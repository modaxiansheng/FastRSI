"""Standalone TargetWS server backed by a REAL LIBERO simulation.

Runs in a LIBERO conda env (for example `libero`, py3.8) which has
libero + robosuite + mujoco. Deliberately self-contained: it does NOT import
the PhyAgentOS package (that would pull pydantic / py3.10+ syntax into py3.8).
It speaks the TargetWS msgpack-over-websocket wire format, so the runtime side
(`LiberoRemoteTargetProxy` + `LiberoTargetAdapter`, in the `paos` env) talks to
it transparently.

Launch (LIBERO env):

  MUJOCO_GL=egl PYTHONWARNINGS=ignore \
  conda run --no-capture-output -n libero python PhyAgentOS/runtime/targets/remote/libero/server.py \
    --host 0.0.0.0 --port 9002 \
    --benchmark-name libero_spatial --task-id 0 --init-state-id 0 \
    --camera-height 256 --camera-width 256 --max-steps 300 --num-steps-wait 10 \
    --control-mode relative
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from urllib import error as url_error, request as url_request

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

# --- msgpack wire codec (kept byte-compatible with PhyAgentOS msgpack_codec) ---
import msgpack

RPC_VERSION = "phyagentos.runtime_rpc.v2"
WEBSOCKET_KEEPALIVE_DISABLED = {
    "ping_interval": None,
    "ping_timeout": None,
}
VERIFIER_CLIENT_TIMEOUT_GRACE_S = 15.0

# LIBERO dummy action keeps the gripper open while the scene settles.
LIBERO_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError("Unsupported dtype: %s" % obj.dtype)
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Dict[Any, Any]) -> Any:
    if obj.get("__ndarray__"):
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=obj["shape"])
    if obj.get("__npgeneric__"):
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def packb(payload: Any) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, default=_pack_array)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_unpack_array)


def _pack_policy_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError("Unsupported dtype: %s" % obj.dtype)
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_policy_array(obj: Dict[Any, Any]) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


def policy_packb(payload: Any) -> bytes:
    return msgpack.packb(payload, default=_pack_policy_array)


def policy_unpackb(payload: bytes) -> Any:
    return _decode_policy_keys(msgpack.unpackb(payload, object_hook=_unpack_policy_array))


def _decode_policy_keys(value: Any) -> Any:
    if isinstance(value, dict):
        decoded: Dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            decoded[key] = _decode_policy_keys(item)
        return decoded
    if isinstance(value, list):
        return [_decode_policy_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_policy_keys(item) for item in value)
    return value


def make_response(request: Dict[str, Any], response_type: str, payload: Dict[str, Any]) -> bytes:
    return packb(
        {
            "version": RPC_VERSION,
            "type": response_type,
            "session_id": request.get("session_id"),
            "target_id": request.get("target_id"),
            "skillruntime_id": request.get("skillruntime_id"),
            "episode_id": request.get("episode_id"),
            "seq": int(request.get("seq", 0)),
            "timestamp_ns": time.time_ns(),
            "trace_id": request.get("trace_id"),
            "payload": payload,
        }
    )


class TargetProtocolError(Exception):
    pass


def status(message: str) -> None:
    print("[libero-target] %s" % message, flush=True)


async def _run_blocking(func: Any, *args: Any, **kwargs: Any) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


# --- real LIBERO runtime ------------------------------------------------------

LIBERO_DEFAULT_CONFIG = {
    "benchmark_name": "libero_spatial",
    "task_id": 0,
    "init_state_id": 0,
    "camera_height": 256,
    "camera_width": 256,
    "max_chunk_size": 50,
    "max_steps": 300,
    "num_steps_wait": 10,
    "control_mode": "relative",
    "seed": 0,
    "record_dir": None,
    "record_fps": 20,
}


class LiberoRealRuntime:
    """Real LIBERO benchmark target runtime (one session == one episode)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(LIBERO_DEFAULT_CONFIG)
        self.config.update(config or {})
        self.session_id = None
        self.env = None
        self._env_key = None
        self.suite = None
        self.task = None
        self.init_states = None
        self.language = "LIBERO task"
        self.step_idx = 0
        self.success = False
        self.done = False
        self._total_reward = 0.0
        self._episode_chunks = []
        self._last_obs = None
        self._initial_obs = None
        self._frames = []
        self._episode_idx = 0
        self._last_video_path = None
        self._last_status = {"accepted": True, "safety_status": "idle", "executed_steps": 0}
        self._benchmark_jobs: Dict[str, Dict[str, Any]] = {}

    # -- lifecycle --
    def _ensure_env(self):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        import os

        bn = str(self.config["benchmark_name"])
        tid = int(self.config["task_id"])
        key = (
            bn,
            tid,
            int(self.config["camera_height"]),
            int(self.config["camera_width"]),
            int(self.config.get("seed", 0)),
        )
        if self.env is not None and self._env_key == key:
            return
        status(
            "loading LIBERO env suite=%s task_id=%d camera=%dx%d"
            % (bn, tid, int(self.config["camera_width"]), int(self.config["camera_height"]))
        )
        suite = benchmark.get_benchmark_dict()[bn]()
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=int(self.config["camera_height"]),
            camera_widths=int(self.config["camera_width"]),
        )
        # Match OpenVLA's official LIBERO eval; the env seed affects object
        # positions even when using fixed LIBERO init states.
        self.env.seed(int(self.config.get("seed", 0)))
        self.suite = suite
        self.task = task
        self.init_states = suite.get_task_init_states(tid)
        self.language = str(getattr(task, "language", task.name))
        self._env_key = key
        status(
            "env ready suite=%s task_id=%d init_states=%d task=\"%s\""
            % (bn, tid, len(self.init_states), self.language)
        )

    def describe(self) -> Dict[str, Any]:
        self._ensure_env()
        return {
            "runtime": "LiberoRealRemoteTargetRuntime",
            "benchmark_name": self.config["benchmark_name"],
            "task_id": int(self.config["task_id"]),
            "task_description": self.language,
            "num_tasks": len(self._task_list()),
            "task_list": self._task_list(),
            "benchmark_capabilities": [{
                "benchmark_id": "libero",
                "suites": ["libero_spatial", "libero_object", "libero_goal", "libero_10"],
                "execution_modes": [
                    {"mode": "policy_loop", "interface": "rollout_episode_v1", "reset_owner": "session_runner"},
                    {"mode": "target_native", "interface": "target_benchmark_job_v1", "reset_owner": "skillruntime"},
                ],
            }],
            "observation_schema": {
                "agentview_image": {"dtype": "uint8", "layout": "HWC"},
                "robot0_eye_in_hand_image": {"dtype": "uint8", "layout": "HWC"},
                "robot0_eef_pos": {"dtype": "float32", "shape": [3]},
                "robot0_eef_quat": {"dtype": "float32", "shape": [4]},
                "robot0_eef_mat": {"dtype": "float32", "shape": [3, 3]},
                "robot0_gripper_qpos": {"dtype": "float32", "shape": [2]},
            },
            "action_contract": {
                "id": "libero_delta_eef_gripper_v1",
                "shape": ["T", 7],
                "dtype": "float32",
                "normalized": False,
                "frame": "base",
                "control_mode": str(self.config.get("control_mode", "relative")),
                "max_chunk_size": int(self.config["max_chunk_size"]),
            },
        }

    def configure_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        self.config.update(ctx.get("libero", {}))
        status("configured session=%s %s" % (self.session_id or "<none>", self._short_config()))
        return {"configured": True, "session_id": self.session_id, "libero": self._libero_metadata()}

    def start_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        return {"started": True, "session_id": self.session_id, "libero": self._libero_metadata()}

    def reset(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        self.config.update(ctx.get("libero", {}))
        status("reset requested session=%s %s" % (self.session_id or "<none>", self._short_config()))
        self._ensure_env()
        self.env.reset()
        isid = int(self.config.get("init_state_id", 0))
        obs = self.env.set_init_state(self.init_states[isid])
        for _ in range(int(self.config.get("num_steps_wait", 10))):
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        # Most OpenPI/OpenVLA LIBERO policies emit relative/delta end-effector
        # actions. Some LeRobot/X-VLA evaluations use absolute control.
        use_delta = str(self.config.get("control_mode", "relative")).lower() not in {"absolute", "abs"}
        for robot in self.env.robots:
            robot.controller.use_delta = use_delta
        status(
            "episode ready session=%s control_mode=%s use_delta=%s task=\"%s\""
            % (self.session_id or "<none>", str(self.config.get("control_mode", "relative")), use_delta, self.language)
        )
        self.step_idx = 0
        self.success = False
        self.done = False
        self._total_reward = 0.0
        self._episode_chunks = []
        self._frames = []
        self._last_video_path = None
        self._record_frame(obs)
        self._last_obs = self._format_obs(obs)
        self._initial_obs = self._last_obs
        self._last_status = {
            "accepted": True,
            "safety_status": "ok",
            "executed_steps": 0,
            "target_step_index": 0,
            "success": False,
            "done": False,
            "reward": 0.0,
            "obs": self._last_obs,
        }
        return self._last_obs

    def observe(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._last_obs is None:
            raise TargetProtocolError("LIBERO observe before reset")
        return self._last_obs

    def action_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        actions = np.asarray(chunk.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise TargetProtocolError("LIBERO expected actions [T,7], got %s" % (actions.shape,))
        if actions.shape[0] > int(self.config["max_chunk_size"]):
            raise TargetProtocolError("LIBERO action chunk too large: %d" % actions.shape[0])
        if not np.isfinite(actions).all():
            raise TargetProtocolError("LIBERO actions contain NaN or Inf")

        max_steps = int(self.config.get("max_steps", 300))
        chunk_reward = 0.0
        first_step = self.step_idx + 1
        obs = self._raw_obs
        for action in actions:
            obs, reward, done, _info = self.env.step(action.tolist())
            self.step_idx += 1
            chunk_reward += float(reward)
            self._record_frame(obs)
            if bool(done):
                self.success = True
                self.done = True
            if self.done or self.step_idx >= max_steps:
                break
        self._total_reward += chunk_reward
        self.done = self.done or self.step_idx >= max_steps
        self._last_obs = self._format_obs(obs)
        if self.done:
            self._last_video_path = self._write_video()
        executed = max(0, self.step_idx - first_step + 1)
        self._episode_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", "libero_chunk"),
                "first_step": first_step,
                "executed_steps": executed,
                "requested_steps": int(actions.shape[0]),
                "reward": chunk_reward,
                "success": bool(self.success),
                "done": bool(self.done),
                "action_shape": [int(actions.shape[0]), int(actions.shape[1])],
            }
        )
        self._last_status = {
            "chunk_id": chunk.get("chunk_id", "libero_chunk"),
            "accepted": True,
            "buffered_steps": 0,
            "executed_steps": self.step_idx,
            "target_step_index": self.step_idx,
            "need_replan": not self.success,
            "safety_status": "ok",
            "success": bool(self.success),
            "done": bool(self.done),
            "reward": chunk_reward,
            "obs": self._last_obs,
            "libero": self._libero_metadata(),
            "episode_summary": self._episode_summary(),
        }
        return dict(self._last_status)

    def execution_status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def cancel(self, reason: str) -> Dict[str, Any]:
        self._last_status = dict(self._last_status)
        self._last_status.update({"cancelled": True, "cancel_reason": reason})
        return {"cancelled": True, "reason": reason}

    def close(self) -> Dict[str, Any]:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self._env_key = None
        return {"closed": True}

    def benchmark_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("schema_version") != "benchmark_job_request_v1":
            raise TargetProtocolError("UNSUPPORTED_BENCHMARK_SCHEMA")
        active = [job for job in self._benchmark_jobs.values() if job["state"] in {"pending", "running", "cancelling"}]
        if active:
            raise TargetProtocolError("a LIBERO benchmark job is already active")
        job_id = "libero_job_%d" % time.time_ns()
        config = dict(payload.get("options") or {})
        config.update({key: value for key, value in payload.items() if key != "options"})
        config["suite"] = payload.get("suite_id")
        config["job_id"] = job_id
        job = {
            "job_id": job_id,
            "state": "pending",
            "completed_episodes": 0,
            "total_episodes": len(config.get("episodes") or []),
            "heartbeat_ns": time.time_ns(),
            "result": None,
            "error_code": None,
            "error_message": None,
        }
        self._benchmark_jobs[job_id] = job

        async def run() -> None:
            job.update({"state": "running", "heartbeat_ns": time.time_ns()})
            try:
                result = await self.benchmark(config)
                result["schema_version"] = "benchmark_execution_result_v1"
                job.update({
                    "state": "succeeded" if result.get("status") == "succeeded" else "failed",
                    "completed_episodes": int(result.get("completed_episodes") or 0),
                    "result": result,
                    "heartbeat_ns": time.time_ns(),
                })
            except asyncio.CancelledError:
                job.update({"state": "cancelled", "heartbeat_ns": time.time_ns()})
                job["result"] = {
                    "schema_version": "benchmark_execution_result_v1", "status": "cancelled",
                    "run_id": str(config.get("run_id") or ""), "suite": str(config.get("suite") or ""),
                    "execution_mode": "target_native", "episodes": [],
                }
            except Exception as exc:  # noqa: BLE001
                job.update({"state": "failed", "error_code": type(exc).__name__, "error_message": str(exc), "heartbeat_ns": time.time_ns()})
                job["result"] = {
                    "schema_version": "benchmark_execution_result_v1", "status": "failed",
                    "run_id": str(config.get("run_id") or ""), "suite": str(config.get("suite") or ""),
                    "execution_mode": "target_native", "episodes": [],
                    "error_code": type(exc).__name__, "error_message": str(exc),
                }

        job["task"] = asyncio.create_task(run())
        return {"schema_version": "benchmark_job_ref_v1", "job_id": job_id}

    def benchmark_status(self, job_id: str) -> Dict[str, Any]:
        job = self._benchmark_job(job_id)
        return {"schema_version": "benchmark_job_status_v1", **{key: value for key, value in job.items() if key in {"job_id", "state", "completed_episodes", "total_episodes", "heartbeat_ns", "error_code", "error_message"}}}

    def benchmark_result(self, job_id: str) -> Dict[str, Any]:
        job = self._benchmark_job(job_id)
        if job["state"] not in {"succeeded", "failed", "cancelled"}:
            raise TargetProtocolError("benchmark job has not completed")
        return dict(job["result"])

    def benchmark_cancel(self, job_id: str, reason: str) -> Dict[str, Any]:
        job = self._benchmark_job(job_id)
        if job["state"] in {"pending", "running"}:
            job.update({"state": "cancelling", "heartbeat_ns": time.time_ns(), "error_message": reason})
            job["task"].cancel()
        return self.benchmark_status(job_id)

    def _benchmark_job(self, job_id: str) -> Dict[str, Any]:
        job = self._benchmark_jobs.get(job_id)
        if job is None:
            raise TargetProtocolError("unknown benchmark job: %s" % job_id)
        return job

    async def benchmark(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = dict(payload or {})
        suite = str(config.get("suite") or config.get("benchmark_name") or self.config.get("benchmark_name"))
        policy_endpoint = str(config.get("policy_endpoint") or "")
        if not policy_endpoint:
            raise TargetProtocolError("benchmark requires policy_endpoint")
        episode_specs = _parse_episode_specs(config)
        max_steps = int(config.get("max_steps", self.config.get("max_steps", 300)))
        policy_timeout_s = float(config.get("policy_timeout_s", 180.0))
        control_mode = str(config.get("control_mode", self.config.get("control_mode", "relative")))
        evidence_mode = str(config.get("evidence_mode") or ("all" if config.get("include_evidence") else "none"))
        attempt_index = int(config.get("attempt_index", 0))
        verification = _verification_config(config)
        verification_enabled = verification["profile"] != "strict"
        if verification_enabled:
            evidence_mode = "failed"
        configured_verifier_budget = max(0, int(verification["max_verifier_calls_per_run"])) if verification_enabled else 0
        effective_verifier_budget = min(configured_verifier_budget, len(episode_specs))
        status(
            "benchmark start session=%s suite=%s episodes=%d policy=%s"
            % (self.session_id or config.get("session_id") or "<none>", suite, len(episode_specs), policy_endpoint)
        )
        episodes = []
        total_steps = 0
        latencies = []
        first_successes = 0
        final_successes = 0
        verifier_attempts = []
        attempts = []
        episodes_replanned = 0
        recovered_after_replan = 0
        started = time.time()
        async with _PolicyWsClient(policy_endpoint, timeout_s=policy_timeout_s) as policy:
            for episode_spec in episode_specs:
                task_id = int(episode_spec["task_id"])
                init_state_id = int(episode_spec["init_state_id"])
                if verification_enabled:
                    episode, episode_attempts, episode_verifier_attempts = await self._run_verified_benchmark_episode(
                        policy,
                        suite=suite,
                        task_id=task_id,
                        init_state_id=init_state_id,
                        max_steps=max_steps,
                        control_mode=control_mode,
                        evidence_mode=evidence_mode,
                        config=config,
                        verification=verification,
                        task_description=episode_spec.get("task_description"),
                        verifier_calls_remaining=max(0, effective_verifier_budget - len(verifier_attempts)),
                    )
                else:
                    episode = await self._run_benchmark_episode(
                        policy,
                        suite=suite,
                        task_id=task_id,
                        init_state_id=init_state_id,
                        max_steps=max_steps,
                        control_mode=control_mode,
                        attempt_index=attempt_index,
                        evidence_mode=evidence_mode,
                        config=config,
                        task_description=episode_spec.get("task_description"),
                    )
                    episode_attempts = [_attempt_record(episode)]
                    episode_verifier_attempts = []
                episodes.append(episode)
                job = self._benchmark_jobs.get(str(config.get("job_id") or ""))
                if job is not None:
                    job.update({"completed_episodes": len(episodes), "heartbeat_ns": time.time_ns()})
                attempts.extend(episode_attempts)
                verifier_attempts.extend(episode_verifier_attempts)
                total_steps += sum(int(item.get("num_steps") or 0) for item in episode_attempts)
                first_successes += 1 if episode.get("first_attempt_success", episode.get("success")) else 0
                final_successes += 1 if episode.get("final_success", episode.get("success")) else 0
                if len(episode_attempts) > 1:
                    episodes_replanned += 1
                    if episode.get("final_success") and not episode.get("first_attempt_success"):
                        recovered_after_replan += 1
                if episode.get("mean_policy_latency_ms") is not None:
                    latencies.append(float(episode["mean_policy_latency_ms"]))
                status(
                    "benchmark episode suite=%s t%d_i%d first=%s final=%s steps=%d rate=%d/%d"
                    % (
                        suite,
                        task_id,
                        init_state_id,
                        bool(episode.get("first_attempt_success", episode.get("success"))),
                        bool(episode.get("final_success", episode.get("success"))),
                        int(episode.get("num_steps") or 0),
                        final_successes,
                        len(episodes),
                    )
                )
        total = len(episodes)
        result = {
            "status": "succeeded" if total else "failed",
            "run_id": str(config.get("run_id") or config.get("session_id") or self.session_id or ""),
            "suite": suite,
            "execution_mode": "target_native",
            "verification_profile": verification["profile"],
            "control_mode": control_mode,
            "max_steps": max_steps,
            "successes": first_successes,
            "first_attempt_successes": first_successes if attempt_index == 0 else 0,
            "final_successes": final_successes,
            "total_episodes": total,
            "completed_episodes": total,
            "valid_episodes": total,
            "total_attempts": len(attempts) if attempts else total,
            "success_rate": float(first_successes / total) if total else 0.0,
            "final_success_rate": float(final_successes / total) if total else 0.0,
            "first_attempt_score": float(first_successes / total) if total else 0.0,
            "assisted_final_score": float(final_successes / total) if total else 0.0,
            "episodes_replanned": episodes_replanned,
            "recovered_after_replan": recovered_after_replan,
            "verifier_calls": len(verifier_attempts),
            "configured_max_verifier_calls_per_run": configured_verifier_budget,
            "effective_max_verifier_calls_per_run": effective_verifier_budget,
            "remaining_verifier_calls": max(0, effective_verifier_budget - len(verifier_attempts)),
            "verifier_skipped": sum(1 for item in verifier_attempts if item.get("verdict") == "skipped"),
            "num_steps": total_steps,
            "mean_policy_latency_ms": float(np.mean(latencies)) if latencies else None,
            "elapsed_s": float(time.time() - started),
            "episodes": episodes,
            "attempts": attempts,
            "verifier_attempts": verifier_attempts,
        }
        status(
            "benchmark finished suite=%s first=%d/%d final=%d/%d elapsed_s=%.1f"
            % (suite, first_successes, total, final_successes, total, float(result["elapsed_s"]))
        )
        return result

    async def _run_verified_benchmark_episode(
        self,
        policy,
        *,
        suite: str,
        task_id: int,
        init_state_id: int,
        max_steps: int,
        control_mode: str,
        evidence_mode: str,
        config: Dict[str, Any],
        verification: Dict[str, Any],
        task_description: Optional[str],
        verifier_calls_remaining: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        profile = str(verification["profile"])
        retry_instruction_mode = str(verification["retry_instruction_mode"])
        max_replans = max(0, int(verification["max_replans_per_episode"])) if profile == "recovery" else 0
        attempts = []
        verifier_attempts = []
        first_episode = None
        final_episode = None
        current_task_description = task_description
        for attempt_index in range(max_replans + 1):
            episode = await self._run_benchmark_episode(
                policy,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                max_steps=max_steps,
                control_mode=control_mode,
                attempt_index=attempt_index,
                evidence_mode=evidence_mode,
                config=config,
                task_description=current_task_description,
                reset_environment=attempt_index == 0,
            )
            if first_episode is None:
                # Preserve the native first attempt before a verifier can revise
                # this attempt's final outcome.
                first_episode = dict(episode)
            attempts.append(_attempt_record(episode))
            final_episode = episode
            if episode.get("success"):
                if attempt_index > 0:
                    status(
                        "verification recovery result suite=%s t%d_i%d attempt=%d success=True steps=%d error_code=%s error_message=%s"
                        % (
                            suite,
                            task_id,
                            init_state_id,
                            attempt_index,
                            int(episode.get("num_steps") or 0),
                            str(episode.get("error_code")),
                            str(episode.get("error_message")),
                        )
                    )
                break
            status(
                "benchmark attempt failed suite=%s t%d_i%d attempt=%d steps=%d max_steps=%d error_code=%s error_message=%s"
                % (
                    suite,
                    task_id,
                    init_state_id,
                    attempt_index,
                    int(episode.get("num_steps") or 0),
                    max_steps,
                    str(episode.get("error_code")),
                    str(episode.get("error_message")),
                )
            )
            if len(verifier_attempts) >= verifier_calls_remaining:
                status(
                    "episode verifier skipped suite=%s t%d_i%d attempt=%d reason=verifier_budget_exhausted"
                    % (suite, task_id, init_state_id, attempt_index)
                )
                break
            status(
                "episode verifier start suite=%s t%d_i%d attempt=%d endpoint=%s"
                % (
                    suite,
                    task_id,
                    init_state_id,
                    attempt_index,
                    str(verification.get("endpoint") or "<none>"),
                )
            )
            verdict = await _verify_episode_attempt(config, episode, verification)
            verifier_attempts.append(verdict)
            episode.pop("evidence", None)
            if verdict.get("verdict") == "success":
                episode["success"] = True
                episode["status"] = "succeeded"
                episode["success_authority"] = "verifier"
                break
            status(
                "episode verifier verdict suite=%s t%d_i%d attempt=%d verdict=%s verifier_status=%s failure_reason=%s"
                % (
                    suite,
                    task_id,
                    init_state_id,
                    attempt_index,
                    str(verdict.get("verdict")),
                    str(verdict.get("verifier_status")),
                    str(verdict.get("failure_reason")),
                )
            )
            if verdict.get("verdict") != "replan" or profile != "recovery" or attempt_index >= max_replans:
                status(
                    "verification recovery skipped suite=%s t%d_i%d attempt=%d verdict=%s"
                    % (suite, task_id, init_state_id, attempt_index, str(verdict.get("verdict")))
                )
                break
            if retry_instruction_mode == "verifier_rewrite":
                current_task_description = str(verdict["replan_task_description"])
            else:
                current_task_description = task_description
            verdict["applied_task_description"] = (
                current_task_description
                if current_task_description is not None
                else episode.get("task_description")
            )
            status(
                "verification recovery start suite=%s t%d_i%d next_attempt=%d"
                % (suite, task_id, init_state_id, attempt_index + 1)
            )
        first_success = bool(first_episode and first_episode.get("success"))
        final_success = bool(final_episode and final_episode.get("success"))
        merged = dict(first_episode or final_episode or {})
        merged.pop("evidence", None)
        merged["first_attempt_success"] = first_success
        merged["final_success"] = final_success
        merged["success"] = final_success
        merged["final_status"] = "succeeded" if final_success else "failed"
        merged["attempts"] = attempts
        if final_episode is not None:
            merged["success_authority"] = final_episode.get("success_authority", "target")
            merged["task_description"] = final_episode.get("task_description")
            merged["num_steps"] = int(final_episode.get("num_steps") or 0)
            merged["return_value"] = float(final_episode.get("return_value") or 0.0)
            merged["mean_policy_latency_ms"] = final_episode.get("mean_policy_latency_ms")
            merged["status"] = str(final_episode.get("status") or merged.get("status") or "failed")
        return merged, attempts, verifier_attempts

    async def _run_benchmark_episode(
        self,
        policy,
        *,
        suite: str,
        task_id: int,
        init_state_id: int,
        max_steps: int,
        control_mode: str,
        attempt_index: int,
        evidence_mode: str,
        config: Dict[str, Any],
        task_description: Optional[str] = None,
        reset_environment: bool = True,
    ) -> Dict[str, Any]:
        self.config.update(
            {
                "benchmark_name": suite,
                "task_id": int(task_id),
                "init_state_id": int(init_state_id),
                "max_steps": int(max_steps),
                "control_mode": control_mode,
                "seed": int(config.get("seed", self.config.get("seed", 0))),
                "camera_height": int(config.get("camera_height", self.config.get("camera_height", 256))),
                "camera_width": int(config.get("camera_width", self.config.get("camera_width", 256))),
                "num_steps_wait": int(config.get("num_steps_wait", self.config.get("num_steps_wait", 10))),
                "record_dir": config.get("record_dir", self.config.get("record_dir")),
            }
        )
        episode_session_id = "%s_t%d_i%d" % (str(config.get("session_id") or self.session_id or "benchmark"), task_id, init_state_id)
        if attempt_index:
            episode_session_id = "%s_a%d" % (episode_session_id, attempt_index)
        episode_id = "%s_t%d_i%d" % (suite, task_id, init_state_id)
        if reset_environment:
            self.reset({"session_id": episode_session_id, "libero": dict(self.config)})
        else:
            self.session_id = episode_session_id
            self.step_idx = 0
            self.success = False
            self.done = False
            self._total_reward = 0.0
            self._episode_chunks = []
            self._initial_obs = self._last_obs
        policy.reset_session(episode_session_id)
        effective_task_description = str(task_description).strip() if task_description is not None else self.language
        replan_every_steps = _benchmark_replan_every_steps(config)
        episode_latencies = []
        error_code = None
        error_message = None
        try:
            while not self.done and self.step_idx < max_steps:
                observation = _policy_observation(self._last_obs, effective_task_description, episode_session_id)
                policy_output = await policy.infer(observation)
                policy_meta = dict(policy_output.get("policy_meta") or {})
                if policy_meta.get("policy_latency_ms") is not None:
                    episode_latencies.append(float(policy_meta["policy_latency_ms"]))
                actions = _policy_actions(policy_output)
                actions = actions[: min(replan_every_steps, actions.shape[0])]
                self.action_chunk(
                    {
                        "chunk_id": "benchmark_policy_chunk_%d" % len(self._episode_chunks),
                        "actions": actions,
                    }
                )
                job = self._benchmark_jobs.get(str(config.get("job_id") or ""))
                if job is not None:
                    job["heartbeat_ns"] = time.time_ns()
        except Exception as exc:  # noqa: BLE001
            if _is_policy_transport_error(exc):
                raise TargetProtocolError(
                    "policy transport failed during %s t%d_i%d attempt=%d: %s"
                    % (suite, task_id, init_state_id, attempt_index, exc)
                ) from exc
            error_code = type(exc).__name__
            error_message = str(exc)
            self.done = True
        summary = self._episode_summary()
        episode = {
            "episode_id": episode_id,
            "suite": suite,
            "task_id": int(task_id),
            "init_state_id": int(init_state_id),
            "attempt_index": int(attempt_index),
            "policy_session_id": episode_session_id,
            "task_description": effective_task_description,
            "success": bool(self.success),
            "first_attempt_success": bool(self.success) if int(attempt_index) == 0 else None,
            "final_success": bool(self.success),
            "status": "succeeded" if bool(self.success) else "failed",
            "num_steps": int(self.step_idx),
            "return_value": float(self._total_reward),
            "mean_policy_latency_ms": float(np.mean(episode_latencies)) if episode_latencies else None,
            "error_code": error_code,
            "error_message": error_message,
            "episode_summary": {
                key: value
                for key, value in summary.items()
                if key not in {"chunks"}
            },
        }
        include_evidence = evidence_mode == "all" or (evidence_mode == "failed" and not bool(self.success))
        if include_evidence:
            episode["evidence"] = {
                "version": "benchmark_episode_evidence_v1",
                "initial_observation": self._initial_obs,
                "final_observation": self._last_obs,
            }
        return episode

    # -- helpers --
    def _format_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self._raw_obs = obs
        # LIBERO renders images upside-down; flip vertically+horizontally to match training.
        agent = np.ascontiguousarray(np.asarray(obs["agentview_image"])[::-1, ::-1])
        wrist = np.ascontiguousarray(np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1])
        return {
            "observation_id": "libero_obs_%d" % self.step_idx,
            "agentview_image": agent.astype(np.uint8, copy=False),
            "robot0_eye_in_hand_image": wrist.astype(np.uint8, copy=False),
            "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            "robot0_eef_mat": np.asarray(self.env.robots[0].controller.ee_ori_mat, dtype=np.float32),
            "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            "benchmark_name": self.config["benchmark_name"],
            "task_id": int(self.config["task_id"]),
            "init_state_id": int(self.config.get("init_state_id", 0)),
            "task_description": self.language,
            "timestamp_ns": time.time_ns(),
        }

    def _record_frame(self, raw_obs: Dict[str, Any]) -> None:
        if not self.config.get("record_dir"):
            return
        # Human-upright view: raw LIBERO images are rendered upside-down, so flip
        # vertically only. Show agentview + wrist side by side.
        agent = np.asarray(raw_obs["agentview_image"])[::-1]
        wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"])[::-1]
        self._frames.append(np.ascontiguousarray(np.hstack([agent, wrist])))

    def _write_video(self):
        record_dir = self.config.get("record_dir")
        if not record_dir or not self._frames:
            return None
        if cv2 is None:
            print("[record] cv2 unavailable; skipping video", flush=True)
            return None
        os.makedirs(record_dir, exist_ok=True)
        self._episode_idx += 1
        tag = "success" if self.success else "fail"
        name = "ep_t%d_i%d_%02d_%s.mp4" % (
            int(self.config["task_id"]),
            int(self.config.get("init_state_id", 0)),
            self._episode_idx,
            tag,
        )
        path = os.path.join(record_dir, name)
        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(self.config.get("record_fps", 20)), (w, h))
        for frame in self._frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR for OpenCV
        writer.release()
        print("[record] wrote %d frames -> %s (%s)" % (len(self._frames), path, tag), flush=True)
        return path

    def _libero_metadata(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.config["benchmark_name"],
            "task_id": int(self.config["task_id"]),
            "init_state_id": int(self.config.get("init_state_id", 0)),
            "step_index": self.step_idx,
            "task_description": self.language,
            "control_mode": str(self.config.get("control_mode", "relative")),
            "seed": int(self.config.get("seed", 0)),
        }

    def _short_config(self) -> str:
        return (
            "suite=%s task_id=%d init_state_id=%d control_mode=%s max_steps=%d seed=%d"
            % (
                str(self.config.get("benchmark_name")),
                int(self.config.get("task_id", 0)),
                int(self.config.get("init_state_id", 0)),
                str(self.config.get("control_mode", "relative")),
                int(self.config.get("max_steps", 300)),
                int(self.config.get("seed", 0)),
            )
        )

    def _episode_summary(self) -> Dict[str, Any]:
        summary = {
            "benchmark_name": self.config["benchmark_name"],
            "task_id": int(self.config["task_id"]),
            "init_state_id": int(self.config.get("init_state_id", 0)),
            "task_description": self.language,
            "target_step_index": self.step_idx,
            "success": bool(self.success),
            "done": bool(self.done),
            "total_reward": float(self._total_reward),
            "num_action_chunks": len(self._episode_chunks),
            "chunks": list(self._episode_chunks),
            "final_observation_id": self._last_obs.get("observation_id") if self._last_obs else None,
        }
        if self._last_video_path:
            summary["video_path"] = self._last_video_path
        return summary

    def _task_list(self):
        if self.suite is None:
            return []
        num_tasks = int(
            getattr(self.suite, "n_tasks", 0)
            or getattr(self.suite, "num_tasks", 0)
            or len(getattr(self.suite, "tasks", []) or [])
        )
        tasks = []
        for task_id in range(num_tasks):
            try:
                task = self.suite.get_task(task_id)
            except Exception:
                continue
            item = {
                "task_id": task_id,
                "task_name": str(getattr(task, "name", "task_%d" % task_id)),
                "language": str(getattr(task, "language", getattr(task, "name", "task_%d" % task_id))),
            }
            problem_folder = getattr(task, "problem_folder", None)
            bddl_file = getattr(task, "bddl_file", None)
            if problem_folder is not None:
                item["problem_folder"] = str(problem_folder)
            if bddl_file is not None:
                item["bddl_file"] = str(bddl_file)
            tasks.append(item)
        return tasks


class _PolicyWsClient:
    def __init__(self, endpoint: str, *, timeout_s: float):
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self._ws = None
        self._last_session_id = None

    async def __aenter__(self):
        import websockets

        self._ws = await asyncio.wait_for(
            websockets.connect(
                _policy_ws_url(self.endpoint),
                max_size=None,
                compression=None,
                **WEBSOCKET_KEEPALIVE_DISABLED,
            ),
            timeout=self.timeout_s,
        )
        metadata = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        if isinstance(metadata, str):
            raise TargetProtocolError("policy server returned text metadata: %s" % metadata)
        self.metadata = policy_unpackb(metadata)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._ws is not None:
            await self._ws.close()
        self._ws = None

    def reset_session(self, session_id: str) -> None:
        self._last_session_id = str(session_id)

    async def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self._ws is None:
            raise TargetProtocolError("policy websocket is not connected")
        if self._last_session_id is not None:
            observation = dict(observation)
            observation["session_id"] = self._last_session_id
        started = time.perf_counter()
        await asyncio.wait_for(self._ws.send(policy_packb(observation)), timeout=self.timeout_s)
        response = await asyncio.wait_for(self._ws.recv(), timeout=self.timeout_s)
        if isinstance(response, str):
            raise TargetProtocolError("policy server returned text error: %s" % response)
        output = policy_unpackb(response)
        if not isinstance(output, dict) or "actions" not in output:
            raise TargetProtocolError("policy response missing actions")
        output.setdefault("policy_meta", {})
        output["policy_meta"]["policy_latency_ms"] = (time.perf_counter() - started) * 1000.0
        return output


def _policy_ws_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme == "openpi":
        return urlunparse(("ws", parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
    if parsed.scheme in {"ws", "wss"}:
        return endpoint
    raise TargetProtocolError("unsupported policy endpoint for benchmark: %s" % endpoint)


def _is_policy_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    name = type(exc).__name__
    if name in {"ConnectionClosed", "ConnectionClosedError", "ConnectionClosedOK"}:
        return True
    message = str(exc)
    return (
        "policy websocket is not connected" in message
        or "policy response missing actions" in message
        or "policy server returned text" in message
        or "timed out" in message.lower()
        or "keepalive ping timeout" in message
        or "no close frame received" in message
    )


def _policy_observation(obs: Dict[str, Any], task: str, session_id: str) -> Dict[str, Any]:
    return {
        "observation/image": np.ascontiguousarray(obs["agentview_image"]),
        "observation/wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
        "observation/state": _libero_state(obs),
        "observation/eef_mat": np.ascontiguousarray(np.asarray(obs["robot0_eef_mat"], dtype=np.float32)),
        "prompt": str(task),
        "session_id": session_id,
    }


def _libero_state(obs: Dict[str, Any]) -> np.ndarray:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(4)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper.size == 0:
        gripper = np.zeros(2, dtype=np.float32)
    if gripper.size == 1:
        gripper = np.repeat(gripper, 2)
    return np.ascontiguousarray(np.concatenate([eef_pos, _quat_to_axisangle(eef_quat), gripper[:2]], axis=0), dtype=np.float32)


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.astype(np.float32, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if den < 1e-8:
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * np.arccos(quat[3]) / den)).astype(np.float32)


def _policy_actions(policy_output: Dict[str, Any]) -> np.ndarray:
    actions = np.asarray(policy_output["actions"], dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise TargetProtocolError("policy action must have shape [A] or [T,A], got %s" % (actions.shape,))
    if actions.shape[1] == 7:
        return np.ascontiguousarray(actions, dtype=np.float32)
    if actions.shape[1] >= 10:
        return _ee6d_action_to_libero(actions)
    raise TargetProtocolError("policy action must have 7 dims or ee6d dims >=10, got %s" % (actions.shape,))


def _benchmark_replan_every_steps(config: Dict[str, Any]) -> int:
    for key in ("replan_every_steps", "preferred_replan_every_steps", "policy_chunk_steps", "replan_every"):
        value = config.get(key)
        if value is None:
            continue
        steps = int(value)
        if steps > 0:
            return steps
    return int(config.get("max_chunk_size", LIBERO_DEFAULT_CONFIG["max_chunk_size"]))


def _ee6d_action_to_libero(actions: np.ndarray) -> np.ndarray:
    target_eef = actions[:, :3]
    target_axis = _rotate6d_to_axis_angle(actions[:, 3:9])
    gripper = np.where(actions[:, 9:10] > 0.5, 1.0, -1.0)
    return np.ascontiguousarray(np.concatenate([target_eef, target_axis, gripper], axis=-1), dtype=np.float32)


def _rotate6d_to_axis_angle(rotation_6d: np.ndarray) -> np.ndarray:
    a1 = rotation_6d[:, 0:3]
    a2 = rotation_6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-6)
    dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2_orth = a2 - dot_prod * b1
    b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + 1e-6)
    b3 = np.cross(b1, b2, axis=-1)
    rotation_matrix = np.stack([b1, b2, b3], axis=-1)
    return np.stack([_quat_to_axis_angle(_mat_to_quat(mat)) for mat in rotation_matrix], axis=0).astype(np.float32)


def _mat_to_quat(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)[:3, :3]
    m00, m01, m02 = mat[0]
    m10, m11, m12 = mat[1]
    m20, m21, m22 = mat[2]
    k = np.array(
        [
            [m00 - m11 - m22, 0.0, 0.0, 0.0],
            [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
            [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ],
        dtype=np.float32,
    )
    k /= 3.0
    values, vectors = np.linalg.eigh(k)
    quat = vectors[[3, 0, 1, 2], np.argmax(values)]
    if quat[0] < 0.0:
        quat = -quat
    return quat[[1, 2, 3, 0]].astype(np.float32)


def _parse_id_list(value: Any, *, default: List[int]) -> List[int]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    spec = str(value).strip()
    if spec == "all":
        return list(default)
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(dict.fromkeys(ids))


def _parse_episode_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    explicit = config.get("episodes")
    if isinstance(explicit, list):
        episodes = []
        for item in explicit:
            if not isinstance(item, dict):
                continue
            episode = {
                "task_id": int(item["task_id"]),
                "init_state_id": int(item["init_state_id"]),
            }
            if item.get("task_description") is not None:
                episode["task_description"] = str(item["task_description"])
            episodes.append(episode)
        return episodes
    task_ids = _parse_id_list(config.get("task_ids"), default=list(range(10)))
    init_state_ids = _parse_id_list(config.get("init_state_ids"), default=list(range(50)))
    return [
        {"task_id": int(task_id), "init_state_id": int(init_state_id)}
        for task_id in task_ids
        for init_state_id in init_state_ids
    ]


def _verification_config(config: Dict[str, Any]) -> Dict[str, Any]:
    profile = str(config.get("verification_profile") or "strict")
    if profile not in {"strict", "audit", "recovery"}:
        raise TargetProtocolError("unsupported verification profile: %s" % profile)
    return {
        "profile": profile,
        "endpoint": config.get("verification_endpoint"),
        "token": config.get("verification_token"),
        "timeout_s": float(config.get("verification_timeout_s") or 180.0),
        "max_replans_per_episode": _int_config(config, "max_replans_per_episode", 2),
        "max_verifier_calls_per_run": _int_config(config, "max_verifier_calls_per_run", 50),
        "retry_instruction_mode": str(config.get("retry_instruction_mode") or "original"),
    }


async def _verify_episode_attempt(
    config: Dict[str, Any],
    episode: Dict[str, Any],
    verification: Dict[str, Any],
) -> Dict[str, Any]:
    endpoint = verification.get("endpoint")
    bundle = _episode_verification_bundle(config, episode)
    started = time.time()
    verdict = await _run_blocking(
        _post_verifier,
        str(endpoint or ""),
        bundle,
        float(verification.get("timeout_s") or 180.0),
        str(verification.get("token") or ""),
    )
    record = {
        "run_id": config.get("run_id") or config.get("session_id"),
        "suite": config.get("suite") or config.get("benchmark_name"),
        "episode_id": episode.get("episode_id"),
        "task_id": episode.get("task_id"),
        "init_state_id": episode.get("init_state_id"),
        "attempt_index": episode.get("attempt_index", 0),
        "verdict": verdict.get("verdict"),
        "evidence": verdict.get("evidence"),
        "failure_reason": verdict.get("failure_reason"),
        "replan_task_description": verdict.get("replan_task_description"),
        "lesson": verdict.get("lesson"),
        "verifier_status": verdict.get("verifier_status"),
        "retry_instruction_mode": verification["retry_instruction_mode"],
        "latency_ms": (time.time() - started) * 1000.0,
        "evidence_digest": hashlib.sha256(
            json.dumps(_jsonable(bundle), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "evidence_retention": "deleted_after_verification",
    }
    return record


def _int_config(config: Dict[str, Any], key: str, default: int) -> int:
    value = config.get(key)
    return int(default if value is None else value)


def _post_verifier(
    endpoint: str,
    bundle: Dict[str, Any],
    timeout_s: float,
    token: str,
) -> Dict[str, Any]:
    if not endpoint:
        return _verifier_error("verifier endpoint is not configured")
    if endpoint.startswith("mock://"):
        mode = endpoint[len("mock://") :]
        if mode in {"replan", "replan_failed"}:
            return {
                "verdict": "replan",
                "evidence": ["mock verifier requested retry"],
                "failure_reason": None,
                "replan_task_description": bundle.get("task_description"),
                "lesson": "Mock verifier retries failed benchmark episodes.",
                "verifier_status": "mock",
            }
        return _verifier_error("mock verifier did not request retry")
    try:
        if endpoint.startswith(("http://", "https://")) and not endpoint.rstrip("/").endswith("/v1/verify-episode"):
            endpoint = endpoint.rstrip("/") + "/v1/verify-episode"
        payload = json.dumps(_jsonable(bundle), ensure_ascii=False).encode("utf-8")
        req = url_request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
            method="POST",
        )
        opener = url_request.build_opener(url_request.ProxyHandler({}))
        client_timeout_s = max(1.0, float(timeout_s) + VERIFIER_CLIENT_TIMEOUT_GRACE_S)
        with opener.open(req, timeout=client_timeout_s) as response:  # noqa: S310 - configured verifier
            data = json.loads(response.read().decode("utf-8"))
    except url_error.HTTPError as exc:
        return _verifier_error(_http_error_message(exc))
    except (OSError, TimeoutError, url_error.URLError, json.JSONDecodeError) as exc:
        return _verifier_error(str(exc))
    if not isinstance(data, dict):
        return _verifier_error("verifier response is not a JSON object")
    verdict = str(data.get("verdict") or "failure")
    if verdict not in {"success", "replan", "failure", "skipped"}:
        verdict = "failure"
    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        evidence = ["verifier did not provide evidence"]
    return {
        "verdict": verdict,
        "evidence": [str(item) for item in evidence],
        "failure_reason": data.get("failure_reason"),
        "replan_task_description": data.get("replan_task_description"),
        "lesson": str(data.get("lesson") or "No verifier lesson was provided."),
        "verifier_status": str(data.get("verifier_status") or "completed"),
    }


def _verifier_error(reason: str) -> Dict[str, Any]:
    return {
        "verdict": "failure",
        "evidence": ["verifier service error: %s" % reason],
        "failure_reason": reason,
        "replan_task_description": None,
        "lesson": "Verification failed and no recovery was attempted.",
        "verifier_status": "error",
    }


def _http_error_message(exc: url_error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # noqa: BLE001
        body = ""
    if not body:
        return "HTTP Error %s: %s" % (exc.code, exc.reason)
    return "HTTP Error %s: %s: %s" % (exc.code, exc.reason, body[:500])


def _episode_verification_bundle(config: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    evidence = episode.get("evidence") if isinstance(episode.get("evidence"), dict) else {}
    return {
        "version": "benchmark_episode_verification_v1",
        "session_id": config.get("session_id"),
        "run_id": config.get("run_id") or config.get("session_id"),
        "suite_id": episode.get("suite") or config.get("suite") or config.get("benchmark_name"),
        "episode_id": episode.get("episode_id"),
        "task_index": episode.get("task_id"),
        "instance_id": episode.get("init_state_id"),
        "attempt_index": episode.get("attempt_index", 0),
        "task_description": episode.get("task_description"),
        "runtime_claim": {
            "success": bool(episode.get("success")),
            "source": "target_environment",
            "authority": "target",
        },
        "initial_observation": evidence.get("initial_observation"),
        "final_observation": evidence.get("final_observation"),
        "final_status": {
            "steps": episode.get("num_steps"),
            "reward": episode.get("return_value"),
            "done": True,
        },
    }


def _attempt_record(episode: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "episode_id": episode.get("episode_id"),
        "suite": episode.get("suite"),
        "task_id": episode.get("task_id"),
        "init_state_id": episode.get("init_state_id"),
        "attempt_index": int(episode.get("attempt_index") or 0),
        "policy_session_id": episode.get("policy_session_id"),
        "task_description": episode.get("task_description"),
        "success": bool(episode.get("success")),
        "status": episode.get("status"),
        "num_steps": episode.get("num_steps"),
        "return_value": episode.get("return_value"),
        "mean_policy_latency_ms": episode.get("mean_policy_latency_ms"),
        "error_code": episode.get("error_code"),
        "error_message": episode.get("error_message"),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


async def _dispatch(runtime: LiberoRealRuntime, request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    rtype = request.get("type")
    payload = request.get("payload") or {}
    if rtype == "target.benchmark":
        raise TargetProtocolError("UNSUPPORTED_BENCHMARK_SCHEMA")
    if rtype == "target.describe":
        return rtype, runtime.describe()
    if rtype == "target.configure_session":
        return rtype, runtime.configure_session(payload)
    if rtype == "target.start_session":
        return rtype, runtime.start_session(payload)
    if rtype == "target.reset":
        return rtype, runtime.reset(payload)
    if rtype == "target.observe":
        return "target.observation", runtime.observe(payload)
    if rtype == "target.action_chunk":
        return rtype, runtime.action_chunk(payload)
    if rtype == "target.benchmark.start":
        return rtype, runtime.benchmark_start(payload)
    if rtype == "target.benchmark.status":
        return rtype, runtime.benchmark_status(str(payload.get("job_id") or ""))
    if rtype == "target.benchmark.result":
        return rtype, runtime.benchmark_result(str(payload.get("job_id") or ""))
    if rtype == "target.benchmark.cancel":
        return rtype, runtime.benchmark_cancel(str(payload.get("job_id") or ""), str(payload.get("reason") or "cancelled"))
    if rtype == "target.execution_status":
        return rtype, runtime.execution_status()
    if rtype == "target.cancel":
        return rtype, runtime.cancel(str(payload.get("reason", "cancelled")))
    if rtype == "target.close":
        return rtype, runtime.close()
    raise TargetProtocolError("unsupported target RPC type: %s" % rtype)


async def serve(runtime: LiberoRealRuntime, host: str, port: int) -> None:
    import websockets

    async def handler(ws):
        async for message in ws:
            if isinstance(message, str):
                await ws.send(packb({"version": RPC_VERSION, "type": "runtime.error", "seq": 0,
                                     "timestamp_ns": time.time_ns(),
                                     "payload": {"error_code": "BAD_PAYLOAD", "message": "expected binary msgpack"}}))
                continue
            request = unpackb(message)
            try:
                rtype, payload = await _dispatch(runtime, request)
                await ws.send(make_response(request, rtype, payload))
            except Exception as exc:  # noqa: BLE001
                await ws.send(
                    make_response(
                        request,
                        "runtime.error",
                        {"error_code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                    )
                )

    async with websockets.serve(
        handler,
        host,
        port,
        max_size=None,
        compression=None,
        **WEBSOCKET_KEEPALIVE_DISABLED,
    ):
        status("listening on ws://%s:%d" % (host, port))
        status("use target endpoint targetws://%s:%d from PAOS" % (host, port))
        status("waiting for target.describe / target.reset requests")
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--benchmark-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--control-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-dir", default=None, help="if set, write an mp4 per episode here")
    parser.add_argument("--record-fps", type=int, default=20)
    args = parser.parse_args()
    status("starting LIBERO TargetWS server")
    status(
        "default suite=%s task_id=%d init_state_id=%d control_mode=%s"
        % (args.benchmark_name, args.task_id, args.init_state_id, args.control_mode)
    )
    status(
        "bind=%s:%d camera=%dx%d max_steps=%d num_steps_wait=%d seed=%d"
        % (
            args.host,
            args.port,
            args.camera_width,
            args.camera_height,
            args.max_steps,
            args.num_steps_wait,
            args.seed,
        )
    )
    if args.record_dir:
        status("recording enabled record_dir=%s fps=%d" % (args.record_dir, args.record_fps))
    runtime = LiberoRealRuntime(
        {
            "benchmark_name": args.benchmark_name,
            "task_id": args.task_id,
            "init_state_id": args.init_state_id,
            "camera_height": args.camera_height,
            "camera_width": args.camera_width,
            "max_steps": args.max_steps,
            "num_steps_wait": args.num_steps_wait,
            "control_mode": args.control_mode,
            "seed": args.seed,
            "record_dir": args.record_dir,
            "record_fps": args.record_fps,
        }
    )
    asyncio.run(serve(runtime, args.host, args.port))


if __name__ == "__main__":
    main()
