"""BEHAVIOR-1K real simulation runtime (policy-free, external action chunks).

Runs inside the ``behavior`` conda env with OmniGibson + Isaac Sim.
Does NOT import PhyAgentOS.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th
from gello.robots.sim_robot.og_teleop_cfg import DISABLED_TRANSITION_RULES
from gello.robots.sim_robot.og_teleop_utils import (
    augment_rooms,
    generate_robot_config,
    get_task_relevant_room_types,
    load_available_tasks,
)
from hydra.utils import instantiate
from omnigibson.envs.env_wrapper import EnvironmentWrapper
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    ROBOT_CAMERA_NAMES,
    TASK_NAMES_TO_INDICES,
    flatten_obs_dict,
    generate_basic_environment_config,
)
from omnigibson.macros import gm, macros
from omnigibson.metrics import AgentMetric, MetricBase, TaskMetric
from omnigibson.robots import BaseRobot
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch

logger = logging.getLogger(__name__)

NUM_EVAL_INSTANCES = 10

# Match upstream eval.py performance macros.
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
with macros.unlocked():
    macros.robots.manipulation_robot.GRASP_WINDOW = 0.75

HEAD_RGB_KEY = ROBOT_CAMERA_NAMES["R1Pro"]["head"] + "::rgb"
LEFT_WRIST_RGB_KEY = ROBOT_CAMERA_NAMES["R1Pro"]["left_wrist"] + "::rgb"
RIGHT_WRIST_RGB_KEY = ROBOT_CAMERA_NAMES["R1Pro"]["right_wrist"] + "::rgb"
PROPRIO_KEY = "robot_r1::proprio"
CAM_REL_POSES_KEY = "robot_r1::cam_rel_poses"

BEHAVIOR1K_DEFAULT_CONFIG: Dict[str, Any] = {
    "task_name": "turning_on_radio",
    "instance_id": 0,
    "max_steps": 200,
    "max_chunk_size": 50,
    "action_dim": 23,
    "headless": True,
    "partial_scene_load": False,
    "close_sim_on_shutdown": False,
}


class Behavior1KRealRuntime:
    """One session == one B1K episode; actions arrive via TargetWS action_chunk."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = dict(BEHAVIOR1K_DEFAULT_CONFIG)
        self.config.update(config or {})
        self.session_id: str | None = None
        self.env: EnvironmentWrapper | None = None
        self.robot: BaseRobot | None = None
        self.metrics: List[MetricBase] = []
        self.human_stats: Dict[str, float] = {}
        self.obs: dict | None = None
        self._env_key: tuple[str, int | None] | None = None
        self.step_idx = 0
        self.success = False
        self.done = False
        self.n_trials = 0
        self.n_success_trials = 0
        self._last_obs: Dict[str, Any] | None = None
        self._last_status: Dict[str, Any] = {
            "accepted": True,
            "safety_status": "idle",
            "executed_steps": 0,
        }
        self._episode_chunks: List[Dict[str, Any]] = []
        self._loaded_tro_instance: int | None = None

    def describe(self) -> Dict[str, Any]:
        return {
            "runtime": "Behavior1KRealRemoteTargetRuntime",
            "task_name": self.config.get("task_name"),
            "task_description": self.config.get("task_description", ""),
            "num_tasks": len(self._task_list()),
            "task_list": self._task_list(),
            "observation_schema": {
                "head_rgb": {"dtype": "uint8", "layout": "HWC"},
                "left_wrist_rgb": {"dtype": "uint8", "layout": "HWC"},
                "right_wrist_rgb": {"dtype": "uint8", "layout": "HWC"},
                "proprio": {"dtype": "float32", "shape": [256]},
                "cam_rel_poses": {"dtype": "float32", "shape": ["*"]},
                "task_id": {"dtype": "int64", "shape": [1]},
            },
            "action_contract": {
                "id": "behavior1k_r1pro_joint_v1",
                "shape": ["T", int(self.config["action_dim"])],
                "dtype": "float32",
                "normalized": False,
                "frame": "robot",
                "max_chunk_size": int(self.config["max_chunk_size"]),
            },
        }

    def configure_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        self._merge_ctx(ctx)
        return {"configured": True, "session_id": self.session_id, "behavior1k": self._metadata()}

    def start_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        self._merge_ctx(ctx)
        return {"started": True, "session_id": self.session_id, "behavior1k": self._metadata()}

    def reset(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        self._merge_ctx(ctx)
        task_name = str(self.config["task_name"])
        instance_slot = int(self.config.get("instance_id", 0))
        self._ensure_env(task_name)
        assert self.env is not None and self.robot is not None

        tro_instance = resolve_tro_instance_id(task_name, instance_slot)
        if self._loaded_tro_instance != tro_instance:
            self.load_task_instance(tro_instance)
            self._loaded_tro_instance = tro_instance

        raw_obs, _ = self.env.reset()
        self.obs = self._preprocess_obs(raw_obs)
        for metric in self.metrics:
            metric.start_callback(self.env)
        self.step_idx = 0
        self.success = False
        self.done = False
        self.n_trials = 0
        self.n_success_trials = 0
        self._episode_chunks = []
        self._stabilize_rendering()
        self._last_obs = self._format_obs(self.obs)
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

    def _stabilize_rendering(self, frames: int = 12) -> None:
        """Warm up Isaac camera Fabric buffers before the first policy observation."""
        if self.env is None:
            return
        try:
            import omnigibson as og

            for _ in range(frames):
                og.sim.render()
            self.obs = self._preprocess_obs(self.env.get_obs())
        except Exception:
            pass

    def observe(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._last_obs is None:
            raise RuntimeError("BEHAVIOR-1K observe before reset")
        return self._last_obs

    def action_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        assert self.env is not None
        actions = np.asarray(chunk.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != int(self.config["action_dim"]):
            raise RuntimeError("BEHAVIOR-1K expected actions [T,%d], got %s" % (self.config["action_dim"], actions.shape))
        if actions.shape[0] > int(self.config["max_chunk_size"]):
            raise RuntimeError("BEHAVIOR-1K action chunk too large: %d" % actions.shape[0])
        if not np.isfinite(actions).all():
            raise RuntimeError("BEHAVIOR-1K actions contain NaN or Inf")

        max_steps = self._env_step_limit()
        if max_steps is None:
            max_steps = 200
        chunk_reward = 0.0
        first_step = self.step_idx + 1
        terminated = truncated = False
        info: dict = {}

        for action in actions:
            action_t = th.as_tensor(action, dtype=th.float32)
            raw_obs, reward, terminated, truncated, info = self.env.step(action_t, n_render_iterations=1)
            self.step_idx += 1
            chunk_reward += float(reward)
            self.obs = self._preprocess_obs(raw_obs)
            for metric in self.metrics:
                metric.step_callback(self.env)
            if terminated or truncated:
                self.n_trials += 1
                if info.get("done", {}).get("success"):
                    self.success = True
                    self.n_success_trials += 1
                self.done = True
                break
            if self.step_idx >= max_steps:
                self.done = True
                break

        if self.done and self.env is not None:
            for metric in self.metrics:
                metric.end_callback(self.env)

        self._last_obs = self._format_obs(self.obs)
        executed = max(0, self.step_idx - first_step + 1)
        self._episode_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", "behavior1k_chunk"),
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
            "chunk_id": chunk.get("chunk_id", "behavior1k_chunk"),
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
            "behavior1k": self._metadata(),
            "episode_summary": self._episode_summary(terminated=terminated, truncated=truncated, info=info),
        }
        return dict(self._last_status)

    def execution_status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def cancel(self, reason: str) -> Dict[str, Any]:
        self._last_status = dict(self._last_status)
        self._last_status.update({"cancelled": True, "cancel_reason": reason})
        return {"cancelled": True, "reason": reason}

    def close(self) -> Dict[str, Any]:
        if self.config.get("close_sim_on_shutdown", False):
            if self.env is not None:
                try:
                    self.env.close()
                except Exception:
                    pass
            try:
                og.shutdown()
            except Exception:
                pass
            self.env = None
            self._env_key = None
        return {"closed": True, "sim_kept_alive": not self.config.get("close_sim_on_shutdown", False)}

    def tick_idle(self) -> None:
        if self.env is None:
            return
        try:
            og.sim.render()
        except Exception:
            pass

    def _merge_ctx(self, ctx: Dict[str, Any]) -> None:
        b1k = dict(ctx.get("behavior1k") or {})
        benchmark = dict(ctx.get("benchmark") or {})
        if benchmark.get("task_name"):
            b1k.setdefault("task_name", benchmark["task_name"])
        if benchmark.get("instance_id") is not None:
            b1k.setdefault("instance_id", benchmark["instance_id"])
        if ctx.get("task_description"):
            b1k.setdefault("task_description", ctx["task_description"])
        self.config.update(b1k)

    def _env_step_limit(self) -> int | None:
        if self.config.get("env_max_steps") is not None:
            return int(self.config["env_max_steps"])
        if self.config.get("max_steps") is not None:
            return int(self.config["max_steps"])
        return None

    def _env_cache_key(self, task_name: str) -> tuple[str, int | None]:
        return (task_name, self._env_step_limit())

    def _ensure_env(self, task_name: str) -> None:
        cache_key = self._env_cache_key(task_name)
        if self.env is not None and self._env_key == cache_key:
            return
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None
            self.robot = None
            self._loaded_tro_instance = None

        for rule in DISABLED_TRANSITION_RULES:
            rule.ENABLED = False

        available_tasks = load_available_tasks()
        if task_name not in available_tasks:
            raise ValueError("invalid BEHAVIOR-1K task_name: %s" % task_name)

        task_idx = TASK_NAMES_TO_INDICES[task_name]
        human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        episodes_path = os.path.join(
            gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "episodes.jsonl"
        )
        with open(episodes_path, "r", encoding="utf-8") as handle:
            episodes = [json.loads(line) for line in handle]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for key in human_stats:
                    human_stats[key].append(episode[key])
        for key in human_stats:
            human_stats[key] = sum(human_stats[key]) / len(human_stats[key])
        self.human_stats = human_stats

        task_cfg = available_tasks[task_name][0]
        cfg = generate_basic_environment_config(task_name=task_name, task_cfg=task_cfg)
        if self.config.get("partial_scene_load"):
            relevant_rooms = get_task_relevant_room_types(activity_name=task_name)
            relevant_rooms = augment_rooms(relevant_rooms, task_cfg["scene_model"], task_name)
            cfg["scene"]["load_room_types"] = relevant_rooms

        cfg["robots"] = [generate_robot_config(task_name=task_name, task_cfg=task_cfg)]
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        cfg["robots"][0]["proprio_obs"] = list(PROPRIOCEPTION_INDICES["R1Pro"].keys())

        max_steps = self._env_step_limit()
        if max_steps is None:
            cfg["task"]["termination_config"]["max_steps"] = int(human_stats["length"] * 2)
        else:
            cfg["task"]["termination_config"]["max_steps"] = int(max_steps)
        cfg["task"]["include_obs"] = False

        env = og.Environment(configs=cfg)
        wrapper_cfg = {"_target_": "omnigibson.learning.wrappers.RGBLowResWrapper"}
        self.env = instantiate(wrapper_cfg, env=env)
        self.robot = self.env.scene.object_registry("name", "robot_r1")
        self.metrics = [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]
        self._env_key = cache_key
        logger.info("Loaded BEHAVIOR-1K env for task %s", task_name)

    def load_task_instance(self, instance_id: int) -> None:
        assert self.env is not None and self.robot is not None
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r", encoding="utf-8") as handle:
            tro_state = recursively_convert_to_torch(json.load(handle))
        for tro_key, tro_value in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = tro_value
                robot_pos = presampled_robot_poses[self.robot.model_name][0]["position"]
                robot_quat = presampled_robot_poses[self.robot.model_name][0]["orientation"]
                self.robot.set_position_orientation(robot_pos, robot_quat)
                self.env.scene.write_task_metadata(key=tro_key, data=tro_value)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_value, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()

    def _preprocess_obs(self, obs: dict) -> dict:
        assert self.robot is not None
        obs = flatten_obs_dict(obs)
        base_pose = self.robot.get_position_orientation()
        cam_rel_poses = []
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = self.robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    th.cat(T.relative_pose_transform(*(camera.get_position_orientation()), *base_pose))
                )
            else:
                cam_pose = T.mat2pose(
                    th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32)
                )
                cam_rel_poses.append(th.cat(T.relative_pose_transform(*cam_pose, *base_pose)))
        obs[CAM_REL_POSES_KEY] = th.cat(cam_rel_poses, axis=-1)
        obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[str(self.config["task_name"])]], dtype=th.int64)
        return obs

    def _format_obs(self, obs: dict) -> Dict[str, Any]:
        def _rgb(key: str) -> np.ndarray:
            tensor = obs[key]
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            array = np.ascontiguousarray(np.asarray(tensor))
            if array.dtype != np.uint8:
                array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
            if array.ndim == 3 and array.shape[-1] == 4:
                array = array[..., :3]
            return array

        def _vec(key: str) -> np.ndarray:
            tensor = obs[key]
            if hasattr(tensor, "detach"):
                tensor = tensor.detach().cpu().numpy()
            return np.asarray(tensor, dtype=np.float32)

        return {
            "observation_id": "behavior1k_obs_%d" % self.step_idx,
            "head_rgb": _rgb(HEAD_RGB_KEY),
            "left_wrist_rgb": _rgb(LEFT_WRIST_RGB_KEY),
            "right_wrist_rgb": _rgb(RIGHT_WRIST_RGB_KEY),
            "proprio": _vec(PROPRIO_KEY),
            "cam_rel_poses": _vec(CAM_REL_POSES_KEY),
            "task_id": int(obs["task_id"].reshape(-1)[0].item()),
            "task_name": str(self.config["task_name"]),
            "instance_id": int(self.config.get("instance_id", 0)),
            "task_description": str(self.config.get("task_description", self.config["task_name"])),
            "timestamp_ns": time.time_ns(),
        }

    def _task_list(self) -> List[Dict[str, Any]]:
        return [{"name": name, "index": idx} for name, idx in sorted(TASK_NAMES_TO_INDICES.items(), key=lambda x: x[1])]

    def _metadata(self) -> Dict[str, Any]:
        return {
            "task_name": self.config.get("task_name"),
            "instance_id": int(self.config.get("instance_id", 0)),
            "task_description": self.config.get("task_description", ""),
            "step_index": self.step_idx,
            "success": bool(self.success),
            "done": bool(self.done),
        }

    def _episode_summary(
        self,
        *,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> Dict[str, Any]:
        metrics_payload: Dict[str, Any] = {}
        if self.done and self.env is not None:
            for metric in self.metrics:
                metrics_payload.update(metric.gather_results())
        q_score = metrics_payload.get("q_score")
        if isinstance(q_score, dict):
            q_score = q_score.get("final")
        if q_score is None:
            q_score = metrics_payload.get("final_q_score")
        return {
            "task_name": self.config.get("task_name"),
            "instance_id": int(self.config.get("instance_id", 0)),
            "executed_steps": self.step_idx,
            "success": bool(self.success),
            "done": bool(self.done),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "n_trials": self.n_trials,
            "n_success_trials": self.n_success_trials,
            "q_score": float(q_score) if q_score is not None else None,
            "metrics": metrics_payload,
            "chunks": list(self._episode_chunks),
        }


def resolve_tro_instance_id(task_name: str, instance_slot: int) -> int:
    if instance_slot < 0 or instance_slot >= NUM_EVAL_INSTANCES:
        raise ValueError("instance_id slot must be in [0, %d), got %d" % (NUM_EVAL_INSTANCES, instance_slot))
    csv_path = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "test_instances.csv")
    with open(csv_path, "r", encoding="utf-8") as handle:
        lines = list(csv.reader(handle))[1:]
    task_idx = TASK_NAMES_TO_INDICES[task_name]
    assert lines[task_idx][1] == task_name
    test_instances = lines[task_idx][2].strip().split(",")
    return int(test_instances[instance_slot])
