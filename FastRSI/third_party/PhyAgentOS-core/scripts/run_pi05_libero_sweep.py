"""Phase 4 evaluation: multi-episode LIBERO success-rate sweep for PI0.5.

Runs N episodes (task_id, init_state_id) through the real runtime path
(real pi05 policy server <-> real LIBERO MuJoCo target) and reports the
success rate. Keep it small (~20 episodes / ~20 min).

Servers must already be running:
  - PI0.5 policy server (lerobotpi): openpi://127.0.0.1:8000
  - real LIBERO target server (liberopi): targetws://127.0.0.1:9002
    (start it on any task; this sweep switches task/init per episode via config)

  PYTHONPATH=$(pwd) conda run -n paos python scripts/run_pi05_libero_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from PhyAgentOS.runtime.communication.target_ws_client import TargetWSClient
from PhyAgentOS.runtime.artifacts.episode_writer import EpisodeWriter
from PhyAgentOS.runtime.policy.factory import build_policy_client
from PhyAgentOS.runtime.schemas import AdapterPlan, SessionSpec, SkillRuntimeSpec, TargetSpec
from PhyAgentOS.runtime.sessions.session_runner import SessionRunner
from PhyAgentOS.runtime.skillruntime.policy import OpenPISkillRuntime
from PhyAgentOS.runtime.targets.remote.libero.proxy import LiberoRemoteTargetProxy

DEFAULT_POLICY_ENDPOINT = "openpi://127.0.0.1:8000"
DEFAULT_TARGET_ENDPOINT = "targetws://127.0.0.1:9002"
DEFAULT_BENCHMARK_NAME = "libero_spatial"
DEFAULT_MAX_STEPS = 280
DEFAULT_REPLAN_EVERY = 10

# task 0..9 each with init_state 0 and 1 -> 20 episodes (same task is contiguous
# so the real server only rebuilds the MuJoCo env once per task).
EPISODES = [(t, i) for t in range(10) for i in (0, 1)]

_SKILL_SPEC = {
    "id": "pi05_libero_remote",
    "runtime": "OpenPISkillRuntime",
    "runtime_kind": "policy",
    "loop_mode": "policy_closed_loop",
    "supported_target_kinds": ["simulation"],
    "policy": {"policy_client": "openpi", "policy_adapter": "policy_adapter://openpi_pi05_adapter"},
    "requires": {"sensors": [], "environment_outputs": [], "strict_environment_contract": True},
    "output_contract": {
        "action": {
            "action_space_id": "libero_pi05_delta_eef_gripper_v1",
            "shape": ["T", 7],
            "dtype": "float32",
            "normalized": False,
            "representation": "delta_eef_pose_gripper",
            "frame": "base",
            "chunk": {"policy_hz": 20},
        }
    },
}


def run_one(args, task_id: int, init_id: int):
    cfg = {
        "benchmark_name": args.benchmark_name,
        "task_id": task_id,
        "init_state_id": init_id,
        "camera_height": 256,
        "camera_width": 256,
        "max_steps": args.max_steps,
        "num_steps_wait": 10,
        "max_chunk_size": 50,
        "action": {"action_dim": 7, "max_chunk_size": 50},
    }
    client = TargetWSClient(args.target_endpoint, target_id="libero_real_remote", timeout_s=180.0)
    target = LiberoRemoteTargetProxy(client, config=cfg)
    policy_client = build_policy_client(args.policy_endpoint, timeout_s=180.0)
    session = SessionSpec.model_validate(
        {
            "session_id": f"sweep_t{task_id}_i{init_id}",
            "target_ref": "target://libero_real_remote",
            "skillruntime_ref": "skillruntime://pi05_libero_remote",
            "task_description": "libero task",
            "routing": {"policy_endpoint": args.policy_endpoint, "target_endpoint": args.target_endpoint},
            "execution": {
                "max_steps": args.max_steps,
                "replan_every_steps": args.replan_every_steps,
                "action_chunk_mode": "chunk_buffer",
            },
        }
    )
    target_spec = TargetSpec.model_validate(
        {
            "id": "libero_real_remote",
            "target_class": "remote",
            "target_kind": "simulation",
            "workspace": "workspaces/libero_real",
            "supported_skillruntimes": ["pi05_libero_remote"],
            "runtime": {
                "target_runtime": "LiberoRemoteTargetProxy",
                "target_endpoint": args.target_endpoint,
                "target_adapter": "target_adapter://libero_adapter",
                "runtime_contract_ref": "configs/runtime/contracts/libero_real.runtime.yaml",
            },
            "config": cfg,
        }
    )
    result = SessionRunner(
        session=session,
        target_spec=target_spec,
        skillruntime_spec=SkillRuntimeSpec.model_validate(_SKILL_SPEC),
        adapter_plan=AdapterPlan(
            target_adapter="target_adapter://libero_adapter",
            policy_adapter="policy_adapter://openpi_pi05_adapter",
            action_bridges=["bridge://safety_clamp"],
        ),
        target=target,
        skill_runtime=OpenPISkillRuntime(),
        policy_client=policy_client,
        perception_runtime=None,
        perception_plan=None,
    ).start()
    for c in (policy_client, client):
        try:
            c.close()
        except Exception:
            pass
    artifact_dir = EpisodeWriter(args.artifacts_root).write_episode(session, target_spec, "pi05_libero_remote", result)
    final_status = (result.metadata or {}).get("final_status", {})
    summary = final_status.get("episode_summary") if isinstance(final_status, dict) else None
    return {
        "task_id": task_id,
        "init_state_id": init_id,
        "success": bool(result.success),
        "status": result.status,
        "num_steps": int(result.num_steps or 0),
        "artifact_dir": str(artifact_dir),
        "episode_summary": summary or {},
        "error_code": result.error_code,
        "error_message": result.error_message,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real LIBERO/pi0.5 multi-episode sweep")
    parser.add_argument("--policy-endpoint", default=DEFAULT_POLICY_ENDPOINT)
    parser.add_argument("--target-endpoint", default=DEFAULT_TARGET_ENDPOINT)
    parser.add_argument("--benchmark-name", default=DEFAULT_BENCHMARK_NAME)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--replan-every-steps", type=int, default=DEFAULT_REPLAN_EVERY)
    parser.add_argument("--artifacts-root", type=Path, default=Path("/tmp/phyagentos_libero_sweep_artifacts"))
    parser.add_argument("--summary-json", type=Path, default=Path("/tmp/phyagentos_libero_sweep_summary.json"))
    args = parser.parse_args()
    args.artifacts_root.mkdir(parents=True, exist_ok=True)
    succ = 0
    t0 = time.perf_counter()
    rows = []
    for k, (task_id, init_id) in enumerate(EPISODES):
        row = run_one(args, task_id, init_id)
        succ += int(row["success"])
        rows.append(row)
        elapsed = time.perf_counter() - t0
        print(
            f"[{k + 1:2d}/{len(EPISODES)}] task{task_id} init{init_id}: "
            f"success={row['success']} steps={row['num_steps']} | running {succ}/{k + 1} | {elapsed:.0f}s elapsed",
            flush=True,
        )
    rate = succ / len(EPISODES)
    print("=" * 50)
    print(f"SUCCESS RATE: {succ}/{len(EPISODES)} = {rate:.1%}")
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row["success"])
    print("per-task:", {t: f"{sum(v)}/{len(v)}" for t, v in sorted(by_task.items())})
    payload = {
        "benchmark_name": args.benchmark_name,
        "successes": succ,
        "episodes": len(EPISODES),
        "success_rate": rate,
        "rows": rows,
        "per_task": {str(t): {"successes": int(sum(v)), "episodes": len(v)} for t, v in sorted(by_task.items())},
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("summary_json:", args.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
