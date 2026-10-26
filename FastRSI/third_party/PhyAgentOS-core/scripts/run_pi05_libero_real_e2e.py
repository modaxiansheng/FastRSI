"""External acceptance: real PI0.5 policy server -> real LIBERO target via Watchdog.

Assumes two servers are already running:
  - PI0.5 policy server: openpi://127.0.0.1:8000
  - real LIBERO target server: targetws://127.0.0.1:9002

This script writes a minimal runtime workspace and executes one Watchdog pass.
It deliberately exercises the HAL v3 path instead of constructing SessionRunner
directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor


DEFAULT_TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one real LIBERO/pi0.5 session through WatchdogSupervisor")
    parser.add_argument("--workspace", default="/tmp/phyagentos_libero_real_e2e")
    parser.add_argument("--policy-endpoint", default="openpi://127.0.0.1:8000")
    parser.add_argument("--target-endpoint", default="targetws://127.0.0.1:9002")
    parser.add_argument("--benchmark-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--task-description", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--replan-every-steps", type=int, default=10)
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser()
    _write_workspace(workspace, args)
    supervisor = WatchdogSupervisor(workspace)
    ran = supervisor.run_once()
    if not ran:
        print("No pending runtime session was executed.")
        return 1

    sessions_doc = read_yaml_block(workspace / "SESSIONS.md")
    session = sessions_doc["sessions"][0]
    result = session.get("result", {})
    print("=== RESULT ===")
    print("status:", session.get("status"))
    print("success:", result.get("success"))
    print("artifact_dir:", result.get("artifact_dir"))
    artifact_dir = result.get("artifact_dir")
    if artifact_dir:
        episode_path = workspace / artifact_dir / "episode.json"
        if episode_path.exists():
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            print("benchmark:", json.dumps(payload.get("benchmark"), ensure_ascii=False, sort_keys=True))
    return 0 if result.get("success") else 2


def _write_workspace(workspace: Path, args) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    contract_path = workspace / "configs/runtime/contracts/libero_real.runtime.yaml"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_src = ROOT / "PhyAgentOS/templates/configs/runtime/contracts/libero_real.runtime.yaml"
    contract_path.write_text(contract_src.read_text(encoding="utf-8"), encoding="utf-8")
    write_yaml_block(
        workspace / "TARGETS.md",
        "Runtime Targets",
        {
            "version": "runtime_target_registry_v1",
            "targets": [
                {
                    "id": "libero_real_remote",
                    "target_class": "remote",
                    "target_kind": "simulation",
                    "enabled": True,
                    "workspace": "workspaces/libero_real",
                    "supported_skillruntimes": ["pi05_libero_remote"],
                    "runtime": {
                        "target_runtime": "LiberoRemoteTargetProxy",
                        "target_endpoint": args.target_endpoint,
                        "target_adapter": "target_adapter://libero_adapter",
                        "runtime_contract_ref": "configs/runtime/contracts/libero_real.runtime.yaml",
                    },
                    "observation": {"observation_type": "multimodal", "empty_observation_allowed": False},
                    "perception": {"enabled": False, "strict_preflight": True},
                    "config": {
                        "benchmark_name": args.benchmark_name,
                        "task_id": args.task_id,
                        "init_state_id": args.init_state_id,
                        "camera_height": 256,
                        "camera_width": 256,
                        "max_steps": args.max_steps,
                        "num_steps_wait": 10,
                        "action": {"action_dim": 7, "max_chunk_size": 50},
                    },
                }
            ],
        },
    )
    write_yaml_block(
        workspace / "SKILLRUNTIME.md",
        "Runtime Skill Runtimes",
        {
            "version": "runtime_skill_registry_v1",
            "skillruntimes": [
                {
                    "id": "pi05_libero_remote",
                    "runtime": "OpenPISkillRuntime",
                    "runtime_kind": "policy",
                    "loop_mode": "policy_closed_loop",
                    "agent_exposure": "none",
                    "supported_target_kinds": ["simulation"],
                    "policy": {"policy_client": "openpi", "policy_adapter": "policy_adapter://openpi_pi05_adapter"},
                    "observation_contract": {"observation_type": "multimodal", "empty_observation_allowed": False},
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
                    "adapter_requirements": {"allowed_bridges": ["bridge://safety_clamp"]},
                }
            ],
        },
    )
    write_yaml_block(
        workspace / "SESSIONS.md",
        "Runtime Sessions",
        {
            "version": "runtime_sessions_v1",
            "sessions": [
                {
                    "session_id": f"libero_t{args.task_id}_i{args.init_state_id}",
                    "target_ref": "target://libero_real_remote",
                    "skillruntime_ref": "skillruntime://pi05_libero_remote",
                    "task_description": args.task_description,
                    "status": "pending",
                    "routing": {"target_endpoint": args.target_endpoint, "policy_endpoint": args.policy_endpoint},
                    "execution": {
                        "max_steps": args.max_steps,
                        "replan_every_steps": args.replan_every_steps,
                        "action_chunk_mode": "chunk_buffer",
                    },
                    "timeouts": {"execute_timeout_s": 600, "policy_timeout_s": 180},
                    "safety_profile": {"profile": "default_simulation", "stop_on_policy_timeout": True},
                    "result": {},
                }
            ],
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
