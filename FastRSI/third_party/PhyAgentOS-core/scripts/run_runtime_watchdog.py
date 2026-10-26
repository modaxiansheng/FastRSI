#!/usr/bin/env python
"""Run the PhyAgentOS runtime v2 watchdog supervisor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _verification_settings(args: argparse.Namespace) -> tuple[bool, dict]:
    local_service_url = args.verification_service_url or os.environ.get("PAOS_VERIFICATION_LOCAL_SERVICE_URL")
    remote_target_url = args.verification_remote_target_url or os.environ.get("PAOS_VERIFICATION_REMOTE_TARGET_URL")
    episode_token = args.verification_episode_token or os.environ.get("PAOS_VERIFICATION_EPISODE_TOKEN")
    enabled = bool(local_service_url) or _truthy(os.environ.get("PAOS_VERIFICATION_ENABLED"))
    settings = {
        "local_service_url": local_service_url,
        "remote_target_url": remote_target_url,
        "max_replans_per_episode": int(
            args.verification_max_replans
            if args.verification_max_replans is not None
            else os.environ.get("PAOS_VERIFICATION_MAX_REPLANS_PER_EPISODE", 2)
        ),
        "max_verifier_calls_per_run": int(
            args.verification_max_calls
            if args.verification_max_calls is not None
            else os.environ.get("PAOS_VERIFICATION_MAX_VERIFIER_CALLS_PER_RUN", 50)
        ),
        "timeout_s": float(
            args.verification_timeout_s
            if args.verification_timeout_s is not None
            else os.environ.get("PAOS_VERIFICATION_TIMEOUT_S", 180.0)
        ),
        "episode_token": episode_token,
    }
    return enabled, settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PhyAgentOS runtime v2 watchdog")
    parser.add_argument("--workspace", required=True, help="Workspace containing TARGETS/SKILLRUNTIME/SESSIONS.md")
    parser.add_argument(
        "--environment-workspace",
        help="Agent/shared workspace where perception writes ENVIRONMENT.md. Defaults to --workspace.",
    )
    parser.add_argument("--once", action="store_true", help="Run one polling pass and exit")
    parser.add_argument(
        "--session-id",
        help="Run this session only (must exist in SESSIONS.md). Resets it to pending before claim.",
    )
    parser.add_argument("--verification-service-url", help="Local verification service URL, e.g. http://127.0.0.1:8100")
    parser.add_argument("--verification-remote-target-url", help="Verification service URL reachable from remote targets")
    parser.add_argument("--verification-episode-token", help="Shared secret for scoped benchmark episode verification")
    parser.add_argument("--verification-timeout-s", type=float, help="Per verification request timeout")
    parser.add_argument("--verification-max-replans", type=int, help="Max recovery replans per episode")
    parser.add_argument("--verification-max-calls", type=int, help="Max verifier calls per benchmark run")
    args = parser.parse_args()

    verification_enabled, verification_settings = _verification_settings(args)
    supervisor = WatchdogSupervisor(
        args.workspace,
        environment_workspace=args.environment_workspace,
        verification_enabled=verification_enabled,
        verification_settings=verification_settings,
    )
    if args.once:
        ok = supervisor.run_once(session_id=args.session_id)
        return 0 if ok else 1

    while True:
        supervisor.run_once(session_id=args.session_id)


if __name__ == "__main__":
    raise SystemExit(main())
