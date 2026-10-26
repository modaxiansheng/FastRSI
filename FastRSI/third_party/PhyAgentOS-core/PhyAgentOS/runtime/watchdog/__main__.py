"""Entry point for `python -m PhyAgentOS.runtime.watchdog`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor

DEFAULT_WORKSPACE = Path(os.path.expanduser("~/.PhyAgentOS/workspace"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PhyAgentOS Runtime Watchdog Supervisor")
    parser.add_argument(
        "--workspace",
        default=str(DEFAULT_WORKSPACE),
        help=f"Workspace path (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--environment-workspace",
        help="Agent/shared workspace for ENVIRONMENT.md (default: same as --workspace)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one polling pass and exit",
    )
    parser.add_argument(
        "--session-id",
        help="Run this session only (must exist in SESSIONS.md). Resets it to pending before claim.",
    )
    args = parser.parse_args()

    supervisor = WatchdogSupervisor(
        args.workspace,
        environment_workspace=args.environment_workspace,
    )

    if args.once:
        return 0 if supervisor.run_once(session_id=args.session_id) else 1

    while True:
        supervisor.run_once(session_id=args.session_id)


if __name__ == "__main__":
    raise SystemExit(main())
