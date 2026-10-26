#!/usr/bin/env python
"""Initialize Runtime v2 protocol files in a workspace."""

from __future__ import annotations

import argparse
import sys
from importlib.resources import files as pkg_files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PhyAgentOS.runtime.workspace.manager import (
    RUNTIME_CONFIG_TEMPLATE_NAMES,
    RUNTIME_TEMPLATE_NAMES,
    RuntimeWorkspaceManager,
)
from PhyAgentOS.config.schema import Config


def init_runtime_workspace(workspace: Path, force: bool = False) -> dict[str, list[str]]:
    """Create or refresh Runtime v2 protocol files."""
    workspace = workspace.expanduser()
    workspace.mkdir(parents=True, exist_ok=True)
    templates = pkg_files("PhyAgentOS") / "templates"

    result: dict[str, list[str]] = {"created": [], "skipped": [], "overwritten": []}
    for name in (*RUNTIME_TEMPLATE_NAMES, "SESSIONS.md"):
        src = templates / name
        dest = workspace / name
        if dest.exists() and not force:
            result["skipped"].append(name)
            continue

        existed = dest.exists()
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        result["overwritten" if existed else "created"].append(name)
    for name in RUNTIME_CONFIG_TEMPLATE_NAMES:
        src = templates.joinpath(*Path(name).parts)
        dest = workspace / name
        if dest.exists() and not force:
            result["skipped"].append(name)
            continue
        existed = dest.exists()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        result["overwritten" if existed else "created"].append(name)
    cfg = Config()
    cfg.agents.defaults.workspace = str(workspace)
    cfg.runtime.workspace = str(workspace)
    cfg.runtime.autostart_watchdog = False
    manager = RuntimeWorkspaceManager(cfg)
    report = manager.write_runtime_instructions()
    for name in report.created:
        if name not in result["created"]:
            result["created"].append(name)
    for name in report.updated:
        if name not in result["created"] and name not in result["overwritten"]:
            result["overwritten"].append(name)
    environment_path = workspace / "ENVIRONMENT.md"
    if environment_path.exists() and not force:
        result["skipped"].append("ENVIRONMENT.md")
    else:
        existed = environment_path.exists()
        environment_path.write_text("", encoding="utf-8")
        result["overwritten" if existed else "created"].append("ENVIRONMENT.md")
    report = manager.write_environment_targets_snapshot()
    for name in report.created:
        if name not in result["created"]:
            result["created"].append(name)
    for name in report.updated:
        if name not in result["created"] and name not in result["overwritten"]:
            result["overwritten"].append(name)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize PhyAgentOS Runtime v2 workspace files")
    parser.add_argument("--workspace", required=True, help="Workspace directory to initialize")
    parser.add_argument("--force", action="store_true", help="Overwrite existing runtime protocol files")
    args = parser.parse_args()

    result = init_runtime_workspace(Path(args.workspace), force=args.force)
    for key in ("created", "overwritten", "skipped"):
        for name in result[key]:
            print(f"{key}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
