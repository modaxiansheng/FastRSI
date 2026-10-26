#!/usr/bin/env python
"""Summarize LIBERO evaluation success rates."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


def read_sessions(workspace: Path) -> list[dict]:
    sessions_md = workspace / "SESSIONS.md"
    text = sessions_md.read_text()
    try:
        block = text.split("```yaml", 1)[1].split("```", 1)[0]
    except IndexError as exc:
        raise ValueError(f"{sessions_md} does not contain a fenced yaml block") from exc
    doc = yaml.safe_load(block)
    return list(doc.get("sessions") or [])


def iter_workspaces(paths: list[Path]) -> list[Path]:
    workspaces: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if (path / "SESSIONS.md").is_file():
            workspaces.append(path)
            continue
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "SESSIONS.md").is_file():
                workspaces.append(child)
    if not workspaces:
        joined = ", ".join(str(path) for path in paths)
        raise ValueError(f"no workspaces containing SESSIONS.md found under: {joined}")
    return workspaces


def summarize_workspace(workspace: Path) -> tuple[int, int, int]:
    sessions = read_sessions(workspace)
    counts = Counter(session.get("status", "unknown") for session in sessions)
    benchmark_first_success = 0
    benchmark_final_success = 0
    benchmark_total = 0
    has_final = False
    for session in sessions:
        result = session.get("result") or {}
        metadata = result.get("metadata") or {}
        benchmark = metadata.get("benchmark_result") or {}
        if not isinstance(benchmark, dict):
            continue
        if benchmark.get("total_episodes") is None:
            continue
        first = int(benchmark.get("first_attempt_successes", benchmark.get("successes") or 0))
        final = int(benchmark.get("final_successes", first))
        benchmark_first_success += first
        benchmark_final_success += final
        benchmark_total += int(benchmark.get("total_episodes") or 0)
        has_final = has_final or final != first or str(benchmark.get("verification_profile", "strict")) == "recovery"
    if benchmark_total:
        success = benchmark_first_success
        total = benchmark_total
    else:
        success = sum(session.get("status") == "succeeded" for session in sessions)
        benchmark_final_success = success
        total = len(sessions)
    first_rate = success / total if total else 0.0
    if has_final and benchmark_total:
        final_rate = benchmark_final_success / total if total else 0.0
        print(
            f"{workspace.name}: first={success}/{total} = {first_rate:.3%}; "
            f"final={benchmark_final_success}/{total} = {final_rate:.3%} statuses={dict(counts)}"
        )
    else:
        print(f"{workspace.name}: {success}/{total} = {first_rate:.3%} statuses={dict(counts)}")
    return success, benchmark_final_success, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        help="Directory containing suite workspaces, e.g. tests/xvla/libero_4suite_...",
    )
    parser.add_argument(
        "--workspace",
        action="append",
        type=Path,
        help="Single workspace containing SESSIONS.md. Can be passed multiple times.",
    )
    args = parser.parse_args()

    paths = (args.run_root or []) + (args.workspace or [])
    if not paths:
        parser.error("pass at least one --run-root or --workspace")

    total_first_success = 0
    total_final_success = 0
    total_episodes = 0
    for workspace in iter_workspaces(paths):
        first_success, final_success, total = summarize_workspace(workspace)
        total_first_success += first_success
        total_final_success += final_success
        total_episodes += total

    first_rate = total_first_success / total_episodes if total_episodes else 0.0
    final_rate = total_final_success / total_episodes if total_episodes else 0.0
    if total_final_success != total_first_success:
        print(
            f"overall: first={total_first_success}/{total_episodes} = {first_rate:.3%}; "
            f"final={total_final_success}/{total_episodes} = {final_rate:.3%}"
        )
    else:
        print(f"overall: {total_first_success}/{total_episodes} = {first_rate:.3%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
