"""Benchmark artifact writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PhyAgentOS.runtime.artifacts.episode_writer import _encode_rgb_png, _find_rgb_arrays, _jsonable, _safe_name
from PhyAgentOS.runtime.schemas import SessionResult, SessionSpec, TargetSpec
from PhyAgentOS.runtime.schemas.common import utc_now
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text


class BenchmarkArtifactWriter:
    """Write suite-level benchmark artifacts under artifacts/benchmarks."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.artifacts_root = workspace / "artifacts" / "benchmarks"

    def write_benchmark(
        self,
        session: SessionSpec,
        target: TargetSpec,
        skillruntime_id: str,
        result: SessionResult,
    ) -> dict[str, Any] | None:
        benchmark = result.metadata.get("benchmark_result")
        if not isinstance(benchmark, dict) or benchmark.get("total_episodes") is None:
            return None
        run_id = str(
            benchmark.get("run_id")
            or (session.benchmark.run_id if session.benchmark else None)
            or session.session_id
        )
        run_dir = self.artifacts_root / _safe_path_name(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        episodes = [_sanitize_episode(run_dir, item) for item in list(benchmark.get("episodes") or [])]
        attempts = _collect_attempts(benchmark, episodes)
        verifier_attempts = list(benchmark.get("verifier_attempts") or [])
        failures = [
            episode
            for episode in episodes
            if not bool(episode.get("final_success", episode.get("success")))
        ]
        summary = _summary(
            benchmark,
            run_id=run_id,
            session=session,
            target=target,
            skillruntime_id=skillruntime_id,
            artifact_dir=run_dir,
            episodes=episodes,
            attempts=attempts,
            verifier_attempts=verifier_attempts,
        )

        _write_json(run_dir / "manifest.json", _manifest(session, target, skillruntime_id, benchmark))
        _write_json(run_dir / "summary.json", summary)
        _write_json(run_dir / "tasks.json", _task_summary(episodes))
        _write_jsonl(run_dir / "episodes.jsonl", episodes)
        _write_jsonl(run_dir / "attempts.jsonl", attempts)
        _write_jsonl(run_dir / "verifier_attempts.jsonl", verifier_attempts)
        _write_jsonl(run_dir / "failures.jsonl", failures)
        return summary


def compact_benchmark_result(benchmark: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    """Return a SESSIONS.md-safe benchmark result without episode arrays or RGB evidence."""
    compact = {
        key: value
        for key, value in benchmark.items()
        if key not in {"episodes", "attempts", "verifier_attempts"}
    }
    if summary is not None:
        compact["summary"] = summary
    return _jsonable(compact)


def _sanitize_episode(run_dir: Path, episode: Any) -> dict[str, Any]:
    if not isinstance(episode, dict):
        return {"raw": _jsonable(episode)}
    item = dict(episode)
    evidence = item.pop("evidence", None)
    if isinstance(evidence, dict):
        evidence_paths = _write_evidence(run_dir, item, evidence)
        if evidence_paths:
            item["evidence"] = evidence_paths
    if "attempts" in item and isinstance(item["attempts"], list):
        item["attempts"] = [_jsonable(_strip_evidence(dict(attempt))) for attempt in item["attempts"]]
    return _jsonable(item)


def _write_evidence(run_dir: Path, episode: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    rgb_dir = run_dir / "rgb"
    episode_id = str(episode.get("episode_id") or f"t{episode.get('task_id')}_i{episode.get('init_state_id')}")
    attempt_index = int(episode.get("attempt_index") or 0)
    paths: dict[str, list[str]] = {}
    for phase in ("initial", "final"):
        phase_value = evidence.get(f"{phase}_observation") or evidence.get(f"{phase}_rgb")
        written: list[str] = []
        for index, (name, array) in enumerate(_find_rgb_arrays(phase_value), start=1):
            rgb_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{_safe_path_name(episode_id)}_attempt{attempt_index}_{phase}_{index:02d}_{_safe_name(name)}.png"
            path = rgb_dir / filename
            path.write_bytes(_encode_rgb_png(array))
            written.append(str(path.relative_to(run_dir)))
        if written:
            paths[f"{phase}_rgb_paths"] = written
    return paths


def _strip_evidence(item: dict[str, Any]) -> dict[str, Any]:
    item.pop("evidence", None)
    return item


def _collect_attempts(benchmark: dict[str, Any], episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts = list(benchmark.get("attempts") or [])
    if attempts:
        return [_jsonable(_strip_evidence(dict(attempt))) for attempt in attempts if isinstance(attempt, dict)]
    collected: list[dict[str, Any]] = []
    for episode in episodes:
        episode_attempts = episode.get("attempts")
        if isinstance(episode_attempts, list) and episode_attempts:
            for attempt in episode_attempts:
                if isinstance(attempt, dict):
                    collected.append(_jsonable(_strip_evidence(dict(attempt))))
            continue
        collected.append(
            {
                "episode_id": episode.get("episode_id"),
                "suite": episode.get("suite"),
                "task_id": episode.get("task_id"),
                "init_state_id": episode.get("init_state_id"),
                "attempt_index": int(episode.get("attempt_index") or 0),
                "success": bool(episode.get("success")),
                "status": episode.get("status"),
                "num_steps": episode.get("num_steps"),
                "return_value": episode.get("return_value"),
                "mean_policy_latency_ms": episode.get("mean_policy_latency_ms"),
                "policy_session_id": episode.get("policy_session_id"),
            }
        )
    return collected


def _summary(
    benchmark: dict[str, Any],
    *,
    run_id: str,
    session: SessionSpec,
    target: TargetSpec,
    skillruntime_id: str,
    artifact_dir: Path,
    episodes: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    verifier_attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    total = int(benchmark.get("total_episodes") or len(episodes))
    first_successes = int(benchmark.get("first_attempt_successes", benchmark.get("successes") or 0))
    final_successes = int(benchmark.get("final_successes", benchmark.get("successes") or first_successes))
    episodes_replanned = int(benchmark.get("episodes_replanned") or 0)
    recovered = int(benchmark.get("recovered_after_replan") or 0)
    summary = {
        "schema_version": "benchmark_result_v1",
        "run_id": run_id,
        "benchmark_id": session.benchmark.benchmark_id if session.benchmark else None,
        "suite_id": benchmark.get("suite") or (session.benchmark.suite_id if session.benchmark else None),
        "policy_id": session.benchmark.policy_id if session.benchmark else None,
        "execution_mode": benchmark.get("execution_mode", "target_native"),
        "verification_profile": benchmark.get("verification_profile", session.verification_profile),
        "status": benchmark.get("status"),
        "execution_success": benchmark.get("status") == "succeeded",
        "target_id": target.id,
        "skillruntime_id": skillruntime_id,
        "episodes": {
            "expected": total,
            "completed": int(benchmark.get("completed_episodes") or total),
            "valid": int(benchmark.get("valid_episodes") or total),
            "first_attempt_successful": first_successes,
            "final_outcome_successful": final_successes,
        },
        "metrics": {
            "first_attempt_score": float(first_successes / total) if total else 0.0,
            "assisted_final_score": float(final_successes / total) if total else 0.0,
            "official_first_attempt_success_rate": float(first_successes / total) if total else 0.0,
            "final_outcome_success_rate": float(final_successes / total) if total else 0.0,
            "mean_policy_latency_ms": benchmark.get("mean_policy_latency_ms"),
        },
        "replan": {
            "logical_episodes": total,
            "total_attempts": int(benchmark.get("total_attempts") or len(attempts)),
            "episodes_replanned": episodes_replanned,
            "recovered_after_replan": recovered,
            "verifier_calls": int(benchmark.get("verifier_calls") or len(verifier_attempts)),
            "verifier_skipped": int(benchmark.get("verifier_skipped") or 0),
            "configured_max_verifier_calls_per_run": int(benchmark.get("configured_max_verifier_calls_per_run") or 0),
            "effective_max_verifier_calls_per_run": int(benchmark.get("effective_max_verifier_calls_per_run") or 0),
            "remaining_verifier_calls": int(benchmark.get("remaining_verifier_calls") or 0),
        },
        "artifacts": {
            "manifest": str((artifact_dir / "manifest.json").relative_to(self_workspace(artifact_dir))),
            "summary": str((artifact_dir / "summary.json").relative_to(self_workspace(artifact_dir))),
            "tasks": str((artifact_dir / "tasks.json").relative_to(self_workspace(artifact_dir))),
            "episodes": str((artifact_dir / "episodes.jsonl").relative_to(self_workspace(artifact_dir))),
            "attempts": str((artifact_dir / "attempts.jsonl").relative_to(self_workspace(artifact_dir))),
            "verifier_attempts": str((artifact_dir / "verifier_attempts.jsonl").relative_to(self_workspace(artifact_dir))),
            "failures": str((artifact_dir / "failures.jsonl").relative_to(self_workspace(artifact_dir))),
        },
    }
    return _jsonable(summary)


def self_workspace(artifact_dir: Path) -> Path:
    # artifact_dir is <workspace>/artifacts/benchmarks/<run_id>
    return artifact_dir.parents[2]


def _manifest(
    session: SessionSpec,
    target: TargetSpec,
    skillruntime_id: str,
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    return _jsonable(
        {
            "schema_version": "benchmark_manifest_v1",
            "created_at": utc_now().isoformat(),
            "session": session.model_dump(mode="json", exclude_none=True),
            "target_id": target.id,
            "skillruntime_id": skillruntime_id,
            "suite": benchmark.get("suite"),
            "verification_profile": benchmark.get("verification_profile", session.verification_profile),
            "control_mode": benchmark.get("control_mode"),
            "max_steps": benchmark.get("max_steps"),
        }
    )


def _task_summary(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[int, dict[str, Any]] = {}
    for episode in episodes:
        task_id = int(episode.get("task_id") or 0)
        item = by_task.setdefault(
            task_id,
            {
                "task_id": task_id,
                "episodes": 0,
                "first_attempt_successful": 0,
                "final_outcome_successful": 0,
            },
        )
        item["episodes"] += 1
        item["first_attempt_successful"] += 1 if episode.get("first_attempt_success", episode.get("success")) else 0
        item["final_outcome_successful"] += 1 if episode.get("final_success", episode.get("success")) else 0
    for item in by_task.values():
        total = int(item["episodes"])
        item["first_attempt_success_rate"] = float(item["first_attempt_successful"] / total) if total else 0.0
        item["final_outcome_success_rate"] = float(item["final_outcome_successful"] / total) if total else 0.0
    return [by_task[key] for key in sorted(by_task)]


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    atomic_write_text(path, json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    text = "".join(json.dumps(_jsonable(row), sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def _safe_path_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return cleaned.strip("_")[:128] or "benchmark_run"
