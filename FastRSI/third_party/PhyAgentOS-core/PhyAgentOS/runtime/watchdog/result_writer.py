"""Write runtime results to artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PhyAgentOS.runtime.artifacts.benchmark_writer import BenchmarkArtifactWriter, compact_benchmark_result
from PhyAgentOS.runtime.artifacts.episode_writer import EpisodeWriter
from PhyAgentOS.runtime.schemas import SessionResult, SessionSpec, TargetSpec
from PhyAgentOS.runtime.schemas.common import utc_now
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block


class ResultWriter:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.benchmark_writer = BenchmarkArtifactWriter(workspace)
        self.episode_writer = EpisodeWriter(workspace / "artifacts" / "runtime")

    def write_episode(
        self,
        session: SessionSpec,
        target: TargetSpec,
        skillruntime_id: str,
        result: SessionResult,
    ) -> SessionResult:
        benchmark = result.metadata.get("benchmark_result")
        if isinstance(benchmark, dict) and benchmark.get("total_episodes") is not None:
            summary = self.benchmark_writer.write_benchmark(session, target, skillruntime_id, result)
            result.metadata["benchmark_summary"] = summary
            result.metadata["benchmark_result"] = compact_benchmark_result(benchmark, summary)
        artifact_dir = self.episode_writer.write_episode(session, target, skillruntime_id, result)
        result.artifact_dir = str(artifact_dir.relative_to(self.workspace))
        return result

    def write_session_history(
        self,
        session: SessionSpec,
        target: TargetSpec,
        result: SessionResult,
    ) -> None:
        """Write transient runtime session history outside ENVIRONMENT.md."""
        path = self.workspace / "LOG.md"
        history = self._load_session_history(path)
        sessions = history.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}

        summary = {
            "session_id": session.session_id,
            "target_id": target.id,
            "status": result.status,
            "success": bool(result.success),
            "artifact_dir": result.artifact_dir or "",
            "num_steps": result.num_steps,
            "return_value": result.return_value,
            "mean_policy_latency_ms": result.mean_policy_latency_ms,
            "error_code": result.error_code,
            "error_message": result.error_message,
            "trace_path": result.trace_path,
            "updated_at": utc_now().isoformat(),
        }
        sessions[session.session_id] = {key: value for key, value in summary.items() if value is not None}
        history.update(
            {
                "version": "runtime_session_history_v1",
                "updated_at": utc_now().isoformat(),
                "last_session_id": session.session_id,
                "last_target_id": target.id,
                "last_status": result.status,
                "last_success": bool(result.success),
                "last_artifact_dir": result.artifact_dir or "",
                "sessions": sessions,
            }
        )
        write_yaml_block(path, "Runtime Session History", history)

    def write_verification_bundle(
        self,
        session: SessionSpec,
        target: TargetSpec,
        skillruntime_id: str,
        result: SessionResult,
        *,
        environment_workspace: Path,
        initial_observation: dict[str, Any] | None,
        final_observation: dict[str, Any] | None,
    ) -> Path:
        """Write the immutable evidence consumed by the Agent verifier."""
        if not result.artifact_dir:
            raise ValueError("verification bundle requires an episode artifact directory")
        artifact_dir = self.workspace / result.artifact_dir
        initial_paths = self.episode_writer.write_rgb_frames(artifact_dir, initial_observation, phase="initial")
        final_paths = self.episode_writer.write_rgb_frames(artifact_dir, final_observation, phase="final")
        payload = {
            "version": "agent_session_verification_v2",
            "session": session.model_dump(mode="json", exclude_none=True),
            "target_id": target.id,
            "skillruntime_id": skillruntime_id,
            "runtime_result": result.model_dump(mode="json", exclude_none=True),
            "initial_rgb_paths": [str(path.relative_to(self.workspace)) for path in initial_paths],
            "final_rgb_paths": [str(path.relative_to(self.workspace)) for path in final_paths],
            "environment_md": self._read_optional(environment_workspace / "ENVIRONMENT.md"),
            "history_md": self._read_optional(self.workspace / "LOG.md"),
            "lessons_md": self._read_optional(self.workspace / "LESSONS.md"),
            "created_at": utc_now().isoformat(),
        }
        path = artifact_dir / "verification_bundle.json"
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return path

    def _load_session_history(self, path: Path) -> dict:
        if not path.exists():
            return {"version": "runtime_session_history_v1", "sessions": {}}
        try:
            payload = read_yaml_block(path)
        except Exception:
            return {"version": "runtime_session_history_v1", "sessions": {}}
        if payload.get("version") != "runtime_session_history_v1":
            return {"version": "runtime_session_history_v1", "sessions": {}}
        return payload

    def write_lesson(
        self,
        session: SessionSpec,
        target_id: str,
        skillruntime_id: str,
        phase: str,
        error_code: str | None,
        summary: str,
        metadata: dict,
    ) -> None:
        path = self.workspace / "LESSONS.md"
        payload = self._load_lessons(path)
        lessons = payload.get("lessons")
        if not isinstance(lessons, list):
            lessons = []
        lessons.append(
            {
                "id": f"lesson_{session.session_id}_{len(lessons) + 1}",
                "timestamp": utc_now().isoformat(),
                "session_id": session.session_id,
                "phase": phase,
                "error_code": error_code,
                "target_id": target_id,
                "skillruntime_id": skillruntime_id,
                "summary": summary,
                "metadata": metadata,
            }
        )
        write_yaml_block(
            path,
            "Runtime Lessons",
            {"version": "runtime_lessons_v1", "updated_at": utc_now().isoformat(), "lessons": lessons},
        )

    def _load_lessons(self, path: Path) -> dict:
        if not path.exists():
            return {"version": "runtime_lessons_v1", "lessons": []}
        try:
            payload = read_yaml_block(path)
        except Exception:
            return {"version": "runtime_lessons_v1", "lessons": []}
        if not isinstance(payload.get("lessons"), list):
            payload["lessons"] = []
        return payload

    @staticmethod
    def _read_optional(path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""
