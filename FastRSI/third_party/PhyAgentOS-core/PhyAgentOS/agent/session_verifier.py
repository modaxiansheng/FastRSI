"""Agent-owned semantic verification for completed runtime sessions."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import socket
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from PhyAgentOS.providers.base import LLMProvider
from PhyAgentOS.runtime.schemas import SessionResult, SessionSpec, SessionStatus
from PhyAgentOS.runtime.schemas.common import utc_now
from PhyAgentOS.runtime.state_io.atomic_file import atomic_write_text
from PhyAgentOS.runtime.watchdog.registry import SessionRegistry
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter
from PhyAgentOS.verification.engine import VerificationEngine
from PhyAgentOS.verification.service import VerificationServiceProcess

logger = logging.getLogger(__name__)

_REVIEWABLE_STATUSES = {
    SessionStatus.SUCCEEDED,
    SessionStatus.FAILED,
    SessionStatus.REPLANNED,
}


class VerificationEvidenceError(ValueError):
    """Raised when persisted evidence cannot support a verification attempt."""


class VerificationVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["success", "replan", "failure"]
    evidence: list[str] = Field(min_length=1)
    failure_reason: str | None = None
    replan_task_description: str | None = None
    lesson: str = Field(min_length=1)
    verifier_status: Literal["completed", "invalid_response"] = "completed"

    @model_validator(mode="after")
    def validate_verdict_fields(self) -> "VerificationVerdict":
        if self.verdict == "replan" and not self.replan_task_description:
            raise ValueError("replan verdict requires replan_task_description")
        if self.verdict == "failure" and not self.failure_reason:
            raise ValueError("failure verdict requires failure_reason")
        return self


class SessionVerifier:
    """Poll completed runtime sessions and apply Agent-level semantic verdicts."""

    def __init__(
        self,
        *,
        workspace: str | Path,
        provider: LLMProvider,
        model: str,
        max_replans: int,
        rgb_retention: Literal["all", "failed", "none"] = "failed",
        poll_interval_s: float = 1.0,
        worker_id: str | None = None,
        timeout_s: float = 180.0,
        service_host: str = "127.0.0.1",
        service_port: int = 8100,
        episode_token: str | None = None,
        service_provider_spec: dict[str, Any] | None = None,
    ):
        self.workspace = Path(workspace).expanduser()
        self.provider = provider
        self.model = model
        self.engine = VerificationEngine(provider=provider, model=model, timeout_s=timeout_s)
        self.max_replans = max_replans
        self.rgb_retention = rgb_retention
        self.poll_interval_s = max(0.05, float(poll_interval_s))
        self.worker_id = worker_id or f"agent-verifier@{socket.gethostname()}"
        self.registry = SessionRegistry(self.workspace / "SESSIONS.md")
        self.result_writer = ResultWriter(self.workspace)
        self._stopped = False
        self.service = VerificationServiceProcess(
            engine=self.engine,
            host=service_host,
            port=service_port,
            episode_token=episode_token or "episode-verification-disabled",
            provider_spec=service_provider_spec,
        )

    async def start(self) -> None:
        """Start the child service and wait until its readiness probe succeeds."""
        if not self._stopped:
            await asyncio.to_thread(self.service.start)

    async def run(self) -> None:
        await self.start()
        while not self._stopped:
            ran = await self.run_once()
            if not ran:
                await asyncio.sleep(self.poll_interval_s)

    def stop(self) -> None:
        self._stopped = True
        self.service.stop()

    async def run_once(self) -> bool:
        session = self.registry.first_awaiting_verification()
        if session is None:
            return False
        outcome = await self.verify_session(session.session_id, source="auto")
        return outcome["code"] != "already_verifying"

    async def verify_session(
        self, session_id: str, *, source: Literal["auto", "tool"] = "tool"
    ) -> dict[str, Any]:
        """Apply a pending verification or review a terminal session without mutating it."""
        try:
            session = self.registry.get_session(session_id)
        except KeyError:
            return self._outcome(False, "not_found", session_id, message="session does not exist")

        if session.status == SessionStatus.AWAITING_VERIFICATION:
            if not self.registry.try_claim_verification(session_id, self.worker_id):
                return self._outcome(
                    False,
                    "already_verifying",
                    session_id,
                    status=SessionStatus.VERIFYING.value,
                    message="session verification was claimed by another worker",
                )
            return await self._run_apply(self.registry.get_session(session_id), source)

        if session.status == SessionStatus.VERIFYING:
            return self._outcome(
                False,
                "already_verifying",
                session_id,
                status=session.status.value,
                message="session verification is already in progress",
            )

        if session.status in _REVIEWABLE_STATUSES:
            return await self._run_review(session, source)

        return self._outcome(
            False,
            "not_ready",
            session_id,
            status=session.status.value,
            message="session execution has not reached a verifiable state",
        )

    async def _run_apply(self, session: SessionSpec, source: Literal["auto", "tool"]) -> dict[str, Any]:
        try:
            verdict = await self._verify(session)
        except asyncio.CancelledError:
            self.registry.release_verification(session.session_id, self.worker_id)
            raise
        except VerificationEvidenceError as exc:
            result = session.result.model_copy(deep=True)
            result.metadata["verification"] = {
                "status": "skipped", "reason": str(exc), "bundle": "agent_session_verification_v2"
            }
            self.registry.mark_finished(session.session_id, result)
            return self._outcome(True, "skipped", session.session_id, mode="apply", status=result.status, message=str(exc))
        except Exception as exc:
            logger.exception("session verification failed for %s", session.session_id)
            self._mark_verification_error(session, exc, source)
            return self._outcome(
                False,
                "verification_error",
                session.session_id,
                mode="apply",
                status=SessionStatus.FAILED.value,
                message=str(exc),
            )

        try:
            final_status, replanned_session_id = self._apply_verdict(session, verdict, source)
        except Exception as exc:
            logger.exception("failed to apply verification verdict for %s", session.session_id)
            self._mark_verification_error(session, exc, source)
            return self._outcome(
                False,
                "verification_error",
                session.session_id,
                mode="apply",
                status=SessionStatus.FAILED.value,
                message=str(exc),
            )

        try:
            retention = self._apply_rgb_retention(session.session_id, final_status)
            self._record_retention(session.session_id, retention)
            code = "verified"
            message = None
        except Exception as exc:
            logger.exception("RGB retention failed for %s", session.session_id)
            retention = {
                "policy": self.rgb_retention,
                "status": "error",
                "updated_at": utc_now().isoformat(),
                "deleted_at": None,
                "deleted_paths": [],
                "errors": [{"error": str(exc)}],
            }
            code = "verified_with_retention_error"
            message = str(exc)
            try:
                self._record_retention(session.session_id, retention)
            except Exception:
                logger.exception("failed to record RGB retention error for %s", session.session_id)

        return self._outcome(
            True,
            code,
            session.session_id,
            mode="apply",
            status=final_status.value,
            verdict=verdict,
            message=message,
            replanned_session_id=replanned_session_id,
            rgb_retention=retention,
        )

    async def _run_review(self, session: SessionSpec, source: Literal["auto", "tool"]) -> dict[str, Any]:
        try:
            verdict = await self._verify(session)
        except VerificationEvidenceError as exc:
            return self._outcome(
                False,
                "evidence_unavailable",
                session.session_id,
                mode="review",
                status=session.status.value,
                message=str(exc),
            )
        except Exception as exc:
            logger.exception("session review failed for %s", session.session_id)
            try:
                self._record_review_error(session, exc, source)
            except Exception:
                logger.exception("failed to record review error for %s", session.session_id)
            return self._outcome(
                False,
                "verification_error",
                session.session_id,
                mode="review",
                status=session.status.value,
                message=str(exc),
            )

        try:
            result = session.result.model_copy(deep=True)
            self._append_attempt(result, verdict, source=source, mode="review")
            self.registry.update_result(session.session_id, result)
            self._write_lesson(session, verdict, error_code=None, phase="agent_reverification")
        except Exception as exc:
            logger.exception("failed to record session review for %s", session.session_id)
            return self._outcome(
                False,
                "verification_error",
                session.session_id,
                mode="review",
                status=session.status.value,
                message=str(exc),
            )
        return self._outcome(
            True,
            "reviewed",
            session.session_id,
            mode="review",
            status=session.status.value,
            verdict=verdict,
        )

    async def _verify(self, session: SessionSpec) -> VerificationVerdict:
        bundle = self._load_bundle(session)
        initial_paths = bundle.get("initial_rgb_paths")
        final_paths = bundle.get("final_rgb_paths")
        if not isinstance(initial_paths, list) or not initial_paths or not isinstance(final_paths, list) or not final_paths:
            raise VerificationEvidenceError(
                "paired initial/final RGB evidence is unavailable"
            )
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "Verify whether the physical task is semantically complete. Runtime success only "
                    "means execution finished; it is not proof of task completion. Use the task, final "
                    "initial/final RGB evidence, runtime result, ENVIRONMENT.md, history, and lessons below.\n\n"
                    f"VERIFICATION_BUNDLE:\n{json.dumps(bundle, ensure_ascii=False, indent=2)}"
                ),
            }
        ]
        for relative_path in [*initial_paths, *final_paths]:
            image_path = self._workspace_path(str(relative_path))
            if not image_path.is_file():
                raise VerificationEvidenceError(f"verification RGB evidence does not exist: {relative_path}")
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )

        await self.start()
        data = await asyncio.to_thread(self.service.verify_session, content)
        return VerificationVerdict.model_validate(data)

    def _load_bundle(self, session: SessionSpec) -> dict[str, Any]:
        if not session.result.artifact_dir:
            raise VerificationEvidenceError("session has no artifact directory for verification")
        path = self._workspace_path(session.result.artifact_dir) / "verification_bundle.json"
        if not path.is_file():
            raise VerificationEvidenceError(f"verification bundle does not exist: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != "agent_session_verification_v2":
            raise VerificationEvidenceError("unsupported verification bundle version")
        return payload

    def _workspace_path(self, relative_path: str) -> Path:
        root = self.workspace.resolve()
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root):
            raise VerificationEvidenceError(
                f"verification artifact escapes runtime workspace: {relative_path}"
            )
        return path

    def _apply_verdict(
        self,
        session: SessionSpec,
        verdict: VerificationVerdict,
        source: Literal["auto", "tool"],
    ) -> tuple[SessionStatus, str | None]:
        result = session.result.model_copy(deep=True)
        self._append_attempt(result, verdict, source=source, mode="apply")
        if verdict.verdict == "success":
            self._write_lesson(session, verdict, error_code=None)
            self.registry.mark_verification_succeeded(session.session_id, result)
            return SessionStatus.SUCCEEDED, None
        if verdict.verdict == "failure":
            result.error_code = "AGENT_VERIFICATION_FAILED"
            result.error_message = verdict.failure_reason
            self._write_lesson(session, verdict, error_code=result.error_code)
            self.registry.mark_verification_failed(session.session_id, result)
            return SessionStatus.FAILED, None
        if session.replan_attempt >= self.max_replans:
            result.error_code = "AGENT_VERIFICATION_REPLAN_LIMIT_REACHED"
            result.error_message = (
                f"replan limit reached ({self.max_replans}): {verdict.replan_task_description}"
            )
            self._write_lesson(session, verdict, error_code=result.error_code)
            self.registry.mark_verification_failed(session.session_id, result)
            return SessionStatus.FAILED, None
        replanned = self._build_replanned_session(session, verdict.replan_task_description or "")
        result.metadata["replanned_session_id"] = replanned.session_id
        self._write_lesson(session, verdict, error_code=None)
        self.registry.mark_replanned(session.session_id, result, replanned)
        return SessionStatus.REPLANNED, replanned.session_id

    def _append_attempt(
        self,
        result: SessionResult,
        verdict: VerificationVerdict | None,
        *,
        source: Literal["auto", "tool"],
        mode: Literal["apply", "review"],
        error: str | None = None,
    ) -> dict[str, Any]:
        verification = result.metadata.setdefault("verification", {"attempts": []})
        attempts = verification.setdefault("attempts", [])
        attempt = {
            "attempt_id": f"verification_{uuid4().hex[:12]}",
            "created_at": utc_now().isoformat(),
            "source": source,
            "mode": mode,
            "verdict": verdict.verdict if verdict is not None else None,
            "evidence": verdict.evidence if verdict is not None else None,
            "lesson": verdict.lesson if verdict is not None else None,
            "details": verdict.model_dump(mode="json") if verdict is not None else None,
            "error": error,
        }
        attempts.append(attempt)
        if mode == "apply":
            verification["applied_attempt_id"] = attempt["attempt_id"]
        else:
            verification["latest_review_attempt_id"] = attempt["attempt_id"]
        return attempt

    def _apply_rgb_retention(
        self, session_id: str, final_status: SessionStatus
    ) -> dict[str, Any]:
        should_delete = self.rgb_retention == "none" or (
            self.rgb_retention == "failed" and final_status == SessionStatus.SUCCEEDED
        )
        session = self.registry.get_session(session_id)
        bundle = self._load_bundle(session)
        paths = [str(path) for key in ("initial_rgb_paths", "final_rgb_paths") for path in bundle.get(key, [])]
        deleted: list[str] = []
        remaining: list[str] = []
        errors: list[dict[str, str]] = []

        if should_delete:
            for relative_path in paths:
                try:
                    path = self._workspace_path(relative_path)
                    path.unlink(missing_ok=True)
                    deleted.append(relative_path)
                except Exception as exc:
                    remaining.append(relative_path)
                    errors.append({"path": relative_path, "error": str(exc)})
        else:
            remaining = paths

        now = utc_now().isoformat()
        record = {
            "policy": self.rgb_retention,
            "status": (
                "retained"
                if not should_delete
                else "deleted"
                if not errors
                else "partial"
            ),
            "updated_at": now,
            "deleted_at": now if deleted else None,
            "deleted_paths": deleted,
            "errors": errors,
        }
        remaining_set = set(remaining)
        bundle["initial_rgb_paths"] = [path for path in bundle.get("initial_rgb_paths", []) if path in remaining_set]
        bundle["final_rgb_paths"] = [path for path in bundle.get("final_rgb_paths", []) if path in remaining_set]
        bundle["rgb_retention"] = record
        bundle_path = self._workspace_path(session.result.artifact_dir or "") / "verification_bundle.json"
        atomic_write_text(bundle_path, json.dumps(bundle, indent=2, sort_keys=True) + "\n")

        rgb_dirs = {self._workspace_path(path).parent for path in deleted}
        for rgb_dir in rgb_dirs:
            try:
                rgb_dir.rmdir()
            except OSError:
                pass
        return record

    def _record_retention(self, session_id: str, retention: dict[str, Any]) -> None:
        session = self.registry.get_session(session_id)
        result = session.result.model_copy(deep=True)
        verification = result.metadata.setdefault("verification", {"attempts": []})
        verification["rgb_retention"] = retention
        self.registry.update_result(session_id, result)

    def _build_replanned_session(self, session: SessionSpec, task_description: str) -> SessionSpec:
        attempt = session.replan_attempt + 1
        replanned = session.model_copy(deep=True)
        replanned.session_id = f"{session.session_id}_replan_{attempt}"
        replanned.parent_session_id = session.session_id
        replanned.replan_attempt = attempt
        replanned.task_description = task_description
        replanned.status = SessionStatus.PENDING
        replanned.created_at = utc_now()
        replanned.updated_at = replanned.created_at
        replanned.claimed_by = None
        replanned.claim_token = None
        replanned.retry.attempted = 0
        replanned.result = SessionResult()
        return replanned

    def _mark_verification_error(
        self, session: SessionSpec, error: Exception, source: Literal["auto", "tool"]
    ) -> None:
        verdict = VerificationVerdict(
            verdict="failure",
            evidence=["Agent verification could not produce a valid semantic verdict."],
            failure_reason=str(error),
            lesson=f"Verification failed for session {session.session_id}: {error}",
        )
        result = session.result.model_copy(deep=True)
        result.error_code = "AGENT_VERIFICATION_ERROR"
        result.error_message = str(error)
        self._append_attempt(result, None, source=source, mode="apply", error=str(error))
        self._write_lesson(session, verdict, error_code=result.error_code)
        self.registry.mark_verification_failed(session.session_id, result)

    def _record_review_error(
        self, session: SessionSpec, error: Exception, source: Literal["auto", "tool"]
    ) -> None:
        result = session.result.model_copy(deep=True)
        self._append_attempt(result, None, source=source, mode="review", error=str(error))
        self.registry.update_result(session.session_id, result)

    def _write_lesson(
        self,
        session: SessionSpec,
        verdict: VerificationVerdict,
        error_code: str | None,
        *,
        phase: str = "agent_verification",
    ) -> None:
        self.result_writer.write_lesson(
            session,
            session.target_ref.removeprefix("target://"),
            session.skillruntime_ref.removeprefix("skillruntime://"),
            phase,
            error_code,
            verdict.lesson,
            verdict.model_dump(mode="json"),
        )

    @staticmethod
    def _outcome(
        ok: bool,
        code: str,
        session_id: str,
        *,
        mode: str | None = None,
        status: str | None = None,
        verdict: VerificationVerdict | None = None,
        message: str | None = None,
        replanned_session_id: str | None = None,
        rgb_retention: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "ok": ok,
                "code": code,
                "session_id": session_id,
                "mode": mode,
                "status": status,
                "verdict": verdict.verdict if verdict is not None else None,
                "evidence": verdict.evidence if verdict is not None else None,
                "lesson": verdict.lesson if verdict is not None else None,
                "message": message,
                "replanned_session_id": replanned_session_id,
                "rgb_retention": rgb_retention,
            }.items()
            if value is not None
        }
