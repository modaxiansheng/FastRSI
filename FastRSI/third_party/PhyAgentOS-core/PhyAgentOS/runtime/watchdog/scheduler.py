"""Session scheduling for the serial runtime watchdog."""

from __future__ import annotations

from dataclasses import dataclass

from PhyAgentOS.runtime.schemas import (
    SessionsDocument,
    SessionSpec,
    SessionStatus,
    SkillRuntimeSpec,
    SkillRuntimeDocument,
    TargetSpec,
    TargetsDocument,
)
from PhyAgentOS.runtime.schemas.common import strip_ref
from PhyAgentOS.runtime.watchdog.errors import SchemaValidationError


@dataclass(frozen=True)
class ScheduledSession:
    session: SessionSpec
    target_spec: TargetSpec
    skillruntime_spec: SkillRuntimeSpec
    target_id: str
    skillruntime_id: str


class SessionScheduleError(SchemaValidationError):
    """Raised when a specific pending session cannot be scheduled."""

    def __init__(self, session_id: str, message: str):
        super().__init__(message)
        self.session_id = session_id


class SessionScheduler:
    """Pick the next executable pending session for a serial worker."""

    _PRIORITY_RANK = {"high": 0, "normal": 1, "low": 2}

    def select_by_id(
        self,
        session_id: str,
        sessions_doc: SessionsDocument,
        targets_doc: TargetsDocument,
        skillruntimes_doc: SkillRuntimeDocument,
    ) -> ScheduledSession:
        for session in sessions_doc.sessions:
            if session.session_id == session_id:
                if session.status != SessionStatus.PENDING:
                    raise SessionScheduleError(
                        session_id,
                        f"session {session_id!r} is not pending (status={session.status})",
                    )
                try:
                    return self.resolve_session(session, targets_doc, skillruntimes_doc)
                except SchemaValidationError as exc:
                    raise SessionScheduleError(session_id, str(exc)) from exc
        raise SessionScheduleError(session_id, f"session not found: {session_id}")

    def select_next(
        self,
        sessions_doc: SessionsDocument,
        targets_doc: TargetsDocument,
        skillruntimes_doc: SkillRuntimeDocument,
        *,
        session_id: str | None = None,
    ) -> ScheduledSession | None:
        if session_id:
            return self.select_by_id(session_id, sessions_doc, targets_doc, skillruntimes_doc)
        pending = [
            (idx, session)
            for idx, session in enumerate(sessions_doc.sessions)
            if session.status == SessionStatus.PENDING
        ]
        if not pending:
            return None
        _, session = min(
            pending,
            key=lambda item: (self._PRIORITY_RANK.get(item[1].priority, 1), item[0]),
        )
        try:
            return self.resolve_session(session, targets_doc, skillruntimes_doc)
        except SchemaValidationError as exc:
            raise SessionScheduleError(session.session_id, str(exc)) from exc

    def resolve_session(
        self,
        session: SessionSpec,
        targets_doc: TargetsDocument,
        skillruntimes_doc: SkillRuntimeDocument,
    ) -> ScheduledSession:
        target_id = strip_ref(session.target_ref, "target://")
        skillruntime_id = strip_ref(session.skillruntime_ref, "skillruntime://")
        target_spec = self._find_target(targets_doc, target_id)
        skillruntime_spec = self._find_skillruntime(skillruntimes_doc, skillruntime_id)
        if skillruntime_id not in target_spec.supported_skillruntimes:
            raise SchemaValidationError(f"target {target_id} does not support skillruntime {skillruntime_id}")
        if target_spec.target_kind not in skillruntime_spec.supported_target_kinds:
            raise SchemaValidationError(
                f"skillruntime {skillruntime_id} does not support target kind {target_spec.target_kind}"
            )
        return ScheduledSession(
            session=session,
            target_spec=target_spec,
            skillruntime_spec=skillruntime_spec,
            target_id=target_id,
            skillruntime_id=skillruntime_id,
        )

    def _find_target(self, document: TargetsDocument, target_id: str) -> TargetSpec:
        for target in document.targets:
            if target.id == target_id:
                return target
        raise SchemaValidationError(f"target not found: {target_id}")

    def _find_skillruntime(self, document: SkillRuntimeDocument, skillruntime_id: str) -> SkillRuntimeSpec:
        for skillruntime in document.skillruntimes:
            if skillruntime.id == skillruntime_id:
                return skillruntime
        raise SchemaValidationError(f"skillruntime not found: {skillruntime_id}")
