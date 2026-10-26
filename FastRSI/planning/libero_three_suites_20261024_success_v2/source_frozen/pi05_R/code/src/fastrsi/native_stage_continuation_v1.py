"""Pause an opt-in root loop without ending or restarting its native Session.

The initial interface supports CONTINUE only. It is an execution-continuity
probe, not a skill selector or a learned controller. The original OpenPI loop
still owns inference, action limits and terminal success. No files, memories,
models or simulator state are updated here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import queue
import threading
import time
from typing import Any


@dataclass(frozen=True)
class StageRequest:
    request_id: str
    session_id: str
    goal: str
    physical_steps: int
    submitted_chunks: int
    observation_id: str
    observation: dict[str, Any]
    max_policy_iterations: int
    component_versions: dict[str, str]
    remaining_seconds: float
    allowed_actions: tuple[str, ...] = ("continue",)


class StageInterrupted(RuntimeError):
    def __init__(self, status: str, code: str):
        self.status = status
        self.code = code
        super().__init__(code)


class NativeStageContinuation:
    """A bounded rendezvous between a native runner and an external consumer.

    Registered elapsed steps are for interface checks only, not learned stage
    labels. A future controller must bind genuine events to available evidence.
    Cancellation and waiting consume the original Session's execution timeout.
    Requests contain copies of the existing legal runtime observation, never
    the private target handle or hidden simulator state.
    """
    def __init__(self, *, session_id: str, goal: str, pause_steps: tuple[int, ...],
                 component_versions: dict[str, str], emit=None):
        if not session_id or not goal or not isinstance(pause_steps, tuple):
            raise ValueError("Bind the root Session, original goal and boundary tuple")
        if (any(type(s) is not int or s < 0 for s in pause_steps)
                or tuple(sorted(set(pause_steps))) != pause_steps):
            raise ValueError("Boundary steps must be distinct ascending nonnegative integers")
        if not component_versions or any(not isinstance(k, str) or not isinstance(v, str)
                                         or not k or not v for k, v in component_versions.items()):
            raise ValueError("Bind the existing component versions")
        self.session_id = session_id
        self.goal = goal
        self.pause_steps = pause_steps
        self.component_versions = dict(component_versions)
        self.emit = emit
        self.events: list[dict[str, Any]] = []
        self.requests: queue.Queue[StageRequest] = queue.Queue()
        self._condition = threading.Condition()
        self._pending: StageRequest | None = None
        self._resolved = False
        self._closed = False
        self._entered = False
        self._seen: set[int] = set()
        self.submitted_chunks = 0
        self._selected_action = "continue"

    def allowed_actions(self):
        return ("continue",)

    def validate_action(self, action):
        if action != "continue":
            raise ValueError("No alternative executable skill is bound in this interface probe")

    def resolved_boundary(self, request, action):
        """Extension point after cancellation/timeout checks, before inference."""

    def resolve(self, request_id: str, *, action: str,
                component_versions: dict[str, str]) -> None:
        """Acknowledge the exact current request; no outcome is inferred."""
        with self._condition:
            if self._closed or self._pending is None or self._resolved:
                raise ValueError("No unresolved live boundary")
            if request_id != self._pending.request_id:
                raise ValueError("Stale or foreign boundary reply")
            self.validate_action(action)
            if component_versions != self.component_versions:
                raise ValueError("Component versions changed during the root Session")
            self._resolved = True
            self._selected_action = action
            self._condition.notify_all()

    def _event(self, event: str, **fields):
        row = dict(event=event, session_id=self.session_id, goal=self.goal, **fields)
        self.events.append(row)
        if self.emit is not None:
            self.emit(dict(row))

    def _check(self, handle, deadline):
        if handle.session_state.cancelled:
            raise StageInterrupted("cancelled", "SESSION_CANCELLED")
        if time.monotonic() >= deadline:
            raise StageInterrupted("timed_out", "EXECUTION_TIMEOUT")

    def _observe(self, runtime, handle, deadline):
        self._check(handle, deadline)
        observation = handle.observe()
        step = handle.session_state.step_index
        if step not in self.pause_steps or step in self._seen:
            return observation
        self._seen.add(step)
        preparation = time.perf_counter()
        obs_id = observation.data.get("observation_id")
        if not isinstance(obs_id, str) or not obs_id:
            raise ValueError("Boundary requires an existing legal observation identity")
        request = StageRequest(
            request_id=f"{self.session_id}:{step}:{len(self._seen)}",
            session_id=self.session_id, goal=self.goal, physical_steps=step,
            submitted_chunks=self.submitted_chunks, observation_id=obs_id,
            observation=copy.deepcopy(observation.data),
            max_policy_iterations=handle.session.execution.max_steps,
            component_versions=dict(self.component_versions),
            remaining_seconds=max(0.0, deadline-time.monotonic()),
            allowed_actions=self.allowed_actions())
        previous_snapshot = dict(runtime._snapshot)
        waiting = time.perf_counter()
        self._event("native_stage_pause", request_id=request.request_id,
                    observation_id=obs_id, physical_steps=step,
                    submitted_chunks=self.submitted_chunks,
                    preparation_seconds=waiting-preparation,
                    component_versions=dict(self.component_versions),
                    authority="INTERFACE_PROBE_NOT_LEARNING_OR_STAGE_COMPLETION")
        runtime._snapshot = {**previous_snapshot, "stage_continuation": {
            "state": "waiting", "request_id": request.request_id,
            "physical_steps": step, "original_goal": self.goal}}
        try:
            with self._condition:
                self._check(handle, deadline)
                self._pending = request
                self._resolved = False
                self._selected_action = "continue"
                self.requests.put(request)
                while not self._resolved:
                    self._check(handle, deadline)
                    self._condition.wait(timeout=min(0.05, max(0.001, deadline-time.monotonic())))
                # Never resume a cancelled/timed-out task after a late reply.
                self._check(handle, deadline)
                action = self._selected_action
            self.resolved_boundary(request, action)
            self._event("native_stage_resume", request_id=request.request_id,
                        observation_id=obs_id, physical_steps=step,
                        wait_seconds=time.perf_counter()-waiting, action=action)
        finally:
            with self._condition:
                self._pending = None
                self._resolved = False
            runtime._snapshot = previous_snapshot
        # Return the actual original observation. The consumer's copy cannot
        # rewrite the subsequent native policy request.
        return observation

    def run(self, original, runtime, skill_ctx, handle, adapter_plan, policy_client):
        if skill_ctx.session.session_id != self.session_id:
            return original(runtime, skill_ctx, handle, adapter_plan, policy_client)
        if skill_ctx.session.task_description != self.goal or handle.session.task_description != self.goal:
            raise ValueError("The original whole-task goal cannot change")
        if self._entered:
            raise ValueError("A root loop cannot be restarted to reset its execution budget")
        self._entered = True
        timeout = float(skill_ctx.session.timeouts.execute_timeout_s)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Invalid original Session execution timeout")
        elapsed = max(0.0, (time.time_ns()-handle.session_state.started_at_ns)/1e9)
        deadline = time.monotonic() + max(0.0, timeout-elapsed)
        owner = self

        class SameHandle:
            def observe(self):
                return owner._observe(runtime, handle, deadline)

            def action_chunk(self, chunk):
                owner._check(handle, deadline)
                status = handle.action_chunk(chunk)
                owner.submitted_chunks += 1
                return status

            def __getattr__(self, name):
                return getattr(handle, name)

        try:
            result = original(runtime, skill_ctx, SameHandle(), adapter_plan, policy_client)
        except StageInterrupted as exc:
            from PhyAgentOS.runtime.sessions.models import SkillRuntimeResult
            self._event("native_stage_interrupted", status=exc.status, error_code=exc.code)
            result = SkillRuntimeResult(status=exc.status, success=False,
                final_status=dict(handle.session_state.last_status),
                error_code=exc.code, error_message=str(exc))
        finally:
            with self._condition:
                self._closed = True
                self._pending = None
                self._condition.notify_all()
        return result

    def audit_snapshot(self):
        """Out-of-band diagnostics; never amend the native verification result."""
        return dict(
            authority="INTERFACE_PROBE_NOT_LEARNING_OR_STAGE_COMPLETION",
            registered_steps=list(self.pause_steps), visited_steps=sorted(self._seen),
            submitted_chunks=self.submitted_chunks, events=copy.deepcopy(self.events),
            executable_alternatives_bound=False, stage_hook_persistent_updates=False)

    def install(self):
        """Explicit process-local opt-in; no changes to the frozen SDK files."""
        from PhyAgentOS.runtime.skillruntime.policy.openpi import OpenPISkillRuntime
        original = OpenPISkillRuntime.run_policy_loop
        if getattr(original, "_fastrsi_stage_gate", False):
            raise ValueError("A continuation gate is already installed in this process")
        def wrapped(runtime, skill_ctx, handle, adapter_plan, policy_client):
            return self.run(original, runtime, skill_ctx, handle, adapter_plan, policy_client)
        wrapped._fastrsi_stage_gate = True
        OpenPISkillRuntime.run_policy_loop = wrapped
        def restore():
            if OpenPISkillRuntime.run_policy_loop is not wrapped:
                raise RuntimeError("Another installer replaced the native runtime during this probe")
            OpenPISkillRuntime.run_policy_loop = original
        return restore
