"""Explicit development-only completion of an already submitted stage chunk.

The native target stops a chunk at its first success.  This opt-in helper may
consume only the remainder of that same chunk before changing stages.  It owns
no model, reset, goal definition or additional execution budget.  A successful
predicate is a fact at a particular step, never a release certificate.
"""
import copy
import hashlib
import json
import math
import time


SCOPE = "submitted_stage_handoff_tail_candidate_not_learning_or_release_certificate"


def _actions(value):
    rows = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty submitted [T,7] action chunk is required")
    result = []
    for row in rows:
        if (not isinstance(row, (list, tuple)) or len(row) != 7
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in row)):
            raise ValueError("Submitted actions must be seven finite numeric values")
        result.append([float(x) for x in row])
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


class StageHandoffTail:
    """Bind submission and optional draining to one SameWorldContinuation.

    ``submit(chunk)`` returns the original target status.  Call ``consume``
    immediately after recording that original result, before another action or
    stage activation.  Disabled mode delegates submission and does no extra
    checks or actions on consumption.  Enabled mode applies only to activation
    one and native-success early stopping; later stages have no tail permission.

    Optional ``action_observer(chunk, current_status, row_event)`` can capture
    images after each real row.  ``event_sink(event)`` receives separate tail
    facts.  Their measured spans are contained in consume time.  Exceptions
    remain exceptions; partial physical work is retained in ``last_report``.
    """
    def __init__(self, owner, *, enabled=False):
        if type(enabled) is not bool:
            raise ValueError("Tail execution must be explicitly boolean opt-in")
        self.owner = owner
        self.enabled = enabled
        self._pending = None
        self.last_report = None

    def submit(self, chunk):
        if not self.enabled:
            return self.owner.action_chunk(chunk)
        if self._pending is not None:
            raise RuntimeError("Consume or reject the previous bound chunk first")
        self.owner._check_identity()
        if not isinstance(chunk, dict):
            raise ValueError("Native action chunk dictionary required")
        rows = _actions(chunk.get("actions"))
        if len(rows) > int(self.owner.runtime.config["max_chunk_size"]):
            raise ValueError("Submitted chunk exceeds native maximum")
        frozen = copy.deepcopy(chunk)
        frozen["actions"] = rows
        runtime = self.owner.runtime
        before = runtime.step_idx
        stage = self.owner.active
        activation = self.owner.activation_count
        goal = copy.deepcopy(runtime.env.env.parsed_problem["goal_state"])
        language = runtime.language
        tick = time.perf_counter()
        status = self.owner.action_chunk(frozen)
        seconds = time.perf_counter() - tick
        used = runtime.step_idx - before
        if not 0 < used <= len(rows):
            raise RuntimeError("Native action count outside the submitted chunk")
        self._pending = dict(chunk=frozen, rows=rows, before=before,
            after=runtime.step_idx, observation=runtime._last_obs, used=used,
            stage=stage, activation=activation, goal=goal, language=language,
            max_steps=runtime.config["max_steps"], status=copy.copy(status),
            submit_seconds=seconds, runtime_success=bool(runtime.success),
            runtime_done=bool(runtime.done))
        return status

    def _check_bound(self, bound):
        self.owner._check_identity()
        r = self.owner.runtime
        if (self.owner.active != bound["stage"]
                or self.owner.activation_count != bound["activation"]
                or r.language != bound["language"]
                or r.config["max_steps"] != bound["max_steps"]
                or r.env.env.parsed_problem["goal_state"] != bound["goal"]):
            raise RuntimeError("Stage, instruction, goal or budget changed before tail consumption")

    def _set_current_fact(self, current, *, preserve_terminal=False):
        """Refresh sticky target fields from its real predicate, preserving raw logs."""
        r = self.owner.runtime
        r.success = bool(current)
        r.done = (bool(current) or (preserve_terminal and bool(r.done))
                  or r.step_idx >= min(self.owner.total_steps, int(r.config["max_steps"])))
        updated = dict(r._last_status)
        updated.update(success=r.success, done=r.done, need_replan=not r.success,
                       current_native_goal_success=r.success)
        if isinstance(updated.get("episode_summary"), dict):
            updated["episode_summary"] = dict(updated["episode_summary"],
                                              success=r.success, done=r.done)
        r._last_status = updated
        return dict(updated)

    def consume(self, *, event_sink=None, action_observer=None):
        started = time.perf_counter()
        report = dict(scope=SCOPE, enabled=self.enabled, status="NO_ACTION",
            stop_reason="disabled", executed_tail_steps=0, rows=[],
            native_check_attempts=0, native_check_completions=0,
            new_policy_calls=0, environment_resets=0,
            release_assessment="NOT_EVALUATED", learning_claim=False,
            costs=dict(submitted_chunk_seconds=0., tail_action_seconds=0.,
                tail_oracle_seconds=0., tail_trace_seconds=0., tail_event_seconds=0.))
        self.last_report = report
        if not self.enabled:
            return report
        bound = self._pending
        if bound is None:
            raise RuntimeError("No bound submitted chunk; tail permission is single-use")
        self._pending = None
        r = self.owner.runtime
        report.update(stage=bound["stage"], source_chunk_id=bound["chunk"].get("chunk_id"),
            source_step_before=bound["before"], source_step_after=bound["after"],
            submitted_actions=copy.deepcopy(bound["rows"]),
            submitted_actions_sha256=_digest(bound["rows"]),
            originally_executed_steps=bound["used"],
            originally_executed_actions=copy.deepcopy(bound["rows"][:bound["used"]]),
            submitted_suffix=copy.deepcopy(bound["rows"][bound["used"]:]),
            original_runtime_success=bound["runtime_success"],
            original_runtime_done=bound["runtime_done"],
            physical_step_before_tail=r.step_idx, total_steps=self.owner.total_steps,
            shared_deadline=self.owner.deadline)
        report["costs"]["submitted_chunk_seconds"] = bound["submit_seconds"]

        def emit(value):
            if event_sink is not None:
                tick = time.perf_counter()
                event_sink(copy.deepcopy(value))
                report["costs"]["tail_event_seconds"] += time.perf_counter()-tick

        def check_success():
            tick = time.perf_counter()
            report["native_check_attempts"] += 1
            try:
                current = bool(r.env.check_success())
                report["native_check_completions"] += 1
                return current
            finally:
                report["costs"]["tail_oracle_seconds"] += time.perf_counter()-tick

        try:
            self._check_bound(bound)
            if r.step_idx != bound["after"] or r._last_obs is not bound["observation"]:
                raise RuntimeError("Another action or reset replaced the bound submitted result")
            if (bound["activation"] != 1 or not bound["runtime_success"]
                    or not bound["runtime_done"] or bound["used"] == len(bound["rows"])):
                report["stop_reason"] = "not_first_stage_native_success_early_stop"
                return report
            current = check_success()
            report["native_success_at_original_stop"] = current
            self._set_current_fact(current, preserve_terminal=not current)
            if not current:
                report["stop_reason"] = "original_stop_not_current_native_success"
                return report
            emit(dict(event="submitted_tail_permission", stage=bound["stage"],
                step=r.step_idx, native_success_at_stop=current,
                original_runtime_success=bound["runtime_success"],
                original_runtime_done=bound["runtime_done"],
                submitted_actions_sha256=report["submitted_actions_sha256"],
                suffix=copy.deepcopy(report["submitted_suffix"]), scope=SCOPE))
            report["status"] = "COMPLETED"
            report["stop_reason"] = "submitted_suffix_consumed"
            for index in range(bound["used"], len(bound["rows"])):
                self._check_bound(bound)
                cap = min(self.owner.total_steps, int(r.config["max_steps"]))
                if r.step_idx >= cap:
                    report.update(status="PARTIAL_STOP", stop_reason="shared_budget_exhausted")
                    break
                if self.owner.clock() >= self.owner.deadline:
                    report.update(status="PARTIAL_STOP", stop_reason="shared_deadline_exhausted")
                    break
                # Permission to finish this committed row only.  Keep the old
                # success as a historical fact; recompute current truth below.
                r.done = False
                chunk = dict(bound["chunk"], actions=[copy.deepcopy(bound["rows"][index])],
                    chunk_id=str(bound["chunk"].get("chunk_id", "chunk"))+"_tail_"+str(index))
                before = r.step_idx
                tick = time.perf_counter()
                try:
                    raw_status = self.owner.action_chunk(chunk)
                except BaseException as error:
                    # A native exception can follow a real row.  Never hide
                    # that cost or silently allow a stale owner to proceed.
                    delta = r.step_idx-before
                    if (r.env is self.owner.environment and r._env_key == self.owner.anchor_key
                            and delta in (0, 1)):
                        self.owner.expected_step = r.step_idx
                        self.owner.expected_observation = r._last_obs
                        report["executed_tail_steps"] += delta
                    report["rows"].append(dict(event="submitted_tail_row_exception",
                        stage=bound["stage"], submitted_row_index=index,
                        step_before=before, step_after=r.step_idx, executed_steps=delta,
                        action=copy.deepcopy(bound["rows"][index]),
                        current_native_goal_success=None,
                        action_seconds=time.perf_counter()-tick, error=repr(error), scope=SCOPE))
                    raise
                finally:
                    action_seconds = time.perf_counter()-tick
                    report["costs"]["tail_action_seconds"] += action_seconds
                delta = r.step_idx-before
                report["executed_tail_steps"] += delta
                if delta != 1:
                    raise RuntimeError("One submitted tail row must execute exactly one physical step")
                self._check_bound(bound)
                oracle_before = report["costs"]["tail_oracle_seconds"]
                try:
                    current = check_success()
                except BaseException as error:
                    report["rows"].append(dict(event="submitted_tail_native_check_exception",
                        stage=bound["stage"], submitted_row_index=index,
                        step_before=before, step_after=r.step_idx, executed_steps=1,
                        action=copy.deepcopy(bound["rows"][index]),
                        raw_native_status=dict(success=bool(raw_status.get("success")),
                                               done=bool(raw_status.get("done"))),
                        current_native_goal_success=None, action_seconds=action_seconds,
                        native_check_seconds=report["costs"]["tail_oracle_seconds"]-oracle_before,
                        observation_id=r._last_obs.get("observation_id"),
                        error=repr(error), scope=SCOPE))
                    raise
                status = self._set_current_fact(current, preserve_terminal=not current)
                row_observation = r._last_obs
                non_goal_terminal = (bool(raw_status.get("done")) and not current
                                     and r.step_idx < cap)
                row = dict(event="submitted_tail_row", stage=bound["stage"],
                    submitted_row_index=index, step_before=before, step_after=r.step_idx,
                    action=copy.deepcopy(bound["rows"][index]),
                    action_seconds=action_seconds,
                    native_check_seconds=report["costs"]["tail_oracle_seconds"]-oracle_before,
                    stage_terminal_permission_cleared=True,
                    raw_native_status=dict(success=bool(raw_status.get("success")),
                                           done=bool(raw_status.get("done"))),
                    current_native_goal_success=current,
                    current_runtime_success=bool(r.success), current_runtime_done=bool(r.done),
                    native_terminal_not_current_goal=non_goal_terminal,
                    observation_id=r._last_obs.get("observation_id"),
                    release_assessment="NOT_EVALUATED", scope=SCOPE)
                report["rows"].append(row)
                if action_observer is not None:
                    tick = time.perf_counter()
                    try:
                        row["trace"] = action_observer(copy.deepcopy(chunk), status, copy.deepcopy(row))
                    finally:
                        report["costs"]["tail_trace_seconds"] += time.perf_counter()-tick
                emit(row)
                self._check_bound(bound)
                if r.step_idx != before+1 or r._last_obs is not row_observation:
                    raise RuntimeError("Tail observation/event callback changed the owned stream")
                if non_goal_terminal:
                    report.update(status="PARTIAL_STOP", stop_reason="native_terminal_not_current_goal")
                    break
            report["native_success_after_tail"] = bool(r.success)
            report["unconsumed_suffix"] = copy.deepcopy(bound["rows"][
                bound["used"]+report["executed_tail_steps"]:])
            return report
        except BaseException as error:
            report.update(status="FAILED", stop_reason="tail_exception", error=repr(error))
            raise
        finally:
            report["unconsumed_suffix"] = copy.deepcopy(bound["rows"][
                bound["used"]+report["executed_tail_steps"]:])
            report["physical_step_after_tail"] = r.step_idx
            report["remaining_shared_steps"] = max(0, self.owner.total_steps-r.step_idx)
            report["consume_seconds"] = time.perf_counter()-started
            report["cost_scope"] = ("Tail action/oracle/trace/event spans are contained in consume_seconds; "
                "submitted_chunk_seconds is the preceding original action span. No model or reset is performed.")
