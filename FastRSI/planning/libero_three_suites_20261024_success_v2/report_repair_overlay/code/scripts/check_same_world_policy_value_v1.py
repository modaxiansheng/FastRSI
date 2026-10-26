"""Bounded registered policy diagnostic in one native LIBERO world.

Stage completion is native environment oracle feedback. This is an operation-
value upper-bound diagnostic, not learning, an official episode score, or a
complete Agent comparison. No Agent, Qwen, persistent memory or training runs.
"""
import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fastrsi.same_world_continuation import SameWorldContinuation, SCOPE
from stage_trace import capture

DIAGNOSTIC_SCOPE = "native_oracle_stage_feedback_operation_value_upper_bound_not_learning_or_full_agent"
ORDER_TASKS = (
    ("goal6", 6, "put the cream cheese in the bowl",
     "81c54eace32b92466598103b3b6e8435ce90f10996ef18cec4a35fe6c4d31a98"),
    ("goal8", 8, "put the bowl on the plate",
     "358b7781437e412c67d6a7648c1c28796b6b1bf117a25e61316cb8dfeb940e5b"),
)
ORDER_ARMS = {"order68": ["goal6", "goal8"], "order86": ["goal8", "goal6"]}
ORDER_WORLD = "4f0e6df6419194a5c60dc4e0b35df8b39930ba68a2de77d3af64cdef477170e6"
CONTEXT_GENERALIZATION_PROTOCOL = "native_order_heldout_initial_states_v1"
HANDOFF_TAIL_PROTOCOL = "complete_submitted_first_stage_prefix_v1"
PUBLIC_RELEASE_RETRACT_PROTOCOL = "public_release_retract_after_first_stage_v1"
ORDER_RELATIONSHIPS = {
    "goal6": ("on", "cream_cheese_1", "akita_black_bowl_1"),
    "goal8": ("on", "akita_black_bowl_1", "plate_1"),
}


def canonical(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected lowercase SHA256")
    return value


def parse_bddl_definition(text):
    """Read source literals for binding only, never substitute for native parsing."""
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unmatched BDDL close")
            stack.pop()
        elif stack:
            stack[-1].append(token)
        else:
            raise ValueError("Atom outside BDDL root")
    if stack or len(roots) != 1 or not roots[0] or roots[0][0] != "define":
        raise ValueError("Invalid BDDL definition")
    if any(not isinstance(node, list) or not node for node in roots[0][1:]):
        raise ValueError("Malformed BDDL field")
    return roots[0]


def parse_world_definition(text):
    """Independently parse the inventory's conservative S-expression signature."""
    world = [node for node in parse_bddl_definition(text)[1:]
             if node[0] not in (":language", ":goal", ":obj_of_interest")]
    return world, sha(canonical(world))


def validate_registration(registration):
    required = {"version", "suite", "anchor_task_id", "init_state_id", "seed",
                "total_steps", "replan_steps", "timeout_seconds", "policy_timeout_seconds",
                "tasks", "policy_metadata_expected", "source_sha256"}
    if not isinstance(registration, dict) or not required.issubset(registration):
        raise ValueError("Incomplete registered diagnostic")
    version = registration["version"]
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Only declared protocol version 1 or 2 supported")
    context_protocol = registration.get("context_generalization_protocol")
    if "context_generalization_protocol" in registration:
        allowed = registration.get("allowed_initial_state_ids")
        if (context_protocol != CONTEXT_GENERALIZATION_PROTOCOL or version != 2
                or not isinstance(allowed, list) or not allowed
                or any(type(value) is not int or value < 0 for value in allowed)
                or len(set(allowed)) != len(allowed)
                or type(registration["init_state_id"]) is not int
                or registration["init_state_id"] not in allowed):
            raise ValueError("Context opt-in requires version2 and a registered unique initialization list")
        if registration["policy_metadata_expected"] != dict(backend="openpi_native",
                policy_config="pi05_libero", action_dim=7, action_horizon=10,
                torch_seed=7, torch_compile_disabled=True):
            raise ValueError("Context comparison requires unchanged native pi05 metadata")
        checked_sha(registration.get("anchor_init_source_sha256"))
    fixed = dict(suite="libero_goal", anchor_task_id=0 if version == 1 else 6,
                 init_state_id=registration["init_state_id"] if context_protocol else 0,
                 seed=7, total_steps=600, replan_steps=10)
    for key, value in fixed.items():
        if registration[key] != value or (type(value) is int and type(registration[key]) is not int):
            raise ValueError("Frozen protocol differs: " + key)
    for key in ("timeout_seconds", "policy_timeout_seconds"):
        value = registration[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1800:
            raise ValueError("Finite bounded timeout required")
    if version == 1:
        if registration.get("combined_instruction") != "open the middle drawer of the cabinet and turn on the stove":
            raise ValueError("Joint instruction changed")
        expected_tasks = (
            ("goal0", 0, "open the middle drawer of the cabinet", None),
            ("goal7", 7, "turn on the stove", None),
        )
    else:
        if registration.get("arm_orders") != ORDER_ARMS:
            raise ValueError("Both exact predeclared order68/order86 sequences required")
        expected_tasks = ORDER_TASKS
    tasks = registration["tasks"]
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise ValueError("Exactly the registered two goals required")
    for row, (key, tid, language, bddl_sha256) in zip(tasks, expected_tasks):
        if (not isinstance(row, dict) or row.get("key") != key or row.get("task_id") != tid or type(row.get("task_id")) is not int
                or row.get("language") != language):
            raise ValueError("Native task order or prompt changed")
        checked_sha(row.get("bddl_sha256")); checked_sha(row.get("world_signature"))
        if version == 2 and (row["bddl_sha256"] != bddl_sha256 or row["world_signature"] != ORDER_WORLD):
            raise ValueError("Declared task6/task8 native source binding changed")
    if tasks[0]["world_signature"] != tasks[1]["world_signature"]:
        raise ValueError("Registered worlds differ")
    if not isinstance(registration["policy_metadata_expected"], dict) or not registration["policy_metadata_expected"]:
        raise ValueError("Expected frozen policy metadata required")
    expected_sources = {"runtime_server", "same_world_continuation", "stage_trace"}
    interventions = [key for key in ("stage_handoff_tail", "public_release_retract")
                     if key in registration]
    if len(interventions) > 1:
        raise ValueError("Only one registered intervention may run")
    if interventions:
        intervention_key = interventions[0]
        candidate = registration[intervention_key]
        protocol = (HANDOFF_TAIL_PROTOCOL if intervention_key == "stage_handoff_tail"
                    else PUBLIC_RELEASE_RETRACT_PROTOCOL)
        if (not isinstance(candidate, dict) or candidate.get("protocol") != protocol
                or candidate.get("arm") != "order86" or version != 2
                or registration["init_state_id"] != 1 or not context_protocol):
            raise ValueError("Only the explicit init1/order86 development intervention is registered")
        if intervention_key == "public_release_retract":
            from fastrsi.public_release_retract_v1 import validate_registration as validate_motion
            operation = validate_motion(candidate.get("operation"))
            if operation["stage_key"] != "goal6":
                raise ValueError("Motion must run after ordinary second-stage activation")
        baseline = candidate.get("baseline_prefix")
        if (not isinstance(baseline, dict) or type(baseline.get("handoff_step")) is not int
                or not 0 < baseline["handoff_step"] <= 90
                or not isinstance(baseline.get("events"), list) or not baseline["events"]
                or len(baseline["events"]) % 2):
            raise ValueError("Exact bounded baseline prefix binding required")
        for field in ("source_result_sha256", "source_events_sha256", "initial_sim_state_sha256"):
            checked_sha(baseline.get(field))
        if set(baseline.get("initial_images", {})) != {"0000_agentview_image.png", "0000_robot0_eye_in_hand_image.png"}:
            raise ValueError("Both original reset images required")
        for digest in baseline["initial_images"].values(): checked_sha(digest)
        for index, event in enumerate(baseline["events"]):
            if (not isinstance(event, dict) or event.get("event") != ("policy_call" if index % 2 == 0 else "execution")
                    or event.get("ordinal") != index // 2 + 1):
                raise ValueError("Ordered policy/execution baseline bindings required")
            checked_sha(event.get("sha256"))
        expected_sources.add(intervention_key)
    if set(registration["source_sha256"]) != expected_sources:
        raise ValueError("Native runtime and continuation source bindings required")
    for value in registration["source_sha256"].values():
        checked_sha(value)
    return registration


def validate_initial_state_count(registration, initial_state_count):
    """Check every predeclared ID against the actual suite, before policy use."""
    validate_registration(registration)
    if type(initial_state_count) is not int or initial_state_count <= 0:
        raise ValueError("Native suite must provide a nonempty initial-state collection")
    ids = registration.get('allowed_initial_state_ids', [registration['init_state_id']])
    if any(type(value) is not int or not 0 <= value < initial_state_count for value in ids):
        raise ValueError("Registered initialization is outside the actual native suite")


def diagnostic_sequence(registration, arm):
    """Select only a declared protocol arm; never infer an unexecuted outcome."""
    if registration.get("stage_handoff_tail") and arm != registration["stage_handoff_tail"]["arm"]:
        raise ValueError("Tail candidate cannot be run on a different arm")
    if registration.get("public_release_retract") and arm != registration["public_release_retract"]["arm"]:
        raise ValueError("Motion candidate cannot be run on a different arm")
    if registration["version"] == 1:
        if arm not in ("flat", "staged"):
            raise ValueError("Version 1 supports flat/staged only")
        return ["joint"] if arm == "flat" else ["goal0", "goal7"]
    if registration["version"] != 2 or registration.get("arm_orders") != ORDER_ARMS:
        raise ValueError("Invalid registered order protocol")
    if arm not in registration["arm_orders"]:
        raise ValueError("Version 2 supports order68/order86 only")
    return list(registration["arm_orders"][arm])


def anchor_task_key(registration):
    keys = [row["key"] for row in registration["tasks"]
            if row["task_id"] == registration["anchor_task_id"]]
    if len(keys) != 1:
        raise ValueError("Exactly one registered anchor task required")
    return keys[0]


def combine_goals(stages):
    predicates = []
    for stage in stages.values():
        goal = stage["goal_state"]
        if not isinstance(goal, list) or not goal:
            raise ValueError("Missing native goal predicates")
        for clause in goal:
            if (not isinstance(clause, list) or len(clause) not in (2, 3)
                    or not all(isinstance(atom, str) for atom in clause)
                    or clause[0].lower() == "and"):
                raise ValueError("Malformed native goal predicate")
            predicates.append(copy.deepcopy(clause))
    # Native LIBERO _check_success ANDs a flat list of unary/binary predicates.
    # It does not evaluate a nested BDDL ['And', ...] expression here.
    return predicates


def bind_native_stages(registration, suite, bddl_root, native_parse):
    """SHA-check actual benchmark BDDL, native goal parse and independent world."""
    stages, bindings, first_world = {}, [], None
    for row in registration["tasks"]:
        task = suite.get_task(row["task_id"])
        if str(task.language) != row["language"]:
            raise ValueError("Benchmark task language/order differs from registration")
        relative = Path(task.problem_folder) / task.bddl_file
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe benchmark BDDL path")
        path = Path(bddl_root) / relative
        raw = path.read_bytes()
        if sha(raw) != row["bddl_sha256"]:
            raise ValueError("Registered native BDDL bytes changed")
        text = raw.decode()
        world, signature = parse_world_definition(text)
        if signature != row["world_signature"]:
            raise ValueError("Registered same-world signature does not match actual BDDL")
        if first_world is not None and world != first_world:
            raise ValueError("Native world definitions differ")
        first_world = world
        parsed = native_parse(str(path))
        native_goal = parsed["goal_state"]
        # The frozen LIBERO scanner lowercases source tokens. Bind relationship
        # name case-insensitively and objects exactly, but retain parser output
        # unchanged for the native evaluator; BDDL literal case is not its API.
        if registration["version"] == 2:
            if (not isinstance(native_goal, list) or len(native_goal) != 1
                    or not isinstance(native_goal[0], list) or len(native_goal[0]) != 3
                    or not all(isinstance(atom, str) for atom in native_goal[0])
                    or (native_goal[0][0].casefold(),) + tuple(native_goal[0][1:]) != ORDER_RELATIONSHIPS[row["key"]]):
                raise ValueError("Native task6/task8 flat on relationship differs")
        stages[row["key"]] = dict(
            language=row["language"], goal_state=copy.deepcopy(native_goal),
            world_signature=signature, max_steps=registration["total_steps"])
        bindings.append(dict(key=row["key"], task_id=row["task_id"], bddl_path=str(path),
                             bddl_sha256=sha(raw), world_signature=signature,
                             benchmark_language=str(task.language),
                             bddl_language=copy.deepcopy(parsed.get("language_instruction", parsed.get("language"))),
                             native_language_instruction_raw=copy.deepcopy(parsed.get("language_instruction")),
                             bddl_goal_literal=next(copy.deepcopy(node[1:])
                                 for node in parse_bddl_definition(text)[1:] if node[0] == ":goal"),
                             native_goal_parser=getattr(native_parse, "__module__", "") + "."
                                 + getattr(native_parse, "__name__", type(native_parse).__name__),
                             native_goal_state=copy.deepcopy(native_goal)))
    combine_goals(stages)
    return stages, bindings


def state_binding(runtime):
    value = runtime.env.get_sim_state()
    raw = value.tobytes() if hasattr(value, "tobytes") else canonical(value)
    return dict(sim_state_sha256=sha(raw), physical_steps=runtime.step_idx,
                environment_identity=id(runtime.env))


def evaluate_goals_same_state(runtime, stages):
    """Evaluate every goal and their conjunction without stepping/resetting."""
    original = copy.deepcopy(runtime.env.env.parsed_problem["goal_state"])
    before = state_binding(runtime)
    goals = {}
    try:
        for key, stage in stages.items():
            runtime.env.env.parsed_problem["goal_state"] = copy.deepcopy(stage["goal_state"])
            goals[key] = bool(runtime.env.check_success())
        runtime.env.env.parsed_problem["goal_state"] = combine_goals(stages)
        joint = bool(runtime.env.check_success())
    finally:
        runtime.env.env.parsed_problem["goal_state"] = original
    after = state_binding(runtime)
    if before != after or joint != all(goals.values()):
        raise RuntimeError("Final goal check changed physics or conjunction disagreed")
    return dict(goals=goals, joint_success=joint, same_state_before=before,
                same_state_after=after, definition="Final native predicates, not historical stage successes")


def array_binding(value):
    if hasattr(value, "tobytes") and hasattr(value, "shape") and hasattr(value, "dtype"):
        return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=sha(value.tobytes()))
    return dict(encoding="synthetic_json_test_only", sha256=sha(canonical(value)))


def prefix_event_binding(event):
    """Bind causal inputs/actions/pixels, excluding noisy clocks and new logging fields."""
    fields = {
        "policy_call": ("event", "ordinal", "stage", "step_before", "prompt", "array_bindings", "submitted_actions"),
        "execution": ("event", "ordinal", "stage", "step_before", "step_after", "actual_actions", "executed_steps",
                      "images", "native_active_goal", "runtime_success", "runtime_done", "accepted"),
    }
    return dict(event=event["event"], ordinal=event["ordinal"],
                sha256=sha(canonical({key:event[key] for key in fields[event["event"]]})))


def finite_actions(output, replan_steps):
    raw = output.get("actions") if isinstance(output, dict) else None
    rows = raw.tolist() if hasattr(raw, "tolist") else raw
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        rows = rows[0]
    if not isinstance(rows, list) or len(rows) < replan_steps:
        raise ValueError("Policy must return at least one complete W10 action prefix")
    result = []
    for row in rows:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("Expected native [T,7] policy actions")
        if any(type(value) not in (float, int) or not math.isfinite(value) for value in row):
            raise ValueError("Nonfinite or nonnumeric policy action")
        result.append([float(value) for value in row])
    return result


async def execute_diagnostic(registration, arm, runtime, policy, stages, output,
                             payload_builder, *, capture_fn=capture):
    """Concrete one-world rollout; injected target/policy allow CPU contract tests."""
    sequence = diagnostic_sequence(registration, arm)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(status="STARTING", arm=arm, scope=DIAGNOSTIC_SCOPE,
                  protocol_version=registration["version"], registered_sequence=sequence,
                  target_scope=SCOPE, environment_resets=0, physical_steps=0,
                  policy_calls=0, policy_attempts=0, qwen_calls=0, learning_updates=0,
                  oracle_feedback_calls=0, transitions=[], exceptions=[],
                  oracle_feedback_count_scope="Explicit extra diagnostic checks only. Native "
                    "activate/check_success and per-action env.step predicate checks are not "
                    "counted here; their work remains in worker/action time.",
                  reset_settling_steps=0,
                  step_scope="physical_steps counts all submitted actions after reset, including "
                    "registered interventions. Native reset "
                    "settling dummy actions are reported separately and charged to reset_seconds. "
                    "Other internal simulator reset work remains in reset time.",
                  costs=dict(reset_seconds=0., policy_seconds=0., action_seconds=0.,
                             trace_seconds=0., oracle_feedback_seconds=0.))
    if registration.get("context_generalization_protocol"):
        # Group identity is kept in results only, never passed to policy inputs.
        report["context_generalization"] = dict(
            protocol=registration["context_generalization_protocol"],
            init_state_id=registration["init_state_id"],
            use="Paired evaluation group identity; not a model feature or learned selector")
    events = output / "EVENTS.jsonl"
    tail_config = registration.get("stage_handoff_tail")
    motion_config = registration.get("public_release_retract")
    candidate = tail_config or motion_config
    baseline = candidate["baseline_prefix"] if candidate else None
    if tail_config:
        report["stage_handoff_tail"] = dict(protocol=tail_config["protocol"],
            scope="One seen development case; operation value only, not learning", reports=[])
    if motion_config:
        report["public_release_retract"] = dict(protocol=motion_config["protocol"],
            scope="One seen development case; public targets are not object-release certificates; not learning",
            reports=[])
    if candidate:
        report["baseline_prefix_binding"] = dict(status="PENDING", matched_events=0,
            source_result_sha256=baseline["source_result_sha256"],
            source_events_sha256=baseline["source_events_sha256"])

    def check_prefix_event(event):
        if not candidate or event["stage"] != sequence[0]: return
        binding = report["baseline_prefix_binding"]
        index = binding["matched_events"]
        if index >= len(baseline["events"]) or prefix_event_binding(event) != baseline["events"][index]:
            binding.update(status="MISMATCH", mismatched_event=prefix_event_binding(event))
            raise RuntimeError("Candidate first-stage baseline prefix differs; intervention not authorized")
        binding["matched_events"] += 1

    def emit(event):
        with events.open("a") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")

    def save():
        report["worker_wall_seconds"] = time.perf_counter() - started
        report["physical_steps"] = runtime.step_idx
        report["counted_target_action_steps"] = runtime.step_idx + report["reset_settling_steps"]
        (output / "RESULT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))

    def trace(method, args):
        tick = time.perf_counter()
        try:
            return capture_fn(runtime, method, args, output / "stage_trace")
        finally:
            report["costs"]["trace_seconds"] += time.perf_counter()-tick

    save()
    owner = None
    try:
        metadata = dict(getattr(policy, "metadata", {}))
        for key, value in registration["policy_metadata_expected"].items():
            if metadata.get(key) != value:
                raise ValueError("Frozen policy metadata differs: " + key)
        report["policy_metadata"] = metadata
        report["policy_metadata_sha256"] = sha(canonical(metadata))
        session = registration.get("session_id", "same_world_policy_value_v1")
        policy.reset_session(session)
        tick = time.perf_counter(); runtime.reset({"session_id": session})
        report["costs"]["reset_seconds"] = time.perf_counter()-tick
        report["environment_resets"] = 1
        report["reset_settling_steps"] = int(runtime.config.get("num_steps_wait", 0))
        anchor_key = anchor_task_key(registration)
        anchor_stage = stages[anchor_key]
        report["anchor_task_key"] = anchor_key
        if runtime.env.env.parsed_problem["goal_state"] != anchor_stage["goal_state"]:
            raise ValueError("Actual anchor goal differs from registered native definition")
        initial_trace = trace("reset", ({"session_id": session},))
        report["initial_state"] = state_binding(runtime)
        report["initial_images"] = initial_trace["images"]
        if candidate:
            if (report["initial_state"]["sim_state_sha256"] != baseline["initial_sim_state_sha256"]
                    or report["initial_images"] != baseline["initial_images"]):
                report["baseline_prefix_binding"]["status"] = "MISMATCH"
                raise RuntimeError("Candidate reset differs from registered baseline")
            report["baseline_prefix_binding"]["initial_state_and_images_match"] = True
        active_stages = copy.deepcopy(stages)
        if arm == "flat":
            active_stages = {"joint": dict(language=registration["combined_instruction"],
                goal_state=combine_goals(stages), world_signature=anchor_stage["world_signature"],
                max_steps=registration["total_steps"])}
        owner = SameWorldContinuation(runtime, active_stages,
            world_signature=anchor_stage["world_signature"],
            total_steps=registration["total_steps"], timeout_seconds=registration["timeout_seconds"])
        tail = None
        if tail_config:
            from fastrsi.stage_handoff_tail import StageHandoffTail
            tail = StageHandoffTail(owner, enabled=True)
        for index, key in enumerate(sequence):
            before = state_binding(runtime)
            owner.activate(key)
            after = state_binding(runtime)
            if before != after:
                raise RuntimeError("Stage activation changed physical world")
            transition = dict(stage=key, step=runtime.step_idx, before=before, after=after,
                              language=runtime.language, shared_max_steps=runtime.config["max_steps"],
                              trigger="registered_start" if index == 0 else "native_environment_stage_success")
            report["transitions"].append(transition); emit(dict(event="stage_activation", **transition))
            tick = time.perf_counter(); active_complete = bool(runtime.env.check_success())
            report["costs"]["oracle_feedback_seconds"] += time.perf_counter()-tick
            report["oracle_feedback_calls"] += 1
            if motion_config and index == 1:
                if report["baseline_prefix_binding"]["status"] != "PASSED":
                    raise RuntimeError("Original first-stage prefix must pass before motion")
                from fastrsi.public_release_retract_v1 import PublicReleaseRetract
                motion = PublicReleaseRetract(owner, motion_config["operation"], enabled=True)

                def motion_event(row):
                    capture_row = trace("action_chunk", ({"actions": [row["action"]]},))
                    emit(dict(event="operation_execution", stage=key,
                              operation="public_release_retract", images=capture_row["images"], **row))

                try:
                    motion.run(event_sink=motion_event)
                finally:
                    if motion.last_report:
                        report["public_release_retract"]["reports"].append(motion.last_report)
                        # Inclusive operation time includes observe, event and trace spans;
                        # only its actual action span joins the existing action ledger.
                        report["costs"]["action_seconds"] += motion.last_report["costs"]["action_seconds"]
                        save()
                tick = time.perf_counter(); active_complete = bool(runtime.env.check_success())
                report["costs"]["oracle_feedback_seconds"] += time.perf_counter()-tick
                report["oracle_feedback_calls"] += 1
                if motion.last_report["status"] in ("CANCELLED", "TIMEOUT"):
                    report["stop_reason"] = "operation_" + motion.last_report["status"].lower()
                    emit(dict(event="stage_end", stage=key, step=runtime.step_idx,
                              native_stage_success=active_complete, runtime_success=bool(runtime.success)))
                    break
                # The next native policy iteration reads owner.observe() afresh.
                # PUBLIC_TARGET_REACHED / qpos checks never count as task completion.
            while not runtime.done and runtime.step_idx < registration["total_steps"]:
                observation = owner.observe()
                payload = payload_builder(observation, runtime.language, session)
                forbidden = {"goal_state", "native_success", "benchmark_success", "final_success", "stage_success"}
                if forbidden.intersection(payload):
                    raise ValueError("Oracle predicate/result leaked into policy observation")
                before_step = runtime.step_idx
                tick = time.perf_counter(); report["policy_attempts"] += 1
                try:
                    policy_output = await asyncio.wait_for(policy.infer(payload),
                        timeout=registration["policy_timeout_seconds"])
                except BaseException as error:
                    report["costs"]["policy_seconds"] += time.perf_counter()-tick
                    emit(dict(event="policy_exception", step=before_step, stage=key,
                              error=repr(error)))
                    raise
                duration = time.perf_counter()-tick
                report["costs"]["policy_seconds"] += duration
                report["policy_calls"] += 1
                predicted = finite_actions(policy_output, registration["replan_steps"])
                submitted = predicted[:registration["replan_steps"]]
                remaining = registration["total_steps"]-runtime.step_idx
                submitted = submitted[:remaining]
                call = dict(event="policy_call", ordinal=report["policy_calls"], stage=key,
                    step_before=before_step, prompt=payload["prompt"], policy_seconds=duration,
                    array_bindings={name:array_binding(payload[name]) for name in
                        ("observation/image", "observation/wrist_image", "observation/state")},
                    public_policy_state=[float(value) for value in payload["observation/state"]],
                    predicted_actions=predicted, predicted_horizon=len(predicted),
                    submitted_actions=submitted, policy_meta=policy_output.get("policy_meta", {}))
                emit(call)
                check_prefix_event(call)
                tick = time.perf_counter()
                submit = tail.submit if tail is not None and index == 0 else owner.action_chunk
                try:
                    status = submit(dict(actions=submitted,
                        chunk_id="same_world_policy_"+str(report["policy_calls"])))
                finally:
                    action_seconds = time.perf_counter()-tick
                    report["costs"]["action_seconds"] += action_seconds
                actual_count = runtime.step_idx-before_step
                if not 0 < actual_count <= len(submitted):
                    raise RuntimeError("Actual action count outside submitted prefix")
                row = trace("action_chunk", ({"actions":submitted},))
                tick = time.perf_counter(); active_complete = bool(runtime.env.check_success())
                report["costs"]["oracle_feedback_seconds"] += time.perf_counter()-tick
                report["oracle_feedback_calls"] += 1
                execution_event = dict(event="execution", ordinal=report["policy_calls"], stage=key,
                          step_before=before_step, step_after=runtime.step_idx,
                          actual_actions=submitted[:actual_count], executed_steps=actual_count,
                          action_seconds=action_seconds, images=row["images"],
                          native_active_goal=active_complete, runtime_success=bool(runtime.success),
                          runtime_done=bool(runtime.done), accepted=bool(status.get("accepted")))
                emit(execution_event)
                check_prefix_event(execution_event)
                if candidate and index == 0 and active_complete:
                    binding = report["baseline_prefix_binding"]
                    if (binding["matched_events"] != len(baseline["events"])
                            or runtime.step_idx != baseline["handoff_step"]):
                        binding["status"] = "MISMATCH"
                        raise RuntimeError("Candidate original handoff differs from baseline")
                    binding["status"] = "PASSED"
                if tail is not None and index == 0:
                    try:
                        tail.consume(event_sink=emit,
                            action_observer=lambda chunk, current, event: trace("action_chunk", (chunk,)))
                    finally:
                        tail_report = tail.last_report
                        if tail_report:
                            report["costs"]["action_seconds"] += tail_report["costs"]["tail_action_seconds"]
                            report["costs"]["oracle_feedback_seconds"] += tail_report["costs"]["tail_oracle_seconds"]
                            report["oracle_feedback_calls"] += tail_report["native_check_attempts"]
                            # trace() already accounts for tail capture; submit span is already
                            # in action_seconds. Inclusive consume time is never added twice.
                            if tail_report["status"] != "NO_ACTION":
                                report["stage_handoff_tail"]["reports"].append(tail_report)
                    if active_complete:
                        active_complete = bool(runtime.success)
                        if not active_complete:
                            report["stop_reason"] = "current_goal_lost_after_submitted_tail_no_extra_repair"
                            save()
                            break
                save()
            emit(dict(event="stage_end", stage=key, step=runtime.step_idx,
                      native_stage_success=active_complete, runtime_success=bool(runtime.success)))
            if not active_complete:
                report.setdefault("stop_reason", "stage_failed_or_shared_budget_exhausted")
                break
            if index + 1 < len(sequence) and runtime.step_idx >= registration["total_steps"]:
                report["stop_reason"] = "shared_budget_exhausted_before_next_stage"
                break
        report["status"] = "COMPLETED"
    except BaseException:
        report["status"] = "FAILED"
        report["exceptions"].append(traceback.format_exc())
        raise
    finally:
        if runtime.env is not None and runtime._last_obs is not None:
            try:
                tick = time.perf_counter()
                report["final_native_evaluation"] = evaluate_goals_same_state(runtime, stages)
                report["costs"]["oracle_feedback_seconds"] += time.perf_counter()-tick
                report["oracle_feedback_calls"] += len(stages)+1
            except BaseException:
                report["final_evaluation_error"] = traceback.format_exc()
                report["status"] = "FAILED"
        if owner is not None:
            report["continuation_snapshot"] = owner.snapshot()
        report["success"] = (report["status"] == "COMPLETED"
            and report.get("final_native_evaluation", {}).get("joint_success") is True)
        report["cost_scope"] = "Worker includes reset, policy, environment, oracle checks and trace; " \
            "model-service startup/loading, GPU reservation and parent scheduling charged separately. " \
            "Component spans are contained in worker wall time; do not add them again."
        save()
    return report


async def run_registered(args):
    started = time.perf_counter()
    raw = args.registration.read_bytes()
    registration = validate_registration(json.loads(raw))
    diagnostic_sequence(registration, args.arm)
    source_paths = dict(runtime_server=args.runtime_server,
        same_world_continuation=ROOT/"src/fastrsi/same_world_continuation.py",
        stage_trace=ROOT/"scripts/stage_trace.py")
    if registration.get("stage_handoff_tail"):
        source_paths["stage_handoff_tail"] = ROOT/"src/fastrsi/stage_handoff_tail.py"
    if registration.get("public_release_retract"):
        source_paths["public_release_retract"] = ROOT/"src/fastrsi/public_release_retract_v1.py"
    for key, path in source_paths.items():
        if file_sha(path) != registration["source_sha256"][key]:
            raise ValueError("Frozen source differs: " + key)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import bddl_utils
    suite = benchmark.get_benchmark_dict()[registration["suite"]]()
    if registration.get("context_generalization_protocol"):
        anchor = suite.get_task(registration["anchor_task_id"])
        init_path = Path(get_libero_path("init_states"))/anchor.problem_folder/anchor.init_states_file
        if file_sha(init_path) != registration["anchor_init_source_sha256"]:
            raise ValueError("Registered anchor initialization source changed")
        validate_initial_state_count(registration,
            len(suite.get_task_init_states(registration["anchor_task_id"])))
    stages, bindings = bind_native_stages(registration, suite, get_libero_path("bddl_files"),
                                          bddl_utils.robosuite_parse_problem)
    spec = importlib.util.spec_from_file_location("same_world_frozen_runtime_v1", args.runtime_server)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    runtime = module.LiberoRealRuntime(dict(
        benchmark_name=registration["suite"], task_id=registration["anchor_task_id"],
        init_state_id=registration["init_state_id"], seed=registration["seed"],
        max_steps=registration["total_steps"], max_chunk_size=10, control_mode="relative"))
    report = None
    connection_started = time.perf_counter()
    try:
        endpoint = "ws://" + args.policy_host + ":" + str(args.policy_port)
        async with module._PolicyWsClient(endpoint, timeout_s=registration["policy_timeout_seconds"]) as policy:
            connection_seconds = time.perf_counter()-connection_started
            report = await execute_diagnostic(registration, args.arm, runtime, policy, stages,
                                               args.output, module._policy_observation)
            report["policy_connection_seconds"] = connection_seconds
    finally:
        tick = time.perf_counter(); runtime.close()
        if args.output.exists() and (args.output/"RESULT.json").exists():
            report = json.loads((args.output/"RESULT.json").read_bytes()) if report is None else report
            report.update(registration_sha256=sha(raw), native_definition_bindings=bindings,
                          source_sha256={key:file_sha(path) for key,path in source_paths.items()},
                          diagnostic_script_sha256=file_sha(Path(__file__)),
                          cleanup_seconds=time.perf_counter()-tick,
                          script_wall_seconds=time.perf_counter()-started)
            (args.output/"RESULT.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--arm", choices=("flat", "staged", "order68", "order86"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--runtime-server", type=Path,
                        default=Path("/core/PhyAgentOS/runtime/targets/remote/libero/server.py"))
    args = parser.parse_args()
    if not 1 <= args.policy_port <= 65535 or args.output.exists():
        parser.error("Valid policy port and a new output directory required")
    report = asyncio.run(run_registered(args))
    print(json.dumps({key:report[key] for key in
        ("status", "success", "physical_steps", "policy_calls", "script_wall_seconds")}, indent=2))
    if report["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
