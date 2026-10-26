"""Typed requests for externally executed, bounded native development checks.

Parsing a request never grants physical dispatch or admits its own candidate.
Development seeds identify native sampler draws, not official initial-state IDs.
Feedback labels are extracted from hash-bound actual consumer result files.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

VERSION = "native_validation_request_v1"
FEEDBACK_VERSION = "native_validation_feedback_v1"
STATE_VERSION = "native_component_state_v1"
DEV_SEEDS = (640711, 640712)
ARMS = ("native_pi05", "dual_anchor_v1")
VERIFICATION = "native_success_primary_semantic_separate_conflicts_quarantined_v1"
GOAL = "stack the right bowl on the left bowl and place them in the tray"
REQUEST_KEYS = {"version", "goal", "suite", "task_id", "development_seeds", "arms",
                "max_steps_per_episode", "verification_protocol", "current_component_state_sha256",
                "candidate_declaration", "reason"}
COST_SECONDS = {"reset_seconds", "binding_seconds", "program_seconds", "vla_inference_seconds",
                "vla_action_seconds", "capture_seconds", "close_seconds"}
COST_COUNTS = {"vla_calls_attempted", "vla_calls_completed", "qwen_calls", "program_actions", "vla_actions",
               "reset_calls_attempted", "settling_actions", "training_updates", "persistent_updates"}
COST_KEYS = COST_SECONDS | COST_COUNTS
ARM_SECONDS = {"model_startup_seconds", "model_stop_seconds", "model_residency_seconds", "full_wall_seconds"}
PI05_MODEL_FILES = {
    "config.json": "06bbdc28bc7b7cf8cdae483099ee626054c0f085c71ed5fc398aa1d7da64715f",
    "conversion_manifest.json": "66d46396ff5e645cebe303ff8989ff47a738d6da60cc7a9fac95aeb71e541179",
    "assets/physical-intelligence/libero/norm_stats.json": "b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84",
    "model.safetensors": "2e967a34251e5f69559314d3f1dd999212993226797f87e67e9ee34084f2fd69",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _shape(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing %s fields" % name)


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Exact lowercase SHA256 required")


def _finite_seconds(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("Costs must be finite nonnegative seconds")


def read_reference(base, reference):
    """Strict, relative, ordinary file; no silent provenance path relocation."""
    _shape(reference, {"path", "sha256"}, "result reference")
    _digest(reference["sha256"])
    rel = Path(reference["path"])
    if (not isinstance(reference["path"], str) or rel.is_absolute() or ".." in rel.parts
            or not rel.parts or rel.as_posix() != reference["path"] or "\\" in reference["path"]):
        raise ValueError("Result reference must be an ordinary relative path")
    base = Path(base).resolve()
    path = base / rel
    for parent in (path, *path.parents):
        if parent == base:
            break
        if parent.is_symlink():
            raise ValueError("Symlink evidence cannot be consumed")
    path.resolve().relative_to(base)
    raw = path.read_bytes()
    if sha(raw) != reference["sha256"]:
        raise ValueError("Actual result evidence changed")
    return raw


def validate_request(value, *, workspace, expected_model_config_sha256,
                     current_component_state_sha256):
    _shape(value, REQUEST_KEYS, "validation request")
    if (value["version"], value["goal"], value["suite"], value["task_id"], value["max_steps_per_episode"],
            value["verification_protocol"]) != (VERSION, GOAL, "libero_90", 64, 400, VERIFICATION):
        raise ValueError("Request changed the predeclared native task/budget/verification")
    if type(value["task_id"]) is not int or type(value["max_steps_per_episode"]) is not int:
        raise ValueError("Integer task/budget required")
    seeds = value["development_seeds"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 2
            or any(type(seed) is not int or seed not in DEV_SEEDS for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Only one or two fresh registered development draws are allowed")
    if value["arms"] != list(ARMS):
        raise ValueError("Both native_pi05 and dual_anchor_v1 must receive the same draws and cap")
    _digest(current_component_state_sha256)
    if value["current_component_state_sha256"] != current_component_state_sha256:
        raise ValueError("Request did not bind the actual current native registry facts")
    if not isinstance(value["reason"], str) or not value["reason"].strip() or len(value["reason"]) > 2000:
        raise ValueError("A bounded, nonempty request reason is required")
    from .native_goal_component_v1 import prepare_component
    prepare_component(workspace, value["candidate_declaration"], expected_goal=GOAL,
                      expected_model_config_sha256=expected_model_config_sha256,
                      expected_target_ref="target://libero_real_remote")
    return copy.deepcopy(value)


def request_schema(*, current_component_state_sha256, declaration_schema):
    """Describe legal operations without suggesting an answer or requesting it."""
    return dict(version=VERSION, legal_decisions=["register", "noop", "request_validation"],
        request_path="reports/VALIDATION_REQUEST.json", required_keys=sorted(REQUEST_KEYS),
        fixed=dict(version=VERSION, goal=GOAL, suite="libero_90", task_id=64,
                   arms=list(ARMS), max_steps_per_episode=400, verification_protocol=VERIFICATION,
                   current_component_state_sha256=current_component_state_sha256),
        development_seeds_allowed=list(DEV_SEEDS), minimum_development_seeds=1,
        maximum_development_seeds=2, candidate_declaration_schema=copy.deepcopy(declaration_schema),
        reason="Model provides its own bounded rationale, not an approval or fabricated outcome",
        status="AWAITING_EXTERNAL_VALIDATION_IF_WRITTEN",
        scope="Request alone is neither execution nor verified skill nor learning gain; no evaluation seeds allowed")


def validate_feedback(value, *, base, request, request_sha256):
    """Extract real facts; unknown/failed branches stay unknown and failures cost."""
    _shape(value, {"version", "status", "request_sha256", "execution_plan", "executions"}, "validation feedback")
    _digest(request_sha256)
    if (value["version"], value["status"], value["request_sha256"]) != (FEEDBACK_VERSION, "COMPLETED", request_sha256):
        raise ValueError("Feedback must finish and bind this exact exported request")
    expected = {(seed, arm) for seed in request["development_seeds"] for arm in ARMS}
    plan_blob = read_reference(base, value["execution_plan"])
    plan_sha = sha(plan_blob)
    plan = json.loads(plan_blob)
    declaration = request["candidate_declaration"]
    transform = declaration["transform"]
    if ((plan.get("version"), plan.get("role"), plan.get("suite"), plan.get("task_id"), plan.get("max_steps"),
            plan.get("replan_steps"), plan.get("program_steps"), plan.get("policy_config"), plan.get("torch_seed"))
            != ("native_development_operation_v1", "DEVELOPMENT_INPUT", "libero_90", 64, 400, 10, 253, "pi05_libero", 7)
            or plan.get("arms") != list(ARMS) or plan.get("development_seeds") != list(DEV_SEEDS)
            or plan.get("model_files_sha256") != PI05_MODEL_FILES):
        raise ValueError("Actual plan differs from original matched task/policy/budget protocol")
    refs = (("candidate", declaration["source_artifact"]), ("source_image", transform["source_rgb"]),
            ("calibration", transform["calibration"]))
    for key, reference in refs:
        if not isinstance(plan.get(key), str) or plan.get("bindings", {}).get(plan[key]) != reference["sha256"]:
            raise ValueError("Executed source/RGB/calibration differs from requested component")
    if (plan.get("source_rois_xyxy") != transform["source_rois_xyxy"]
            or plan.get("anchor_plan") != transform["anchor_plan"] or plan.get("max_offset_m") != transform["max_offset_m"]
            or transform["version"] != "dual_anchor_v1"):
        raise ValueError("Executed geometry definition differs from requested candidate")
    seen, facts = set(), []
    pair_signatures = {}
    if not isinstance(value["executions"], list):
        raise ValueError("Actual consumer executions required")
    for execution in value["executions"]:
        _shape(execution, {"development_seed", "arm", "result"}, "execution evidence")
        identity = (execution["development_seed"], execution["arm"])
        if identity not in expected or identity in seen:
            raise ValueError("Unrequested/evaluation/duplicate execution")
        arm_result = json.loads(read_reference(base, execution["result"]))
        required_arm = {"development_seed", "arm", "status", "raw_probe", *ARM_SECONDS,
                        "full_agent_chain", "learning_gain_claimed"}
        if (not isinstance(arm_result, dict) or not required_arm.issubset(arm_result)
                or (arm_result["development_seed"], arm_result["arm"]) != identity
                or arm_result["status"] not in ("COMPLETED", "FAILED")
                or arm_result["full_agent_chain"] is not False or arm_result["learning_gain_claimed"] is not False):
            raise ValueError("Actual ARM_RESULT identity/cost/evidence scope is missing")
        if arm_result.get("plan_sha256") != plan_sha:
            raise ValueError("ARM_RESULT is not bound to the executed comparison plan")
        for key in ARM_SECONDS:
            _finite_seconds(arm_result[key])
        arm_path = Path(base) / execution["result"]["path"]
        source = json.loads(read_reference(arm_path.parent, arm_result["raw_probe"]))
        required = {"development_seed", "arm", "status", "native_success", "physical_steps", "seconds", "costs",
                    "actual_goal", "suite", "task_id", "max_steps", "full_agent_chain", "learning_gain_claimed"}
        if not isinstance(source, dict) or not required.issubset(source):
            raise ValueError("Raw actual result is missing outcome or complete cost accounting")
        if (source["development_seed"], source["arm"]) != identity or source["status"] not in ("COMPLETED", "FAILED"):
            raise ValueError("Actual result identity/terminal status differs")
        if source.get("plan_sha256") != plan_sha:
            raise ValueError("Raw probe executed another plan")
        if (source["actual_goal"], source["suite"], source["task_id"], source["max_steps"]) != (GOAL, "libero_90", 64, 400):
            raise ValueError("Raw actual probe changed task/complete goal/cap")
        if source["full_agent_chain"] is not False or source["learning_gain_claimed"] is not False:
            raise ValueError("Operation diagnostic cannot be relabelled full Agent or learning gain")
        if (type(source["physical_steps"]) is not int or not 0 <= source["physical_steps"] <= 400
                or source["native_success"] is not None and type(source["native_success"]) is not bool):
            raise ValueError("Invalid observed native outcome or shared cap")
        if source["status"] == "COMPLETED" and source["native_success"] is None:
            raise ValueError("A completed execution must include actual native evaluation")
        _finite_seconds(source["seconds"])
        _shape(source["costs"], COST_KEYS, "full cost breakdown")
        for key in COST_SECONDS:
            _finite_seconds(source["costs"][key])
        for key in COST_COUNTS:
            if type(source["costs"][key]) is not int or source["costs"][key] < 0:
                raise ValueError("Actual operation counters must be nonnegative integers")
        if (source["costs"]["program_actions"] + source["costs"]["vla_actions"] != source["physical_steps"]
                or source["costs"]["vla_calls_completed"] > source["costs"]["vla_calls_attempted"]
                or any(source["costs"][key] for key in ("qwen_calls", "training_updates", "persistent_updates"))):
            raise ValueError("Probe action/call counters or operation-only scope differ")
        if arm_result["full_wall_seconds"] < source["seconds"]:
            raise ValueError("External arm total cannot omit its actual probe time")
        binding = source.get("initial_input_binding", {})
        binding_keys = ("collection_manifest_sha256", "sample_id", "xml_sha256", "pre_settle_state", "post_settle_state",
                        "public_rgb", "public_robot8")
        if (binding.get("status") != "MATCHED_FROZEN_INPUT" or binding.get("sample_id") != "sample_" + str(identity[0])
                or binding.get("public_rgb_exact") is not True or binding.get("public_robot8_exact") is not True
                or any(key not in binding for key in binding_keys)):
            raise ValueError("Actual sampler/initial observation was not exactly recovered")
        signature = {key:binding[key] for key in binding_keys}
        if identity[0] in pair_signatures and signature != pair_signatures[identity[0]]:
            raise ValueError("Two arms did not use the same actual initial state/RGB/robot8")
        pair_signatures[identity[0]] = signature
        # Nested/overlapping phases and model residency are kept separate, never summed.
        facts.append(dict(development_seed=identity[0], arm=identity[1], status=source["status"],
                          controller_status=arm_result["status"],
                          native_success=source["native_success"], physical_steps=source["physical_steps"],
                          probe_seconds=source["seconds"], probe_costs=copy.deepcopy(source["costs"]),
                          external_controller_cost={key:arm_result[key] for key in ARM_SECONDS},
                          full_agent_chain=False, learning_gain_claimed=False,
                          semantic_status=source.get("semantic_status", "NOT_REPORTED"),
                          physical_completion=source.get("physical_completion", source.get("physical_release", "UNKNOWN")),
                          semantic_conflict_text="QUARANTINED_NOT_A_LEARNING_LABEL",
                          actual_result_sha256=arm_result["raw_probe"]["sha256"],
                          actual_arm_result_sha256=execution["result"]["sha256"]))
        seen.add(identity)
    if seen != expected:
        raise ValueError("Missing actual branches; never fabricate unexecuted comparison outcomes")
    return dict(version=FEEDBACK_VERSION, request_sha256=request_sha256, goal=GOAL,
                executions=facts, scope="Completed development evidence; not independent evaluation or net learning benefit")
