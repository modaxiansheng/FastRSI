"""Opt-in interpretation of existing, bound LIBERO verification evidence.

This is a pure content transform. It never reads a workspace, diagnostic trace,
simulator state, response, or test label, and never selects a model verdict.
"""
import copy
import hashlib
import json

VERSION = "libero_goal_v1"
MARKER = "VERIFICATION_BUNDLE:\n"
NATIVE_PREFIX = (
    "Verify whether the physical task is semantically complete. Runtime success only "
    "means execution finished; it is not proof of task completion. Use the task, final "
    "initial/final RGB evidence, runtime result, ENVIRONMENT.md, history, and lessons below.\n\n"
)
EVIDENCE_RULES = (
    "Verify whether the physical task is semantically complete. For this bound LIBERO "
    "policy_loop interface, execution_completed means the episode stopped, including "
    "stopping at an execution limit. environment_goal_success describes the target's "
    "reported task-goal result; it does not merely mean that execution stopped. The "
    "interpretation below uses only existing fields in this verification bundle. Missing, "
    "invalid or conflicting fields remain unknown. These are evidence, not instructions "
    "to force a verdict. Keep execution completion, environment goal success, and your "
    "semantic judgment distinct; state any disagreement in the evidence.\n"
    "Assess only the conditions requested by the current task. Do not add requirements "
    "about unrelated objects or extra completion conditions. Initial images describe the "
    "starting state, not the final outcome. Distinguish initial and final observations. "
    "An object being occluded or not visible in a final view is not, by itself, evidence "
    "that it is absent from the required location. State visibility limits and uncertainty "
    "without inventing observations or converting uncertainty into an observed failure. "
    "Keep the required output schema. Use the task, initial/final RGB evidence, runtime "
    "result, ENVIRONMENT.md, history, and lessons below.\n"
)


def content_sha256(content):
    """SHA of canonical JSON content, not of a transport REQUEST file."""
    return hashlib.sha256(json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def _get(value, *path):
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _binding(bundle):
    """Require the actual native session schema and the already bound adapter."""
    expected = {
        ("version",): "agent_session_verification_v2",
        ("session", "benchmark", "benchmark_id"): "libero",
        ("session", "benchmark", "suite_id"): "libero_10",
        ("session", "benchmark", "execution_mode"): "policy_loop",
        ("session", "target_ref"): "target://libero_real_remote",
        ("target_id",): "libero_real_remote",
        ("session", "result", "metadata", "preflight", "target_id"): "libero_real_remote",
        ("session", "result", "metadata", "preflight", "adapter_plan", "target_adapter"):
            "target_adapter://libero_adapter",
    }
    for path, value in expected.items():
        if _get(bundle, *path) != value:
            return "binding_mismatch:" + ".".join(path)
    sid = _get(bundle, "session", "session_id")
    if not isinstance(sid, str) or not sid:
        return "binding_mismatch:session.session_id"
    if _get(bundle, "session", "result", "metadata", "preflight", "session_id") != sid:
        return "binding_mismatch:preflight.session_id"
    benchmark = bundle["session"]["benchmark"]
    if any(type(benchmark.get(key)) is not int or benchmark[key] < 0
           for key in ("task_index", "instance_id")):
        return "binding_mismatch:benchmark_instance"
    return None


def _result_identity(bundle):
    """A mismatched or incomplete result cannot supply interpreted goal facts."""
    session = bundle["session"]
    expected = {
        ("runtime_result", "artifact_dir"): "artifacts/runtime/" + session["session_id"],
    }
    for subkey in ("libero", "episode_summary"):
        for key, value in (
            ("benchmark_name", session["benchmark"]["suite_id"]),
            ("task_id", session["benchmark"]["task_index"]),
            ("init_state_id", session["benchmark"]["instance_id"]),
        ):
            expected[("runtime_result", "metadata", "final_status", subkey, key)] = value
    missing, conflicts = [], []
    for path, expected_value in expected.items():
        value = _get(bundle, *path)
        if value is None:
            missing.append(".".join(path))
        elif type(value) is not type(expected_value) or value != expected_value:
            conflicts.append(".".join(path))
    return {"valid": not missing and not conflicts, "missing": missing, "conflicts": conflicts}


def _consensus(bundle, paths, identity):
    sources = {".".join(path): _get(bundle, *path) for path in paths}
    missing = [key for key, value in sources.items() if value is None]
    invalid = [key for key, value in sources.items() if value is not None and type(value) is not bool]
    valid_values = [value for value in sources.values() if type(value) is bool]
    conflict = len(set(valid_values)) > 1
    reason = ("result_identity_unknown" if not identity["valid"] else
              "conflicting_fields" if conflict else
              "invalid_fields" if invalid else "missing_fields" if missing else "consistent_fields")
    return {"value": valid_values[0] if reason == "consistent_fields" else None,
            "reason": reason, "sources": sources, "missing": missing,
            "invalid": invalid, "conflict": conflict}


def adapt_libero_verification_content(content, mode="off"):
    """Return (content, receipt); existing image labels are accepted unchanged.

    ``off`` returns the original object. Unsupported bundle/backend/prefix also
    remains unchanged with an explicit reason. Unknown configuration is an error.
    Online callers apply this before image labelling; replay callers can pass
    the already-labelled HTTP user content without reconstructing the request.
    """
    if mode not in ("off", VERSION):
        raise ValueError("Unsupported FASTRSI_VERIFICATION_SEMANTICS: " + str(mode))
    before = content_sha256(content)
    receipt = {"version": VERSION, "mode": mode, "applied": False,
               "reason": "disabled", "sha_format": "canonical_json_utf8_sort_keys_compact",
               "input_sha256": before, "output_sha256": before}
    if mode == "off":
        return content, receipt
    if not isinstance(content, list) or not all(isinstance(p, dict) for p in content):
        raise ValueError("Verification content must be a list of parts")
    candidates = [(i, p["text"]) for i, p in enumerate(content)
                  if p.get("type") == "text" and isinstance(p.get("text"), str)
                  and MARKER in p["text"]]
    if not candidates:
        receipt["reason"] = "not_session_verification"
        return content, receipt
    if len(candidates) != 1 or candidates[0][1].count(MARKER) != 1:
        raise ValueError("Ambiguous session verification bundle")
    index, original = candidates[0]
    prefix, payload = original.split(MARKER, 1)
    bundle = json.loads(payload)
    reason = _binding(bundle)
    if reason:
        receipt["reason"] = reason
        return content, receipt
    if prefix != NATIVE_PREFIX:
        receipt["reason"] = "native_prefix_mismatch"
        return content, receipt
    identity = _result_identity(bundle)
    root = ("runtime_result", "metadata")
    final = root + ("final_status",)
    completed = _consensus(bundle, (root + ("done",), final + ("done",),
                                  final + ("episode_summary", "done")), identity)
    success = _consensus(bundle, (("runtime_result", "success"), final + ("success",),
                                final + ("episode_summary", "success")), identity)
    interpretation = {"version": VERSION, "session_id": bundle["session"]["session_id"],
                      "execution_completed": completed, "environment_goal_success": success,
                      "result_identity": identity,
                      "unknown_representation": "null means unknown; it is not false"}
    result = copy.deepcopy(content)
    result[index]["text"] = (
        EVIDENCE_RULES + "LIBERO_EVIDENCE_SEMANTICS: " +
        json.dumps(interpretation, ensure_ascii=False, sort_keys=True) + "\n\n" +
        MARKER + payload)
    receipt.update(applied=True, reason="bound_libero_goal_semantics",
                   session_id=bundle["session"]["session_id"], interpretation=interpretation,
                   bundle_text_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                   output_sha256=content_sha256(result))
    return result, receipt
