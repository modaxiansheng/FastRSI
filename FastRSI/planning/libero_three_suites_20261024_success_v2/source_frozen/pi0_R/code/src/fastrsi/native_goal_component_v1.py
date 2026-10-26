"""Typed source31 program references for opt-in native component consumption.

This helper neither installs a consumer nor grants execution/admission authority.
The source remains an unchanged compiled success. Derived geometry is a separate,
unvalidated artifact. The caller's MODEL_CONFIG raw SHA identifies an externally
bound service/model contract (currently a Qwen service configuration), not pi0.5
weights or compiled-source VLA provenance; that source has no weight digest.

The initial RGB matcher is the exact source-order/disjoint-pair definition from
probe_dual_anchor_operation_value_v1.bind_initial_rgb, not a semantic detector.
The .06 bound and source annotations are the existing development diagnostic
definition; neither is a calibrated applicability or physical-completion test.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

from .dual_anchor_retarget_v1 import retarget_dual_anchors
from .experience_compiler import canonical_sha, validate_compiled


FIELD = "native_goal_component"
VERSION = "native_goal_component_v1"
TRANSFORM_VERSION = "dual_anchor_v1"
SOURCE_GOAL = "stack the right bowl on the left bowl and place them in the tray"
SOURCE31_RAW_SHA256 = "56141fcb7ed616d16a8766e24a821d0f055c59dfd94f75d1e43935758d204601"
SOURCE31_WAYPOINTS_SHA256 = "e4c25c989c9b63437a405cc49d7f7fd74ad70c0e0f58e872a9874e242acb8cf4"
SOURCE_RGB_RAW_SHA256 = "b06b6b870d9e7dadcb7cf7b264007801272e4c89c309e9e1afe1968e5ef32f3d"
CALIBRATION_RAW_SHA256 = "889aeec1de4c34dc46e53260fef6e984bfa340643bc21ac05c6ed55fb266de12"
SOURCE_ROIS = [[164, 122, 207, 153], [85, 130, 129, 163]]
ANCHOR_PLAN = [dict(anchor="first_object", start=0, close=41, release=125,
                    stop=126, terminal_kind="observed_open_command"),
               dict(anchor="second_object", start=126, close=197, release=None,
                    stop=253, terminal_kind="source_native_goal_truncated")]


def _mapping(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("Unexpected or missing %s fields" % name)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid %s SHA256" % name)
    return value


def _reference(value, name):
    _mapping(value, ("path", "sha256"), name)
    _digest(value["sha256"], name)
    raw = value["path"]
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError("Workspace configs-relative reference required")
    path = Path(raw)
    if (path.is_absolute() or ".." in path.parts or len(path.parts) < 2
            or path.parts[0] != "configs" or path.as_posix() != raw):
        raise ValueError("Workspace configs-relative reference required")


def validate_declaration(declaration):
    """Validate the only registered declaration, without filesystem/execution."""
    if isinstance(declaration, dict) and declaration.get('version') == 'native_goal_component_v2':
        from .native_goal_component_v2 import validate_declaration as validate_successor
        return validate_successor(declaration)
    _mapping(declaration, ("version", "source_artifact", "transform", "bindings"), "component")
    if declaration["version"] != VERSION:
        raise ValueError("Unknown native goal component version")
    _reference(declaration["source_artifact"], "source_artifact")
    transform = declaration["transform"]
    _mapping(transform, ("version", "source_rgb", "calibration", "source_rois_xyxy",
                         "anchor_plan", "max_offset_m"), "transform")
    if transform["version"] != TRANSFORM_VERSION:
        raise ValueError("Only the registered dual_anchor_v1 transform is allowed")
    _reference(transform["source_rgb"], "source_rgb")
    _reference(transform["calibration"], "calibration")
    refs = (declaration["source_artifact"], transform["source_rgb"], transform["calibration"])
    if len({ref["path"] for ref in refs}) != 3:
        raise ValueError("Source program, image and calibration must be distinct artifacts")
    if (declaration["source_artifact"]["sha256"] != SOURCE31_RAW_SHA256
            or transform["source_rgb"]["sha256"] != SOURCE_RGB_RAW_SHA256
            or transform["calibration"]["sha256"] != CALIBRATION_RAW_SHA256):
        raise ValueError("References differ from the registered source31 inputs")
    rois = transform["source_rois_xyxy"]
    if (not isinstance(rois, list) or len(rois) != 2 or
            any(not isinstance(roi, list) or len(roi) != 4 or
                any(type(v) is not int for v in roi) for roi in rois)
            or rois != SOURCE_ROIS):
        raise ValueError("Source-only region annotations differ from registered diagnostic")
    plans = transform["anchor_plan"]
    if not isinstance(plans, list) or plans != ANCHOR_PLAN:
        raise ValueError("Anchor plan differs from registered source31 command geometry")
    # Equality alone would accept True for index 1 in other declarations.
    for plan in plans:
        _mapping(plan, ANCHOR_PLAN[0].keys(), "anchor_plan")
        if any(type(plan[k]) is not int for k in ("start", "close", "stop")):
            raise ValueError("Anchor indices must be integers")
        if plan["release"] is not None and type(plan["release"]) is not int:
            raise ValueError("Release command index must be integer or absent")
    limit = transform["max_offset_m"]
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit != .06:
        raise ValueError("Use the exact registered .06m diagnostic bound")
    binding = declaration["bindings"]
    _mapping(binding, ("source_goal", "model_config_sha256", "target_ref",
                       "action_dim", "robot_state_dim"), "bindings")
    if binding["source_goal"] != SOURCE_GOAL:
        raise ValueError("The complete source task goal must remain unchanged")
    _digest(binding["model_config_sha256"], "model_config")
    if not isinstance(binding["target_ref"], str) or not binding["target_ref"].startswith("target://") or not binding["target_ref"][9:]:
        raise ValueError("Bind the original native target reference")
    if type(binding["action_dim"]) is not int or binding["action_dim"] != 7:
        raise ValueError("Bind the original seven-dimensional action interface")
    if type(binding["robot_state_dim"]) is not int or binding["robot_state_dim"] != 8:
        raise ValueError("Bind the original public robot8 interface")
    return copy.deepcopy(declaration)


def extract_component(skill_spec):
    """Missing typed field returns None; it never enables a default program."""
    if hasattr(skill_spec, "model_dump"):
        skill_spec = skill_spec.model_dump(mode="json")
    if not isinstance(skill_spec, dict):
        raise ValueError("Native SkillRuntime specification required")
    contract = skill_spec.get("output_contract", {})
    if not isinstance(contract, dict):
        raise ValueError("Native output contract must be a mapping")
    action = contract.get("action", {})
    if not isinstance(action, dict):
        raise ValueError("Native action contract must be a mapping")
    chunk = action.get("chunk", {})
    if not isinstance(chunk, dict):
        raise ValueError("Native action chunk contract must be a mapping")
    if FIELD not in chunk:
        return None
    policy = skill_spec.get("policy")
    if (skill_spec.get("runtime") != "OpenPISkillRuntime" or not isinstance(policy, dict)
            or policy.get("policy_adapter") != "policy_adapter://openpi_pi05_adapter"):
        raise ValueError("Registered component requires the original native pi05 runtime/adapter")
    if "preferred_replan_after_steps" in chunk:
        raise ValueError("Do not compose the goal program with the prefix experiment")
    return validate_declaration(chunk[FIELD])


def _read(workspace, reference):
    root = Path(workspace).resolve()
    path = root/reference["path"]
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError("Symlink component references are forbidden")
    try:
        path.resolve().relative_to(root)
    except ValueError:
        raise ValueError("Component reference escaped the workspace")
    if not path.is_file():
        raise ValueError("Component reference is not an ordinary existing file")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != reference["sha256"]:
        raise ValueError("Component referenced bytes changed: "+reference["path"])
    return blob


def bind_initial_rgb(source, current, rois, calibration):
    """Exact frozen probe matcher: appearance/order correspondence, not identity."""
    import numpy as np
    from .planar_binding import translation
    from .visual_binding import match_template
    matches = [match_template(source, current, roi, count=4) for roi in rois]
    center = lambda box: np.array([(box[0]+box[2]-1)/2, (box[1]+box[3]-1)/2])
    source_order = np.sign(center(rois[0])[0]-center(rois[1])[0])
    pairs = []
    for a in matches[0]['peaks']:
        for b in matches[1]['peaks']:
            aa, bb = a['box_xyxy'], b['box_xyxy']
            disjoint = aa[2] <= bb[0] or bb[2] <= aa[0] or aa[3] <= bb[1] or bb[3] <= aa[1]
            if disjoint and np.sign(center(aa)[0]-center(bb)[0]) == source_order:
                pairs.append((a['score']+b['score'], aa, bb))
    if not pairs:
        raise ValueError('No two disjoint, source-order-preserving correspondences')
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected = pairs[0]
    offsets = [translation(center(roi), center(box), calibration).tolist()
               for roi, box in zip(rois, selected[1:])]
    return offsets, dict(matches=matches, selected_boxes=list(selected[1:]),
        assignment_score=selected[0], competing_assignment_score=pairs[1][0] if len(pairs)>1 else None,
        assignment='source initial screen order, not verified semantic identity',
        calibrated_confidence=False, execution_applicability_certified=False)


def prepare_component(workspace, declaration, *, expected_goal,
                      expected_model_config_sha256, expected_target_ref, current_rgb=None):
    """Load bound inputs; optionally derive once from the initial raw front RGB.

    Returned JSON is a declaration/derivation receipt, never an executed skill or
    success label. Without current_rgb, derived_waypoints remains None. Runtime
    consumers must independently bind initial observation/Session, verify static
    camera/controller contracts, and debit actual native execution budgets.
    """
    if isinstance(declaration, dict) and declaration.get('version') == 'native_goal_component_v2':
        from .native_goal_component_v2 import prepare_component as prepare_successor
        return prepare_successor(workspace, declaration, expected_goal=expected_goal,
            expected_model_config_sha256=expected_model_config_sha256,
            expected_target_ref=expected_target_ref, current_rgb=current_rgb)
    value = validate_declaration(declaration)
    binding = value["bindings"]
    _digest(expected_model_config_sha256, "expected_model_config")
    if (binding["source_goal"] != expected_goal
            or binding["model_config_sha256"] != expected_model_config_sha256
            or binding["target_ref"] != expected_target_ref):
        raise ValueError("Component changes the current full goal/model contract/target")
    transform = value["transform"]
    source_blob = _read(workspace, value["source_artifact"])
    image_blob = _read(workspace, transform["source_rgb"])
    calibration_blob = _read(workspace, transform["calibration"])
    source = validate_compiled(json.loads(source_blob))
    if (source["source_goal"] != binding["source_goal"] or len(source["waypoints"]) != 253
            or canonical_sha(source["waypoints"]) != SOURCE31_WAYPOINTS_SHA256):
        raise ValueError("Source is not the unchanged registered source31 compiled program")
    calibration = json.loads(calibration_blob)
    import cv2
    import numpy as np
    source_rgb = cv2.imdecode(np.frombuffer(image_blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if source_rgb is None:
        raise ValueError("Registered source RGB is not decodable")
    source_rgb = cv2.cvtColor(source_rgb, cv2.COLOR_BGR2RGB)
    shape = (calibration["height"], calibration["width"], 3)
    if source_rgb.shape != shape:
        raise ValueError("Registered source RGB and calibration dimensions differ")
    receipt = dict(version=VERSION, declaration_sha256=canonical_sha(value),
                   source_artifact_sha256=value["source_artifact"]["sha256"],
                   source_waypoints_sha256=canonical_sha(source["waypoints"]),
                   source_rgb_raw_sha256=transform["source_rgb"]["sha256"],
                   source_rgb_array_sha256=hashlib.sha256(source_rgb.tobytes()).hexdigest(),
                   calibration_raw_sha256=transform["calibration"]["sha256"],
                   model_config_sha256=binding["model_config_sha256"],
                   model_identity_scope="EXTERNAL_SERVICE_MODEL_CONTRACT_NOT_PI_WEIGHTS_OR_SOURCE_VLA_PROVENANCE",
                   transform_version=TRANSFORM_VERSION, current_rgb_array_sha256=None,
                   current_rgb_shape=None, current_rgb_dtype=None, offsets=None,
                   derived_waypoints_sha256=None, matching=None,
                   status="REFERENCES_VALIDATED_NOT_DERIVED", execution_applicability_certified=False,
                   semantic_identity_certified=False, physical_release_status="UNKNOWN",
                   native_goal_status="NOT_EVALUATED", source_candidate_unchanged=True,
                   original_default_changed=False, model_calls=0, physics_steps=0)
    prepared = dict(version=VERSION, declaration=value, source_candidate=copy.deepcopy(source),
                    source_artifact_sha256=value["source_artifact"]["sha256"],
                    source_waypoints_sha256=receipt["source_waypoints_sha256"],
                    calibration=calibration, derived_waypoints=None,
                    derivation_receipt=receipt)
    if current_rgb is not None:
        if not isinstance(current_rgb, np.ndarray) or current_rgb.dtype != np.uint8 or current_rgb.shape != shape:
            raise ValueError("Current raw front RGB must match registered HWC uint8 dimensions")
        offsets, matching = bind_initial_rgb(source_rgb, current_rgb,
                                            transform["source_rois_xyxy"], calibration)
        named_offsets = {plan["anchor"]: offset for plan, offset in zip(transform["anchor_plan"], offsets)}
        geometry = retarget_dual_anchors(source["waypoints"], transform["anchor_plan"], named_offsets,
                                        max_offset_m=transform["max_offset_m"])
        prepared["derived_waypoints"] = geometry["waypoints"]
        receipt.update(status="UNVALIDATED_DERIVED_OPERATION_CANDIDATE",
                       current_rgb_array_sha256=hashlib.sha256(current_rgb.tobytes()).hexdigest(),
                       current_rgb_shape=list(current_rgb.shape), current_rgb_dtype=str(current_rgb.dtype),
                       offsets=named_offsets, derived_waypoints_sha256=geometry["result_waypoints_sha256"],
                       matching=matching, geometry={k:v for k,v in geometry.items() if k != "waypoints"})
    if canonical_sha(source["waypoints"]) != receipt["source_waypoints_sha256"]:
        raise ValueError("Derivation altered the compiled source")
    return prepared


def bind_selection(workspace, skill_spec, *, expected_goal,
                   expected_model_config_sha256, expected_target_ref, current_rgb=None):
    declaration = extract_component(skill_spec)
    if declaration is None:
        return None
    return prepare_component(workspace, declaration, expected_goal=expected_goal,
                             expected_model_config_sha256=expected_model_config_sha256,
                             expected_target_ref=expected_target_ref, current_rgb=current_rgb)
