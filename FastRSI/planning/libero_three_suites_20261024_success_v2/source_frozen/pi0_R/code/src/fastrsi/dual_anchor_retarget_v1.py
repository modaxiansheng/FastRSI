"""Unvalidated, two-anchor geometry for the recorded source31 program.

This pure component consumes declared initial RGB-plane offsets. It does not
estimate object identity/poses, recognize a held object, infer actions, or run
physics. Both approach offsets fade out before the unchanged destination poses.
The final source command is closed: native goal truncation is explicitly distinct
from observing a release. No physical postcondition is inferred here.

The close/apex/halfway-blend construction follows phase_binding.transport_weights.
This version additionally binds two disjoint command spans and ramps entry offsets
by source path length, avoiding an offset jump at the first release boundary.
It does not turn the earlier single-anchor negative probe into a positive result.
"""

import copy
import hashlib
import json
import math
from numbers import Real


VERSION = "dual_anchor_retarget_v1"
SOURCE_STEPS = 253
PLAN_KEYS = {"anchor", "start", "close", "release", "stop", "terminal_kind"}
SOURCE_SPANS = ((0, 41, False), (41, 125, True),
                (125, 197, False), (197, 253, True))


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("%s must be a finite real number" % name)
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        raise ValueError("%s must be a finite real number" % name)
    if not math.isfinite(converted):
        raise ValueError("%s must be a finite real number" % name)
    return converted


def _vector(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("%s must have three components" % name)
    return [_number(v, name) for v in value]


def _waypoints(waypoints):
    if not isinstance(waypoints, list) or len(waypoints) != SOURCE_STEPS:
        raise ValueError("Bind the complete 253-command source31 program")
    positions = []
    for row in waypoints:
        if not isinstance(row, dict) or set(row) != {"target_pos", "target_rot", "gripper_command"}:
            raise ValueError("Only source robot target and gripper fields are accepted")
        positions.append(_vector(row["target_pos"], "target_pos"))
        r = row["target_rot"]
        if not isinstance(r, (list, tuple)) or len(r) != 3:
            raise ValueError("target_rot must be a proper 3x3 rotation")
        r = [_vector(v, "target_rot") for v in r]
        for a in range(3):
            for b in range(3):
                dot = sum(r[k][a] * r[k][b] for k in range(3))
                if abs(dot - (1.0 if a == b else 0.0)) > 1e-6:
                    raise ValueError("target_rot must be a proper rotation")
        determinant = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
                       - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
                       + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
        if abs(determinant - 1.0) > 1e-6:
            raise ValueError("target_rot must be a proper rotation")
        grip = _number(row["gripper_command"], "gripper_command")
        if grip == 0:
            raise ValueError("Ambiguous zero gripper command")
    for start, stop, closed in SOURCE_SPANS:
        if any((row["gripper_command"] > 0) != closed for row in waypoints[start:stop]):
            raise ValueError("Source31 open/closed command spans differ from the recorded program")
    return positions


def _plans(anchor_plan):
    if not isinstance(anchor_plan, list) or len(anchor_plan) != 2:
        raise ValueError("Declare exactly two object-anchor plans")
    for row in anchor_plan:
        if not isinstance(row, dict) or set(row) != PLAN_KEYS:
            raise ValueError("Unexpected or missing anchor-plan fields")
        if not isinstance(row["anchor"], str) or not row["anchor"].strip():
            raise ValueError("Declare a nonempty object anchor")
        if any(type(row[k]) is not int for k in ("start", "close", "stop")):
            raise ValueError("Command boundaries must be integer indices")
        if not 0 <= row["start"] < row["close"] < row["stop"] <= SOURCE_STEPS:
            raise ValueError("Invalid command span")
        if row["release"] is not None and type(row["release"]) is not int:
            raise ValueError("Release index must be integer or explicitly absent")
    if anchor_plan[0]["anchor"] == anchor_plan[1]["anchor"]:
        raise ValueError("Two distinct object anchors are required")
    if anchor_plan[0]["stop"] > anchor_plan[1]["start"]:
        raise ValueError("Object-anchor spans overlap")
    expected = [(0, 41, 125, 126, "observed_open_command"),
                (126, 197, None, 253, "source_native_goal_truncated")]
    for row, bounds in zip(anchor_plan, expected):
        if tuple(row[k] for k in ("start", "close", "release", "stop", "terminal_kind")) != bounds:
            raise ValueError("Bind exact source31 spans; its final release was not observed")
    return copy.deepcopy(anchor_plan)


def _path_progress(positions, start, stop):
    """Inclusive endpoint progress by source 3D target path length."""
    cumulative = [0.0]
    for index in range(start + 1, stop + 1):
        cumulative.append(cumulative[-1] + math.hypot(*(
            positions[index][d] - positions[index - 1][d] for d in range(3))))
        if not math.isfinite(cumulative[-1]):
            raise ValueError("Source geometric progress is nonfinite")
    if cumulative[-1] <= 1e-12:
        raise ValueError("Source interval has no geometric progress")
    return [distance / cumulative[-1] for distance in cumulative]


def retarget_dual_anchors(waypoints, anchor_plan, offsets, *, max_offset_m):
    """Return auditable geometry; indices are zero-based, stop is exclusive.

    ``offsets`` maps the two explicitly named anchors to [dx, dy, 0] in metres.
    ``max_offset_m`` is an explicit protocol limit, not a learned/safety threshold.
    First plan: start=0, close=41, release=125, stop=126,
    terminal_kind='observed_open_command'. Second: start=126, close=197,
    release=None, stop=253, terminal_kind='source_native_goal_truncated'.

    Entry weight starts at zero and reaches one at the close command. It remains
    one until halfway from close to the source commanded-height apex, then decays
    by path length to zero at that apex. All later targets remain exactly source.
    Command transitions are not object-stage or physical-release observations.
    """
    positions = _waypoints(waypoints)
    plans = _plans(anchor_plan)
    limit = _number(max_offset_m, "max_offset_m")
    if limit < 0:
        raise ValueError("max_offset_m must be nonnegative")
    anchors = {row["anchor"] for row in plans}
    if not isinstance(offsets, dict) or set(offsets) != anchors:
        raise ValueError("Offsets must bind exactly the declared object anchors")
    declared_offsets = {}
    for anchor in anchors:
        offset = _vector(offsets[anchor], "planar offset")
        if offset[2] != 0:
            raise ValueError("Only XY-plane offsets are allowed; Z must be zero")
        if math.hypot(offset[0], offset[1]) > limit:
            raise ValueError("Offset exceeds the explicitly declared protocol bound")
        declared_offsets[anchor] = offset

    result = copy.deepcopy(waypoints)
    rows = []
    segments = []
    for plan in plans:
        start, close, stop = (plan[k] for k in ("start", "close", "stop"))
        release = plan["release"]
        closed_stop = release if release is not None else stop
        apex = max(range(close, closed_stop), key=lambda i: positions[i][2])
        blend_begin = (close + apex) // 2
        if not close < blend_begin < apex < closed_stop - (release is None):
            raise ValueError("Source span lacks an elevated transfer interval before its terminal")
        if positions[apex][2] <= positions[blend_begin][2] + 1e-12:
            raise ValueError("Source lift interval has no commanded-height increase")
        entry = _path_progress(positions, start, close)
        decay = _path_progress(positions, blend_begin, apex)
        offset = declared_offsets[plan["anchor"]]
        weights = []
        for index in range(start, stop):
            if index <= close:
                weight, phase = entry[index - start], "approach_entry"
            elif index <= blend_begin:
                weight, phase = 1.0, "close_command_hold"
            elif index < apex:
                weight, phase = 1.0 - decay[index - blend_begin], "source_lift_decay"
            else:
                weight, phase = 0.0, "unchanged_destination"
            applied = [weight * v for v in offset]
            if weight != 0 and any(v != 0 for v in offset):
                result[index]["target_pos"] = [p + delta for p, delta in zip(positions[index], applied)]
                _vector(result[index]["target_pos"], "retargeted target_pos")
            weights.append(weight)
            rows.append(dict(index=index, anchor=plan["anchor"], phase=phase,
                             weight=weight, offset_m=applied,
                             source_target_pos=copy.deepcopy(waypoints[index]["target_pos"]),
                             target_pos=copy.deepcopy(result[index]["target_pos"])))
        segments.append(dict(**plan, apex=apex, blend_begin=blend_begin,
                             offset_m=offset, weights=weights,
                             release_command_observed=release is not None,
                             physical_release_status="UNKNOWN",
                             geometry_scope="SOURCE_COMMANDS_NOT_OBSERVED_OBJECT_STAGE"))

    for index in (0, 125, 126, SOURCE_STEPS - 1):
        if result[index] != waypoints[index]:
            raise ValueError("A source entry/release/terminal endpoint changed")
    return dict(version=VERSION, status="UNVALIDATED_OPERATION_CANDIDATE",
                source_waypoints_sha256=_sha(waypoints),
                result_waypoints_sha256=_sha(result), max_offset_m=limit,
                waypoints=result, segments=segments, rows=rows,
                destination_assumption="SOURCE_DESTINATION_TARGETS_UNCHANGED_NOT_CURRENT_POSTCONDITION_CERTIFIED",
                terminal=dict(index=SOURCE_STEPS - 1,
                              terminal_kind="source_native_goal_truncated",
                              final_closed_command=True, release_not_observed=True,
                              physical_release_status="UNKNOWN",
                              native_goal_status="NOT_EVALUATED_BY_GEOMETRY_COMPONENT"),
                online_input_scope="DECLARED_INITIAL_RGB_XY_OFFSETS_AND_SOURCE_ROBOT_TARGETS_ONLY",
                hidden_state_used=False, model_calls=0, physics_steps=0)
