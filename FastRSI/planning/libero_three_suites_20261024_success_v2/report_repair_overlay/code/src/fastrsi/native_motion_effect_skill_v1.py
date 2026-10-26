"""Opt-in development: at most two native relative chunks with live feedback.

Entry is the first actually executed all-closed chunk after a mixed closing
chunk, never an elapsed-step oracle. Prototype examples and measured response
are learned records, not grasp or task-phase certificates. The original Agent,
loop, goal, target, feedback window and termination remain authoritative.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import time

from fastrsi.native_stage_continuation_v1 import NativeStageContinuation

AUTHORITY = "MOTION_EFFECT_SKILL_DEVELOPMENT_VALUE"


def public_vector(image, wrist, robot, previous_robot, recent_commands):
    import numpy as np
    pools = []
    for pixels in (image, wrist):
        a = np.asarray(pixels)
        if a.shape != (256, 256, 3) or a.dtype != np.uint8:
            raise ValueError("Unchanged legal dual RGB required")
        pools.append(a.astype(np.float32).reshape(8, 32, 8, 32, 3).mean(axis=(1, 3)).ravel() / 127.5 - 1)
    robot = np.asarray(robot, np.float32)
    previous_robot = np.asarray(previous_robot, np.float32)
    commands = np.asarray(recent_commands, np.float32)
    if robot.shape != (8,) or previous_robot.shape != (8,) or not commands.size:
        raise ValueError("Public state and actual action history required")
    state = robot * np.array([5, 5, 5, .3125, .3125, .3125, 25, 25], np.float32)
    delta = (robot - previous_robot) * np.array([20, 20, 20, 2, 2, 2, 25, 25], np.float32)
    result = np.concatenate([np.concatenate(pools) / np.sqrt(384),
                             np.r_[state, delta, np.mean(commands > 0), commands[-1] > 0] / np.sqrt(18)])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite public input")
    return result.astype(np.float32)


def load_skill(path, digest, goal, model_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Frozen skill changed")
    skill = json.loads(raw)
    if (skill.get("version") != "native_motion_effect_skill_v1"
            or skill.get("status") != "UNVALIDATED_DEVELOPMENT_CANDIDATE"
            or skill.get("goal") != goal or skill.get("model_config_sha256") != model_sha
            or skill.get("action_units") != "original_relative_OSC_native_commands"
            or skill.get("second_chunk_feedback_window") != 10 or skill.get("output_cap") != 2
            or skill.get("grasp_certified") is not False):
        raise ValueError("Skill scope changed")
    finite = lambda v: type(v) in (int, float) and math.isfinite(v)
    chunks = skill["chunks"]
    if (not isinstance(chunks, list) or len(chunks) != 2
            or any(not isinstance(chunk, list) or len(chunk) != 10 for chunk in chunks)
            or any(not isinstance(row, list) or len(row) != 7 for chunk in chunks for row in chunk)
            or any(not finite(v) or abs(v) > 2 for chunk in chunks for row in chunk for v in row)):
        raise ValueError("Two bounded original-relative chunks required")
    if any(row[6] <= 0 for chunk in chunks for row in chunk):
        raise ValueError("No release or new grasp command is permitted")
    if hashlib.sha256(struct.pack("<140f", *(v for chunk in chunks for row in chunk for v in row))).hexdigest() != skill["actions_sha256"]:
        raise ValueError("Native actions changed")
    for name in ("positive_vectors", "negative_vectors"):
        values = skill[name]
        if (not isinstance(values, list) or not values
                or any(not isinstance(row, list) or len(row) != 402 or not all(finite(v) for v in row) for row in values)):
            raise ValueError("Bound public condition examples required")
    for name in ("response_min", "response_max", "entry_min", "entry_max"):
        values = skill[name]
        if not isinstance(values, list) or len(values) != 3 or not all(finite(v) for v in values):
            raise ValueError("Invalid motion contract")
    for prefix in ("response", "entry"):
        if any(a > b for a, b in zip(skill[prefix + "_min"], skill[prefix + "_max"])):
            raise ValueError("Reversed observed bounds")
    if not skill["source_bindings"] or any(len(h) != 64 for h in skill["source_bindings"].values()):
        raise ValueError("Missing feedback provenance")
    return skill


class NativeMotionEffectSkill(NativeStageContinuation):
    def __init__(self, *, skill, enabled, **kwargs):
        super().__init__(pause_steps=(), **kwargs)
        self.skill = copy.deepcopy(skill)
        self.enabled = bool(enabled)
        self.commands = []
        self.previous_robot = None
        self.last_commands = []
        self.mixed_close_seen = False
        self.entry_considered = False
        self.stage = 0
        self.entry_robot = None
        self.outputs = 0
        self.decisions = 0

    def _decision(self, observation):
        import numpy as np
        robot = np.asarray(observation["observation/state"], np.float32)
        if robot.shape != (8,) or not np.isfinite(robot).all() or observation.get("prompt") != self.goal:
            raise ValueError("Original public observation changed")
        decision, reason = None, "native"
        details = {}
        if self.stage == 1:
            movement = robot[:3] - self.entry_robot[:3]
            valid = (np.all(movement >= np.asarray(self.skill["response_min"]) - .01)
                     and np.all(movement <= np.asarray(self.skill["response_max"]) + .01)
                     and np.all(np.abs(robot[6:]) >= .001)
                     and np.all(np.abs(robot[6:]) <= .025)
                     and len(self.last_commands) == 10 and min(self.last_commands) > 0)
            details["measured_movement"] = movement.tolist()
            decision, reason = (1, "measured_response_inside_learned_envelope") if valid else (None, "response_or_public_state_mismatch")
            self.stage = 2
        elif (not self.entry_considered and self.mixed_close_seen
              and len(self.last_commands) == 10 and min(self.last_commands) > 0):
            self.entry_considered = True
            if self.previous_robot is None:
                reason = "history_missing"
            else:
                feature = public_vector(observation["observation/image"], observation["observation/wrist_image"],
                                        robot, self.previous_robot, self.commands[-20:])
                pos = float(np.min(np.sum((np.asarray(self.skill["positive_vectors"]) - feature) ** 2, axis=1)))
                neg = float(np.min(np.sum((np.asarray(self.skill["negative_vectors"]) - feature) ** 2, axis=1)))
                valid = (pos < neg
                         and np.all(robot[:3] >= np.asarray(self.skill["entry_min"]) - .04)
                         and np.all(robot[:3] <= np.asarray(self.skill["entry_max"]) + .04)
                         and np.all(np.abs(robot[6:]) >= .001)
                         and np.all(np.abs(robot[6:]) <= .025))
                details.update(positive_distance=pos, negative_distance=neg)
                decision, reason = (0, "closer_to_observed_positive_operation") if valid else (None, "entry_outside_observed_scope")
                if valid and self.enabled:
                    self.entry_robot = robot.copy()
                    self.stage = 1
        self.previous_robot = robot.copy()
        if decision is not None or reason != "native":
            self.decisions += 1
            self._event("native_motion_effect_skill_decision", enabled=self.enabled,
                        selected_chunk=decision if self.enabled else None, proposed_chunk=decision,
                        reason=reason, public_robot8=robot.tolist(), **details)
        return decision if self.enabled else None

    def run(self, original, runtime, skill_ctx, handle, adapter_plan, policy_client):
        if skill_ctx.session.session_id != self.session_id:
            return original(runtime, skill_ctx, handle, adapter_plan, policy_client)
        owner = self

        class TrackedHandle:
            def action_chunk(self, chunk):
                import numpy as np
                before = handle.session_state.step_index
                status = handle.action_chunk(chunk)
                consumed = handle.session_state.step_index - before
                actions = np.asarray(chunk["actions"], np.float32)
                if actions.ndim != 2 or actions.shape[1] != 7 or consumed > len(actions):
                    raise ValueError("Actual original decoded actions required")
                owner.last_commands = actions[:consumed, 6].tolist()
                owner.commands.extend(owner.last_commands)
                if owner.last_commands and min(owner.last_commands) < 0 < max(owner.last_commands):
                    owner.mixed_close_seen = True
                return status

            def __getattr__(self, name):
                return getattr(handle, name)

        class SamePolicy:
            def infer(self, observation):
                import numpy as np
                tick = time.perf_counter()
                chunk = owner._decision(observation)
                owner._event("native_motion_effect_condition_cost", seconds=time.perf_counter() - tick,
                             authority=AUTHORITY, actual_policy_call=chunk is None)
                if chunk is None:
                    return policy_client.infer(observation)
                owner.outputs += 1
                actions = np.asarray(owner.skill["chunks"][chunk], np.float32).copy()
                owner._event("native_motion_effect_skill_output", authority=AUTHORITY,
                             chunk_index=chunk, actions_float32_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                             actual_policy_call=False, grasp_certified=False,
                             physical_steps=handle.session_state.step_index)
                return dict(actions=actions, policy_meta=dict(policy_seq=0))

            def __getattr__(self, name):
                return getattr(policy_client, name)

        return super().run(original, runtime, skill_ctx, TrackedHandle(), adapter_plan, SamePolicy())

    def audit_snapshot(self):
        return {**super().audit_snapshot(), "authority": AUTHORITY, "enabled": self.enabled,
                "relative_outputs": self.outputs, "condition_decisions": self.decisions,
                "source_feedback_updates": len(self.skill["positive_vectors"]),
                "grasp_certified": False, "candidate_accepted": False}
