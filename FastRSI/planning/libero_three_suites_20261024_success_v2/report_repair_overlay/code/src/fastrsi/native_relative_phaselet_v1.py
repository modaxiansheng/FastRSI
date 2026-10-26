"""One source-native relative action chunk, never an absolute waypoint replay.

This implements an opt-in developmental operation-value probe. Eligibility is
not learned or certified. The registered boundary is not a grasp label. The
original SDK loop/adapter/handle owns actions, termination and the shared budget.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import struct

from fastrsi.native_stage_continuation_v1 import NativeStageContinuation


AUTHORITY = "RELATIVE_PHASELET_OPERATION_VALUE_PROBE"


def load_phaselet(path, expected_sha, *, goal, model_config_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Relative phaselet changed")
    artifact = json.loads(raw)
    if (artifact.get("version") != "source_native_relative_phaselet_v1"
            or artifact.get("status") != "UNVALIDATED_CANDIDATE"
            or artifact.get("goal") != goal
            or artifact.get("model_config_sha256") != model_config_sha
            or artifact.get("source_native_success") is not True
            or artifact.get("source_experience_admission") != "ADMITTED"
            or artifact.get("action_units") != "original_relative_OSC_native_commands"
            or artifact.get("stage_or_grasp_label_certified") is not False):
        raise ValueError("Phaselet scope/provenance does not match this probe")
    values = artifact["actions"]
    if (not isinstance(values,list) or len(values)!=10
            or any(not isinstance(row,list) or len(row)!=7 for row in values)
            or any(type(v) not in (float,int) or not math.isfinite(v) or abs(v)>2
                   for row in values for v in row)
            or hashlib.sha256(struct.pack("<70f",*(v for row in values for v in row))).hexdigest()
               != artifact.get("actions_float32_sha256")):
        raise ValueError("Invalid original ten-step native action chunk")
    if not artifact.get("source_bindings") or any(len(v) != 64 for v in artifact["source_bindings"].values()):
        raise ValueError("Missing source file bindings")
    return artifact


class NativeRelativePhaseletChoice(NativeStageContinuation):
    def __init__(self, *, phaselet, **kwargs):
        super().__init__(**kwargs)
        self.phaselet = copy.deepcopy(phaselet)
        self._replacement = None
        self.relative_outputs = 0
        self.skipped_policy_calls = 0

    def allowed_actions(self):
        return ("continue", "relative_phaselet")

    def validate_action(self, action):
        if action not in self.allowed_actions():
            raise ValueError("Unbound operation")
        if action == "relative_phaselet" and self.relative_outputs:
            raise ValueError("This candidate permits one ten-step submission only")

    def resolved_boundary(self, request, action):
        if action == "relative_phaselet":
            if self._replacement is not None:
                raise ValueError("Unconsumed phaselet reply")
            self._replacement = request

    def run(self, original, runtime, skill_ctx, handle, adapter_plan, policy_client):
        if skill_ctx.session.session_id != self.session_id:
            return super().run(original, runtime, skill_ctx, handle, adapter_plan, policy_client)
        # The original policy loop does not consume the illustrative Session
        # replan_every_steps field. The entry binds the real fixed10 action
        # adapter; check the actual submitted shape in the execution receipt.
        owner = self
        class SamePolicy:
            def infer(self, observation):
                import numpy as np
                request = owner._replacement
                if request is None:
                    return policy_client.infer(observation)
                # Called by the original loop only after observing the same
                # handle. No teacher draw is spent to disguise reuse costs.
                state = observation.get("observation/state")
                if (observation.get("prompt") != owner.goal or state is None
                        or np.asarray(state).shape != (8,)):
                    raise ValueError("Original whole-goal observation changed")
                owner._replacement = None
                owner.relative_outputs += 1
                owner.skipped_policy_calls += 1
                owner._event("native_relative_phaselet_output",
                    authority=AUTHORITY, request_id=request.request_id,
                    observation_id=request.observation_id, physical_steps=request.physical_steps,
                    actions_float32_sha256=owner.phaselet["actions_float32_sha256"],
                    source_bindings=owner.phaselet["source_bindings"],
                    actual_policy_call=False, stage_or_grasp_label_certified=False,
                    current_public_robot8=np.asarray(state).tolist(),
                    current_arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                        sha256=hashlib.sha256(v.tobytes()).hexdigest())
                        for k,v in observation.items() if hasattr(v,"tobytes")})
                return dict(actions=np.asarray(owner.phaselet["actions"], dtype=np.float32).copy(),
                            policy_meta=dict(policy_seq=0))

            def __getattr__(self, name):
                return getattr(policy_client, name)
        result = super().run(original, runtime, skill_ctx, handle, adapter_plan, SamePolicy())
        return result

    def audit_snapshot(self):
        return {**super().audit_snapshot(), "authority": AUTHORITY,
                "executable_alternatives_bound": True,
                "relative_outputs": self.relative_outputs,
                "skipped_policy_calls": self.skipped_policy_calls,
                "candidate_accepted": False, "eligibility_learned": False}
