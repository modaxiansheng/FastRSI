"""A fixed developer intervention, never an inferred stage or learned policy.

Known init26 receives the original full goal before step120, a remaining-goal
instruction for 100 steps, then the full goal permanently. No completion facts
are invented and no ObjectStageMemory confirmation is bypassed.
"""
import hashlib
import re
from .object_stage_memory_v1 import GOAL, REMAINING

AUTHORITY = 'FIXED_DEVELOPMENT_INTERVENTION_NOT_LEARNED_OR_DEPLOYABLE'


def bindings(payload):
    return {k: dict(sha256=hashlib.sha256(payload[k].tobytes()).hexdigest(),
        shape=list(payload[k].shape), dtype=str(payload[k].dtype))
        for k in ('observation/image', 'observation/wrist_image', 'observation/state')}


class RemainingGoalValueProbe:
    def __init__(self, case, artifact, model_sha, *, enabled):
        if (case.get('mode'), case.get('suite'), case.get('task_id'), case.get('init_state_id'),
                case.get('max_steps'), case.get('task_description'), case.get('feedback_window_mode')) != (
                'development', 'libero_10', 4, 26, 520, GOAL, 'fixed10'):
            raise ValueError('Only registered known development init26/H10/W10 may run')
        if (artifact.get('version'), artifact.get('authority'), artifact.get('start_step'),
                artifact.get('end_step'), artifact.get('remaining_goal'), artifact.get('model_sha256')) != (
                1, AUTHORITY, 120, 220, REMAINING['white_mug'], model_sha):
            raise ValueError('Unbound developer intervention')
        if type(enabled) is not bool or set(artifact['prefix_bindings']) != {str(s) for s in range(0, 121, 10)}:
            raise ValueError('Explicit arm and complete source prefix required')
        self.case, self.artifact, self.enabled = dict(case), artifact, enabled
        self.previous = None; self.retired = False

    def input(self, payload, observation, context):
        if context.get('session_id') != self.case['id']:
            self.retired = True
            return payload, None
        if context.get('task_description') != GOAL or payload.get('prompt') != GOAL:
            raise ValueError('Original complete task changed')
        match = re.fullmatch(r'libero_obs_(\d+)', str(observation.get('observation_id')))
        if match is None: raise ValueError('Actual observation boundary required')
        step = int(match[1])
        if (not 0 <= step < 520 or step % 10 or
                (self.previous is None and step != 0) or
                (self.previous is not None and step != self.previous + 10)):
            raise ValueError('Actual consecutive H10 boundaries required')
        current = bindings(payload)
        if step <= 120 and current != self.artifact['prefix_bindings'][str(step)]:
            raise ValueError('Observed prefix differs from preregistered source')
        applied = self.enabled and not self.retired and 120 <= step < 220
        if step >= 220: self.retired = True
        self.previous = step
        result = {**payload, 'prompt': self.artifact['remaining_goal']} if applied else payload
        receipt = dict(authority=AUTHORITY, episode_id=self.case['id'], step=step,
            original_goal=GOAL, prompt=result['prompt'], applied=applied, enabled=self.enabled,
            array_bindings=current, prefix_source_match=step <= 120,
            new_model_calls=0, physical_completion_claim=None, persistent_update=False)
        return result, receipt
