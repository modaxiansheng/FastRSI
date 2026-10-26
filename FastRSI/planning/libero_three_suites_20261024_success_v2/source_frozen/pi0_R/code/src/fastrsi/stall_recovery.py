"""Bounded simulation-only retreat candidate, not a learned or accepted skill.

The trigger reads only past/current policy proprioception. A stationary open
gripper is a measurable condition, not proof of task failure or an empty hand.
All replacement actions consume the original episode budget. Native inference
is retained during the probe so pre-intervention RNG and cost remain comparable.
"""
import math


SPEC = dict(version='stall_retreat_probe_v1', history_steps=50,
            max_position_extent_m=0.01, minimum_open_finger_m=0.035,
            lift_m=0.08, recovery_steps=30, maximum_uses=1,
            translation_output_scale=0.05, action_limit=0.6)


class StallRecovery:
    def __init__(self, case, enabled, spec):
        if spec != SPEC:
            raise ValueError('Unbound recovery specification')
        if (case.get('mode') != 'development' or case.get('suite') != 'libero_10'
                or case.get('feedback_window_mode') != 'fixed10'
                or case.get('max_steps') != 520):
            raise ValueError('Recovery probe requires an H10/W10 development episode')
        self.case, self.enabled = dict(case), bool(enabled)
        self.history = []
        self.home = None
        self.trigger_step = None
        self.origin = None
        self.used = 0

    def choose(self, state, step, session_id, prompt):
        if session_id != self.case['id'] or prompt != self.case['task_description']:
            return None, dict(reason='outside_root_session', applied=False)
        if (type(step) is not int or step < 0 or step % 10 or len(state) != 8
                or any(not math.isfinite(float(x)) for x in state)):
            raise ValueError('Invalid current policy observation')
        state = list(map(float, state))
        if self.history and step != self.history[-1][0] + 10:
            raise ValueError('Reset, missing or repeated policy observation')
        if not self.history:
            if step != 0:
                raise ValueError('Missing initial state')
            self.home = state[:3]
        self.history.append((step, state))
        self.history = [(s, x) for s, x in self.history if s >= step - SPEC['history_steps']]
        extent = math.sqrt(sum((max(x[j] for _, x in self.history) -
                                min(x[j] for _, x in self.history)) ** 2 for j in range(3)))
        observed_open = all(x[6] >= SPEC['minimum_open_finger_m'] and
                            x[7] <= -SPEC['minimum_open_finger_m'] for _, x in self.history)
        eligible = (step - self.history[0][0] == SPEC['history_steps'] and
                    extent < SPEC['max_position_extent_m'] and observed_open and
                    step + SPEC['recovery_steps'] + 50 <= self.case['max_steps'])
        event = dict(step=step, position_extent_m=extent,
                     observed_open=observed_open, eligible=eligible, applied=False,
                     assumption='Stationarity/open fingers do not establish semantic failure')
        if self.enabled and eligible and self.used == 0:
            self.trigger_step, self.origin = step, state[:3]
            self.used += 1
        if self.trigger_step is None or not 0 <= step-self.trigger_step < SPEC['recovery_steps']:
            return None, event
        offset = step-self.trigger_step
        # First lift vertically; then retreat toward the episode's observed
        # starting XY at elevated Z. No object positions or goal predicates.
        height = max(self.origin[2], self.home[2]) + SPEC['lift_m']
        target = [*self.origin[:2], height] if offset == 0 else [*self.home[:2], height]
        remaining = 10 if offset == 0 else SPEC['recovery_steps']-offset
        delta = [max(-SPEC['action_limit'], min(SPEC['action_limit'],
                  (target[j]-state[j])/(SPEC['translation_output_scale']*remaining))) for j in range(3)]
        actions = [delta + [0.0, 0.0, 0.0, -1.0] for _ in range(10)]
        event.update(applied=True, trigger_step=self.trigger_step,
                     phase='lift' if offset == 0 else 'retreat', target_xyz=target,
                     policy_prediction_discarded=True, replacement_steps=10)
        return actions, event
