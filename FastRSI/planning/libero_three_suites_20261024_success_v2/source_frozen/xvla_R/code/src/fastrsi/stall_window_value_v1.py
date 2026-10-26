"""One bounded development question: can native short feedback recover a stall?

This reuses the previously recorded proprioceptive trigger unchanged. It never
emits replacement motion, declares object facts, or constitutes a learned skill.
"""
import math
from .stall_recovery import StallRecovery, SPEC


class StallWindowValue:
    def __init__(self, case, enabled):
        if case.get('mode') != 'development' or case.get('inference_horizon', 10) != 10:
            raise ValueError('Only registered H10 development')
        self.case = dict(case)
        self.detector = StallRecovery(case, False, dict(SPEC))
        self.enabled = bool(enabled)
        self.trigger_step = None
        self.current_step = None
        self.current_event = None

    def observe(self, state, step, session_id, prompt):
        if session_id != self.case['id'] or prompt != self.case['task_description']:
            raise ValueError('Wrong root or goal')
        if (type(step) is not int or not 0 <= step < self.case['max_steps']
                or len(state) != 8 or any(not math.isfinite(float(v)) for v in state)):
            raise ValueError('Invalid public observation or exhausted root budget')
        if self.current_step is not None and step <= self.current_step:
            raise ValueError('Repeated or nonmonotonic observation')
        if self.trigger_step is None:
            actions, event = self.detector.choose(state, step, session_id, prompt)
            assert actions is None  # The old retreat is never enabled or executed.
            if self.enabled and event['eligible']:
                self.trigger_step = step
        else:
            if step != self.current_step + 5:
                raise ValueError('Short-feedback continuation lost its boundary')
            event = dict(step=step, reason='already_switched_once')
        self.current_step = step
        self.current_event = dict(event, trigger_step=self.trigger_step,
                                  requested_steps=5 if self.trigger_step is not None else 10,
                                  replaced_motion=False, learned=False)
        return dict(self.current_event)

    def horizon(self, step):
        if step != self.current_step:
            raise ValueError('No current legal robot observation at this boundary')
        return 5 if self.trigger_step is not None else 10
