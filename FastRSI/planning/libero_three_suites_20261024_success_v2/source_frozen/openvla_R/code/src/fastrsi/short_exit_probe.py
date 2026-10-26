"""One actual intervention; experimental execution policy, not a learned selector."""
import time

from .feedback_window import FeedbackWindow


class ShortExitOnce(FeedbackWindow):
    def __init__(self, case):
        super().__init__(case, 'gripper_transition')
        self.last_submitted = None
        self.intervened = False
        self.current_probe = None

    def _choose_horizon(self, signs, within, boundary, step):
        base = super()._choose_horizon(signs, within, boundary, step)
        eligible = (not self.intervened and self.last_submitted == 5 and base == 5
                    and self.case['max_steps'] - step >= 10)
        self.current_probe = {'baseline_requested_steps': base,
                              'prior_submitted_steps': self.last_submitted,
                              'eligible': eligible, 'intervention_applied': eligible,
                              'rule': 'First consecutive short commitment: force ten once, then resume original rule',
                              'scope': 'Predeclared development intervention; no learned threshold'}
        return 10 if eligible else base

    def apply(self, chunk, context):
        started = time.perf_counter()
        out, event = super().apply(chunk, context)
        event['mode'] = 'short_exit_once'
        event['short_exit_probe'] = self.current_probe.copy()
        if self.current_probe['intervention_applied']:
            self.intervened = True
        self.last_submitted = event['submitted_steps']
        event['selector_seconds'] = time.perf_counter() - started
        return out, event
