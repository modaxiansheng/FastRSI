"""Developmental low-translation signal; not a task-progress certificate.

Three distinct native H10 boundaries must span twenty actually executed steps.
The 1 cm spatial scale is the existing local controller tolerance, not a fitted
classifier or the joint-space threshold from PIP. This is a falsifiable rule.
"""
import time
import numpy as np

KEY = "public_translation_history_v1"


class PublicTranslationGate:
    def __init__(self, recipe):
        self.tolerance = recipe["position_tolerance"]
        self.open_min = recipe["open_finger_min"]
        self.history = []
        self.closed = False
        self.seconds = 0.

    def close(self):
        self.closed = True
        self.history.clear()

    def observe(self, step, state):
        tick = time.perf_counter()
        try:
            state = np.asarray(state, dtype=np.float64)
            if type(step) is not int or step < 0 or state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Fresh public step and robot8 required")
            row = dict(version=KEY, step=step, eligible=False,
                       physical_stall_certified=False, threshold_fitted=False)
            if self.closed:
                return {**row, "reason":"WATCH_CLOSED"}
            if self.history and step <= self.history[-1][0]:
                return {**row, "reason":"CACHED_OR_OLD_OBSERVATION"}
            if not (state[6] >= self.open_min and state[7] <= -self.open_min):
                self.history.clear()
                return {**row, "reason":"OPEN_FINGER_EVIDENCE_ABSENT"}
            if self.history and step-self.history[-1][0] != 10:
                self.history.clear()
            self.history.append((step,state[:3].copy()))
            self.history = self.history[-3:]
            if len(self.history) != 3:
                return {**row, "reason":"INSUFFICIENT_FRESH_EXECUTION_HISTORY",
                        "history_steps":[s for s,_ in self.history]}
            positions = np.array([p for _,p in self.history])
            diameter = float(np.linalg.norm(positions[:,None]-positions[None,:],axis=-1).max())
            return {**row, "eligible":diameter <= self.tolerance,
                    "reason":"LOW_PUBLIC_TRANSLATION" if diameter <= self.tolerance else "TRANSLATION_PROGRESS",
                    "history_steps":[s for s,_ in self.history], "xyz_diameter_m":diameter,
                    "xyz_tolerance_m":self.tolerance, "actual_step_span":20}
        finally:
            self.seconds += time.perf_counter()-tick
