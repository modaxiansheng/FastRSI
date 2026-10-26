"""Native policy-input observation point. V1 is explicitly shadow-only.

Collects legal pre-action signals and development audit requests. It does not
claim that finite proprioception establishes a skill's physical applicability.
No shadow wake is counted as an actual decision call or saved invocation.
"""
import hashlib
import math
import re
import time
from .decision_validity import Contract, Dependency, Fact, assess, audit_selected


class ValidityObserver:
    def __init__(self, case, model_version):
        config = case.get('decision_validity_shadow')
        if config != {'version': 1, 'audit_seed': 17, 'audit_denominator': 20}:
            raise ValueError('Unbound validity shadow configuration')
        self.case = dict(case)
        self.model_version = model_version
        self.previous = None
        self.events = self.requests = self.audits = 0
        self.seconds = 0.0

    def observe(self, payload, observation, context):
        tick = time.perf_counter()
        sid = str(context.get('session_id', ''))
        obs_id = str(observation.get('observation_id', ''))
        match = re.fullmatch(r'libero_obs_(\d+)', obs_id)
        step = int(match.group(1)) if match else None
        state = payload.get('observation/state')
        try:
            values = [float(x) for x in state]
            finite = len(values) == 8 and all(math.isfinite(x) for x in values)
        except (TypeError, ValueError):
            values, finite = None, False
        goal = str(context.get('task_description', ''))
        goal_sha = hashlib.sha256(goal.encode()).hexdigest()
        # This probe asks whether the CURRENT recording supplies the evidence a
        # real applicability decision would need. No physical fact is invented.
        facts = {'proprio_finite': Fact(finite, step or 0, obs_id or 'missing_observation')}
        contract = None if step is None else Contract(
            'evidence-coverage-probe', sid or 'unknown', goal_sha,
            'native-full-goal-v1', self.model_version, step, step + 40,
            tick, tick + 30,
            (Dependency('proprio_finite', True, 20), Dependency('skill_applicable', True, 40)))
        previous_step = self.previous[1] if self.previous and self.previous[0] == sid else None
        result = assess(contract, session_id=sid, goal_sha256=goal_sha,
                        skill_version='native-full-goal-v1', model_version=self.model_version,
                        step=step, now=tick, facts=facts, previous_step=previous_step)
        audit = audit_selected(sid, obs_id, mode=self.case['mode'], seed=17, denominator=20)
        self.previous = (sid, step)
        self.events += 1; self.requests += int(result['wake']); self.audits += int(audit)
        elapsed = time.perf_counter() - tick; self.seconds += elapsed
        event = dict(event='decision_validity_shadow', case=self.case['id'], session_id=sid,
            observation_id=obs_id, step=step, mode=self.case['mode'],
            robot_state=values, goal=goal, **result, development_audit_selected=audit,
            controller_changed=False, actual_model_calls=0, actual_extra_observations=0,
            persistent_update=False, label=None, label_status='UNLABELLED', seconds=elapsed,
            scope='Evidence coverage probe, not an accepted validity contract or learned savings')
        return payload, event

    def summary(self):
        return dict(status='SHADOW_ONLY', observations=self.events, wake_requests=self.requests,
                    audit_selections=self.audits, compute_seconds=self.seconds,
                    actual_model_calls=0, persistent_updates=0, actual_saved_calls=None)


def install(adapter_cls, observer, emit):
    original = adapter_cls.to_policy_input
    def wrapped(adapter, observation, context):
        payload = original(adapter, observation, context)
        unchanged, event = observer.observe(payload, observation, context)
        emit(event)
        return unchanged
    adapter_cls.to_policy_input = wrapped
    def restore():
        adapter_cls.to_policy_input = original
    return restore
