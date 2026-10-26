"""Explicit decision validity, not a safety certificate or learned policy.

Unknown, stale and contradicted evidence are different wake reasons. This module
never infers object facts from missing events, never changes VLA feedback, and
cannot renew its own contract. A caller must supply a new validated decision.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name)


@dataclass(frozen=True)
class Fact:
    value: bool | None
    observed_step: int
    source: str

    def __post_init__(self):
        if self.value is not None and type(self.value) is not bool:
            raise ValueError('Fact value must be true, false or unknown')
        _integer(self.observed_step, 'observed_step')
        if not isinstance(self.source, str) or not self.source:
            raise ValueError('Missing evidence provenance')


@dataclass(frozen=True)
class Dependency:
    name: str
    expected: bool
    max_age_steps: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or type(self.expected) is not bool:
            raise ValueError('Invalid fact dependency')
        _integer(self.max_age_steps, 'max_age_steps', 1)


@dataclass(frozen=True)
class Contract:
    decision_id: str
    session_id: str
    goal_sha256: str
    skill_version: str
    model_version: str
    issued_step: int
    expires_step: int
    issued_monotonic: float
    expires_monotonic: float
    dependencies: tuple[Dependency, ...]

    def __post_init__(self):
        for key in ('decision_id', 'session_id', 'goal_sha256', 'skill_version', 'model_version'):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError('Missing contract binding: ' + key)
        _integer(self.issued_step, 'issued_step')
        _integer(self.expires_step, 'expires_step', self.issued_step + 1)
        times = (self.issued_monotonic, self.expires_monotonic)
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in times):
            raise ValueError('Invalid monotonic clock')
        if not 0 <= times[0] < times[1]:
            raise ValueError('Invalid contract time range')
        if not self.dependencies or len({d.name for d in self.dependencies}) != len(self.dependencies):
            raise ValueError('Missing or duplicate dependencies')


def assess(contract, *, session_id, goal_sha256, skill_version, model_version,
           step, now, facts, previous_step=None, tool_error=False, unresolved=False):
    """Read-only, O(number of declared dependencies), no model or image encoding."""
    if type(step) is not int or step < 0 or type(now) not in (int, float) or not math.isfinite(now):
        return {'wake': True, 'reasons': ['invalid_clock_or_observation'], 'missing': []}
    reasons, missing = [], []
    if previous_step is not None and step <= previous_step:
        reasons.append('reset_or_nonadvancing_observation')
    if tool_error:
        reasons.append('tool_error')
    if unresolved:
        reasons.append('unresolved_issue')
    if contract is None:
        reasons.append('no_decision_contract')
    else:
        for key, value in (('session_id', session_id), ('goal_sha256', goal_sha256),
                           ('skill_version', skill_version), ('model_version', model_version)):
            if getattr(contract, key) != value:
                reasons.append('changed_' + key)
        if step < contract.issued_step or now < contract.issued_monotonic:
            reasons.append('clock_precedes_decision')
        if step >= contract.expires_step or now >= contract.expires_monotonic:
            reasons.append('decision_expired')
        for dep in contract.dependencies:
            fact = facts.get(dep.name)
            if not isinstance(fact, Fact) or fact.value is None:
                reasons.append('unknown:' + dep.name); missing.append(dep.name)
            elif fact.observed_step > step:
                reasons.append('future_evidence:' + dep.name)
            elif step - fact.observed_step >= dep.max_age_steps:
                reasons.append('stale:' + dep.name); missing.append(dep.name)
            elif fact.value != dep.expected:
                reasons.append('contradicted:' + dep.name)
    return {'wake': bool(reasons), 'reasons': reasons, 'missing': missing}


def audit_selected(session_id, observation_id, *, mode, seed, denominator=20):
    """Outcome-independent development sampling; selection is not a label."""
    if mode not in ('development', 'evaluation'):
        raise ValueError('Unknown data role')
    _integer(denominator, 'audit denominator', 1)
    _integer(seed, 'audit seed')
    if mode != 'development':
        return False
    key = f'{seed}:{session_id}:{observation_id}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % denominator == 0


def validate_choice(response, allowed):
    """Reject invented options, non-finite scores and malformed distributions."""
    if not allowed or len(set(allowed)) != len(allowed):
        raise ValueError('Empty or duplicate legal choices')
    if not isinstance(response, dict) or response.get('choice') not in allowed:
        raise ValueError('Illegal decision choice')
    probs = response.get('probabilities')
    if not isinstance(probs, dict) or set(probs) != set(allowed):
        raise ValueError('Incomplete candidate probabilities')
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError('Invalid probabilities')
    if abs(sum(probs.values()) - 1) > 0.002:
        raise ValueError('Probabilities not normalized')
    if probs[response['choice']] < max(probs.values()):
        raise ValueError('Choice does not match distribution')
    return dict(response)
