"""Immutable round versions; evaluation can read but never publish updates."""
import hashlib
import json
import math
import copy
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_state():
    return {'version': 0, 'lessons': [], 'skills': {}, 'history_md': '', 'token_budget': 384,
            'accepted_completion_tokens': [], 'update_receipts': []}


def publish(path, state, *, mode):
    if mode != 'development':
        raise PermissionError('Evaluation cannot publish persistent versions')
    with Path(path).open('x') as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def learned_budget(accepted_tokens):
    # A bounded engineering policy, not a calibrated quality guarantee.
    if len(accepted_tokens) < 2:
        return 384
    return min(384, max(192, int(math.ceil((2*max(accepted_tokens)+32)/32)*32)))


def with_fixed_experience(bundle, state):
    """Supply accepted skills to the real verifier, not just the shadow model."""
    value=copy.deepcopy(bundle)
    value['accepted_skills']={key:copy.deepcopy(rule) for key,rule in state['skills'].items()
                              if rule.get('validation_status')=='accepted'}
    value['fixed_experience_version']=state['version']
    value['fixed_reasoning_budget']=state['token_budget']
    return value
