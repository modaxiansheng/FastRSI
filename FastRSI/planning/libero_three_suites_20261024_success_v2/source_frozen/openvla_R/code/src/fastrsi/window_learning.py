"""Episode-level policy selection. Native outcome facts never admit VLM narratives."""
import copy
import hashlib
import json
import math
import time
from pathlib import Path

from .feedback_window import MODES


def context_key(goal):
    # Public instruction only. No task ID, init ID, simulator state or future label.
    words = goal.lower().split()
    return 'articulated_goal' if ('turn on' in goal.lower() or
                                  ('drawer' in words and 'close' in words)) else 'other'


def empty():
    return {'version': 1, 'default': 'gripper_transition', 'records': [],
            'scope': 'articulated_goal', 'updates': 0}


def update(state, records, *, phase):
    if phase != 'development':
        raise ValueError('Persistent window learning is development-only')
    out = copy.deepcopy(state)
    seen = {r['evidence_id'] for r in out['records']}
    for record in records:
        r = copy.deepcopy(record)
        if (r['mode'] != 'development' or r['policy'] not in MODES or
                r['native_facts_consistent'] is not True or
                type(r['native_goal_success']) is not bool or
                r['chain_status'] != 'PASSED' or not r['sources']):
            raise ValueError('Invalid native episode evidence')
        for key in ('full_wall_seconds', 'vla_calls', 'steps'):
            if not math.isfinite(r[key]) or r[key] <= 0:
                raise ValueError('Invalid paid episode cost')
        if r['evidence_id'] in seen:
            raise ValueError('Duplicate episode evidence')
        seen.add(r['evidence_id'])
        out['records'].append(r)
    out['updates'] += 1
    return out


def decide(state, goal):
    if state['version']==2:
        from .paired_window_learning import decide as paired_decide
        return paired_decide(state,goal)
    started = time.perf_counter()
    key = context_key(goal)
    rows = [r for r in state['records'] if context_key(r['goal']) == key]
    stats = {}
    for policy in MODES:
        rs = [r for r in rows if r['policy'] == policy]
        if rs:
            stats[policy] = {'n': len(rs),
                             'failures': sum(not r['native_goal_success'] for r in rs),
                             'mean_full_seconds': sum(r['full_wall_seconds'] for r in rs)/len(rs),
                             'mean_vla_calls': sum(r['vla_calls'] for r in rs)/len(rs)}
    # Two real episodes per option is a wiring-pilot minimum, NOT a quality guarantee.
    ready = key == state['scope'] and all(stats.get(p, {}).get('n', 0) >= 2 for p in MODES)
    selected = min(MODES, key=lambda p: (stats[p]['failures']/stats[p]['n'],
                                        stats[p]['mean_full_seconds'],
                                        stats[p]['mean_vla_calls'])) if ready else state['default']
    return {'selected_mode': selected, 'context': key, 'stats': stats,
            'reason': 'observed_failure_then_full_cost' if ready else 'insufficient_in_scope_evidence',
            'evidence_ids': [r['evidence_id'] for r in rows],
            'updates': state['updates'], 'selection_seconds': time.perf_counter()-started,
            'scope': 'Development mechanism pilot; no noninferiority or global promotion'}


def load(path, expected_sha):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError('Window snapshot hash mismatch')
    state = json.loads(data)
    if state['version'] not in (1,2) or state['default'] not in MODES:
        raise ValueError('Unsupported window snapshot')
    if state['version']==2 and state.get('incumbent') not in MODES:
        raise ValueError('Invalid paired incumbent')
    return state
