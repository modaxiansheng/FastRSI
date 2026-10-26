"""Pure metadata factory for the fixed 12-identity capability-gated trial.

No environment/model imports, state generation, dispatch, or outcome reads.
Each physical arm receives its own capability row; null is intentional.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MODELS = ('pi05', 'pi0', 'openvla', 'xvla')
BRANCHES = ('ordinary', 'frozen', 'candidate')
LIMITS = {'libero_spatial': 220, 'libero_object': 280,
          'libero_goal': 300, 'libero_10': 520}
HORIZONS = {'pi05': (10, 10), 'pi0': (50, 10),
            'openvla': (1, 1), 'xvla': (1, 1)}
OLD_OBJECT = {0: 'alphabet_transfer_v1', 3: 'dev12', 5: 'dev13', 8: 'dev15'}
NEW_OBJECT = dict(OLD_OBJECT)
NEW_OBJECT.update({1: 'newtask_02', 4: 'newtask_07_prefix_v1', 6: 'dev14',
                   7: 'milk_transfer_v1', 9: 'newtask_13'})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposal():
    value = read(ROOT / 'CONFIRMATION_PROPOSAL.json')
    rows = value['rows']
    assert len(rows) == len({r['identity_id'] for r in rows}) == 12
    assert value['model_conditions'] == 48
    assert value['normal_physical_attempts_max'] == 288
    assert value['branches'] == list(BRANCHES)
    assert sum(r['suite'] == 'libero_object' for r in rows) == 6
    for suite in ('libero_spatial', 'libero_goal', 'libero_10'):
        group = [r for r in rows if r['suite'] == suite]
        assert len(group) == 2 and {r['task_id'] for r in group} == {0}
        assert [r['variant'] for r in group] == ['position', 'environment']
    for row in rows:
        assert row['max_steps'] == LIMITS[row['suite']]
        assert row['skill_by_branch'] == {
            branch: capability(row['suite'], row['task_id'], branch)
            for branch in BRANCHES}
    return value, rows


def capability(suite, task_id, branch):
    assert branch in BRANCHES and suite in LIMITS
    assert type(task_id) is int and task_id in range(10)
    if suite != 'libero_object' or branch == 'ordinary':
        return None
    return (OLD_OBJECT if branch == 'frozen' else NEW_OBJECT).get(task_id)


def physical_row(state):
    """Require a separately verified generated state, without reading an outcome."""
    _, design = proposal()
    expected = next(r for r in design if r['identity_id'] == state['identity_id'])
    for key in ('suite', 'task_id', 'variant', 'max_steps', 'language'):
        assert state[key] == expected[key], ('PRECOMMITTED_DESIGN_MISMATCH', key)
    for key in ('initial_state_sha256', 'physical_state_sha256', 'asset_sha256',
                'source_receipt_sha256'):
        value = state[key]
        assert isinstance(value, str) and len(value) == 64
        assert set(value) <= set('0123456789abcdef')
    assert state['state_path'].startswith('states/')
    assert '..' not in Path(state['state_path']).parts
    assert state.get('source', 'official_initializer_bounded_development') == \
        'official_initializer_bounded_development'
    return expected


def condition_row(state, model, node, gpu):
    expected = physical_row(state)
    assert model in MODELS and isinstance(node, str)
    assert type(gpu) is int and gpu >= 0
    source = deepcopy(state)
    source.update(subject=expected.get('subject'),
                  support='CAPABILITY_GATED_PRECOMMITTED_DEVELOPMENT')
    identity = dict(identity_id=state['identity_id'], suite=state['suite'],
                    perturbation=state['variant'], task_id=state['task_id'],
                    source='official_initializer_bounded_development', split='development')
    for key in ('initial_state_sha256', 'physical_state_sha256', 'asset_sha256',
                'source_receipt_sha256'):
        identity[key] = state[key]
    wire, execution = HORIZONS[model]
    return dict(identity_id=model + '-' + state['identity_id'],
                physical_identity_id=state['identity_id'], identity=identity,
                model=model, node=node, gpu=gpu, source=source,
                wire_horizon=wire, execution_horizon=execution,
                skill=capability(state['suite'], state['task_id'], 'candidate'),
                skill_by_branch=deepcopy(expected['skill_by_branch']))


def branch_row(condition, branch, after_baseline_rng_hashes,
               baseline_reset_state_sha256, baseline_first_input_sha256):
    assert branch in BRANCHES
    row = deepcopy(condition)
    canonical = condition['identity_id']
    row['canonical_condition'] = canonical
    row['identity_id'] = canonical if branch == 'ordinary' else canonical + '-' + branch
    row['skill'] = capability(row['source']['suite'], row['source']['task_id'], branch)
    row['source']['skill'] = row['skill']
    row['support'] = ('SOURCE_SKILL_UNAVAILABLE_NATIVE_CONTINUATION' if row['skill'] is None
                      else 'EXACT_IMMUTABLE_SOURCE_CAPABILITY')
    row['after_baseline_rng_hashes'] = deepcopy(after_baseline_rng_hashes)
    row['baseline_reset_state_sha256'] = baseline_reset_state_sha256
    row['baseline_first_input_sha256'] = baseline_first_input_sha256
    return row


def baseline_roster_row(condition):
    s = condition['source']
    return dict(id=condition['identity_id'] + '--base', suite=s['suite'],
                task_id=s['task_id'], task_description=s['language'],
                development_identity=condition['identity'],
                development_state_path=s['state_path'])
