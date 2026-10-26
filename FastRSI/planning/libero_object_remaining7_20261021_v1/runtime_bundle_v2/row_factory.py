"""Pure metadata adaptation of the successful confirm12 factory: 14 new Object IDs.

No environment/model imports or outcome-dependent selection. The old recovery is
the original three-object release, so none of these seven new objects has an old
skill. Alphabet transfer is a NEW-arm skill here, never silently an old skill.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MODELS = ('pi05', 'pi0', 'openvla', 'xvla')
BRANCHES = ('ordinary', 'frozen', 'candidate')
LIMITS = {'libero_object': 280}
HORIZONS = {'pi05': (10, 10), 'pi0': (50, 10),
            'openvla': (1, 1), 'xvla': (1, 1)}
TASKS = (0, 1, 2, 4, 6, 7, 9)
SKILLS = {0: 'alphabet_transfer_v1', 1: 'newtask_02', 2: None,
          4: 'newtask_07_prefix_v1', 6: 'dev14',
          7: 'milk_transfer_v1', 9: 'newtask_13'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_capabilities(root=None):
    root = ROOT if root is None else Path(root)
    manifest = read(root / 'TASK_CAPABILITY_ADMISSION.json')
    assert manifest['schema'] == 'fastrsi_object7_independent_capability_admission_v1'
    assert manifest['old_all_six_gate_unchanged'] is True
    assert set(manifest['tasks']) == {str(v) for v in TASKS}
    for task in TASKS:
        item = manifest['tasks'][str(task)]
        assert item['task_id'] == task and item['skill'] == SKILLS[task]
        assert item['capability_pass'] is (task != 2)
        if task == 2:
            assert item['status'] == 'SKIPPED_UNSUPPORTED'
            assert item['physical_attempts_in_new_arm'] == 0
        for proof in item['proofs']:
            p = root / proof['path']
            assert not Path(proof['path']).is_absolute() and '..' not in p.relative_to(root).parts
            assert sha(p) == proof['sha256']
            value = read(p)
            for component in proof['json_pointer'].split('/'):
                if component:
                    value = value[int(component)] if isinstance(value, list) else value[component]
            assert value is proof['expected'], ('CAPABILITY_PROOF_CHANGED', task, proof)
    return manifest


def proposal():
    value = read(ROOT / 'CONFIRMATION_PROPOSAL.json')
    rows = value['rows']
    assert len(rows) == len({r['identity_id'] for r in rows}) == 14
    assert value['model_conditions'] == 56
    assert value['normal_physical_attempts_max'] == 336
    assert value['pi05_first_stage_normal_attempts_max'] == 84
    assert value['branches'] == list(BRANCHES)
    assert [(r['task_id'], r['variant']) for r in rows] == [
        (task, variant) for task in TASKS for variant in ('position', 'environment')]
    for row in rows:
        assert row['suite'] == 'libero_object' and row['max_steps'] == 280
        assert row['skill_by_branch'] == {
            branch: capability(row['suite'], row['task_id'], branch)
            for branch in BRANCHES}
    return value, rows


def capability(suite, task_id, branch):
    assert branch in BRANCHES and suite == 'libero_object'
    assert type(task_id) is int and task_id in TASKS
    return SKILLS[task_id] if branch == 'candidate' else None


def branch_disposition(condition, branch):
    assert branch in BRANCHES
    return ('SKIPPED_UNSUPPORTED' if branch == 'candidate'
            and condition['source']['task_id'] == 2 else 'EXECUTE_ON_BASELINE_FAILURE')


def final_outcomes(baseline_success, arms):
    """Unknown/unsupported is null, never copied from another arm or scored false."""
    assert type(baseline_success) is bool
    if baseline_success:
        return {branch: True for branch in BRANCHES}
    result = {}
    for branch in BRANCHES:
        arm = arms[branch]
        if arm.get('status') == 'SKIPPED_UNSUPPORTED':
            assert branch == 'candidate' and arm['physical_attempts'] == 0
            result[branch] = None
        else:
            success = arm['result']['physical_success']
            assert type(success) is bool
            result[branch] = success
    return result


def paired_difference(candidate, control):
    assert candidate is None or type(candidate) is bool
    assert control is None or type(control) is bool
    return None if candidate is None or control is None else int(candidate) - int(control)


def physical_row(state):
    _, design = proposal()
    expected = next(r for r in design if r['identity_id'] == state['identity_id'])
    for key in ('suite', 'task_id', 'variant', 'max_steps', 'language', 'subject', 'seed'):
        assert state[key] == expected[key], ('PRECOMMITTED_DESIGN_MISMATCH', key)
    for key in ('initial_state_sha256', 'physical_state_sha256', 'asset_sha256',
                'source_receipt_sha256'):
        value = state[key]
        assert isinstance(value, str) and len(value) == 64
        assert set(value) <= set('0123456789abcdef')
    path = Path(state['state_path'])
    assert not path.is_absolute() and path.parts[0] == 'states' and '..' not in path.parts
    assert state['nq'] == 58, 'Object actual nq is 58, never the LIBERO10 nq65 projection'
    assert state.get('source', 'official_initializer_bounded_development') == \
        'official_initializer_bounded_development'
    return expected


def condition_row(state, model, node, gpu):
    expected = physical_row(state)
    assert model in MODELS and node in ('187', '188')
    assert type(gpu) is int and gpu in range(8)
    source = deepcopy(state)
    source['support'] = 'PER_TASK_CAPABILITY_FIXED_NEW_DEVELOPMENT'
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
                skill=expected['skill_by_branch']['candidate'],
                skill_by_branch=deepcopy(expected['skill_by_branch']),
                candidate_status=('SKIPPED_UNSUPPORTED' if state['task_id'] == 2
                                  else 'FROZEN_SOURCE_CAPABILITY_AVAILABLE'))


def branch_row(condition, branch, after_baseline_rng_hashes,
               baseline_reset_state_sha256, baseline_first_input_sha256):
    assert branch_disposition(condition, branch) == 'EXECUTE_ON_BASELINE_FAILURE'
    row = deepcopy(condition)
    canonical = condition['identity_id']
    row['canonical_condition'] = canonical
    row['identity_id'] = canonical if branch == 'ordinary' else canonical + '-' + branch
    row['skill'] = capability(row['source']['suite'], row['source']['task_id'], branch)
    row['source']['skill'] = row['skill']
    row['support'] = ('ORIGINAL_NATIVE_FALLBACK_NO_OBJECT_SKILL' if row['skill'] is None
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
