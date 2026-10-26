"""Pure contracts shared by the isolated coordinator and static validation."""
from pathlib import Path
import hashlib
import json
from row_factory import MODELS, BRANCHES, LIMITS, HORIZONS, condition_row, proposal, TASKS, check_capabilities

SUITES = ('libero_object',)
AUTHORIZATION_STATUS = 'EXISTING_USER_AUTHORIZATION_RECORDED'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups():
    _, rows = proposal()
    result = []
    for model in MODELS:
        for task in TASKS:
            suite = 'libero_object'
            selected = [r for r in rows if r['task_id'] == task]
            result.append(dict(model=model, suite=suite, task_id=task,
                               physical_ids=[r['identity_id'] for r in selected],
                               conditions=len(selected), wire_horizon=HORIZONS[model][0],
                               execution_horizon=HORIZONS[model][1],
                               per_attempt_cap=LIMITS[suite],
                               normal_attempts_max=6 * len(selected)))
    assert len(result) == 28 and sum(r['conditions'] for r in result) == 56
    assert sum(r['normal_attempts_max'] for r in result) == 336
    return result


def validate_authorization(root, authorization, resource_binding, node, gpu, model, suite):
    # This records already granted user authority; it requests no new approval.
    assert authorization['status'] == AUTHORIZATION_STATUS
    assert authorization['full_evaluation'] == 'PAUSED'
    assert authorization['normal_physical_attempts_max'] == 336
    assert authorization['model_conditions'] == 56
    assert authorization['proposal_sha256'] == sha(root / 'CONFIRMATION_PROPOSAL.json')
    assert node in authorization['authorized_gpus'] and gpu in authorization['authorized_gpus'][node]
    assert authorization['canonical_actors_unchanged'] is True
    assert authorization['per_task_source_gates_verified'] is True
    check_capabilities(root)
    direct = authorization['direct_user_authorization']
    assert sha(direct['path']) == direct['sha256']
    human_record = read(direct['path'])
    assert human_record['schema'] == 'fastrsi_object7_direct_human_authorization_v1'
    assert human_record['full_evaluation'] == 'PAUSED'
    approved = human_record['approved_stage']
    assert approved['tasks'] == list(TASKS) and approved['physical_identities'] == 14
    assert approved['models'] == list(MODELS) and approved['first_model'] == 'pi05'
    assert approved['first_stage_max_physical_attempts'] == 84
    assert approved['all_stage_max_physical_attempts'] == 336
    assert approved['official_action_cap_each'] == 280
    assert approved['resources'] == ['187', '188'] and approved['new_training_started'] is False
    assert human_record['proposal']['sha256'] == read(root / 'CONFIRMATION_PROPOSAL.json')['source_proposal_sha256']
    assert authorization['pi05_first_stage_normal_attempts_max'] == 84
    if model != 'pi05':
        verify_pi05_stage(root)
    assert authorization['source_and_failed_predecessor_costs_retained'] is True
    receipt = Path(resource_binding['external_binding_receipt']['path'])
    assert receipt.is_file() and sha(receipt) == resource_binding['external_binding_receipt']['sha256']
    bindings = read(receipt)
    assert bindings['accepted'] is True
    assert bindings['actual_used_inputs_only'] is True
    assert bindings['frozen_weight_identity_proved'] is True
    key = model + ':' + suite
    assert key in bindings['applicable_groups'], ('Host dependency identity not bound for group', key)
    checkpoint = ({'libero_object': 'openvla_object', 'libero_spatial': 'openvla_spatial',
                   'libero_goal': 'openvla_goal', 'libero_10': 'openvla_libero10'}[suite]
                  if model == 'openvla' else
                  {'pi05': 'pi05_libero_pytorch_fp32', 'pi0': 'pi0_libero_pytorch_fp32',
                   'xvla': 'xvla_libero'}[model])
    assert checkpoint in bindings['checkpoint_identities'], ('Checkpoint identity not yet bound', checkpoint)
    assert bindings['checkpoint_identities'][checkpoint]['fresh_sha_verified'] is True
    # Root verifies/reuses the immutable checkpoint manifests once, not per lane.
    # Each lane checks the small receipt/manifests and explicitly used file hashes.
    for item in bindings['pinned_manifests'] + bindings['used_source_and_config_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    for dependency in bindings['dependency_identities']:
        assert dependency['scope'] and dependency['binding_files']
        for item in dependency['binding_files']:
            assert sha(item['path']) == item['sha256'], item['path']
    contract = read(root / 'LAUNCH_AND_COST_CONTRACT.json')
    assert set(resource_binding['images']) == {
        contract['policy_image'], contract['environment_and_cpu_vision_image']}
    return resource_binding


def verified_states(root):
    binding = read(root / 'STATE_OUTPUT_BINDING.json')
    assert binding['status'] == 'INDEPENDENT_FROZEN_ROSTER_BOUND_FOR_RUNTIME'
    generated = Path(binding['generated_host_root'])
    verification = Path(binding['verification_host_root'])
    assert generated != verification
    for directory, files in [(generated, binding['files']),
                             (verification, binding['verification_files'])]:
        for name, digest in files.items():
            assert sha(directory / name) == digest, name
    frozen = read(verification / 'FROZEN_ROSTER.json')
    assert frozen['accepted'] is True and frozen['overlap'] == 0
    assert frozen['generation_read_only'] is True
    declaration = read(root / 'STATE_PREPARATION_BINDING.json')
    assert frozen['task_design_sha256'] == declaration['task_design_sha256']
    assert frozen['before_generation_freeze_sha256'] == declaration['before_generation_freeze_sha256']
    assert len(frozen['rows']) == len(frozen['identities']) == 14
    return frozen['rows']


def group_rows(states, model, suite, node, gpu, task_id):
    expected = next(r for r in groups() if r['model'] == model and r['suite'] == suite and r['task_id'] == task_id)
    selected = [r for r in states if r['identity_id'] in expected['physical_ids']]
    assert [r['identity_id'] for r in selected] == expected['physical_ids']
    rows = [condition_row(state, model, node, gpu) for state in selected]
    assert len(rows) == expected['conditions']
    assert all(r['source']['suite'] == suite and r['model'] == model for r in rows)
    assert all(r['source']['max_steps'] == LIMITS[suite] for r in rows)
    return rows


def frozen_files(root):
    receipt = read(root / 'BUNDLE_BYTE_COPY_RECEIPT.json')
    assert receipt['local_metadata_only'] is False
    assert receipt['historical_sources_modified'] is False
    assert receipt['source_manifest_sha256'] == sha(root / 'BUILD_INPUT_MANIFEST.json')
    for name, digest in receipt['files'].items():
        assert sha(root / name) == digest, name
    manifest = read(root / 'BUILD_INPUT_MANIFEST.json')
    for name, digest in manifest['package_files'].items():
        assert sha(root / name) == digest, name
    for entry in manifest['files']:
        assert sha(root / entry['destination']) == entry['sha256'], entry['destination']
    return receipt


def branch_final(baseline, physical):
    assert type(baseline) is bool and type(physical) is bool
    return baseline or physical


def validate_arm(result, row):
    assert result['software_ok'] is True and result['error'] is None
    assert result['episode_completed'] is True
    assert type(result['physical_success']) is bool
    assert result['native_accounting_complete'] is True
    assert result['GT_control_input'] is False
    assert 0 <= result['official_actions'] <= row['source']['max_steps']
    if row['skill'] is None:
        assert result['visual_requests'] == 0
        assert result['stop_reason'] in ['SOURCE_SKILL_UNAVAILABLE', None]
    return result


def verify_pi05_stage(root, marker=None):
    marker = read(root / 'PI05_STAGE_COMPLETE.json') if marker is None else marker
    assert marker['schema'] == 'fastrsi_object7_pi05_stage_complete_v1'
    assert marker['frozen_runtime_scope_sha256'] == sha(root / 'FROZEN_RUNTIME_SCOPE.json')
    scope = read(root / 'FROZEN_RUNTIME_SCOPE.json')
    expected = [g for g in scope['groups'] if g['model'] == 'pi05']
    assert len(expected) == len(marker['groups']) == 7
    total = 0
    for group, pinned in zip(expected, marker['groups']):
        assert pinned['task_id'] == group['task_id']
        lane = root / 'campaigns/libero_object__pi05' / ('task' + str(group['task_id'])) / ('node' + group['node']) / ('gpu' + str(group['gpu']))
        for name in ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json'):
            assert sha(lane / name) == pinned['files'][name]
        result = read(lane / 'GROUP_RESULT.json')
        assert result['status'] == 'FROZEN_GROUP_COMPLETED' and result['completed'] == group['condition_ids']
        total += result['normal_physical_attempts']
        closed = read(lane / 'OWNED_STOP_RECEIPT.json')
        assert closed['cleanup_errors'] == []
        assert all(item['state']['Running'] is False for item in closed['states'].values())
        lifecycle = read(lane / 'LIFECYCLE.json')
        assert lifecycle['error'] is None and lifecycle['cleanup_errors'] == []
        for condition in group['condition_ids']:
            path = lane / condition / 'CONDITION_RESULT.json'
            assert sha(path) == pinned['conditions'][condition]
            record = read(path)
            assert record['software_ok'] is True
            assert all(type(record['final'][b]) is bool for b in ('ordinary', 'frozen'))
            if group['task_id'] == 2 and record['native_up_to3'] is False:
                assert record['final']['candidate'] is None
                assert record['branches']['candidate']['status'] == 'SKIPPED_UNSUPPORTED'
            else:
                assert type(record['final']['candidate']) is bool
    assert total == marker['normal_physical_attempts_completed'] <= 84
    return marker
