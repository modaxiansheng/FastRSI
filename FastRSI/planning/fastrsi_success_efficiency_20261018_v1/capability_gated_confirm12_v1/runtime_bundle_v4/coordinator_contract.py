"""Pure contracts shared by the isolated coordinator and static validation."""
from pathlib import Path
import hashlib
import json
from row_factory import MODELS, BRANCHES, LIMITS, HORIZONS, condition_row, proposal

SUITES = ('libero_object', 'libero_spatial', 'libero_goal', 'libero_10')
AUTHORIZATION_STATUS = 'EXISTING_USER_AUTHORIZATION_RECORDED'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups():
    _, rows = proposal()
    result = []
    for model in MODELS:
        for suite in SUITES:
            selected = [r for r in rows if r['suite'] == suite]
            result.append(dict(model=model, suite=suite,
                               physical_ids=[r['identity_id'] for r in selected],
                               conditions=len(selected), wire_horizon=HORIZONS[model][0],
                               execution_horizon=HORIZONS[model][1],
                               per_attempt_cap=LIMITS[suite],
                               normal_attempts_max=6 * len(selected)))
    assert len(result) == 16 and sum(r['conditions'] for r in result) == 48
    assert sum(r['normal_attempts_max'] for r in result) == 288
    return result


def validate_authorization(root, authorization, resource_binding, node, gpu, model, suite):
    # This records already granted user authority; it requests no new approval.
    assert authorization['status'] == AUTHORIZATION_STATUS
    assert authorization['full_evaluation'] == 'PAUSED'
    assert authorization['normal_physical_attempts_max'] == 288
    assert authorization['model_conditions'] == 48
    assert authorization['proposal_sha256'] == sha(root / 'CONFIRMATION_PROPOSAL.json')
    assert node in authorization['authorized_gpus'] and gpu in authorization['authorized_gpus'][node]
    assert authorization['canonical_actors_unchanged'] is True
    assert authorization['source_gate_not_bypassed'] is True
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
    assert len(frozen['rows']) == len(frozen['identities']) == 12
    return frozen['rows']


def group_rows(states, model, suite, node, gpu):
    expected = next(r for r in groups() if r['model'] == model and r['suite'] == suite)
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
