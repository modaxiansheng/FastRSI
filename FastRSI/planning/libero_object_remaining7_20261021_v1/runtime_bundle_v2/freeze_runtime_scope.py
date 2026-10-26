"""Freeze 28 two-identity groups after independent state verification; no launch."""
from pathlib import Path
import argparse
import json
import time
from coordinator_contract import groups, read, sha, verified_states, group_rows, frozen_files
from row_factory import check_capabilities

ROOT = Path(__file__).resolve().parent


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--placement', type=Path, required=True)
    args = parser.parse_args()
    frozen_files(ROOT)
    check_capabilities(ROOT)
    states = verified_states(ROOT)
    placed = read(args.placement)['groups']
    expected = groups()
    key = lambda item: (item['model'], item['suite'], item['task_id'])
    assert len(placed) == len(expected) == 28
    assert len({key(g) for g in placed}) == 28
    assert {key(g) for g in placed} == {key(g) for g in expected}
    rows, group_docs = [], []
    for group in expected:
        location = next(p for p in placed if key(p) == key(group))
        assert location['node'] in ('187', '188') and type(location['gpu']) is int
        assert location['gpu'] in range(8)
        conditions = group_rows(states, group['model'], group['suite'],
                                location['node'], location['gpu'], group['task_id'])
        rows.extend(conditions)
        group_docs.append(dict(group, node=location['node'], gpu=location['gpu'],
                               condition_ids=[r['identity_id'] for r in conditions]))
    assert len(rows) == len({r['identity_id'] for r in rows}) == 56
    assert len({r['physical_identity_id'] for r in rows}) == 14
    assert sum(r['skill'] is not None for r in rows) == 48
    scope = dict(status='ALL56_OBJECT7_CONDITIONS_FROZEN_BEFORE_TRIAL', time=time.time(),
                 input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
                 state_output_binding_sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'),
                 proposal_sha256=sha(ROOT / 'CONFIRMATION_PROPOSAL.json'),
                 placement_sha256=sha(args.placement), groups=group_docs, conditions=rows,
                 normal_physical_attempts_max=336, pi05_first_stage_normal_attempts_max=84,
                 branches=['ordinary', 'frozen', 'candidate'],
                 additional_attempts_per_failed_condition_per_branch=1,
                 primary_model_conditions=56, physical_identities=14,
                 unsupported_candidate_conditions=8,
                 unsupported_candidate_status='SKIPPED_UNSUPPORTED',
                 unsupported_is_not_a_physical_failure=True,
                 success_pooling=False, compute_budget_matched=False, training=False,
                 full_evaluation='PAUSED', old_six_source_study='CLOSED5OF6_UNCHANGED',
                 execution_started=False)
    save(ROOT / 'FROZEN_RUNTIME_SCOPE.json', scope)
    manifest = read(ROOT / 'BUILD_INPUT_MANIFEST.json')
    save(ROOT / 'EXECUTION_BINDINGS.json', dict(
        status='EXECUTION_BYTES_AND56_OBJECT7_CONDITIONS_FROZEN', time=time.time(),
        input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
        frozen_runtime_scope_sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'),
        state_output_binding_sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'),
        authorization_scope_sha256=sha(ROOT / 'AUTHORIZATION_SCOPE.json'),
        source_files={e['destination']: e['sha256'] for e in manifest['files']},
        package_files=manifest['package_files'], full_evaluation='PAUSED'))
    print(json.dumps(dict(status=scope['status'], groups=28, model_conditions=56)))


if __name__ == '__main__':
    main()
