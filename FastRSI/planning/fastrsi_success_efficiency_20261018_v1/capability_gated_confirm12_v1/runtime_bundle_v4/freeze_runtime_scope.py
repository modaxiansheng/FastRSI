"""Freeze all48 metadata conditions after independent state verification.

Consumes an explicit16-group placement document. Does not start subprocesses,
construct environments, infer, generate states, or inspect trial outcomes.
"""
from pathlib import Path
import argparse
import json
import time
from coordinator_contract import groups, read, sha, verified_states, group_rows, frozen_files

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--placement', required=True, type=Path)
    args = parser.parse_args()
    frozen_files(ROOT)
    states = verified_states(ROOT)
    placed = read(args.placement)['groups']
    expected = groups()
    assert len(placed) == len(expected) == 16
    assert {(r['model'], r['suite']) for r in placed} == \
        {(r['model'], r['suite']) for r in expected}
    assert len({(r['model'], r['suite']) for r in placed}) == 16
    conditions, group_docs = [], []
    for group in expected:
        placement = next(r for r in placed if r['model'] == group['model']
                         and r['suite'] == group['suite'])
        assert placement['node'] in ['187', '188'] and placement['gpu'] in range(8)
        rows = group_rows(states, group['model'], group['suite'],
                          placement['node'], placement['gpu'])
        conditions.extend(rows)
        group_docs.append(dict(group, node=placement['node'], gpu=placement['gpu'],
                               condition_ids=[r['identity_id'] for r in rows]))
    assert len(conditions) == len({r['identity_id'] for r in conditions}) == 48
    assert len({r['physical_identity_id'] for r in conditions}) == 12
    assert sum(r['skill'] is not None for r in conditions) == 20
    scope = dict(status='ALL48_RUNTIME_CONDITIONS_FROZEN_BEFORE_TRIAL', time=time.time(),
                 input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
                 state_output_binding_sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'),
                 proposal_sha256=sha(ROOT / 'CONFIRMATION_PROPOSAL.json'),
                 placement_sha256=sha(args.placement), groups=group_docs,
                 conditions=conditions, normal_physical_attempts_max=288,
                 additional_attempts_per_failed_condition_per_branch=1,
                 branches=['ordinary', 'frozen', 'candidate'],
                 primary_denominator=48, per_model_denominator=12,
                 unsupported_candidate_conditions=28, success_pooling=False,
                 compute_budget_matched=False, training=False,
                 full_evaluation='PAUSED', old_six_source_study='CLOSED5OF6_UNCHANGED',
                 execution_started=False)
    with (ROOT / 'FROZEN_RUNTIME_SCOPE.json').open('x') as stream:
        json.dump(scope, stream, indent=2)
        stream.write('\n')
    manifest = read(ROOT / 'BUILD_INPUT_MANIFEST.json')
    execution = dict(status='EXECUTION_BYTES_AND48_CONDITIONS_FROZEN', time=time.time(),
                     input_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
                     frozen_runtime_scope_sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'),
                     state_output_binding_sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'),
                     authorization_scope_sha256=sha(ROOT / 'AUTHORIZATION_SCOPE.json'),
                     source_files={e['destination']: e['sha256'] for e in manifest['files']},
                     package_files=manifest['package_files'], full_evaluation='PAUSED')
    with (ROOT / 'EXECUTION_BINDINGS.json').open('x') as stream:
        json.dump(execution, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=scope['status'], conditions=48,
                         sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'))))


if __name__ == '__main__':
    main()
