"""Bind exact generated/independently verified bytes; no state deserialization."""
from pathlib import Path
import argparse
import json
import time
from row_factory import physical_row, read, sha, proposal

ROOT = Path(__file__).resolve().parent


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bind(generated, verification, manifest):
    generated, verification, manifest = Path(generated).resolve(), Path(verification).resolve(), Path(manifest).resolve()
    assert generated != verification
    frozen_path = verification / 'FROZEN_ROSTER.json'
    verified = read(frozen_path)
    proposed, design = proposal()
    assert verified['schema'] == 'fastrsi_object7_verified_development_roster_v1'
    assert verified['accepted'] is True and verified['overlap'] == 0
    assert verified['generation_read_only'] is True
    assert verified['task_design_sha256'] == proposed['source_proposal_sha256']
    assert verified['before_generation_freeze_sha256'] == sha(manifest)
    assert verified['manifest']['sha256'] == sha(manifest)
    assert len(verified['rows']) == len(verified['identities']) == 14
    assert [r['identity_id'] for r in verified['rows']] == [r['identity_id'] for r in design]
    assert len({r['physical_state_sha256'] for r in verified['rows']}) == 14
    assert sha(generated / 'ROSTER_RAW.json') == verified['raw_roster_sha256']
    assert sha(generated / 'GENERATION_COST.json') == verified['generation_cost']['sha256']
    files = {}
    for name in ('ROSTER_RAW.json', 'GENERATION_COST.json', 'GENERATION_EVENTS.jsonl',
                 'OFFICIAL_RESET_CALLS.jsonl', 'CONSTRUCTORS.jsonl'):
        files[name] = sha(generated / name)
    for row, identity in zip(verified['rows'], verified['identities']):
        physical_row(row)
        for key in ('identity_id', 'initial_state_sha256', 'physical_state_sha256',
                    'asset_sha256', 'source_receipt_sha256'):
            assert row[key] == identity[key]
        for name, digest in ((row['state_path'], row['initial_state_sha256']),
                             (row['settled_state_path'], row['settled_state_sha256']),
                             (row['source_receipt'], row['source_receipt_sha256'])):
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(generated / relative) == digest
            files[str(relative)] = digest
    declaration = dict(status='FOURTEEN_INDEPENDENT_VERIFIED_STATES_DECLARED',
                       task_design_sha256=verified['task_design_sha256'],
                       before_generation_freeze_sha256=sha(manifest),
                       expected_frozen_roster_sha256=sha(frozen_path),
                       expected_raw_roster_sha256=verified['raw_roster_sha256'],
                       generation_manifest_path=str(manifest),
                       generation_manifest_sha256=sha(manifest))
    binding = dict(status='INDEPENDENT_FROZEN_ROSTER_BOUND_FOR_RUNTIME', time=time.time(),
                   generated_host_root=str(generated), verification_host_root=str(verification),
                   verification_files={'FROZEN_ROSTER.json': sha(frozen_path)},
                   frozen_roster_host_path=str(frozen_path), frozen_roster_sha256=sha(frozen_path),
                   files=files, source_output_modified=False,
                   new_states_generated_by_this_helper=0, model_or_environment_calls=0,
                   full_evaluation='PAUSED')
    save(ROOT / 'STATE_PREPARATION_BINDING.json', declaration)
    save(ROOT / 'STATE_OUTPUT_BINDING.json', binding)
    return binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-root', required=True, type=Path)
    parser.add_argument('--verification-root', required=True, type=Path)
    parser.add_argument('--generation-manifest', required=True, type=Path)
    args = parser.parse_args()
    value = bind(args.generated_root, args.verification_root, args.generation_manifest)
    print(json.dumps(dict(status=value['status'], state_identity_count=14,
                          sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'))))


if __name__ == '__main__':
    main()
