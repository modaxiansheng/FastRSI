"""Bind the separate independently verified CPU state output; no generation.

The exact generated directory is supplied explicitly after verification. This
does not launch, read policy outcomes, change a state, or edit the source output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
from row_factory import physical_row

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-root', required=True, type=Path)
    parser.add_argument('--verification-root', required=True, type=Path)
    args = parser.parse_args()
    state_root = args.generated_root.resolve()
    verification_root = args.verification_root.resolve()
    assert verification_root != state_root
    declaration = json.loads((ROOT / 'STATE_PREPARATION_BINDING.json').read_text())
    path = verification_root / 'FROZEN_ROSTER.json'
    assert sha(path) == declaration['expected_frozen_roster_sha256']
    verified = json.loads(path.read_text())
    assert verified['status'] == '12_STATE_ONLY_EXCLUSIONS_EXACTLY_VERIFIED'
    assert verified['accepted'] is True and verified['same_states_for_all_models'] is True
    assert verified['generation_read_only'] is True
    assert verified['task_design_sha256'] == declaration['task_design_sha256']
    assert verified['before_generation_freeze_sha256'] == declaration['before_generation_freeze_sha256']
    assert verified['overlap'] == 0
    assert set(verified['models']) == {'pi0', 'pi05', 'openvla', 'xvla'}
    assert len(verified['rows']) == len(verified['identities']) == 12
    assert sha(state_root / 'ROSTER_RAW.json') == verified['raw_roster_sha256']
    assert verified['raw_roster_sha256'] == declaration['expected_raw_roster_sha256']
    files = {'ROSTER_RAW.json': verified['raw_roster_sha256']}
    for name, key in [('GENERATION_COST.json', 'generation_cost_sha256'),
                      ('GENERATION_EVENTS.jsonl', 'generation_events_sha256'),
                      ('OFFICIAL_RESET_CALLS.jsonl', 'generation_reset_journal_sha256')]:
        assert sha(state_root / name) == verified[key]
        files[name] = verified[key]
    for row, identity in zip(verified['rows'], verified['identities']):
        physical_row(row)
        assert identity['identity_id'] == row['identity_id']
        for key in ('initial_state_sha256', 'physical_state_sha256', 'asset_sha256',
                    'source_receipt_sha256'):
            assert identity[key] == row[key]
        for name, digest in [(row['state_path'], row['initial_state_sha256']),
                             (row['source_receipt'], row['source_receipt_sha256'])]:
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            assert sha(state_root / relative) == digest
            files[name] = digest
    result = dict(status='INDEPENDENT_FROZEN_ROSTER_BOUND_FOR_RUNTIME', time=time.time(),
                  generated_host_root=str(state_root),
                  verification_host_root=str(verification_root),
                  verification_files={'FROZEN_ROSTER.json': sha(path)},
                  frozen_roster_host_path=str(path), frozen_roster_sha256=sha(path),
                  files=files, source_output_modified=False,
                  new_states_generated_by_this_helper=0,
                  model_or_environment_calls=0, full_evaluation='PAUSED')
    with (ROOT / 'STATE_OUTPUT_BINDING.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(status=result['status'], sha256=sha(ROOT / 'STATE_OUTPUT_BINDING.json'))))


if __name__ == '__main__':
    main()
