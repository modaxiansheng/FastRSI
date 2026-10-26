"""Pin already-closed first-stage outputs; never infer or dispatch another model."""
from pathlib import Path
import json
from coordinator_contract import read, sha, verify_pi05_stage

ROOT = Path(__file__).resolve().parent


def main():
    scope = read(ROOT / 'FROZEN_RUNTIME_SCOPE.json')
    records, count = [], 0
    for group in [g for g in scope['groups'] if g['model'] == 'pi05']:
        lane = ROOT / 'campaigns/libero_object__pi05' / ('task' + str(group['task_id'])) / ('node' + group['node']) / ('gpu' + str(group['gpu']))
        row = dict(task_id=group['task_id'], files={}, conditions={})
        for name in ('GROUP_RESULT.json', 'OWNED_STOP_RECEIPT.json', 'LIFECYCLE.json'):
            row['files'][name] = sha(lane / name)
        count += read(lane / 'GROUP_RESULT.json')['normal_physical_attempts']
        for condition in group['condition_ids']:
            row['conditions'][condition] = sha(lane / condition / 'CONDITION_RESULT.json')
        records.append(row)
    marker = dict(schema='fastrsi_object7_pi05_stage_complete_v1',
                  frozen_runtime_scope_sha256=sha(ROOT / 'FROZEN_RUNTIME_SCOPE.json'),
                  groups=records, normal_physical_attempts_completed=count,
                  unsupported_candidate_is_coverage_gap=True,
                  full_evaluation='PAUSED')
    # Validate actual existing outputs before publishing the exclusive phase marker.
    verify_pi05_stage(ROOT, marker)
    with (ROOT / 'PI05_STAGE_COMPLETE.json').open('x') as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status='PI05_SEVEN_GROUPS_CLOSED_NO_NEW_LAUNCH', normal_attempts=count)))


if __name__ == '__main__':
    main()
