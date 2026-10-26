"""Summarize exact replay without granting simulator evidence online authority."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = dict(status='ANALYZED', scope='Three execution settings, one seen development initial state; exact CPU replay only.',
        arms={}, online_label_permission=False, skill_updates=0, model_updates=0)
    for arm in ('N', 'F', 'R'):
        source = json.loads((args.run / arm / 'REPLAY.json').read_text())
        assert source['status'] == 'EXACT_IMAGE_REPLAY'
        rows = source['rows']
        report['arms'][arm] = dict(observations=len(rows), steps=rows[-1]['step'],
            first_inside_observed=next(r['step'] for r in rows if r['native_inside']),
            inside_while_grasped_observations=[r['step'] for r in rows if r['native_inside'] and r['native_grasp']],
            inside_while_contacting_observations=[r['step'] for r in rows if r['native_inside'] and r['gripper_contacts_bowl']],
            first_inside_without_gripper_contact_observed=next(r['step'] for r in rows if r['native_inside'] and not r['gripper_contacts_bowl']),
            terminal_success=rows[-1]['overall_success'], replay_seconds=source['seconds'])
    report['contact_semantics'] = 'In calls contact AND containment, but the region SiteObjectState check_contact unconditionally returns True. Field cabinet_contacts_bowl in raw v1 means region method return, NOT physical cabinet contact. Gripper contact and grasp are separate simulator checks. Preserve raw output and source definitions.'
    report['decision'] = 'Reject inside-alone closing trigger. Native closing exists and succeeds in all three traces; pause this candidate expansion. Keep data for progress-memory invalidation checks; do not turn privileged labels into online features.'
    report['cost'] = dict(parallel_wall_seconds=json.loads((args.run / 'STATUS.json').read_text())['seconds'],
        sum_replay_seconds=sum(v['replay_seconds'] for v in report['arms'].values()), new_policy_rollouts=0, model_calls=0)
    (args.run / 'ANALYSIS.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
