"""Audit saved decision-time signals; no simulated outcomes or policy fitting."""
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastrsi.feedback_window import transition_diagnostics, prediction_revision_diagnostics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision(previous, current):
    """Align already predicted gripper signs by absolute action index."""
    if previous is None:
        return {'available': False, 'overlap': 0, 'changed': None, 'changed_in_next_five': None}
    offset = current['step'] - previous['step']
    old = previous['predicted_gripper_positive'][offset:]
    new = current['predicted_gripper_positive'][:len(old)]
    changes = [a != b for a, b in zip(old, new)]
    return {'available': bool(changes), 'overlap': len(changes),
            'changed': sum(changes) if changes else None,
            'changed_in_next_five': sum(changes[:5]) if changes else None}


def main():
    rows, exclusions, sources, paired = [], [], [], []
    decisions, traces = {}, {}
    for run in sorted((ROOT / 'runs').glob('*window*')):
        manifest = json.loads((run / 'DELIVERY_MANIFEST.json').read_text())['files']

        def read(path):
            relative = str(path.relative_to(run))
            sha = digest(path)
            assert sha == manifest[relative]['sha256'], str(path)
            sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha})
            return path.read_text()

        for path in sorted(run.glob('pair*/*/stream/OS_COST_EVENTS.jsonl')):
            events = list(map(json.loads, read(path).splitlines()))
            choices = [e for e in events if e.get('event') == 'feedback_window_choice']
            result_path = path.parent / 'RESULT.json'
            if not result_path.exists() or not choices:
                exclusions.append({'source': str(path.relative_to(ROOT)),
                                   'reason': 'no completed result' if not result_path.exists()
                                   else 'no saved full predicted gripper sequence'})
                continue
            result = json.loads(read(result_path))['cases'][0]
            case = result['case']
            trace_path, = path.parent.glob('stage_trace/*/TRACE.jsonl')
            trace = list(map(json.loads, read(trace_path).splitlines()))
            chunks = [t for t in trace if t['method'] == 'action_chunk']
            assert len(chunks) == len(choices)
            direction, count, previous, records = None, 0, None, []
            executed_actions = []
            for index, (event, chunk) in enumerate(zip(choices, chunks)):
                executed = chunk['step'] - event['step']
                assert 0 < executed <= event['submitted_steps']
                assert executed == event['submitted_steps'] or (index == len(choices)-1 and result['environment_success'])
                assert len(chunk['actions']) == event['submitted_steps']
                executed_actions.extend(chunk['actions'][:executed])
                assert [a[6] > 0 for a in chunk['actions']] == event['predicted_gripper_positive'][:event['submitted_steps']]
                assert previous is None or event['step'] == previous['step'] + previous['submitted_steps']
                d, direction, count = transition_diagnostics(
                    event['predicted_gripper_positive'], event['previous_committed_gripper_positive'],
                    event['submitted_steps'], direction, count)
                rev = revision(previous, event)
                live = prediction_revision_diagnostics(event['predicted_gripper_positive'], event['step'],
                    previous['predicted_gripper_positive'] if previous else None,
                    previous['step'] if previous else None)
                assert (rev['available'], rev['overlap'], rev['changed']) == (
                    live['available'], live['overlap_steps'], live['changed_steps'])
                records.append({'step': event['step'], 'submitted_steps': event['submitted_steps'],
                    'executed_steps_analysis_only': executed, 'before_choice': {
                    k: d[k] for k in ('first_predicted_transition_index', 'transition_direction',
                                      'same_direction_deferrals_before')},
                    'prediction_revision_before_choice': rev,
                    'after_actual_commit': {k: d[k] for k in
                        ('predicted_transition_outside_committed_prefix', 'consecutive_deferrals_after')}})
                previous = event
            key = str(path.parent.relative_to(ROOT))
            decisions[key] = choices
            traces[key] = trace
            rows.append({'key': key, 'run': run.name, 'pair': path.parts[-4], 'arm': path.parts[-3],
                'task': case['task_id'], 'init': case['init_state_id'],
                'mode': choices[0]['mode'], 'calls': len(choices),
                'steps': chunks[-1]['step'], 'native_success_analysis_only': result['environment_success'],
                'verification_status_analysis_only': result['verification_status'],
                'chain_status': result['chain_status'],
                'max_deferrals_after': max(r['after_actual_commit']['consecutive_deferrals_after'] for r in records),
                'max_deferrals_before': max(r['before_choice']['same_direction_deferrals_before'] for r in records),
                'revised_calls': sum((r['prediction_revision_before_choice']['changed'] or 0) > 0 for r in records),
                'overlap_calls': sum(r['prediction_revision_before_choice']['overlap'] > 0 for r in records),
                'behavior_sha256': hashlib.sha256(json.dumps(executed_actions).encode()).hexdigest(),
                'decisions': records})

    # Only compare until first actual submission divergence. Later matching step IDs
    # do not imply matching states. Native results are episode labels, not local labels.
    for a, b in itertools.combinations(rows, 2):
        if (a['run'], a['pair']) != (b['run'], b['pair']):
            continue
        first = None
        for index, (ea, eb) in enumerate(zip(decisions[a['key']], decisions[b['key']])):
            ta, tb = traces[a['key']][index], traces[b['key']][index]
            same_observation = ta['images'] == tb['images'] and ta['step'] == tb['step']
            ca, cb = traces[a['key']][index+1], traces[b['key']][index+1]
            if not same_observation or ca['actions'] != cb['actions']:
                first = {'step_a': ea['step'], 'step_b': eb['step'],
                         'same_saved_images': same_observation,
                         'same_predicted_gripper_signs': ea['predicted_gripper_positive'] == eb['predicted_gripper_positive'],
                         'submitted_a': ea['submitted_steps'], 'submitted_b': eb['submitted_steps'],
                         'before_a': a['decisions'][index]['before_choice'],
                         'before_b': b['decisions'][index]['before_choice'],
                         'numeric_robot_state_equality': 'not recorded; not established'}
                break
        paired.append({'a': a['key'], 'b': b['key'], 'first_divergence': first,
                       'native_success_a': a['native_success_analysis_only'],
                       'native_success_b': b['native_success_analysis_only']})
    report = {'scope': 'Existing development trajectories only; no training, held-out test, or new robot run',
              'source_integrity': 'all read sources matched delivery SHA256',
              'episodes': len(rows), 'unique_task_initial_conditions': len({(r['task'], r['init']) for r in rows}),
              'unique_executed_behaviors': len({r['behavior_sha256'] for r in rows}),
              'rows': rows, 'paired_first_divergence': paired, 'exclusions': exclusions, 'sources': sources,
              'limitations': ['Decisions in an episode are correlated, not independent samples.',
                'Deferral is caused partly by the chosen prefix; fixed10 cannot defer a transition within ten predicted steps.',
                'Prediction overlap exists only after short submission; absence after fixed10 is structural missingness, not stability.',
                'Episode outcomes do not label the causal benefit of each local short-window decision.',
                'After branch divergence no counterfactual outcome is inferred.',
                'No thresholds fitted, no held-out claims, no cross-protocol latency pooling.']}
    out = ROOT / 'planning/window_signal_audit_v1_20261003'
    out.mkdir(exist_ok=True)
    (out / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('episodes', 'unique_task_initial_conditions', 'unique_executed_behaviors', 'source_integrity')}))
    for r in rows:
        print(r['task'], r['init'], r['arm'], r['mode'], r['max_deferrals_after'], r['revised_calls'], r['overlap_calls'], r['native_success_analysis_only'])


if __name__ == '__main__':
    main()
