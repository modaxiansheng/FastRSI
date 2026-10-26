"""Bind actual feedback-only execution and all three-arm development costs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import struct


def read(p): return json.loads(p.read_text())
def rows(p): return [json.loads(x) for x in p.read_text().splitlines()]
def packed(a):
    values = [v for row in a for v in row]
    return struct.pack('<%df' % len(values), *values)


def main(run, out):
    manifest = read(run/'RAW_MANIFEST.json')['files']
    for r in manifest:
        p = run/r['path']
        assert p.stat().st_size == r['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r['sha256'], p
    state = read(run/'STATUS.json')
    assert state['status'] == 'COMPLETED'
    stops = read(run/'FRESH_CLEANUP_CHECK.json')['containers']
    assert len(stops) == 12 and all(not c['state']['Running'] and not c['state']['OOMKilled'] for c in stops)
    arms, traces, inputs, sources = {}, {}, {}, {}
    for arm in 'NFR':
        stream = run/arm/'stream'
        report = read(stream/'RESULT.json')['cases'][0]
        case = report['case']
        assert case['mode'] == 'development' and case['init_state_id'] == 31
        assert case['max_steps'] == 520
        events = rows(stream/'OS_COST_EVENTS.jsonl')
        windows = [e for e in events if e['event'] == 'feedback_window_choice']
        actual_inputs = [e for e in events if e['event'] == 'actual_policy_input']
        assert len(actual_inputs) == len(windows)
        assert all(e['prompt'] == case['task_description'] for e in actual_inputs)
        ts = {p.parent.name:rows(p) for p in (stream/'stage_trace').glob('*/TRACE.jsonl')}
        # This registered check cannot silently become a reset/retry study.
        assert len(ts) == 1 and case['id'] in ts
        trace = ts[case['id']]
        assert trace[0]['method'] == 'reset'
        assert len(trace) == len(windows) + 1
        for i, e in enumerate(windows):
            submitted = trace[i+1]
            assert submitted['method'] == 'action_chunk'
            assert e['step'] == trace[i]['step']
            assert e['original_steps'] == 10
            assert 0 < submitted['step'] - trace[i]['step'] <= e['submitted_steps']
            assert packed(submitted['actions']) == packed(e['predicted_actions'][:e['submitted_steps']])
        receipt = read(stream/'STALL_WINDOW_RECEIPT.json') if arm != 'F' else None
        if receipt:
            assert receipt['completed'] and receipt['replaced_motion'] is False
            assert receipt['learned'] is False and receipt['original_actions_and_sampling']
            assert receipt['observation_count'] == len(actual_inputs)
        trigger = receipt['trigger_step'] if receipt else None
        assert all(e['requested_steps'] == (5 if arm == 'F' or (trigger is not None and e['step'] >= trigger) else 10)
                   for e in windows)
        calls = [read(p) for p in sorted((stream/'service_calls').glob('call_*/RECEIPT.json'))]
        ends = [e for e in events if e['event'] == 'end']
        stage = {}
        for e in ends: stage[e['name']] = stage.get(e['name'], 0.) + e['duration_seconds']
        lifecycle = read(stream/'STOP_RECEIPT.json')
        card_seconds = sum((datetime.datetime.fromisoformat(s['finished'].replace('Z','+00:00')) -
                            datetime.datetime.fromisoformat(s['started'].replace('Z','+00:00'))).total_seconds()
                           for s in lifecycle if s['role'] in ('pi05','qwen'))
        worker = read(run/arm/'STATUS.json')
        arms[arm] = dict(environment_success=report['environment_success'],
            verification_status=report['verification_status'], chain_status=report['chain_status'],
            admission_status=report.get('experience_admission',{}).get('status'),
            episodes=len(ts), physical_steps=trace[-1]['step'], policy_calls=len(actual_inputs),
            action_submissions=len(windows), short_submissions=sum(e['requested_steps']==5 for e in windows),
            trigger_step=trigger, qwen_calls=len(calls), qwen_generation_seconds=sum(c['generation_seconds'] for c in calls),
            qwen_prompt_tokens=sum(c['prompt_tokens'] for c in calls),
            qwen_completion_tokens=sum(c['completion_tokens'] for c in calls),
            stage_seconds=stage, startup_seconds=worker['startup_seconds'],
            controller_seconds=worker['controller_seconds'], worker_seconds=worker['full_wall_seconds'],
            allocated_model_card_seconds=card_seconds, action_prefix_only=True,
            detector_time_separately_measured=False)
        traces[arm] = trace
        inputs[arm] = actual_inputs
        sources[arm] = report['accepted_source']
    trigger = arms['R']['trigger_step']
    assert trigger is not None, 'No actual recovery operation was tested'
    strip = lambda t:[{k:v for k,v in r.items() if k != 'session_id'} for r in t if r['step'] <= trigger]
    assert strip(traces['N']) == strip(traces['R'])
    count = trigger//10+1  # Include current observation before the first short commitment.
    assert [i['array_bindings'] for i in inputs['N'][:count]] == [i['array_bindings'] for i in inputs['R'][:count]]
    assert sources['N'] == sources['F'] == sources['R']
    result = dict(status='VERIFIED_DEVELOPMENT_OPERATION', arms=arms,
        pre_intervention_steps_equal=trigger, intervention_input_equal=True,
        same_initial_experience=True, actual_source_actions_unchanged=True,
        verified_raw_files=len(manifest), owned_containers_stopped=12,
        parent_seconds=state['seconds'],
        sum_worker_seconds=sum(a['worker_seconds'] for a in arms.values()),
        allocated_model_card_seconds=sum(a['allocated_model_card_seconds'] for a in arms.values()),
        learned=False, unseen_evaluation=False, default_changed=False,
        economic_goal_achieved=False,
        timing_note='Stages are nested; worker, parent throughput and allocated card time are separate accounts. Trigger CPU time is included in worker but not measured separately.')
    out.mkdir(parents=True, exist_ok=False)
    (out/'SUMMARY.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run,a.out)
