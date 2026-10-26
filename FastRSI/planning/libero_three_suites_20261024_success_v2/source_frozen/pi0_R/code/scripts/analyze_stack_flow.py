"""Report actual task outcomes and complete costs; never compare unlike goals."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]


GOALS = {
    63: 'stack the left bowl on the right bowl and place them in the tray',
    64: 'stack the right bowl on the left bowl and place them in the tray',
}


def validate_case_binding(task_id, binding, plan, roster, row):
    """Reject a result assigned to the wrong task directory before aggregation."""
    expected = dict(suite='libero_90', task_id=task_id, init_state_id=0,
                    task_description=GOALS[task_id], mode='development', max_steps=400,
                    native_experience_maintenance=True,
                    development_suite_extension='libero90_stack_flow_v1')
    assert binding['task_id'] == task_id and binding['suite'] == 'libero_90', 'Static task identity'
    assert binding['task_order_index'] == 0 and binding['init_state_id'] == 0, 'Static initial state'
    assert binding['language'] == GOALS[task_id], 'Static task language'
    assert len(plan['cases']) == len(roster) == 1, 'One root per worker'
    for record in (plan['cases'][0], roster[0], row['case']):
        for key, value in expected.items():
            assert type(record.get(key)) is type(value) and record[key] == value, 'Case binding: ' + key
    assert row['case'] == roster[0], 'Executed case differs from dispatched ROSTER'
    assert row['case']['id'] == f'net_t{task_id}_i0', 'Root identity'
    assert row['case']['feedback_window_mode'] == 'fixed10', 'Execution window'
    assert row['case']['planning_entry'] == 'public_session_flow', 'Planning entry'


def summarize_attempts(row, traces, windows, target_events):
    """Keep native facts, semantic outcomes and reset retries separately visible."""
    case = row['case']
    sessions = sorted(row['sessions'], key=lambda s: s['replan_attempt'])
    assert 1 <= len(sessions) <= 3, 'Native attempt budget'
    assert [s['replan_attempt'] for s in sessions] == list(range(len(sessions))), 'Attempt sequence'
    ids = [s['session_id'] for s in sessions]
    assert len(set(ids)) == len(ids) and set(ids) == set(traces), 'Attempt trace identities'
    outcomes = {o['session_id']: o for o in row['outcomes']}
    assert len(outcomes) == len(row['outcomes']) and set(outcomes) == set(ids), 'Verifier outcomes'
    assert set(row['native_terminal_by_session']) == set(ids), 'Native terminal identities'
    assert all(e['runtime_session_id'] in ids for e in windows), 'Window event ownership'
    result = []
    for index, session in enumerate(sessions):
        sid = session['session_id']; trace = traces[sid]; outcome = outcomes[sid]
        benchmark = session['benchmark']; execution = session['execution']
        assert (benchmark['benchmark_id'], benchmark['suite_id'], benchmark['task_index'],
                benchmark['instance_id'], benchmark['execution_mode']) == (
                    'libero', case['suite'], case['task_id'], case['init_state_id'], 'policy_loop'), 'Session task binding'
        assert execution['max_steps'] == 400 and execution['reset_policy'] == 'session_runner', 'Attempt execution contract'
        if index == 0:
            assert sid == case['id'] and session['task_description'] == case['task_description'], 'Original whole goal'
            assert not session.get('parent_session_id'), 'Unexpected root parent'
        else:
            previous = sessions[index - 1]
            assert session['parent_session_id'] == previous['session_id'], 'Retry parent'
            assert outcomes[previous['session_id']].get('replanned_session_id') == sid, 'Applied retry lineage'
        observations = [e for e in target_events if e.get('event') == 'native_observation' and e.get('session_id') == sid]
        assert observations and sum(e['method'] == 'reset' for e in observations) == 1, 'Actual physical reset'
        terminal = observations[-1]
        assert terminal == row['native_terminal_by_session'][sid], 'Native terminal changed'
        assert trace[0]['method'] == 'reset' and trace[0]['step'] == 0, 'Trace starts at reset'
        assert all(t['session_id'] == sid for t in trace), 'Trace session identity'
        assert trace[-1]['step'] == terminal['steps'] == session['result']['num_steps'], 'Attempt step accounting'
        assert 0 <= terminal['steps'] <= 400, 'Attempt action budget'
        assert type(terminal['native_success']) is bool and type(terminal['runtime_success']) is bool
        assert trace[-1]['native_success'] == terminal['runtime_success'], 'Trace runtime result'
        episode = session['result']['metadata']['final_status']['episode_summary']
        assert (episode['benchmark_name'], episode['task_id'], episode['init_state_id'], episode['task_description']) == (
            case['suite'], case['task_id'], case['init_state_id'], case['task_description']), 'Native task language/identity'
        own_windows = [e for e in windows if e['runtime_session_id'] == sid]
        assert len(own_windows) == len(trace) - 1, 'Attempt VLA accounting'
        assert all(e['native_attempt'] == index and e['native_attempt_goal'] == session['task_description'] for e in own_windows), 'Attempt policy binding'
        result.append(dict(attempt=index, session_id=sid, parent_session_id=session.get('parent_session_id'),
            task_description=session['task_description'], environment_success=terminal['native_success'],
            runtime_success=terminal['runtime_success'], semantic_verdict=outcome.get('verdict'),
            verification_code=outcome.get('code'), session_status=session['status'],
            physical_resets=1, steps=terminal['steps'], vla_calls=len(own_windows)))
    assert row['environment_success'] == result[-1]['environment_success'], 'Final native outcome'
    return result


def analyze(run):
    status=read(run/'STATUS.json');assert status['status']=='COMPLETED'
    bindings=read(run/'TASKS.json')
    assert len(bindings['tasks']) == 2 and {b['task_id'] for b in bindings['tasks']} == set(GOALS)
    task_bindings={b['task_id']: b for b in bindings['tasks']}
    rows=[];sources={}
    for task in ('task63','task64'):
        group=run/task/'N';stream=group/'stream';result=read(stream/'RESULT.json')
        assert result['status']=='COMPLETED' and len(result['cases'])==1
        row=result['cases'][0];case=row['case'];events=lines(stream/'OS_COST_EVENTS.jsonl')
        validate_case_binding(int(task[4:]), task_bindings[int(task[4:])], read(run/task/'PLAN.json'), read(group/'ROSTER.json'), row)
        assert row['chain_status']=='PASSED'
        windows=[e for e in events if e['event']=='feedback_window_choice']
        assert windows and all(e['mode']=='fixed10' for e in windows)
        # Native retries, if any, retain distinct traces and all their costs.
        traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in row['sessions']}
        assert sum(len(t)-1 for t in traces.values())==len(windows)
        attempts=summarize_attempts(row,traces,windows,lines(stream/'TARGET_COST_EVENTS.jsonl'))
        spans=defaultdict(float)
        for e in events:
            if e['event']=='end':spans[e['name']]+=e['duration_seconds']
        agent_calls=list((stream/case['id']).glob('agent_call_*/RESPONSE.json'))
        service_calls=list((stream/'service_calls').glob('*/REQUEST.json'))
        verifier_calls=0
        for p in service_calls:
            request=read(p)
            if str(request.get('messages',[{}])[0].get('content','')).startswith('Verify the Agent session'):verifier_calls+=1
        inheritance=read(stream/case['id']/'INHERITANCE.json')
        assert inheritance['preserved_before_agent']
        worker=read(group/'STATUS.json')
        rows.append(dict(task=case['task_id'],init=case['init_state_id'],goal=case['task_description'],
            environment_success=row['environment_success'],semantic_verdicts=[o.get('verdict') for o in row['outcomes']],
            first_attempt_environment_success=attempts[0]['environment_success'],
            final_environment_success=attempts[-1]['environment_success'],attempts=attempts,
            chain_status=row['chain_status'],admission=row['experience_admission']['status'],
            resets=sum(a['physical_resets'] for a in attempts),steps=sum(a['steps'] for a in attempts),vla_calls=len(windows),
            agent_calls=len(agent_calls),verifier_calls=verifier_calls,all_qwen_requests=len(service_calls),
            case_seconds=row['seconds'],worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],
            nested_stage_seconds=dict(spans),initial_experience_seal=inheritance['source_seal_sha256']))
        for p in [run/'TASKS.json',run/task/'PLAN.json',group/'ROSTER.json',group/'STATUS.json',
                  stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/'TARGET_COST_EVENTS.jsonl',
                  stream/case['id']/'INHERITANCE.json',
                  *[stream/'stage_trace'/sid/'TRACE.jsonl' for sid in traces]]:
            sources[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len({r['initial_experience_seal'] for r in rows})==1
    return dict(status='ANALYZED',rows=rows,batch_wall_seconds=status['seconds'],
        interpretation='Opposite stacking roles, NOT comparative method arms. Parallelism is throughput, not method speedup. Stage durations are nested.',
        learning_net_gain_established=False,new_policy_training=False,source_sha256=sources)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    result=analyze(a.run)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
