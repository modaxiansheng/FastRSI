"""Report actual task outcomes and complete costs; never compare unlike goals."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(s) for s in path.read_text().splitlines()]


def analyze(run):
    status=read(run/'STATUS.json');assert status['status']=='COMPLETED'
    rows=[];sources={}
    for task in ('task1','task6'):
        group=run/task/'N';stream=group/'stream';result=read(stream/'RESULT.json')
        assert result['status']=='COMPLETED' and len(result['cases'])==1
        row=result['cases'][0];case=row['case'];events=lines(stream/'OS_COST_EVENTS.jsonl')
        assert row['chain_status']=='PASSED'
        windows=[e for e in events if e['event']=='feedback_window_choice']
        assert windows and all(e['mode']=='fixed10' for e in windows)
        # Native retries, if any, retain distinct traces and all their costs.
        traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in row['sessions']}
        assert sum(len(t)-1 for t in traces.values())==len(windows)
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
            chain_status=row['chain_status'],admission=row['experience_admission']['status'],
            resets=len(traces),steps=sum(t[-1]['step'] for t in traces.values()),vla_calls=len(windows),
            agent_calls=len(agent_calls),verifier_calls=verifier_calls,all_qwen_requests=len(service_calls),
            case_seconds=row['seconds'],worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],
            nested_stage_seconds=dict(spans),initial_experience_seal=inheritance['source_seal_sha256']))
        for p in [group/'STATUS.json',stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl']:
            sources[str(p.relative_to(run))]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len({r['initial_experience_seal'] for r in rows})==1
    return dict(status='ANALYZED',rows=rows,batch_wall_seconds=status['seconds'],
        interpretation='Different task families, NOT comparative arms. Parallelism is throughput, not method speedup. Stage durations are nested.',
        learning_net_gain_established=False,new_policy_training=False,source_sha256=sources)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args()
    result=analyze(a.run)
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
