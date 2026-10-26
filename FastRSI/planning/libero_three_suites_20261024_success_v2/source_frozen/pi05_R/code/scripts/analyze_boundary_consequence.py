"""Real branch consequences and separate saved-request interpretation costs."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(run):
    status=read(run/'STATUS.json');assert status['status']=='COMPLETED'
    rows=[];traces={};sources={}
    for arm in ('N','F'):
        group=run/arm;stream=group/'stream';report=read(stream/'RESULT.json')
        assert report['status']=='COMPLETED' and len(report['cases'])==1
        case=report['cases'][0];sid=case['case']['id'];worker=read(group/'STATUS.json')
        guard=read(stream/f'BOUNDARY_{sid}.json')
        assert guard['status']=='PREFIX_MATCHED' and guard['verified_observations']==21 and guard['verified_action_chunks']==20
        assert guard['teacher_calls']==0 and guard['learning_label'] is None
        events=lines(stream/'OS_COST_EVENTS.jsonl');root=lines(stream/'stage_trace'/sid/'TRACE.jsonl');traces[arm]=root
        alltraces=[lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in case['sessions']]
        choices=[e for e in events if e['event']=='boundary_choice']
        inputs=[e for e in events if e['event']=='actual_policy_input']
        assert len(inputs)==sum(len(t)-1 for t in alltraces)
        spans=defaultdict(float)
        for e in events:
            if e['event']=='end':spans[e['name']]+=e['duration_seconds']
        boundary=next(e for e in choices if e['step']==200)
        end=next(e for e in events if e['event']=='end' and e['name']=='os_execution' and e.get('attempt')==0)
        rows.append(dict(arm=arm,environment_success=case['environment_success'],
            semantic=[o['verdict'] for o in case['outcomes']],chain=case['chain_status'],admission=case['experience_admission']['status'],
            attempts=len(alltraces),root_steps=root[-1]['step'],root_remaining_steps=root[-1]['step']-200,
            all_steps=sum(t[-1]['step'] for t in alltraces),vla_calls=len(inputs),
            root_vla_calls=len(root)-1,root_remaining_vla_calls=len(root)-1-20,
            qwen_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),
            agent_calls=len(list((stream/sid).glob('agent_call_*/REQUEST.json'))),
            changed_prompts=guard['changed_policy_prompts'],teacher_calls=0,
            case_seconds=case['seconds'],controller_seconds=worker['controller_seconds'],
            worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],
            root_continuation_wall_seconds=(end['wall_ns']-boundary['wall_ns'])/1e9,
            nested_stage_seconds=dict(spans)))
        for p in (stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/f'BOUNDARY_{sid}.json',stream/'stage_trace'/sid/'TRACE.jsonl'):
            sources[str(p.relative_to(run))]=sha(p)
    prefix={a:[r for r in t if r['step']<=200] for a,t in traces.items()}
    assert prefix['N']==prefix['F'] and len(prefix['N'])==21
    entity=read(run/'ENTITY/stream/RESULT.json');assert entity['status']=='COMPLETED' and len(entity['rows'])==6
    return dict(status='ANALYZED',rows=rows,paired_prefix=dict(boundaries=21,actions=200,images=42,exact_trace_rows_equal=True,
        scope='Observed RGB/state and regenerated submitted actions; no original complete simulator-state snapshot'),
        entity_diagnostic=entity,batch_wall_seconds=status['seconds'],entity_startup_seconds=status['entity_startup_seconds'],
        entity_full_wall_seconds=status['entity_full_wall_seconds'],source_sha256=sources,
        scope='Post-hoc intervention in one previously seen development initial state; not learned online timing or transfer.',
        cost_scope='Both full prefixes and checks are executed and charged; reference loading is included in controller time. '
                   'Nested spans do not sum; the separate entity diagnostic is project validation expense, not method inference.',
        trained=False,learning_gain_established=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    result=analyze(args.run)
    with args.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='entity_diagnostic'},indent=2))
