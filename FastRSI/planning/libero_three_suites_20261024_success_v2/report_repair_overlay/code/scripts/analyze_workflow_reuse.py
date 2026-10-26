"""Summarize paired adoption evidence without promoting warm-start to learning gain."""
import argparse,json,hashlib
from collections import Counter,defaultdict
from pathlib import Path
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);args=ap.parse_args()
    run=ROOT/'runs'/args.run_id;stream=run/'stream';protocol=read(run/'PROTOCOL.json');result=read(stream/'RESULT.json',{})
    rows=[];events=lines(stream/'OS_COST_EVENTS.jsonl');starts={e['span_id']:e for e in events if e.get('event')=='begin'}
    def case_of(e):
        while e:
            if e.get('case'):return e['case']
            e=starts.get(e.get('parent_id'))
    spans=defaultdict(lambda:defaultdict(float))
    for e in events:
        if e.get('event')=='end':spans[case_of(e)][e['name']]+=e['duration_seconds']
    service=defaultdict(list)
    def key(r):return hashlib.sha256(json.dumps(r['messages'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    for p in stream.glob('service_calls/*/REQUEST.json'):service[key(read(p))].append(p.parent)
    target=lines(stream/'TARGET_COST_EVENTS.jsonl')
    for kind in ['planning_cases','full_cases']:
        for item in protocol[kind]:
            folder=stream/item['label'];r=read(folder/'RESULT.json')
            if not r:continue
            tools=lines(folder/'AGENT_EVENTS.jsonl');phases=defaultdict(lambda:defaultdict(float))
            for p in folder.glob('*_call_*/REQUEST.json'):
                req=read(p);phase='verifier' if p.parent.name.startswith('verifier') else 'planning'
                users=[m.get('content','') for m in req['messages'] if m['role']=='user']
                if phase=='planning' and users and 'Execution and its original verification have finished.' in str(users[-1]):phase='report'
                st=phases[phase];st['requests']+=1;bound=service[key(req)]
                if len(bound)!=1:raise ValueError('Missing or ambiguous raw model binding')
                receipt=read(bound[0]/'RECEIPT.json',{});st['rejected_or_incomplete']+=receipt.get('status')!='OK'
                for k in ['generation_seconds','prompt_tokens','completion_tokens']:st[k]+=receipt.get(k,0)
            obs=[e for e in target if e.get('event')=='native_observation' and str(e.get('session_id','')).startswith(item['label'])]
            rows.append({'kind':kind,'label':item['label'],'arm':item['arm'],'task':item['task'],'status':r.get('chain_status',r['status']),
                'seconds':r.get('seconds'),'environment_success':r.get('environment_success'),
                'verdicts':[o.get('verdict') for o in r.get('outcomes',[])],
                'attempts':len(r.get('outcomes',[])),'last_native_observation':obs[-1] if obs else None,
                'phases':{k:dict(v) for k,v in phases.items()},'stage_intervals':dict(spans[item['label']]),
                'tools':dict(Counter(e['name'] for e in tools if e.get('event')=='tool')),
                'workflow_expansions':sum(e.get('event')=='workflow_expanded' for e in tools),
                'agent_reply_complete':'maximum number of tool call iterations' not in r.get('reply',r.get('agent_response','')),
                'planned_goal_changed':r.get('planned_session',{}).get('task_description',item['task']['task_description'])!=item['task']['task_description'],
                'fixed_experience_unchanged':r.get('experience_unchanged',r.get('experience_before')==r.get('experience_after')),
                'error':r.get('error')})
    planning=[r for r in rows if r['kind']=='planning_cases'];full=[r for r in rows if r['kind']=='full_cases']
    groups={a:[r for r in full if r['arm']==a] for a in ['A','B']}
    comparison=None
    if all(len(groups[a])==1 for a in groups):
        a,b=groups['A'][0],groups['B'][0]
        comparison={'scope':'Single paired warm-start native evaluation; fixed B-first order; not statistical or cold-start result',
            'A_seconds':a['seconds'],'B_seconds':b['seconds'],'observed_savings_seconds':a['seconds']-b['seconds'],
            'observed_savings_fraction':1-b['seconds']/a['seconds'],'both_environment_success':a['environment_success'] is True and b['environment_success'] is True,
            'both_chain_passed':a['status']==b['status']=='PASSED'}
    old=read(ROOT/'runs/native_long_v1_20261002/stream/native_long_t0_i8/RESULT.json')
    bill=summarize(run);bill.pop('native_run',None)
    value={'run_id':args.run_id,'status':result.get('status',read(run/'STATUS.json')['status']),'rows':rows,
        'planning_pairs':[{'task_id':t,'A':next((r for r in planning if r['task']['task_id']==t and r['arm']=='A'),None),
            'B':next((r for r in planning if r['task']['task_id']==t and r['arm']=='B'),None)} for t in sorted({r['task']['task_id'] for r in planning})],
        'full_comparison':comparison,'bill':bill,'compilation':result.get('compilation'),
        'historical_source_episode_seconds':old['seconds'],
        'accounting':'Includes all new failed and successful requests, tests, service startup and shutdown in campaign bill. Historical source episode cost is disclosed separately; shared historical startup is not allocated. Local editing/transfer/human work unmeasured. Nested stages are not additive. No end-to-end payback claim.',
        'mechanism':'Fixed procedural extraction plus common batch-read tool; no policy learning, no native memory-trigger modification, no Laya training',
        'development_split':'All cases are development diagnostics using no-update evaluation machinery; these observed instances are not held-out final tests.',
        'limitations':['The native policy server seeds its RNG at startup and does not reset it between episodes; full A/B trajectories are not matched random draws. Fixed B-first order can affect timing.',
            'B procedure is an engineering-authored template populated from an admitted trace, not autonomous induction.',
            'Same prior sealed experience and same tools; B adds procedural skill. This isolates warm-start adoption, not cold-start acquisition.',
            'One full pair cannot establish 2pp noninferiority or 30% population benefit.']}
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({'status':value['status'],'full_comparison':comparison,'rows':[{k:r[k] for k in ['label','status','seconds','tools','environment_success','verdicts']} for r in rows],
        'bill':{k:bill[k] for k in ['full_wall_seconds','model_requests','model_rejected','incomplete_model_requests','hcu_allocation_seconds']}},indent=2))
if __name__=='__main__':main()
