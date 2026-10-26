"""Fresh-seed pair, outcome alignment and complete development bill."""
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);args=ap.parse_args()
    run=ROOT/'runs'/args.run_id;protocol=read(run/'PROTOCOL.json');cases=[];bills=[]
    for arm in ['A','B']:
        group=run/arm;stream=group/'stream';row=read(stream/'RESULT.json')['cases'][0];folder=stream/row['case']['id'];w=folder/'workspace'
        stage=defaultdict(float)
        for e in lines(stream/'OS_COST_EVENTS.jsonl'):
            if e.get('event')=='end':stage[e['name']]+=e['duration_seconds']
        phases=defaultdict(lambda:defaultdict(int))
        for p in folder.glob('*_call_*/REQUEST.json'):
            req=read(p);response=read(p.parent/'RESPONSE.json');phase='verifier' if p.parent.name.startswith('verifier') else 'planning'
            users=[m.get('content','') for m in req['messages'] if m['role']=='user']
            if phase=='planning' and users and 'Execution and its original verification have finished.' in str(users[-1]):phase='report'
            phases[phase]['calls']+=1
            for key in ['prompt_tokens','completion_tokens']:phases[phase][key]+=response.get('usage',{}).get(key,0)
        events=lines(folder/'AGENT_EVENTS.jsonl');target_events=lines(stream/'TARGET_COST_EVENTS.jsonl');obs=[e for e in target_events if e.get('event')=='native_observation']
        target_spans=defaultdict(float)
        for e in target_events:
            if e.get('event')=='end':target_spans[e['name']]+=e['duration_seconds']
        images={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (w/'artifacts/runtime'/row['case']['id']/'rgb').glob('*.png')}
        argv=read(stream/'LAUNCH_pi05.json')['argv'];assert '--torch-seed 7' in argv[-1]
        bill=summarize(group);bill.pop('native_run',None);bills.append(bill)
        cases.append({'arm':arm,'seconds':row['seconds'],'controller_seconds':read(stream/'RESULT.json')['seconds'],'environment_success':row['environment_success'],'chain_status':row['chain_status'],
            'verdicts':[o.get('verdict') for o in row['outcomes']],'attempts':len(row['outcomes']),'steps':obs[-1]['steps'],
            'goal_unchanged':row['planned_session']['task_description']==protocol['task']['task_description'],
            'experience_unchanged':row['experience_before']==row['experience_after'],
            'policy_container':read(stream/'LAUNCH_pi05.json')['stdout'].strip(),'images':images,
            'phases':{k:dict(v) for k,v in phases.items()},'stage_intervals':dict(stage),'target_intervals':dict(target_spans),
            'tools':dict(Counter(e['name'] for e in events if e.get('event')=='tool')),
            'workflow_expansions':sum(e.get('event')=='workflow_expanded' for e in events)})
    a,b=cases
    initial=lambda c:{k:v for k,v in c['images'].items() if k.startswith('initial_')}
    comparison={'initial_images_identical':bool(initial(a)) and initial(a)==initial(b),'all_images_identical':bool(a['images']) and a['images']==b['images'],
        'distinct_fresh_policy_processes':a['policy_container']!=b['policy_container'],
        'identical_goal':a['goal_unchanged'] and b['goal_unchanged'],'steps_equal':a['steps']==b['steps'],
        'both_environment_success':a['environment_success'] and b['environment_success'],
        'observed_controller_savings_seconds':a['controller_seconds']-b['controller_seconds'],'observed_controller_savings_fraction':1-b['controller_seconds']/a['controller_seconds'],
        'scope':'One development pair, fresh seed 7 per arm and A-first order; no population noninferiority, learned-selection or cold-start payoff claim'}
    history=[]
    for n in [1,2,3]:
        r=ROOT/f'runs/workflow_reuse_v{n}_20261002';x=summarize(r);x.pop('native_run',None);history.append(x)
    keys=['full_wall_seconds','model_requests','model_rejected','incomplete_model_requests','prompt_tokens','completion_tokens','hcu_allocation_seconds','cpu_quota_seconds']
    new_bill={k:sum(x.get(k,0) for x in bills) for k in keys}
    new_bill['full_wall_seconds']=read(run/'STATUS.json')['full_wall_seconds']
    development={k:sum(x.get(k,0) for x in history+bills) for k in keys}
    development['full_wall_seconds']=sum(x['full_wall_seconds'] for x in history)+new_bill['full_wall_seconds']
    out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
    value={'run_id':args.run_id,'protocol':protocol,'cases':cases,'comparison':comparison,'group_bills':bills,
        'matched_pair_bill':new_bill,'all_four_campaigns_bill':development,
        'historical_source_episode_seconds':read(ROOT/'runs/native_long_v1_20261002/stream/native_long_t0_i8/RESULT.json')['seconds'],
        'accounting':'All four campaigns include unsuccessful development, validation, startup and stopping. Prior source acquisition disclosed separately; local editing/transfer/human time unmeasured. Service spans are nested and not additive. Allocation seconds are not utilization.'}
    with out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps({'comparison':comparison,'cases':[{k:c[k] for k in ['arm','seconds','environment_success','verdicts','steps','phases','stage_intervals','tools']} for c in cases], 'total_bill':development},indent=2))
if __name__=='__main__':main()
