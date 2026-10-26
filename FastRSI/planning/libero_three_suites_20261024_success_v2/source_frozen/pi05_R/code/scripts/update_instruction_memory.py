"""Paid development admission and next-Session choice receipt, never from evaluation."""
import json,sys,hashlib,time
from pathlib import Path
from fastrsi.instruction_memory import update,select

def read(p):return json.loads(p.read_text())
def make_update(campaign):
    start=time.perf_counter();campaign=Path(campaign);state=read(campaign/'PROPOSED.json');observed={}
    for arm,action in [('A','native'),('B','candidate')]:
        stream=campaign/('dev_'+arm)/'stream';report=read(stream/'RESULT.json');row=report['cases'][0];case=row['case'];folder=stream/case['id']
        events=[json.loads(x) for x in (stream/'OS_COST_EVENTS.jsonl').read_text().splitlines()]
        prompts=[e['prompt'] for e in events if e.get('event')=='actual_policy_input' and e.get('case')==case['id']]
        rgb=folder/'workspace/artifacts/runtime'/case['id']/'rgb'
        expected=state['original_goal'] if action=='native' else state['candidate_instruction']
        observed[action]={'mode':case['mode'],'goal':case['task_description'],'instruction':row['planned_session']['task_description'],
            'seconds':report['seconds'],'init_state_id':case['init_state_id'],
            'initial_images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in rgb.glob('initial_*.png')},
            'chain_passed':row['chain_status']=='PASSED','feedback_admitted':row.get('experience_admission',{}).get('status')=='ADMITTED',
            'environment_success':row['environment_success'],'verification_success':bool(row['outcomes']) and all(o.get('verdict')=='success' for o in row['outcomes']),
            'prompt_verified':bool(prompts) and set(prompts)=={expected},
            'evidence':str(folder/'RESULT.json'),'evidence_sha256':hashlib.sha256((folder/'RESULT.json').read_bytes()).hexdigest(),
            'all_attempts':len(row['outcomes'])}
    value=update(state,observed);p=campaign/'LEARNED.json'
    with p.open('x') as f:json.dump(value,f,indent=2)
    case={**value['scope'],'task_description':value['original_goal']}
    receipt={'before':select(state,case,value['scope']['model_config_sha256']),
             'after':select(value,case,value['scope']['model_config_sha256']),
             'state_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start,
             'scope':'Development-only two-path diagnostic. Both action acquisition costs paid; not isolated end-to-end A/B/C.'}
    with (campaign/'POLICY_UPDATE.json').open('x') as f:json.dump(receipt,f,indent=2)
    return value
if __name__=='__main__':print(json.dumps(make_update(sys.argv[1]),indent=2))
