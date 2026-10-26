"""Audit held-instance outcome, actual prompts, immutable experience and paid bill."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);args=ap.parse_args();run=ROOT/'runs'/args.run_id
 report={'run_id':args.run_id,'status':read(run/'STATUS.json'),'scope':'One held development initialization, not final test or main A/B/C','arms':[]}
 bill=summarize(run);receipts=[read(p) for p in run.glob('*/stream/service_calls/*/RECEIPT.json')]
 bill['model_calls']=len(receipts);bill['model_requests']=len(list(run.glob('*/stream/service_calls/*/REQUEST.json')));bill['incomplete_model_requests']=bill['model_requests']-bill['model_calls'];bill['model_rejected']=sum(r.get('status')!='OK' for r in receipts)
 for k in ['generation_seconds','prompt_tokens','completion_tokens']:bill[k]=sum(r.get(k,0) for r in receipts)
 for name in ['eval_A','eval_B']:
  group=run/name;stream=group/'stream';p=stream/'RESULT.json'
  if not p.exists():continue
  result=read(p);row=result['cases'][0];folder=stream/row['case']['id'];w=folder/'workspace'
  spans={};prompts=[]
  for file in ['OS_COST_EVENTS.jsonl','TARGET_COST_EVENTS.jsonl']:
   totals=defaultdict(float);q=stream/file
   for e in map(json.loads,q.read_text().splitlines() if q.exists() else []):
    if e.get('event')=='end':totals[e['name']]+=e['duration_seconds']
    if e.get('event')=='actual_policy_input':prompts.append(e['prompt'])
   spans[file]=dict(totals)
  req=read(w/'REQUEST.json');expected=req['instruction_decision']['instruction']
  policy=read(stream/'LAUNCH_pi05.json');assert '--torch-seed 7' in policy['argv'][-1]
  arm={'arm':name,'controller_seconds':result['seconds'],'chain_status':row['chain_status'],'environment_success':row['environment_success'],'verification':[o.get('verdict') for o in row['outcomes']],
      'steps':{sid:e['steps'] for sid,e in row.get('native_terminal_by_session',{}).items()},'actual_instruction':expected,'actual_prompt_verified':bool(prompts) and set(prompts)=={expected},'policy_calls':len(prompts),
      'initial_images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (w/'artifacts/runtime'/row['case']['id']/'rgb').glob('initial_*.png')},
      'experience_unchanged':row['experience_before']==row['experience_after'],'learned_state_sha256':hashlib.sha256((w/'skills/instruction-reuse/EXPERIENCE.json').read_bytes()).hexdigest(),
      'fresh_policy_container':policy['stdout'].strip(),'spans':spans,'full_wall_seconds':read(group/'STATUS.json')['full_wall_seconds']}
  report['arms'].append(arm)
 if len(report['arms'])==2:
  a,b=report['arms'];good=all(x['chain_status']=='PASSED' and x['actual_prompt_verified'] and x['experience_unchanged'] for x in [a,b])
  report['comparison']={'both_chain_legal_fixed':good,'initial_images_identical':bool(a['initial_images']) and a['initial_images']==b['initial_images'],'both_environment_success':all(x['environment_success'] is True for x in [a,b]),
      'same_fixed_learned_state':a['learned_state_sha256']==b['learned_state_sha256']==hashlib.sha256((run/'LEARNED.json').read_bytes()).hexdigest(),
      'fresh_policy_processes':a['fresh_policy_container']!=b['fresh_policy_container'],'controller_savings_fraction':1-b['controller_seconds']/a['controller_seconds'] if good else None,
      'full_arm_wall_savings_fraction':1-b['full_wall_seconds']/a['full_wall_seconds'] if good else None,
      'scope':'Single predeclared held development pair. Failures count; no population, 2pp or 30% claim.'}
 if (run/'probe/stream/RESULT.json').exists():report['probe']=read(run/'probe/stream/RESULT.json')
 report['this_run_bill']=bill
 prior=read(ROOT/'planning/instruction_trial_v3/FINAL_BILL.json')['totals'];report['branch_total_bill']={k:v+bill.get(k,0) for k,v in prior.items()}
 report['additional_prior_bills']=[]
 for other in sorted([*ROOT.glob('runs/instruction_followup_*/DELIVERY_MANIFEST.json'),*ROOT.glob('runs/instruction_patch_check_*/DELIVERY_MANIFEST.json')]):
  other=other.parent
  if other==run:continue
  b=summarize(other);rs=[read(p) for p in other.glob('*/stream/service_calls/*/RECEIPT.json')]
  b['model_calls']=len(rs);b['model_requests']=len(list(other.glob('*/stream/service_calls/*/REQUEST.json')));b['incomplete_model_requests']=b['model_requests']-len(rs);b['model_rejected']=sum(r.get('status')!='OK' for r in rs)
  for k in ['generation_seconds','prompt_tokens','completion_tokens']:b[k]=sum(r.get(k,0) for r in rs)
  report['additional_prior_bills'].append(b)
  for k in prior:report['branch_total_bill'][k]+=b.get(k,0)
 report['accounting']='Trial attempts, CPU repair and this followup. Prior baseline/other branches and local editing/transfer/human time excluded; spans overlap; HCU/CPU capacity-time is not utilization.'
 out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps({k:report[k] for k in ['status','arms','branch_total_bill']},indent=2))
if __name__=='__main__':main()
