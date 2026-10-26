"""Read complete paired evidence; static engineering savings never become learning."""
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def normalized_request(value):
 value=json.loads(json.dumps(value));value['benchmark']['run_id']='ARM';return value

def normalized_launch(group,role):
 stream=group/'stream';argv=list(read(stream/('LAUNCH_'+role+'.json'))['argv']);argv[argv.index('--name')+1]='ARM_CONTAINER'
 raw=str(stream);inside=raw.replace(str(ROOT),'/project');remote='/mnt/data/task/yuchen/FastPhyagentos/FastRSI/'+stream.relative_to(ROOT).as_posix()
 return [v.replace(raw,'STREAM').replace(inside,'STREAM').replace(remote,'STREAM') for v in argv]

def analyze(run):
 status=read(run/'STATUS.json');report={'run_id':run.name,'status':status['status'],'scope':'Public static Session construction pilot; no learned FastRSI claim','arms':[]}
 for arm in ['A','B']:
  group=run/arm;stream=group/'stream';p=stream/'RESULT.json'
  if not p.exists():continue
  result=read(p);row=result['cases'][0];folder=stream/row['case']['id'];w=folder/'workspace'
  events=lines(folder/'AGENT_EVENTS.jsonl');cost=lines(stream/'OS_COST_EVENTS.jsonl');target=lines(stream/'TARGET_COST_EVENTS.jsonl')
  queued=next((e for e in events if e['event']=='queue_accepted'),None);dispatch_ns=round(queued['time']*1e9) if queued else None
  model_begins=[e for e in cost if e.get('event')=='begin' and e.get('name')=='agent_model_call']
  stage=defaultdict(float)
  for e in cost+target:
   if e.get('event')=='end':stage[e['name']]+=e['duration_seconds']
  prompts=[e['prompt'] for e in cost if e.get('event')=='actual_policy_input']
  receipts=[read(p) for p in stream.glob('service_calls/*/RECEIPT.json')]
  request=read(w/'REQUEST.json');image_root=w/'artifacts/runtime'/row['case']['id']/'rgb'
  report['arms'].append({'arm':arm,'chain_status':row['chain_status'],'environment_success':row['environment_success'],
    'verification_verdicts':[o.get('verdict') for o in row['outcomes']], 'controller_seconds':result['seconds'],
    'full_wall_seconds':read(group/'STATUS.json')['full_wall_seconds'],'stage_seconds':dict(stage),
    'planning_model_calls':sum(e['wall_ns']<dispatch_ns for e in model_begins) if dispatch_ns else None,
    'agent_model_calls_total':len(model_begins),'service_requests':len(list(stream.glob('service_calls/*/REQUEST.json'))),
    'service_receipts':len(receipts),'service_generation_seconds':sum(r.get('generation_seconds',0) for r in receipts),
    'prompt_tokens':sum(r.get('prompt_tokens',0) for r in receipts),'completion_tokens':sum(r.get('completion_tokens',0) for r in receipts),
    'queue_origin':queued.get('origin','agent') if queued else None,'flow_fallbacks':sum(e['event']=='session_flow_handoff' for e in events),
    'physical_steps':{k:e['steps'] for k,e in row.get('native_terminal_by_session',{}).items()},'policy_calls':len(prompts),
    'same_goal_at_vla':bool(prompts) and set(prompts)=={request['goal']},'persistent_state_unchanged':row['experience_before']==row['experience_after'],
    'eval_closed':(w/'.FASTRSI_EVALUATION_CLOSED.json').exists(),'request':request,
    'initial_images':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in image_root.glob('initial_*.png')},
    'initial_fixed_state':row['experience_before'],'accepted_source':row.get('accepted_source'),'report_passed':row.get('terminal_report',{}).get('status')=='PASSED'})
 if len(report['arms'])==2:
  a,b=report['arms'];launch_match={role:normalized_launch(run/'A',role)==normalized_launch(run/'B',role) for role in ['qwen','pi05','target']}
  equal={'immutable_intent_equal':normalized_request(a['request'])==normalized_request(b['request']),
    'initial_images_equal':bool(a['initial_images']) and a['initial_images']==b['initial_images'],
    'same_initial_experience':a['accepted_source']==b['accepted_source'] and {k:v for k,v in a['initial_fixed_state'].items() if k!='REQUEST.json'}=={k:v for k,v in b['initial_fixed_state'].items() if k!='REQUEST.json'},
    'matching_launch_config':launch_match,'both_fixed_and_closed':all(r['persistent_state_unchanged'] and r['eval_closed'] for r in [a,b]),
    'both_actual_vla_goals_match':all(r['same_goal_at_vla'] for r in [a,b]),'both_chain_and_report_complete':all(r['chain_status']=='PASSED' and r['report_passed'] for r in [a,b])}
  valid=all(v for k,v in equal.items() if k!='matching_launch_config') and all(launch_match.values())
  prelude=status['groups'][0]['started_at']-status['started_at'];postlude=max(0,status['finished_at']-status['groups'][-1]['finished_at'])
  report['comparison']={**equal,'matched_pair_valid':valid,'static_structure_replaced':b['planning_model_calls']==0 and b['queue_origin']=='public_session_flow',
     'controller_difference_seconds':a['controller_seconds']-b['controller_seconds'],
     'full_arm_difference_seconds':a['full_wall_seconds']-b['full_wall_seconds'],
     'measured_candidate_setup_and_cpu_checks_seconds':prelude,'postlude_seconds':postlude,
     'B_with_all_measured_prelude_postlude_seconds':b['full_wall_seconds']+prelude+postlude,
     'difference_after_charging_all_measured_setup_to_B_seconds':a['full_wall_seconds']-b['full_wall_seconds']-prelude-postlude,
     'quality_note':'Single paired development instance: outcomes are observations, not noninferiority, representative quality or learning proof',
     'accounting_note':'All measured campaign prelude incl acquisition, code snapshot and CPU checks charged conservatively to B for H=1 diagnostic. Local implementation, human work and network transfer unmeasured and disclosed. Study campaign bill includes both arms; no algorithm cumulative savings claim.'}
 bill=summarize(run);all_receipts=[read(p) for p in run.glob('*/stream/service_calls/*/RECEIPT.json')]
 bill['model_calls']=len(all_receipts);bill['model_requests']=len(list(run.glob('*/stream/service_calls/*/REQUEST.json')))
 bill['incomplete_model_requests']=bill['model_requests']-len(all_receipts)
 bill['model_rejected']=sum(r.get('status')!='OK' for r in all_receipts)
 bill['service_call_seconds']=sum(r['seconds'] for r in all_receipts if 'seconds' in r)
 bill['service_seconds_receipt_coverage']=sum('seconds' in r for r in all_receipts)
 for k in ['generation_seconds','prompt_tokens','completion_tokens']:bill[k]=sum(r.get(k,0) for r in all_receipts)
 report['complete_study_bill']=bill
 report['not_claimed']=['learning benefit','30% cumulative method savings','2pp noninferiority','task-general robustness','candidate generation by autonomous model','native mechanism default changed']
 return report

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);args=ap.parse_args()
 result=analyze(ROOT/'runs'/args.run_id);out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps({'status':result['status'],'arms':[{k:r[k] for k in ['arm','environment_success','verification_verdicts','planning_model_calls','controller_seconds','full_wall_seconds']} for r in result['arms']],'comparison':result.get('comparison')},ensure_ascii=False,indent=2))
