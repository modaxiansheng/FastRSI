"""Account for the actual learning/transfer experiment without implied branches."""
import argparse,collections,hashlib,json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def events(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def summarize(group):
 stream=group/'stream';report=read(stream/'RESULT.json');row=report['cases'][0];s=read(group/'STATUS.json');ev=events(stream/'OS_COST_EVENTS.jsonl')
 assert report['status']==s['status']=='COMPLETED' and row['chain_status']=='PASSED'
 resident=read(stream/'RESIDENT_SESSION.json');assert resident['status']=='COMPLETED' and resident['release_confirmed']
 startup=read(group.parents[3]/'resident'/group.name/'pool/pi05/resident_receipts/STARTUP.json')
 assert resident['begin']['receipt']['policy_state']==startup['policy_state'],'Service state did not match its own loaded baseline'
 inputs=[e for e in ev if e['event']=='actual_policy_input'];choices=[e for e in ev if e['event']=='feedback_window_choice']
 assert inputs and len(inputs)==len(choices)
 modes={e['mode'] for e in choices};assert len(modes)==1
 case=row['case'];choice=row.get('task_window_decision');expected=choice['selected_mode'] if choice else case['feedback_window_mode'];assert modes=={expected}
 if choice:
  artifact=group.parent/('UPDATED.json' if group.name=='F' else 'FROZEN.json');assert digest(artifact)==choice['snapshot_sha256']
 if case['mode']=='evaluation':assert row['experience_before']==row['experience_after']
 counts=collections.Counter();seconds=collections.defaultdict(float)
 for e in ev:
  if e['event']=='end':counts[e['name']]+=1;seconds[e['name']]+=e['duration_seconds']
 target=collections.defaultdict(float)
 for e in events(stream/'TARGET_COST_EVENTS.jsonl'):
  if e['event']=='end':target[e['name']]+=e['duration_seconds']
 trace=events(stream/'stage_trace'/case['id']/'TRACE.jsonl')
 signature=hashlib.sha256(json.dumps([{k:e.get(k) for k in ('method','step','images','native_success','actions')} for e in trace],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 native=list(row['native_terminal_by_session'].values());assert native
 return dict(trajectory_signature=signature,task=case['task_id'],init=case['init_state_id'],phase=case['mode'],mode=expected,task_window_decision=choice,
  native_success=row['environment_success'],semantic_verdicts=[x['verdict'] for x in row['outcomes']],chain_status=row['chain_status'],
  admission=row.get('experience_admission'),vla_calls=len(inputs),agent_calls=counts['agent_model_call'],steps=sum(x['steps'] for x in native),
  case_seconds=row['seconds'],worker_seconds=s['full_wall_seconds'],stage_seconds=dict(seconds),target_seconds=dict(target),
  initial_rng_sha256=resident['begin']['receipt']['rng_sha256'],initial_policy_state_sha256=resident['begin']['receipt']['policy_state']['state_sha256'],initial_policy_sources=resident['begin']['receipt']['policy_state']['sources_sha256'],service_state_matches_own_startup=True,
  initial_input={k:inputs[0][k] for k in ('prompt','array_bindings','robot_state')},model_config_sha256=digest(group/'MODEL_CONFIG.json'),
  evaluation_isolation_passed=(row['experience_before']==row['experience_after']) if case['mode']=='evaluation' else None)

def analyze(run):
 s=read(run/'STATUS.json');assert s['status']=='COMPLETED'
 warmup=read(run/'WARMUP_CHECK.json');assert warmup['status']=='PASSED' and warmup['policy_released'] and warmup['tools_executed']==0
 proto=read(run/'PROTOCOL.json');update=read(run/'UPDATE_RECEIPT.json')
 assert digest(run/'FROZEN.json')==update['frozen_sha256'] and digest(run/'UPDATED.json')==update['updated_sha256']
 result={'development':[],'evaluation':[]}
 for phase,arms in [('development',('N','F')),('evaluation',('N','F','R'))]:
  for i,c in enumerate(proto[phase]):
   pair_root=run/phase/('pair'+str(i))
   if phase=='development' and str(i) in proto.get('reused_development',{}):pair_root=run.parent.parent/proto['reused_development'][str(i)]
   rows={a:summarize(pair_root/('campaign_'+a)/a) for a in arms}
   assert all(r['initial_input']==rows[arms[0]]['initial_input'] for r in rows.values()),'Initial legal inputs differ'
   assert len({r['model_config_sha256'] for r in rows.values()})==1
   assert len({r['initial_rng_sha256'] for r in rows.values()})==1
   assert all(r['initial_policy_sources']==rows[arms[0]]['initial_policy_sources'] for r in rows.values())
   for r in rows.values():r.pop('initial_input')
   result[phase].append(dict(task=c['task_id'],init=c['init_state_id'],arms=rows,initial_inputs_exact=True,same_executed_trajectory=len({r['trajectory_signature'] for r in rows.values()})==1))
 totals={}
 for a in ('N','F','R'):
  rows=[r['arms'][a] for r in result['evaluation']]
  totals[a]={k:sum(r[k] for r in rows) for k in ('native_success','vla_calls','agent_calls','steps','case_seconds','worker_seconds')}
  totals[a]['semantic_success']=sum(r['semantic_verdicts']==['success'] for r in rows)
  totals[a]['episodes']=len(rows)
 dev_cost=sum(r['worker_seconds'] for pair in result['development'] for r in pair['arms'].values())
 oracle=[]
 for pair in result['development']:
  rows=pair['arms'];available=[a for a,r in rows.items() if r['native_success']]
  best=min(available,key=lambda a:rows[a]['worker_seconds']) if available else None
  oracle.append(dict(task=pair['task'],init=pair['init'],best_observed_native_success_arm=best,default_native_success=rows['N']['native_success'],observed_saving_vs_fixed10=rows['N']['worker_seconds']-rows[best]['worker_seconds'] if best and rows['N']['native_success'] else None,scope='Development hindsight on actually executed arms, native outcome only; not a legal selector, not an admitted update, not generalization.'))
 diff=[]
 for pair in result['evaluation']:
  a,b=pair['arms']['R'],pair['arms']['F'];diff.append(dict(task=pair['task'],init=pair['init'],frozen_mode=a['mode'],updated_mode=b['mode'],changed=a['mode']!=b['mode'],observed_worker_saving=a['worker_seconds']-b['worker_seconds'],quality=[a['native_success'],b['native_success']]))
 return dict(status='COMPLETED_AND_CHECKED',**result,update=update,evaluation_totals=totals,decisions=diff,development_hindsight=oracle,
  ledger=dict(frozen_warmup={k:warmup[k] for k in ('status','seconds','model_calls','failed_http_attempts','policy_released','tools_executed')},batch_wall_seconds=s['full_wall_seconds'],prior_interrupted_batch_seconds=read(run.parent/proto['resume_from']/'STATUS.json')['full_wall_seconds'] if proto.get('resume_from') else 0,pool_startup_seconds=s['pool_startup_seconds'],development_worker_seconds_sum=dev_cost,update_seconds=update['update_seconds'],evaluation_worker_seconds_by_arm={a:r['worker_seconds'] for a,r in totals.items()},
    scope='Wall time and worker-time sums are separate, nested stage timings not added. Development is a shared mechanism-diagnostic input; each independently reproducing acquisition would pay it. Not a cold-start superiority trial.'),
  learning_net_gain_established=False,production_promotion=False,
  interpretation='Small development holdout mechanism experiment. Same-action timing differences cannot establish learning gain; no 2pp or 30% claim. Historical acquisition fees retained in preparation/HISTORY_COST.json.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=analyze(a.run)
 with a.output.open('x') as f:json.dump(r,f,indent=2)
 print(json.dumps({k:r[k] for k in ('status','update','evaluation_totals','decisions','ledger')},indent=2))
