"""Account for real branch executions and verify pre-choice observable matching."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.decision_corpus import checked_image
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def analyze(run):
 assert read(run/'STATUS.json')['status']=='COMPLETED';reg=read(run/'REGISTRATION.json');rows=[];prefixes={};inputs={};manifests={}
 for job in reg['jobs']:
  folder=run/job['child'];group=folder/'F';stream=group/'stream';result=read(stream/'RESULT.json');assert result['status']=='COMPLETED'
  row=result['cases'][0];assert row['chain_status']=='PASSED';sid=row['case']['id'];trace=lines(stream/'stage_trace'/sid/'TRACE.jsonl')
  calls=lines(stream/'GOAL_SKILL_CALLS.jsonl');events=lines(stream/'OS_COST_EVENTS.jsonl');actual=[x for x in events if x.get('event')=='actual_policy_input'];assert len(calls)==len(actual)
  execution=lines(stream/'GOAL_SKILL_EXECUTION.jsonl');used=0;image_checks=0
  for c,e in zip(calls,actual):
   if c['runtime_session']!=sid:continue
   tr=next(x for x in trace if x['step']==c['step'])
   for key,cam in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
    file=f"{c['step']:04d}_{cam}.png";checked_image((stream/'stage_trace'/sid/file).read_bytes(),tr['images'][file],e['array_bindings'][key]);image_checks+=1
   if c['origin']=='goal_skill':
    rec=execution[used];used+=1;assert rec['start_step']==c['step'] and rec['object_pose_used'] is False
    after=next(x for x in trace if x['step']==rec['end_step']);np.testing.assert_array_equal(np.asarray(rec['actual_actions'],np.float32),np.asarray(after['actions'],np.float32))
   assert (c['origin']=='goal_skill')==(c['step']<job['extent'])
  assert used==len(execution)
  prefixes[job['child']]=[dict(step=x['step'],images=x['images'],actions=x.get('actions',[])) for x in trace if x['step']<=130]
  at=next(e for c,e in zip(calls,actual) if c['runtime_session']==sid and c['step']==130);inputs[job['child']]={k:at[k] for k in ('prompt','robot_state','array_bindings')}
  cost=read(group/'STATUS.json');receipt=read(stream/'GOAL_SKILL_RECEIPT.json');assert receipt['actual_vla_calls']==sum(c['origin']=='vla' for c in calls)
  rows.append(dict(child=job['child'],role=job['role'],extent=job['extent'],native_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],chain=row['chain_status'],admission=row['experience_admission']['status'],physical_steps=trace[-1]['step'],sessions=len(row['sessions']),vla_calls=receipt['actual_vla_calls'],skill_requests=receipt['skill_requests'],boundary_requests=receipt['boundary_requests'],qwen_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),controller_seconds=cost['controller_seconds'],worker_seconds=cost['full_wall_seconds'],startup_seconds=cost['startup_seconds'],image_checks=image_checks,
    result_sha256=sha(stream/'RESULT.json'),initial_experience_seal=read(stream/sid/'INHERITANCE.json')['source_seal_sha256']))
  manifests[job['child']]=sha(group/'CODE_MANIFEST.json')
 pairs=[]
 for init in (4,5):
  a,b=f'init{init}_h130',f'init{init}_h262';assert prefixes[a]==prefixes[b] and inputs[a]==inputs[b]
  short,long=[next(r for r in rows if r['child']==x) for x in (a,b)];assert short['initial_experience_seal']==long['initial_experience_seal']
  pairs.append(dict(init=init,matched_observable_prefix=True,prefix_actions=sum(len(x['actions']) for x in prefixes[a]),
   vla_calls_saved=short['vla_calls']-long['vla_calls'],controller_seconds_saved=short['controller_seconds']-long['controller_seconds'],both_native_success=short['native_success'] and long['native_success']))
 return dict(status='ANALYZED',rows=rows,pairs=pairs,batch_wall_seconds=read(run/'STATUS.json')['seconds'],code_manifests=manifests,
  scope='Real same-code fixed-branch development executions. Model selector was not in this live loop. Native primary; semantic conflicts quarantined. No original full simulator snapshot equality or generalization guarantee asserted.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();value=analyze(a.run)
 with a.output.open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps(value,indent=2))
