"""Join the completed branch, reference and learner evidence without net-gain claims."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 branch=read(R/'planning/phase_choice_v1_20261005/ANALYSIS.json');reference=R/'runs/phase_reference_v1_20261005';assert read(reference/'STATUS.json')['status']=='COMPLETED'
 refs=[];bindings={}
 for i in (4,5):
  g=reference/f'init{i}_native/F';s=g/'stream';result=read(s/'RESULT.json');assert result['status']=='COMPLETED' and len(result['cases'])==1;row=result['cases'][0];assert row['chain_status']=='PASSED'
  case=row['case'];assert case['goal_skill_probe']['enabled'] is False and case['init_state_id']==i
  calls=read(s/'GOAL_SKILL_RECEIPT.json');assert calls['skill_requests']==0
  trace=s/'stage_trace'/case['id']/'TRACE.jsonl';tr=[json.loads(t) for t in trace.read_text().splitlines()]
  reset=json.loads((R/f'runs/phase_choice_v1_20261005/init{i}_h262/F/stream/stage_trace'/case['id']/'TRACE.jsonl').read_text().splitlines()[0]);assert reset==tr[0]
  cost=read(g/'STATUS.json');refs.append(dict(init=i,native_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],chain=row['chain_status'],vla_calls=calls['actual_vla_calls'],qwen_calls=len(list((s/'service_calls').glob('*/REQUEST.json'))),physical_steps=tr[-1]['step'],controller_seconds=cost['controller_seconds'],worker_seconds=cost['full_wall_seconds'],same_initial_observation=True))
  for p in (s/'RESULT.json',s/'GOAL_SKILL_RECEIPT.json',g/'STATUS.json',trace):bindings[str(p.relative_to(R))]=sha(p)
 learning=read(R/'runs/phase_learning_v1_20261005/learning/RESULT.json');assert learning['status']=='COMPLETED'
 stops=[]
 for name in ('phase_choice_v1_20261005','phase_reference_v1_20261005','phase_learning_v1_20261005'):
  for p in (R/'runs'/name).rglob('STOP_RECEIPT.json'):
   for item in read(p):
    assert item['running'] is False;stops.append(item['container'])
 value=dict(status='COMPLETED',branches=branch['rows'],original_policy_references=refs,learning=learning,
  resources={'max_concurrent_cards':9,'robot_host':'187','learner_host':'188','unique_stopped_containers':len(set(stops))},
  costs={'branch_parallel_wall_seconds':branch['batch_wall_seconds'],'reference_parallel_wall_seconds':read(reference/'STATUS.json')['seconds'],'learner_parallel_wall_seconds':read(R/'runs/phase_learning_v1_20261005/STATUS.json')['seconds'],
   'robot_controller_seconds_sum':sum(x['controller_seconds'] for x in branch['rows'])+sum(x['controller_seconds'] for x in refs),
   'learner_active_calls_seconds':sum(x['seconds'] for x in learning['calls']),'learner_fit_seconds':sum(learning['fit_seconds'].values()),
   'accounting':'Per-worker controller sums are aggregate work, not elapsed wall. Learning overlaps the branch batch and includes load/wait. Historical source and prior experiments remain charged in their ledgers. References ran later with two rather than four workers; do not claim matched wall-time speedup.'},
  new_robot_episodes=6,actual_training_versions=2,live_learned_policy_episodes=0,default_changed=False,learning_net_gain_established=False,source_sha256=bindings)
 with (R/'planning/phase_choice_v1_20261005/STUDY.json').open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({'reference':refs,'resources':value['resources'],'costs':value['costs']},indent=2))
if __name__=='__main__':main()
