"""One verified-prefix operation consequence; no cross-batch speed claim."""
import argparse,hashlib,json
from pathlib import Path
from collections import defaultdict
from analyze_stack_flow import summarize_attempts,GOALS

def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(v) for v in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(run,reference):
 status=read(run/'STATUS.json');assert status['status']=='COMPLETED'
 group=run/'F';stream=group/'stream';result=read(stream/'RESULT.json');assert result['status']=='COMPLETED' and len(result['cases'])==1
 row=result['cases'][0];case=row['case'];sid='net_t63_i0'
 expected={'suite':'libero_90','task_id':63,'init_state_id':0,'id':sid,'mode':'development','max_steps':400,'task_description':GOALS[63],'feedback_window_mode':'fixed10','development_suite_extension':'libero90_stack_flow_v1'}
 for key,value in expected.items():assert type(case.get(key)) is type(value) and case[key]==value,key
 assert case==read(group/'ROSTER.json')[0]
 spec=read(run/'BOUNDARY.json');assert spec['boundary_step']==120 and spec['max_steps']==400
 guard=read(stream/f'BOUNDARY_{sid}.json')
 assert guard['status']=='PREFIX_MATCHED' and guard['prefix_matched'] and guard['failure'] is None
 assert guard['verified_observations']==13 and guard['verified_action_chunks']==12
 assert guard['teacher_calls']==0 and guard['learning_label'] is None and guard['artifact_sha256']==sha(run/'BOUNDARY.json')
 events=lines(stream/'OS_COST_EVENTS.jsonl');windows=[e for e in events if e['event']=='feedback_window_choice'];inputs=[e for e in events if e['event']=='actual_policy_input']
 traces={s['session_id']:lines(stream/'stage_trace'/s['session_id']/'TRACE.jsonl') for s in row['sessions']}
 attempts=summarize_attempts(row,traces,windows,lines(stream/'TARGET_COST_EVENTS.jsonl'))
 root=traces[sid];prior=lines(reference/'stage_trace'/sid/'TRACE.jsonl')
 assert sha(reference/'stage_trace'/sid/'TRACE.jsonl')==spec['reference_trace']['sha256']
 assert sha(reference/'OS_COST_EVENTS.jsonl')==spec['reference_cost']['sha256']
 assert [v for v in root if v['step']<=120]==[v for v in prior if v['step']<=120]
 assert len(inputs)==len(windows)
 root_inputs=[e for e,w in zip(inputs,windows) if w['runtime_session_id']==sid]
 assert len(root_inputs)==len(root)-1,'Unexpected input count/attempt attribution'
 for step,e in zip([r['step'] for r in root[:-1]],root_inputs):
  assert e['prompt']==(spec['goal'] if step<120 else spec['remaining_goal'])
 assert guard['changed_policy_prompts']==len(root_inputs)-12
 assert len(inputs)==sum(a['vla_calls'] for a in attempts)
 spans=defaultdict(float)
 for e in events:
  if e['event']=='end':spans[e['name']]+=e['duration_seconds']
 worker=read(group/'STATUS.json');priorrow=read(reference/'RESULT.json')['cases'][0]
 assert row['chain_status']=='PASSED' and row['case']['task_description']==priorrow['case']['task_description']
 sources=[run/'BOUNDARY.json',stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/f'BOUNDARY_{sid}.json',stream/'stage_trace'/sid/'TRACE.jsonl']
 return dict(status='ANALYZED',original=dict(environment_success=priorrow['environment_success'],semantic=[o['verdict'] for o in priorrow['outcomes']],root_steps=prior[-1]['step'],root_vla_calls=len(prior)-1,case_seconds=priorrow['seconds']),
  intervention=dict(environment_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],chain=row['chain_status'],admission=row['experience_admission']['status'],attempts=attempts,
     root_steps=root[-1]['step'],root_remaining_steps=root[-1]['step']-120,root_vla_calls=len(root)-1,root_remaining_vla_calls=len(root)-13,
     all_steps=sum(a['steps'] for a in attempts),vla_calls=len(inputs),changed_prompts=guard['changed_policy_prompts'],teacher_calls=0,
     qwen_calls=len(list((stream/'service_calls').glob('*/REQUEST.json'))),agent_calls=len(list((stream/sid).glob('agent_call_*/REQUEST.json'))),
     case_seconds=row['seconds'],controller_seconds=worker['controller_seconds'],worker_seconds=worker['full_wall_seconds'],startup_seconds=worker['startup_seconds'],nested_stage_seconds=dict(spans)),
  prefix=dict(observations=13,images=26,actions=120,action_chunks=12,exact_trace_equal=True,guard_before_submission=True),
  batch_wall_seconds=status['seconds'],trained=False,learning_gain_established=False,
  scope='One seen developer case, manually selected remaining-action prompt at post-hoc step120; no learned timing, transfer, or matched cross-batch wall-time speed claim.',
  cost_scope='Original and new root costs retained; new root physically regenerates and pays its prefix. Nested spans must not be added. Different batches/resources prevent interpreting wall-time difference as method speed.',
  source_sha256={str(p.relative_to(run)):sha(p) for p in sources})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 v=analyze(a.run,a.reference)
 with a.output.open('x') as f:json.dump(v,f,indent=2)
 print(json.dumps(v,indent=2))
