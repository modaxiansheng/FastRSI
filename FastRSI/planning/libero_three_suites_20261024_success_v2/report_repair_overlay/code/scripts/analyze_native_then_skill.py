"""Verify actual native-prefix/source-suffix composition, never splice outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=R/'runs/native_then_skill_v1_20261005';assert read(run/'STATUS.json')['status']=='COMPLETED';g=run/'init5_composed/F';s=g/'stream';result=read(s/'RESULT.json');row=result['cases'][0];assert row['chain_status']=='PASSED'
 sid=row['case']['id'];trace=lines(s/'stage_trace'/sid/'TRACE.jsonl');reference=R/'runs/phase_reference_v1_20261005/init5_native/F/stream';native=lines(reference/'stage_trace'/sid/'TRACE.jsonl')
 def before(rows):return [dict(step=x['step'],images=x['images'],actions=x.get('actions',[])) for x in rows if x['step']<=130]
 assert before(trace)==before(native),'Native-prefix actions or observations changed'
 calls=lines(s/'GOAL_SKILL_CALLS.jsonl');execs=lines(s/'GOAL_SKILL_EXECUTION.jsonl');receipt=read(s/'GOAL_SKILL_RECEIPT.json');assert receipt['source_start_step']==130
 for c in calls:
  assert (c['origin']=='goal_skill')==(c['runtime_session']==sid and 130<=c['step']<262)
 for e in execs:
  assert e['start_step']>=130 and e['end_step']<=262 and e['object_pose_used'] is False
  actual=next(x for x in trace if x['step']==e['end_step']);np.testing.assert_array_equal(np.asarray(e['actual_actions'],np.float32),np.asarray(actual['actions'],np.float32))
 assert len(execs)==receipt['skill_requests']
 events=lines(s/'TARGET_COST_EVENTS.jsonl');resets=[e for e in events if e.get('event')=='native_observation' and e.get('method')=='reset'];assert sum(e['session_id']==sid for e in resets)==1
 cost=read(g/'STATUS.json');value=dict(status='ANALYZED',native_success=row['environment_success'],semantic=[o['verdict'] for o in row['outcomes']],admission=row['experience_admission']['status'],chain=row['chain_status'],physical_steps=trace[-1]['step'],vla_calls=receipt['actual_vla_calls'],skill_requests=receipt['skill_requests'],boundary_requests=receipt['boundary_requests'],qwen_calls=len(list((s/'service_calls').glob('*/REQUEST.json'))),controller_seconds=cost['controller_seconds'],worker_seconds=cost['full_wall_seconds'],batch_wall_seconds=read(run/'STATUS.json')['seconds'],native_prefix_actions=130,matched_native_prefix=True,same_episode=True,
  automatic_discovery=False,learned_selector=False,scope='One manually composed candidate on already-seen failure. Actual new execution, not concatenated outcomes. Not independent generalization or RSI gain.',
  sources={str(p.relative_to(R)):sha(p) for p in (s/'RESULT.json',s/'GOAL_SKILL_CALLS.jsonl',s/'GOAL_SKILL_EXECUTION.jsonl',s/'stage_trace'/sid/'TRACE.jsonl')})
 out=R/'planning/phase_choice_v1_20261005/COMPOSITION_ANALYSIS.json'
 with out.open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps(value,indent=2))
if __name__=='__main__':main()
