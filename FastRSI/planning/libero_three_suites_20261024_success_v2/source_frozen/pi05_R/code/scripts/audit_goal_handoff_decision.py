"""Recover a real phase-boundary contrast from recorded actions, not imagined branches."""
import hashlib,json,struct,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from fastrsi.decision_corpus import checked_image
NAME='goal_applicability_shadow_v1_20261005';SID='net_t64_i1';STEP=130
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 branches=[];prefixes=[];inputs=[];source_files={}
 for name,action in [('goal_skill_handoff_v1_20261005','handoff_to_pi05'),('source_completion_handoff_v1_20261005','continue_source_until_262_then_pi05')]:
  run=R/'runs'/name;s=run/'F/stream';tr=s/'stage_trace'/SID/'TRACE.jsonl'
  rows=lines(tr);prefix=[x for x in rows if x['step']<=STEP]
  assert [x['step'] for x in prefix]==list(range(0,131,10))
  prefixes.append([dict(step=x['step'],images=x['images'],actions=x.get('actions',[]),native_success=x['native_success']) for x in prefix])
  events=lines(s/'OS_COST_EVENTS.jsonl');actual=[e for e in events if e.get('event')=='actual_policy_input']
  choices=lines(s/'GOAL_SKILL_CALLS.jsonl')
  assert len(actual)==len(choices)
  selected=[(e,c) for e,c in zip(actual,choices) if c['step']==STEP];assert len(selected)==1
  e,c=selected[0];assert c['runtime_session']==SID and e['case']==SID
  assert c['origin']==('vla' if action=='handoff_to_pi05' else 'goal_skill'),c
  inputs.append({k:e[k] for k in ('prompt','robot_state','array_bindings')})
  assert hashlib.sha256(struct.pack('<8f',*e['robot_state'])).hexdigest()==e['array_bindings']['observation/state']['sha256']
  for x in prefix:
   for file,digest in x['images'].items():assert sha(tr.parent/file)==digest
  for key,camera in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
   fn=f'{STEP:04d}_{camera}.png';checked_image((tr.parent/fn).read_bytes(),prefix[-1]['images'][fn],e['array_bindings'][key])
  result=read(s/'RESULT.json');summary=read(R/'planning'/name/'ANALYSIS.json');row=next(x for x in summary['rows'] if x['arm']=='F')
  branches.append(dict(action=action,native_success=rows[-1]['native_success'],semantic=row['semantic'],terminal_step=rows[-1]['step'],remaining_physics_steps=rows[-1]['step']-STEP,remaining_vla_calls=sum(c['step']>=STEP and c['origin']=='vla' for c in choices),source=str(s.relative_to(R))))
  for p in (tr,s/'OS_COST_EVENTS.jsonl',s/'GOAL_SKILL_CALLS.jsonl',s/'RESULT.json',run/'MODEL_CONFIG.json'):
   source_files[str(p.relative_to(R))]=sha(p)
 assert prefixes[0]==prefixes[1], 'Pre-branch action/pixel prefixes differ'
 assert inputs[0]==inputs[1], 'Actual pre-branch robot/model inputs differ'
 assert sha(R/'runs/goal_skill_handoff_v1_20261005/MODEL_CONFIG.json')==sha(R/'runs/source_completion_handoff_v1_20261005/MODEL_CONFIG.json')
 assert len([a for x in prefixes[0] for a in x['actions']])==130
 out=R/'planning'/NAME/'PHASE130_DECISION_EVIDENCE.json'
 value=dict(status='MATCHED_OBSERVABLE_PREFIX',phase=STEP,prefix_actions=130,prefix_boundaries=14,images_per_branch=28,pre_decision_input=inputs[0],branches=branches,source_sha256=source_files,primary_protocol='native_primary_v1',observed_benchmark_preference='continue_source_until_262_then_pi05',training_scope='One retrospective developer contrast, not a validated applicability rule or independent generalization.',limitations=['Source skill suffix availability differs intentionally.','Public action-chunk accounting was fixed between runs; wall time differences are confounded and not assigned to the phase decision.','Exact pixels/actions/robot state are verified; original complete simulator snapshots and RNG state were not saved.','Only one matched context and realization; no confidence threshold, generalized rule, or model training inferred.','No future outcomes included in pre_decision_input; branch labels must stay outside inference payload.'],new_robot_calls=0,new_model_calls=0,trained_model=False)
 with out.open('x') as f:json.dump(value,f,indent=2)
 print(json.dumps({k:value[k] for k in ('status','prefix_actions','branches')}))
if __name__=='__main__':main()
