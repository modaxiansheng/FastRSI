"""Bind pre-execution source/current observations for an unlabelled Clef probe.

No terminal images, target outcomes, hidden object positions or initial-state IDs
are sent to the model. IDs in the sidecar remain audit metadata only.
"""
import hashlib,json,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from fastrsi.decision_corpus import checked_image,project_input
from fastrsi.clef_adapter import request_for
NAME='goal_applicability_shadow_v1_20261005'
GOAL='stack the right bowl on the left bowl and place them in the tray'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,value):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(value,f,indent=2)

def boundary(stream):
 roster=read(stream.parent/'ROSTER.json');assert len(roster)==1
 case=roster[0];assert case['task_description']==GOAL
 sid=case['id'];trace=stream/'stage_trace'/sid/'TRACE.jsonl'
 reset=json.loads(trace.read_text().splitlines()[0]);assert reset['method']=='reset' and reset['step']==0
 log=stream/'OS_COST_EVENTS.jsonl'
 event=next(json.loads(l) for l in log.read_text().splitlines() if json.loads(l).get('event')=='actual_policy_input')
 assert event['case']==sid
 current,missing=project_input(event,'fixed10');assert not missing,missing
 images=[]
 for key,camera in [('observation/image','agentview_image'),('observation/wrist_image','robot0_eye_in_hand_image')]:
  name=f'0000_{camera}.png';p=trace.parent/name
  pixel_sha=checked_image(p.read_bytes(),reset['images'][name],event['array_bindings'][key])
  images.append(dict(path=str(p.resolve()),sha256=sha(p),pixel_sha256=pixel_sha,role=key))
 return current,images,dict(log=str(log.relative_to(R)),log_sha256=sha(log),trace=str(trace.relative_to(R)),trace_sha256=sha(trace),roster_sha256=sha(stream.parent/'ROSTER.json'))

def main():
 out=R/'planning'/NAME;out.mkdir(exist_ok=False);inputs=out/'requests';inputs.mkdir()
 source,source_images,source_binding=boundary(R/'runs/stack_flow_v1_20261004/task64/N/stream')
 candidate=R/'runs/source_completion_handoff_v1_20261005/GOAL_SKILL.json'
 assert sha(candidate)=='f8836351128078d6267784ef7664a82ae1d4aa11408c5012910271991e369034'
 paths=[R/'runs/source_completion_handoff_v1_20261005/N/stream']+[R/f'runs/m2_transfer_v1_20261005/init{i}/N/stream' for i in (2,3)]
 manifest=[];bindings=[]
 for number,stream in enumerate(paths):
  current,current_images,binding=boundary(stream)
  # Pair initial observations must be identical before any intervention.
  other,other_images,_=boundary(stream.parents[1]/'F/stream')
  assert current==other and [x['sha256'] for x in current_images]==[x['sha256'] for x in other_images]
  state=dict(goal=GOAL,observation_id='current_initial_observation',robot_state=current['robot_state'],
   current_execution_rule='At the initial observation choose whether to reuse the fixed source procedure or retain the original pi05 procedure.',
   candidate=dict(kind='Recorded robot controller target trajectory',steps=262,source_robot_state=source['robot_state'],
    parameter_binding='Absolute controller targets. Current robot feedback is used each step; no object pose retargeting.',
    handoff='After the recorded trajectory, pi05 resumes in the same physical episode with the original complete goal.',
    source_evidence='One historical source execution completed the native task. This does not prove applicability to the current scene.',
    current_applicability='Not yet established; infer only from the provided current/source observations, not from task identity alone.'),
   image_order=['Source initial front','Source initial wrist','Current initial front','Current initial wrist'],
   decision_consequences=dict(continue_='Reuse the recorded source trajectory, then resume pi05.',inspect='Request a specific missing applicability observation before deciding; evidence acquisition costs time.',escalate='Keep original pi05 execution and ask the existing high-level Agent to analyze unsupported reuse.'),
   instructions='Judge reuse of the candidate, not whether the robot should continue its original task. The source and current images are different roles. All images precede execution. Do not infer future success or calibrated confidence.')
  state['decision_consequences']['continue']=state['decision_consequences'].pop('continue_')
  record=dict(request=request_for(state),images=[{**x,'role':'source/'+x['role']} for x in source_images]+[{**x,'role':'current/'+x['role']} for x in current_images],label=None,label_status='UNLABELLED',mode='development')
  p=inputs/f'{number:03d}.json';save(p,record)
  manifest.append(dict(file=p.name,sha256=sha(p),images=4))
  bindings.append(dict(file=p.name,source=source_binding,current=binding,candidate_sha256=sha(candidate),paired_initial_equal=True,known_development_record=True))
 save(inputs/'MANIFEST.json',dict(status='BOUND_PRE_EXECUTION_INPUTS',records=manifest,actual_model_calls=0,training_labels=0))
 save(out/'INPUT_BINDING.json',dict(records=bindings,not_model_input=True,notes='Initial-state IDs and outcomes deliberately excluded from model payload. Source controller goals contain no object poses; no target terminal evidence is provided.'))
 save(out/'PLAN.json',dict(scope='Three real pre-execution paired observations; zero-shot unlabelled applicability shadow only',question='Does the existing decision model make differentiated, legal choices on actual source/current evidence?',success_definition='Complete bound inputs and legal choices; downstream usefulness and quality remain unproven without valid labels and executed selection',fixed_rule='Evidence is not calibrated for target applicability: defer reuse; do not assume every similar task is covered',model='clef-flash',revision='17f0b0ad64efb65d273590632833508766b2aae6',new_robot_runs=0,training_runs=0,control_enabled=False,no_expansion='No repeated prompt search or confidence threshold fitting on these three records'))
 run=R/'runs'/NAME;run.mkdir(exist_ok=False);code=run/'code'
 for name in ('clef_adapter.py','decision_validity.py','__init__.py'):
  p=code/'src/fastrsi'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/'src/fastrsi'/name,p)
 p=code/'scripts/run_clef_validity_probe.py';p.parent.mkdir(parents=True);shutil.copyfile(R/'scripts/run_clef_validity_probe.py',p)
 save(run/'CODE_MANIFEST.json',{str(p.relative_to(code)):sha(p) for p in code.rglob('*') if p.is_file()})
 print(json.dumps(dict(requests=len(manifest),images_per_request=4,labels=0)))
if __name__=='__main__':main()
