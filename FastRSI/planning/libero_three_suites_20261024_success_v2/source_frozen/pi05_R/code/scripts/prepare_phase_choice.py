"""Freeze four same-code, real phase-130 branch executions before outcomes."""
import hashlib,json,shutil
from pathlib import Path
NAME='phase_choice_v1_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 root=Path(__file__).resolve().parents[1];source=root/'runs/source_completion_handoff_v1_20261005'
 out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 for rel,d in json.loads((source/'CODE_MANIFEST.json').read_text()).items():assert sha(source/'code'/rel)==d,rel
 full=json.loads((source/'GOAL_SKILL.json').read_text())
 short=json.loads((root/'runs/goal_skill_handoff_v1_20261005/GOAL_SKILL.json').read_text())
 assert short['waypoints']==full['waypoints'][:130]
 shutil.copytree(source/'code',out/'code')
 jobs=[]
 for index,(init,extent) in enumerate([(4,130),(4,262),(5,130),(5,262)]):
  name=f'init{init}_h{extent}';folder=out/name;folder.mkdir()
  for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,folder/rel)
  shutil.copytree(source/'initial_experience',folder/'initial_experience')
  save(folder/'GOAL_SKILL.json',short if extent==130 else full)
  plan=json.loads((source/'PLAN.json').read_text());plan['cases'][0]['init_state_id']=init
  save(folder/'PLAN.json',plan)
  # Only F is launched. All four workers have the identical executable snapshot.
  shift=-24000+index*100;port=49500+shift
  protocol=json.loads((source/'PROTOCOL.json').read_text());protocol.pop('absolute_deadline',None)
  protocol.update(port_offset=shift-100,gpu_bases={'F':index*2},max_campaign_seconds=1200,
   interpretation='Same-code phase130 development contrast; init4 update, init5 held-out diagnostic. No online learner or default promotion.',
   goal_skill_probe={'enabled':{'F':True},'sha256':sha(folder/'GOAL_SKILL.json')})
  save(folder/'PROTOCOL.json',protocol)
  group=folder/'F';group.mkdir();shutil.copytree(source/'code',group/'code')
  for p in (group/'code/scripts').glob('*.py'):
   text=p.read_text()
   for old in (49500,49502,49510,49520):text=text.replace(str(old),str(old+shift))
   p.write_text(text)
  save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):sha(p) for p in (group/'code').rglob('*') if p.is_file()})
  jobs.append(dict(child=name,init=init,extent=extent,gpus=[index*2,index*2+1],ports=[port+a for a in (0,2,10,20)],role='update' if init==4 else 'heldout_diagnostic'))
 save(out/'REGISTRATION.json',dict(jobs=jobs,primary='native_primary_v1',seed_context=1,update_context=4,heldout_context=5,
  source_code_manifest_sha256=sha(source/'CODE_MANIFEST.json'),train_test='All branches of a context stay together. Heldout labels never train. Known init1 old-code cost excluded from cost fitting.',
  method_scope='One meaningful phase choice. Same full goal, models, execution caps, initial experience, necessary verifier and independent memories.',
  accounting='Complete worker/controller cost, VLA/Qwen requests, physical steps, source acquisition and training separate; parallel wall is throughput.',
  learning_plan='Frozen Clef features; fit external observed-outcome head on init1 then init1+4. Compare both on init5 and include fixed continue/fixed handoff. No future input or fabricated alternatives.',
  fit={'seed':17,'iterations':200,'learning_rate':0.02,'weight_decay':0.01},max_seconds=1200))
 print(out)
if __name__=='__main__':main()
