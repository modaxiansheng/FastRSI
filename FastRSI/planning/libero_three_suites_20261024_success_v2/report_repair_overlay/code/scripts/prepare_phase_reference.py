"""Original-policy references on both phase-study contexts; no new candidate."""
import hashlib,json,shutil
from pathlib import Path
NAME='phase_reference_v1_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 root=Path(__file__).resolve().parents[1];source=root/'runs/source_completion_handoff_v1_20261005';out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 for rel,d in json.loads((source/'CODE_MANIFEST.json').read_text()).items():assert sha(source/'code'/rel)==d,rel
 shutil.copytree(source/'code',out/'code');jobs=[]
 for index,init in enumerate((4,5)):
  child=f'init{init}_native';folder=out/child;folder.mkdir()
  for rel in ('MODEL_CONFIG.json','FLOW.json','GOAL_SKILL.json'):shutil.copyfile(source/rel,folder/rel)
  shutil.copytree(source/'initial_experience',folder/'initial_experience')
  plan=json.loads((source/'PLAN.json').read_text());plan['cases'][0]['init_state_id']=init;save(folder/'PLAN.json',plan)
  shift=-22000+100*index;p=json.loads((source/'PROTOCOL.json').read_text());p.pop('absolute_deadline',None)
  p.update(port_offset=shift-100,gpu_bases={'F':index*2},goal_skill_probe={'enabled':{'F':False},'sha256':sha(folder/'GOAL_SKILL.json')},max_campaign_seconds=1200,interpretation='Original policy full-task reference; no reuse and no online learner. Follow-up after phase study; known developer contexts, never new heldout evidence.')
  save(folder/'PROTOCOL.json',p);group=folder/'F';group.mkdir();shutil.copytree(source/'code',group/'code')
  for f in (group/'code/scripts').glob('*.py'):
   s=f.read_text()
   for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+shift))
   f.write_text(s)
  save(group/'CODE_MANIFEST.json',{str(f.relative_to(group/'code')):sha(f) for f in (group/'code').rglob('*') if f.is_file()})
  jobs.append(dict(child=child,init=init,extent=0,gpus=[index*2,index*2+1],ports=[49500+shift+i for i in (0,2,10,20)],role='posthoc_native_reference'))
 save(out/'REGISTRATION.json',dict(jobs=jobs,max_seconds=1200,primary='native_primary_v1',scope='Both current contexts; diagnose source-prefix versus native capability. No selection by favorable result. No learner updates from this reference run.'))
 print(out)
if __name__=='__main__':main()
