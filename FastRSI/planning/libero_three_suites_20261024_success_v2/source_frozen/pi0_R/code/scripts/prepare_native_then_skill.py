"""One evidence-driven, manually composed development candidate; no search."""
import hashlib,json,shutil
from pathlib import Path
NAME='native_then_skill_v1_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2)
def main():
 r=Path(__file__).resolve().parents[1];s=r/'runs/source_completion_handoff_v1_20261005';out=r/'runs'/NAME;out.mkdir(exist_ok=False);child=out/'init5_composed';child.mkdir()
 for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(s/rel,child/rel)
 shutil.copytree(s/'initial_experience',child/'initial_experience')
 skill=json.loads((s/'GOAL_SKILL.json').read_text());skill['start_step']=130;save(child/'GOAL_SKILL.json',skill)
 plan=json.loads((s/'PLAN.json').read_text());plan['cases'][0]['init_state_id']=5;save(child/'PLAN.json',plan)
 p=json.loads((s/'PROTOCOL.json').read_text());p.pop('absolute_deadline',None);shift=-21000
 p.update(port_offset=shift-100,gpu_bases={'F':0},goal_skill_probe={'enabled':{'F':True},'sha256':sha(child/'GOAL_SKILL.json')},max_campaign_seconds=1000,interpretation='Known init5: native steps0:130 then frozen source suffix130:262 then native. One manually composed candidate based on actual prior trajectories; not learned routing or independent test.')
 save(child/'PROTOCOL.json',p)
 for group,portshift in [(out,0),(child/'F',shift)]:
  group.mkdir(exist_ok=True);shutil.copytree(s/'code',group/'code')
  for rel in ('src/fastrsi/goal_skill.py','scripts/run_goal_skill_native_baseline.py','tests/test_goal_skill_entry.py'):
   shutil.copyfile(r/rel,group/'code'/rel)
  if portshift:
   for f in (group/'code/scripts').glob('*.py'):
    text=f.read_text()
    for old in (49500,49502,49510,49520):text=text.replace(str(old),str(old+portshift))
    f.write_text(text)
  save(group/'CODE_MANIFEST.json',{str(f.relative_to(group/'code')):sha(f) for f in (group/'code').rglob('*') if f.is_file()})
 save(out/'REGISTRATION.json',dict(jobs=[dict(child='init5_composed',init=5,extent=262,start_step=130,gpus=[0,1],ports=[28500,28502,28510,28520])],max_seconds=1000,
  rationale='At native phase130 a bowl is visibly in tray; source-only phase130 leaves both bowls on table. Reuse the observed working native prefix and test the stored source suffix as a real continuous execution.',
  evidence={str(p.relative_to(r)):sha(p) for p in [r/'runs/phase_reference_v1_20261005/init5_native/F/stream/stage_trace/net_t64_i5/0130_agentview_image.png',r/'runs/phase_choice_v1_20261005/init5_h262/F/stream/stage_trace/net_t64_i5/0130_agentview_image.png']},
  stopping='Exactly one composed candidate on known init5. No timing sweep, no learned-policy claim, no default change. Full original goal and verifier retained.'))
 print(out)
if __name__=='__main__':main()
