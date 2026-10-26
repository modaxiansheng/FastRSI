"""Instantiate registered init2/3 without changing M1 executable/model/experience."""
import hashlib,json,shutil
from pathlib import Path
NAME='m2_transfer_v1_20261005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,indent=2)
def main():
 root=Path(__file__).resolve().parents[1];source=root/'runs/source_completion_handoff_v1_20261005'
 registration=root/'planning/source_completion_handoff_v1_20261005/M2_REGISTERED_PLAN.json';reg=json.loads(registration.read_text())
 assert reg['preselected_new_initial_states']==[2,3] and sha(source/'GOAL_SKILL.json')==reg['candidate_sha256']
 for rel,d in json.loads((source/'CODE_MANIFEST.json').read_text()).items():assert sha(source/'code'/rel)==d,rel
 out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 shutil.copytree(source/'code',out/'code');shutil.copyfile(source/'CODE_MANIFEST.json',out/'CODE_MANIFEST.json');shutil.copyfile(registration,out/'REGISTRATION.json')
 for index,init in enumerate(reg['preselected_new_initial_states']):
  child=out/('init'+str(init));child.mkdir()
  for rel in ('MODEL_CONFIG.json','FLOW.json','GOAL_SKILL.json'):shutil.copyfile(source/rel,child/rel)
  shutil.copytree(source/'initial_experience',child/'initial_experience')
  plan=json.loads((source/'PLAN.json').read_text());plan['cases'][0]['init_state_id']=init;save(child/'PLAN.json',plan)
  protocol=json.loads((source/'PROTOCOL.json').read_text());protocol.pop('absolute_deadline',None)
  offset=-25000+index*200;protocol.update(port_offset=offset,gpu_bases={'N':index*4,'F':index*4+2},interpretation='M2 preregistered developer initial state '+str(init)+'. Fixed source complete skill. All executable code and candidate identical to M1 except service ports. No online candidate update or between-instance experience transfer.',max_campaign_seconds=1800)
  save(child/'PROTOCOL.json',protocol)
  for group,shift in [(child,0),(child/'N',offset),(child/'F',offset+100)]:
   group.mkdir(exist_ok=True);shutil.copytree(source/'code',group/'code')
   if shift:
    for p in (group/'code/scripts').glob('*.py'):
     text=p.read_text()
     for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
     p.write_text(text)
   save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):sha(p) for p in (group/'code').rglob('*') if p.is_file()})
  assert sha(child/'GOAL_SKILL.json')==reg['candidate_sha256']
 save(out/'DISPATCH_PLAN.json',dict(registration_sha256=sha(registration),children=['init2','init3'],arms=['N','F'],resources={'init2/N':[0,1],'init2/F':[2,3],'init3/N':[4,5],'init3/F':[6,7]},max_seconds=1800,runtime_code_source=str(source.relative_to(root)),runtime_code_manifest_sha256=sha(source/'CODE_MANIFEST.json'),prior_checks='M1 same runtime:30 server checks passed. No new algorithm code; verify artifacts/ports before dispatch.',claim_scope='New developer conditions, no final test or learned applicability claim. Parallel batch measures throughput, each group wall remains method cost.'))
 print(out)
if __name__=='__main__':main()
