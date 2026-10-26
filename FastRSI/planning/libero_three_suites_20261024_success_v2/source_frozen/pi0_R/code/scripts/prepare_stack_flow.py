"""Freeze two natural ordered-role developer tasks, with the full goal intact."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.development_suite import STACK_GOALS,STACK_EXTENSION
NAME='stack_flow_v1_20261004'
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 source=root/'runs/prefix_memory_v1_20261004'
 for index,(tid,goal) in enumerate(STACK_GOALS.items()):
  c=out/f'task{tid}';c.mkdir()
  for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,c/rel)
  shutil.copytree(source/'initial_experience',c/'initial_experience')
  save(c/'PLAN.json',dict(cases=[dict(suite='libero_90',task_id=tid,init_state_id=0,task_description=goal,
      mode='development',max_steps=400,native_experience_maintenance=True,development_suite_extension=STACK_EXTENSION)]))
  shift=-27000+index*1000
  save(c/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',arm_modes={'N':'fixed10'},
    port_offset=shift,gpu_bases={'N':index*2},terminal_evidence_view=True,planning_entries={'N':'public_session_flow'},
    max_root_cases=1,max_native_attempts=3,max_campaign_seconds=1500,
    interpretation='N is a worker label, not a comparative arm. Original whole-goal execution; no forced stage switching.'))
 for group,offset in [(out,0)]+[(out/f'task{tid}',0) for tid in STACK_GOALS]+[(out/f'task{tid}'/'N',-27000+i*1000) for i,tid in enumerate(STACK_GOALS)]:
  group.mkdir(exist_ok=True)
  for rel in ('scripts','src/fastrsi','tests','configs'):
   shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  if offset:
   for p in (group/'code/scripts').glob('*.py'):
    s=p.read_text()
    for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+offset))
    p.write_text(s)
  save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (group/'code').rglob('*') if p.is_file()})
 save(out/'PROTOCOL.json',dict(cases=[[63,0],[64,0]],max_root_cases=2,max_total_native_attempts=6,max_campaign_seconds=1500,
    selection='Original LIBERO90 dual-bowl stacking and tray placement, role-reversed pair. First init selected before any outcome. Developer capability probe, not held-out evaluation.',
    purpose='Locate actual composition/order failures and viable existing actions. Preserve whole-goal policy as strong comparator.',
    unchanged=['original BDDL/init states','Qwen and VLA','whole goal','H10/W10','native trigger/replan/feedback','success standard'],
    next='Inspect actual trajectory before proposing any alternative. No automatic fixed-time split or positive labels from terminal success.',
    interpretation='Parallel throughput only. No comparative speed or learning claim. Init 0 is exposed developer data from this point.'))
 print(out)
if __name__=='__main__':main()
