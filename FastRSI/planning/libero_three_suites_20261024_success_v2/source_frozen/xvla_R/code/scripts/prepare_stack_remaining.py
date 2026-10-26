"""One post-hoc operation probe; regenerate and pay for the original prefix."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.boundary_probe import stack_artifact
NAME='stack_remaining_v1_20261004'
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 source=root/'runs/stack_flow_v1_20261004/task63'
 for rel in ('MODEL_CONFIG.json','FLOW.json','PLAN.json'):shutil.copyfile(source/rel,out/rel)
 shutil.copytree(source/'initial_experience',out/'initial_experience')
 trace=source/'N/stream/stage_trace/net_t63_i0/TRACE.jsonl';cost=source/'N/stream/OS_COST_EVENTS.jsonl'
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 pp=lambda p:'/project/'+str(p.relative_to(root))
 save(out/'BOUNDARY.json',stack_artifact(pp(trace),digest(trace),pp(cost),digest(cost)))
 save(out/'SOURCE_MANIFEST.json',{str(p.relative_to(root)):digest(p) for p in [trace,cost]})
 save(out/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',arm_modes={'F':'fixed10'},
    port_offset=-26000,gpu_bases={'F':0},terminal_evidence_view=True,planning_entries={'F':'public_session_flow'},
    boundary_probe={'sha256':digest(out/'BOUNDARY.json'),'enabled':{'F':True}},
    max_root_cases=1,max_native_attempts=3,max_campaign_seconds=1100,
    interpretation='Single operation probe after the original task63 result. Reference prefix120 is regenerated, checked before submission and charged. Original full goal and400-step cap unchanged. No teacher, no learned trigger; no matched speed claim across batches.',
    selection='Observed RGB at120 depicts first bowl in tray and other bowl on table. Offline replay locates headroom but is not read by the online branch.',
    stopping='One precise remaining-action candidate. Do not search wording or timing variants after failure. Success requires new-instance checks before learned reuse.'))
 for group,offset in [(out,0),(out/'F',-25900)]:
  group.mkdir(exist_ok=True)
  for rel in ('scripts','src/fastrsi','tests','configs'):
   shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  if offset:
   for p in (group/'code/scripts').glob('*.py'):
    s=p.read_text()
    for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+offset))
    p.write_text(s)
  save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):digest(p) for p in (group/'code').rglob('*') if p.is_file()})
 print(out)
if __name__=='__main__':main()
