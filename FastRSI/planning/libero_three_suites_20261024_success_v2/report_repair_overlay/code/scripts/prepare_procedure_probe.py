"""Freeze first two paired developer interventions; no test or learned claim."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.procedure_probe import artifact,GOALS
NAME='procedure_probe_v1_20261004'

def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)

def main():
 root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME;out.mkdir(exist_ok=False)
 source=root/'runs/prefix_memory_v1_20261004'
 pools=json.loads((root/'planning/representative_coverage_v1/RESERVED_SPLIT.json').read_text())['tasks']
 chosen=[(2,9),(9,15)]
 for i,(task,init) in enumerate(chosen):
  pool=next(v for v in pools if v['task_id']==task)
  assert init in pool['held_development_init_ids'] and init not in pool['reserved_final_method_test_init_ids']
  c=out/f'task{task}';c.mkdir()
  for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,c/rel)
  shutil.copytree(source/'initial_experience',c/'initial_experience')
  save(c/'PROCEDURE.json',artifact(task));sha=hashlib.sha256((c/'PROCEDURE.json').read_bytes()).hexdigest()
  save(c/'PLAN.json',{'cases':[dict(suite='libero_10',task_id=task,init_state_id=init,task_description=GOALS[task],mode='development',max_steps=520,native_experience_maintenance=True)]})
  shift=-29000+i*1000
  save(c/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',arm_modes={'N':'fixed10','F':'fixed10'},
    port_offset=shift,gpu_bases={'N':i*4,'F':i*4+2},terminal_evidence_view=True,planning_entries={'N':'public_session_flow','F':'public_session_flow'},
    procedure_probe={'sha256':sha,'enabled':{'N':False,'F':True}},max_root_cases=2,max_native_attempts=3,max_campaign_seconds=1600,
    interpretation='Full goal native versus unvalidated paid remaining-subgoal teacher procedure. No learned policy; teacher overhead belongs to F.'))
  for group,offset in [(c,0),(c/'N',shift),(c/'F',shift+100)]:
   group.mkdir(exist_ok=True)
   for rel in ('scripts','src/fastrsi','tests','configs'):
    shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
   if offset:
    for p in (group/'code/scripts').glob('*.py'):
     text=p.read_text()
     for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+offset))
     p.write_text(text)
   save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (group/'code').rglob('*') if p.is_file()})
 save(out/'PROTOCOL.json',dict(cases=chosen,root_executions=4,max_native_attempts=12,max_steps_per_attempt=520,max_seconds=1600,
   teacher_queries_per_root=4,teacher_steps=[100,200,300,400],teacher_timeout_seconds=45,
   selection='Two pre-existing held-development instances. Task 2 has no enforced order; task 9 microwave initially open. Neither is a held-out final test.',
   hypotheses='Remaining official subgoal may improve the action continuation; current full-goal native may dominate. No artificial baseline reasoning.',
   no_changes=['VLA weights','Qwen weights','success criteria','window10','native verifier/feedback'],
   next='If no useful effect, do not train on teacher proposals or repeat same procedure with new wording. Only real outcomes inform choice data.'))
 print(out)
if __name__=='__main__':main()
