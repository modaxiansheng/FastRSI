"""Freeze one new-initial-state mechanism probe, no threshold search."""
import hashlib,json,shutil
from pathlib import Path
NAME='action_memory_v1_20261004'
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)
def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME;out.mkdir(exist_ok=False)
    source=root/'runs/stack_flow_v1_20261004/task64'
    for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,out/rel)
    shutil.copytree(source/'initial_experience',out/'initial_experience')
    memory=root/'planning'/NAME/'MEMORY.json';shutil.copyfile(memory,out/'MEMORY.json')
    plan=json.loads((source/'PLAN.json').read_text());plan['cases'][0]['init_state_id']=1
    save(out/'PLAN.json',plan)
    save(out/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',
        arm_modes={'N':'fixed10','F':'fixed10'},port_offset=-25000,gpu_bases={'N':0,'F':2},
        terminal_evidence_view=True,planning_entries={'N':'public_session_flow','F':'public_session_flow'},
        action_memory_probe={'sha256':hashlib.sha256(memory.read_bytes()).hexdigest(),'enabled':{'N':False,'F':True}},
        max_root_cases_per_arm=1,max_native_attempts_per_arm=3,max_campaign_seconds=1100,
        interpretation='New developer init1 after source init0; mechanism feasibility only. N and F can access identical frozen source memory. F uses observed-action lookup; N invokes VLA. No claim of learned applicability outperforming fixed retrieval.',
        stopping='One fixed similarity rule and two real runs. No threshold/wording/init search after outcome.',
        input_scope='Current RGB and proprio only. No source or target hidden predicates, initial-state ID, elapsed step, future results or test labels as selector features.',
        unchanged='Full original task,400-step cap,H10,W10,necessary verification,Qwen/VLA weights. Every boundary obtains new feedback.',
        accounting='Actual VLA calls separate from memory lookup and total action decisions; include discovery, compilation, validation, fallbacks and failures. Do not sum overlapping spans.'))
    for group,offset in [(out,0),(out/'N',-25000),(out/'F',-24900)]:
        group.mkdir(exist_ok=True)
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if offset:
            for p in (group/'code/scripts').glob('*.py'):
                s=p.read_text()
                for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port+offset))
                p.write_text(s)
        save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (group/'code').rglob('*') if p.is_file()})
    print(out)
if __name__=='__main__':main()
