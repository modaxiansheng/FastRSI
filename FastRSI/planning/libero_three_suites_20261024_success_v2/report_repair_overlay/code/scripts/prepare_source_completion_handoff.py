"""Freeze one new-initial-state mechanism probe, no threshold search."""
import hashlib,json,shutil
from pathlib import Path
NAME='source_completion_handoff_v1_20261005'
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)
def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME;out.mkdir(exist_ok=False)
    source=root/'runs/action_memory_v1_20261004'
    for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,out/rel)
    shutil.copytree(source/'initial_experience',out/'initial_experience')
    skill=root/'planning/skill_completion_v1_20261005/SKILL.json';shutil.copyfile(skill,out/'GOAL_SKILL.json')
    shutil.copyfile(source/'PLAN.json',out/'PLAN.json')
    save(out/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',
        arm_modes={'N':'fixed10','F':'fixed10'},port_offset=-25000,gpu_bases={'N':0,'F':2},
        terminal_evidence_view=True,planning_entries={'N':'public_session_flow','F':'public_session_flow'},
        goal_skill_probe={'sha256':hashlib.sha256(skill.read_bytes()).hexdigest(),'enabled':{'N':False,'F':True}},
        max_root_cases_per_arm=1,max_native_attempts_per_arm=3,max_campaign_seconds=1500,
        interpretation='M1: previously seen developer init1. Fixed source-goal prefix through the source completion boundary262 followed by real pi05 in the same episode. Full public-optimized native Agent, not original unoptimized reproduction and not learned applicability.',
        stopping='One source-complete skill, then real VLA; no outcome-driven extent/pose/prompt search. Failures retained. No default promotion.',
        input_scope='Source control targets and current robot controller state; no source/target object pose or visual calibration. Fixed phase boundary is permitted for this manual mechanism intervention; not claimed as learned selector.',
        unchanged='Full original goal,400-step cap,H10,W10 except explicit final2 skill commands at260,necessary verification,Qwen/VLA weights,common initial experience. N/F isolated mutable experience.',
        rng_scope='Identical seed7 but F omits27 policy draws. Later policy draws differ by design; no fake inference to equalize RNG index.',
        accounting='Actual VLA,Qwen,skill requests,control updates,physics steps and complete runtime separate. Source acquisition/compilation/validation are inherited costs, not free or this turn wall time. Diagnostic capture is shared; no summing overlapping spans.'))
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
