"""Freeze a post-hoc developer action contrast and independent saved-input audit."""
import hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.boundary_probe import artifact
NAME='boundary_consequence_v1_20261004'

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2)

def main():
    root=Path(__file__).resolve().parents[1];run=root/'runs'/NAME;run.mkdir(exist_ok=False)
    source=root/'runs/procedure_probe_v1_20261004/task9'
    for rel in ('MODEL_CONFIG.json','FLOW.json','PLAN.json'):shutil.copyfile(source/rel,run/rel)
    shutil.copytree(source/'initial_experience',run/'initial_experience')
    trace=source/'N/stream/stage_trace/net_t9_i15/TRACE.jsonl'
    costs=source/'N/stream/OS_COST_EVENTS.jsonl'
    path=lambda p:'/project/'+str(p.relative_to(root))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    spec=artifact(path(trace),digest(trace),path(costs),digest(costs))
    save(run/'BOUNDARY.json',spec)
    save(run/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',
        arm_modes={'N':'fixed10','F':'fixed10'},port_offset=-27000,gpu_bases={'N':0,'F':2},
        terminal_evidence_view=True,planning_entries={'N':'public_session_flow','F':'public_session_flow'},
        boundary_probe={'sha256':digest(run/'BOUNDARY.json'),'enabled':{'N':False,'F':True}},
        max_root_cases=2,max_native_attempts=3,max_campaign_seconds=1300,
        interpretation='Post-hoc development intervention at fixed step200 of the unmodified native prefix. '
                       'Not an online learned trigger or held-out evaluation; no intermediate teacher. '
                       'Prefix is re-executed and fully charged in both arms. Original total520 budget unchanged.'))
    entity=run/'ENTITY';entity.mkdir()
    shutil.copyfile(run/'MODEL_CONFIG.json',entity/'MODEL_CONFIG.json')
    inputs=root/'planning/entity_identity_v1_20261004'
    shutil.copytree(inputs,entity,dirs_exist_ok=True)
    for group,offset in [(run,0),(run/'N',-27000),(run/'F',-26900),(entity,0)]:
        group.mkdir(exist_ok=True)
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if offset:
            for p in (group/'code/scripts').glob('*.py'):
                text=p.read_text()
                for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+offset))
                p.write_text(text)
        save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):digest(p) for p in (group/'code').rglob('*') if p.is_file()})
    save(run/'SOURCE_MANIFEST.json',{str(p.relative_to(root)):digest(p) for p in (trace,costs)})
    print(run)

if __name__=='__main__':main()
