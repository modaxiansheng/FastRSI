"""Two untouched held-development instances; no method or window comparison."""
import hashlib
import json
from pathlib import Path
import shutil

NAME='remaining_families_v1_20261004'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2)


def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs'/NAME
    out.mkdir(exist_ok=False)
    source=root/'runs/prefix_memory_v1_20261004'
    tasks=json.loads((root/'planning/representative_coverage_v1/RUNTIME_TASKS.json').read_text())['tasks']
    split=json.loads((root/'planning/representative_coverage_v1/RESERVED_SPLIT.json').read_text())['tasks']
    chosen=[(1,34),(6,15)]
    inputs={}
    for tid,iid in chosen:
        pool=next(x for x in split if x['task_id']==tid)
        assert iid in pool['held_development_init_ids'] and iid not in pool['reserved_final_method_test_init_ids']
    for rel in ['planning/representative_coverage_v1/RUNTIME_TASKS.json','planning/representative_coverage_v1/RESERVED_SPLIT.json']:
        inputs[rel]=hashlib.sha256((root/rel).read_bytes()).hexdigest()
    for index,(tid,iid) in enumerate(chosen):
        campaign=out/('task'+str(tid));campaign.mkdir()
        for rel in ('MODEL_CONFIG.json','FLOW.json'):
            shutil.copyfile(source/rel,campaign/rel);inputs[str((source/rel).relative_to(root))]=hashlib.sha256((source/rel).read_bytes()).hexdigest()
        shutil.copytree(source/'initial_experience',campaign/'initial_experience')
        goal=next(t['language'] for t in tasks if t['task_id']==tid)
        save(campaign/'PLAN.json',dict(cases=[dict(suite='libero_10',task_id=tid,init_state_id=iid,
            task_description=goal,mode='development',max_steps=520,native_experience_maintenance=True)]))
        shift=-29000+index*1000
        save(campaign/'PROTOCOL.json',dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',
            arm_modes=dict(N='fixed10'),port_offset=shift,gpu_bases=dict(N=index*2),
            terminal_evidence_view=True,planning_entries=dict(N='public_session_flow'),
            scope='Single default-configuration development execution. N is a worker label, not a comparative arm.',
            max_root_cases=1,max_native_attempts=3,max_steps_per_attempt=520,max_campaign_seconds=1500,
            verifier_prompt='unchanged Session prompt',public_fast='shared existing constructor',
            no_learning_comparison=True,no_parameter_training=True))
        for group,offset in [(campaign,0),(campaign/'N',shift)]:
            group.mkdir(exist_ok=True)
            for rel in ('scripts','src/fastrsi','tests','configs'):
                shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            if offset:
                for p in (group/'code/scripts').glob('*.py'):
                    text=p.read_text()
                    for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+offset))
                    p.write_text(text)
            save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (group/'code').rglob('*') if p.is_file()})
    for p in (source/'initial_experience').rglob('*'):
        if p.is_file():inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save(out/'INPUT_MANIFEST.json',inputs)
    save(out/'PROTOCOL.json',dict(cases=chosen,max_root_cases=2,max_total_native_attempts=6,
        max_campaign_seconds=1500,selection='Previously frozen held-development pool for task families 1 and 6, absent from the 20-pair learning corpus.',
        exclusion='No final-test pool; no instruction/window edits; no forced verifier replans.',
        purpose='Locate genuine remaining work under the strong default; preserve complete failed and successful trajectories.',
        interpretation='Throughput parallelism only; no speed, learning or benchmark-rate claim.',
        next='Do not repeat if both are already reliably handled. If a failure occurs, inspect actual remaining action before proposing a paid intervention.'))
    print(out)


if __name__=='__main__':main()
