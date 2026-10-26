"""Freeze one paid capability comparison before any candidate execution."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.stall_recovery import SPEC, StallRecovery


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def main():
    root=Path(__file__).resolve().parents[1]
    name='stall_recovery_v1_20261004'; out=root/'runs'/name
    out.mkdir(exist_ok=False)
    sources={}; historical=[]
    corpus=root/'planning/decision_corpus_v1_20261004'
    read=lambda p:json.loads(p.read_text())
    # Existing complete inputs only; no new labels or future signals used online.
    meta=[json.loads(s) for s in (corpus/'METADATA.jsonl').read_text().splitlines()]
    inputs=[json.loads(s) for s in (corpus/'INPUTS.jsonl').read_text().splitlines()]
    outcomes={r['episode_id']:r for r in map(json.loads,(corpus/'EPISODE_OUTCOMES.jsonl').read_text().splitlines())}
    by={}
    for m,x in zip(meta, inputs):
        if x['model_input']:
            by.setdefault(m['episode_id'], []).append((m, x['model_input']))
    for episode, records in by.items():
        o=outcomes[episode]
        # Probe is H10/W10. Other windows are historical context, not pooled labels.
        if o['mode']!='fixed10':
            continue
        case=read(root/o['source']/'stream/RESULT.json')['cases'][0]['case']
        if case.get('mode')!='development':
            raise ValueError('No evaluation data')
        p=StallRecovery(case, False, SPEC);eligible=[]
        for m,x in records:
            _,e=p.choose(x['robot_state'],m['step'],case['id'],case['task_description'])
            if e['eligible']:eligible.append(e['step'])
        historical.append(dict(source=o['source'],native_success=o['native_success'],
                               first_eligible=eligible[0] if eligible else None,
                               eligible_steps=eligible,independent_instance=o['instance_group']))
    origin=root/'runs/short_exit_v1_20261003/pair0/N/stream/OS_COST_EVENTS.jsonl'
    sources[str(origin.relative_to(root))]=hashlib.sha256(origin.read_bytes()).hexdigest()
    old=root/'runs/prefix_memory_v1_20261004'
    for rel in ('MODEL_CONFIG.json','FLOW.json'):
        shutil.copyfile(old/rel,out/rel)
        sources[str((old/rel).relative_to(root))]=hashlib.sha256((old/rel).read_bytes()).hexdigest()
    shutil.copytree(old/'initial_experience',out/'initial_experience')
    for p in (old/'initial_experience').rglob('*'):
        if p.is_file():sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    artifact=dict(status='PROPOSED',spec=SPEC,
                  model_config_sha256=hashlib.sha256((out/'MODEL_CONFIG.json').read_bytes()).hexdigest(),
                  origin='Handwritten physical retreat candidate; no learned applicability claim.',
                  rationale='Visible open-gripper positional stationarity in paid failed task4/init31. No success/oracle input online.',
                  sources=sources)
    save(out/'STALL.json',artifact)
    goal='put the white mug on the left plate and put the yellow and white mug on the right plate'
    save(out/'PLAN.json',dict(cases=[dict(suite='libero_10',task_id=4,init_state_id=31,
         task_description=goal,mode='development',max_steps=520,native_experience_maintenance=True,
         previously_observed=True)]))
    protocol=dict(slot=0,diagnostic_trace=True,control_kind='feedback_window',
        arm_modes=dict(N='fixed10',F='fixed10'),port_offset=-30000,gpu_bases=dict(N=0,F=2),
        terminal_evidence_view=True,planning_entries=dict(N='public_session_flow',F='public_session_flow'),
        stall_recovery_probe=dict(artifact_sha256=hashlib.sha256((out/'STALL.json').read_bytes()).hexdigest(),enabled=dict(N=False,F=True)),
        maximum_campaign_seconds=1000,max_root_cases=2,max_steps_per_attempt=520,
        hypothesis='A short physical retreat exits a stalled policy state where instruction-only correction failed.',
        permissible_input='Current/past policy proprioception, initial robot position, original goal, elapsed steps. No object truth, future labels or simulation state.',
        action='Once only: 10 lift steps, 20 retreat steps toward observed start XY; zero orientation delta; open gripper; all within native budget.',
        no_inference_saving_claim='Native inference is paid even for replacement chunks; preserving RNG until intervention. Primary test is recovery outcome and remaining work.',
        minimum_success='Candidate native success plus semantic agreement and chain/admission; same initial state; control records retained. One success is feasibility only.',
        on_failure='Reject this physical candidate, no amplitude/threshold sweep on this initial state.',
        next_if_success='Freeze proposed recovery, compare retention/new development initial states before training or promotion.',
        evaluation='No final test data, no default change, no Clef training; native feedback isolation unchanged.')
    save(out/'PROTOCOL.json',protocol)
    save(out/'INPUT_MANIFEST.json',sources)
    planning=root/'planning'/name
    save(planning/'HISTORICAL_TRIGGER_AUDIT.json',dict(rows=historical,spec=SPEC,
        scope='Retrospective trigger coverage, not recovery effects or learned prediction quality'))
    save(planning/'PREDECLARED_PROTOCOL.json',protocol)
    for group,shift in ((out,0),(out/'N',-30000),(out/'F',-29900)):
        group.mkdir(exist_ok=True)
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(root/rel,group/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if shift:
            for p in (group/'code/scripts').glob('*.py'):
                text=p.read_text()
                for port in (49500,49502,49510,49520):text=text.replace(str(port),str(port+shift))
                p.write_text(text)
        save(group/'CODE_MANIFEST.json',{str(p.relative_to(group/'code')):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (group/'code').rglob('*') if p.is_file()})
    print(json.dumps(dict(campaign=str(out),historical=historical),indent=2))


if __name__=='__main__':
    main()
