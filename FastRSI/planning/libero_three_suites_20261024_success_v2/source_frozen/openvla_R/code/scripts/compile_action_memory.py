"""Compile successful source observations/actions; never inspect target init1."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.action_memory import GOAL,LIMITS,ActionMemory,feature,state,open_motion
from fastrsi.decision_corpus import checked_image,png_pixels


def compile_memory(root):
    started=time.perf_counter()
    source=root/'runs/stack_flow_v1_20261004/task64/N/stream'
    result=json.loads((source/'RESULT.json').read_text())['cases'][0]
    if (result['case']['task_description']!=GOAL or result['case']['init_state_id']!=0 or
            not result['environment_success'] or result['chain_status']!='PASSED' or
            result['experience_admission']['status']!='ADMITTED' or
            len(result['sessions'])!=1 or [o['verdict'] for o in result['outcomes']]!=['success']):
        raise ValueError('Need an admitted, independently verified source execution')
    trace=source/'stage_trace/net_t64_i0/TRACE.jsonl'
    rows=list(map(json.loads,trace.read_bytes().splitlines()))
    events=[e for e in map(json.loads,(source/'OS_COST_EVENTS.jsonl').read_bytes().splitlines()) if e.get('event')=='actual_policy_input']
    if len(events)!=len(rows)-1:raise ValueError('Missing source input')
    entries=[]
    for before,after,event in zip(rows,rows[1:],events):
        if event['prompt']!=GOAL or event['case']!='net_t64_i0':raise ValueError('Source goal/owner mismatch')
        s=state(event['robot_state']); binding=event['array_bindings']['observation/state']
        if binding!={'shape':[8],'dtype':'float32','sha256':hashlib.sha256(s.tobytes()).hexdigest()}:
            raise ValueError('Reconstructed state does not match actual policy input')
        # The final submitted chunk was only partially executed: never reuse it.
        if after['step']-before['step']!=10 or not open_motion(s,after['actions']):continue
        entry=dict(source_step=before['step'],state=s.tolist(),actions=after['actions'])
        for name,key,camera in [('image','observation/image','agentview_image'),('wrist','observation/wrist_image','robot0_eye_in_hand_image')]:
            filename=f'{before["step"]:04d}_{camera}.png';blob=(trace.parent/filename).read_bytes()
            checked_image(blob,before['images'][filename],event['array_bindings'][key])
            pixels,shape=png_pixels(blob)
            entry[name]=feature(np.frombuffer(pixels,dtype=np.uint8).reshape(shape)).tolist()
        entries.append(entry)
    paths=[source/'RESULT.json',source/'OS_COST_EVENTS.jsonl',trace,source/'LAUNCH_pi05.json']
    artifact=dict(version=1,goal=GOAL,limits=LIMITS,entries=entries,
        source_case='libero90/task64/init0', target_scope='libero90/task64/init1 development only',
        sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Instance-based observed action memory; fixed heuristic eligibility, not learned applicability, causal skill validation or trained decision weights.',
        policy_rng='Skipping inference may change subsequent stochastic policy draws; no exact future-policy identity claim.',
        source_cost='Full paid task64 discovery execution and preceding project work remain in their original ledgers; not free training data.',
        compile_seconds=time.perf_counter()-started)
    ActionMemory(artifact)
    return artifact

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    out=root/'planning/action_memory_v1_20261004/MEMORY.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    value=compile_memory(root)
    with out.open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(dict(path=str(out),entries=len(value['entries']),source_steps=[e['source_step'] for e in value['entries']],compile_seconds=value['compile_seconds'])))
