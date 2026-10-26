"""Exact simulator gate before adding policy RNG reuse or live candidate trials."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.trial_prefix import execute_prefix, signature
from stage_trace import png_bytes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stream',type=Path,required=True)
    ap.add_argument('--boundary',type=int,required=True);ap.add_argument('--mode',choices=['full','reuse'],required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    case=json.loads((a.stream.parent/'ROSTER.json').read_text())[0]
    if case['mode']!='development':raise ValueError('Development evidence only')
    trace=a.stream/'stage_trace'/case['id']/'TRACE.jsonl';raw=trace.read_bytes();rows=list(map(json.loads,raw.splitlines()))
    spec=importlib.util.spec_from_file_location('native_target','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rt=mod.LiberoRealRuntime(dict(benchmark_name=case['suite'],task_id=case['task_id'],init_state_id=case['init_state_id'],
                                seed=7,max_steps=case['max_steps'],control_mode='relative'))
    result=dict(status='RUNNING',mode=a.mode,boundary=a.boundary,rows=[],source_sha256=hashlib.sha256(raw).hexdigest())
    start=time.perf_counter()
    def check(row, rendered):
        state=dict(step=rt.step_idx,physics=signature(rt.env.env.sim.get_state().flatten()),native_success=bool(rt.env.check_success()))
        if rt.step_idx!=row['step'] or bool(rt.success)!=row['native_success']:raise ValueError('Outcome/step mismatch')
        if rendered:
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                if hashlib.sha256(png_bytes(rt._last_obs[camera])).hexdigest()!=row['images'][f"{row['step']:04d}_{camera}.png"]:
                    raise ValueError('Image mismatch '+str((row['step'],camera)))
            state['policy_observation']={k:signature(v) for k,v in mod._policy_observation(rt._last_obs,case['task_description'],case['id']).items() if hasattr(v,'tobytes')}
        result['rows'].append(state)
    try:
        rt.reset({'session_id':case['id']});check(rows[0],True)
        result['reset_seconds']=time.perf_counter()-start
        tick=time.perf_counter()
        if a.mode=='reuse':
            result['prefix']=execute_prefix(rt,rows,a.boundary,on_boundary=lambda rt,row:check(row,False))
            # Replace the checkpoint's unrendered record with its exact refreshed input.
            result['rows'].pop();check(next(r for r in rows if r['step']==a.boundary),True)
        else:
            for row in rows[1:]:
                if row['step']>a.boundary:break
                rt.action_chunk({'actions':row['actions']});check(row,True)
        result['prefix_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        for row in rows:
            if row['step']<=a.boundary:continue
            rt.action_chunk({'actions':row['actions']});check(row,True)
        result.update(tail_seconds=time.perf_counter()-tick,status='EXACT_BOUNDARY_REPLAY',success=bool(rt.success))
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:
        result['seconds']=time.perf_counter()-start
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        rt.close()
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
