"""Replay paid actions, measure env work, and hash every step and visible boundary."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np
from stage_trace import png_bytes


def array_signature(value):
    arr=np.ascontiguousarray(value)
    return dict(shape=list(arr.shape),dtype=str(arr.dtype),sha256=hashlib.sha256(arr.tobytes()).hexdigest())


def cpu_stat():
    path=Path('/sys/fs/cgroup/cpu.stat')
    return path.read_text() if path.exists() else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stream',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    roster=json.loads((args.stream.parent/'ROSTER.json').read_text());assert len(roster)==1
    case=roster[0];assert case['mode']=='development' and case['suite']=='libero_10'
    trace=args.stream/'stage_trace'/case['id']/'TRACE.jsonl'
    raw=trace.read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
    inputs=[json.loads(x) for x in (args.stream/'OS_COST_EVENTS.jsonl').read_text().splitlines()
            if json.loads(x).get('event')=='actual_policy_input']
    assert len(inputs)==len(rows)-1
    spec=importlib.util.spec_from_file_location('runtime','/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rt=mod.LiberoRealRuntime(dict(benchmark_name=case['suite'],task_id=case['task_id'],
        init_state_id=case['init_state_id'],seed=7,max_steps=case['max_steps'],control_mode='relative'))
    start=time.perf_counter();result=dict(status='RUNNING',source_sha256=hashlib.sha256(raw).hexdigest(),
        lp_num_threads=os.environ.get('LP_NUM_THREADS'),openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'),cpu_count=os.cpu_count(),
        cpu_stat_before=cpu_stat(),steps=[],boundaries=[],timings={},
        scope='Offline fixed-action simulation replay; not new policy execution, learning or complete OS speed')
    try:
        t=time.perf_counter();rt._ensure_env();result['timings']['cold_environment_create']=time.perf_counter()-t
        original=rt.env.step
        def measured_step(action):
            t=time.perf_counter();value=original(action);seconds=time.perf_counter()-t
            obs,reward,done,_=value
            result['steps'].append(dict(index=len(result['steps']),seconds=seconds,
                raw_observation={k:array_signature(v) for k,v in sorted(obs.items()) if isinstance(v,np.ndarray)},
                physics=array_signature(rt.env.env.sim.get_state().flatten()),reward=float(reward),done=bool(done)))
            return value
        rt.env.step=measured_step
        for index,row in enumerate(rows):
            t=time.perf_counter()
            if row['method']=='reset':rt.reset({'session_id':row['session_id']})
            else:rt.action_chunk({'actions':row['actions']})
            seconds=time.perf_counter()-t
            result['timings'][row['method']]=result['timings'].get(row['method'],0)+seconds
            assert rt.step_idx==row['step']
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                assert hashlib.sha256(png_bytes(rt._last_obs[camera])).hexdigest()==row['images'][f"{row['step']:04d}_{camera}.png"]
            if index<len(inputs):
                policy=mod._policy_observation(rt._last_obs,case['task_description'],case['id'])
                for key,binding in inputs[index]['array_bindings'].items():
                    assert array_signature(policy[key])==binding,(index,key)
            result['boundaries'].append(dict(step=row['step'],images_match=True,
                policy_input_match=index<len(inputs),native_success=bool(rt.env.check_success())))
        try:
            from OpenGL.GL import glGetString,GL_RENDERER,GL_VERSION
            result['gl_renderer']=str(glGetString(GL_RENDERER));result['gl_version']=str(glGetString(GL_VERSION))
        except Exception as exc:result['gl_query_error']=repr(exc)
        result['process_threads']=len(list(Path('/proc/self/task').iterdir()))
        try:
            from threadpoolctl import threadpool_info
            result['threadpools']=threadpool_info()
        except ImportError:
            result['threadpools']='threadpoolctl unavailable; no dependency installed'
        result['native_steps']=rt.step_idx
        result['success']=bool(rt.env.check_success())
        result['timings']['env_step_sum']=sum(s['seconds'] for s in result['steps'])
        result['status']='EXACT_BOUNDARY_REPLAY'
    except BaseException as exc:
        result.update(status='FAILED',error=repr(exc));raise
    finally:
        result['seconds']=time.perf_counter()-start;result['cpu_stat_after']=cpu_stat()
        with args.output.open('x') as f:json.dump(result,f,indent=2)
        if rt.env is not None:rt.env.close()
    print(json.dumps({k:v for k,v in result.items() if k not in ('steps','boundaries')}),flush=True)


if __name__=='__main__':main()
