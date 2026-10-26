"""Registered later-state alternative; factual default outcome remains separately sourced."""
import asyncio,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb
from replay_stack_progress import RUNTIME_SOURCE

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

async def main():
    p=Path('/project');o=Path('/check');spec=json.loads((o/'SPEC.json').read_text());ctx=spec['context'];fork=ctx['step']
    assert fork in (80,160) and ctx['initial'] in (17,18)
    for key in ('request','response'):
        assert sha(p/ctx[key])==ctx[key+'_sha256']
    assert sha(p/ctx['full_actual_baseline_outcome'])==ctx['outcome_sha256']
    expected=unpackb((p/ctx['request']).read_bytes());original=unpackb((p/ctx['response']).read_bytes())
    stream=(p/ctx['request']).parent.parent
    call=next(r for r in map(json.loads,(stream/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()) if r['step']==fork)
    trace=list(map(json.loads,(stream/'stage_trace'/call['session_id']/'TRACE.jsonl').read_text().splitlines()))
    prefix=[r for r in trace if r['method']=='action_chunk' and r['step']<=fork]
    baseline=json.loads((p/ctx['full_actual_baseline_outcome']).read_text())
    runtime=None;started=time.perf_counter()
    result=dict(status='RUNNING',initial=ctx['initial'],fork_step=fork,rows=[],actor_inferences=0,
        prefix_steps=fork,parameter_updates=0,persistent_updates=0,full_agent_chain=False,
        scope='Actual development action branch, not held-out method performance',context=ctx,
        baseline_outcome_reference_reused=True,baseline_outcome=baseline,
        reference_reproduction='planning/native_value_branch_audit_v1_20261010/RESULT.json')
    def save():(o/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        module=importlib.util.spec_from_file_location('branch_runtime',RUNTIME_SOURCE)
        mod=importlib.util.module_from_spec(module);module.loader.exec_module(mod)
        runtime=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=ctx['initial'],seed=7,
            camera_height=256,camera_width=256,max_steps=400,max_chunk_size=50,num_steps_wait=10,
            control_mode='relative',record_dir=None));sid=f'branch_v2_i{ctx["initial"]}_s{fork}'
        runtime.reset(dict(session_id=sid))
        def inputs():return {'observation/image':np.asarray(runtime._last_obs['agentview_image']).copy(),
            'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
            'observation/state':mod._libero_state(runtime._last_obs).copy(),'prompt':runtime.language}
        for i,r in enumerate(prefix):
            runtime.action_chunk(dict(actions=r['actions'],chunk_id=f'prefix{i}'))
            assert runtime.step_idx==r['step'] and not runtime.success
        assert runtime.step_idx==fork
        current=inputs();result['fork_inputs_exact']={k:bool(np.array_equal(current[k],expected[k])) for k in current if k!='prompt'}
        assert all(result['fork_inputs_exact'].values()) and current['prompt']==expected['prompt']
        result['prefix_reconstruction_seconds']=time.perf_counter()-started
        async with websockets.connect(spec['endpoint'],compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
            metadata=unpackb(await ws.recv());assert metadata['branch_baseline_protocol']=='same_future_noise_v2'
            result['policy_metadata']=metadata
            while runtime.step_idx<400 and not runtime.success:
                before=runtime.step_idx;idx=before//10;request=inputs()
                request['_branch_baseline_v2']=dict(session_id=sid,step=before,noise_base=call['noise_base'],fork_step=fork)
                tick=time.perf_counter();await ws.send(packb(request));raw=await ws.recv();rpc=time.perf_counter()-tick
                assert not isinstance(raw,str),raw
                reply=unpackb(raw);candidates=np.asarray(reply['candidates'],np.float32)
                count=2 if before==fork else 1;assert candidates.shape==(count,10,7) and np.isfinite(candidates).all()
                for j in range(count):
                    noise=np.random.default_rng(np.random.SeedSequence([call['noise_base'],idx,j])).standard_normal((10,32)).astype(np.float32)
                    assert reply['branch_baseline']['noise_sha256'][j]==hashlib.sha256(noise.tobytes()).hexdigest()
                if before==fork:
                    result['default_candidate_exact']=bool(np.array_equal(candidates[0],original['actions']))
                    assert result['default_candidate_exact'],'Do not reuse a reference with different actions'
                selected=1 if before==fork else 0;actions=candidates[selected]
                (o/f'{idx:03d}_request.msgpack').write_bytes(packb(request));(o/f'{idx:03d}_response.msgpack').write_bytes(raw)
                np.savez_compressed(o/f'{idx:03d}_command.npz',actions=actions)
                tick=time.perf_counter();runtime.action_chunk(dict(actions=actions.tolist(),chunk_id=f'branch{idx}'));env=time.perf_counter()-tick
                result['actor_inferences']+=count
                result['rows'].append(dict(before=before,after=runtime.step_idx,selected_index=selected,candidates=count,
                    rpc_seconds=rpc,environment_seconds=env));save()
        result.update(status='COMPLETED',native_success=bool(runtime.env.check_success()),physical_steps=runtime.step_idx,
            remaining_actions=runtime.step_idx-fork,remaining_control_requests=len(result['rows']))
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        if runtime is not None:runtime.close()
        result['seconds_including_close']=time.perf_counter()-started;save()
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','baseline_outcome','policy_metadata')}),flush=True)

if __name__=='__main__':asyncio.run(main())
