"""Execute each stored candidate at the same observable state, then default pi."""
import asyncio,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb
from replay_stack_progress import RUNTIME_SOURCE

async def main():
    p=Path('/project');o=Path('/check');specification=json.loads((o/'SPEC.json').read_text())
    source=p/specification['source'];plan=json.loads((source/'PLAN.json').read_text());arm=specification['arm']
    assert arm in (0,1) and plan['fork_step']==10 and plan['max_steps']==400
    for key in ('request','response'):
        assert hashlib.sha256((p/plan['source_'+key]).read_bytes()).hexdigest()==plan['source_'+key+'_sha256']
    expected=unpackb((p/plan['source_request']).read_bytes())
    candidates=np.load(source/'CANDIDATES.npz',allow_pickle=False)['candidates']
    result=dict(status='RUNNING',initial=plan['initial'],arm=arm,rows=[],actor_inferences=0,physical_steps=0,
        prefix_steps=10,fork_step=10,parameter_updates=0,persistent_updates=0,full_agent_chain=False,
        scope='Same-state branch development supervision; not method deployment or held-out result.')
    runtime=None;started=time.perf_counter()
    def save():(o/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        module=importlib.util.spec_from_file_location('branch_runtime',RUNTIME_SOURCE)
        mod=importlib.util.module_from_spec(module);module.loader.exec_module(mod)
        runtime=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=plan['initial'],seed=7,
            camera_height=256,camera_width=256,max_steps=400,max_chunk_size=50,num_steps_wait=10,
            control_mode='relative',record_dir=None));sid=f'branch_i{plan["initial"]}_a{arm}'
        runtime.reset(dict(session_id=sid))
        def inputs():return {'observation/image':np.asarray(runtime._last_obs['agentview_image']).copy(),
            'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
            'observation/state':mod._libero_state(runtime._last_obs).copy(),'prompt':runtime.language}
        for i,actions in enumerate(plan['prefix']):runtime.action_chunk(dict(actions=actions,chunk_id=f'prefix{i}'))
        current=inputs();assert runtime.step_idx==10 and not runtime.success
        result['fork_inputs_exact']={k:bool(np.array_equal(current[k],expected[k])) for k in current if k!='prompt'}
        assert all(result['fork_inputs_exact'].values()) and current['prompt']==expected['prompt']
        tick=time.perf_counter();runtime.action_chunk(dict(actions=candidates[arm].tolist(),chunk_id='isolated_candidate'))
        result['rows'].append(dict(kind='isolated_candidate',before=10,after=runtime.step_idx,seconds=time.perf_counter()-tick,
            submitted_sha256=hashlib.sha256(candidates[arm].tobytes()).hexdigest()))
        async with websockets.connect(specification['endpoint'],compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
            metadata=unpackb(await ws.recv());assert metadata['branch_baseline_protocol']=='same_future_noise_v1'
            while runtime.step_idx<400 and not runtime.success:
                before=runtime.step_idx;request=inputs();request['_branch_baseline_v1']=dict(session_id=sid,step=before,noise_base=plan['noise_base'])
                tick=time.perf_counter();await ws.send(packb(request));raw=await ws.recv();rpc=time.perf_counter()-tick
                assert not isinstance(raw,str),raw
                reply=unpackb(raw);actions=np.asarray(reply['actions'],np.float32);assert actions.shape==(10,7) and np.isfinite(actions).all()
                idx=before//10;noise=np.random.default_rng(np.random.SeedSequence([plan['noise_base'],idx,0])).standard_normal((10,32)).astype(np.float32)
                assert reply['branch_baseline']['index']==idx and reply['branch_baseline']['noise_sha256']==hashlib.sha256(noise.tobytes()).hexdigest()
                (o/f'{idx:03d}_request.msgpack').write_bytes(packb(request));(o/f'{idx:03d}_response.msgpack').write_bytes(raw)
                np.savez_compressed(o/f'{idx:03d}_command.npz',actions=actions)
                tick=time.perf_counter();runtime.action_chunk(dict(actions=actions.tolist(),chunk_id=f'continuation{idx}'));env=time.perf_counter()-tick
                result['actor_inferences']+=1;result['rows'].append(dict(kind='default_pi',before=before,after=runtime.step_idx,rpc_seconds=rpc,environment_seconds=env));save()
        result.update(status='COMPLETED',native_success=bool(runtime.env.check_success()),physical_steps=runtime.step_idx,
            remaining_actions=runtime.step_idx-10,remaining_control_requests=len(result['rows']))
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        if runtime is not None:runtime.close()
        result['seconds_including_close']=time.perf_counter()-started;save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':asyncio.run(main())
