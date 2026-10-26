"""Execute the first learned change, then default pi; actual diagnostic, not a benchmark."""
import asyncio,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb
from action_expert_ranker_v1 import ExpertPairRanker
from replay_stack_progress import RUNTIME_SOURCE

async def main():
    p=Path('/project');o=Path('/check');spec=json.loads((o/'SPEC.json').read_text());source=p/spec['source']
    plan=json.loads((source/'PLAN.json').read_text());fork=plan['fork_step'];assert plan['initial'] in (36,37,38,39) and 0<=fork<400 and fork%10==0
    for key in ('source_request','source_response','source_default_response','default_outcome','full_learned_outcome','model'):
        assert hashlib.sha256((p/plan[key]).read_bytes()).hexdigest()==plan[key+'_sha256']
    expected=unpackb((p/plan['source_request']).read_bytes());cached=np.load(source/'CANDIDATES.npz',allow_pickle=False)['candidates']
    pred=ExpertPairRanker(p/plan['model']);runtime=None;started=time.perf_counter()
    result=dict(status='RUNNING',initial=plan['initial'],fork_step=fork,rows=[],prefix_steps=fork,actor_inferences=0,
        parameter_updates=0,persistent_updates=0,full_agent_chain=False,scope='First learned-change causal development diagnostic, not held-out method speed',plan=plan)
    def save():(o/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    try:
        module=importlib.util.spec_from_file_location('first_choice_runtime',RUNTIME_SOURCE);mod=importlib.util.module_from_spec(module);module.loader.exec_module(mod)
        runtime=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=plan['initial'],seed=7,
            camera_height=256,camera_width=256,max_steps=400,max_chunk_size=50,num_steps_wait=10,control_mode='relative',record_dir=None))
        sid=f'first_choice_i{plan["initial"]}';runtime.reset(dict(session_id=sid))
        def inputs():return {'observation/image':np.asarray(runtime._last_obs['agentview_image']).copy(),
            'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
            'observation/state':mod._libero_state(runtime._last_obs).copy(),'prompt':runtime.language}
        for i,actions in enumerate(plan['prefix']):runtime.action_chunk(dict(actions=actions,chunk_id=f'prefix{i}'))
        assert runtime.step_idx==fork and not runtime.success
        current=inputs();result['fork_inputs_exact']={k:bool(np.array_equal(current[k],expected[k])) for k in current if k!='prompt'}
        assert all(result['fork_inputs_exact'].values()) and current['prompt']==expected['prompt']
        result['prefix_reconstruction_seconds']=time.perf_counter()-started
        async with websockets.connect(spec['endpoint'],compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
            metadata=unpackb(await ws.recv());assert metadata['branch_baseline_protocol']=='same_future_noise_v2';result['policy_metadata']=metadata
            while runtime.step_idx<400 and not runtime.success:
                before=runtime.step_idx;idx=before//10;request=inputs()
                request['_branch_baseline_v2']=dict(session_id=sid,step=before,noise_base=plan['noise_base'],fork_step=fork)
                tick=time.perf_counter();await ws.send(packb(request));raw=await ws.recv();rpc=time.perf_counter()-tick;assert not isinstance(raw,str),raw
                reply=unpackb(raw);candidates=np.asarray(reply['candidates'],np.float32);count=2 if before==fork else 1
                assert candidates.shape==(count,10,7) and np.isfinite(candidates).all()
                for j in range(count):
                    noise=np.random.default_rng(np.random.SeedSequence([plan['noise_base'],idx,j])).standard_normal((10,32)).astype(np.float32)
                    assert reply['branch_baseline']['noise_sha256'][j]==hashlib.sha256(noise.tobytes()).hexdigest()
                if before==fork:
                    result['cross_host_original_candidates_exact']=bool(np.array_equal(candidates,cached));assert result['cross_host_original_candidates_exact']
                    source_reply=unpackb((p/plan['source_response']).read_bytes())
                    assert np.array_equal(reply['action_expert_tokens'],source_reply['action_expert_tokens'])
                    selected,rank=pred.choose(reply['action_expert_tokens'])
                    assert abs(rank['pair_preference_score']-plan['model_scores']['pair_preference_score'])<1e-9
                    assert selected==1;result['first_choice_model_scores']=rank
                else:selected=0
                actions=candidates[selected];(o/f'{idx:03d}_request.msgpack').write_bytes(packb(request));(o/f'{idx:03d}_response.msgpack').write_bytes(raw)
                np.savez_compressed(o/f'{idx:03d}_command.npz',actions=actions)
                tick=time.perf_counter();runtime.action_chunk(dict(actions=actions.tolist(),chunk_id=f'first_choice{idx}'));env=time.perf_counter()-tick
                result['actor_inferences']+=count;result['rows'].append(dict(before=before,after=runtime.step_idx,selected_index=selected,candidates=count,rpc_seconds=rpc,environment_seconds=env));save()
        result.update(status='COMPLETED',native_success=bool(runtime.env.check_success()),physical_steps=runtime.step_idx,
            remaining_actions=runtime.step_idx-fork,remaining_control_requests=len(result['rows']))
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        if runtime is not None:runtime.close()
        result['seconds_including_close']=time.perf_counter()-started;save()
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','plan','policy_metadata')}),flush=True)
if __name__=='__main__':asyncio.run(main())
