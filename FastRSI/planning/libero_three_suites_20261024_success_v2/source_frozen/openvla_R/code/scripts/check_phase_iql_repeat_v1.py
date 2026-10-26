"""Predeclared feedback/frozen adaptive-budget comparison; no Agent, learning or hidden features."""
import argparse,asyncio,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb
from replay_stack_progress import bind_goal_predicates,RUNTIME_SOURCE
from stage_trace import png_bytes

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initial',type=int,required=True)
    ap.add_argument('--arm',choices=['single','frozen'],required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--endpoint',required=True);a=ap.parse_args()
    assert a.initial in (20,21,22,23)
    a.output.mkdir(exist_ok=False,parents=True);case='init%d_%s'%(a.initial,a.arm)
    started=time.perf_counter();runtime=None
    report=dict(status='RUNNING',case=case,init_state_id=a.initial,arm=a.arm,rows=[],
        training_steps=0,persistent_updates=0,complete_agent_comparison=False,
        scope='Candidate qualification from normal native task start; all outcomes development',total_task_budget=400)
    report['learned_selection']=a.arm in ('frozen','updated')
    report['sampling_noise_base']=8101000+a.initial
    def save():(a.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        spec=importlib.util.spec_from_file_location('native_runtime',RUNTIME_SOURCE)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        runtime=mod.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=a.initial,seed=7,
            camera_height=256,camera_width=256,max_steps=400,max_chunk_size=50,num_steps_wait=10,
            control_mode='relative',record_dir=None))
        runtime.reset({'session_id':case})
        report['goal_predicates']=bind_goal_predicates(runtime.env.env,64)
        report['language']=runtime.language
        assert runtime.language=='stack the right bowl on the left bowl and place them in the tray'
        assert not runtime.env.check_success()
        def inputs():return {'observation/image':np.asarray(runtime._last_obs['agentview_image']).copy(),
            'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
            'observation/state':mod._libero_state(runtime._last_obs).copy(),'prompt':runtime.language}
        initial=inputs();report['initial_input_sha256']={key:hashlib.sha256(np.ascontiguousarray(initial[key]).tobytes()).hexdigest() for key in initial if key!='prompt'}
        np.savez_compressed(a.output/'initial_input.npz',**initial)
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            (a.output/('initial_'+camera+'.png')).write_bytes(png_bytes(runtime._last_obs[camera]))
        async with websockets.connect(a.endpoint,compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
            metadata=unpackb(await ws.recv());report['policy_metadata']=metadata
            assert metadata['candidate_protocol']=='phase_iql_repeat.v1'
            report['value_model_sha256']=metadata.get(a.arm+'_model_sha256')
            index=0
            while runtime.step_idx<400 and not runtime.success:
                n=1 if a.arm=='single' else (2 if index%2==0 else 1) if a.arm=='mix' else -1
                request=inputs();request.update(_candidate_case=case,_candidate_chunk=index,_candidate_count=n,_remaining_budget=400-runtime.step_idx)
                tick=time.perf_counter();(a.output/('%03d_request.msgpack'%index)).write_bytes(packb(request));save_pre=time.perf_counter()-tick
                tick=time.perf_counter();await ws.send(packb(request));raw=await ws.recv();rpc=time.perf_counter()-tick
                assert not isinstance(raw,str),raw
                response=unpackb(raw);candidates=np.asarray(response['candidates'],dtype=np.float32);n=response['candidate_count']
                assert candidates.shape==(n,10,7) and np.isfinite(candidates).all()
                selected=response['selected_index'];choice_meta=response.get('choice_meta',{})
                proposal=candidates[selected];before=runtime.step_idx
                tick=time.perf_counter()
                np.savez_compressed(a.output/('%03d_command.npz'%index),planned_actions=proposal,
                    submitted_actions=proposal[:min(10,400-before)])
                save_pre+=time.perf_counter()-tick
                tick=time.perf_counter();runtime.action_chunk(dict(actions=proposal[:min(10,400-before)].tolist(),chunk_id=case+'-'+str(index)));env_seconds=time.perf_counter()-tick
                tick=time.perf_counter();(a.output/('%03d_response.msgpack'%index)).write_bytes(raw)
                for camera in ('agentview_image','robot0_eye_in_hand_image'):
                    (a.output/('%04d_%s.png'%(runtime.step_idx,camera))).write_bytes(png_bytes(runtime._last_obs[camera]))
                evidence=time.perf_counter()-tick+save_pre
                report['rows'].append(dict(chunk=index,step_before=before,step_after=runtime.step_idx,
                    native_success=bool(runtime.success),candidate_count=n,selected_index=selected,
                    rpc_seconds=rpc,environment_seconds=env_seconds,evidence_seconds=evidence,
                    candidate_seconds=response['candidate_seconds'],choice_meta=choice_meta,candidate_noise_sha256=response['candidate_noise_sha256']))
                index+=1;save();print(case,runtime.step_idx,runtime.success,flush=True)
        report.update(status='COMPLETED',native_success=bool(runtime.env.check_success()),executed_actions=runtime.step_idx,
            control_requests=len(report['rows']),candidate_inferences=sum(r['candidate_count'] for r in report['rows']),
            selected_indices=[r['selected_index'] for r in report['rows']])
    except BaseException as e:report.update(status='ERROR',error=repr(e));raise
    finally:
        report['seconds_before_close']=time.perf_counter()-started;save()
        if runtime is not None:runtime.close()
        report['seconds_including_close']=time.perf_counter()-started;save()
if __name__=='__main__':asyncio.run(main())
