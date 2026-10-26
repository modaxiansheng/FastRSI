"""Data qualification: original policy on the three predeclared public-demo starts.
Not a complete-Agent comparison or a learning/held-out result.
"""
import argparse, asyncio, hashlib, importlib.util, json, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb, unpackb
from replay_stack_progress import bind_goal_predicates, RUNTIME_SOURCE
from stage_trace import png_bytes

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--demo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--endpoint',required=True)
    args=parser.parse_args();args.camera_size=256
    args.output.mkdir(exist_ok=False,parents=True)
    case=args.demo.name
    assert case in ('demo_0','demo_7','demo_25')
    started=time.perf_counter();runtime=None
    report=dict(status='RUNNING',case=case,rows=[],persistent_updates=0,
        scope='Source behavior qualification only; no Agent/Qwen, training or online hidden-state features',
        intermediate_state_restores=0)
    try:
        manifest = json.loads((args.demo / 'MANIFEST.json').read_text())
        for name, item in manifest['files'].items():
            assert sha(args.demo / name) == item['sha256'], f'Export SHA mismatch: {name}'
        data = np.load(args.demo / 'replay.npz', allow_pickle=False)
        actions = data['actions']
        assert actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
        assert len(actions) <= 400
        spec = importlib.util.spec_from_file_location('native_runtime', RUNTIME_SOURCE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        runtime = mod.LiberoRealRuntime(dict(benchmark_name='libero_90', task_id=64,
            init_state_id=0, seed=7, camera_height=args.camera_size, camera_width=args.camera_size,
            max_steps=400, max_chunk_size=50, num_steps_wait=10, control_mode='relative', record_dir=None))
        runtime.reset({'session_id': 'public-demo-' + args.demo.name})
        # Current LIBERO updates distractor meshes/names. Loading the historical
        # XML fails current Python object references. Keep the native model and
        # validate the entire initial-state layout before restoring any values.
        env = runtime.env.env
        native_xml = env.sim.model.get_xml()
        (args.output / 'native_model.xml').write_text(native_xml)
        original_tree = ET.fromstring((args.demo / 'model.xml').read_text())
        old_joints = [(j.get('name'), j.get('type', 'hinge'))
                      for j in original_tree.find('worldbody').iter('joint')]
        type_names = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}
        new_joints = [(str(name), type_names[int(env.sim.model.jnt_type[i])])
                      for i, name in enumerate(env.sim.model.joint_names)]
        aliases = {'new_salad_dressing_1_joint0': 'salad_dressing_1_joint0',
                   'new_chocolate_pudding_1_joint0': 'chocolate_pudding_1_joint0'}
        canonical = [(aliases.get(name, name), kind) for name, kind in new_joints]
        report['state_layout'] = dict(source_joints=old_joints, native_joints=new_joints,
            aliases_used={a: b for a, b in aliases.items() if a in dict(new_joints)},
            native_nq=int(env.sim.model.nq), native_nv=int(env.sim.model.nv),
            native_xml_sha256=hashlib.sha256(native_xml.encode()).hexdigest(),
            mode='Native XML retained; no historical geometry loaded')
        assert old_joints == canonical, 'Saved state joint ordering/types differ from native model'
        assert len(data['states'][0]) == 1 + env.sim.model.nq + env.sim.model.nv
        native_tree = ET.fromstring(native_xml)
        source_cameras = {c.get('name'): dict(c.attrib) for c in original_tree.find('worldbody').iter('camera')}
        native_cameras = {c.get('name'): dict(c.attrib) for c in native_tree.find('worldbody').iter('camera')}
        report['camera_xml_comparison'] = {name: dict(source=source_cameras[name], native=native_cameras[name])
                                         for name in ('agentview', 'robot0_eye_in_hand')}
        # First saved state is paired with the first action; the declared original
        # init_state may precede preprocessing's discarded leading idle actions.
        raw = runtime.env.set_init_state(data['states'][0])
        runtime.env.env.robots[0].controller.use_delta = True
        runtime._last_obs = runtime._format_obs(raw)
        runtime._initial_obs = runtime._last_obs
        runtime.step_idx = 0
        runtime.success = runtime.done = False
        runtime._total_reward = 0.0
        runtime._episode_chunks = []
        goals = bind_goal_predicates(runtime.env.env, 64)
        report.update(goal_predicates=goals, initial_native_success=bool(runtime.env.check_success()),
            restoration='Current native XML, verified joint layout, states[0] once; no settling after restore',
            declared_init_vs_saved_initial_max_abs=manifest['initial_attr_vs_first_saved_state_max_abs'],
            source_action_count=len(actions))
        assert not report['initial_native_success'], 'Already-successful initial state'
        controller = runtime.env.env.robots[0].controller
        report['controller'] = dict(type=type(controller).__name__, use_delta=bool(controller.use_delta),
            control_freq=runtime.env.env.control_freq,
            output_min=np.asarray(controller.output_min).tolist(),
            output_max=np.asarray(controller.output_max).tolist())
        assert report['controller']['control_freq'] == 20
        np.testing.assert_allclose(controller.output_max, [.05, .05, .05, .5, .5, .5])

        def inputs():
            return {'observation/image':np.asarray(runtime._last_obs['agentview_image']).copy(),
                'observation/wrist_image':np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
                'observation/state':mod._libero_state(runtime._last_obs).copy(),
                'prompt':'stack the right bowl on the left bowl and place them in the tray'}
        initial=inputs()
        reference_path=Path('/project/runs/bowl_demo_collection_v1_20261009')/case/'samples/0000.npz'
        reference=np.load(reference_path,allow_pickle=False)
        report['initial_input_exact']={k:bool(np.array_equal(initial[k],reference[k])) for k in initial if k!='prompt'}
        report['source_reference_sha256']=sha(reference_path)
        assert all(report['initial_input_exact'].values()), 'Initial policy inputs differ from qualified source replay'
        np.savez_compressed(args.output/'initial_input.npz',**initial)
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            (args.output/('initial_'+camera+'.png')).write_bytes(png_bytes(runtime._last_obs[camera]))
        async with websockets.connect(args.endpoint,compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
            metadata=unpackb(await ws.recv());report['policy_metadata']=metadata
            assert metadata['checkpoint_dir']=='/model' and metadata['torch_compile_disabled'] and metadata['torch_seed']==7
            index=0
            while runtime.step_idx<400 and not runtime.success:
                request=inputs();request['_source_case']=case;request['_source_chunk']=index
                (args.output/('%03d_request.msgpack'%index)).write_bytes(packb(request))
                tick=time.perf_counter();await ws.send(packb(request));response_raw=await ws.recv()
                if isinstance(response_raw,str):raise RuntimeError(response_raw)
                (args.output/('%03d_response.msgpack'%index)).write_bytes(response_raw)
                response=unpackb(response_raw);model_seconds=time.perf_counter()-tick
                proposed=np.asarray(response['actions'],dtype=np.float32)
                assert proposed.ndim==2 and proposed.shape[1]==7 and len(proposed)>=10 and np.isfinite(proposed).all()
                chunk=proposed[:min(10,400-runtime.step_idx)]
                before=runtime.step_idx;tick=time.perf_counter()
                runtime.action_chunk(dict(actions=chunk.tolist(),chunk_id=case+'-'+str(index)))
                report['rows'].append(dict(chunk=index,step_before=before,step_after=runtime.step_idx,
                    native_success=bool(runtime.success),model_rpc_seconds=model_seconds,
                    environment_seconds=time.perf_counter()-tick))
                index+=1
                print(case,runtime.step_idx,runtime.success,flush=True)
        report.update(status='COMPLETED',native_success=bool(runtime.env.check_success()),
            executed_actions=runtime.step_idx,vla_calls=len(report['rows']),control_requests=len(report['rows']))
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            (args.output/('final_'+camera+'.png')).write_bytes(png_bytes(runtime._last_obs[camera]))
    except BaseException as error:
        report.update(status='ERROR',error=repr(error));raise
    finally:
        report['seconds_before_close']=time.perf_counter()-started
        (args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
        if runtime is not None:runtime.close()
        report['seconds_including_close']=time.perf_counter()-started
        (args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':asyncio.run(main())
