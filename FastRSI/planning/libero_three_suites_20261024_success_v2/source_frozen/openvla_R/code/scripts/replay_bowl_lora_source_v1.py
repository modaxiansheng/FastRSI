"""Exact-action offline diagnosis of completed source demo0 arms, no model calls."""
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from replay_stack_progress import bind_goal_predicates,facts,summarize_rows,RUNTIME_SOURCE
from stage_trace import png_bytes

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['original','adapted'],required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.demo=Path('/project/planning/bowl_behavior_adaptation_v1_20261009/exports/demo_0');args.camera_size=256
    source=Path('/project/runs')/('bowl_lora_source_'+args.arm+'_v1_20261009')/'demo_0'
    live=json.loads((source/'RESULT.json').read_text());assert live['status']=='COMPLETED'
    args.output.mkdir(exist_ok=False,parents=True);runtime=None;started=time.perf_counter()
    report=dict(status='RUNNING',arm=args.arm,model_calls=0,training_steps=0,persistent_updates=0,rows=[],scope='Privileged native predicates for offline diagnosis only, not online features or causal intervention labels')
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

        report['rows'].append(dict(step=0,**facts(runtime.env.env,runtime._last_obs,64,goals)))
        for row in live['rows']:
            index=row['chunk'];request=unpackb((source/('%03d_request.msgpack'%index)).read_bytes());response=unpackb((source/('%03d_response.msgpack'%index)).read_bytes())
            assert np.array_equal(mod._libero_state(runtime._last_obs),request['observation/state'])
            for camera,key in [('agentview_image','observation/image'),('robot0_eye_in_hand_image','observation/wrist_image')]:
                assert np.array_equal(runtime._last_obs[camera],request[key]),(index,camera)
            runtime.action_chunk(dict(actions=np.asarray(response['actions'])[:min(10,400-runtime.step_idx)].tolist(),chunk_id='replay-'+str(index)))
            assert runtime.step_idx==row['step_after'] and bool(runtime.success)==row['native_success']
            report['rows'].append(dict(step=runtime.step_idx,**facts(runtime.env.env,runtime._last_obs,64,goals)))
        for camera in ['agentview_image','robot0_eye_in_hand_image']:
            assert png_bytes(runtime._last_obs[camera])==(source/('final_'+camera+'.png')).read_bytes()
        report.update(status='EXACT_OBSERVATION_AND_ACTION_REPLAY',progress_summary=summarize_rows(report['rows']),native_success=bool(runtime.success),executed_actions=runtime.step_idx)
    except BaseException as error:report.update(status='FAILED',error=repr(error));raise
    finally:
        report['seconds_before_close']=time.perf_counter()-started;(args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
        if runtime is not None:runtime.close()
        report['seconds_including_close']=time.perf_counter()-started;(args.output/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],arm=args.arm,seconds=report['seconds_including_close'])))
if __name__=='__main__':main()
