"""Continuous action replay of a public demo in the unchanged native task.

Initial state / XML are offline diagnostics only. No policy, teacher, learning,
or intermediate simulator-state restoration is performed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from replay_stack_progress import bind_goal_predicates, facts, RUNTIME_SOURCE
from stage_trace import png_bytes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate_assets(xml):
    from libero.libero.utils.utils import postprocess_model_xml
    from libero.libero import get_libero_path
    tree = ET.fromstring(postprocess_model_xml(xml))
    mapping = []
    for element in tree.find('asset'):
        old = element.get('file')
        if not old:
            continue
        path = Path(old)
        # Public LIBERO-v1 demos retain the project's older chiliocosm name.
        if ('libero' in path.parts or 'chiliocosm' in path.parts) and 'assets' in path.parts:
            index = max(i for i, part in enumerate(path.parts) if part == 'assets')
            new = Path(get_libero_path('assets')).joinpath(*path.parts[index + 1:])
            element.set('file', str(new))
            mapping.append(dict(original=old, local=str(new)))
            path = new
        if not path.is_file():
            raise FileNotFoundError(f'Unresolved demonstration asset: {path}')
    return ET.tostring(tree, encoding='unicode'), mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--camera-size', type=int, choices=(128, 256), default=256)
    parser.add_argument('--observation-only', action='store_true')
    parser.add_argument('--limit-actions', type=int, default=400)
    parser.add_argument('--collect-stride', type=int, default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert 1 <= args.limit_actions <= 400
    assert args.collect_stride >= 0
    start = time.perf_counter()
    report = dict(status='RUNNING', demo=args.demo.name, camera_size=args.camera_size,
        scope='Public demonstration compatibility, not policy improvement or held-out evaluation',
        model_calls=0, training_steps=0, persistent_updates=0, intermediate_state_restores=0,
        rows=[], script_sha256=sha(Path(__file__)), runtime_source_sha256=sha(RUNTIME_SOURCE))
    runtime = None
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

        def snapshot(label):
            for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                (args.output / f'{label}_{camera}.png').write_bytes(png_bytes(runtime._last_obs[camera]))

        snapshot('initial')
        initial_state = mod._libero_state(runtime._last_obs)
        report['initial_public_state'] = initial_state.tolist()
        report['initial_reference_public_state'] = np.concatenate((data['ee_pos'][0], data['ee_ori'][0],
                                                                   data['gripper_states'][0])).tolist()
        def image_comparison(reference_slot, label):
            comparisons = {}
            for camera, key in (('agentview_image', 'agentview_rgb'),
                                ('robot0_eye_in_hand_image', 'eye_in_hand_rgb')):
                reference = data[key][reference_slot]
                raw_pixels = np.asarray(runtime._raw_obs[camera])
                candidates = dict(raw=raw_pixels, flip_vertical=raw_pixels[::-1],
                                  flip_horizontal=raw_pixels[:, ::-1], rotate_180=raw_pixels[::-1, ::-1])
                comparisons[camera] = {k: float(np.mean(np.abs(v.astype(float) - reference)))
                                       for k, v in candidates.items()}
                (args.output / f'{label}_reference_{camera}.png').write_bytes(png_bytes(reference))
            report[label + '_source_image_orientation_mae'] = comparisons
        if args.camera_size == 128:
            image_comparison(0, 'pre_action')
        if not args.observation_only:
            samples = []
            if args.collect_stride:
                (args.output / 'samples').mkdir()
            for index, action in enumerate(actions[:args.limit_actions]):
                before = mod._libero_state(runtime._last_obs)
                if args.collect_stride and index % args.collect_stride == 0 and index + 10 <= len(actions):
                    assert runtime.step_idx == index, 'Observation/action time mismatch'
                    path = args.output / 'samples' / f'{index:04d}.npz'
                    np.savez_compressed(path, **{
                        'observation/image': np.asarray(runtime._last_obs['agentview_image']).copy(),
                        'observation/wrist_image': np.asarray(runtime._last_obs['robot0_eye_in_hand_image']).copy(),
                        'observation/state': before.copy(),
                        'actions': actions[index:index + 10].astype(np.float32),
                        'prompt': np.asarray('stack the right bowl on the left bowl and place them in the tray')})
                    samples.append(dict(action_index=index, path='samples/' + path.name, sha256=sha(path),
                        observation_step=runtime.step_idx, label_start=index, label_end_exclusive=index+10))
                reference = np.concatenate((data['ee_pos'][index], data['ee_ori'][index],
                                            data['gripper_states'][index]))
                runtime.action_chunk(dict(actions=[action.tolist()], chunk_id=f'demo-{index}'))
                after = mod._libero_state(runtime._last_obs)
                report['rows'].append(dict(action_index=index, step=runtime.step_idx,
                    native_success=bool(runtime.env.check_success()),
                    eef_before_reference_l2=float(np.linalg.norm(before[:3] - reference[:3])),
                    eef_after_reference_l2=float(np.linalg.norm(after[:3] - reference[:3])),
                    state_before_reference_max_abs=float(np.max(np.abs(before - reference))),
                    state_after_reference_max_abs=float(np.max(np.abs(after - reference)))))
                if runtime.success:
                    break
            if args.collect_stride:
                report['samples'] = [dict(s, eligible=bool(runtime.success and s['label_end_exclusive'] <= runtime.step_idx))
                                     for s in samples]
                report['sample_alignment'] = 'Native PRE-action observation with next 10 actual source actions; no post-action input'
        snapshot('final')
        if args.camera_size == 128 and runtime.step_idx == 1:
            image_comparison(0, 'after_action_0')
        report.update(status='OBSERVATION_DIAGNOSTIC' if args.observation_only else
                      ('NATIVE_SUCCESS' if runtime.success else
                       ('PREFIX_DIAGNOSTIC' if args.limit_actions < len(actions) else 'NATIVE_FAILURE')),
            executed_actions=runtime.step_idx, native_success=bool(runtime.env.check_success()),
            action_protocol_requests=len(runtime._episode_chunks),
            final_facts=facts(runtime.env.env, runtime._last_obs, 64, goals),
            alignment_scope='Continuous replay divergence included; not a state-forced trajectory')
        if report['rows']:
            report['state_alignment'] = {key: dict(median=float(np.median([r[key] for r in report['rows']])),
                max=float(max(r[key] for r in report['rows']))) for key in
                ('eef_before_reference_l2', 'eef_after_reference_l2', 'state_before_reference_max_abs')}
    except BaseException as error:
        report.update(status='REPLAY_ERROR', error=repr(error))
        raise
    finally:
        report['elapsed_seconds_before_close'] = time.perf_counter() - start
        # Persist evidence before renderer close; successful process exit must
        # separately be checked by the host to qualify this result.
        (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
        if runtime is not None:
            runtime.close()
        report['elapsed_seconds_including_close'] = time.perf_counter() - start
        (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'demo', 'executed_actions', 'native_success',
                                             'elapsed_seconds_including_close')}))


if __name__ == '__main__':
    main()
