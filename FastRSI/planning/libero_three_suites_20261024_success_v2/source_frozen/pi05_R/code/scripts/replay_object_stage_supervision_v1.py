"""Exact developmental action replay; hidden facts stay in offline supervision.

No policy/model calls or current OS memory writes. Replay matches every saved
camera boundary before annotating concrete objects and release conditions.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

from stage_trace import png_bytes

GOAL = 'put the white mug on the left plate and put the yellow and white mug on the right plate'
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
RUNTIME = Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_source(result, rows, init):
    if result.get('status') != 'COMPLETED' or len(result.get('cases', [])) != 1:
        raise ValueError('Complete bound developmental run required')
    row = result['cases'][0]
    case = row['case']
    if (case.get('suite'), case.get('task_id'), case.get('init_state_id'),
            case.get('mode'), case.get('max_steps'), case.get('task_description')) != (
            'libero_10', 4, init, 'development', 520, GOAL):
        raise ValueError('Only registered developmental task4 source may supervise')
    if row.get('chain_status') != 'PASSED':
        raise ValueError('Source OS chain failed')
    sid = case['id']
    if not rows or rows[0].get('method') != 'reset' or rows[0].get('step') != 0:
        raise ValueError('Original reset boundary required')
    previous = -1
    for index, entry in enumerate(rows):
        step = entry.get('step')
        if (entry.get('session_id') != sid or type(step) is not int
                or not previous < step <= 520
                or entry.get('method') != ('reset' if index == 0 else 'action_chunk')
                or type(entry.get('native_success')) is not bool):
            raise ValueError('Malformed original action boundary')
        if index:
            actions = entry.get('actions')
            if (not isinstance(actions, list) or not actions or step - previous > len(actions)
                    or any(not isinstance(a, list) or len(a) != 7
                           or any(type(v) not in (float, int) or not math.isfinite(v) for v in a)
                           for a in actions)):
                raise ValueError('Invalid actual action buffer')
            if rows[index - 1]['native_success']:
                raise ValueError('Action after terminal boundary')
        for camera in CAMERAS:
            h = entry.get('images', {}).get(f'{step:04d}_{camera}.png')
            if not isinstance(h, str) or len(h) != 64:
                raise ValueError('Missing original image binding')
        previous = step
    return case


def static_roles(env):
    goals = env.parsed_problem['goal_state']
    # Static task identity, not a guess from object names or privileged poses.
    expected = {
        'white_mug': ['on', 'porcelain_mug_1', 'plate_1'],
        'yellow_white_mug': ['on', 'white_yellow_mug_1', 'plate_2'],
    }
    if (len(goals) != 2 or any(len(g) != 3 for g in goals)
            or {tuple(g) for g in goals} != {tuple(g) for g in expected.values()}):
        raise ValueError('Unexpected task4 object/target definitions')
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init', type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable replay output already exists')
    blob = args.trace.read_bytes()
    rows = [json.loads(line) for line in blob.splitlines()]
    result_blob = args.result.read_bytes()
    case = validate_source(json.loads(result_blob), rows, args.init)
    started = time.perf_counter()
    report = dict(status='RUNNING', source_trace_sha256=sha(blob),
                  source_result_sha256=sha(result_blob), case=case,
                  online_input_used=False, new_model_calls=0, new_policy_episodes=0,
                  scope='Exact replay of already executed developmental actions; privileged facts are supervision only.', rows=[])
    runtime = None
    prior_facts = {}
    try:
        spec = importlib.util.spec_from_file_location('object_stage_native_runtime', RUNTIME)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['runtime_source_sha256'] = sha(RUNTIME.read_bytes())
        runtime = module.LiberoRealRuntime(dict(benchmark_name='libero_10', task_id=4,
            init_state_id=args.init, seed=7, max_steps=520, control_mode='relative'))
        for index, entry in enumerate(rows):
            if index == 0:
                runtime.reset({'session_id': case['id']})
            else:
                runtime.action_chunk({'actions': entry['actions']})
            if runtime.step_idx != entry['step'] or bool(runtime.success) != entry['native_success']:
                raise RuntimeError('Actual replay boundary/result mismatch')
            for camera in CAMERAS:
                name = f'{entry["step"]:04d}_{camera}.png'
                if sha(png_bytes(runtime._last_obs[camera])) != entry['images'][name]:
                    raise RuntimeError('Replay pixels differ at ' + name)
            env = runtime.env.env
            if index == 0:
                if runtime.language != GOAL:
                    raise ValueError('Loaded original task differs')
                roles = static_roles(env)
                report['offline_goal_roles'] = roles
            # Quaternion conversion matches the public OpenPI adapter's robot8.
            from scipy.spatial.transform import Rotation
            obs = runtime._last_obs
            robot8 = (list(map(float, obs['robot0_eef_pos']))
                      + Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec().tolist()
                      + list(map(float, obs['robot0_gripper_qpos'])))
            if len(robot8) != 8 or not all(math.isfinite(v) for v in robot8):
                raise ValueError('Public proprioception malformed')
            current = {}
            for subject, predicate in roles.items():
                obj = env.get_object(predicate[1])
                satisfied = bool(env._eval_predicate(predicate))
                grasp = bool(env._check_grasp(env.robots[0].gripper, obj.contact_geoms))
                contact = bool(env.check_contact(env.robots[0].gripper, obj))
                qualified = satisfied and not grasp and not contact
                current[subject] = dict(native_subgoal=satisfied, native_grasp=grasp,
                    native_gripper_contact=contact, released_support_proxy=qualified,
                    eligible_to_skip=bool(qualified and prior_facts.get(subject, False)),
                    label_scope='native_subgoal_plus_release_two_observed_boundaries',
                    stability_certificate=False)
                prior_facts[subject] = qualified
            report['rows'].append(dict(step=entry['step'], observation_id=entry['observation_id'],
                images=entry['images'], robot8=robot8, offline_supervision=current,
                actual_native_success=bool(runtime.success), exact_images=True))
        report.update(status='EXACT_IMAGE_REPLAY', source_boundaries=len(rows),
            reconstructed_action_steps=rows[-1]['step'], training_permission='DEVELOPMENT_SUPERVISION_ONLY')
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        report['seconds'] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump(report, handle, indent=2)
        if runtime is not None and runtime.env is not None:
            runtime.env.close()
    print(json.dumps({k: report[k] for k in ('status', 'seconds', 'source_boundaries')}), flush=True)


if __name__ == '__main__':
    main()
