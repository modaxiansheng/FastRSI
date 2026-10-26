"""Offline, privileged diagnostic. Never imported by the online policy."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import time
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastrsi.retarget_geometry import rotation_vector
from replay_stack_progress import RUNTIME_SOURCE, validate_case, validate_trace, facts, bind_goal_predicates
from stage_trace import png_bytes


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    start = time.perf_counter()
    case = validate_case(json.loads((a.stream.parent / 'ROSTER.json').read_text()))
    trace_path = a.stream / 'stage_trace' / case['id'] / 'TRACE.jsonl'
    raw = trace_path.read_bytes()
    rows = [json.loads(v) for v in raw.splitlines()]
    validate_trace(rows)
    spec = importlib.util.spec_from_file_location('extract_runtime', RUNTIME_SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    runtime = m.LiberoRealRuntime(dict(benchmark_name='libero_90', task_id=case['task_id'],
        init_state_id=case['init_state_id'], seed=7, max_steps=400, control_mode='relative'))
    report = dict(status='RUNNING', scope='Offline privileged controller diagnostic only',
                  model_calls=0, persistent_learning_updates=0, case=case,
                  source_trace_sha256=sha(raw), runtime_sha256=sha(RUNTIME_SOURCE.read_bytes()),
                  controls=[], boundaries=[])
    try:
        runtime.reset({'session_id': case['id']})
        env = runtime.env.env
        goals = bind_goal_predicates(env, case['task_id'])
        c = env.robots[0].controller
        original = c.set_goal
        import robosuite.utils.transform_utils as T
        report['controller_source'] = inspect.getsource(type(c).set_goal)
        report['controller_file_sha256'] = sha(Path(inspect.getsourcefile(type(c))).read_bytes())
        config = {}
        for key in ('input_min', 'input_max', 'output_min', 'output_max', 'use_delta',
                    'use_ori', 'impedance_mode', 'position_limits', 'orientation_limits'):
            v = getattr(c, key)
            config[key] = v.tolist() if isinstance(v, np.ndarray) else v
        report['controller'] = config
        assert c.use_delta and c.use_ori and c.impedance_mode == 'fixed'
        assert c.position_limits is None and c.orientation_limits is None

        def logged(action, *args, **kwargs):
            result = original(action, *args, **kwargs)
            clipped = np.clip(action, c.input_min, c.input_max)
            position = (c.goal_pos - c.ee_pos)
            rotation = rotation_vector(c.goal_ori.dot(c.ee_ori_mat.T))
            scaled = np.r_[position, rotation]
            inverse = (scaled - c.action_output_transform) / c.action_scale + c.action_input_transform
            # Zero orientation commands preserve the previous goal. Record this
            # exception instead of claiming exact inverse reconstruction there.
            zero_ori = bool(np.all(np.asarray(action)[3:] == 0))
            error = float(np.max(np.abs(inverse[:3] - clipped[:3])))
            if not zero_ori:
                error = max(error, float(np.max(np.abs(inverse[3:] - clipped[3:]))))
            report['controls'].append(dict(step=runtime.step_idx,
                controller_action=np.asarray(action).tolist(), clipped_action=clipped.tolist(),
                eef_pos=c.ee_pos.tolist(), eef_rot=c.ee_ori_mat.tolist(),
                target_pos=c.goal_pos.tolist(), target_rot=c.goal_ori.tolist(),
                reconstructed_action=inverse.tolist(), inversion_error=error,
                zero_ori_preserves_goal=zero_ori))
            return result

        c.set_goal = logged
        for index, row in enumerate(rows):
            if index:
                before = len(report['controls'])
                runtime.action_chunk({'actions': row['actions']})
                generated = report['controls'][before:]
                assert len(generated) == row['step'] - rows[index-1]['step']
                for control, action in zip(generated, row['actions']):
                    control['gripper_command'] = action[-1]
            assert runtime.step_idx == row['step'] and runtime.success == row['native_success']
            for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                name = f'{row["step"]:04d}_{camera}.png'
                assert sha((trace_path.parent/name).read_bytes()) == row['images'][name]
                assert sha(png_bytes(runtime._last_obs[camera])) == row['images'][name]
            report['boundaries'].append(dict(step=row['step'],
                **facts(env, runtime._last_obs, case['task_id'], goals)))
        maximum = max(v['inversion_error'] for v in report['controls'])
        report.update(status='EXACT_REPLAY', controls_count=len(report['controls']),
            maximum_inversion_error=maximum,
            zero_orientation_commands=sum(v['zero_ori_preserves_goal'] for v in report['controls']),
            inversion_check=maximum < 1e-4)
        assert report['inversion_check'], maximum
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report['seconds'] = time.perf_counter() - start
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as f:
            json.dump(report, f, indent=2)
        if runtime.env is not None:
            runtime.env.close()
    print(json.dumps({k: report[k] for k in ('status','controls_count','maximum_inversion_error','seconds')}))


if __name__ == '__main__':
    main()
