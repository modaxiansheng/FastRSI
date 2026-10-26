"""Exact-action offline procedure audit for LIBERO-10 tasks 2 and 9.

The privileged facts in this output are diagnostics / supervision candidates.
This script does not call a model, publish experience, or modify policy inputs.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import time
from pathlib import Path

from stage_trace import png_bytes


RUNTIME_SOURCE = Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
EXPECTED_GOALS = {
    2: (('turnon', 'flat_stove_1'), ('on', 'moka_pot_1', 'flat_stove_1_cook_region')),
    9: (('in', 'white_yellow_mug_1', 'microwave_1_heating_region'), ('close', 'microwave_1')),
}


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def validate_trace(rows):
    """A complete root trace must start at reset and retain every boundary."""
    if not rows or rows[0].get('method') != 'reset' or rows[0].get('step') != 0:
        raise ValueError('Trace must begin with its original reset at step 0')
    session = rows[0].get('session_id')
    if not session:
        raise ValueError('Missing trace session identity')
    previous = -1
    for index, row in enumerate(rows):
        method = 'reset' if index == 0 else 'action_chunk'
        if row.get('method') != method or row.get('session_id') != session:
            raise ValueError(f'Unexpected method/session at trace row {index}')
        step = row.get('step')
        if not isinstance(step, int) or step <= previous or step > 520:
            raise ValueError(f'Invalid or non-increasing step at trace row {index}')
        if not isinstance(row.get('native_success'), bool):
            raise ValueError(f'Missing native success at trace row {index}')
        for camera in CAMERAS:
            digest = row.get('images', {}).get(f'{step:04d}_{camera}.png')
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f'Missing camera hash at trace row {index}: {camera}')
        if index:
            actions = row.get('actions')
            if not isinstance(actions, list) or not actions or any(
                not isinstance(action, list) or len(action) != 7 for action in actions
            ):
                raise ValueError(f'Invalid action chunk at trace row {index}')
            # The final chunk may stop early at native success / the step cap.
            delta = step - previous
            if delta > len(actions) or (delta < len(actions) and not (row['native_success'] or step == 520)):
                raise ValueError(f'Missing action boundary at trace row {index}')
        previous = step


def bind_goal_predicates(env, task):
    """Read the loaded native BDDL goal, then require the audited task binding."""
    goals = env.parsed_problem['goal_state']
    normalized = [tuple(str(value).lower() for value in goal) for goal in goals]
    if len(normalized) != len(EXPECTED_GOALS[task]) or set(normalized) != set(EXPECTED_GOALS[task]):
        raise ValueError(f'Native BDDL goal differs from task {task}: {goals!r}')
    return [list(goal) for goal in goals]


def sim_state_binding(env):
    """Hash every element of the reconstructed MuJoCo flattened state."""
    flattened = env.sim.get_state().flatten()
    blob = flattened.tobytes(order='C')
    return dict(sha256=sha256(blob), dtype=str(flattened.dtype),
                shape=list(flattened.shape), nbytes=len(blob),
                encoding='sim.get_state().flatten().tobytes(order=C)',
                scope='Reconstructed replay state; source trace did not save a state hash')


def facts(env, task, goals):
    values = [dict(predicate=goal, value=bool(env._eval_predicate(goal))) for goal in goals]
    by_name = {str(item['predicate'][0]).lower(): item['value'] for item in values}
    obj_name = 'moka_pot_1' if task == 2 else 'white_yellow_mug_1'
    obj = env.get_object(obj_name)
    gripper = env.robots[0].gripper
    result = dict(goal_predicates=values, native_grasp=bool(env._check_grasp(gripper, obj.contact_geoms)),
                  grasp_object=obj_name, gripper_contacts_object=bool(env.check_contact(gripper, obj)))
    if task == 2:
        result.update(native_stove_on=by_name['turnon'], native_pot_on_stove=by_name['on'])
    else:
        result.update(native_mug_inside=by_name['in'], native_microwave_closed=by_name['close'])
    return result


def source_definitions(env, task):
    from libero.libero.envs.predicates.base_predicates import Close, In, On, TurnOn
    entries = [('eval_predicate', env._eval_predicate), ('native_grasp', env._check_grasp),
               ('object_contact', env.check_contact)]
    if task == 2:
        stove = env.object_states_dict['flat_stove_1']
        region = env.object_states_dict['flat_stove_1_cook_region']
        entries += [('turnon_predicate', TurnOn), ('on_predicate', On),
                    ('turn_on', stove.turn_on), ('region_check_ontop', region.check_ontop)]
    else:
        microwave = env.object_states_dict['microwave_1']
        region = env.object_states_dict['microwave_1_heating_region']
        entries += [('in_predicate', In), ('close_predicate', Close),
                    ('region_contact', region.check_contact), ('region_contain', region.check_contain),
                    ('microwave_is_close', microwave.is_close)]
    result = {}
    for name, function in entries:
        source = inspect.getsource(function)
        source_file = inspect.getsourcefile(function)
        result[name] = dict(source=source, source_sha256=sha256(source.encode()), source_file=source_file,
                            file_sha256=sha256(Path(source_file).read_bytes()) if source_file else None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', type=int, choices=(2, 9), required=True)
    parser.add_argument('--init', dest='init_state_id', type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.init_state_id < 50:
        parser.error('--init must belong to the bound 50-initial-state task')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite replay receipt: {args.output}')
    raw = args.trace.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    start = time.time()
    report = dict(status='RUNNING', task=args.task, init_state_id=args.init_state_id,
                  source_trace=str(args.trace), source_trace_sha256=sha256(raw),
                  scope='Offline privileged diagnostics / supervision candidates only. Never online inputs, '
                        'automatic decision labels, accepted experience, or a learned skill.',
                  online_input_used=False, model_calls=0, persistent_learning_updates=0, rows=[])
    runtime = None
    try:
        validate_trace(rows)
        spec = importlib.util.spec_from_file_location('runtime', RUNTIME_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['runtime_source_sha256'] = sha256(RUNTIME_SOURCE.read_bytes())
        runtime = module.LiberoRealRuntime(dict(benchmark_name='libero_10', task_id=args.task,
            init_state_id=args.init_state_id, seed=7, max_steps=520, control_mode='relative'))
        for index, row in enumerate(rows):
            report['processing_trace_row'] = index
            if row['method'] == 'reset':
                runtime.reset({'session_id': row['session_id']})
            else:
                runtime.action_chunk({'actions': row['actions']})
            if runtime.step_idx != row['step']:
                raise RuntimeError(f"Replay step mismatch at row {index}: {runtime.step_idx} != {row['step']}")
            images = {}
            for camera in CAMERAS:
                name = f"{row['step']:04d}_{camera}.png"
                digest = sha256(png_bytes(runtime._last_obs[camera]))
                if digest != row['images'][name]:
                    raise RuntimeError(f'Replay image mismatch at step {row["step"]}: {camera}')
                images[name] = digest
            if bool(runtime.success) != row['native_success']:
                raise RuntimeError(f'Runtime success mismatch at step {row["step"]}')
            env = runtime.env.env
            if index == 0:
                goals = bind_goal_predicates(env, args.task)
                report['native_goal_predicates'] = goals
                report['task_description'] = runtime.language
                report['source_definitions'] = source_definitions(env, args.task)
            state = sim_state_binding(env)
            if index == 0:
                report['reset_sim_state'] = state
            current = facts(env, args.task, goals)
            overall = bool(runtime.env.check_success())
            if overall != all(item['value'] for item in current['goal_predicates']):
                raise RuntimeError(f'Goal conjunction disagrees with native success at step {row["step"]}')
            report['rows'].append(dict(trace_row=index, method=row['method'], step=row['step'],
                observation_id=row.get('observation_id'), session_id=row['session_id'],
                images=images, images_exact_match=True, sim_state=state,
                trace_native_success=row['native_success'], runtime_success=bool(runtime.success),
                overall_success=overall,
                gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(), **current))
        report.update(status='EXACT_IMAGE_REPLAY', all_trace_rows_replayed=len(report['rows']) == len(rows),
                      source_trace_rows=len(rows))
        report.pop('processing_trace_row', None)
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        raise
    finally:
        report['seconds'] = time.time() - start
        try:
            with args.output.open('x') as handle:
                json.dump(report, handle, indent=2)
        finally:
            if runtime is not None and runtime.env is not None:
                runtime.env.close()
    print(json.dumps(dict(status=report['status'], rows=len(report['rows']), seconds=report['seconds'])))


if __name__ == '__main__':
    main()
