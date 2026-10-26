"""Offline exact-action diagnosis for the two bound LIBERO-90 stacking tasks.

No model calls, policy edits, accepted experience, or online privileged features.
A native grasp/contact reading is not a release or stability certificate.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import time

from stage_trace import png_bytes

RUNTIME_SOURCE = Path('/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')
BOWLS = ('akita_black_bowl_1', 'akita_black_bowl_2')
TASKS = {
    63: dict(language='stack the left bowl on the right bowl and place them in the tray',
             top=BOWLS[0], bottom=BOWLS[1]),
    64: dict(language='stack the right bowl on the left bowl and place them in the tray',
             top=BOWLS[1], bottom=BOWLS[0]),
}


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def validate_case(roster):
    if not isinstance(roster, list) or len(roster) != 1:
        raise ValueError('Exactly one bound root case is required')
    case = roster[0]
    if (case.get('suite') != 'libero_90' or case.get('task_id') not in TASKS
            or case.get('max_steps') != 400 or case.get('mode') != 'development'):
        raise ValueError('Only LIBERO-90 development tasks 63/64 with cap 400 are bound')
    task = TASKS[case['task_id']]
    if case.get('task_description') != task['language']:
        raise ValueError('Roster goal language differs from the bound full goal')
    if (type(case.get('init_state_id')) is not int or not 0 <= case['init_state_id'] < 50
            or case.get('seed', 7) != 7 or case.get('task_order_index', 0) != 0):
        raise ValueError('Unsupported initial state, seed, or task ordering')
    identity = case.get('id')
    if not isinstance(identity, str) or not identity or Path(identity).name != identity:
        raise ValueError('Missing or nonlocal case ID')
    return case


def validate_trace(rows):
    if not rows or rows[0].get('method') != 'reset' or rows[0].get('step') != 0:
        raise ValueError('Trace must start with its original reset at step 0')
    session = rows[0].get('session_id')
    if not isinstance(session, str) or not session:
        raise ValueError('Missing root session identity')
    previous = -1
    for index, row in enumerate(rows):
        if row.get('method') != ('reset' if index == 0 else 'action_chunk') or row.get('session_id') != session:
            raise ValueError(f'Unexpected method or session at row {index}')
        step = row.get('step')
        if type(step) is not int or step <= previous or not 0 <= step <= 400:
            raise ValueError(f'Invalid step at row {index}')
        if type(row.get('native_success')) is not bool:
            raise ValueError(f'Missing native success at row {index}')
        for camera in CAMERAS:
            digest = row.get('images', {}).get(f'{step:04d}_{camera}.png')
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError(f'Missing or invalid image hash at row {index}: {camera}')
        if index:
            if rows[index - 1]['native_success'] or previous == 400:
                raise ValueError('Action submitted after terminal trace boundary')
            actions = row.get('actions')
            if not isinstance(actions, list) or not actions:
                raise ValueError(f'Empty action chunk at row {index}')
            for action in actions:
                if (not isinstance(action, list) or len(action) != 7
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in action)):
                    raise ValueError(f'Invalid action vector at row {index}')
            delta = step - previous
            if delta > len(actions) or (delta < len(actions) and not (row['native_success'] or step == 400)):
                raise ValueError(f'Skipped action boundary at row {index}')
        previous = step


def bind_goal_predicates(env, task_id):
    task = TASKS[task_id]
    expected = {('on', task['top'], task['bottom']),
                ('in', task['bottom'], 'wooden_tray_1_contain_region')}
    goals = env.parsed_problem['goal_state']
    actual = [tuple(str(v).lower() for v in goal) for goal in goals]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'Loaded native goals differ from task {task_id}: {goals!r}')
    return [list(goal) for goal in goals]


def sim_state_binding(env):
    state = env.sim.get_state().flatten()
    blob = state.tobytes(order='C')
    return dict(sha256=sha256(blob), dtype=str(state.dtype), shape=list(state.shape),
                nbytes=len(blob), encoding='sim.get_state().flatten().tobytes(order=C)',
                scope='Reconstructed replay state; not an original saved simulator snapshot')


def source_definitions(env):
    from libero.libero.envs.predicates.base_predicates import In, On
    entries = [('eval_predicate', env._eval_predicate), ('native_grasp', env._check_grasp),
               ('object_contact', env.check_contact), ('on_predicate', On), ('in_predicate', In)]
    result = {}
    for name, fn in entries:
        source = inspect.getsource(fn)
        filename = inspect.getsourcefile(fn)
        result[name] = dict(source=source, source_sha256=sha256(source.encode()), source_file=filename,
                            file_sha256=sha256(Path(filename).read_bytes()) if filename else None)
    return result


def facts(env, observation, task_id, goals):
    task = TASKS[task_id]
    values = [dict(predicate=g, value=bool(env._eval_predicate(g))) for g in goals]
    gripper = env.robots[0].gripper
    eef = observation.get('robot0_eef_pos')
    eef_position = [float(v) for v in eef] if eef is not None else None
    records = {}
    for name in BOWLS:
        obj = env.get_object(name)
        body_id = env.sim.model.body_name2id(obj.root_body)
        position = [float(v) for v in env.sim.data.body_xpos[body_id]]
        quaternion = [float(v) for v in env.sim.data.body_xquat[body_id]]
        records[name] = dict(role='top' if name == task['top'] else 'bottom',
            native_grasp=bool(env._check_grasp(gripper, obj.contact_geoms)),
            gripper_contact=bool(env.check_contact(gripper, obj)),
            body_name=obj.root_body, world_position=position, world_quaternion_wxyz=quaternion,
            eef_to_body_origin_distance=math.dist(eef_position, position) if eef_position is not None else None,
            native_in_tray=bool(env._eval_predicate(['in', name, 'wooden_tray_1_contain_region'])),
            release_status='UNKNOWN', stable_support_status='UNKNOWN')
    top = env.get_object(task['top']); bottom = env.get_object(task['bottom'])
    return dict(goal_predicates=values,
                native_top_on_bottom=bool(env._eval_predicate(['on', task['top'], task['bottom']])),
                native_bottom_in_tray=records[task['bottom']]['native_in_tray'],
                bowls_contact=bool(env.check_contact(top, bottom)), bowls=records,
                eef_world_position=eef_position,
                eef_position_source='runtime._last_obs.robot0_eef_pos' if eef_position is not None else 'UNAVAILABLE',
                eef_to_object_distance_scope='Euclidean distance to body origin, not surface clearance or grasp quality')


def summarize_rows(rows):
    """Discrete boundary evidence only; no inferred events between observations."""
    result = dict(scope='Recorded root-episode boundaries only; no interpolation or release inference',
                  observation_count=len(rows), goal_predicates=[])
    if not rows:
        return result
    for index, entry in enumerate(rows[0]['goal_predicates']):
        states = [(row['step'], row['goal_predicates'][index]['value']) for row in rows]
        true_steps = [step for step, value in states if value]
        transitions = [dict(step=step, value=value) for (previous_step, previous_value), (step, value)
                       in zip(states, states[1:]) if value != previous_value]
        result['goal_predicates'].append(dict(predicate=entry['predicate'],
            first_satisfied_step=true_steps[0] if true_steps else None,
            last_satisfied_step=true_steps[-1] if true_steps else None,
            satisfied_boundary_count=len(true_steps), final_value=states[-1][1],
            observed_lost_after_satisfaction=any(not value and true_steps and step > true_steps[0]
                                                for step, value in states),
            transitions=transitions))
    result['bowls'] = {name: dict(
        native_grasp_true_steps=[r['step'] for r in rows if r['bowls'][name]['native_grasp']],
        gripper_contact_true_steps=[r['step'] for r in rows if r['bowls'][name]['gripper_contact']],
        first_world_position=rows[0]['bowls'][name]['world_position'],
        final_world_position=rows[-1]['bowls'][name]['world_position'],
        final_world_quaternion_wxyz=rows[-1]['bowls'][name]['world_quaternion_wxyz'],
        release_status='UNKNOWN', stable_support_status='UNKNOWN') for name in BOWLS}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite receipt: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    runtime = None
    report = dict(status='RUNNING', scope='Offline privileged diagnostic only; not online input, automatic '
                  'training labels, admitted experience, or a learned skill. Discrete native grasp/contact '
                  'readings do not prove release or stable support.', model_calls=0,
                  online_privileged_inputs=0, persistent_learning_updates=0,
                  root_scope='Only the recorded root episode is replayed; retries and full Agent costs are not inferred or counted here.', rows=[])
    try:
        roster_path = args.stream.parent / 'ROSTER.json'
        roster_raw = roster_path.read_bytes()
        case = validate_case(json.loads(roster_raw))
        trace_path = args.stream / 'stage_trace' / case['id'] / 'TRACE.jsonl'
        trace_raw = trace_path.read_bytes()
        rows = [json.loads(line) for line in trace_raw.splitlines()]
        validate_trace(rows)
        report.update(case=case, source_roster=str(roster_path), source_roster_sha256=sha256(roster_raw),
                      source_trace=str(trace_path), source_trace_sha256=sha256(trace_raw),
                      source_trace_rows=len(rows),
                      other_trace_paths=[str(path) for path in sorted((args.stream / 'stage_trace').glob('*/TRACE.jsonl')) if path != trace_path],
                      terminal_source_trace=bool(rows[-1]['native_success'] or rows[-1]['step'] == 400))
        # Check the saved image bytes as well as the regenerated image bytes below.
        for row in rows:
            for name, digest in row['images'].items():
                if Path(name).name != name or sha256((trace_path.parent / name).read_bytes()) != digest:
                    raise ValueError(f'Saved source image mismatch: {name}')
        spec = importlib.util.spec_from_file_location('stack_replay_runtime', RUNTIME_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report['runtime_source_sha256'] = sha256(RUNTIME_SOURCE.read_bytes())
        runtime = module.LiberoRealRuntime(dict(benchmark_name='libero_90', task_id=case['task_id'],
            init_state_id=case['init_state_id'], seed=7, max_steps=400, control_mode='relative'))
        for index, row in enumerate(rows):
            report['processing_trace_row'] = index
            if row['method'] == 'reset':
                runtime.reset({'session_id': row['session_id']})
            else:
                runtime.action_chunk({'actions': row['actions']})
            if runtime.step_idx != row['step']:
                raise RuntimeError(f'Replay step mismatch at row {index}')
            images = {}
            for camera in CAMERAS:
                name = f'{row["step"]:04d}_{camera}.png'
                digest = sha256(png_bytes(runtime._last_obs[camera]))
                if digest != row['images'][name]:
                    raise RuntimeError(f'Replay image mismatch at step {row["step"]}: {camera}')
                images[name] = digest
            if bool(runtime.success) != row['native_success']:
                raise RuntimeError(f'Replay success mismatch at step {row["step"]}')
            env = runtime.env.env
            if index == 0:
                if runtime.language != case['task_description']:
                    raise ValueError('Loaded runtime language differs from roster goal')
                goals = bind_goal_predicates(env, case['task_id'])
                report['native_goal_predicates'] = goals
                report['source_definitions'] = source_definitions(env)
            current = facts(env, runtime._last_obs, case['task_id'], goals)
            success = bool(runtime.env.check_success())
            if success != all(v['value'] for v in current['goal_predicates']):
                raise RuntimeError('Native success disagrees with goal conjunction')
            report['rows'].append(dict(trace_row=index, method=row['method'], step=row['step'],
                observation_id=row.get('observation_id'), session_id=row['session_id'],
                images=images, images_exact_match=True, sim_state=sim_state_binding(env),
                runtime_success=bool(runtime.success), trace_native_success=row['native_success'],
                overall_success=success,
                gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(), **current))
        report.update(status='EXACT_IMAGE_REPLAY', all_trace_rows_replayed=len(report['rows']) == len(rows),
                      progress_summary=summarize_rows(report['rows']))
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
