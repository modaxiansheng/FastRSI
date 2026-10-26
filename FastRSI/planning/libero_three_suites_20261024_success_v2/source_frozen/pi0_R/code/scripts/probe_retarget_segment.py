"""Execute one preregistered paired local probe using privileged object anchors.

This is a mechanical diagnostic, not a full Agent, learned policy, or cost win.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.retarget_geometry import retarget, rotation_vector
from replay_stack_progress import RUNTIME_SOURCE, facts, bind_goal_predicates, validate_trace
from stage_trace import png_bytes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--arm', choices=['recorded', 'retarget'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text())
    root = Path('/project')
    for rel, digest in plan['source_sha256'].items():
        assert sha(root/rel) == digest, rel
    source = json.loads((root/plan['source_controls']).read_text())
    target = json.loads((root/plan['target_controls']).read_text())
    assert source['status'] == target['status'] == 'EXACT_REPLAY_RECHECKED'
    assert source['case']['task_id'] == target['case']['task_id'] == 64
    start, end = plan['start_step'], plan['end_step']
    assert type(start) is int and type(end) is int and 0 <= start < end <= 262
    assert plan['anchor'] in ['akita_black_bowl_1', 'akita_black_bowl_2']
    original = [json.loads(v) for v in (root/plan['target_trace']).read_text().splitlines()]
    validate_trace(original)
    by_step = {r['step']:r for r in original}
    actions = [action for row in original[1:] for action in row['actions']]
    source_boundary = next(r for r in source['boundaries'] if r['step'] == start)
    source_anchor = source_boundary['bowls'][plan['anchor']]
    spec = importlib.util.spec_from_file_location('probe_runtime', RUNTIME_SOURCE)
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime = m.LiberoRealRuntime(dict(benchmark_name='libero_90', task_id=64,
        init_state_id=target['case']['init_state_id'], seed=7, max_steps=400, control_mode='relative'))
    report = dict(status='RUNNING', arm=a.arm, plan_sha256=sha(a.plan),
        scope='Privileged fixed local mechanical comparison; no learned model or full Agent claim',
        model_calls=0, persistent_learning_updates=0, rows=[], actions=[], prefix_image_checks=0)
    started = time.perf_counter()
    try:
        import robosuite.utils.transform_utils as T
        runtime.reset({'session_id': target['case']['id']})
        env = runtime.env.env
        goals = bind_goal_predicates(env, 64)
        c = env.robots[0].controller
        assert c.use_delta and c.use_ori and c.impedance_mode == 'fixed'
        assert c.position_limits is None and c.orientation_limits is None
        def verify(step):
            row = by_step[step]
            for camera in ('agentview_image','robot0_eye_in_hand_image'):
                name = f'{step:04d}_{camera}.png'
                assert hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest() == row['images'][name]
                report['prefix_image_checks'] += 1
        verify(0)
        for row in original[1:]:
            if row['step'] > start:
                break
            runtime.action_chunk({'actions':row['actions']});verify(row['step'])
        assert runtime.step_idx == start and not runtime.done
        initial = facts(env,runtime._last_obs,64,goals)
        anchor = initial['bowls'][plan['anchor']]
        sr = T.quat2mat(np.roll(source_anchor['world_quaternion_wxyz'], -1))
        tr = T.quat2mat(np.roll(anchor['world_quaternion_wxyz'], -1))
        report.update(initial=initial, source_anchor=source_anchor, target_anchor=anchor,
                      prefix_steps=start, probe_cap=end-start)
        for step in range(start,end):
            saturation = 0
            if a.arm == 'recorded':
                action = np.asarray(actions[step],dtype=np.float32)
            else:
                control = source['controls'][step]
                assert control['step'] == step
                pos, rot = retarget(control['target_pos'],control['target_rot'],
                    source_anchor['world_position'],sr,anchor['world_position'],tr)
                c.update()
                delta = np.r_[pos-c.ee_pos,rotation_vector(rot @ c.ee_ori_mat.T)]
                scale = np.abs(c.output_max-c.output_min)/np.abs(c.input_max-c.input_min)
                command = (delta-(c.output_max+c.output_min)/2)/scale+(c.input_max+c.input_min)/2
                saturation = int(np.count_nonzero((command<c.input_min)|(command>c.input_max)))
                action = np.r_[np.clip(command,c.input_min,c.input_max),control['gripper_command']].astype(np.float32)
            runtime.action_chunk({'actions':[action.tolist()]})
            report['actions'].append(dict(step=step, command=action.tolist(), saturated_dimensions=saturation))
            report['rows'].append(dict(step=runtime.step_idx, **facts(env,runtime._last_obs,64,goals)))
            if a.arm == 'recorded' and runtime.step_idx in by_step:
                verify(runtime.step_idx)
            if runtime.done:
                break
        report.update(status='COMPLETED', environment_success=bool(runtime.success),
            probe_steps=len(report['actions']), final=report['rows'][-1],
            grasp_steps=[r['step'] for r in report['rows'] if r['bowls'][plan['anchor']]['native_grasp']],
            any_clipping=sum(r['saturated_dimensions']>0 for r in report['actions']))
    except BaseException as error:
        report.update(status='FAILED',error=repr(error));raise
    finally:
        report['seconds'] = time.perf_counter()-started
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x') as f:json.dump(report,f,indent=2)
        if runtime.env is not None:runtime.env.close()
    print(json.dumps({k:report[k] for k in ['status','arm','probe_steps','environment_success','grasp_steps','seconds']}))


if __name__ == '__main__':
    main()
