"""Fixed 70-step source/cached/fresh-state diagnostic; privileged, not training."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.retarget_geometry import retarget, rotation_vector
from replay_stack_progress import RUNTIME_SOURCE, facts, bind_goal_predicates
from stage_trace import png_bytes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--arm', choices=['source', 'cached', 'fresh'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    plan = json.loads(a.plan.read_text());root = Path('/project')
    assert plan['steps'] == 70
    for rel, sha in plan['source_sha256'].items():
        assert digest(root/rel) == sha, rel
    src = json.loads((root/plan['source_controls']).read_text())
    prior = json.loads((root/plan['prior_retarget']).read_text())
    is_source = a.arm == 'source'
    case = src['case'] if is_source else json.loads((root/plan['target_controls']).read_text())['case']
    trace_file = root/plan['source_trace' if is_source else 'target_trace']
    trace = {r['step']:r for r in map(json.loads, trace_file.read_text().splitlines())}
    spec = importlib.util.spec_from_file_location('tracking_runtime', RUNTIME_SOURCE)
    m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime = m.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,
        init_state_id=case['init_state_id'],seed=7,max_steps=400,control_mode='relative'))
    r = dict(status='RUNNING',arm=a.arm,plan_sha256=digest(a.plan),model_calls=0,
        persistent_learning_updates=0,scope='Privileged controller tracking diagnostic',rows=[],image_checks=0)
    started = time.perf_counter()
    try:
        import robosuite.utils.transform_utils as T
        runtime.reset({'session_id':case['id']})
        env=runtime.env.env;c=env.robots[0].controller;goals=bind_goal_predicates(env,64)
        assert 'force' in inspect.signature(c.update).parameters
        r['update_source']=inspect.getsource(type(c).update)
        r['initial']=facts(env,runtime._last_obs,64,goals)
        anchor=r['initial']['bowls']['akita_black_bowl_1']
        source_anchor=src['boundaries'][0]['bowls']['akita_black_bowl_1']
        sr=T.quat2mat(np.roll(source_anchor['world_quaternion_wxyz'],-1))
        tr=T.quat2mat(np.roll(anchor['world_quaternion_wxyz'],-1))
        for camera in ['agentview_image','robot0_eye_in_hand_image']:
            assert hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest()==trace[0]['images'][f'0000_{camera}.png']
            r['image_checks']+=1
        for step in range(70):
            control=src['controls'][step]
            pos,rot=retarget(control['target_pos'],control['target_rot'],
                source_anchor['world_position'],sr,anchor['world_position'],tr)
            # Direct simulator reads below are diagnostic and do not change the
            # cached arm's update flag. Only fresh changes the control operation.
            true_pos=env.sim.data.get_site_xpos(c.eef_name).copy()
            true_rot=env.sim.data.get_site_xmat(c.eef_name).copy()
            if is_source:
                action=np.r_[control['controller_action'],control['gripper_command']].astype(np.float32)
                used_pos=None
            else:
                c.update(force=True) if a.arm=='fresh' else c.update()
                used_pos=c.ee_pos.copy()
                delta=np.r_[pos-c.ee_pos,rotation_vector(rot@c.ee_ori_mat.T)]
                scale=np.abs(c.output_max-c.output_min)/np.abs(c.input_max-c.input_min)
                command=(delta-(c.output_max+c.output_min)/2)/scale+(c.input_max+c.input_min)/2
                action=np.r_[np.clip(command,c.input_min,c.input_max),control['gripper_command']].astype(np.float32)
                if a.arm=='cached':
                    np.testing.assert_array_equal(action,np.asarray(prior['actions'][step]['command'],dtype=np.float32))
            runtime.action_chunk({'actions':[action.tolist()]})
            row=dict(step=step+1,action=action.tolist(),desired_pos=pos.tolist(),desired_rot=rot.tolist(),
                true_pre_pos=true_pos.tolist(),used_pre_pos=None if used_pos is None else used_pos.tolist(),
                pose_staleness_m=None if used_pos is None else float(np.linalg.norm(used_pos-true_pos)),
                submitted_goal_pos_error_m=float(np.linalg.norm(c.goal_pos-pos)),
                submitted_goal_rotation_error=float(np.linalg.norm(rotation_vector(c.goal_ori@rot.T))),
                gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(),
                **facts(env,runtime._last_obs,64,goals))
            if a.arm=='cached':
                previous=prior['rows'][step]
                assert row['bowls']==previous['bowls']
                assert row['eef_world_position']==previous['eef_world_position']
            if is_source and step+1 in trace:
                for camera in ['agentview_image','robot0_eye_in_hand_image']:
                    name=f'{step+1:04d}_{camera}.png'
                    assert hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest()==trace[step+1]['images'][name]
                    r['image_checks']+=1
            if step+1 in [50,54,60,65,68,69,70]:
                row['images']={}
                for camera in ['agentview_image','robot0_eye_in_hand_image']:
                    name=f'{step+1:04d}_{camera}.png';blob=png_bytes(runtime._last_obs[camera])
                    (a.output.parent/name).write_bytes(blob)
                    row['images'][name]=hashlib.sha256(blob).hexdigest()
            r['rows'].append(row)
        r.update(status='COMPLETED',steps=70,grasp_steps=[x['step'] for x in r['rows'] if x['bowls']['akita_black_bowl_1']['native_grasp']],
                 final_grasp=r['rows'][-1]['bowls']['akita_black_bowl_1']['native_grasp'])
    except BaseException as error:
        r.update(status='FAILED',error=repr(error));raise
    finally:
        r['seconds']=time.perf_counter()-started
        with a.output.open('x') as f:json.dump(r,f,indent=2)
        if runtime.env is not None:runtime.env.close()
    print(json.dumps({k:r[k] for k in ['status','arm','steps','grasp_steps','final_grasp','seconds']}),flush=True)


if __name__=='__main__':
    main()
