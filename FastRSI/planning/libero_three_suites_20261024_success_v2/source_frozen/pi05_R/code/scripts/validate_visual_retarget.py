"""Bounded RGB+static-calibration translation vs unchanged waypoint control.

Live object state is read only into evaluator records, never into action binding.
"""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.visual_binding import match_template
from fastrsi.planar_binding import translation,camera_rotation
from fastrsi.retarget_geometry import rotation_vector
from replay_stack_progress import RUNTIME_SOURCE,facts,bind_goal_predicates
from stage_trace import png_bytes

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--arm',choices=['rgb','zero'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();plan=json.loads(a.plan.read_text());root=Path('/project')
    for rel,sha in plan['sha256'].items():assert digest(root/rel)==sha,rel
    skill=json.loads((root/plan['skill']).read_text());cal=json.loads((root/plan['calibration']).read_text())
    spec=importlib.util.spec_from_file_location('visual_runtime',RUNTIME_SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    runtime=m.LiberoRealRuntime(dict(benchmark_name='libero_90',task_id=64,init_state_id=1,seed=7,max_steps=400,control_mode='relative'))
    result=dict(status='RUNNING',arm=a.arm,plan_sha256=digest(a.plan),rows=[],model_calls=0,training_updates=0,online_default_changed=False)
    start=time.perf_counter()
    try:
        import cv2
        runtime.reset({'session_id':'visual_retarget_'+a.arm})
        assert hashlib.sha256(png_bytes(runtime._last_obs['agentview_image'])).hexdigest()==plan['target_initial_image_sha256']
        env=runtime.env.env;c=env.robots[0].controller;goals=bind_goal_predicates(env,64)
        # Check fixed calibration only; no object locations enter binding.
        cam=env.sim.model.camera_name2id('agentview')
        np.testing.assert_allclose(env.sim.data.cam_xpos[cam],cal['position'],atol=1e-8)
        np.testing.assert_allclose(env.sim.data.cam_xmat[cam].reshape(3,3),camera_rotation(cal['quaternion_wxyz']),atol=1e-7)
        assert abs(float(env.sim.model.cam_fovy[cam])-cal['fovy_degrees'])<1e-8
        bind_start=time.perf_counter()
        source=cv2.cvtColor(cv2.imread(str(root/plan['source_image'])),cv2.COLOR_BGR2RGB)
        match=match_template(source,runtime._last_obs['agentview_image'],plan['source_roi_xyxy'])
        sb=plan['source_roi_xyxy'];tb=match['peaks'][0]['box_xyxy']
        center=lambda box: [(box[0]+box[2]-1)/2,(box[1]+box[3]-1)/2]
        estimate=translation(center(sb),center(tb),cal)
        delta=estimate if a.arm=='rgb' else np.zeros(3)
        result.update(binding_seconds=time.perf_counter()-bind_start,matching=match,estimated_delta=estimate.tolist(),applied_delta=delta.tolist(),opencv_version=cv2.__version__)
        result['initial_evaluator_only']=facts(env,runtime._last_obs,64,goals)
        for i,waypoint in enumerate(skill['waypoints']):
            pos=np.asarray(waypoint['target_pos'])+delta;rot=np.asarray(waypoint['target_rot']);c.update()
            relative=np.r_[pos-c.ee_pos,rotation_vector(rot@c.ee_ori_mat.T)]
            scale=np.abs(c.output_max-c.output_min)/np.abs(c.input_max-c.input_min)
            command=(relative-(c.output_max+c.output_min)/2)/scale+(c.input_max+c.input_min)/2
            action=np.r_[np.clip(command,c.input_min,c.input_max),waypoint['gripper_command']].astype(np.float32)
            runtime.action_chunk({'actions':[action.tolist()]})
            row=dict(step=i+1,action=action.tolist(),gripper_qpos=runtime._last_obs['robot0_gripper_qpos'].tolist(),evaluator=facts(env,runtime._last_obs,64,goals))
            if i+1 in [70,120,130]:
                row['images']={}
                for camera in ['agentview_image','robot0_eye_in_hand_image']:
                    name=f'{i+1:04d}_{camera}.png';blob=png_bytes(runtime._last_obs[camera]);(a.output.parent/name).write_bytes(blob);row['images'][name]=hashlib.sha256(blob).hexdigest()
            result['rows'].append(row)
        row=result['rows'][-1];f=row['evaluator'];b=f['bowls']['akita_black_bowl_1']
        result.update(status='COMPLETED',steps=len(result['rows']),local_release=bool(f['native_bottom_in_tray'] and not b['native_grasp'] and not b['gripper_contact'] and all(abs(v)>.03 for v in row['gripper_qpos']) and row['action'][-1]<0),full_task_success=all(g['value'] for g in f['goal_predicates']))
    except BaseException as e:result.update(status='FAILED',error=repr(e));raise
    finally:
        result['seconds']=time.perf_counter()-start
        with a.output.open('x') as f:json.dump(result,f,indent=2)
        if runtime.env is not None:runtime.env.close()
    print(json.dumps({k:result[k] for k in ['status','arm','steps','local_release','full_task_success','seconds']}),flush=True)
if __name__=='__main__':main()
