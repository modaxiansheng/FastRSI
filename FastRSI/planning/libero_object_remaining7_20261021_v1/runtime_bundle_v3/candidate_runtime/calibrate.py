"""Compile six permitted development task models for static camera calibration only.
No task initial-state files, dynamic object poses, simulator actions, policy or learning.
"""
import hashlib,inspect,json,time,traceback
from pathlib import Path
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
import robosuite.macros as macros
import robosuite.utils.transform_utils as T
OUT=Path('/out/calibration'); OLD=Path('/eval'); OUT.mkdir(exist_ok=False)
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def write(n,v):
 with (OUT/n).open('x') as f:json.dump(plain(v),f,indent=2,allow_nan=False)
def pose(p,q):
 x=np.eye(4);x[:3,3]=p;x[:3,:3]=T.quat2mat(np.asarray(q)[[1,2,3,0]]);return x
def path_to_root(model,bid):
 out=[]
 while bid!=0:
  out.append(bid);bid=int(model.body_parentid[bid])
 return out[::-1]
def static_body(model,bid):
 out=np.eye(4)
 for i in path_to_root(model,bid):out=out@pose(model.body_pos[i],model.body_quat[i])
 return out
start=time.time();rows=json.loads((OLD/'ASSET_INDEX_FINAL.json').read_text())
rows=[r for r in rows if r['suite']=='libero_object' and r['variant'] in ['position','environment'] and r['task_id'] in [3,5,8]]
assert len(rows)==6
write('STATIC_CALIBRATION_PREREGISTERED.json',dict(rows=[{k:r[k] for k in ['suite','variant','task_id','bddl_rel','bddl_sha256']} for r in rows],max_scene_constructors=6,normal_physical_attempts=0,explicit_env_steps=0,initial_state_files_read=0,neural_calls=0,scope='Static task/robot/camera geometry only. Constructor random placement is not used as pose input or scored trial.'))
result=[];error=None
try:
 for row in rows:
  path=OLD/'assets'/row['bddl_rel'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['bddl_sha256']
  before=time.time();wrapper=None
  try:
   wrapper=OffScreenRenderEnv(bddl_file_name=str(path),camera_heights=256,camera_widths=256)
   env=wrapper.env;model=env.sim.model;data=env.sim.data
   c=env.robots[0].controller;sid=model.site_name2id(c.eef_name);sb=int(model.site_bodyid[sid])
   eef_static=static_body(model,sb)@pose(model.site_pos[sid],model.site_quat[sid])
   eef_actual=np.eye(4);eef_actual[:3,3]=data.site_xpos[sid];eef_actual[:3,:3]=data.site_xmat[sid].reshape(3,3)
   cameras={}
   for name in ['agentview','robot0_eye_in_hand']:
    cid=model.camera_name2id(name);cb=int(model.cam_bodyid[cid]);bp=path_to_root(model,cb);sp=path_to_root(model,sb)
    common=0
    while common<min(len(bp),len(sp)) and bp[common]==sp[common]:common+=1
    camera_path=bp[common:];site_path=sp[common:]
    if name=='robot0_eye_in_hand':
     assert all(int(model.body_jntnum[i])==0 for i in camera_path+site_path),'Unmodelled joint between EEF and camera'
    camera_static=static_body(model,cb)@pose(model.cam_pos[cid],model.cam_quat[cid])
    camera_actual=np.eye(4);camera_actual[:3,3]=data.cam_xpos[cid];camera_actual[:3,:3]=data.cam_xmat[cid].reshape(3,3)
    camera_cv_static=camera_static@np.diag([1.,-1.,-1.,1.])
    f=.5*256/np.tan(float(model.cam_fovy[cid])*np.pi/360.)
    d=dict(body_id=cb,body_name=model.body_id2name(cb),cam_local_pos=model.cam_pos[cid],cam_local_quat_wxyz=model.cam_quat[cid],fovy=model.cam_fovy[cid],K=[[f,0,128],[0,f,128],[0,0,1]],eef_site=c.eef_name,relative_path_joint_counts=[int(model.body_jntnum[i]) for i in camera_path+site_path],model_body_path=[model.body_id2name(i) for i in bp])
    if name=='agentview':
     assert all(int(model.body_jntnum[i])==0 for i in bp),'Front camera not static'
     err=float(np.max(np.abs(camera_static-camera_actual)));assert err<1e-6,err
     d.update(T_world_camera_cv=camera_cv_static,static_world_frame_error=err)
    else:
     rel=np.linalg.inv(eef_static)@camera_static@np.diag([1.,-1.,-1.,1.])
     actual=np.linalg.inv(eef_actual)@camera_actual@np.diag([1.,-1.,-1.,1.])
     err=float(np.max(np.abs(rel-actual)));assert err<1e-6,err
     d.update(T_eef_camera_cv=rel,relative_frame_error=err)
    cameras[name]=d
   record=dict(task_id=row['task_id'],variant=row['variant'],bddl_sha256=row['bddl_sha256'],cameras=cameras,seconds=time.time()-before,constructor_calls=1,explicit_env_steps=0,policy_calls=0,initial_state_files_read=0,dynamic_object_poses_read=False)
   write('CALIBRATION_'+row['variant']+'_task'+str(row['task_id'])+'.json',record);result.append(record)
   if len(result)==1:
    import robosuite.utils.binding_utils as binding
    sources=dict(camera_sensors=inspect.getsource(env._create_camera_sensors),renderer=inspect.getsource(binding.MjSim.render),image_convention=getattr(macros,'IMAGE_CONVENTION',None))
    write('ACTUAL_RENDER_CONVENTION_SOURCE.json',sources)
   print(json.dumps(dict(task=row['task_id'],variant=row['variant'],status='STATIC_CAMERA_CALIBRATED',seconds=time.time()-before)),flush=True)
  finally:
   if wrapper is not None:wrapper.close()
except BaseException:error=traceback.format_exc()
write('STATIC_CALIBRATION_RESULT.json',dict(status='STATIC_CALIBRATION_CLOSED' if error is None else 'STATIC_CALIBRATION_FAILED_NEEDS_DIAGNOSIS',error=error,scene_models_completed=len(result),seconds=time.time()-start,normal_physical_attempts=0,explicit_env_steps=0,policy_generations=0,money=None,constructor_internal_initialization_cost_included=True,results=result))
if error:raise RuntimeError(error)
