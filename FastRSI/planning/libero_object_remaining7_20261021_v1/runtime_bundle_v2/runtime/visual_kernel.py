"""Visual recovery calculations. No simulator, truth file, outcome label or policy access.

Source demonstration geometry is fixed training/development input. Current poses
must be RGB predictions; current EEF and finger positions are public sensors.
"""
import numpy as np
from vision_skill import public_eef,correction_plan,action_for_pose
from object_frame_skill import rotation_vector

def rigid(v):
 t=np.asarray(v,dtype=float)
 return t.shape==(4,4) and np.isfinite(t).all() and np.allclose(t[3],[0,0,0,1],atol=1e-6) and np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-4) and abs(np.linalg.det(t[:3,:3])-1)<1e-4

def initial_pose_gate(rgb_pose,skill):
 if not rigid(rgb_pose):return dict(status='UNKNOWN',reason='INVALID_RGB_POSE')
 t=np.asarray(rgb_pose);source=np.asarray(skill['source_object_pose']);local_up=source[:3,:3].T@np.array([0.,0.,1.])
 tilt=float(np.degrees(np.arccos(np.clip((t[:3,:3]@local_up)[2],-1.,1.))))
 return dict(status='ACCEPT' if tilt<=15. else 'UNKNOWN',reason=None if tilt<=15. else 'SOURCE_SUPPORTED_RESTING_ORIENTATION_UNCERTAIN',tilt_deg=tilt)

def held_gate(obs,rgb_object,release_skill,previous=None):
 if not rigid(rgb_object):return dict(status='UNKNOWN',reason='INVALID_CURRENT_RGB_SUBJECT_POSE')
 q=np.asarray(obs['robot0_gripper_qpos'],dtype=float);assert q.shape==(2,) and np.isfinite(q).all()
 eef=public_eef(obs);obj=np.asarray(rgb_object,dtype=float);distance=float(np.linalg.norm(obj[:3,3]-eef[:3,3]));width=float(abs(q[0]-q[1]));relative=np.linalg.inv(eef)@obj
 metrics=dict(finger_width_m=width,subject_eef_distance_m=distance,relative_eef_object=relative.tolist())
 # Physical clearance and source-held envelope, fixed before new test identities.
 if width<=.002:return dict(status='UNKNOWN',reason='FINGERS_AT_NEAR_EMPTY_CLOSE',**metrics)
 if distance>release_skill['source_held_center_radius_m']+2*release_skill['position_tolerance_m']:
  return dict(status='UNKNOWN',reason='SUBJECT_NOT_LOCALIZED_IN_SOURCE_HELD_ENVELOPE',**metrics)
 if previous is None:return dict(status='UNKNOWN',reason='NEED_TWO_CURRENT_VISUAL_PROPRIOCEPTIVE_SAMPLES',**metrics)
 prior=np.asarray(previous['relative_eef_object']);motion=float(np.linalg.norm(relative[:3,3]-prior[:3,3]));metrics['relative_translation_change_m']=motion
 if motion>release_skill['source_relative_hold_translation_step_max_m']+2*release_skill['position_tolerance_m']:
  return dict(status='UNKNOWN',reason='VISUAL_SUBJECT_MOTION_NOT_CONSISTENT_WITH_GRIPPER',**metrics)
 return dict(status='HELD_TARGET_ESTIMATE',reason='TWO_RGB_POSES_MOVE_WITH_PUBLIC_EEF_AND_NONEMPTY_FINGER_GAP',**metrics)

def place_plan(obs,release_skill,rgb_object,rgb_basket):
 assert rigid(rgb_object) and rigid(rgb_basket)
 obj=np.asarray(rgb_object);basket=np.asarray(rgb_basket);now=public_eef(obs);grasp=np.linalg.inv(obj)@now
 sb=np.asarray(release_skill['source_basket_pose']);so=np.asarray(release_skill['source_object_release_pose']);se=np.asarray(release_skill['source_eef_release_pose']);sg=np.linalg.inv(so)@se
 goal=basket@np.linalg.inv(sb)@so@grasp
 transported=[basket@np.linalg.inv(sb)@np.asarray(x)@np.linalg.inv(sg)@grasp for x in release_skill['source_held_commanded_poses']]
 z=max([now[2,3],goal[2,3]]+[x[2,3] for x in transported]);up=now.copy();up[2,3]=z;across=goal.copy();across[2,3]=z
 suffix=[dict(pose=(basket@np.linalg.inv(sb)@np.asarray(x['pose'])@np.linalg.inv(sg)@grasp).tolist(),grip=x['grip'],source_step=x['source_step']) for x in release_skill['source_release_suffix']]
 return dict(targets=[x.tolist() for x in [up,across,goal]],suffix=suffix,close_gripper=release_skill['close_gripper'],
  position_tolerance_m=release_skill['position_tolerance_m'],rotation_tolerance_rad=release_skill['rotation_tolerance_rad'],max_steps_per_phase=release_skill['max_steps_per_phase'],
  current_rgb_object_pose=obj.tolist(),current_rgb_basket_pose=basket.tolist(),public_eef=now.tolist(),current_estimated_grasp_transform=grasp.tolist(),GT_read=False)
