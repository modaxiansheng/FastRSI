"""Only permitted successful development source trajectories become fixed skill data."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from object_frame_skill import source_object_pose
H=Path('/out');W=Path('/source')
def pose(s):
 t=np.eye(4);t[:3,3]=s['eef_position'];t[:3,:3]=s['eef_rotation'];return t
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def main():
 (H/'release_skills').mkdir(exist_ok=False);(H/'grasp_skills').mkdir(exist_ok=False)
 for src in ['dev12','dev13','dev15']:
  skill=json.loads((W/'skills'/(src+'.json')).read_text());subject=skill['subject'];root=W/'cases/node187'/src;events=[json.loads(x) for x in (root/'DIAGNOSTIC_EVENTS.jsonl').read_text().splitlines()];states={x['step']:x for x in events if x['event']=='diagnostic_state'};commands={x['step']:x for x in events if x['event']=='commanded_target'};start=skill['source_segment_end_exclusive'];pre=start-2
  assert states[pre]['objects'][subject]['two_finger_contact'] and states[start]['objects'][subject]['two_finger_contact']
  release=next(x for x in range(start+1,max(states)+1) if states[x]['action'][-1]<0);before=release-1;assert states[before]['objects'][subject]['two_finger_contact'] and states[max(states)]['official_success']
  def cmd(i):t=np.eye(4);t[:3,3]=commands[i]['goal_pos'];t[:3,:3]=commands[i]['goal_ori'];return t
  held=[i for i in range(start,before+1) if states[i]['objects'][subject]['two_finger_contact']];g=[np.linalg.inv(pose(states[i]))@source_object_pose(states[i]['objects'][subject]) for i in held]
  radius=max(float(np.linalg.norm(x[:3,3])) for x in g);motion=max(float(np.linalg.norm(a[:3,3]-b[:3,3])) for a,b in zip(g,g[1:]))
  value=dict(source_identity=src,subject=subject,source_grasp_end_step=start,source_before_grasp_step=pre,
   source_pregrasp_object_pose=source_object_pose(states[pre]['objects'][subject]),source_initial_object_pose=np.asarray(skill['source_object_pose']),
   source_held_center_radius_m=radius,source_relative_hold_translation_step_max_m=motion,
   source_basket_pose=source_object_pose(states[before]['objects']['basket_1']),source_object_release_pose=source_object_pose(states[before]['objects'][subject]),source_eef_release_pose=pose(states[before]),source_held_commanded_poses=[cmd(i) for i in range(start+1,before+1)],source_release_suffix=[dict(pose=cmd(i),grip=states[i]['action'][-1],source_step=i) for i in range(release,max(states)+1)],
   close_gripper=float(np.median([states[i]['action'][-1] for i in range(before-4,before+1)])),position_tolerance_m=skill['source_p95_translation_step'],rotation_tolerance_rad=skill['source_max_rotation_step'],max_steps_per_phase=2*(before-start),
   source_events_sha256=hashlib.sha256((root/'DIAGNOSTIC_EVENTS.jsonl').read_bytes()).hexdigest(),source_GT_supervision_only=True)
  (H/'release_skills'/(src+'.json')).write_text(json.dumps(plain(value),indent=2));(H/'grasp_skills'/(src+'.json')).write_bytes((W/'skills'/(src+'.json')).read_bytes());print(json.dumps(dict(source=src,held_radius=radius,relative_motion=motion)),flush=True)
if __name__=='__main__':main()
