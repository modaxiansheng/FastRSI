"""Explicit, opt-in source-goal prefix. No live object poses.

The experimental grasp-only binding additionally consumes public initial RGB
and explicitly supplied static calibration. Ordinary prefixes are unchanged.

The action array on a tagged request is a transport placeholder, never an
actuator command. Missing or inconsistent tags fail before stepping physics.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .retarget_geometry import rotation_vector,pose

STEPS=130

def load_skill(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Skill SHA mismatch')
    value=json.loads(raw)
    if value.get('kind')=='experience_bank':
        from .decision_bank import validate_bank
        return validate_bank(value)
    size=len(value.get('waypoints',[]))
    if value.get('version')==2:
        from .experience_compiler import validate_compiled
        validate_compiled(value)
    elif size!=STEPS and not (size in (60,262) and value.get('source_completed_step')==262):raise ValueError('Unbound source waypoint extent')
    start=value.get('start_step',0)
    if type(start) is not int or start not in (0,130) or start>=size:raise ValueError('Unbound source entry boundary')
    for w in value['waypoints']:
        pose(w['target_pos'],w['target_rot'])
        if not np.isfinite(w['gripper_command']):raise ValueError('Nonfinite gripper command')
    if value.get('rgb_binding'):
        binding=value['rgb_binding']
        if size!=60 or start!=0:raise ValueError('RGB grasp binding is limited to the registered 60-step probe')
        if hashlib.sha256(Path(binding['source_image']).read_bytes()).hexdigest()!=binding['source_image_sha256']:raise ValueError('Binding source image mismatch')
    return value

def active(enabled,session,root_session,step,steps=STEPS,start_step=0):
    return bool(enabled and session==root_session and start_step<=step<steps)

def marker(sha,session,step,count=None,steps=STEPS):
    if type(steps) is not int or not 0<steps<=400 or type(step) is not int or step<0 or step%10 or step>=steps:raise ValueError('Unbound prefix boundary')
    expected=min(10,steps-step)
    if count is None:count=expected
    if type(count) is not int or count!=expected:raise ValueError('Unbound skill chunk length')
    return dict(version=1,sha256=sha,session_id=session,start_step=step,count=count)

def execute(runtime,chunk,skill,sha,root_session,enabled,original):
    if skill.get('kind')=='experience_bank':
        from .decision_bank import execute_bank
        return execute_bank(runtime,chunk,skill,sha,root_session,enabled,original)
    tag=chunk.get('provenance',{}).get('goal_skill')
    steps=len(skill['waypoints'])
    is_prefix=active(enabled,runtime.session_id,root_session,runtime.step_idx,steps,skill.get('start_step',0))
    if not is_prefix:
        if tag is not None:raise ValueError('Unexpected skill request outside prefix')
        return original(runtime,chunk),None
    actions=np.asarray(chunk.get('actions'),dtype=np.float32)
    expected=marker(sha,runtime.session_id,runtime.step_idx,steps=steps)
    count=expected['count']
    if tag!=expected or actions.shape!=(count,7) or not np.array_equal(actions,np.zeros((count,7),np.float32)):
        raise ValueError('Skill request marker/placeholder mismatch')
    binding_record=None;offset=np.zeros(3)
    if skill.get('rgb_binding'):
        key=(sha,runtime.session_id)
        cached=getattr(runtime,'_fastrsi_rgb_binding',None)
        if cached is None or cached['key']!=key:
            if runtime.step_idx!=0:raise ValueError('RGB binding must originate at the initial observation')
            import cv2
            from .visual_binding import match_template
            from .planar_binding import translation
            tick=time.perf_counter();b=skill['rgb_binding'];source=cv2.cvtColor(cv2.imread(b['source_image']),cv2.COLOR_BGR2RGB)
            match=match_template(source,runtime._last_obs['agentview_image'],b['source_roi_xyxy'])
            center=lambda box:[(box[0]+box[2]-1)/2,(box[1]+box[3]-1)/2]
            offset=translation(center(b['source_roi_xyxy']),center(match['peaks'][0]['box_xyxy']),b['calibration'])
            binding_record=dict(matching=match,offset=offset.tolist(),seconds=time.perf_counter()-tick,source_image_sha256=b['source_image_sha256'],online_inputs=['initial_agentview_rgb','shared_static_calibration'])
            cached=dict(key=key,offset=offset.tolist());runtime._fastrsi_rgb_binding=cached
        offset=np.asarray(cached['offset'])
    begin=runtime.step_idx;submitted=[];reward=0.;chunk_begin=len(runtime._episode_chunks)
    for index in range(begin,begin+count):
        w=skill['waypoints'][index];c=runtime.env.env.robots[0].controller;c.update()
        desired=np.r_[np.asarray(w['target_pos'])+offset-c.ee_pos,rotation_vector(np.asarray(w['target_rot'])@c.ee_ori_mat.T)]
        scale=np.abs(c.output_max-c.output_min)/np.abs(c.input_max-c.input_min)
        command=(desired-(c.output_max+c.output_min)/2)/scale+(c.input_max+c.input_min)/2
        action=np.r_[np.clip(command,c.input_min,c.input_max),w['gripper_command']].astype(np.float32)
        submitted.append(action.tolist());out=original(runtime,{**chunk,'actions':[action.tolist()]});reward+=float(out.get('reward',0))
        if runtime.done:break
    # One public request remains one chunk. Preserve all internal control
    # records in the separate execution receipt, not as extra public calls.
    substeps=runtime._episode_chunks[chunk_begin:]
    if len(substeps)!=len(submitted):raise ValueError('Native control record count mismatch')
    combined={**substeps[-1],'first_step':begin+1,'executed_steps':runtime.step_idx-begin,
              'requested_steps':count,'reward':reward,'action_shape':[count,7]}
    runtime._episode_chunks[chunk_begin:]=[combined]
    runtime._last_status.update(reward=reward,episode_summary=runtime._episode_summary())
    return dict(runtime._last_status),dict(start_step=begin,end_step=runtime.step_idx,skill_sha256=sha,actual_actions=submitted,internal_control_records=substeps,object_pose_used=False,rgb_binding=binding_record)
