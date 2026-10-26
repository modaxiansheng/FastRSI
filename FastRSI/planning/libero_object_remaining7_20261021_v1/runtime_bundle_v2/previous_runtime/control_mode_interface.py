"""Use byte-identical frozen relative skill commands, restore the backend wire mode at handoff."""
import hashlib,numpy as np
from vision_skill import action_for_pose
def backend_action_for_pose(obs,desired,grip,skill,model):
 assert model in ['pi0','pi05','openvla','xvla']
 return action_for_pose(obs,desired,grip,skill)
def switch_xvla_controller(runtime,use_delta,emit,phase):
 controllers=[robot.controller for robot in runtime.env.robots]
 if all(c.use_delta==use_delta for c in controllers):return
 before=hashlib.sha256(np.asarray(runtime.env.get_sim_state()).tobytes()).hexdigest();step=runtime.step_idx
 for c in controllers:c.use_delta=use_delta
 after=hashlib.sha256(np.asarray(runtime.env.get_sim_state()).tobytes()).hexdigest();assert before==after and runtime.step_idx==step
 emit(dict(event='CONTROL_REPRESENTATION_HANDOFF',phase=phase,step=step,use_delta=use_delta,physics_unchanged=True,state_sha256=before,environment_reset=False,source='Explicit actuator-mode interface; no coordinate offset or mechanical action changes'))
