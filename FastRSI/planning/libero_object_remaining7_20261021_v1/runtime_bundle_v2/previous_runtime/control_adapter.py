"""Absolute wire equivalent of the frozen clipped relative OSC pose increment."""
import numpy as np
from vision_skill import action_for_pose,public_eef
from object_frame_skill import rotation_vector

def relative_to_absolute(obs,relative,skill):
    a=np.asarray(relative,dtype=np.float64);assert a.shape==(7,)
    c=skill['controller_parameters'];scale=np.asarray(c['output_max'],float)
    delta=np.clip(a[:6],-1,1)*scale
    now=public_eef(obs);angle=float(np.linalg.norm(delta[3:]));axis=delta[3:]/angle if angle else np.array([1.,0.,0.])
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    rotation=np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)
    return np.r_[now[:3,3]+delta[:3],rotation_vector(rotation@now[:3,:3]),a[6]].astype(np.float32)

def backend_action_for_pose(obs,desired,grip,skill,model):
    relative=action_for_pose(obs,desired,grip,skill)
    return relative_to_absolute(obs,relative,skill) if model=='xvla' else relative
