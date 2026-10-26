"""RGB-only geometric input adapter; no simulator import or object-state access.

Static calibration is explicit. Current target pose is NOT provided by this module.
Model RGB is horizontal reflection of the top-left OpenCV view, as verified from
actual renderer -> observable -> frozen runtime source. EEF uses controller matrix,
not an assumed equivalent quaternion frame. Plane intersection is conditional on a
known support plane and a known object reference height; it is not general 6D pose.
"""
import numpy as np

def model_rgb_projection(world_points, calibration, image_width=256):
    points=np.asarray(world_points,dtype=float)
    assert points.shape[-1]==3
    T=np.asarray(calibration['T_world_camera_cv'],dtype=float)
    K=np.asarray(calibration['K'],dtype=float)
    camera=(np.linalg.inv(T)@np.concatenate([points,np.ones(points.shape[:-1]+(1,))],axis=-1)[...,None])[...,0]
    pixels=(K@camera[...,:3,None])[...,0]
    assert np.all(camera[...,2]>0),'Points behind camera are not usable evidence'
    uv=pixels[...,:2]/pixels[...,2:3]
    uv[...,0]=image_width-1-uv[...,0]
    return uv

def model_rgb_ray_to_zplane(uv, calibration, z_world, image_width=256):
    uv=np.asarray(uv,dtype=float);assert uv.shape==(2,)
    K=np.asarray(calibration['K'],float);T=np.asarray(calibration['T_world_camera_cv'],float)
    cv=np.array([image_width-1-uv[0],uv[1],1.])
    direction=T[:3,:3]@np.linalg.solve(K,cv)
    assert abs(direction[2])>1e-10,'Ray parallel to declared plane'
    scale=(float(z_world)-T[2,3])/direction[2]
    assert scale>0,'Plane behind camera'
    return T[:3,3]+scale*direction

def wrist_camera_from_public_eef(eef_pos,eef_controller_matrix,calibration):
    T=np.eye(4);T[:3,3]=np.asarray(eef_pos,float);T[:3,:3]=np.asarray(eef_controller_matrix,float)
    assert np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5)
    return T@np.asarray(calibration['T_eef_camera_cv'],float)
