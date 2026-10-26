"""Approximate planar translation from RGB correspondence and fixed calibration.

Calibration is an explicit shared setup input, never inferred from a live object
pose. A constant feature height and unchanged object orientation are assumptions.
"""
import numpy as np

def camera_rotation(quaternion_wxyz):
    q=np.asarray(quaternion_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
        raise ValueError('Invalid camera quaternion')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def intersect_plane(pixel_xy,calibration):
    c=calibration;uv=np.asarray(pixel_xy,dtype=float);origin=np.asarray(c['position'],dtype=float)
    if uv.shape!=(2,) or origin.shape!=(3,) or not np.isfinite(uv).all() or not np.isfinite(origin).all():
        raise ValueError('Invalid pixel or origin')
    width,height=c['width'],c['height'];fovy=c['fovy_degrees'];z=c['plane_z']
    if width<=0 or height<=0 or not 0<fovy<180 or not np.isfinite(z):raise ValueError('Invalid calibration')
    if c['image_convention']!='libero_policy_180_from_bottom_up':raise ValueError('Unknown image orientation')
    f=height/(2*np.tan(np.deg2rad(fovy)/2));center=np.array([(width-1)/2,(height-1)/2])
    # Bottom-up OpenGL -> vertical+horizontal flip in the current policy adapter.
    # Relative to upright top-left pixels, only the horizontal axis is mirrored.
    ray=camera_rotation(c['quaternion_wxyz'])@np.r_[-(uv-center)/f,-1.0]
    if abs(ray[2])<1e-8:raise ValueError('Ray parallel to assumed plane')
    distance=(z-origin[2])/ray[2]
    if distance<=0:raise ValueError('Plane behind camera')
    return origin+distance*ray

def translation(source_pixel,target_pixel,calibration):
    return intersect_plane(target_pixel,calibration)-intersect_plane(source_pixel,calibration)
