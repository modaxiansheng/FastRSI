"""Explicit native-image/CV/world conventions extracted from the accepted runtime."""
import numpy as np


def source_seed(box, template_crop, source_pose, source_camera, current_camera):
    box = np.asarray(box, float)
    crop = np.asarray(template_crop, float)
    assert box.shape == crop.shape == (4,) and np.all(box[2:] > box[:2])
    sw = np.asarray(source_camera['T_world_camera_cv'])
    cw = np.asarray(current_camera['T_world_camera_cv'])
    sk = np.asarray(source_camera['K'])
    ck = np.asarray(current_camera['K'])
    source = np.asarray(source_pose)
    camera_source = np.linalg.inv(sw) @ source
    source_center = sk @ camera_source[:3, 3]
    source_center = source_center[:2] / source_center[2]
    source_box_cv = np.array([255-crop[2], crop[1], 255-crop[0], crop[3]])
    current_box_cv = np.array([255-box[2], box[1], 255-box[0], box[3]])
    scale = np.sqrt(np.prod((box[2:]-box[:2]) / (crop[2:]-crop[:2])))
    center = ((current_box_cv[:2]+current_box_cv[2:])/2 +
              (source_center-(source_box_cv[:2]+source_box_cv[2:])/2)*scale)
    depth = camera_source[2, 3] / scale
    seed = np.eye(4)
    seed[:3, :3] = cw[:3, :3].T @ source[:3, :3]
    seed[:3, 3] = (np.linalg.inv(ck) @ np.r_[center, 1.])*depth
    assert depth > 0 and np.isfinite(seed).all()
    return seed, dict(method='FIXED_SUCCESS_SOURCE_ORIENTATION_AND_RGB_BBOX_SCALE_DEPTH',
                     bbox=box.tolist(), scale=float(scale),
                     camera_depth_m=float(depth), GT_read=False)


def put_on_support_plane(seed, source_pose, current_camera, plane_delta):
    """Keep the observed center ray; impose the demonstrated static support plane."""
    seed = np.asarray(seed).copy()
    cw = np.asarray(current_camera['T_world_camera_cv'])
    z = np.asarray(source_pose)[2, 3] + plane_delta
    ray = seed[:3, 3] / seed[2, 3]
    depth = (z-cw[2, 3]) / (cw[:3, :3] @ ray)[2]
    assert depth > 0 and np.isfinite(depth)
    seed[:3, 3] = ray*depth
    return seed, float(z)


def source_centers(xx, yy, width, height, crop, source_pose, source_camera,
                   current_camera, z):
    """Vectorized source_seed center rays for pre-existing static table bounds."""
    source = np.asarray(source_pose)
    sw = np.asarray(source_camera['T_world_camera_cv'])
    cw = np.asarray(current_camera['T_world_camera_cv'])
    crop = np.asarray(crop, float)
    camera_source = np.linalg.inv(sw) @ source
    center = np.asarray(source_camera['K']) @ camera_source[:3, 3]
    center = center[:2]/center[2]
    source_box = np.array([255-crop[2], crop[1], 255-crop[0], crop[3]])
    scale = np.sqrt(width/(crop[2]-crop[0])*height/(crop[3]-crop[1]))
    offset = (center-(source_box[:2]+source_box[2:])/2)*scale
    uv = np.stack([255-(xx+width/2)+offset[0], yy+height/2+offset[1],
                   np.ones_like(xx)], axis=-1)
    rays = uv @ np.linalg.inv(np.asarray(current_camera['K'])).T @ cw[:3, :3].T
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (z-cw[2, 3])/rays[:, :, 2]
    return cw[:3, 3]+rays*depth[:, :, None], depth
