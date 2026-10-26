"""Fixed development-only, truth-assisted source transfer. No simulator writes.

Pose convention follows the pinned MimicGen RobosuiteInterface and object-frame
transform references in references/mimicgen. This is not an RGB pose estimator.
"""
import math
import numpy as np
import robosuite.utils.transform_utils as T


def rotation_vector(matrix):
    # Native mat2quat returns float32. sqrt(1-w*w) loses precision for small
    # rotations; atan2(norm(xyz), w) retains the represented rotation instead.
    q = np.asarray(T.mat2quat(matrix), dtype=np.float64)
    if q[3] < 0: q = -q
    n = float(np.linalg.norm(q[:3]))
    return np.zeros(3) if n < 1e-15 else q[:3] * (2*math.atan2(n, float(q[3]))/n)


def pose(pos, rot):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = pos
    return out


def source_object_pose(record):
    w, x, y, z = record['quaternion']
    return pose(record['position'], T.quat2mat(np.array([x, y, z, w])))


def current_eef(env):
    c = env.robots[0].controller
    sid = env.sim.model.site_name2id(c.eef_name)
    return pose(env.sim.data.site_xpos[sid].copy(), env.sim.data.site_xmat[sid].reshape(3, 3).copy())


def current_object(env, name):
    bid = env.obj_body_id[name]
    return pose(env.sim.data.body_xpos[bid].copy(), env.sim.data.body_xmat[bid].reshape(3, 3).copy())


def action_for_pose(env, desired, grip):
    c = env.robots[0].controller
    assert c.use_delta and c.position_limits is None and c.orientation_limits is None
    assert np.array_equal(c.input_max, np.ones(6)) and np.array_equal(c.input_min, -np.ones(6))
    assert np.array_equal(c.output_max, -c.output_min)
    now = current_eef(env)
    dp = (desired[:3, 3] - now[:3, 3]) / c.output_max[:3]
    dr = rotation_vector(desired[:3, :3].dot(now[:3, :3].T)) / c.output_max[3:6]
    return np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip].astype(np.float32)


def interpolate(a, b, count):
    qa = T.mat2quat(a[:3, :3]); qb = T.mat2quat(b[:3, :3])
    for i in range(1, count + 1):
        t = i / count
        yield pose((1-t)*a[:3, 3] + t*b[:3, 3], T.quat2mat(T.quat_slerp(qa, qb, t)))


def correction_plan(env, skill):
    """One fixed source and one bounded plan; no candidate search or offsets."""
    obj = current_object(env, skill['subject'])
    transform = obj @ np.linalg.inv(np.asarray(skill['source_object_pose']))
    targets = np.asarray([transform @ np.asarray(p) for p in skill['commanded_poses']])
    now = current_eef(env)
    # Lift and travel at the successful source's observed clearance, then descend.
    # The height and spacing are source-derived, never tuned on target outcomes.
    clearance = max(float(now[2, 3]), float(obj[2, 3] + skill['source_clearance_above_object']))
    up = now.copy(); up[2, 3] = clearance
    across = targets[0].copy(); across[2, 3] = clearance
    points = [now, up, across, targets[0]]
    connections = []
    for a, b in zip(points, points[1:]):
        distance = np.linalg.norm(b[:3, 3] - a[:3, 3])
        angle = np.linalg.norm(rotation_vector(b[:3, :3].dot(a[:3, :3].T)))
        n = max(1, math.ceil(distance / skill['source_p95_translation_step']), math.ceil(angle / skill['source_max_rotation_step']))
        connections.extend((p, skill['gripper_actions'][0], 'source_clearance_connection') for p in interpolate(a, b, n))
    sequence = connections + [(p, g, 'source_grasp_segment') for p, g in zip(targets, skill['gripper_actions'])]
    return sequence, dict(connection_actions=len(connections), source_actions=len(targets), clearance_z=clearance, object_pose=obj.tolist(), first_transferred_target=targets[0].tolist(), source_to_target_transform=transform.tolist())
