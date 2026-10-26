"""Fixed W skill with RGB-estimated object pose and public proprioception only.

No environment/simulator is accepted by these control functions. The imported
interpolation and rotation math is the frozen source skill's implementation.
"""
import math
import numpy as np
from object_frame_skill import pose, rotation_vector, interpolate


def public_eef(obs):
    pos = np.asarray(obs['robot0_eef_pos'], dtype=np.float64)
    rot = np.asarray(obs['robot0_eef_mat'], dtype=np.float64)
    assert pos.shape == (3,) and rot.shape == (3, 3)
    assert np.isfinite(pos).all() and np.isfinite(rot).all()
    assert np.max(np.abs(rot.T @ rot - np.eye(3))) < 1e-5
    assert abs(np.linalg.det(rot) - 1) < 1e-5
    return pose(pos, rot)


def action_for_pose(obs, desired, grip, skill):
    # Frozen static action scales, not dynamic privileged controller state.
    c = skill['controller_parameters']
    assert c['use_delta'] and c['position_limits'] is None and c['orientation_limits'] is None
    assert np.array_equal(c['input_max'], np.ones(6))
    assert np.array_equal(c['input_min'], -np.ones(6))
    scale = np.asarray(c['output_max'], dtype=np.float64)
    assert np.array_equal(scale, -np.asarray(c['output_min']))
    now = public_eef(obs)
    dp = (desired[:3, 3] - now[:3, 3]) / scale[:3]
    dr = rotation_vector(desired[:3, :3] @ now[:3, :3].T) / scale[3:6]
    return np.r_[np.clip(dp, -1, 1), np.clip(dr, -1, 1), grip].astype(np.float32)


def correction_plan(obs, skill, rgb_object_pose):
    obj = np.asarray(rgb_object_pose, dtype=np.float64)
    assert obj.shape == (4, 4) and np.isfinite(obj).all()
    transform = obj @ np.linalg.inv(np.asarray(skill['source_object_pose']))
    targets = np.asarray([transform @ np.asarray(p) for p in skill['commanded_poses']])
    now = public_eef(obs)
    clearance = max(float(now[2, 3]), float(obj[2, 3] + skill['source_clearance_above_object']))
    up = now.copy(); up[2, 3] = clearance
    across = targets[0].copy(); across[2, 3] = clearance
    points = [now, up, across, targets[0]]
    connections = []
    for a, b in zip(points, points[1:]):
        distance = np.linalg.norm(b[:3, 3] - a[:3, 3])
        angle = np.linalg.norm(rotation_vector(b[:3, :3] @ a[:3, :3].T))
        n = max(1, math.ceil(distance / skill['source_p95_translation_step']),
                math.ceil(angle / skill['source_max_rotation_step']))
        connections.extend((p, skill['gripper_actions'][0], 'source_clearance_connection')
                           for p in interpolate(a, b, n))
    sequence = connections + [(p, g, 'source_grasp_segment')
                              for p, g in zip(targets, skill['gripper_actions'])]
    return sequence, dict(connection_actions=len(connections), source_actions=len(targets),
                          clearance_z=clearance, object_pose=obj.tolist(),
                          first_transferred_target=targets[0].tolist(),
                          source_to_target_transform=transform.tolist(),
                          object_pose_source='FROZEN_RGB_NEURAL_CACHE',
                          feedback_source='PUBLIC_EEF_POS_AND_MAT',
                          current_simulator_object_pose_read_by_planner=False)


def require_cached_rgb(obs, phase, cache):
    import hashlib
    pixels = np.ascontiguousarray(obs['agentview_image'], dtype=np.uint8)
    expected = cache[phase]
    actual = hashlib.sha256(pixels.tobytes()).hexdigest()
    assert list(pixels.shape) == expected['shape'] and actual == expected['decoded_model_rgb_sha256'], \
        'RGB cache miss: no simulator pose fallback and no physical recovery'
    return dict(phase=phase, decoded_model_rgb_sha256=actual,
                original_png_sha256=expected['original_png_sha256'],
                exact_input_match=True, fresh_pose_inference=False)
