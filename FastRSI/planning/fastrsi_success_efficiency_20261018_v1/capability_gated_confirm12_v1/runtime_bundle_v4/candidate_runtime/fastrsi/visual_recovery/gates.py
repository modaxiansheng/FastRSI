"""Existing initial-pose acceptance, with source data and variant explicit."""
import numpy as np


def initial_pose_gate(rgb_pose, skill, variant, static_calibration):
    t = np.asarray(rgb_pose, dtype=float)
    rigid = (t.shape == (4, 4) and np.isfinite(t).all() and
             np.allclose(t[3], [0, 0, 0, 1], atol=1e-6) and
             np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4) and
             abs(np.linalg.det(t[:3, :3])-1) < 1e-4)
    if not rigid:
        return dict(status='UNKNOWN', reason='INVALID_RGB_POSE')
    source = np.asarray(skill['source_object_pose'])
    local_up = source[:3, :3].T @ np.array([0., 0., 1.])
    tilt = float(np.degrees(np.arccos(np.clip((t[:3, :3] @ local_up)[2], -1., 1.))))
    decision = dict(status='ACCEPT' if tilt <= 15. else 'UNKNOWN',
                    reason=None if tilt <= 15. else 'SOURCE_SUPPORTED_RESTING_ORIENTATION_UNCERTAIN',
                    tilt_deg=tilt)
    if decision['status'] != 'ACCEPT' or skill['subject'] != 'chocolate_pudding_1':
        return decision
    assert variant in ('position', 'environment')
    expected = source[2, 3]-(.445 if variant == 'position' else 0.)
    if variant == 'environment' and (abs(t[0, 3]) > .35 or abs(t[1, 3]) > .8):
        return dict(status='UNKNOWN', reason='INITIAL_RGB_POSE_OUTSIDE_STATIC_SUPPORT_FIXTURE',
                    original_gate=decision)
    cal = static_calibration[variant]['cameras']['agentview']
    depth = (np.linalg.inv(np.asarray(cal['T_world_camera_cv'])) @ t)[2, 3]
    tolerance = 3*depth/cal['K'][0][0]
    if depth <= 0 or abs(t[2, 3]-expected) > tolerance:
        return dict(status='UNKNOWN', reason='INITIAL_RGB_POSE_NOT_ON_STATIC_SUPPORTED_RESTING_PLANE',
                    original_gate=decision, height_error_m=float(t[2, 3]-expected),
                    pixel_metric_envelope_m=float(tolerance))
    return dict(**decision, static_geometry_admissible=True,
                static_fixture_input=True, current_simulator_pose_input=False)
