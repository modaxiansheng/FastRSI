"""RGB proposals; version 1 preserves the accepted global-template behavior."""
from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import source_seed, put_on_support_plane, source_centers


@dataclass(frozen=True)
class ProposalPolicy:
    scales: tuple = tuple(.4+.05*i for i in range(23))
    size_envelope_px: int = 3


def _responses(image, template, source_pose, source_camera, current_camera,
               plane_delta, policy):
    sw = np.asarray(source_camera['T_world_camera_cv'])
    cw = np.asarray(current_camera['T_world_camera_cv'])
    K = np.asarray(current_camera['K'])
    source = np.asarray(source_pose)
    source_depth = (np.linalg.inv(sw) @ source)[2, 3]
    z = source[2, 3]+plane_delta
    for scale in policy.scales:
        width = max(3, round(template.shape[1]*scale))
        height = max(3, round(template.shape[0]*scale))
        q = cv2.resize(template, (width, height), interpolation=(
            cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
        response = cv2.matchTemplate(image, q, cv2.TM_CCOEFF_NORMED)
        yy, xx = np.indices(response.shape)
        uv = np.stack([255-(xx+width/2), yy+height/2, np.ones_like(xx)], axis=-1)
        rays = uv @ np.linalg.inv(K).T @ cw[:3, :3].T
        depth = (z-cw[2, 3])/rays[:, :, 2]
        expected = source_depth/depth
        valid = ((depth > 0) &
                 (abs(width-template.shape[1]*expected) <= policy.size_envelope_px) &
                 (abs(height-template.shape[0]*expected) <= policy.size_envelope_px))
        yield scale, width, height, response, expected, valid


def legacy_detection(image, template, record, source_pose, source_camera,
                     current_camera, plane_delta, policy=ProposalPolicy()):
    peaks = []
    for scale, width, height, response, expected, valid in _responses(
            image, template, source_pose, source_camera, current_camera, plane_delta, policy):
        masked = np.where(valid, response, -np.inf)
        _, score, _, loc = cv2.minMaxLoc(masked)
        if not np.isfinite(score):
            continue
        peaks.append(dict(template=record['path'], scale=scale, score=float(score),
                          bbox=[loc[0], loc[1], loc[0]+width, loc[1]+height],
                          expected_scale=float(expected[loc[1], loc[0]])))
    assert peaks, 'No source-size-compatible RGB location'
    selected = max(peaks, key=lambda p: p['score'])
    seed, proof = source_seed(selected['bbox'], record['source_crop'], source_pose,
                              source_camera, current_camera)
    seed, z = put_on_support_plane(seed, source_pose, current_camera, plane_delta)
    proof.update(method='SOURCE_RGB_MATCH_WITH_STATIC_SUPPORT_PLANE_SIZE_CONSTRAINT',
                 plane_delta_m=plane_delta, expected_initial_object_center_z=z,
                 size_quantization_envelope_px=policy.size_envelope_px)
    return selected, seed, proof, peaks


def distinct_rgb_proposals(image, template, record, source_pose, source_camera,
                           current_camera, plane_delta, limit=6):
    """Bounded alternate locations, not six scale variants of one false maximum.

    Proposal locations are determined only by current RGB and fixed calibration.
    Spatial suppression uses half a proposed object footprint, not result labels.
    No pose gate is relaxed. Later ranking uses the frozen pose model, not NCC.
    """
    assert limit == 6
    maps = []
    for scale, width, height, response, expected, valid in _responses(
            image, template, source_pose, source_camera, current_camera,
            plane_delta, ProposalPolicy()):
        if plane_delta == 0.:
            yy, xx = np.indices(response.shape)
            world, seed_depth = source_centers(
                xx, yy, width, height, record['source_crop'], source_pose,
                source_camera, current_camera, np.asarray(source_pose)[2, 3])
            # Same table bounds as the existing gate. No new per-instance ROI.
            valid &= (np.isfinite(world).all(axis=2) & (seed_depth > 0) &
                      (abs(world[:, :, 0]) <= .35) & (abs(world[:, :, 1]) <= .8))
        maps.append([scale, width, height, np.where(valid, response, -np.inf), expected])
    proposals = []
    for index in range(limit):
        best = None
        for scale, width, height, response, expected in maps:
            _, score, _, loc = cv2.minMaxLoc(response)
            if np.isfinite(score) and (best is None or score > best['score']):
                best = dict(template=record['path'], scale=scale, score=float(score),
                            bbox=[loc[0], loc[1], loc[0]+width, loc[1]+height],
                            expected_scale=float(expected[loc[1], loc[0]]))
        if best is None:
            break
        seed, proof = source_seed(best['bbox'], record['source_crop'], source_pose,
                                  source_camera, current_camera)
        seed, _ = put_on_support_plane(seed, source_pose, current_camera, plane_delta)
        proposals.append(dict(index=index, detection=best, seed=seed, proof=proof))
        x1, y1, x2, y2 = best['bbox']
        center = ((x1+x2)/2, (y1+y2)/2)
        for _, w, h, response, _ in maps:
            yy, xx = np.indices(response.shape)
            near = ((abs(xx+w/2-center[0]) < max(w, x2-x1)/2) &
                    (abs(yy+h/2-center[1]) < max(h, y2-y1)/2))
            response[near] = -np.inf
    return proposals
