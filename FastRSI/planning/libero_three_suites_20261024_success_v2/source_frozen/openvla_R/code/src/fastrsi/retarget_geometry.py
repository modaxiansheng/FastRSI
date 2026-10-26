"""Rigid object-relative waypoint transform for offline feasibility probes.

Standard rigid-body algebra; no third-party implementation copied. This does not
estimate object poses, detect task phases, or certify collision-free motion.
"""
import numpy as np


def pose(position, rotation):
    p, r = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
    if p.shape != (3,) or r.shape != (3, 3) or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError('Invalid rigid pose')
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
        raise ValueError('Invalid proper rotation')
    return p, r


def retarget(position, rotation, source_position, source_rotation, target_position, target_rotation):
    """Preserve the source waypoint's pose relative to its object anchor."""
    p, r = pose(position, rotation)
    sp, sr = pose(source_position, source_rotation)
    tp, tr = pose(target_position, target_rotation)
    change = tr @ sr.T
    return tp + change @ (p-sp), change @ r


def rotation_vector(rotation):
    """Double precision SO(3) logarithm, including small angles and half turns."""
    _, r = pose([0, 0, 0], rotation)
    skew = np.array([r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]) / 2
    sine = np.linalg.norm(skew)
    cosine = np.clip((np.trace(r)-1)/2, -1, 1)
    theta = np.arctan2(sine, cosine)
    if cosine < 0 and sine < 1e-6:
        _, vectors = np.linalg.eigh((r+r.T)/2)
        axis = vectors[:, -1]
        if np.dot(axis, skew) < 0:
            axis = -axis
        return theta*axis
    return skew if sine < 1e-10 else theta/sine*skew
