"""Offline evidence for contact-supported object motion, not a grasp certificate.

Privileged replay quantities are label evidence only. This module does not
authorize a skill, route a policy, or interpret whole-episode success as a
counterfactual decision label.
"""
from __future__ import annotations

import math


PROTOCOL = {
    "version": "contact_supported_motion_evidence_v1",
    "boundary_count": 3,
    "min_object_height_change_m": 0.03,
    "min_ee_translation_m": 0.01,
    "max_object_relative_translation_drift_m": 0.01,
    "max_boundary_gap_steps": 10,
    "label_is_grasp_certificate": False,
    "label_is_reuse_value": False,
    "online_privileged_input": False,
}


def carry_evidence(history):
    """Classify the most recent three recorded boundaries of one object.

    The positions use the same world frame. Two-sided finger contact is not a
    requirement: some real source motions do not satisfy native _check_grasp.
    Unknown covers brief contact, little motion, and inconsistent attachment.
    """
    if len(history) < 3:
        return {"label": "UNKNOWN", "reason": "insufficient_boundaries"}
    rows = history[-3:]
    for row in rows:
        if (type(row.get("step")) is not int or type(row.get("contact")) is not bool
                or type(row.get("closed_command")) is not bool
                or not math.isfinite(row.get("initial_object_z", float("nan")))):
            raise ValueError("Malformed offline carry evidence")
        for key in ("object_world_position", "grip_site_world_position"):
            value = row.get(key)
            if (not isinstance(value, list) or len(value) != 3
                    or any(type(x) not in (float, int) or not math.isfinite(x) for x in value)):
                raise ValueError("Malformed world-frame replay positions")
    gaps = [b["step"] - a["step"] for a, b in zip(rows, rows[1:])]
    if any(not 0 < gap <= PROTOCOL["max_boundary_gap_steps"] for gap in gaps):
        return {"label": "UNKNOWN", "reason": "discontinuous_boundaries"}
    if len({r["initial_object_z"] for r in rows}) != 1:
        raise ValueError("Initial object reference changed")
    relative = [[a-b for a, b in zip(r["object_world_position"], r["grip_site_world_position"])]
                for r in rows]
    drift = max(math.dist(relative[0], rel) for rel in relative[1:])
    movement = max(math.dist(rows[0]["grip_site_world_position"], r["grip_site_world_position"])
                   for r in rows[1:])
    lift = rows[-1]["object_world_position"][2] - rows[-1]["initial_object_z"]
    facts = {"relative_translation_drift_m": drift, "ee_translation_m": movement,
             "current_object_height_change_m": lift,
             "contact_boundaries": sum(r["contact"] for r in rows)}
    if not any(r["contact"] for r in rows):
        return {**facts, "label": "NO_CURRENT_CONTACT", "reason": "three_contact_free_boundaries"}
    supported = (all(r["contact"] and r["closed_command"] for r in rows)
                 and lift >= PROTOCOL["min_object_height_change_m"]
                 and movement >= PROTOCOL["min_ee_translation_m"]
                 and drift <= PROTOCOL["max_object_relative_translation_drift_m"])
    return {**facts, "label": "SUPPORTED_MOTION_WITH_CONTACT" if supported else "UNKNOWN",
            "reason": "contact_lift_and_comotion" if supported else "attachment_not_established"}
