"""OmniGibson observation keys for BEHAVIOR-1K R1Pro (matches eval.py / serve_b1k)."""

from __future__ import annotations

HEAD_RGB_KEY = "robot_r1::robot_r1:zed_link:Camera:0::rgb"
LEFT_WRIST_RGB_KEY = "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"
RIGHT_WRIST_RGB_KEY = "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"
PROPRIO_KEY = "robot_r1::proprio"

B1K_ACTION_DIM = 23
