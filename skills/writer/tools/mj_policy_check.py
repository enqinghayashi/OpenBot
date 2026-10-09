#!/usr/bin/env python3
"""mj_policy_check.py — MuJoCo sim gate for the C3 policy stack (laptop only).

Drives the GR00T Balance/Walk ONNX policies through the EXACT obs stack the
writer uses (obs_builder + policy_runner) in the g1_gear_wbc.xml sim.

Gates:
  stand:  base z settles within [0.65, 0.85] m of height_cmd 0.74, no fall
          for 12 s with zero cmd
  walk:   cmd (0.3, 0, 0) -> forward displacement 0.15-0.45 m/s avg, no fall
          for 15 s

Usage:  python3 tools/mj_policy_check.py [--xml <path>] [--onnx-dir <dir>]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mujoco

from obs_builder import ObsBuilder
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")
# self-contained: meshes pinched from unitree_ros g1_d_description
# (GRAIL's own are just Git-LFS stubs)
XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")

# XML order, matches G1JointIndex 0..14
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]


def quat_to_wxyz_identity():
    return np.array([1.0, 0.0, 0.0, 0.0])
