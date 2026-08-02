#!/usr/bin/env python3
"""
manual_arm_joints.py — manual arm-joint manipulation tool for Stuart.

Run in a clean shell with the tv conda env (NO ROS2 env), like the pick daemon:
    /home/unitree/miniconda3/envs/tv/bin/python3 manual_arm_joints.py eth0 [--upper-only]
        ... then interactive REPL, OR one of:
    --right <pitch roll yaw elbow wr wp wy>     move RIGHT arm to these 7 joints
    --left  <pitch roll yaw elbow wr wp wy>     move LEFT arm to these 7 joints
    --both  <14 values: left 7 then right 7>    move both arms
    --pose <name>                               load a pose from manual_poses.json
    --state                                     print current joints and exit

Dual-arm q layout (14-DoF, same as the pick daemon):
    q[0:7]  = LEFT  arm   [shoulder_pitch, shoulder_roll, shoulder_yaw,
                           elbow, wrist_roll, wrist_pitch, wrist_yaw]
    q[7:14] = RIGHT arm   (same order)

REPL commands:
    right <7 vals> | left <7 vals> | both <14 vals> | state
    ready | lift | carry_tuck | carry_tuck_right | carry_tuck_left |
    neutral | both | home | quit

Moves are smooth cosine interpolations (0.3 rad/s, min 1.5 s) — never instant.
Values beyond +/-3.0 rad trigger a warning but still execute.

TASK 7 GUARD (2026-08-16): refuses to start while the pick daemon stack is
alive (pick_node / pick_daemon processes).  Running this tool while the
upper-only daemon holds the arm channels puts a SECOND arm_sdk writer on the
bus — the 2026-08-16 incident flipped fsm_mode 0 -> 3 (gait blocked).  Stop
the stack first; the guard applies with AND without --upper-only.
"""
import os
import sys
import json
import time
import numpy as np

os.chdir('/home/unitree/xr_teleoperate/teleop/robot_control')
sys.path.insert(0, '/home/unitree/xr_teleoperate')
sys.path.insert(0, '/home/unitree/unitree_sdk2_python')
sys.path.insert(0, '/home/unitree/leon')

from teleop.robot_control.robot_arm import G1_29_ArmController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

MOVE_SPEED = 0.3          # rad/s, same as the pick daemon
MOVE_HZ = 50
MAX_WARN_RAD = 3.0

JOINT_NAMES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
               'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']

# preset constants, identical to the pick daemon's
# per-arm: pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
# READY elbow -1.0 fits URDF limit [-1.0472, +2.0944];
# -1.2 overshoots by 0.153 rad; -1.0 stays inside the URDF limit
READY_LEFT = np.array([0.40, 0.30, 0.50, -1.00, 0.00, 0.00, -0.01])
READY_RIGHT = np.array([0.40, -0.30, -0.50, -1.00, 0.00, 0.00, 0.01])
LIFT_LEFT = np.array([0.00, 0.20, 0.30, -1.00, 0.00, -0.60, 0.20])
LIFT_RIGHT = np.array([0.00, -0.20, -0.30, -1.00, 0.00, -0.60, -0.20])
CARRY_TUCK_LEFT = np.array([0.50, 0.05, 0.00, 0.00, -1.50, 0.00, -1.00])
CARRY_TUCK_RIGHT = np.array([0.50, -0.05, 0.00, 0.00, 1.50, 0.00, 1.00])
# NEUTRAL: lowstate-sourced hanging rest.  SIGN RISK:
# yaw/elbow signs may be flipped
# vs the arm_sdk COMMAND convention
# HARDWARE-VERIFY on Stuart first
# flip only acting-up joints: yaw, then roll.  elbow is out:
# -1.0 is in-limits and user-approved
NEUTRAL_LEFT = np.array([0.25522, 0.27104, -0.08654, 0.83254,
                         -0.00101, 0.01032, -0.00063])
NEUTRAL_RIGHT = np.array([0.26688, -0.28410, 0.09247, 0.80183,
                          -0.00034, 0.00915, 0.00730])

# the LEFT arm
# twisted toward the chest on same-sign yaw pairs ->
# LEFT yaw flipped POSITIVE (+0.5 ready / +0.3 lift)
# right arm fine, unchanged

PRESETS = {
    'ready': (READY_LEFT, READY_RIGHT),
    'lift': (LIFT_LEFT, LIFT_RIGHT),
    'carry_tuck': (CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT),
    'carry_tuck_right': (None, CARRY_TUCK_RIGHT),
    'carry_tuck_left': (CARRY_TUCK_LEFT, None),
    'neutral': (NEUTRAL_LEFT, NEUTRAL_RIGHT),
    # 'both': both arms in carry_tuck walking posture
    # (carry_tuck_* single-arm = per-arm verification)
    'both': (CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT),
}

POSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'manual_poses.json')


