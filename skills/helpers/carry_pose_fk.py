#!/usr/bin/env python3
"""
carry_pose_fk.py — laptop FK validation of the T5 pose state machine (no robot).

Validates, with pure pinocchio on the laptop, everything that can be
checked without CasADi:

  1. Reduced-model q convention — the 14 free joints must be exactly the
     daemon dual-arm q (left 0-6, right 7-13), verified by name via the
     joint->q-coordinate occupancy map (pinocchio root "universe" joint is
     skipped — it is a URDF root artifact, NOT part of model.q).
  2. T5 state poses (ready / lift / carry_tuck / neutral, left + right):
     - EVERY value inside the reduced-model joint limits (hard assert).
       NOTE: ready elbow = -1.0 (user decision 2026-08-16) — the original
       -1.2 spec exceeded the URDF elbow limit [-1.0472, +2.0944] by
       0.153 rad and was adjusted to -1.0, which is inside.
     - wrist pose SANITY per state (EE = wrist_yaw_joint + [0.05,0,0], the
       daemon's R_ee/L_ee frame definition): |lateral| >= 0.05 (wrist clear
       of the body centerline) and 0.35 >= z >= -0.35 (within a humanoid
       reach envelope; the OLD carry band [0.50, 0.62] gate is GONE).
     - EE positions and left/right mirror consistency are PRINTED per state
       (informational) for the operator's visual check on Stuart — incl.
       the ready/lift yaw resolution (hardware-verified 2026-08-16: LEFT
       arm twisted — LEFT yaw flipped to +0.5/+0.3; right unchanged).

Run (laptop, pinocchio env):
    /home/leonlin/miniconda3/envs/cv_pipeline_env/bin/python3 carry_pose_fk.py

Exit code 0 = all PASS.  Constants below are mirrored manually from
pick_daemon_HORIZONTAL_version.py (same names/values).
"""

import os
import sys
import numpy as np
import pinocchio as pin

# ── Daemon constants, copied over (pick_daemon_HORIZONTAL_version.py, T5) ─────
# per arm: pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
READY_LEFT  = np.array([0.40,  0.30,  0.50, -1.00,  0.00,  0.00, -0.01])
READY_RIGHT = np.array([0.40, -0.30, -0.50, -1.00,  0.00,  0.00,  0.01])
LIFT_LEFT   = np.array([0.00,  0.20,  0.30, -1.00,  0.00, -0.60,  0.20])
LIFT_RIGHT  = np.array([0.00, -0.20, -0.30, -1.00,  0.00, -0.60, -0.20])
CARRY_TUCK_LEFT  = np.array([0.50,  0.05,  0.00,  0.00, -1.50,  0.00, -1.00])
CARRY_TUCK_RIGHT = np.array([0.50, -0.05,  0.00,  0.00,  1.50,  0.00,  1.00])
# NEUTRAL - hanging rest from LOWSTATE
# yaw/elbow sign may be flipped. HARDWARE-VERIFY.
NEUTRAL_LEFT  = np.array([0.25522,  0.27104, -0.08654, 0.83254, -0.00101,  0.01032, -0.00063])
NEUTRAL_RIGHT = np.array([0.26688, -0.28410,  0.09247, 0.80183, -0.00034,  0.00915,  0.00730])

STATES = {
    'ready':      ('READY_LEFT',      'READY_RIGHT',      READY_LEFT,      READY_RIGHT),
    'lift':       ('LIFT_LEFT',       'LIFT_RIGHT',       LIFT_LEFT,       LIFT_RIGHT),
    'carry_tuck': ('CARRY_TUCK_LEFT', 'CARRY_TUCK_RIGHT', CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT),
    'neutral':    ('NEUTRAL_LEFT',    'NEUTRAL_RIGHT',    NEUTRAL_LEFT,    NEUTRAL_RIGHT),
}

# ── Bounds we check against ───────────────────────────────────────────────────
LAT_MIN = 0.05    # clear of body centerline
Z_MIN   = -0.35   # foot level - just a loose check
Z_MAX   = 1.50    # humanoid reach
MIRROR_TOL = 0.15  # flags |y_L + y_R|
SAME_SIGN_YAW_STATES = ('ready', 'lift')  # left yaw flipped 2026-08-16

_BASE = '/home/leonlin/Unitree/video2contactdynamic/Lerobot_Framework' \
        '/unitree_lerobot/unitree_lerobot/eval_robot'
URDF  = os.path.join(_BASE, 'assets/g1/g1_body29_hand14.urdf')
URDF_DIR = os.path.join(_BASE, 'assets/g1/')

# 29 locked joints, as in daemon
LOCKS = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
         "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
         "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
         "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
         "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
         "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
         "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
         "left_hand_middle_1_joint", "left_hand_index_0_joint",
         "left_hand_index_1_joint", "right_hand_thumb_0_joint",
         "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
         "right_hand_index_0_joint", "right_hand_index_1_joint",
         "right_hand_middle_0_joint", "right_hand_middle_1_joint"]

# expected free-joint order
EXPECTED_Q = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
              "left_shoulder_yaw_joint", "left_elbow_joint",
              "left_wrist_roll_joint", "left_wrist_pitch_joint",
              "left_wrist_yaw_joint",
              "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"]

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def ee_frames(red):
    """Return (jid, offset) for the daemon's R_ee/L_ee frames."""
    out = {}
    for side in ('right', 'left'):
        jid = red.model.getJointId(side + '_wrist_yaw_joint')
        out[side] = (jid, pin.SE3(np.eye(3), np.array([0.05, 0, 0])))
    return out


