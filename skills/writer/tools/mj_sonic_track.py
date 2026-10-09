#!/usr/bin/env python3
"""mj_sonic_track.py — staged closed-loop SONIC stack test in the OpenBot MuJoCo scene.

Runs the FULL SONIC stack (planner -> encoder -> decoder tracker) closed-loop in
skills/writer/sim/g1_gear_wbc.xml (loaded READ-ONLY; byte-pinned md5
8ffca967c299400100d80d2ce40a917f) and reports whether the robot stands.

Three staged tests, one JSON line each (A first, and non-negotiable):
  A) TRACKER-ONLY  : constant standing reference (SONIC default_angles), no planner. 6 s
  B) PLANNER IDLE  : planner mode 0, then tracker.                                    6 s
  C) PLANNER WALK  : planner mode 2, target_vel 1.0, then tracker.                    8 s
A result is only interpretable if A stands.

ACTUATION MODEL — a 50 Hz POSITION setpoint on a 500 Hz PD law:
  The reference deploy never computes a joint torque. Its 50 Hz Control thread publishes
  a position setpoint (q_target, kp, kd) to Unitree's motor controllers, which evaluate the
  PD law themselves, continuously, at ~1 kHz. This harness reproduces that: `q_target` is
  refreshed on the 50 Hz control tick, and the identical PD law is evaluated on EVERY
  500 Hz sim step from that setpoint. Holding the TORQUE across the intervening sim steps
  (a zero-order hold) instead of the SETPOINT is a sim-fidelity defect that makes the loop
  diverge for every reference; see the comment at the control seam for the evidence.

FALL CRITERION — NOT-FALLEN test, not a positive standing test:
  fell = (root tilt from vertical > 45 deg) OR (root z < 0.35)

HONESTY NOTE on that criterion: the nominal standing height in this scene is
0.75 m, so 0.35 is ~53% of body height — a robot collapsed into an upright
crouch/kneel with root z anywhere in [0.35, 0.75] and torso tilt < 45 deg is NOT
flagged. `CONTROL_A_STANDS` is the final line's string because the task spec
mandates it verbatim; read it as "A did not fall". The thresholds are the spec's
and are deliberately NOT tuned here (tuning a fall threshold is the single worst
outcome for this task). The raw z_min / max_tilt_deg numbers in each JSON record,
not the boolean label, are the trustworthy evidence.

The exit status is a machine-readable gate in addition to the printed lines:
0 only when A stands AND B stands AND C walks, 1 otherwise (see main()).
A alone is the non-negotiable precondition; B and C are now meaningful, so they gate too.

===============================================================================
CONSTANTS — imported from skills/sonic/sonic_core.py (SINGLE SOURCE OF TRUTH)
  Originally copied verbatim from the SONIC reference deploy header
  /home/leonlin/Unitree/GRAIL/imports/SONIC/gear_sonic_deploy/src/g1/
      g1_deploy_onnx_ref/include/policy_parameters.hpp
These are SONIC OWN gains. NOT the GR00T / merge_config gains — using OpenBot
leg_kp (150/200/40) or its default_angles here is a DEFECT.

That block was previously duplicated in this file. It now lives only in
sonic_core, because the writer seam and the satellite consume the same tables and
a drifted copy would silently invalidate the closed-loop proof below. The names
are re-exported unchanged; test_sonic_equivalence.py T1 asserts exact
np.array_equal parity for every table against sonic_core.

policy_parameters.hpp:100-101  (isaaclab order in mujoco index)
  const std::array<int, 29> isaaclab_to_mujoco = {0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
                                                  11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28};

policy_parameters.hpp:103-104  (mujoco order in isaaclab index)
  const std::array<int, 29> mujoco_to_isaaclab = {0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
                                                  16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28};

policy_parameters.hpp:109-139  g1_action_scale[i] = 0.25 * EFFORT_LIMIT_x / STIFFNESS_x   (muJoCo order)
policy_parameters.hpp:143-173  kps[i]             = STIFFNESS_x  (ankles/waist-roll/pitch x2)
policy_parameters.hpp:177-207  kds[i]             = DAMPING_x    (ankles/waist-roll/pitch x2)
policy_parameters.hpp:210-240  default_angles     = {-0.312,0,0,0.669,-0.363,0,  -0.312,0,0,0.669,-0.363,0,
                                                     0,0,0,  0.2,0.2,0,0.6,0,0,0,  0.2,-0.2,0,0.6,0,0,0}
policy_parameters.hpp:40-65    ARMATURE_{5020,7520_14,7520_22,4010}
                               EFFORT_LIMIT_{5020,7520_14,7520_22,4010} = 25, 88, 139, 5
policy_parameters.hpp:46-47    NATURAL_FREQ = 10*2*pi, DAMPING_RATIO = 2
                               STIFFNESS_x = ARMATURE_x * NATURAL_FREQ**2
                               DAMPING_x   = 2 * DAMPING_RATIO * ARMATURE_x * NATURAL_FREQ

Usage: python3 tools/mj_sonic_track.py [--xml PATH] [--models DIR] [--dump-states PATH]

--dump-states writes ONE .npz holding every control tick of all three tests
(tagged by test_index) for the offline equivalence test T2. Off by default; it
does not alter any control path.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
# sonic_core in skills/sonic
# imported top-level
SONIC_DIR = os.path.join(os.path.dirname(PKG), "sonic")
sys.path.insert(0, SONIC_DIR)

XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")
MODELS_DEFAULT = "/home/leonlin/Unitree/planner_models"

# ---------------------------------------------------------------------------
# SONIC constants, imported
# one copy: no drift
# T1: exact np.array_equal parity
# ---------------------------------------------------------------------------
from sonic_core import (  # noqa: E402
    NUM_MOTOR, QPOS_DIM, SIM_DT, CONTROL_EVERY, PLANNER_EVERY, PLANNER_HZ,
    CONTROL_HZ, DEFAULT_HEIGHT,
    ARMATURE_5020, ARMATURE_7520_14, ARMATURE_7520_22, ARMATURE_4010,
    NATURAL_FREQ, DAMPING_RATIO,
    EFFORT_LIMIT_5020, EFFORT_LIMIT_7520_14, EFFORT_LIMIT_7520_22,
    EFFORT_LIMIT_4010,
    STIFFNESS_5020, STIFFNESS_7520_14, STIFFNESS_7520_22, STIFFNESS_4010,
    DAMPING_5020, DAMPING_7520_14, DAMPING_7520_22, DAMPING_4010,
    ISAC_TO_MUJOCO, MUJOCO_TO_ISAC, DEFAULT_ANGLES_MUJOCO,
    KP_MUJOCO, KD_MUJOCO, ACTION_SCALE_MUJOCO, ALLOWED_PRED_NUM_TOKENS,
    ENC_MODE_OFF, ENC_JPOS_OFF, ENC_JVEL_OFF, ENC_ANCHOR_OFF, ENC_DIM,
    DEC_TOKEN_OFF, DEC_GYRO_OFF, DEC_JPOS_OFF, DEC_JVEL_OFF, DEC_ACT_OFF,
    DEC_GRAV_OFF, DEC_DIM, ENC_STEP, HIST_LEN, TILT_LIMIT_DEG, Z_FALL,
    quat_mul, quat_conj, quat_normalize, quat_to_mat, quat_rotate_inverse,
    quat_slerp, tilt_deg,
    Motion, make_standing_motion, resample_30_to_50, SonicStack,
    build_planner_context, anchor_orientation_6d, build_encoder_obs,
    build_decoder_obs,
)


# ---------------------------------------------------------------------------
# the sim loop
# ---------------------------------------------------------------------------
