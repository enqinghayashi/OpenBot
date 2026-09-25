#!/usr/bin/env python3
"""sonic_core.py — THE single source of truth for the SONIC stack's constants
and pure-numpy helpers.

Extracted VERBATIM from skills/writer/tools/mj_sonic_track.py so that the writer
seam, the satellite process, and the offline harness CANNOT drift apart. Any
table edited here changes all three; the equivalence test
(skills/sonic/tools/test_sonic_equivalence.py, T1) asserts exact array equality
between this module and the harness's re-exported names.

PROVENANCE of every constant below: the SONIC reference deploy header
  /home/leonlin/Unitree/GRAIL/imports/SONIC/gear_sonic_deploy/src/g1/
      g1_deploy_onnx_ref/include/policy_parameters.hpp

These are SONIC OWN gains. They are NOT the GR00T / merge_config gains — using
OpenBot leg_kp (150/200/40) or its default_angles with SONIC's action output is a
DEFECT (knee 200 vs 99.098 = 2.02x, waist-pitch 250 vs 28.501 = 8.77x).

  policy_parameters.hpp:100-101  isaaclab_to_mujoco
  policy_parameters.hpp:103-104  mujoco_to_isaaclab
  policy_parameters.hpp:109-139  g1_action_scale[i] = 0.25*EFFORT_LIMIT_x/STIFFNESS_x
  policy_parameters.hpp:143-173  kps[i]  = STIFFNESS_x   (ankles/waist-roll/pitch x2)
  policy_parameters.hpp:177-207  kds[i]  = DAMPING_x     (ankles/waist-roll/pitch x2)
  policy_parameters.hpp:210-240  default_angles
  policy_parameters.hpp:40-65    ARMATURE_*, EFFORT_LIMIT_*
  policy_parameters.hpp:46-47    NATURAL_FREQ = 10*2*pi, DAMPING_RATIO = 2

IMPORT CONTRACT: this module imports numpy (and math) only. It must remain
importable inside a 250 Hz control process that is forbidden from loading
onnxruntime or mujoco.
"""

import math
import os

import numpy as np

# ---------------------------------------------------------------------------
# sizes and rates
# ---------------------------------------------------------------------------
NUM_MOTOR = 29
QPOS_DIM = 36          # 3 root pos + 4 quat + 29 joints
SIM_DT = 0.002         # 500 Hz tick
CONTROL_EVERY = 10     # 10 steps = 50 Hz
PLANNER_EVERY = 50     # 50 steps = 10 Hz
PLANNER_HZ = 30.0      # authored at 30 Hz
CONTROL_HZ = 50.0
LEG_WAIST_DOF = 15     # 12 leg + 3 waist, motor order

# harness only
# not a SONIC constant
# sim reset qpos[2]
# planner `height` input
# mj_sonic_track.py
# localmotionkplanner.hpp:216: 0.788740
# keep 0.75: the recorded run
# stays bit-identical
# satellite has its own
# `planner_height`
# don't fix without rerun
DEFAULT_HEIGHT = 0.75

# ---------------------------------------------------------------------------
# SONIC constants
# ---------------------------------------------------------------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# policy_parameters.hpp:100-101
# MOTOR in, ISAACLAB out
# (naming trap below)
# q_target = default + act[i2m]*scale
ISAC_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=np.int64)
# policy_parameters.hpp:103-104
# ISAACLAB in, MOTOR out
# np.argsort(i2m)
MUJOCO_TO_ISAC = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=np.int64)

# policy_parameters.hpp:210-240, mujoco
DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # left-leg joints
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,      # right-leg joints
    0.0, 0.0, 0.0,                              # waist joints
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,          # left-arm joints
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,         # right-arm joints
], dtype=np.float32)

# policy_parameters.hpp:143-173, mujoco
_KP_TAGS = [
    "7520_22", "7520_22", "7520_14", "7520_22", "2*5020", "2*5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "2*5020", "2*5020",
    "7520_14", "2*5020", "2*5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
_KD_TAGS = [
    "7520_22", "7520_22", "7520_14", "7520_22", "2*5020", "2*5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "2*5020", "2*5020",
    "7520_14", "2*5020", "2*5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
# policy_parameters.hpp:109-139
# action_scale = 0.25*EFFORT/STIFFNESS
# ankle/waist-rp: "5020"
# not "2*5020" like _KP_TAGS:
# divide by SINGLE-motor
# stiffness, two motors or not
# copying _KP_TAGS doubles 6
_AS_TAGS = [
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
    "7520_14", "5020", "5020",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",
]
_MOTOR_K = {
    "5020": (STIFFNESS_5020, EFFORT_LIMIT_5020),
    "7520_14": (STIFFNESS_7520_14, EFFORT_LIMIT_7520_14),
    "7520_22": (STIFFNESS_7520_22, EFFORT_LIMIT_7520_22),
    "4010": (STIFFNESS_4010, EFFORT_LIMIT_4010),
}
_MOTOR_D = {
    "5020": DAMPING_5020, "7520_14": DAMPING_7520_14,
    "7520_22": DAMPING_7520_22, "4010": DAMPING_4010,
}


def _gain(tag):
    return 2.0 * _MOTOR_K[tag[2:]][0] if tag.startswith("2*") else _MOTOR_K[tag][0]


def _damp(tag):
    return 2.0 * _MOTOR_D[tag[2:]] if tag.startswith("2*") else _MOTOR_D[tag]


KP_MUJOCO = np.array([_gain(t) for t in _KP_TAGS], dtype=np.float32)
KD_MUJOCO = np.array([_damp(t) for t in _KD_TAGS], dtype=np.float32)
ACTION_SCALE_MUJOCO = np.array(
    [0.25 * _MOTOR_K[t[2:] if t.startswith("2*") else t][1]
     / _MOTOR_K[t[2:] if t.startswith("2*") else t][0] for t in _AS_TAGS],
    dtype=np.float32)

# g1: 6 tokens
# (planner_models/observation_config.yaml)
ALLOWED_PRED_NUM_TOKENS = np.array(
    [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64)

# encoder 1762, g1 offsets
ENC_MODE_OFF = 0        # [0,4) mode_4 == 0 in g1
ENC_JPOS_OFF = 4        # [4,294) 10x29, ISAACLAB order
ENC_JVEL_OFF = 294      # [294,584) 10x29 jvel
ENC_ANCHOR_OFF = 601    # [601,661) 10x6 anchor
ENC_DIM = 1762

# decoder obs 994
DEC_TOKEN_OFF = 0       # [0,64)  the tokens
DEC_GYRO_OFF = 64       # [64,94) 10x3 gyro, body
DEC_JPOS_OFF = 94       # [94,384) 10x29 q - def
DEC_JVEL_OFF = 384      # [384,674) 10x29 dq
DEC_ACT_OFF = 674       # [674,964) prev actions
DEC_GRAV_OFF = 964      # [964,994) 10x3 gravity
DEC_DIM = 994

ENC_STEP = 5            # frames 0,5,...,45
HIST_LEN = 10           # SHARED: 2 jobs
                        # decoder history, oldest first
                        # encoder look-ahead 10 x ENC_STEP
                        # change it, both move

TILT_LIMIT_DEG = 45.0
Z_FALL = 0.35

# ---------------------------------------------------------------------------
# NAMING TRAP: the two conversions are mirrored;
# five call sites depend on the direction
#
# ISAC_TO_MUJOCO: MOTOR in, ISAACLAB out
# MUJOCO_TO_ISAC: ISAACLAB in, MOTOR out
#
# names say the opposite of the use
# right:
#   q_isaaclab = q_motor[MUJOCO_TO_ISAC]
# wrong (name-following trap):
#   q_isaaclab = q_motor[ISAC_TO_MUJOCO]
#   q_target_motor = default
#     + action[MUJOCO_TO_ISAC]
# standing can't tell them apart
# (legs + defaults are symmetric)
# walking does:
#   action[ISAC_TO_MUJOCO] -> walks,
#     tilt 10.3 deg, vx +1.462 m/s
#   action[MUJOCO_TO_ISAC] -> falls,
#     tilt 121.9 deg
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# all quats w,x,y,z
# ---------------------------------------------------------------------------
def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_to_mat(q):
    """Rotation matrix R (world-from-body) for q=(w,x,y,z)."""
    w, x, y, z = quat_normalize(np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_rotate_inverse(q_wxyz, v):
    """Rotate v into the body frame: R(q)^T @ v."""
    return quat_to_mat(q_wxyz).T @ np.asarray(v, dtype=np.float64)


