#!/usr/bin/env python3
"""sim_joint_map.py — NAME-KEYED joint / actuator indexing for the Dex3-unlocked
derived scene (tools/g1_gear_wbc_fingers.xml).

WHY THIS FILE EXISTS
--------------------
The Dex3 unlock (see mj_finger_grip.unlock_scene_text) un-comments the 14 hand
joints of the byte-pinned base scene.  The 7 LEFT-hand joints are inserted at
qposadr 29..35, which SHIFTS the whole right-arm chain by +7:

    right_shoulder_pitch_joint  29 -> 36      right_wrist_roll_joint   33 -> 40
    right_shoulder_roll_joint   30 -> 37      right_wrist_pitch_joint  34 -> 41
    right_shoulder_yaw_joint    31 -> 38      right_wrist_yaw_joint    35 -> 42
    right_elbow_joint           32 -> 39

No pre-existing joint disappears.  Any simulator code that addresses joints by a
hardcoded qpos/qvel SLICE, or that selects "the arm actuators" as
`[i for i in range(nu) if i not in policy_aids]`, will silently drive the WRONG
joint after the unlock (with 14 finger actuators that expression returns 28, not
14, and mj_payload_sweep.py:574's assert aborts).  Everything here is BY NAME.

The EXPECTED_QPOSADR fingerprint below is the assertion that CATCHES a silent
index shift; assert_fingerprint() names the offending joint on failure.

Document order (verified against the compiled unlocked model, both hands):
    thumb_0, thumb_1, thumb_2, middle_0, middle_1, index_0, index_1
NOTE this is the URDF/XML DECLARATION order.  It is NOT the DDS motor order in
grip_server.py (right: thumb, index, middle; left: thumb, middle, index) — that
motor order lives in DEX3_JOINT_NAMES and is consumed only to key presets onto
joint NAMES, never onto positions.
"""

import numpy as np

# --- arm / policy joint names, in the order arm_guard and mj_payload_sweep use
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# --- all 14 finger joints, XML declaration order, both hands ------------------
FINGER_ORDER = ["thumb_0", "thumb_1", "thumb_2",
                "middle_0", "middle_1", "index_0", "index_1"]

# DDS motor order, NOT declaration order
DEX3_JOINT_NAMES = {
    "right": {0: "thumb_0", 1: "thumb_1", 2: "thumb_2",
              3: "index_0", 4: "index_1", 5: "middle_0", 6: "middle_1"},
    "left":  {0: "thumb_0", 1: "thumb_1", 2: "thumb_2",
              3: "middle_0", 4: "middle_1", 5: "index_0", 6: "index_1"},
}
DEX3_JOINTS = [0, 1, 2, 3, 4, 5, 6]


def finger_joint_names(side):
    """The 7 joint names of one hand in XML declaration order."""
    return [f"{side}_hand_{f}_joint" for f in FINGER_ORDER]


FINGER_JOINT_NAMES = finger_joint_names("left") + finger_joint_names("right")


def preset_to_joint_map(side, preset):
    """Turn a grip_server {motor_id: q} preset into {joint_name: q}.

    The ONLY legal mapping.  Uses DEX3_JOINT_NAMES so a preset index lands on
    the joint it actually names (right 3 == index_0, left 3 == middle_0).
    """
    return {f"{side}_hand_{DEX3_JOINT_NAMES[side][jid]}_joint": float(preset[jid])
            for jid in DEX3_JOINTS}


# --- model lookups, everything by NAME ---------------------------------------
def has_joint(model, name):
    """True iff the model declares a joint with this name.

    model.joint(name) RAISES KeyError for an undeclared name in MuJoCo 3.11
    (it does not return a sentinel), so existence must be tested by scanning.
    """
    return any(model.joint(i).name == name for i in range(model.njnt))


def joint_id(model, name):
    i = model.joint(name).id
    if i < 0:
        raise KeyError(f"joint {name!r} not in model")
    return i


def qposadr(model, name):
    return int(model.jnt_qposadr[joint_id(model, name)])


def dofadr(model, name):
    return int(model.jnt_dofadr[joint_id(model, name)])


def actuator_id(model, name):
    i = model.actuator(name).id
    if i < 0:
        raise KeyError(f"actuator {name!r} not in model")
    return i


def arm_actuator_ids(model):
    return np.array([actuator_id(model, n) for n in ARM_JOINT_NAMES], dtype=int)


def finger_actuator_ids(model, side=None):
    names = finger_joint_names(side) if side else FINGER_JOINT_NAMES
    return np.array([actuator_id(model, n) for n in names], dtype=int)
