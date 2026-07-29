#!/usr/bin/env python3
"""
upper_arm_controller.py — G1_29_ArmControllerUpperOnly (T4, Task 2)

Upper-only variant of G1_29_ArmController: the daemon streams arm joint
commands while the robot holds a pick (stationary, non-801), so this
controller drives only the two arms (15-28) and the arm_sdk enable-gate
joint 29 (kNotUsedJoint0).  The waist (12-14) AND the legs (0-11) are
locked (mode=1) at the captured pose and held — never driven.  Locking
the legs mirrors the stock G1_29_ArmController (all 35 joints mode=1),
which the arm_sdk gate requires: with the legs limp (mode=0) the gate is
not honored and the arms run limp (effective kp ~2 Nm/rad vs commanded).

Deployment: copy to Stuart's /home/unitree/leon/upper_arm_controller.py
(already on the daemon's sys.path).  Selected by the daemon argv flag
`upper` (pick_daemon_HORIZONTAL_version.py main()).

Differences vs G1_29_ArmController (robot_arm.py):
  * init locks arm joints 15-28 (mode=1, kp/kd per joint class), the
    gate 29, the waist 12-14 (mode=1, kp_low/kd_low, q seeded at the
    captured pose — held, not driven), the legs 0-11 (mode=1,
    kp_high/kd_high, q seeded at the captured pose — held, not driven),
    AND the trailing unused slots 30-34 (mode=1, kp_high/kd_high, q
    seeded at the captured pose — persist at the init value, never
    driven; the stock locks ALL 35 joints, and the G1 firmware trips if
    the trailing slots are mode=0 in Running Mode (801)).
  * the ctrl loop writes arm joints 15-28 + gate 29 + the locked
    waist 12-14 AND legs 0-11 (q held at the captured pose, dq/tau=0).
  * WRITE_WAIST = True (waist locked for the arm_sdk gate).
  * adds release_arms() / acquire_arms() (101-step gate ramps) and
    hold_only().

API-compatible with G1_29_ArmController: ctrl_dual_arm,
get_current_dual_arm_q, get_current_dual_arm_dq, ctrl_dual_arm_go_home,
speed_gradual_max, speed_instant_max, clip_arm_q_target,
get_current_motor_q, get_mode_machine.
"""

import numpy as np
import sys
import threading
import time

from unitree_sdk2py.core.channel import (ChannelPublisher, ChannelSubscriber,
                                         ChannelFactoryInitialize)  # dds imports
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (LowCmd_ as hg_LowCmd,
                                                    LowState_ as hg_LowState)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

# joint enums + lowstate from the stock controller
from teleop.robot_control.robot_arm import (G1_29_ArmController,
                                            G1_29_JointArmIndex,
                                            G1_29_JointIndex,
                                            G1_29_LowState,
                                            G1_29_Num_Motors,
                                            DataBuffer)

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)

kTopicLowCommand_Motion = "rt/arm_sdk"
kTopicLowState = "rt/lowstate"

GATE_JOINT = G1_29_JointIndex.kNotUsedJoint0  # joint 29, arm_sdk gate
GATE_RAMP_STEPS = 101
GATE_RAMP_DT = 0.02

# waist 12-14 locked at the captured pose, NOT driven, so the arm_sdk
# gate (29) is honored -- same as the stock waist lock.  Legs 0-11 too:
# limp legs (mode=0) leave the gate unhonored and the arms limp
# (effective kp ~2 Nm/rad vs the commanded 80).
WAIST_JOINTS = (G1_29_JointIndex.kWaistYaw,
                G1_29_JointIndex.kWaistRoll,
                G1_29_JointIndex.kWaistPitch)

# legs 0-11 locked at the captured pose (mode=1, kp_high/kd_high), like
# the stock locking ALL 35.  Limp legs (mode=0) leave the gate unhonored
# and the arms at ~2 Nm/rad vs 80.  The pick is stationary (non-801).
LEG_JOINTS = (G1_29_JointIndex.kLeftHipPitch,
              G1_29_JointIndex.kLeftHipRoll,
              G1_29_JointIndex.kLeftHipYaw,
              G1_29_JointIndex.kLeftKnee,
              G1_29_JointIndex.kLeftAnklePitch,
              G1_29_JointIndex.kLeftAnkleRoll,
              G1_29_JointIndex.kRightHipPitch,
              G1_29_JointIndex.kRightHipRoll,
              G1_29_JointIndex.kRightHipYaw,
              G1_29_JointIndex.kRightKnee,
              G1_29_JointIndex.kRightAnklePitch,
              G1_29_JointIndex.kRightAnkleRoll)

# slots 30-34 locked too, like the stock locking ALL 35.  At mode=0 (the
# DDS default) they may trip the G1 firmware in Running Mode (801): arm
# oscillation / fsm_mode 0->3.  Held at the captured pose, never driven.
NOT_USED_JOINTS = (G1_29_JointIndex.kNotUsedJoint1,
                   G1_29_JointIndex.kNotUsedJoint2,
                   G1_29_JointIndex.kNotUsedJoint3,
                   G1_29_JointIndex.kNotUsedJoint4,
                   G1_29_JointIndex.kNotUsedJoint5)


class G1_29_ArmControllerUpperOnly(G1_29_ArmController):
    """Arms-driven variant: legs 0-11 + waist 12-14 + trailing 30-34 held.

    The arms (15-28) and the arm_sdk gate joint (29) are driven; the
    waist (12-14), legs (0-11), and trailing unused slots (30-34) are
    locked (mode=1) at the captured pose and held, never driven.  Locking
    mirrors the stock G1_29_ArmController (all 35 joints mode=1) so the
    arm_sdk gate is honored and the arms don't run limp.  Pick runs
    stationary (non-801).
    """

