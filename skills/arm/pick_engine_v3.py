#!/usr/bin/env python3
"""
Pick daemon — persistent subprocess for autonomous pick + motion operations.
Run in clean shell with tv conda env. NO ROS2 env.

Protocol (stdin/stdout):
    IN:  "pick <x> <y> <z> [nx ny nz] [orientation] [height_m]"
                               — move right arm to torso-frame coords and grasp
                               — orientation: "upright" (default) or "horizontal"
                               — height_m: optional T2 object-height (m);
                                 6-arg = ... <orientation> <height_m>,
                                 9-arg = ... <nx> <ny> <nz> <orientation> <height_m>;
                                 invalid float → ERROR:bad height_m;
                                 absent/implausible (<=0 or >0.5) → legacy lift
    IN:  "pick_left ..."      — same forms for the left arm
    IN:  "open"                — open right hand
    IN:  "close"               — close right hand
    IN:  "open_left"           — open left hand
    IN:  "close_left"          — close left hand
    IN:  "open_both"           — open both hands
    IN:  "close_both"          — close both hands
    IN:  "home"                — return arms to home position
    IN:  "ready"               — go to tucked ready pose
    IN:  "carry"               — right arm to T1 carry posture (upright frame)
    IN:  "carry_left"          — left arm to T1 carry posture (upright frame)
    IN:  "play <name>"         — play motion sequence from ~/motion_sequences/<name>.json
    IN:  "list"                — list available motion sequences
    OUT: "READY"               — daemon initialised
    OUT: "DONE"                — command completed
    OUT: "ERROR:<msg>"         — something went wrong
"""
# ─────────────────────────────────────────────────────────────────────────────
# Behaviour contracts kept by this daemon:
# - horizontal wp1/wp2 use a raw swing; wrist-GN polish tilts the palm
#   (0.18-0.22 rad orientation error).
# - horizontal descend stops on the middle pad (pkg 5) only, with one
#   resume/depth-push slot. a reach stall (gap > H_DESCEND_STALL_GAP) is not
#   palm contact; pushes are bounded (H_DESCEND_MAX_PUSHES = 3) and gated on
#   progress.
# - the close-joint block (CLOSURE_BLOCKED_TOL 0.06 rad) is the grip evidence,
#   because pressure is flat. any promote path needs a block or a real
#   pressure delta.
# - vertical grasp z is tz + GRASP_Z_UP_BIAS (0.01 m).
# - horizontal wp2/wp3/retract park at fwd_offset=-HAND_GRIP_CENTER_OFFSET so
#   the grip zone (+17..+69 mm ahead of R_ee) covers the centroid.
# - the reach gate refuses above REACH_REFUSE_POS (0.15 m) and keeps a
#   diagnostic plus best-effort in (0.03, 0.15].
# - lateral sign: torso +x is robot right, so ik_y = -torso_x.
# - solver-fresh clears the filter queue before each waypoint phase.
# - writer mode is a non-owning socket client with no arm_sdk dds writer.
#   PICK_ARM_WRITER=on|off|auto, where auto means writer if the socket exists.
# ─────────────────────────────────────────────────────────────────────────────



import os
import sys
import time
import math
import json
import glob
import numpy as np

os.chdir('/home/unitree/xr_teleoperate/teleop/robot_control')

sys.path.insert(0, '/home/unitree/xr_teleoperate')
sys.path.insert(0, '/home/unitree/unitree_sdk2_python')
sys.path.insert(0, '/home/unitree/leon')

import pinocchio as pin
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.robot_control.robot_arm import G1_29_ArmController
from unitree_sdk2py.core.channel import (
    ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
from real_grasp_detector import RealGraspDetector, GraspState

# ── bailing out mid-task ─────────────────────────────────────────────────────
# the cancel flag lives in a separate module and is set by the entrypoint's socket
# thread without importing this module (os.chdir above makes import fragile).
# imported defensively: the standalone lane (PICK_ARM_WRITER=off) may have no
# module, and that must not take pick down, so choke calls become no-ops.
# Cancelled derives from BaseException, so plain `except Exception` handlers
# never see it; it reaches the unwind at the foot of the command loop. this
# file must keep zero bare `except:` clauses.
try:
    import engine_cancel
except ImportError:  # pragma: no cover — standalone lane with no module
    class _Cancelled(BaseException):
        """Never raised; exists so `except engine_cancel.Cancelled` below
        stays a valid clause when the module is absent."""

        def __init__(self, where=''):
            self.where = where
            super().__init__(where)

    class _NoCancel(object):
        """No-op stand-in: check() never raises, so nothing is cancellable
        and the engine behaves exactly as it did before the feature."""
        Cancelled = _Cancelled

        @staticmethod
        def check(where):
            return None

        @staticmethod
        def clear():
            return None

        @staticmethod
        def pending():
            return False

    engine_cancel = _NoCancel()


def _check_cancel(where):
    """Motion choke-point probe: raise Cancelled if a cancel is pending.

    Called from the arm-move and hand-move loops.  ``where`` names the PHASE
    (not the function) because that is what makes the reply actionable — an
    operator who cancels needs to know whether the arm stopped mid-descent
    (hovering over the object) or mid-lift (holding it).  Those leave
    different recovery options.
    """
    engine_cancel.check(where)

# the grip server owns the rt/dex3/*/cmd writers; the daemon only routes hand
# commands via GripClient and imports the constants it prints or force-closes
# with. there must be zero ChannelPublisher or Dex3Hand here.
sys.path.insert(0, '/home/unitree/leon/openbot')  # where the grip server is deployed
from grip_server import (  # noqa: E402
    DEX3_JOINTS, CLOSE_PRESETS, R_HAND_OPEN, R_HAND_CLOSE, L_HAND_OPEN, L_HAND_CLOSE,
    OPEN_PRESETS,   # closing sign source
    R_HAND_CYLINDRICAL, R_HAND_PINCH, R_HAND_WIDE,
    F_MAX_SAFE, clamped_close_target,
)
from grip_client import GripClient, GripError  # noqa: E402

# merged-writer socket client; the engine-lane client is already on sys.path
# and the standalone lane never touches it.
from writer_arm_client import WriterArmClient, WriterError  # noqa: E402

# ── left arm neutral (sits here while the right arm picks) ───────────────────
L_NEUTRAL = pin.SE3(
    pin.Quaternion(1, 0, 0, 0),
    np.array([0.25, 0.25, 0.1]),
).homogeneous

# ── right arm neutral (mirror of L_NEUTRAL: ik_y negative on the right) ──────
R_NEUTRAL = pin.SE3(
    pin.Quaternion(1, 0, 0, 0),
    np.array([0.25, -0.25, 0.1]),
).homogeneous

# ── the neutral seed for IK ──────────────────────────────────────────────────
NEUTRAL_SEED = np.array([
    0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   # left arm
    0.0, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   # right arm
])

# ── pose state machine ───────────────────────────────────────────────────────
# joint order: pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
# q = [left 0..6] + [right 7..13]
# pick: ready -> carry_tuck (balance) -> approach/grasp -> LIFT -> CARRY_TUCK
# LIFT and CARRY_TUCK are fixed joint poses. height_m is still parsed (the
# HEIGHT_M_MAX gate) but does not drive the lift.
# READY elbow is -0.80 (URDF limit [-1.0472, +2.0944]); a -1.2 spec exceeded
# that limit by 0.153 rad. left yaw is positive and right yaw negative:
# same-sign yaw pairs twist the left arm into the chest.
READY_LEFT  = np.array([0.40,  0.30,  0.50, -0.80,  0.00,  0.00, -0.01])
READY_RIGHT = np.array([0.40, -0.30, -0.50, -0.80,  0.00,  0.00,  0.01])
LIFT_LEFT   = np.array([0.00,  0.20,  0.30, -0.80,  0.00, -0.60,  0.20])
LIFT_RIGHT  = np.array([0.00, -0.20, -0.30, -0.80,  0.00, -0.60, -0.20])
CARRY_TUCK_LEFT  = np.array([0.50,  0.05,  0.00,  0.00, -1.50,  0.00, -1.00])
CARRY_TUCK_RIGHT = np.array([0.50, -0.05,  0.00,  0.00,  1.50,  0.00,  1.00])
# low hanging-carry pose. the mirror convention is +roll left and -roll right.
# it is the quit down-move target so the grip clears the legs before
# release_arms, which otherwise swings the arms into them.
LOW_CARRY_LEFT  = np.array([0.20,  0.15,  0.00, 1.20, 0.00,  0.1,  0.00])
LOW_CARRY_RIGHT = np.array([0.20, -0.15,  0.00, 1.20, 0.00,  0.1,  0.00])
# NEUTRAL: lowstate-sourced hanging rest.
# sign risk: lowstate yaw and elbow may be flipped against the arm_sdk command
# convention, so verify on hardware before trusting it and flip only a
# misbehaving joint, one at a time (yaw first, roll only if still wrong).

NEUTRAL_LEFT  = np.array([0.25522,  0.27104, -0.08654, 0.83254, -0.00101,  0.01032, -0.00063])
NEUTRAL_RIGHT = np.array([0.26688, -0.28410,  0.09247, 0.80183, -0.00034,  0.00915,  0.00730])

# HOME: both arms hang at the torso sides, palms in, thumbs forward. mirrors
# the arm guard's HOME pose. simulated only, unlike READY/LIFT/CARRY_TUCK and
# NEUTRAL: in the sim plant (root z=0.793, MuJoCo contact at margin 0.12) the
# minimum arm-to-leg clearance is +0.0851 m (right_hip_pitch to
# right_wrist_pitch) and arm-to-torso +0.1449 m (pelvis to left_elbow). the
# collision hulls are fatter than the real shells, so 85 mm is a conservative
# lower bound. verify on hardware before trusting it.
# joint by joint:
#   shoulder_pitch 0.00 has the upper arm plumb; positive swings backward
#   shoulder_roll ±0.30 because the shoulder is only ~16 mm outboard of the
#       hip (shoulder_pitch_link y=+0.100 against hip_roll y=+0.1165); a plumb
#       arm rams the hip and at roll=0 the sim reports hand-to-hip penetrations
#   elbow 1.50 has the elbow 90 deg flexed (forearm horizontal at 0);
#       +pi/2 points the forearm straight down
#   wrist_roll ∓1.5708: at 0 the thumb points inboard at the thigh; ∓pi/2
#       turns the palm inboard with the thumb forward (+X), worth +38 mm of
#       hip clearance. the sign matches CARRY_TUCK and NEUTRAL (left negative,
#       right positive).
HOME_LEFT  = np.array([0.00,  0.30,  0.00, 1.50, -1.5708,  0.00,  0.00])
HOME_RIGHT = np.array([0.00, -0.30,  0.00, 1.50,  1.5708,  0.00,  0.00])
HOME_POSE  = np.concatenate([HOME_LEFT, HOME_RIGHT]).astype(np.float64)

# READY_POSE: both arms ready; ready/ready_right/ready_left are identical
READY_POSE = np.concatenate([READY_LEFT, READY_RIGHT]).astype(np.float64)

# ── knobs for the pick ────────────────────────────────────────────────────────
# a bigger approach height makes the wrist arrive above the object and clear
# the top rim before descent; a bigger forward offset parks farther behind
# contact depth, which the guarded wp4 insert then walks. horizontal keeps a
# forward offset of -HAND_GRIP_CENTER_OFFSET so the hand does not go past the
# object.
APPROACH_HEIGHT_OFFSET = 0.15
APPROACH_FORWARD_OFFSET = -0.13
MOVE_SPEED = 0.3  # rad/s; 0.6 is visibly faster
MOVE_HZ    = 50

# ── how the arms get moved ────────────────────────────────────────────────────
# writer mode is a non-owning socket client of the merged writer (engine
# lane); standalone uses the arm_sdk streaming path.

WRITER_SOCK_DEFAULT = '/tmp/merged_writer.sock'
ARM_WRITER_ENABLED = False  # main() sets this via _detect_writer_mode()


def _detect_writer_mode():
    """PICK_ARM_WRITER: on|1|true|yes → writer mode (fail-fast if the writer
    is unreachable); off|0|false|no → standalone arm_sdk; auto (default) →
    writer mode iff the writer's socket exists (the merged writer is running
    and owns rt/lowcmd + the arms — a second arm_sdk writer in this process
    would fight it and freeze the arms)."""
    _env = os.environ.get('PICK_ARM_WRITER', 'auto').strip().lower()
    if _env in ('on', '1', 'true', 'yes'):
        return True
    if _env in ('off', '0', 'false', 'no'):
        return False
    return os.path.exists(WRITER_SOCK_DEFAULT)


