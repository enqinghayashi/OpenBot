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
