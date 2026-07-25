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


def _hold_rc_writer(arm, enabled, context=''):
    """Writer-mode RC interlock (engine hold_rc timeline: ON at pick start,
    OFF after CARRY_TUCK / failure / quit / exception).  Standalone mode:
    NO-OP — the arm_sdk controller has no such verb."""
    if not ARM_WRITER_ENABLED:
        return
    try:
        arm.hold_rc(enabled)
    except Exception as _e:
        sys.stderr.write(
            f'[pick_daemon] hold_rc {enabled} ({context}) FAILED: {_e} — '
            'writer keeps the clamp; writer console `hold_rc off`\n')
        sys.stderr.flush()


def _close_hand_at_closeout(hand, side, verb):
    """Task #9 (2026-09-13): park a hand CLOSED at a verb's closeout.

    Operator report after the first routed place: "after place task is done,
    the hand is not closed at all, it remained open which i do not want that,
    i want the hand to be fully closed" — plus, "for the hand closure at
    neutral pose as well".

    **CALL THIS FROM `neutral` ONLY.  NEVER FROM A PLACE EXIT.**  That is not
    a style preference; a close on a place path is a RELEASE-OBSERVABLE
    violation and it breaks EVERY place, on all four exits:

      * the engine writes DONE after this returns, and the gateway then takes
        a FRESH grip read (gateway/gateway_node.py:2171 _refresh_grip_state());
      * verify_place requires `grip.is_open is True` (brain/verifier.py:1098),
        per the place post_condition "grip open AND object out of hand"
        (brain/tool_contract.py `"place"`);
      * GripClient.close() cosine-ramps 40 x 0.02 s = 0.8 s of BLOCKING sleep
        (grip_server.py `set`), so the DONE cannot even be written first.

    Net effect of closing here from a place: the gateway measures a closed
    hand and `place` returns in_progress ("post-drop hand not measured-open")
    on the whole success path, deterministically.  This was tried and
    reverted on 2026-09-13.  `neutral` carries no such contract (it has no
    entry in brain/tool_contract.py), which is why the resting-pose half of
    the operator's request lives there instead.
    tests/test_place_closeout_closed_hand.py pins all of this at the source
    level — it fails if a fifth place path re-adds the call.

    BEST-EFFORT BY DESIGN: whatever verb calls this, a hand-controller fault
    must never turn a completed (unundoable) action into an ERROR.  Log and
    continue, same policy as the open-verify and payload-off steps nearby.

    `hand` is the GripClient for `side` ('right' -> rhand, 'left' -> lhand);
    GripClient.close() cosine-ramps to that side's CLOSE preset.

    2026-09-27: the close is now THUMB-FIRST and STAGED rather than one ramp --
    see `_staged_thumb_first_close` below for the operator report and the
    mechanism.  Endpoints are unchanged; only the order is.  BOTH stages live
    in that helper, so this body no longer calls `hand.close()` itself.
    """
    try:
        stage = _staged_thumb_first_close(hand, side, verb)
        sys.stderr.write(
            f'[pick_daemon] {verb}: {side} hand CLOSED at closeout '
            f'(CLOSE preset; thumb-first {stage})\n')
        sys.stderr.flush()
    except Exception as _ce:
        sys.stderr.write(
            f'[pick_daemon] {verb}: {side} hand close at closeout FAILED '
            f'({_ce}) — the hand stays open; continuing\n')
        sys.stderr.flush()


# place hand-park: the holding hand parks closed right after LIFT. closing
# before the hand descends to NEUTRAL keeps open fingers from sweeping down
# past the drop height, where they can hit the table.
# cost: the place verifier wants the post-drop hand measured open, and a hand
# parked closed reads as not open, so a place can stay in progress. the
# release-open token carries the release measurement instead.
# latch risk: this is a full close preset with no pressure stop at 20 Hz. a
# close stalling on a retained object can latch thumb_1, index_0 and middle_0,
# which need a power cycle, so release_ok must cover an object on the palm at
# open. no channel answers "hand empty?" directly.
PLACE_PARK_HAND_CLOSED = True


def _park_hand_after_lift(hand, side, verb, release_ok):
    """Park the HOLDING hand CLOSED right after the place closeout's LIFT move.

    `release_ok` is the place's own `_verify_hand_open` result, and it is the
    ONLY guard this stack can offer against the one way this park is dangerous:
    a close that GRASPS.  The right Dex3 hand latches three of its seven motors
    (thumb_1 / index_0 / middle_0 -> mode=0, motorstate=512, ~0 N.m) on a close
    that stalls against a retained object — measured three times on 2026-09-19,
    cleared only by a power cycle.  This park drives the FULL CLOSE preset:
    `hand.close()` sends the raw `close` verb, which grip_server resolves to
    CLOSE_PRESETS with no pressure stop, and `_writer_loop` re-asserts it at
    20 Hz for as long as the hand is held there (grip_server.py:444-451,
    WRITE_HZ = 20.0).  F_MAX_SAFE does NOT apply on this path — it lives only
    in `clamped_close_target`, which only `close_adaptive` calls — so restoring
    it would not have capped this.  An object still in the fingers at the LIFT
    pose therefore gets a sustained full-preset stall: the latch trigger.

    `release_ok` is a JOINT-POSITION test (every finger within
    VERIFY_OPEN_TOLERANCE_RAD of the open preset).  It catches a finger that
    never reached open — a wedged object, an unreleased object, a latched
    finger — and it does NOT catch an object resting on the palm with the
    fingers at open.  That gap is real and is documented rather than papered
    over: no channel on this stack can answer "is the hand empty?" — the
    pressure path reads 0 and the grasp verdict's own `tactile` flag is False.
    A position test is all there is, so this refuses on doubt rather than
    closing blind.

    Best-effort by design: a hand-controller fault must never turn a completed
    (unundoable) action into an ERROR — log and continue, the same policy as
    `_close_hand_at_closeout` and the payload-off step nearby.

    `hand` is the GripClient for `side` ('right' -> rhand, 'left' -> lhand).
    GripClient.close() cosine-ramps 40 x 0.02 s = 0.8 s of BLOCKING sleep, so
    this delays the closeout's NEUTRAL move by that much when ON.

    2026-09-27: the park is now THUMB-FIRST and STAGED rather than one ramp --
    see `_staged_thumb_first_close` below for the operator report and the
    mechanism.  Endpoints are unchanged; only the order is.  BOTH stages live
    in that helper, so this body no longer calls `hand.close()` itself.  This
    adds a second 0.8 s stage-1 ramp, so the NEUTRAL move is delayed ~1.6 s
    when it fires.
    """
    if not PLACE_PARK_HAND_CLOSED:
        return
    if not release_ok:
        sys.stderr.write(
            f'[pick_daemon] {verb}: park SKIPPED — the {side} hand was not '
            f'verified open after the release, so something may still be in '
            f'the fingers; refusing to close on it (latch risk)\n')
        sys.stderr.flush()
        return
    try:
        stage = _staged_thumb_first_close(hand, side, verb)
        sys.stderr.write(
            f'[pick_daemon] {verb}: {side} hand PARKED CLOSED at the LIFT pose '
            f'(CLOSE preset; thumb-first {stage})\n')
        sys.stderr.flush()
    except Exception as _ke:
        sys.stderr.write(
            f'[pick_daemon] {verb}: {side} hand park at LIFT FAILED ({_ke}) '
            f'— the hand stays open; continuing\n')
        sys.stderr.flush()


GRASP_Z_OFFSET = 0.04   # grasp 4cm above the YOLO detection centroid
# upright grasp sits 1 cm higher. applies only to the upright grasp z, not
# horizontal tz; knob range 0.005..0.02.
GRASP_Z_UP_BIAS = 0.01
# z-frame diagnostic only: perception z is torso_link, IK base is pelvis.
# the waist_roll origin is (-0.0039635, 0, +0.044) in the URDF, so torso_link
# is 0.044 m above pelvis and ik_z = torso_z lands ~4.4 cm low. this roughly
# cancels GRASP_Z_OFFSET + GRASP_Z_UP_BIAS (0.04+0.01-0.044 ~= +0.006 m net),
# so it is logged, not commanded.
SOLVER_Z_TORSO_OFFSET = 0.044
DESCENT_DURATION = 3.0     # descent time (s); higher = slower
# h-wp3 can finish its ramp with the wrist still above tz (end-of-ramp lag)
# while the pad has not touched; more than this above tz triggers one guarded
# push-down.
H_DESCEND_DEPTH_TOL = 0.03

# ── object-height chain (protocol compatibility) ─────────────────────────────
# height_m is parsed and plausibility-gated (<=0 or > HEIGHT_M_MAX means
# unknown); the lift does not use it.
HEIGHT_M_MAX = 0.5

# ── note for the hardware check ──────────────────────────────────────────────
# READY/LIFT yaw pairs are same-sign (-0.5/-0.5, -0.3/-0.3), breaking the
# mirrored-sign convention; check them visually on hardware.

# ── palm-contact detection during the wp4 insert ────────────────────────────
# joints watched (14-DOF q): left 0=shoulder_pitch, 3=elbow; right
# 7=shoulder_pitch, 10=elbow
PALM_CONTACT_MONITOR_JOINTS_RIGHT = [7, 10]
PALM_CONTACT_MONITOR_JOINTS_LEFT  = [0, 3]
# [7,10]/[0,3] are blind to a vertical wall-grasp wp4 push: sideways motion
# comes from shoulder roll 8 and yaw 9 plus wrist flex (12/13 right, 5/6
# left). these monitors are for the vertical wp4 only. if a false trip
# occurs, drop to [8,9]/[0,1] or raise the consecutive count to 3.
WALL_PUSH_MONITOR_JOINTS_RIGHT = [7, 8, 9, 10]
WALL_PUSH_MONITOR_JOINTS_LEFT  = [0, 1, 2, 3]
# 0.04 rad is normal tracking lag and false-triggers; 0.15 is a real contact.
PALM_CONTACT_POSITION_THRESHOLD = 0.15   # sum of abs pos error (rad) across the joints
PALM_CONTACT_CONSECUTIVE_STEPS  = 2       # overruns in a row before contact
INSERT_GUARD_HOLD_TIME = 0.3              # hold after the contact stop
RETRACT_AFTER_CONTACT = -0.03             # torso Y retract after contact; unloads it
# extra standoff (m) for the vertical wp4 insert and its retract, on top of
# approach_offset; 10-30 mm, default 20. keep it below the 0.15 rad contact
# threshold.
GRASP_INSERT_MARGIN = 0.02

# ── relative insert-guard pressure ───────────────────────────────────────────
# an absolute threshold of 60000 sits below the resting open-hand floor
# (~105600), so it fires before any descent and freezes every guarded insert,
# which then closes on air and reports a false solid grasp.
# the trigger is a delta over the baseline captured at guard build (hand open,
# in air). floor noise is ±30; 30000 is three orders above it and below
# GRIP_PRESSURE_DELTA_EVIDENCE; a real first touch reads tens of thousands.
DELTA_PRESS = 30000.0

# ── gating verdicts on evidence ──────────────────────────────────────────────
# close_until_contact using 200000 as an absolute ceiling makes a light object
# squeezed all the way read lower, so it can return 'full' (no grip). a veto
# then demotes solid_grasp to grasped_nothing and the retry reopens.
# that veto buys nothing, since an empty-hand false solid fires via 'contact',
# not 'full', and it kills real grips. the close's pressure delta (final minus
# baseline) proves the fingers pressed something, so use it to upgrade a
# missed light grip.
GRIP_PRESSURE_DELTA_EVIDENCE = 40000.0   # a late-close rise => object in hand
# the light-band upper bound stays 40000 because the band promote is ungated.
# an empty self-pinch ceiling delta is ~94400 (200000 stop minus ~105600 rest
# floor), so a widened band would promote it without the support gate. a
# 'full' delta in (40000, 100000] goes to the support_ratio >= 0.5 bearing
# gate.
GRIP_CLOSE_STOP_DELTA = 200000.0   # close_until_contact stop; server 200000
# the close stop must stay well above the resting finger-pressure floor. the
# server compares absolutely (no baseline) while finger_pressure rests around
# 104-110k, so a 100000 stop fires before the fingers move and makes a
# free-air close read as contact, the false-success class. 200000 sits between
# the rest floor and a hard pad press (223272), so it is inert in
# normal use. the comparison is absolute, not a delta.
# keep evidence is not this channel: it comes from _score_closure_blocked
# (actual-vs-commanded margin, CLOSURE_BLOCKED_TOL=0.06 rad), which is
# pressure-independent. finger pressure is flat here (bottle in hand: +2584
# over 50 steps; empty close +1760; span ~120000).
# light-grip delta floor: a real wall or table jam and an empty self-pinch
# both press the pads hard. at the rest floor ~105600 the 200000 ceiling caps
# the squeeze delta at ~94400, below the 100000 stop, so both read
# closure='full'. a delta between this floor and the evidence threshold marks
# a real rim or edge grip that the context-free detector calls hit_obstacle;
# partial rim touches and ±30 noise stay below it.
_CLOSE_DELTA_LIGHT_GRIP_FLOOR = 9000.0
# support_ratio bearing gate. the detector knows only object_mass, so the
# independent signal is support_ratio = sum |tau_est| over DISTAL_JOINT_IDS
# divided by object weight. a real grip on the 0.30 kg object bears it and the
# ratio climbs toward 1.0; a self-pinch stays low. promote hit_obstacle to
# light_object only then. 0.5 is conservative.
# limitation: a hard wall or table jam also loads distal joints, so this is
# not a perfect jam discriminator, just more physical than pressure.
GRIP_SUPPORT_RATIO_KEEP = 0.5
# reach-refuse split. an in-reach residual is 0.068-0.092 m against
# 0.17-0.32 m truly out. a hard 0.03 gate false-refused reachable grasps, so
# 0.15 sits between the groups.
def _env_reach_refuse_pos():
    """REACH_REFUSE_POS default 0.15; OPENBOT_REACH_REFUSE_POS overrides when
    0.03 < v <= 0.50 (v5.4).  Missing / garbage / out-of-range -> 0.15."""
    try:
        _v = float(os.environ.get('OPENBOT_REACH_REFUSE_POS', ''))
    except (TypeError, ValueError):
        return 0.15
    if _v == _v and 0.03 < _v <= 0.50:
        return _v
    return 0.15


REACH_REFUSE_POS = _env_reach_refuse_pos()  # solved insert residual above this = out of reach
# closure-block evidence: pressure is flat on both hands (delta 0-16 on an
# ambient ~103.8k), so it cannot verify a grip. close joints can: an air close
# reaches the preset exactly, while a real wrap stops short. any close joint
# farther than this from the preset pressed something.
CLOSURE_BLOCKED_TOL = 0.06        # rad, checked on any of DEX3_JOINTS
# ── bounded-lead close, the latched-motor guard ─────────────────────────────
# tau_est ~= bias + 16.6 * lead with lead = |q_cmd - q_meas| closing, and bias
# = parked-hold tau at lead ~0 (0.25-1.09 per joint). the Dex3 rails at
# tau_est ~= 15, so lead_rail ~= 0.83 rad. the rail is a clip, not a torque:
# three motors reported 14.99599, so read 15 as ">= 15, clipped". rail onset
# to first latch trip is 1.16 s.
# a full-preset close on a seated object can rail several joints before the
# pressure stop sees contact; the ramp then closes further into the jam and
# thumb_1, index_0 and middle_0 can trip to motorstate 512 (limp, ~0 N.m).
# that is a motor fault, not an in-hand signal, and the loosened grip lets the
# object slip.
# a bounded press works: tau plateaus around 1.5-5.6 N.m, far below the rail,
# and the grip holds. leads arrive uneven (0.067-0.276 rad) for one uniform
# command because joints block at different travel fractions, so lead is set
# per joint against a live pose, not as a preset offset.
# no stop threshold fixes a full-preset ramp: it is back-loaded (65% by step
# 30, 90% by step 40), so at first contact the commanded lead is already ~0.76
# rad, ~92% of the rail. lead must never be commanded large.
# HAND_CLOSE_LEAD_EPS is the per-write closing-lead bound and therefore the
# force command: the design window is 0.06 < lead < 0.83 rad. 0.06 is
# CLOSURE_BLOCKED_TOL, below which a blocked hand reads blocked=False and the
# pre-lift pin aborts; 0.83 is the rail. the sweet spot is 0.15-0.35, and 0.25
# is mid and within 10% of the proven peak lead 0.276. worst case tau is
# 1.09+16.6*0.25 = 5.24 N.m, 35% of rail. do not push toward 0.83 and do not
# halve toward 0.06: below CLOSURE_BLOCKED_TOL every pick false-fails.
HAND_CLOSE_LEAD_EPS = 0.25        # rad per joint, in the closing direction
# one write per press, and that bit is load-bearing. the grip server's set()
# ramps from its own last_q, not the target, so a multi-step press writes all
# intermediate points; if last_q is stale (boot seed, another session, rail
# recovery) those points carry |last_q - q_meas|, which no target bound can
# reach. at steps=1 r = 0.5-0.5*cos(pi) = 1.0, so the only write is the
# endpoint, lead = eps by construction and last_q is irrelevant.
HAND_CLOSE_PRESS_WRITES = 1
HAND_CLOSE_PRESS_DT = 0.02        # server pacing per write
HAND_CLOSE_WATCH_BUDGET_S = 0.35  # per-press watch window (worst case)
HAND_CLOSE_POLL_S = 0.02          # q_meas poll period in that window
HAND_CLOSE_ARRIVE_FRAC = 0.90     # within (1-frac)*eps of target means arrived; end watch early
HAND_CLOSE_STILL_TOL = 0.004      # rad moved in one poll = still moving
HAND_CLOSE_STILL_HOLD_S = 0.10    # motionless this long = press done
HAND_CLOSE_LOST_READS = 3         # consecutive empty reads = state lost
HAND_CLOSE_TARGET_TOL = 2.0e-3    # two commands this close are a fixed point
HAND_CLOSE_CEILING_TOL = 1.0e-6   # command is at the close preset
HAND_CLOSE_MAX_PRESSES = 160      # press cap
# a free-air close needs ceil(1.745/0.25) = 7 presses. a free-air close still
# reads blocked=False down to 0.15 rad/s with 160 presses over 15 s, about 14x
# margin under the ~2.1 rad/s tracking. a blocked finger exits at the
# command fixed point after ~6 presses, so only a slow free finger pays this
# cap, and that case must finish. do not shrink toward 48: the false-capture
# boundary, where a free close reads blocked=True, is set by this number.
HAND_CLOSE_DEADLINE_S = 15.0      # wall-clock cap on one full close
# safe to be generous: lead is bounded at eps for every write, so a long walk
# is a long bounded walk.
HAND_CLOSE_STATE_RETRIES = 3      # state channel can flicker
HAND_CLOSE_STATE_RETRY_S = 0.05   # a transient empty read must not read as "blind"
# no-reopen, squeeze-hold and pre-lift constants.
# _CLOSE_DELTA_TOUCH_MIN: the close's pressure delta reads 0-24 even on
# contact (dead channel), so a delta at or above this proves the fingers
# pressed; the closure-block and contact-context signals cover the rest.
_CLOSE_DELTA_TOUCH_MIN = 250.0
# pre-lift pin: a closure margin that regresses by more than this between the
# close and the pre-lift re-read means the grip slipped; tuck re-pin evidence only.
PRE_LIFT_SLIP_REGRESS_TOL = 0.03
# a gentler carry_tuck move (slower wrist roll at the lift handoff).
CARRY_TUCK_DURATION = 4.0

# ── horizontal (middle-down) grasp tuning ───────────────────────────────────
# HAND_GRIP_CENTER_OFFSET: the wrist parks this far behind the centroid (in
# -torso Y) so the palm grip midpoint covers the object centre. the URDF grip
# zone runs +17..+69 mm forward of R_ee, midpoint ~43 mm, so a wrist at the
# centroid puts the grip zone over the front edge.
HAND_GRIP_CENTER_OFFSET = 0.045
# H_PALM_MOUNT_OFFSET: how far the pad hangs below the wrist in the
# middle-down pose, empirically in [0.005, 0.017], not 0.030.
H_PALM_MOUNT_OFFSET = 0.015
# honest-descend constants.
# the descend separates a reach stall (guarded pass stopped via position_error
# with the wrist still high above tz) from true palm contact. treating a stall
# as contact skips the rest of the descent and closes on air.
H_DESCEND_MAX_PUSHES = 3            # how many depth push re-solves one descend flow gets
H_DESCEND_MIN_PROGRESS = 0.01       # if wrist-z progress per push drops below this -> stop pushing
# gap = wrist_z - tz (get_wrist_pose, solver frame):
# gap > 0.045 (pad mount + depth tol) means reach stall
# gap <= 0.035 (pad mount + 2 cm swing) means pad contact
# in between: contact-style, no pad-band evidence
H_DESCEND_STALL_GAP = H_PALM_MOUNT_OFFSET + H_DESCEND_DEPTH_TOL
H_PALM_TOUCH_BAND = H_PALM_MOUNT_OFFSET + 0.02
# ~20 mm off the table
H_DESCEND_CLEARANCE = 0.02
# grasp-geometry x_off to object half-radius: compute_x_off = -(r + 0.065),
# so r = -x_off - 0.065. for x_off -0.0924, r is 0.0274.
H_X_OFF_RADIUS_INV = 0.065
H_RADIUS_MIN = 0.005
H_RADIUS_MAX = 0.08
# horizontal flows command wp2/wp3/retract at (tx, ty_use, tz), so
# H_PALM_MOUNT_OFFSET, H_DESCEND_CLEARANCE, H_RADIUS_MIN and H_RADIUS_MAX do
# not shape them. HAND_GRIP_CENTER_OFFSET still feeds WALL_GRASP_PAD_GAP, and
# H_X_OFF_RADIUS_INV feeds the wall-triad diagnostic log.

# ── upright (vertical-object) grasp template ────────────────────────────────
# UPRIGHT_GRASP: wrist template for the vertical pick. the upright-grasp
# builder puts the wrist on the object's forward axis, so the rotation equals
# identity and there is no wrist flexion; open fingers then lead the insert
# into the object's flank. the claw template instead gives real flexion: palm
# normal down, claw over the top and grip zone centred on grasp height. the
# orientation string stays 'upright' for protocol compatibility.
UPRIGHT_GRASP = 'upright'
# claw descend starts this far above the final wrist z so fingertips clear
# the top flank; palm-down fingers hang ~0.10 m below the wrist site.
UPRIGHT_CLAW_GUARD_START = 0.14

# ── wall grip ───────────────────────────────────────────────────────────────
# a palm-down claw freezes above the object because its up-down flexion is the
# wrong axis for a vertical object. the wall grip instead makes the clamp plane
# vertical along the long axis: the palm normal points horizontally at the near
# wall, fingers up, and a guarded lateral push comes from a side standoff using
# left/right wrist flexion instead of descending from above.
WALL_GRASP_R_DEFAULT = 0.05     # fallback half-radius, no x_off
WALL_GRASP_STANDOFF = 0.10      # wrist lateral distance off the wall
# at wp2/wp3; the guarded wp4 push leaves standoff minus pad_gap ~= 55 mm
WALL_GRASP_PAD_GAP = HAND_GRIP_CENTER_OFFSET   # wrist-to-wall at contact: 45 mm
# centre of the 17-69 mm URDF grip zone

# ── the wrist-space orientation solver + verified-solve wash ─────────────────
# DA14 index map, per-arm 7-vector:
#   [pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
#   [6] is wrist flexion (palm toward a vertical surface), [5] up/down and
#   [4] forearm yaw. dual q: left 0-6, right 7-13 (right wrist_yaw 13, left 6).
#   URDF limits: wrist_yaw and wrist_roll at 4,11 are ±1.9722; 5,6,12,13 are
#   ±1.6144.
WRIST_LIMITS = {4: 1.9722, 5: 1.6144, 6: 1.6144,
                11: 1.9722, 12: 1.6144, 13: 1.6144}
RIGHT_WRIST_Q_IDX = [11, 12, 13]   # forearm yaw / up-down / flexion
LEFT_WRIST_Q_IDX  = [4, 5, 6]
WRIST_GN_ITERATIONS = 40
WRIST_GN_TOL = 0.15               # rad; log3 residual norm, same as pose tol
SOLVE_VERIFY_POS_TOL = 0.03       # m; residual on EE position
SOLVE_VERIFY_ORI_TOL = 0.15       # rad; residual on EE rotation
SOLVE_WASH_ATTEMPTS = 5           # solve_ik attempts per waypoint

# ── ulnar deviation ─────────────────────────────────────────────────────────
# the upright wall-grasp approach rolls the hand about the palm normal, an
# in-palm-plane ulnar deviation. d=0 is straight; d>0 tips the finger line
# ~30 deg away from the thumb, pinky-ward, so the long index does not reach the
# surface before the palm bed.
# per-object target via DEV_DELTA_TGT. deviation happens slowly during the
# approach.
# R_dev = Rot(n_palm, s_dev*d) . R_base, with n_palm=[0,1,0] (right palm +Y,
# left -Y) and s_dev = DEV_ROT_SIGN*sign(dir_lateral). R_base is identity, and
# [0,1,0] is identity's own palm-normal column, so the contact-face projection
# is unchanged and only the finger line rolls.
# the 3-DOF wrist GN (solve_wrist_orientation_in_place) realizes it. in the
# wall pose palm roll maps to wrist_roll [11]/[4] with a cross-axis
# leak, so only a body-frame policy is valid.
# sign: a right-hand deviation tips away from the thumb, rotating about
# +[0,1,0] by -d for right (+1 dir_lateral); +d mirrors the left (-1).
DEV_MAX = 0.5236            # 30 deg hard bound (photo angle)
DEV_STEP_MAX = np.pi / 12.0 # 15 deg/waypoint; 2*pi/12 = 0.5236 = DEV_MAX
# two ramps from wp3 to wp4 land at 0.520 rad, never 30 deg
DEV_ROT_SIGN = -1.0         # x sign(dir_lateral): right -1, left +1
DEV_DELTA_TGT = DEV_MAX     # flat 30 deg target
# wire the object law in here
PITCH_FROZEN_TOL = 0.15     # rad; no-pitch rule at contact, index 5 must not move
# pitch about the fingers axis is up/down [5]/[12]. diagnostic only: the
# corrective gate is removed and the decomposition is logged.
# the upright flows use identity R at wp2/wp3/wp4/retract
# (make_wrist_target_upright_v4, no claw chase) with the insert guard and
# retract unchanged. deviation is an add-on: True enables it, False gives the
# straight control arm, and commanded positions are otherwise unchanged.
DEVIATION_ENABLED = True

# ── where the motion sequences live ───────────────────────────────────────────
SEQUENCES_DIR = os.path.expanduser('~/motion_sequences')


# ── muting Ipopt's stdout (CasADi solve_ik) ──────────────────────────────────
# Ipopt dumps license text on the first solve and stdout is the protocol
# (DONE/ERROR), so redirect it to /dev/null during every solve_ik.
class _SuppressStdout:
    """Redirect stdout+stderr to /dev/null.  Suppresses Ipopt/CasADi output.

    READ THIS BEFORE "SIMPLIFYING" THE dup2 BACK TO THE OBVIOUS FORM
    ---------------------------------------------------------------
    dup2 ALONE IS NOT ENOUGH, and it was not enough for two months of runs.
    Ipopt's licence banner is written **through the Python ``sys.stdout``
    OBJECT**, not to file descriptor 1 (MEASURED 2026-10-01; casadi 3.7.2 /
    Ipopt 3.14.11: a recorder installed as ``sys.stdout`` received the whole
    banner while fd 1 -- dup2'd to a pipe -- received ZERO bytes).  So
    redirecting fd 1 cannot touch it.

    That mattered because of how the daemon is served.  The entrypoint
    replaces ``sys.stdout`` with ``_ReplyRouter`` and, while a command is in
    flight, routes every Python-level write on it to the CLIENT that asked
    (``tabletop_pick_place_engine.py:179-187, 258``).  The engine's own
    command loop is INLINE -- the pick handler writes its ``DONE`` only after
    ``CARRY_TUCK`` -- so the banner landed in the CLIENT'S REPLY STREAM, ahead
    of the real reply, and then drained one line per command.

    run10 is the cost, and it is exactly the operator's report: the banner
    block emitted at pick#1's first solve was consumed by SIX successive
    commands (``gateway.log:26,36,45,85,96,103``).  ``_do_pick`` treats any
    non-``ERROR`` reply as an executed pick, so three engine REFUSALS
    (OUT OF REACH, residuals 0.397/0.348/0.352 m) were reported to the brain
    as ``DONE pick executed`` -- and so were run10's TWO real successes
    (y=0.249 and y=0.435; each followed by LIFT + CARRY_TUCK in ``pick.log``):
    neither got its own ``DONE`` read, and the verdict was stamped ~5 s after
    dispatch against a reply that was a banner line.  "The pickup succeeded
    but was never confirmed" is this bug.

    So the suppression has TWO halves and both are load-bearing: dup2 the
    fds (for anything that writes to fd 1 directly) AND rebind ``sys.stdout``
    / ``sys.stderr`` to a stream whose fd is already the devnull one (for
    anything that writes through the Python object, which is what CasADi
    does).  The rebind is restored in ``__exit__``, so the router comes back
    intact and the client still gets its replies.

    Re-entrant by construction: every call site is ``with _SuppressStdout():``
    -- a FRESH instance -- so nesting unwinds LIFO with each instance holding
    its own saved object.
    """
    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved_stdout = os.dup(1)
        self._saved_stderr = os.dup(2)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        os.close(self._devnull)
        # second half: a real file object (not a discard stub) so .fileno() and
        # .isatty() answer; a fresh devnull fd, separate from fd 1.
        
        self._saved_sys_stdout = sys.stdout
        self._saved_sys_stderr = sys.stderr
        self._sink = open(os.devnull, 'w')
        sys.stdout = self._sink
        sys.stderr = self._sink

    def __exit__(self, *args):
        sys.stdout = self._saved_sys_stdout
        sys.stderr = self._saved_sys_stderr
        try:
            self._sink.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved_stdout, 1)
        os.dup2(self._saved_stderr, 2)
        os.close(self._saved_stdout)
        os.close(self._saved_stderr)


def make_wrist_target(torso_x, torso_y, torso_z, fwd_offset=0.0):
    ik_x = torso_y + fwd_offset
    # lateral-sign mirror: perception torso +x is robot right, but the IK
    # solver URDF has +Y = left (left_shoulder y=+0.10022, right_shoulder
    # y=-0.10021), so an un-negated channel aims every target at the mirrored
    # side and the hand misses the object's side. every other builder defers to
    # this one, so the negation propagates to all of them.
    ik_y = -torso_x
    ik_z = torso_z
    R = np.eye(3)
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_with_normal(torso_x, torso_y, torso_z,
                                   normal_x, normal_y, normal_z,
                                   fwd_offset=0.0):
    """Build an oriented wrist SE3 target from a surface normal.

    The normal is in torso frame (X=lateral, +x = robot RIGHT, Y=forward,
    Z=height).
    Position remapping follows ``make_wrist_target``:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)

    The normal is remapped to IK frame and used as the approach direction (Z).
    X is a horizontal vector perpendicular to Z; Y = cross(Z, X).

    Parameters
    ----------
    torso_x, torso_y, torso_z : float
        Torso-frame target position.
    normal_x, normal_y, normal_z : float
        Torso-frame surface normal components (from depth_projector_node).
    fwd_offset : float
        Forward offset applied in IK frame.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix (pin.SE3).
    """
    # same remap as builder
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    # remap the normal from the torso frame (X=lateral, Y=forward, Z=height)
    # into the IK frame (X=forward, Y=lateral, Z=height).
    normal_ik = np.array([normal_y, normal_x, normal_z], dtype=np.float64)
    n = np.linalg.norm(normal_ik)
    if n < 1e-9:
        # degenerate normal: fall back to the identity orientation
        R = np.eye(3)
        return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous
    normal_ik = normal_ik / n

    # X = palm direction, pointing at the object (opposite the surface normal)
    X = -normal_ik
    X = X / np.linalg.norm(X)
    # Z = finger-spread axis: vertical for the horizontal curl plane
    Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    # Y completes the right-handed frame
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)
    # re-orthogonalize Z
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)

    R = np.column_stack([X, Y, Z])
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_middle_down(torso_x, torso_y, torso_z, fwd_offset=0.0):
    """Build an oriented wrist SE3 target with middle finger pointing DOWN.

    The middle finger (wrist Y-axis, thumb-to-pinky direction) is set to
    point straight down (perpendicular to the table surface).  This ensures
    the middle finger contacts a horizontally-lying object first during a
    guarded top-down descent.

    Position remapping follows ``make_wrist_target``:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)

    Rotation (in IK frame):
      Y = [0, 0, -1]   middle finger DOWN
      X = [1, 0, 0]    palm forward
      Z = cross(X, Y)  completes right-handed frame

    Parameters
    ----------
    torso_x, torso_y, torso_z : float
        Torso-frame target position.
    fwd_offset : float
        Forward offset applied in IK frame.

    Returns
    -------
    np.ndarray
        4x4 homogeneous transformation matrix (pin.SE3).
    """
    # same remap as builder
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    # Y = middle finger points down (-Z in the IK frame)
    Y = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    # X = palm pointing forward (horizontal)
    X = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # Z = completes the right-handed frame
    Z = np.cross(X, Y)

    R = np.column_stack([X, Y, Z])
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_middle_down_left(torso_x, torso_y, torso_z, fwd_offset=0.0):
    """LEFT-arm middle-down wrist SE3 target — palm DOWN.

    The left hand is mirror-built: its physical palm normal is -Y_local
    (measured R_ee convention: Y = palm normal, right +Y / left -Y).  So
    the LEFT-frame mirror of make_wrist_target_middle_down has the palm
    DIRECTION in the Y column sign-flipped with respect to the right:
      right:  Y = [0, 0, -1]  =>  right palm (-Y_local ... +Y_local) down
      left:   Y = [0, 0, +1]  =>  left palm (-Y_local) down
    LATENT BUG FIX: the left horizontal flow used the RIGHT unmirrored
    (make_wrist_target_middle_down) at all three sites 2026-08-29, which
    commanded the left palm normal UP.  det(R) = +1 for both mirrors.

    Position remapping follows make_wrist_target:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)
    """
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    X = np.array([1.0, 0.0, 0.0], dtype=np.float64)   # fingers point forward
    Y = np.array([0.0, 0.0, 1.0], dtype=np.float64)   # left palm normal points down
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)
    R = np.column_stack([X, Y, Z])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, f'middle_down_left R det != +1: {np.linalg.det(R)}'
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_upright_grasp(torso_x, torso_y, torso_z, fwd_offset=0.0,
                                    pad_normal=None):
    """DEPRECATED (2026-08-26) — superseded by make_wrist_target_claw();
    kept as a dead reference for the trial record.  v1: geometrically
    degenerate — the palm normal always collapses to the approach axis, so
    the rotation is numerically the old identity (no wrist flexion) and the
    open fingers lead the insert into the object (fingertip knock-offs).
    Old v1 docstring: Build the UPRIGHT_GRASP wrist SE3 target (vertical /
    upright-bottle pick).

    Orientation: the PALM normal points along the approach axis — from the
    wrist toward the object centre at the grasp height, projected into the
    horizontal plane — so the palm is face-on to the object and the closure
    wraps around its sides.  This replaces the old identity rotation, which
    made the palm contact the object first while the wrist site was still
    ~2 cm short of the insert target, so the guarded insert kept advancing
    and shoved the bottle.  Reuses the make_wrist_target_with_normal
    rotation construction (X = palm direction, Z = vertical, Y = cross).

    Position remapping follows make_wrist_target:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)

    pad_normal (torso frame, e.g. the depth projector's surface normal) is
    used only when the geometric palm normal is degenerate (wrist parked on
    the approach axis at the object-centre height) — the fallback treats it
    as a surface normal (palm = -normal, like make_wrist_target_with_normal)
    and warns; never crashes.
    """
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    # palm direction (IK frame): wrist to object centre, flattened to horizontal.
    palm_dir = np.array([torso_y - ik_x, torso_x - ik_y, 0.0], dtype=np.float64)
    n = np.linalg.norm(palm_dir)
    if n < 1e-9:
        sys.stderr.write(
            '[pick_daemon] WARNING: UPRIGHT_GRASP palm normal degenerate '
            '(wrist on the approach axis at the object-centre height) — '
            'falling back to the approach-pad normal\n')
        sys.stderr.flush()
        X = None
        if pad_normal is not None:
            pad_ik = np.array([pad_normal[1], pad_normal[0], 0.0],
                              dtype=np.float64)
            pn = np.linalg.norm(pad_ik)
            if pn > 1e-9:
                X = -pad_ik / pn  # palm faces object, against surface normal
        if X is None:
            X = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # approach from the front
    else:
        X = palm_dir / n

    Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    Y = np.cross(Z, X)
    Y = Y / np.linalg.norm(Y)
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)
    R = np.column_stack([X, Y, Z])
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_claw(torso_x, torso_y, torso_z, fwd_offset=0.0):
    """UPRIGHT_GRASP v2 — real wrist flexion: palm-down claw from above.

    The 2026-08-25 trial showed make_wrist_target_upright_grasp is
    geometrically degenerate: the wrist target sits ON the object's forward
    axis, so 'wrist -> object centre' always equals the approach axis and
    the resulting rotation is numerically the old identity — zero visible
    wrist flexion, and the open fingers led the wp4 insert so the
    fingertips stabbed the bottle's upper flank and knocked it over.

    v2 flexes the wrist ~90 degrees: the palm normal (wrist X) points DOWN,
    the three fingers hang as a claw, and the wrist parks directly ABOVE
    the object centre, with the URDF grip zone (17-69 mm along the palm
    normal in front of the wrist site -> [z-69 mm, z-17 mm]) centred on
    the grasp height.  Closure pinches the object's waist from the two
    lateral sides; no fingertip-first insert against a cylindrical flank
    remains — the only possible contact during the guarded descend is the
    claw's inner gap (guarded by the pressure insert guard).

    Position remapping follows make_wrist_target:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)
    """
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    # IK frame: X=forward, Y=lateral+left, Z=height.
    #   X = palm normal down (palm faces the floor, which is the flexion)
    #   Y = lateral: the finger pair and thumb pinch the object's two sides
    #   Z = cross(X, Y) = forward
    X = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    Y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)
    R = np.column_stack([X, Y, Z])
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def _rotate_about(R, axis, angle):
    """Rotate frame R by angle (rad) about axis (3-vector, right-hand rule):
    R' = Rot(axis, angle) @ R.  Rodrigues closed form — no pinocchio
    dependency (the tv env pinocchio version is not guaranteed to expose
    AxAng).  Angle 0 / zero axis -> R unchanged."""
    ax = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(ax))
    if norm < 1e-12 or abs(float(angle)) < 1e-12:
        return np.asarray(R, dtype=np.float64)
    ax = ax / norm
    th = float(angle)
    K = np.array([[0.0, -ax[2], ax[1]],
                  [ax[2], 0.0, -ax[0]],
                  [-ax[1], ax[0], 0.0]])
    return ((np.eye(3) + np.sin(th) * K
             + (1.0 - np.cos(th)) * (K @ K)) @ np.asarray(R, dtype=np.float64))


def make_wrist_target_claw_v3(torso_x, torso_y, torso_z, dir_lateral,
                              deviation=0.0, fwd_offset=0.0):
    """UPRIGHT_GRASP v3.1 — WALL grip: [6]-index wrist flexion, palm to the
    wall, fingers vertical.
    v4.0 (2026-08-30): RETIRED from the commanded flow — the upright
    waypoints return to the ORIGINAL identity design
    (make_wrist_target_upright_v4).  The fixed base-frame wall rotation
    made the wrist chase a pose the arm rarely reached; the solver dumped
    the residual into the wrist (palm-to-sky / pitch / yaw in the
    v3.1-v3.3 trials) — the original never did this (identity = a
    position-only target inside reach).  Kept as history.
    v3.1.1 adds the optional ulnar DEVIATION (2026-08-30 user design): a
    slow in-palm-plane roll about the palm normal during the approach,
    straight->deviate (see the constants block and the wall flows).
    deviation=0.0 keeps the exact v3.1 behaviour (all call sites may omit
    it).

    v3.1 (2026-08-30) fixes the v3 orientation bug.  v3 wrote
      X = [0, dir_lateral, 0]  ("palm normal")
      Y = [0, 0, 1]            ("fingers")
    but the MEASURED R_ee convention is: X = FINGERS-extend, Y = physical
    PALM NORMAL (right +Y, left -Y), Z = finger-spread (live pinocchio FK,
    same URDF the solver uses — the R_ee frame is
    <arm>_wrist_yaw_joint + [0.05, 0, 0]).  So v3's X column actually put
    the FINGERS laterally into the wall and its Y column put the right
    palm normal UP -> the observed 2026-08-29 failure (palm facing the
    sky/ceiling, forearm yawed, [5] moved up/down instead of [6] flexion).
    The "X = palm normal" mislabel traces to mjlab IK_FK_REFERENCE.md
    ("palm-facing = X") and the old daemon builder docstrings.

    Corrected frame — ONE frame for both arms (the per-arm Y sign flip,
    right physical palm +Y / left physical palm -Y, gives both arms the
    palm face-on to the object along the approach axis):
    v3.3 (2026-08-30) note: the original FSM parked BOTH wrists at the
    object's own lateral tx (not on the -/+ side of it) — the wall slide
    triad that replaced it put the wrist radius+standoff to the side and
    its target sat outside the reachable shell (2026-08-30 trial residual
    == the displacement exactly), which the solver baked into the wrist
    pitch.  All call sites now pass tx; the wall-triad values remain as
    DIAGNOSTIC log numbers only.  v3.4 (2026-08-30): the corrective
    NO-PITCH gate was REMOVED per user directive (DIAG-ONLY now — see the
    wp4-FK decomposition) and the upright flows were restored to the
    original insert guard + retract; see the upright waypoint comments.
      X = [0, 0, 1]      FINGERS UP (vertical)      <- user's wrist_yaw [6]
      Y = [0, 1, 0]      palm-normal sign basis (right palm = +Y, left = -Y)
      Z = cross(X, Y)    spread = [-1, 0, 0]  (det(R) = +1, R^T R = I)
    The rotation is authored in SOLVER/base frame coordinates (the same
    frame FK reports — positions use the ik remap below, the rotation does
    NOT; see robot_arm_ik.py rotational_error = log3(oMf @ cTf[:3,:3].T)).

    NOTE on the flexion index: with fingers up and palm normal horizontal,
    the wrist rotation about the vertical object is achieved by the dual-14
    index 13 (right) / 6 (left) — the LAST element of the per-arm 7-vector
    = the user's [6] = wrist_yaw = in/out flexion; [12]/[5] (up/down) and
    [11]/[4] (forearm yaw) must stay near zero.  dir_lateral is accepted
    for signature compatibility and carries the arm side: the base frame
    is ONE frame both arms (its Y sign basis + position-side flip cancel);
    when deviation != 0 the sign of dir_lateral selects the mirrored
    deviation sense (DEV_ROT_SIGN * sign(dir_lateral)).

    DEVIATION (v3.1.1): if deviation != 0, the target becomes
    R_dev = Rot(n_palm, s_dev * deviation) @ R_WALL with
    n_palm = [0,1,0] (R_WALL's own Y column) — the palm NORMAL is
    unchanged (the contact plane still projects onto the wall) and the
    finger line rolls inside the palm plane, straight->deviate during the
    approach (waypoint-ramped delta; frozen at contact; the wrist GN
    realizes it in joint space — the joint map at the wall pose is
    pose-dependent, so this is a body-frame policy, never a joint-index
    policy).

    Position remapping follows make_wrist_target:
      ik_x = torso_y + fwd_offset   (forward; fwd_offset default 0.0)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)

    fwd_offset (v3.2 restore, 2026-08-30): the approach waypoints (wp2/wp3
    only) pass APPROACH_FORWARD_OFFSET (-0.08) so the wrist parks BEHIND the
    contact depth exactly like the original make_wrist_target FSM did — the
    approach then verifies inside the reachable shell.  wp4 / retract keep
    the default 0.0 (full contact depth): their targets are beyond the reach
    cap BY CONSTRUCTION — the guarded insert + pressure stop is the safety
    when the wrist walks the final 8 cm (original FSM semantics).
    """
    ik_x = torso_y + fwd_offset
    ik_y = -torso_x
    ik_z = torso_z

    X = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    Y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Z = np.cross(X, Y)
    Z = Z / np.linalg.norm(Z)
    R = np.column_stack([X, Y, Z])
    assert abs(np.linalg.det(R) - 1.0) < 1e-6, f'claw_v3 R det != +1: {np.linalg.det(R)}'
    if abs(float(deviation)) > 1e-9:
        # ulnar roll about the palm-normal basis [0,1,0]; DEV_ROT_SIGN mirrors
        # per dir_lateral (-1 right, +1 left) and moves the tip line away from
        # the thumb.
        s_dev = DEV_ROT_SIGN * (1.0 if dir_lateral >= 0 else -1.0)
        R = _rotate_about(R, np.array([0.0, 1.0, 0.0], dtype=np.float64),
                          s_dev * float(deviation))
        # defensive guard: Rodrigues about a unit axis cannot change det (max
        # |det-1| ~1 ulp over d in [0,1]); this catches future edits with an
        # unnormalized axis.
        
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, \
            f'claw_v3 deviated R det != +1: {np.linalg.det(R)}'
    return pin.SE3(R, np.array([ik_x, ik_y, ik_z])).homogeneous


def make_wrist_target_upright_v4(torso_x, torso_y, torso_z, dir_lateral,
                                 deviation=0.0, fwd_offset=0.0):
    """v4.0 upright wrist target — ORIGINAL identity design + ADD-ON.

    v4.0 (2026-08-30, user: "restore all design accordingly from the
    original file, while maintain the new features as add-on"): the base
    target is the ORIGINAL's make_wrist_target — identity rotation
    (R = np.eye(3)), original position channel (ik_x = torso_y +
    fwd_offset, ik_y = -torso_x, ik_z = torso_z).  deviation=0.0 returns a
    byte-identical homogeneous target to the original's builder; the v3
    WALL frame (make_wrist_target_claw_v3) is RETIRED from the commanded
    flow — the fixed base-frame rotation produced the palm-to-sky / pitch
    / yaw motions of the v3.1-v3.3 trials, and the original never had them
    because an identity orientation needs no fixed base-frame rotation and
    its targets sat inside reach.

    ADD-ON (maintained, default OFF — flip DEVIATION_ENABLED): the
    v3.1.1 ulnar deviation, R_dev = Rot([0,1,0], s_dev*deviation) @ R_base.
    [0,1,0] is the identity frame's OWN palm-normal column (FK-verified:
    R_ee Y = physical palm normal, right +Y / left -Y), so the deviation is
    FRAME-INDEPENDENT — the palm normal / contact-face projection is
    unchanged; only the finger line rolls in the palm plane (straight ->
    deviate during the approach, frozen at contact).

    Position remapping follows make_wrist_target:
      ik_x = torso_y + fwd_offset   (forward)
      ik_y = -torso_x  (lateral; v5.2 negated: torso +x = RIGHT, URDF +y = LEFT)
      ik_z = torso_z                (height)

    dir_lateral: arm side (+1 right / -1 left) — selects the mirrored
    deviation sense (DEV_ROT_SIGN * sign(dir_lateral)); the position
    channel is side-independent (original).
    """
    target = make_wrist_target(torso_x, torso_y, torso_z, fwd_offset=fwd_offset)
    if abs(float(deviation)) > 1e-9:
        s_dev = DEV_ROT_SIGN * (1.0 if dir_lateral >= 0 else -1.0)
        R = _rotate_about(target[:3, :3],
                          np.array([0.0, 1.0, 0.0], dtype=np.float64),
                          s_dev * float(deviation))
        # defensive: Rodrigues keeps det(R) = det(base); guards future edits
        assert abs(np.linalg.det(R) - 1.0) < 1e-6, \
            f'upright_v4 deviated R det != +1: {np.linalg.det(R)}'
        target = pin.SE3(R, target[:3, 3]).homogeneous
    return target
# 4 cm suits a ~20 cm object but overshoots a ~6 cm one, so clamp to 20% of
# object height when height_m is available.
GRASP_Z_OFFSET_HEIGHT_FRACTION = 0.20
_grasp_z_offset_no_height_warned = False
# last good open-hand pressure baseline, reused if a per-attempt read fails or
# max=0.0: a fresh relative baseline must be live at verdict time, or the
# absolute tactile compare arms again. [0]-indexed so assignment needs no
# global.
_TACTILE_BASELINE_LAST_GOOD = [None]


def _clamped_grasp_z_offset(height_m):
    """Effective vertical grasp offset above the detection centroid:
    min(GRASP_Z_OFFSET, GRASP_Z_OFFSET_HEIGHT_FRACTION * object_height_m).
    Falls back to plain GRASP_Z_OFFSET (with one warning) when the object
    height is unavailable — never crashes, never refuses a pick."""
    global _grasp_z_offset_no_height_warned
    if height_m is not None and 0.0 < float(height_m) <= HEIGHT_M_MAX:
        return min(GRASP_Z_OFFSET,
                   GRASP_Z_OFFSET_HEIGHT_FRACTION * float(height_m))
    if not _grasp_z_offset_no_height_warned:
        _grasp_z_offset_no_height_warned = True
        sys.stderr.write(
            '[pick_daemon] WARNING: object height unavailable — using plain '
            f'GRASP_Z_OFFSET ({GRASP_Z_OFFSET:.3f} m) for the grasp-z clamp\n')
        sys.stderr.flush()
    return GRASP_Z_OFFSET


# ── the extra place lift ─────────────────────────────────────────────────────
# a place first raises the holding arm alone to 2.5 times the detected object
# height, clamped to the maximum safe rise. it is transient clearance: the
# release point and drop height stay unchanged, pz and hover_z are not raised,
# and the arm gives the z back before the release.
PLACE_EXTRA_LIFT_HEIGHT_FRACTION = 2.5   # x object height (cf. GRASP_Z_OFFSET_HEIGHT_FRACTION)
# ceiling: the largest rise above the LIFT-class carry pose with x and y held
# is 0.409 m, about 0.43 m in practice. the 2.5x law holds only for
# H <= 0.164 m; above that this cap saturates and the achieved rise is logged
# on every place. 0.40 leaves margin under that ceiling.
PLACE_EXTRA_LIFT_MAX_M = 0.40
PLACE_EXTRA_LIFT_TOL_M = SOLVE_VERIFY_POS_TOL   # m; same FK residual the solver checks
PLACE_EXTRA_LIFT_ATTEMPTS = 3                   # halving retries, then no-raise
_place_extra_lift_no_height_warned = False
# last valid pick-time object height (m); None means unknown. [0]-indexed so
# assignment needs no `global`. written unconditionally at every pick
# plausibility gate, so a height-less pick clears it and a later place cannot
# reuse the wrong object. process-local: a daemon restart between pick and
# place loses it, and the place grammar's height token is the durable path.
_PLACE_LAST_OBJECT_HEIGHT_M = [None]


def _place_extra_lift(arm, arm_ik, side, height_m, verb, log_prefix=''):
    """Raise ONE arm vertically by min(2.5*H, PLACE_EXTRA_LIFT_MAX_M).

    Returns (requested_m, achieved_m, ok).  ok=False means NO arm motion was
    commanded at all — never raises, never refuses (warn-and-degrade, matching
    _clamped_grasp_z_offset's contract).  A place whose object height is
    unknown or implausible still PLACES, it simply does not raise.

    CARTESIAN, not joint-space: no single joint delivers +0.15 m from the
    carry pose (best dz/dq is shoulder_pitch at 0.121 m/rad, so 1.5 rad buys
    0.135 m), and a joint-space raise both undershoots a 2.5×H law
    systematically and drives joints toward their stops.  The target is
    translation-ONLY (the achieved wrist rotation is kept) so the held object
    does not rotate or re-seat in the grip — the grip is closed on the object
    throughout, so any wrist rotation would drag it.

    The IK is limit-bounded by construction, but it SOFT-FAILS loud-and-
    best-effort rather than returning None, so the FK residual check below is
    mandatory, not belt-and-braces: an unverified over-range solve would be
    commanded verbatim to the writer (no joint-limit check, no reach gate, no
    stall watchdog on any layer) and would press the URDF stop at full kp.
    """
    global _place_extra_lift_no_height_warned

    _requested = 0.0
    try:
        # plausibility: same bound as the pick gate (HEIGHT_M_MAX). the range
        # test also rejects non-finite values, since `0.0 < nan` and
        # `nan <= 0.5` are both False.
        _rise = 0.0
        if height_m is not None:
            try:
                _h_m = float(height_m)
            except (TypeError, ValueError):
                _h_m = None
            if _h_m is not None and 0.0 < _h_m <= HEIGHT_M_MAX:
                _rise = min(PLACE_EXTRA_LIFT_HEIGHT_FRACTION * _h_m,
                            PLACE_EXTRA_LIFT_MAX_M)
        if _rise <= 0.0:
            if not _place_extra_lift_no_height_warned:
                _place_extra_lift_no_height_warned = True
                sys.stderr.write(
                    f'[pick_daemon] {log_prefix}no object height '
                    f'(fallback rise 0.00) — proceeding\n')
                sys.stderr.flush()
            return 0.0, 0.0, False
        _requested = _rise

        # 2. measured baseline: the writer's `q` verb returns actual lowstate q.
        _q_now = arm.get_current_dual_arm_q().copy()
        _T_ee = get_ee_pose(arm_ik, _q_now, side)
        if _T_ee is None:
            sys.stderr.write(
                f'[pick_daemon] {log_prefix}FK failed — no raise, '
                f'place proceeds\n')
            sys.stderr.flush()
            return _requested, 0.0, False
        _R0 = np.asarray(_T_ee.rotation, dtype=np.float64).copy()
        _p0 = np.asarray(_T_ee.translation, dtype=np.float64).copy()

        # idle-arm target is its current wrist pose (idle-pin idiom), already
        # parked at CARRY_TUCK by step 2 on the coords exits.
        _other = 'left' if side == 'right' else 'right'
        _idle_wrist = get_wrist_pose(arm_ik, _q_now, _other)
        _idle_t = (pin.SE3(np.eye(3), _idle_wrist).homogeneous
                   if _idle_wrist is not None
                   else (R_NEUTRAL if _other == 'right' else L_NEUTRAL))

        _try_rise = _rise
        for _attempt in range(1, PLACE_EXTRA_LIFT_ATTEMPTS + 1):
            # 4. target is translation only, so the grip does not rotate.
            _target = pin.SE3(
                _R0, _p0 + np.array([0.0, 0.0, _try_rise])).homogeneous
            # 6. arg order: the first positional is always the left target.
            if side == 'left':
                _left_t, _right_t = _target, _idle_t
            else:
                _left_t, _right_t = _idle_t, _target
            _sol_q, _sol_tau = _solve_ik_verified(
                arm_ik, _left_t, _right_t, _q_now, side, log_prefix)
            if _sol_q is None:
                break
            _pos_err, _ori_err, _, _ = verify_ik_target(
                arm_ik, _sol_q, side, _target)
            if _pos_err is None:
                break
            if _pos_err <= PLACE_EXTRA_LIFT_TOL_M:
                _T_got = get_ee_pose(arm_ik, _sol_q, side)
                if _T_got is None:
                    break
                _achieved = (float(np.asarray(_T_got.translation,
                                              dtype=np.float64)[2])
                             - float(_p0[2]))
                # idle dofs: transport needs 14 floats, so pin the idle slice to
                # the measured q (the _t5_joint_move pattern). not freeze_idle_arm,
                # which would hard-command CARRY_TUCK when step 2 never ran.
                _cmd_q = np.array(_sol_q, dtype=np.float64).copy()
                _idle_idx = slice(0, 7) if side == 'right' else slice(7, 14)
                _cmd_q[_idle_idx] = _q_now[_idle_idx]
                sys.stderr.write(
                    f'[pick_daemon] {log_prefix}raise {side} arm: '
                    f'H={float(height_m):.3f} m requested='
                    f'{_requested:.3f} m clamped={_try_rise:.3f} m '
                    f'achieved={_achieved:.3f} m (pos_err={_pos_err:.4f})\n')
                sys.stderr.flush()
                # tau from the same 14-vector that is commanded (_t5_joint_move),
                # or the idle arm's gravity comp will not match its own q.
                move_arm_smooth(arm, _cmd_q, _arm_gravity_ff(_cmd_q,
                                                            np.zeros(14)))
                return _requested, _achieved, True
            # 7. residual too large: halve and retry, which clamps to the
            # maximum safe rise without a magic number.
            sys.stderr.write(
                f'[pick_daemon] {log_prefix}raise attempt {_attempt}/'
                f'{PLACE_EXTRA_LIFT_ATTEMPTS}: pos_err={_pos_err:.4f} m > tol '
                f'{PLACE_EXTRA_LIFT_TOL_M:.3f} at rise={_try_rise:.3f} m — '
                f'halving\n')
            sys.stderr.flush()
            _try_rise *= 0.5

        sys.stderr.write(
            f'[pick_daemon] {log_prefix}no FK-verified raise within '
            f'{PLACE_EXTRA_LIFT_ATTEMPTS} attempts (requested='
            f'{_requested:.3f} m) — commanding NO raise, place proceeds\n')
        sys.stderr.flush()
        return _requested, 0.0, False
    except Exception as _el_e:
        # a raise fault must never reach the per-command handler, which would
        # freeze the arm, drop the RC interlock and skip DONE.
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}raise failed: {_el_e} — no raise, '
            f'place proceeds\n')
        sys.stderr.flush()
        return _requested, 0.0, False


def get_wrist_pose(arm_ik, current_q, side='right'):
    """Return wrist position [x,y,z] in IK frame using forward kinematics.

    Uses the parent-joint method: FK → joint placement → apply frame offset.
    This avoids ``data.oMf`` which may not include end-effector frames
    (R_ee / L_ee at IDs 107/106) in all Pinocchio versions.

    Returns None if FK fails.
    """
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    frame_id = arm_ik.R_hand_id if side == 'right' else arm_ik.L_hand_id

    try:
        pin.forwardKinematics(model, data, current_q)
        frame = model.frames[frame_id]
        parent_joint_id = frame.parentJoint
        # oMi has all the joint placements (forwardKinematics always fills them in)
        placement = data.oMi[parent_joint_id] * frame.placement
        return placement.translation.copy()
    except Exception as e:
        try:
            _pj = model.frames[frame_id].parentJoint
        except Exception:
            _pj = 'N/A'
        sys.stderr.write(
            f'[pick_daemon] get_wrist_pose failed: {e} '
            f'(|q|={len(current_q)}, model.nq={model.nq}, '
            f'nframes={model.nframes}, njoints={model.njoints}, '
            f'frame_id={frame_id}, parent_joint={_pj})\n')
        sys.stderr.flush()
        return None


# ── the verified-solve machinery ──────────────────────────────────────────────
# solve_ik never returns None: its exception path returns the passed q. every
# output is also blended 0.4/0.3/0.2/0.1, with identical duplicates skipped, so
# a 4-solve pick's wp4 command is only ~40% wp4 and a failed solve is commanded
# as-is with the Ipopt errors hidden. the wash, FK verify and wrist GN restore
# a verified target without touching the read-only solver.


def get_ee_pose(arm_ik, q14, side='right'):
    """Full R_ee frame FK — pin.SE3 (rotation + translation).

    Same parent-joint construction as get_wrist_pose:
    ``data.oMi[parent_joint] * frame.placement`` with
    frame = <side>_ee = <side>_wrist_yaw_joint + [0.05, 0, 0] (identity
    rotation).  THIS is the frame whose columns are measured as X =
    fingers-extend, Y = physical palm normal (right +Y, left -Y), Z =
    finger-spread.  Returns None on failure; never raises.
    """
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    frame_id = arm_ik.R_hand_id if side == 'right' else arm_ik.L_hand_id

    try:
        pin.forwardKinematics(model, data, q14)
        frame = model.frames[frame_id]
        parent_joint_id = frame.parentJoint
        placement = data.oMi[parent_joint_id] * frame.placement
        return placement
    except Exception:
        return None


def verify_ik_target(arm_ik, q14, side, target_T):
    """FK-verify a solve result against its SE3 target.

    pos_err = |p_ee - p_target|  (m); ori_err = angle of R_ee . R_target^T
    (rad) — the same residual the solver minimizes (robot_arm_ik.py:127-128,
    log3 of oMf @ cTf^T).  Also returns det(R_ee); a |det| != 1 result means
    the achieved frame is a mirror (asserted never to happen).
    Returns (pos_err, ori_err, R_ee, det) or (None,)*4 if FK fails.
    """
    T_ee = get_ee_pose(arm_ik, q14, side)
    if T_ee is None:
        return None, None, None, None
    R_ee = np.asarray(T_ee.rotation, dtype=np.float64)
    p_ee = np.asarray(T_ee.translation, dtype=np.float64)
    R_t = np.asarray(target_T[:3, :3], dtype=np.float64)
    p_t = np.asarray(target_T[:3, 3], dtype=np.float64)
    pos_err = float(np.linalg.norm(p_ee - p_t))
    R_err = R_ee @ R_t.T
    ori_err = float(np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)))
    return pos_err, ori_err, R_ee, float(np.linalg.det(R_ee))


def solve_wrist_orientation_in_place(arm_ik, q14, side, target_T,
                                     iters=WRIST_GN_ITERATIONS,
                                     tol=WRIST_GN_TOL):
    """Wrist-space Gauss-Newton — the W2 spec restore (REVIEW_TASK1 W2,
    IK_SOLVER_RESEARCH §6.1, 02-daemon-poses.md:296-328).

    Minimizes ‖pin.log3(R_ee(q|wrist) @ R_targetᵀ)‖ over the side's THREE
    wrist joints (user map: [11,12,13] right = forearm-yaw / up-down /
    [6]-index FLEXION; [4,5,6] left), all other joints held:
      - forward-difference Jacobian h=1e-3
      - lstsq(J, -r) step
      - clip to WRIST_LIMITS
      - warm start = current wrist q
      - stagnation guard: 5 iters without improvement -> converged=False
      - best iterate retained; never raises
    Returns (q_new, ori_err, pos_err, converged) — the full 14-qq copy with
    the side's wrist replaced.
    """
    iw = RIGHT_WRIST_Q_IDX if side == 'right' else LEFT_WRIST_Q_IDX
    qb = np.array(q14, dtype=np.float64)
    qw = qb[iw].copy()
    R_target = np.asarray(target_T[:3, :3], dtype=np.float64)

    def _resid(qw_):
        qq = qb.copy()
        qq[iw] = qw_
        T_ee = get_ee_pose(arm_ik, qq, side)
        if T_ee is None:
            return None
        R_ee = np.asarray(T_ee.rotation, dtype=np.float64)
        return np.asarray(pin.log3(R_ee @ R_target.T), dtype=np.float64)

    h = 1e-3
    qw_best = qw.copy()
    r_best = _resid(qw_best)
    err_best = float(np.linalg.norm(r_best)) if r_best is not None else float('inf')
    converged = False
    stagnant = 0
    for _ in range(iters):
        r = _resid(qw)
        if r is None:
            break
        err = float(np.linalg.norm(r))
        if err < err_best:
            err_best = err
            qw_best = qw.copy()
        if err <= tol:
            converged = True
            break
        # finite-difference Jacobian
        J = np.zeros((3, 3))
        ok = True
        for j in range(3):
            qw_h = qw.copy()
            qw_h[j] += h
            rh = _resid(qw_h)
            if rh is None:
                ok = False
                break
            J[:, j] = (rh - r) / h
        if not ok:
            break
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        qw_try = np.clip(qw + step,
                         [-WRIST_LIMITS[int(i)] for i in iw],
                         [WRIST_LIMITS[int(i)] for i in iw])
        if np.allclose(qw_try, qw, atol=1e-9):
            break
        r_try = _resid(qw_try)
        if r_try is not None and float(np.linalg.norm(r_try)) < err:
            qw = qw_try
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= 5:
                break
        if converged:
            break

    q_out = qb.copy()
    q_out[iw] = qw_best
    pos_err, ori_err, _, _ = verify_ik_target(arm_ik, q_out, side, target_T)
    if pos_err is None:
        pos_err, ori_err = float('inf'), float('inf')
    converged = converged or (err_best <= tol)
    return q_out, err_best, pos_err, converged


def _solve_ik_verified(arm_ik, left_target, right_target, cur_q, side,
                       log_prefix):
    """solve_ik + verify-wash + wrist-GN polish; SOFT-FAIL on exhaustion
    (v3.2 restore, 2026-08-30).

    v3.4 (2026-08-30 user directive): the v3.3 NO-PITCH corrective gate
    (`_enforce_pitch_freeze` + the enforce_pitch_freeze flag) is REMOVED
    COMPLETELY — "do not pitch hands upwards or downwards" stays as a
    DIAGNOSTIC only (the wp4-FK body-axis decomposition logs flag the
    achieved pitch vs PITCH_FROZEN_TOL above; nothing corrects it).  The
    upright flows were restored to the ORIGINAL insert guard + retract at
    the same time (see the upright waypoint comments).

    Wash loop (<= SOLVE_WASH_ATTEMPTS): each solve runs with warm = the
    previous attempt's OUTPUT sol_q (identical warm start is deterministic,
    so re-solving it is a filter no-op — verified separately by the dedup);
    the filter weights [0.4, 0.3, 0.2, 0.1] sum to 1, so repeated solves of
    the same waypoint converge the blend to the raw solution.  After each
    attempt, FK-verify (pos <= SOLVE_VERIFY_POS_TOL, ori <= SOLVE_VERIFY_ORI_TOL).
    Exception signature (sol_q ~ warm AND pos_err large = Ipopt raised:
    robot_arm_ik.py:246-271 returns the passed q): log loud, break the wash
    (warm retries cannot help — the filter already got the input).

    Fail path: GN the best iterate; if wrist-GN converged AND pos ok,
    re-tau via _arm_gravity_ff (GN changed q); else loud
    'ERROR:IK verify failed: …' and return the GN-polished best-effort
    (gn_q, re-tau) — v3.1's hard-fail (None, None) killed the whole FSM at
    the first waypoint whose target sat beyond the reach cap (before any
    arm motion); the original FSM commanded whatever the solver returned
    and let the guarded move be the safety.  The failure stays LOUD (the
    ERROR line above); the post-move FK checks + gates judge what happened.
    Only when the solver returned NO usable q at all (best is None) is
    (None, None) returned — the original's True-fail, the caller's
    ERROR:IK failed branches.
    """
    if (side == 'right' and (left_target is None or right_target is None)) or \
       (side == 'left' and (left_target is None or right_target is None)):
        return None, None
    target_T = right_target if side == 'right' else left_target
    warm = np.array(cur_q, dtype=np.float64)
    best = None          # (sol_q, sol_tauff, pos_err, ori_err)
    best_pos_err = float('inf')
    raised = False
    for attempt in range(1, SOLVE_WASH_ATTEMPTS + 1):
        with _SuppressStdout():
            sol_q, sol_tauff = arm_ik.solve_ik(
                left_target, right_target,
                current_lr_arm_motor_q=warm.copy(),
                current_lr_arm_motor_dq=np.zeros(14))
        pos_err, ori_err, _, _ = verify_ik_target(arm_ik, sol_q, side, target_T)
        if pos_err is None:
            sys.stderr.write(
                f'[pick_daemon] {log_prefix}IK verify attempt {attempt}/'
                f'{SOLVE_WASH_ATTEMPTS}: FK failed — aborting wash\n')
            sys.stderr.flush()
            break
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}IK verify attempt {attempt}/'
            f'{SOLVE_WASH_ATTEMPTS}: pos_err={pos_err:.4f} m '
            f'ori_err={ori_err:.4f} rad\n')
        sys.stderr.flush()
        if pos_err < best_pos_err:
            best_pos_err = pos_err
            best = (sol_q, sol_tauff, pos_err, ori_err)
        if (np.allclose(sol_q, warm, atol=1e-4)
                and pos_err > SOLVE_VERIFY_POS_TOL):
            sys.stderr.write(
                f'[pick_daemon] {log_prefix}Ipopt solve raised — warm retries '
                f'cannot dilute the filter, continuing to GN polish\n')
            sys.stderr.flush()
            raised = True
            break
        if pos_err <= SOLVE_VERIFY_POS_TOL and ori_err <= SOLVE_VERIFY_ORI_TOL:
            return sol_q, sol_tauff
        warm = sol_q

    # wash exhausted: wrist-space GN on the best iterate.
    gn_base = best[0] if best is not None else np.array(cur_q, dtype=np.float64)
    gn_q, gn_ori_err, gn_pos_err, gn_conv = solve_wrist_orientation_in_place(
        arm_ik, gn_base, side, target_T)
    if gn_conv and gn_pos_err is not None and gn_pos_err <= SOLVE_VERIFY_POS_TOL:
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}wrist-GN polish: '
            f'ori_err={gn_ori_err:.4f} rad pos_err={gn_pos_err:.4f} m '
            f'(wash attempts exhausted{", solver raised" if raised else ""})\n')
        sys.stderr.flush()
        return gn_q, _arm_gravity_ff(gn_q, np.zeros(14))

    if best is None:
        # no usable q: solve_ik returned (None, None) or FK failed every
        # attempt. this is a true fail, left to the caller's ERROR:IK failed
        # branch; cur_q would only no-op the waypoint.
        sys.stderr.write(
            f'[pick_daemon] ERROR:IK verify failed: no usable solve from '
            f'the solver (None / FK failed every attempt) — True-fail, '
            f'caller\'s ERROR:IK failed branch (no best-effort exists)\n')
        sys.stderr.flush()
        return None, None

    sys.stderr.write(
        f'[pick_daemon] ERROR:IK verify failed: pos={gn_pos_err if gn_pos_err is not None else float("inf"):.4f} '
        f'm ori={gn_ori_err:.4f} rad (wash + wrist GN exhausted) — v3.2 '
        f'SOFT-FAIL: commanding the GN-polished best-effort pose (restored '
        f'original FSM semantics — the guarded move + stop-FN is the '
        f'safety; see DEPLOY docs)\n')
    sys.stderr.flush()
    # command whatever wash+GN returned, never (None, None): no guard on the
    # current q. the verify and GN polish stay, since their logs report the
    # waypoint and the caller keeps its gates (pressure evidence, GATE_CONF, FK
    # orientation checks).
    return gn_q, _arm_gravity_ff(gn_q, np.zeros(14))


# ── the deviation ramp + body-axis decomposition ──────────────────────────────
# upright right pick tilt, per attempt. deviation is rotation about the palm
# normal; with the palm sideways it tilts the fingers up or down. attempt 1
# tilts up, 2 is straight and 3 tilts down, so there are 3 attempts. right
# arm only; the left arm keeps _l_dev_tgt. the off flag
# UPRIGHT_TILT_OFF_FLAG makes every attempt +1 on the next pick.
UPRIGHT_TILT_SIGNS = (1.0, 0.0, -1.0)   # attempt 1 up, 2 straight, 3 down
# per-attempt grasp-height shift (m), upright right: the straight attempt
# descends 5 cm more. the z-floor clamp and insert guard stay in force, so a
# shift the table forbids gets clamped.
UPRIGHT_TILT_DZ = (0.0, -0.05, 0.0)


def _upright_tilt_dz(attempt):
    """Grasp-height shift (m) for the attempt; 0.0 when the off flag exists."""
    try:
        if os.path.exists(UPRIGHT_TILT_OFF_FLAG):
            return 0.0
    except Exception:
        pass
    i = min(max(int(attempt), 0), len(UPRIGHT_TILT_DZ) - 1)
    return float(UPRIGHT_TILT_DZ[i])
UPRIGHT_TILT_OFF_FLAG = '/tmp/openbot_upright_tilt_off'
UPRIGHT_ATTEMPTS = len(UPRIGHT_TILT_SIGNS)   # one attempt per tilt sign


def _upright_attempts():
    """3 (one per tilt) normally; the B8i/B8j 2 when the off flag exists."""
    try:
        if os.path.exists(UPRIGHT_TILT_OFF_FLAG):
            return 2
    except Exception:
        pass
    return UPRIGHT_ATTEMPTS


def _upright_tilt_sign(attempt):
    """+1.0 / 0.0 / -1.0 for attempt index 0 / 1 / 2 (later attempts keep
    -1.0); +1.0 for every attempt when the off flag file exists."""
    try:
        if os.path.exists(UPRIGHT_TILT_OFF_FLAG):
            return 1.0
    except Exception:
        pass
    i = min(max(int(attempt), 0), len(UPRIGHT_TILT_SIGNS) - 1)
    return float(UPRIGHT_TILT_SIGNS[i])


def _deviation_ramped(waypoint_step, delta_tgt=DEV_DELTA_TGT):
    """Deviation delta (rad) for approach waypoint `waypoint_step`
    (1-based from the FIRST ramping waypoint): monotone ramp 0 ->
    delta_tgt, bounded by DEV_STEP_MAX per waypoint, clamped to
    [0, min(delta_tgt, DEV_MAX)] — never negative (the wrist never
    deviates past straight the other way).  The contact waypoint is the
    LAST ramping waypoint; callers reuse its delta for the retract
    (frozen at contact — 2026-08-30 user rule)."""
    return float(np.clip(waypoint_step * DEV_STEP_MAX, 0.0,
                         min(float(delta_tgt), DEV_MAX)))


def wrist_rotation_decomposition(arm_ik, q14, side, R_ref):
    """Body-axis decomposition of the achieved rotation (log-only; the
    joint-to-motion assignment is pose-dependent, so the gate asserts on
    BODY axes at the ACTUAL pose, never on joint indices).

    D = log3(R_act @ R_ref^T), projected onto R_act's own columns:
      - fingers axis  = R_act[:,0]  -> PITCH  (up/down: the user's
        NO-PITCH rule during upright contact — |pitch| <= PITCH_FROZEN_TOL)
      - palm normal   = R_act[:,1]  -> DEVIATION (the ulnar roll, ===
        s_dev * commanded delta at contact: s_dev = DEV_ROT_SIGN *
        sign(dir_lateral) — +1 LEFT, -1 RIGHT — so a perfect right-arm
        execution logs the NEGATIVE of the printed positive target;
        compare magnitudes across the side, or read the signed target)
      - spread axis   = R_act[:,2]  -> FLEXION (the in/out palm hinge,
        dominant during the approach)
    Returns (pitch, deviation, flexion) rad or (None,)*3 if FK fails."""
    T_ee = get_ee_pose(arm_ik, q14, side)
    if T_ee is None:
        return None, None, None
    R_act = np.asarray(T_ee.rotation, dtype=np.float64)
    R_ref = np.asarray(R_ref[:3, :3], dtype=np.float64)
    try:
        D = np.asarray(pin.log3(R_act @ R_ref.T), dtype=np.float64)
    except Exception:
        return None, None, None
    return (float(np.dot(D, R_act[:, 0])),
            float(np.dot(D, R_act[:, 1])),
            float(np.dot(D, R_act[:, 2])))


def world_to_ik_frame(world_x, world_y, world_z, robot_x, robot_y, robot_yaw):
    """Convert MuJoCo world-frame coordinates to IK-frame coordinates.

    Transform chain:  world → torso → IK

    world → torso:
        Rotate world displacement by -robot_yaw around world Z, giving
        torso-frame coordinates where X = lateral (+left), Y = forward,
        Z = height.

    torso → IK (G1 arm IK convention — mirrors make_wrist_target):
        ik_x = torso_y   (forward)
        ik_y = torso_x   (LEGACY sign — dead helper, body unchanged; the live
        builders negate: ik_y = -torso_x)
        ik_z = torso_z   (height)

    Parameters
    ----------
    world_x, world_y, world_z : float
        Point in MuJoCo world frame.
    robot_x, robot_y : float
        Robot base position in world frame.
    robot_yaw : float
        Robot yaw angle (radians) in world frame.

    Returns
    -------
    np.ndarray, shape (3,)
        Point in IK frame: [ik_x, ik_y, ik_z].
    """
    # world frame → torso frame
    dx = world_x - robot_x
    dy = world_y - robot_y
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)

    torso_x =  cos_yaw * dx + sin_yaw * dy   # lateral (+left)
    torso_y = -sin_yaw * dx + cos_yaw * dy   # forward
    torso_z = world_z                         # height (yaw does not touch it)

    # torso frame → IK frame
    ik_x = torso_y   # forward
    ik_y = torso_x   # lateral sign (+left)
                     # live builders negate: ik_y = -torso_x
    ik_z = torso_z   # height

    return np.array([ik_x, ik_y, ik_z])


# per-step gravity feedforward: the 250 Hz controller feeds the tau to lowcmd
# and the arm fights gravity at its current pose, so a constant endpoint
# torque (sol_tauff or zeros) is only correct at the target. rnea every 50 Hz
# step is cheap (~0.1 ms on the G1 URDF), so every interpolated pose gets its
# own compensation. this mirrors solve_ik's sol_tauff: pin.rnea on the reduced
# arm model with v=0 and a=0, whose q is the 14-DOF arm vector.
_ARM_IK = None  # main() sets it after building G1_29_ArmIK


def _arm_gravity_ff(q14, fallback):
    """[FF-RE-ENABLED for bisection] Gravity-compensation tau (14-vector) at
    pose q14, or fallback when the IK model is unavailable."""
    global _ARM_IK
    if _ARM_IK is None or not hasattr(_ARM_IK, 'reduced_robot'):
        return fallback
    try:
        m = _ARM_IK.reduced_robot.model
        d = _ARM_IK.reduced_robot.data
        return pin.rnea(m, d, q14, np.zeros(m.nv), np.zeros(m.nv))
    except Exception:
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
# table z-floor clamp: the arm must not be driven into the support surface.
# the risky transitions are the LIFT after a grasp and the return to NEUTRAL
# after a release; both end at NEUTRAL, the one named pose that does not fit
# at a table. in the pelvis frame, with the table plane at -0.110 m (derived in
# _latch_table_floor):
#     NEUTRAL      lowest limb z = -0.2523 -> 0.142 m inside the table
#     READY                      +0.1349   -> clear
#     CARRY_TUCK                 -0.0142   -> clear
# so the clamp only stops NEUTRAL and any low descent short of the surface;
# every already-safe move comes back unchanged.
# this is not collision avoidance: it guards exactly one plane, the table the
# last pick or place latched. the arm lane has no collision model.

# limb mesh AABBs in each link's own frame, from the URDF collision meshes.
# the hand and three distal wrist links are included: they sit lower than the
# hand at READY (45 mm) and CARRY_TUCK (35 mm), so a hand-only clamp
# under-protects by more than the 30 mm margin. constant; regenerating from the
# STL files must reproduce these values.
_LIMB_BBOX = {
    'right_hand_palm_link':     ((-0.00000, -0.02142, -0.04390), (+0.08670, +0.02032, +0.04390)),
    'right_hand_thumb_0_link':  ((-0.01150, +0.00000, -0.00950), (+0.01010, +0.02830, +0.00950)),
    'right_hand_thumb_1_link':  ((-0.01260, -0.00700, -0.01280), (+0.01465, +0.05480, +0.01290)),
    'right_hand_thumb_2_link':  ((-0.00892, -0.00700, -0.01290), (+0.00877, +0.05196, +0.01310)),
    'right_hand_index_0_link':  ((-0.00700, -0.01260, -0.01290), (+0.05480, +0.01465, +0.01280)),
    'right_hand_index_1_link':  ((-0.00700, -0.00892, -0.01310), (+0.05196, +0.00877, +0.01290)),
    'right_hand_middle_0_link': ((-0.00700, -0.01260, -0.01290), (+0.05480, +0.01465, +0.01280)),
    'right_hand_middle_1_link': ((-0.00700, -0.00892, -0.01310), (+0.05196, +0.00877, +0.01290)),
    'right_wrist_roll_link':    ((-0.00000, -0.03000, -0.03000), (+0.05599, +0.03049, +0.03000)),
    'right_wrist_pitch_link':   ((-0.02850, -0.02849, -0.02849), (+0.07450, +0.02849, +0.02849)),
    'right_wrist_yaw_link':     ((-0.01799, -0.03000, -0.03049), (+0.04150, +0.02386, +0.03000)),
    'left_hand_palm_link':      ((+0.00000, -0.02032, -0.04390), (+0.08670, +0.02142, +0.04390)),
    'left_hand_thumb_0_link':   ((-0.01150, -0.02830, -0.00950), (+0.01010, +0.00000, +0.00950)),
    'left_hand_thumb_1_link':   ((-0.01260, -0.05480, -0.01290), (+0.01465, +0.00700, +0.01280)),
    'left_hand_thumb_2_link':   ((-0.00892, -0.05196, -0.01310), (+0.00877, +0.00700, +0.01290)),
    'left_hand_index_0_link':   ((-0.00700, -0.01465, -0.01280), (+0.05480, +0.01260, +0.01290)),
    'left_hand_index_1_link':   ((-0.00700, -0.00877, -0.01290), (+0.05196, +0.00892, +0.01310)),
    'left_hand_middle_0_link':  ((-0.00700, -0.01465, -0.01280), (+0.05480, +0.01260, +0.01290)),
    'left_hand_middle_1_link':  ((-0.00700, -0.00877, -0.01290), (+0.05196, +0.00892, +0.01310)),
    'left_wrist_roll_link':     ((-0.00000, -0.03049, -0.03000), (+0.05599, +0.03000, +0.03000)),
    'left_wrist_pitch_link':    ((-0.02850, -0.02849, -0.02849), (+0.07450, +0.02849, +0.02849)),
    'left_wrist_yaw_link':      ((-0.01799, -0.02386, -0.03049), (+0.04150, +0.03000, +0.03000)),
}

_ZFLOOR_CLEARANCE_M = 0.03     # the hand keeps 3 cm above the latched table plane
_ZFLOOR_STEP = 32              # coarse samples along the interpolation
_ZFLOOR_ITERS = 20             # bisection iterations

# kill switch: without it the clamp cannot be disabled without editing, and a
# clamp slightly too high fails quietly, leaving grasps short.
# PICK_ZFLOOR=0 in the engine environment disables it.
_ZFLOOR_ENABLED = os.environ.get('PICK_ZFLOOR', '1').strip().lower() not in (
    '0', 'off', 'false', 'no')

_TABLE_FLOOR_PELVIS = None     # latched floor, pelvis frame; None = no table
_TABLE_FLOOR_SRC = None        # (tz, height_m, orientation), for the log
_LIMB_FRAME_CACHE = None       # (id(model), [(frame_id, bbox), ...])

# per-arm z-floor: each arm gets its own scale over its 7 joints, so the idle
# arm cannot throttle the active one. switches are read at call time;
# PICK_ZFLOOR_PERARM=0 restores the single-scalar clamp.
_ZF_RATCHET_TOL_M = 0.001      # below-floor arm may only drop this much more
_ZF_LIMITED_S = 0.98           # below this = floor-limited (reply token)
_ZF_FLOW = [False]             # True inside pick / pick_left / place / place_left
_ZF_LAST_SENT = [None]         # post-clamp 14-vector from the last move (pin source)
_ZF_IDLE_PIN = {'left': None, 'right': None}   # idle arm's commanded post-clamp slice
_ZF_CMD = {'moves': 0, 'limited': 0, 'min_s': {'left': 1.0, 'right': 1.0}, 'min_any': 1.0}
_ZF_ACTIVE = [None]            # command's own arm during a pick / place flow
_ZF_PRIOR = [None]             # (floor, src) latched at pick / place start
_ZF_MEAS_Q = [None]            # measured 14-vector at pick / place start (idle precheck)
_ZF_TOKEN_VERBS = ('place', 'place_left', 'home', 'ready', 'ready_right', 'ready_left',
                   'carry', 'carry_left', 'neutral', 'play', 'quit')   # verbs that carry the reply token


def _latch_table_floor(tz, height_m, orientation):
    """Latch the table-plane z-floor from a pick/place's own object.

    The detection is the object's CENTROID in the TORSO frame, and the engine
    commands ``tz = wire_z + GRASP_Z_OFFSET`` as if it were a PELVIS-frame
    value (v5.2 logs this 4.4 cm discrepancy and leaves the channel unchanged).
    So, walking from the commanded tz down to the table surface in the pelvis
    frame::

        centroid_torso  = tz - GRASP_Z_OFFSET
        centroid_pelvis = centroid_torso - SOLVER_Z_TORSO_OFFSET
        table_pelvis    = centroid_pelvis - h/2      (upright: centroid mid-height)

    Checked against run11's own log ("Pick target torso frame: x=-0.012
    y=0.302 z=0.056 orientation=upright height_m=0.164", from the wire command
    ``pick -0.012 0.302 0.016 upright 0.1641``): tz=0.056 -> table -0.110 m.

    NOTE the sign of the h/2 term.  A larger object puts the table FURTHER
    below its centroid, so h/2 SUBTRACTS.  (An earlier draft of this design
    added it, which put the floor 20 cm above the table — above the commanded
    grasp point itself, clamping away every grasp.)

    Horizontal: the object lies on its side, so h is its LENGTH, not the
    height of its axis above the table, and cannot be used.  Fall back to the
    centroid with no h term — a lying object's axis sits one radius above the
    table, so this is conservative by that radius (~3 cm on a bottle).
    """
    global _TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC
    if tz is None:
        return None
    h = None
    try:
        if height_m is not None and 0.0 < float(height_m) <= HEIGHT_M_MAX:
            h = float(height_m)
    except (TypeError, ValueError):
        h = None
    try:
        tz = float(tz)
    except (TypeError, ValueError):
        return None
    floor = tz - GRASP_Z_OFFSET - SOLVER_Z_TORSO_OFFSET + _ZFLOOR_CLEARANCE_M
    if orientation != 'horizontal' and h is not None:
        floor -= h / 2.0
    _TABLE_FLOOR_PELVIS = floor
    _TABLE_FLOOR_SRC = (tz, h, orientation)
    sys.stderr.write(
        f'[pick_daemon][zfloor] latched table floor z={floor:+.4f} (pelvis) '
        f'from tz={tz:+.4f} h={h} orientation={orientation} '
        f'clearance={_ZFLOOR_CLEARANCE_M:.3f}\n')
    sys.stderr.flush()
    return floor


def _clear_table_floor(reason=''):
    """Drop the latched floor — the robot is going somewhere else."""
    global _TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC
    if _TABLE_FLOOR_PELVIS is not None:
        sys.stderr.write(f'[pick_daemon][zfloor] cleared ({reason})\n')
        sys.stderr.flush()
    _TABLE_FLOOR_PELVIS = None
    _TABLE_FLOOR_SRC = None


def _limb_frames(model, side=None):
    """[(frame_id, bbox), ...] for the limb links present in ``model``.

    ``side`` ('left' / 'right') keeps only that arm's links (B8g per-arm clamp).
    None = both arms: the cached list the single-scalar clamp has always used.
    """
    global _LIMB_FRAME_CACHE
    if side is None and _LIMB_FRAME_CACHE is not None and _LIMB_FRAME_CACHE[0] == id(model):
        return _LIMB_FRAME_CACHE[1]
    ids = []
    for name, bbox in _LIMB_BBOX.items():
        if side is not None and not name.startswith(side + '_'):
            continue
        try:
            if model.existFrame(name):
                ids.append((model.getFrameId(name), bbox))
        except Exception:
            pass
    if side is None:
        _LIMB_FRAME_CACHE = (id(model), ids)
    return ids


def _lowest_limb_z(arm_ik, q14, side=None):
    """Lowest point of BOTH arms' hand+wrist meshes, in the PELVIS frame.

    FK of the real URDF links, so the wrist orientation enters correctly (the
    lowest limb is middle_1 at NEUTRAL, thumb_2 at READY, palm at CARRY_TUCK —
    no single fixed point or constant offset is right for all poses).  Uses
    ``oMi[parentJoint] * frame.placement``, the same construction get_ee_pose
    uses, so no ``updateFramePlacements`` is needed.

    z is linear in the corner coordinates, so over the 8 bbox corners the
    minimum sits at the corner that takes min or max per axis — 3 sign tests
    per link instead of 8 transformed corners.  Returns None on failure.

    B8g: ``side`` ('left' / 'right') takes only that arm's links, so one arm's
    value never depends on the other arm's joints (None = both arms, as before).
    """
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    pin.forwardKinematics(model, data, q14)
    lo = None
    for fid, (mn, mx) in _limb_frames(model, side):
        fr = model.frames[fid]
        pl = data.oMi[fr.parentJoint] * fr.placement
        r2 = pl.rotation[2]
        z = pl.translation[2]
        for j in range(3):
            z += r2[j] * (mn[j] if r2[j] > 0.0 else mx[j])
        if lo is None or z < lo:
            lo = z
    return lo


def _zfloor_target(arm_ik, start_q, target_q, floor=None, ratchet=False):
    """Clamp ``target_q`` so the whole commanded move stays above the floor.

    Both transports traverse the SAME joint-space segment: the writer's slew
    (skills/writer/arm_guard.py: `s = 0.5*(1 - cos(pi*r))`) and this engine's
    standalone loop (`r = 0.5 - 0.5*cos(pi*i/steps)`, q = start + r*(t-start))
    are the same law.  Cosine vs linear is a time reparameterization, NOT a
    different path, so clamping one target vector covers both.

    Returns ``(q_out, clamped, info)``.  ``q_out is target_q`` (the same
    object) whenever nothing was clamped, so safe moves are bit-identical.

    B8g: by default each arm is clamped on its OWN 7-joint slice
    (_zfloor_target_perarm).  ``ratchet=True`` (the pick / place flows) lets an
    arm that starts below the floor move at most _ZF_RATCHET_TOL_M lower;
    otherwise such an arm is skipped alone.  PICK_ZFLOOR_PERARM=0 selects the
    pre-B8g single-scalar body (_zfloor_target_single), which ignores ``ratchet``.
    """
    if not _ZFLOOR_ENABLED:
        return target_q, False, None          # kill switch
    if floor is None:
        floor = _TABLE_FLOOR_PELVIS
    if floor is None:
        return target_q, False, None          # fail-open: no table latched yet
    if arm_ik is None or not hasattr(arm_ik, 'reduced_robot'):
        return target_q, False, None
    if _zfloor_perarm_on():
        return _zfloor_target_perarm(arm_ik, start_q, target_q, floor, ratchet)
    return _zfloor_target_single(arm_ik, start_q, target_q, floor)


def _zfloor_target_single(arm_ik, start_q, target_q, floor):
    """The pre-B8g single-scalar clamp (PICK_ZFLOOR_PERARM=0), unchanged."""
    try:
        start_q = np.asarray(start_q, dtype=np.float64)
        target_q = np.asarray(target_q, dtype=np.float64)
        z_start = _lowest_limb_z(arm_ik, start_q)
        if z_start is None:
            return target_q, False, None
        if z_start < floor:
            # already below the floor before this move; clamping would trap the
            # arm in the table and block the move that gets it out. fail open,
            # loudly.
            sys.stderr.write(
                f'[pick_daemon][zfloor] START already below the floor '
                f'({z_start:+.4f} < {floor:+.4f}) — clamp SKIPPED, move allowed '
                f'(recovery/escape)\n')
            sys.stderr.flush()
            return target_q, False, ('start_below', z_start)
        worst = z_start
        k = None
        for i in range(1, _ZFLOOR_STEP + 1):
            s = i / float(_ZFLOOR_STEP)
            z = _lowest_limb_z(arm_ik, start_q + s * (target_q - start_q))
            if z < worst:
                worst = z
            if z < floor and k is None:
                k = i
        if k is None:
            return target_q, False, ('clear', worst)
        lo_s, hi_s = (k - 1) / float(_ZFLOOR_STEP), k / float(_ZFLOOR_STEP)
        for _ in range(_ZFLOOR_ITERS):
            mid = 0.5 * (lo_s + hi_s)
            if _lowest_limb_z(arm_ik, start_q + mid * (target_q - start_q)) >= floor:
                lo_s = mid
            else:
                hi_s = mid
        return (start_q + lo_s * (target_q - start_q)), True, (lo_s, worst)
    except Exception as _ze:
        sys.stderr.write(f'[pick_daemon][zfloor] clamp skipped ({_ze!r})\n')
        sys.stderr.flush()
        return target_q, False, None


def _zfloor_perarm_on():
    """B8g kill switch PICK_ZFLOOR_PERARM, read at CALL time (default on).
    '0' / 'off' / 'false' / 'no' selects the single-scalar clamp, byte-for-byte."""
    return os.environ.get('PICK_ZFLOOR_PERARM', '1').strip().lower() not in (
        '0', 'off', 'false', 'no')


def _zf_wrist_short_m(arm_ik, side, target_q, out_q):
    """B8g E4 logging: how far (m) one arm's hand point moves when the clamp
    shortens its move.  Computed OUTSIDE the clamp: a failure returns nan and
    can never make the clamp fail open."""
    try:
        model = arm_ik.reduced_robot.model
        data = arm_ik.reduced_robot.data
        fid = arm_ik.L_hand_id if side == 'left' else arm_ik.R_hand_id
        pts = []
        for q in (target_q, out_q):
            pin.forwardKinematics(model, data, q)
            fr = model.frames[fid]
            pts.append((data.oMi[fr.parentJoint] * fr.placement).translation.copy())
        return float(np.linalg.norm(pts[0] - pts[1]))
    except Exception:
        return float('nan')


def _zfloor_target_perarm(arm_ik, start_q, target_q, floor, ratchet):
    """B8g per-arm clamp.  Each arm is scanned over its OWN 7-joint slice with
    the same 32-step scan and 20-step bisection as the single-scalar clamp, so
    one arm's lowest limb never depends on the other arm's joints.

    An arm that starts below the floor: with ``ratchet`` (pick / place flow) it
    may move, but its lowest limb never goes more than _ZF_RATCHET_TOL_M below
    where it started; without it, that arm alone is skipped (the legacy escape),
    logged per arm.  Returns ``(q_out, clamped, info)``; ``info`` when clamped is
    (smallest applied scale, worst lowest-z, {side: {'s', 'lw', 'mode', 'short_m'}}).
    """
    try:
        start_q = np.asarray(start_q, dtype=np.float64)
        target_q = np.asarray(target_q, dtype=np.float64)
        z0 = {}
        for side in ('left', 'right'):
            # None means no limb frames for this arm; only that arm is skipped,
            # the other is still clamped.
            z0[side] = _lowest_limb_z(arm_ik, start_q, side)
        if z0['left'] is None and z0['right'] is None:
            return target_q, False, None
        out = None
        per = {}
        skipped = None
        worst_all = None
        for side, sl in (('left', slice(0, 7)), ('right', slice(7, 14))):
            zs = z0[side]
            if zs is None:
                continue                      # no limb frames: skip this arm
            if not np.any(target_q[sl] != start_q[sl]):
                continue                      # this arm does not move: nothing to clamp or log
            eff = floor
            mode = 'clamp'
            if zs < floor:
                if not ratchet:
                    sys.stderr.write(
                        f'[pick_daemon][zfloor] START already below the floor '
                        f'({zs:+.4f} < {floor:+.4f}) arm={side} - clamp SKIPPED for '
                        f'this arm, move allowed (recovery/escape)\n')
                    sys.stderr.flush()
                    skipped = zs if skipped is None else min(skipped, zs)
                    continue
                eff = zs - _ZF_RATCHET_TOL_M
                mode = 'ratchet'
            worst = zs
            k = None
            for i in range(1, _ZFLOOR_STEP + 1):
                s = i / float(_ZFLOOR_STEP)
                z = _lowest_limb_z(arm_ik, start_q + s * (target_q - start_q), side)
                if z < worst:
                    worst = z
                if z < eff and k is None:
                    k = i
            worst_all = worst if worst_all is None else min(worst_all, worst)
            if k is None:
                continue                      # this arm stays above the floor on its own segment
            lo_s, hi_s = (k - 1) / float(_ZFLOOR_STEP), k / float(_ZFLOOR_STEP)
            for _ in range(_ZFLOOR_ITERS):
                mid = 0.5 * (lo_s + hi_s)
                if _lowest_limb_z(arm_ik, start_q + mid * (target_q - start_q), side) >= eff:
                    lo_s = mid
                else:
                    hi_s = mid
            if out is None:
                out = target_q.copy()
            out[sl] = start_q[sl] + lo_s * (target_q[sl] - start_q[sl])
            per[side] = {'s': lo_s, 'lw': worst, 'mode': mode,
                         'short_m': _zf_wrist_short_m(arm_ik, side, target_q, out)}
        if out is None:
            if skipped is not None:
                return target_q, False, ('start_below', skipped)
            present = [v for v in z0.values() if v is not None]
            return target_q, False, ('clear', worst_all if worst_all is not None
                                     else min(present))
        s_min = min(d['s'] for d in per.values())
        lw_min = min(d['lw'] for d in per.values())
        return out, True, (s_min, lw_min, per)
    except Exception as _ze:
        sys.stderr.write(f'[pick_daemon][zfloor] clamp skipped ({_ze!r})\n')
        sys.stderr.flush()
        return target_q, False, None


def _apply_zfloor(start_q, target_q, where, zf_label=None, zf_legacy=False):
    """Clamp + log.  Returns (q_out, was_clamped).

    B8g: ``zf_label`` names the waypoint in the per-arm log (``where`` still names
    the cancel phase).  ``zf_legacy`` keeps closeouts, retracts and the safe exit
    on the legacy skip even inside a pick / place flow.
    """
    q_out, clamped, info = _zfloor_target(
        _ARM_IK, start_q, target_q, ratchet=bool(_ZF_FLOW[0]) and not zf_legacy)
    if not _zfloor_perarm_on():
        if clamped:
            sys.stderr.write(
                f'[pick_daemon][zfloor] CLAMPED {where}: floor={_TABLE_FLOOR_PELVIS:+.4f} '
                f'lowest-would-be={info[1]:+.4f} -> s={info[0]:.3f} of the move\n')
            sys.stderr.flush()
        return q_out, clamped
    _zf_account(start_q, target_q, where, zf_label, clamped, info, zf_legacy)
    return q_out, clamped


def _zf_account(start_q, target_q, where, zf_label, clamped, info, zf_legacy=False):
    """B8g: one CLAMPED line per clamped arm, and the per-command tallies behind
    the E4 summary and the E5 reply token.  Logging and bookkeeping only: it never
    raises and never changes the clamp's output.

    E5 scope: inside a pick / place flow only the command's OWN arm counts, and
    its post-drop closeouts (zf_legacy) do not; outside a flow (the pose verbs)
    every arm the move changes counts.  Idle-arm moves are logged, never counted.
    """
    try:
        st = np.asarray(start_q, dtype=np.float64)
        tq = np.asarray(target_q, dtype=np.float64)
        per = info[2] if (clamped and info is not None and len(info) > 2) else {}
        s_arm = {'left': 1.0, 'right': 1.0}
        for side in ('left', 'right'):
            d = per.get(side)
            if d is not None and d['s'] < 1.0:
                s_arm[side] = d['s']
                sys.stderr.write(
                    f'[pick_daemon][zfloor] CLAMPED {where}: floor={_TABLE_FLOOR_PELVIS:+.4f} '
                    f'lowest-would-be={d["lw"]:+.4f} -> s={d["s"]:.3f} of the move '
                    f'arm={side} label={zf_label or where} '
                    f'wrist_short={100.0 * d["short_m"]:.1f}cm\n')
                sys.stderr.flush()
        moving = [side for side, sl in (('left', slice(0, 7)), ('right', slice(7, 14)))
                  if np.any(tq[sl] != st[sl])]
        if _ZF_FLOW[0]:
            # closeouts and retracts (zf_legacy) never count toward the token.
            # NEUTRAL does not fit at a table, so counting it would mark every
            # table place as not at the drop point; for place the token covers
            # moves up to and including the release.
            counted = [] if zf_legacy else [side for side in moving if side == _ZF_ACTIVE[0]]
        else:
            counted = moving
        if moving:
            _ZF_CMD['moves'] += 1      # a move that changes some arm ("before any arm motion")
        for side in moving:
            if s_arm[side] < _ZF_CMD['min_s'][side]:
                _ZF_CMD['min_s'][side] = s_arm[side]
        if moving and min(s_arm[side] for side in moving) < _ZF_LIMITED_S:
            _ZF_CMD['limited'] += 1
        if counted:
            s_any = min(s_arm[side] for side in counted)
            if s_any < _ZF_CMD['min_any']:
                _ZF_CMD['min_any'] = s_any
    except Exception as _ae:
        sys.stderr.write(f'[pick_daemon][zfloor] accounting skipped ({_ae!r})\n')
        sys.stderr.flush()


def _zf_cmd_begin():
    """Top of every command: no flow context, no commanded-target record, and
    empty per-command tallies."""
    _ZF_FLOW[0] = False
    _ZF_ACTIVE[0] = None
    _ZF_PRIOR[0] = None
    _ZF_MEAS_Q[0] = None
    _ZF_LAST_SENT[0] = None
    _ZF_CMD['moves'] = 0
    _ZF_CMD['limited'] = 0
    _ZF_CMD['min_s'] = {'left': 1.0, 'right': 1.0}
    _ZF_CMD['min_any'] = 1.0


def _zf_cmd_end(cmd):
    """E4: one summary line per command (per-arm clamp on), and the E5 LIMITED
    line when the command's own arm was floor-limited.  Runs in the command
    loop's finally; never raises."""
    try:
        if not _zfloor_perarm_on():
            return
        words = cmd.split()
        verb = words[0] if words else ''
        fl = ('%+.4f' % _TABLE_FLOOR_PELVIS) if _TABLE_FLOOR_PELVIS is not None else 'none'
        sys.stderr.write(
            f'[pick_daemon][zfloor] summary {verb}: floor={fl} '
            f'min_s_right={_ZF_CMD["min_s"]["right"]:.3f} '
            f'min_s_left={_ZF_CMD["min_s"]["left"]:.3f} '
            f'limited_moves={_ZF_CMD["limited"]}\n')
        _s_lim = _zf_token_s()          # same value the reply token carries
        if _s_lim is not None:
            sys.stderr.write(
                f'[pick_daemon][zfloor] LIMITED {verb}: s={_s_lim:.2f}\n')
        sys.stderr.flush()
    except Exception:
        pass


def _zf_token_s():
    """E5: the applied scale printed for a floor-limited command, or None when no
    move of the command was limited.  A limited move never prints 0.98: the verifier
    reads s < 0.98 as limited (brain/verifier.py _zfloor_limited_note), so 0.975-0.979
    is printed as 0.97.  The reply token and the LIMITED log line share this value."""
    if _ZF_CMD['min_any'] >= _ZF_LIMITED_S:
        return None
    return min(round(float(_ZF_CMD['min_any']), 2), round(_ZF_LIMITED_S - 0.01, 2))


def _zf_reply_suffix(verb):
    """E5: the `; zfloor_limited=<s>` token for this command's ONE stdout reply
    line, or '' unless the command's own moves were floor-limited (applied scale
    < 0.98).  Only the place / place_left and pose verbs carry it: a pick is
    reported on stderr alone.  Never raises."""
    try:
        if not _zfloor_perarm_on() or verb not in _ZF_TOKEN_VERBS:
            return ''
        s_tok = _zf_token_s()
        if s_tok is None:
            return ''
        return '; zfloor_limited=%.2f' % s_tok
    except Exception:
        return ''


def _zf_pick_begin(active, arm=None):
    """First statement of a pick / place handler: this command runs in the flow
    (ratchet for in-flow moves), ``active`` is its own arm, any idle-arm pin of an
    earlier command is void, and the floor it found is kept for E6."""
    _ZF_FLOW[0] = True
    _ZF_ACTIVE[0] = active
    _ZF_PRIOR[0] = (_TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC)
    _ZF_IDLE_PIN['left'] = None
    _ZF_IDLE_PIN['right'] = None
    # precheck: measured arm state at command start, before any motion; read
    # only when the per-arm clamp is on.
    _ZF_MEAS_Q[0] = None
    if arm is not None and _zfloor_perarm_on():
        try:
            _ZF_MEAS_Q[0] = np.array(arm.get_current_dual_arm_q(), dtype=np.float64).copy()
        except Exception:
            _ZF_MEAS_Q[0] = None


def _zf_clear_pins():
    """floor_clear: the idle-arm pin goes with the floor it was recorded under."""
    _ZF_IDLE_PIN['left'] = None
    _ZF_IDLE_PIN['right'] = None


def _zf_pin_idle(side, q=None):
    """E2: pin the idle arm to its COMMANDED post-clamp slice: ``q`` when given,
    else the 14-vector the last move commanded.  Per-arm only; a no-op otherwise."""
    if not _zfloor_perarm_on():
        return
    src_q = _ZF_LAST_SENT[0] if q is None else q
    if src_q is None:
        return
    lo, hi = (0, 7) if side == 'left' else (7, 14)
    _ZF_IDLE_PIN[side] = np.array(src_q[lo:hi], dtype=np.float64).copy()


def _zf_idle_target(side, tuck):
    """E2: the idle arm's 7-joint target: the pin when one is set (per-arm on),
    else the CARRY_TUCK pose as before."""
    pinned = _ZF_IDLE_PIN.get(side)
    if pinned is not None and _zfloor_perarm_on():
        return pinned
    return tuck


def _zf_idle_pre(side, tuck, start_q=None):
    """E2, the prechecks: they run BEFORE the command moves its idle arm, so no pin
    is set yet.  Returns the idle 7-slice the approach will run with: the pin when
    set; else where the coming tuck move leaves the idle arm, i.e. the per-arm clamp
    of ``start_q`` -> ``start_q`` with the idle slice at ``tuck`` under the latched
    floor (pick: T5 step 1 puts both arms at READY, so start_q = READY_POSE; place:
    step 2 parks from the measured pose).  Nothing clamps (no floor, or a clear
    path) -> exactly ``tuck``, the pre-B8g answer.  start_q None = the measured q at
    the start of the command.  Switch off, no start, or any failure: ``tuck``."""
    if not _zfloor_perarm_on():
        return tuck
    pinned = _ZF_IDLE_PIN.get(side)
    if pinned is not None:
        return pinned
    start = start_q if start_q is not None else _ZF_MEAS_Q[0]
    if start is None:
        return tuck
    try:
        lo, hi = (0, 7) if side == 'left' else (7, 14)
        s0 = np.array(start, dtype=np.float64).copy()
        tgt = s0.copy()
        tgt[lo:hi] = tuck
        q_out = _zfloor_target(_ARM_IK, s0, tgt, ratchet=True)[0]
        return np.array(q_out[lo:hi], dtype=np.float64).copy()
    except Exception:
        return tuck


def _zf_restore_floor():
    """E6: a pick / place refused BEFORE any arm motion gives back the floor it
    found when the command began.  No-op when nothing changed."""
    global _TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC
    prior = _ZF_PRIOR[0]
    if prior is None or not _zfloor_perarm_on():
        return
    if prior == (_TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC):
        return
    _TABLE_FLOOR_PELVIS, _TABLE_FLOOR_SRC = prior
    sys.stderr.write(
        f'[pick_daemon][zfloor] refused before arm motion: floor restored to '
        f'{_TABLE_FLOOR_PELVIS!r}\n')
    sys.stderr.flush()


def move_arm_smooth(arm, target_q, target_tauff, duration=None, where='arm_move',
                    zf_label=None, zf_legacy=False):
    """Smooth move of the 14-DOF arm target.

    ``where`` names the PHASE for the cancel reply (see ``_check_cancel``).
    Cancellable in both transports: the writer branch through
    ``WriterArmClient.slew(abort=...)`` (20 ms poll inside the slew wait) and
    the standalone branch through the per-step probe below.

    The table z-floor clamp is applied HERE, on the target, before the
    transport branch — both transports walk the same joint-space segment, so
    one clamped target covers both (see _zfloor_target).
    """
    start_q = arm.get_current_dual_arm_q().copy()
    target_q, _zf_clamped = _apply_zfloor(start_q, target_q, where,
                                          zf_label=zf_label, zf_legacy=zf_legacy)
    _ZF_LAST_SENT[0] = np.array(target_q, dtype=np.float64).copy()   # pin source
    if _zf_clamped:
        # gravity FF belongs to the pose actually commanded, not the one asked for.
        target_tauff = _arm_gravity_ff(target_q, target_tauff)
    if ARM_WRITER_ENABLED:
        # writer-slew move: raw 14-DOF target over the socket. the writer's
        # ArmGuard cosine slew (0.3 rad/s, min 1.5 s) replaces the 50 Hz
        # interpolation loop, and the explicit duration keeps both sides on one
        # clock. target_tauff (pin.rnea at the target) rides as the writer's arm
        # gravity FF, so the guarded position-error monitor sees servo error, not
        # gravity droop.
        max_delta = float(np.max(np.abs(target_q - start_q)))
        if duration is None:
            duration = max(1.5, max_delta / MOVE_SPEED)
        sys.stderr.write(
            f'[pick_daemon][writer] move_arm_smooth: max_delta={max_delta:.3f} '
            f'rad, duration={duration:.1f}s (writer slew, tau FF)\n')
        sys.stderr.flush()
        arm.slew(target_q, duration=duration, tau=target_tauff,
                 abort=lambda: _check_cancel(where))
        return
    max_delta = float(np.max(np.abs(target_q - start_q)))
    if duration is None:
        duration = max(1.5, max_delta / MOVE_SPEED)
    steps = max(1, int(duration * MOVE_HZ))
    dt = duration / steps

    sys.stderr.write(
        f'[pick_daemon] move_arm_smooth: max_delta={max_delta:.3f} rad, '
        f'duration={duration:.1f}s, steps={steps}\n')
    sys.stderr.flush()

    for i in range(1, steps + 1):
        # cancel probe before the next command, so the arm never gets a forward
        # command after the cancel (same ordering as the contact check).
        _check_cancel(where)
        r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
        q_interp = start_q + r * (target_q - start_q)
        # constant FF (target_tauff) for every step: a per-step recompute gave a
        # stiff, locked feel mid-motion.
        tau_step = target_tauff  # constant FF
        arm.ctrl_dual_arm(q_interp, tau_step)
        time.sleep(dt)

    arm.ctrl_dual_arm(target_q, target_tauff)


def move_arm_smooth_guarded(arm, target_q, target_tauff, monitor_indices,
                             pos_threshold, consecutive_threshold,
                             stop_fn=None, duration=None, where='guarded_move',
                             zf_label=None, zf_legacy=False):
    """Like move_arm_smooth but stops early on contact.

    ``where`` names the PHASE for the cancel reply (see ``_check_cancel``).
    The cancel probe is checked FIRST in each step, before the contact checks
    and before the next forward command — a cancel is strictly higher
    priority than continuing to press, and the phase it names is what the
    operator is told the arm stopped in.

    Two detection mechanisms (either triggers a stop):

    1. **Position error**: monitors |cmd_q - actual_q| on ``monitor_indices``.
       If cumulative error exceeds ``pos_threshold`` for
       ``consecutive_threshold`` consecutive steps → contact stop.

    2. **stop_fn callback** (optional): called at each step as
       ``stop_fn() → str or None``.  If it returns a non-empty string, the
       motion stops immediately with that string as the reason.  Use this
       for palm pressure sensors or other fast contact signals.

    **Contact is checked BEFORE sending each new command** (except step 1),
    so the arm never receives a forward command after contact is detected.

    On stop, the arm freezes at its current actual position.

    Returns (completed: bool, stop_reason: str or None).

    The table z-floor clamp applies here too.  On a well-formed descent it
    should be a NO-OP: the floor sits (GRASP_Z_OFFSET + SOLVER_Z_TORSO_OFFSET +
    h/2 + clearance) below the commanded grasp z, and the descent stops AT that
    grasp z.  It bites only if a target is itself commanded at or into the
    surface.

    THAT CLAIM IS NOT MEASURED.  The margin depends on the grasp orientation's
    fingertip drop, which varies with the object and the wrist roll — a fixed
    constant is exactly what this clamp exists to avoid assuming.  Watch for
    `[zfloor] CLAMPED ... guarded` in engine_grasp.log on the first table run;
    if it appears during a DESCENT, the floor is too high and grasps will come
    up short.
    """
    start_q = arm.get_current_dual_arm_q().copy()
    target_q, _zf_clamped = _apply_zfloor(start_q, target_q, where,
                                          zf_label=zf_label, zf_legacy=zf_legacy)
    _ZF_LAST_SENT[0] = np.array(target_q, dtype=np.float64).copy()   # pin source
    if _zf_clamped:
        target_tauff = _arm_gravity_ff(target_q, target_tauff)
    if ARM_WRITER_ENABLED:
        # writer-slew guarded move: start the slew and monitor daemon-side at
        # MOVE_HZ; the explicit duration time-aligns both sides. contact
        # semantics: stop_fn first, then |profile - actual| position error on
        # monitor_indices; on stop, arm.freeze(). actual_q is polled from the
        # writer over the socket.
        max_delta = float(np.max(np.abs(target_q - start_q)))
        if duration is None:
            duration = max(1.5, max_delta / MOVE_SPEED)
        steps = max(1, int(duration * MOVE_HZ))
        dt = duration / steps
        sys.stderr.write(
            f'[pick_daemon][writer] move_arm_smooth_guarded: '
            f'max_delta={max_delta:.3f} rad, duration={duration:.1f}s, '
            f'steps={steps}, monitoring joints={monitor_indices}\n')
        sys.stderr.flush()
        # target_tauff rides the socket as the writer gravity FF: the monitor
        # compares the daemon profile against physical q, so the arm must track
        # with servo-scale error. without FF the wrist and elbow droop past the
        # 0.15 rad sum threshold and the guard false-freezes mid-air.
        arm.slew_start(target_q, duration=duration, tau=target_tauff)
        consecutive_exceeded = 0
        prev_cmd_q = start_q.copy()
        for i in range(1, steps + 1):
            # cancel first: it outranks continuing to press and is checked
            # before the next forward command.
            _check_cancel(where)
            # contact check from the previous step before the next profile point,
            # so the arm never gets a forward command after contact.
            if i > 1:
                # stop_fn first: tactile is the fastest contact signal
                if stop_fn is not None:
                    reason = stop_fn()
                    if reason:
                        sys.stderr.write(
                            f'[pick_daemon][writer] PALM CONTACT (sensor): '
                            f'{reason} (step {i}/{steps}), freezing\n')
                        sys.stderr.flush()
                        arm.freeze()
                        return False, reason
                # position error: previous profile point against the actual q polled from the writer
                actual_q = arm.get_current_dual_arm_q()
                error = sum(abs(prev_cmd_q[j] - actual_q[j])
                            for j in monitor_indices)
                if error > pos_threshold:
                    consecutive_exceeded += 1
                    if consecutive_exceeded >= consecutive_threshold:
                        sys.stderr.write(
                            f'[pick_daemon][writer] PALM CONTACT (pos err): '
                            f'{error:.4f} rad exceeded threshold '
                            f'{pos_threshold:.4f} for {consecutive_exceeded} '
                            f'consecutive steps (step {i}/{steps}), '
                            f'freezing\n')
                        sys.stderr.flush()
                        arm.freeze()
                        return False, f'position_error:{error:.4f}rad'
                else:
                    consecutive_exceeded = 0
            # advance the daemon-side profile; the writer tracks its own cosine
            # slew of the same duration, time-aligned by construction.
            r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            prev_cmd_q = start_q + r * (target_q - start_q)
            time.sleep(dt)
        return True, None
    max_delta = float(np.max(np.abs(target_q - start_q)))
    if duration is None:
        duration = max(1.5, max_delta / MOVE_SPEED)
    steps = max(1, int(duration * MOVE_HZ))
    dt = duration / steps

    sys.stderr.write(
        f'[pick_daemon] move_arm_smooth_guarded: max_delta={max_delta:.3f} rad, '
        f'duration={duration:.1f}s, steps={steps}, '
        f'monitoring joints={monitor_indices}\n')
    sys.stderr.flush()

    consecutive_exceeded = 0
    prev_cmd_q = start_q.copy()
    prev_actual_q = start_q.copy()

    for i in range(1, steps + 1):
        # ── cancel first (see the writer branch above).
        _check_cancel(where)
        # check contact from the previous step before sending the next command,
        # so no forward command goes out after contact.
        if i > 1:
            # stop_fn first: tactile is the fastest contact signal
            if stop_fn is not None:
                reason = stop_fn()
                if reason:
                    sys.stderr.write(
                        f'[pick_daemon] PALM CONTACT (sensor): {reason} '
                        f'(step {i}/{steps}, before sending cmd), freezing\n')
                    sys.stderr.flush()
                    # gravity FF at the frozen actual pose; endpoint torque at an
                    # intermediate pose gives a stiff hold.
                    arm.ctrl_dual_arm(
                        prev_actual_q, _arm_gravity_ff(prev_actual_q,
                                                       target_tauff))
                    return False, reason

            # position error: previous commanded against the current actual
            actual_q = arm.get_current_dual_arm_q()
            error = sum(abs(prev_cmd_q[j] - actual_q[j]) for j in monitor_indices)

            if error > pos_threshold:
                consecutive_exceeded += 1
                if consecutive_exceeded >= consecutive_threshold:
                    sys.stderr.write(
                        f'[pick_daemon] PALM CONTACT (pos err): {error:.4f} rad '
                        f'exceeded threshold {pos_threshold:.4f} for '
                        f'{consecutive_exceeded} consecutive steps '
                        f'(step {i}/{steps}, before sending cmd), freezing\n')
                    sys.stderr.flush()
                    # gravity FF at the frozen actual pose; endpoint torque there
                    # gives a stiff hold.
                    
                    arm.ctrl_dual_arm(
                        actual_q, target_tauff)
                    return False, f'position_error:{error:.4f}rad'
            else:
                consecutive_exceeded = 0

            prev_actual_q = actual_q

        # ── send the next command, only if no contact is detected ──
        r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
        cmd_q = start_q + r * (target_q - start_q)
        prev_cmd_q = cmd_q
        # constant FF (target_tauff) for every step: a per-step recompute gave a
        # stiff, locked feel mid-motion.
        tau_step = target_tauff  # constant FF
        arm.ctrl_dual_arm(cmd_q, tau_step)
        time.sleep(dt)

    arm.ctrl_dual_arm(target_q, target_tauff)
    return True, None


def _make_pressure_insert_stop(client, side):
    """Item 2 (v3 RELATIVE): build the insert-guard pressure check.

    v4.0 (2026-08-30): RETIRED — no call sites remain (the upright flow's
    pressure-stop-first went in v3.4; the horizontal flows' went in v4.0;
    every guarded move uses the ORIGINAL middle-finger + position-err
    backstop).  Kept as history.
    Polls grip_server `pressure [side]` once per guarded-move loop iteration
    (same socket verb pattern as GripClient.state — see grip_client.py
    pressure()).  The baseline is captured lazily on the FIRST successful
    read — the guard is built right before the guarded move with the hand
    open in the air (the 2026-08-29 freezes triggered at loop iteration 2,
    still in the approach pose), so the first read IS the open-hand floor.
    Fires when pmax - baseline > DELTA_PRESS.
    ok:false (no state data) → skip the pressure check and fall through to
    the position-err backstop; logged once per guarded insert, not per loop
    step.  Never raises.
    """
    _warned = {'flag': False}
    _baseline = {'p': None}

    def _check():
        try:
            payload = client.pressure(side)
        except Exception:
            payload = None
        if payload is None or not payload.get(side, {}).get('ok'):
            if not _warned['flag']:
                _warned['flag'] = True
                sys.stderr.write(
                    f'[pick_daemon] pressure insert-guard: no pressure data '
                    f'({side}) — position-err backstop only\n')
                sys.stderr.flush()
            return None
        pmax = float(payload[side].get('max', 0.0))
        if _baseline['p'] is None:
            _baseline['p'] = pmax
            sys.stderr.write(
                f'[pick_daemon] pressure insert-guard baseline ({side}): '
                f'{pmax:.0f} (DELTA_PRESS={DELTA_PRESS:.0f})\n')
            sys.stderr.flush()
        delta = pmax - _baseline['p']
        if delta > DELTA_PRESS:
            sys.stderr.write(
                f'[pick_daemon] palm/finger pressure trigger '
                f'(pmax={pmax:.0f}, baseline={_baseline["p"]:.0f}, '
                f'delta=+{delta:.0f}) — stopping insert\n')
            sys.stderr.flush()
            return f'pressure:{pmax:.0f}'
        return None

    return _check


# ── verify-open tolerance ─────────────────────────────────────────────────────
# the tolerance must stay below the smallest open-to-close span, or a
# still-closed joint reads as open. the smallest span is thumb_0 at 0.480 rad;
# 0.2 tolerates normal tracking lag while still catching a fist.
VERIFY_OPEN_TOLERANCE_RAD = 0.2


# release-open evidence for the gateway's place verdict. the place verifier
# wants the post-drop hand measured open, but with PLACE_PARK_HAND_CLOSED on,
# the LIFT closeout closes the holding hand before DONE, so the post-DONE read
# correctly sees closed and place could never verify. the release is measured
# separately: step 6 _verify_hand_open compares every finger joint against the
# OPEN preset in measured motor-q space, and the DONE line carries
#     DONE; release_open=<side>:<True|False>
# the verifier takes the token for the release half, measured at the release
# rather than after the closeout re-closed. the park is unchanged. the token
# cannot be a second stdout line since the transport is one line in, one out.
# absent means unmeasured (a refusal that never reached step 6), never a pass.
_RELEASE_OPEN = {}


def _note_release_open(side, ok):
    """Record the measured-open result of a place release for ``side``."""
    if side in ("right", "left"):
        _RELEASE_OPEN[side] = bool(ok)


def _release_open_token(side):
    """'; release_open=<side>:<True|False>' for the DONE line, or ''.

    Empty when this process never measured a release for ``side`` — the
    gateway/verifier then treat the release as UNMEASURED and fall back to
    the post-state read (the pre-2026-09-27 behaviour, byte-identical).
    """
    if side not in _RELEASE_OPEN:
        return ''
    return '; release_open=%s:%s' % (side, _RELEASE_OPEN[side])


def _verify_hand_open(client, side, open_preset):
    """Item 5: verify the hand actually opened before a wp4 insert.

    Queries the grip_server `state [side]` reply's ``actual`` (live motor q
    per joint — the same source grip_server's _seed_side uses) and compares
    every finger joint against the OPEN preset within
    VERIFY_OPEN_TOLERANCE_RAD.  One retry (re-open + settle) on unknown /
    no-state or any finger far from open; then reports failure.

    Returns (ok: bool, reason: str); reason is meaningful only when not ok.
    Never raises: any transport error counts as 'no actual state'.
    """
    for _round in (0, 1):
        try:
            payload = client.state(side)
        except Exception:
            payload = None
        actual = payload.get(side, {}).get('actual') if payload else None
        if actual:
            for jid in DEX3_JOINTS:
                q = float(actual.get(str(jid), 0.0))
                if abs(q - open_preset[jid]) > VERIFY_OPEN_TOLERANCE_RAD:
                    if _round == 0:
                        sys.stderr.write(
                            f'[pick_daemon] hand {side} not fully open '
                            f'(finger {jid} q={q:.2f} rad, open preset '
                            f'{open_preset[jid]:.2f}) — re-opening\n')
                        sys.stderr.flush()
                    else:
                        return False, (f'finger {jid} at {q:.2f} rad '
                                       f'(open preset {open_preset[jid]:.2f})')
                    break
            else:
                return True, None
        elif _round == 0:
            sys.stderr.write(
                f'[pick_daemon] hand {side} actual state unavailable — '
                're-opening\n')
            sys.stderr.flush()
        if _round == 0:
            try:
                client.open()
            except Exception:
                pass
            time.sleep(0.3)
    return False, 'no actual state'


def _read_hand_actual(client, side):
    """v5.2 (2026-09-07) Edit E: side-unwrapped live motor q from the
    grip_server `state` reply.

    The reply is SIDE-KEYED — {side: {joint: q}} — the `actual` mapping
    lives under the side key, NOT at the top level.  (The v5.1 blocked-read
    did `_cs.get('actual', {})` and read NOTHING — so `_close_blocked` never
    fired; see _score_closure_blocked.)  Mirrors the unwrap _verify_hand_open
    uses: payload.get(side, {}).get('actual').

    Never raises: transport error → {}.
    """
    try:
        payload = client.state(side)
    except Exception:
        return {}
    if not payload:
        return {}
    return payload.get(side, {}).get('actual') or {}


# ── thumb-first staged close ────────────────────────────────────────────────
# a full close is one cosine ramp over all seven joints, so thumb_0 travels
# R_HAND_OPEN[0]=+0.48 to R_HAND_CLOSE[0]=-0.0268 (0.507 rad base rotation)
# while joints 3..6 curl to 1.571/1.745. the thumb sweeps diagonally through
# where index and middle curl, and nothing stages them, so the thumb can
# collide with the fingers.
# this is a path change only: no preset and no endpoint moves. the order is
# thumb base to its centreline in its own ramp first, every other joint held
# at the measured q, then the ordinary full close. by stage 2 thumb_0 has
# ~0.027 rad left.
# the other joints are held at live q because `set` commands absolute targets
# and both call sites can be part-closed; commanding OPEN would open a loaded
# hand before closing. a read failure falls back to OPEN, the pose both call
# sites are in by construction (the place park is guarded by release_ok).
# staging is skipped when the thumb base is within STAGED_CLOSE_SKIP_TOL of
# the centreline, since there is no diagonal to remove.
THUMB_BASE_JID = 0
THUMB_BASE_CENTRE = 0.0
STAGED_CLOSE_SKIP_TOL = 0.10   # rad; base this close to centre = no staging


def _staged_thumb_first_close(hand, side, verb):
    """Stage 1 = thumb base to centre, stage 2 = the ordinary full close.

    BOTH STAGES LIVE HERE, deliberately: the operator's requirement is one
    motion in two phases ("the thumb should first return to neutural centre
    then close"), so a caller that staged and then forgot to close would leave
    the fingers OPEN -- the exact condition the operator's 2026-09-21 directive
    exists to prevent.  Keeping them in one function makes that unrepresentable
    rather than merely discouraged.

    Returns the stage-1 outcome as a short string, for the caller's log line:
    'staged' | 'already-centre:<q>' | 'no-state' | 'stage1-failed:<err>'.

    BEST-EFFORT ON STAGE 1 ONLY.  `hand.close()` is UNCONDITIONAL and is NOT
    inside the stage-1 try, so a staging fault can never skip the close.  A
    stage-2 exception PROPAGATES -- both call sites already wrap this and log
    'the hand stays open; continuing', and swallowing here would hide a real
    hand-controller fault from those handlers.
    """
    actual = _read_hand_actual(hand, side)
    open_preset = OPEN_PRESETS[side]

    # held pose: OPEN preset overwritten joint by joint from the live read.
    # `actual` is JSON, so its keys are strings.
    held = {jid: float(open_preset[jid]) for jid in DEX3_JOINTS}
    q_thumb = None
    for jid in DEX3_JOINTS:
        raw = actual.get(str(jid))
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        held[jid] = val
        if jid == THUMB_BASE_JID:
            q_thumb = val

    if q_thumb is None:
        # no live thumb reference: fall back to OPEN's base, the abducted start
        # this staging unwinds.
        q_thumb = float(open_preset[THUMB_BASE_JID])
        held[THUMB_BASE_JID] = q_thumb
        note = 'no-state'
    else:
        note = 'staged'

    if abs(q_thumb - THUMB_BASE_CENTRE) <= STAGED_CLOSE_SKIP_TOL:
        note = 'already-centre:%.3f' % q_thumb
    else:
        stage1 = dict(held)
        stage1[THUMB_BASE_JID] = float(THUMB_BASE_CENTRE)
        try:
            hand.set(stage1)
        except Exception as _se:                              # noqa: BLE001
            note = 'stage1-failed:%s' % _se

    # stage 2: ordinary full close, outside the try above; staging is
    # best-effort, the close is not.
    
    hand.close()
    return note


def _score_closure_blocked(client, side, preset=None):
    """v5.2 (2026-09-07) Edit E: score how far the CLOSED hand sits from the
    preset it was COMMANDED to — the physical wrap evidence (an air close
    reaches the preset exactly; a real wrap is BLOCKED short of it).

    margin = max over DEX3_JOINTS of |actual - preset[jid]|, NaN-skipped
    (no-state joints are not evidence).  blocked = margin > CLOSURE_BLOCKED_TOL.

    `preset` DEFAULTS to CLOSE_PRESETS[side], which is what every caller wants
    when the close was commanded to the full preset.  It must be passed
    EXPLICITLY whenever the close was capped (F_MAX_SAFE): comparing a capped
    close against the UNCAPPED preset makes the margin |q(0.600) - q(1.0)|,
    which for joint 2 is 0.40 rad >> CLOSURE_BLOCKED_TOL — every free-air close
    would then report a false capture.  Score against what you commanded.

    Returns (blocked: bool, margin: float, worst_jid: int|None).  Never
    raises; a no-data read returns (False, NaN, None).
    """
    actual = _read_hand_actual(client, side)
    if not actual:
        return False, float('nan'), None
    if preset is None:
        preset = CLOSE_PRESETS[side]
    margin = 0.0
    worst = None
    found = False
    for jid in DEX3_JOINTS:
        q = float(actual.get(str(jid), float('nan')))
        if q != q:  # skip NaN (no state data)
            continue
        found = True
        _m = abs(q - float(preset[jid]))
        if _m > margin:
            margin = _m
            worst = jid
    if not found:
        return False, float('nan'), None
    return margin > CLOSURE_BLOCKED_TOL, margin, worst


def _capped_close_preset(side):
    """The preset the F_MAX_SAFE-capped closure actually commands.

    EVERY post-close re-press and EVERY closure-block score must use this
    rather than CLOSE_PRESETS[side].  Two distinct failures otherwise:

      * re-pressing R_HAND_CLOSE/L_HAND_CLOSE transits the self-collision band
        at f=0.63..0.81 again — the exact defect the cap exists to remove;
      * scoring against the UNCAPPED preset reads |q(0.600) - q(1.0)| as the
        margin, which for joint 2 is 0.40 rad >> CLOSURE_BLOCKED_TOL, so every
        free-air close reports a FALSE capture.

    Returns a {jid: q} dict.
    """
    return clamped_close_target(side, 1.0)[0]


def _close_dir(side, jid):
    """+1.0 / -1.0: the sign q MOVES in when `jid` closes; 0.0 if pinned.

    Derived from the presets, exactly as the ramp is, so it cannot disagree
    with the ramp and cannot be hand-typed wrong per joint.  Verified against
    the DEPLOY build's dicts (the engine imports grip_server from
    /home/unitree/leon/openbot): right = {0,1,2: -1.0, 3,4,5,6: +1.0}, left =
    the exact mirror {0,1,2: +1.0, 3,4,5,6: -1.0}.  No Dex3 joint is pinned
    (OPEN == CLOSE) today, so this never returns 0.0 in practice; a future
    pinned joint falls out here rather than being handed invented travel.
    """
    delta = float(CLOSE_PRESETS[side][jid]) - float(OPEN_PRESETS[side][jid])
    if delta > 0.0:
        return 1.0
    if delta < 0.0:
        return -1.0
    return 0.0


def _bounded_lead_target(client, side, preset, eps=None):
    """The {jid: q} to command so the LEAD over the measured q is <= eps.

    THE FIX IS THIS FUNCTION.  tau_est ~= bias + 16.6 * lead, so bounding the
    lead IS bounding the torque, with no torque sensor and no model in the loop:

        target[jid] = clamp(q_meas[jid] + eps * closing_dir, toward preset)

    eps PAST where the joint actually is, and NEVER past the close preset.
    PER JOINT against a LIVE q_meas, which is required and not a refinement: the
    joints BLOCK AT DIFFERENT FRACTIONS of travel, so a preset-relative offset
    is not a bounded lead at all (the operator's one uniform proven press landed
    0.067-0.276 rad of lead across the seven joints).

    The clamp toward the preset is what makes the walk FINITE — a joint already
    at the preset is re-commanded AT the preset — so this can never command
    travel beyond the aperture.  And because the reference is a FRESH q_meas,
    never the previous command, the lead cannot accumulate across presses: a
    finger stopped by the object is re-commanded to the SAME place every time.

    INT-KEYED ON PURPOSE: GripClient.set() does `float(targets.get(jid, 0.0))`
    over DEX3_JOINTS with an INT jid, so a str-keyed target would be sent as
    0.0 on all 7 joints — a full OPEN command mid-pick.

    A non-finite q_meas is REJECTED, not passed through: inf would satisfy
    `min(inf, preset) == preset` and produce exactly the full-preset command
    this function exists to remove, and NaN would be written to the MCU.

    Returns None when the hand state is unavailable or incomplete — it never
    invents a command.  Callers MUST treat None as "write nothing".
    """
    if eps is None:
        eps = HAND_CLOSE_LEAD_EPS
    q = _read_hand_actual(client, side)
    if not q:
        return None
    target = {}
    for jid in DEX3_JOINTS:
        try:
            qm = float(q[str(jid)])
        except (KeyError, TypeError, ValueError):
            return None                       # a joint with no state read: no bound
        if not math.isfinite(qm):
            return None
        try:
            ceil_j = float(preset[jid])
        except (KeyError, TypeError, ValueError):
            return None
        cdir = _close_dir(side, jid)
        if cdir == 0.0:
            # pinned joint: no closing direction, so there is no lead to bound
            # and it must not get invented travel. hold it where it is.
            target[jid] = qm
            continue
        want = qm + cdir * float(eps)
        if cdir > 0.0:
            want = min(want, ceil_j)
        else:
            want = max(want, ceil_j)
        target[jid] = float(want)
    return target


def _targets_within(a, b, tol):
    """True when two {jid: q} commands agree on every Dex3 joint within `tol`."""
    for jid in DEX3_JOINTS:
        try:
            if abs(float(a[jid]) - float(b[jid])) > tol:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _at_close_ceiling(target, side, preset, tol=None):
    """True when `target` IS the close preset on every joint that can move.

    A joint with no closing direction (pinned) has no travel to place, so it is
    not required to sit at the preset.
    """
    if tol is None:
        tol = HAND_CLOSE_CEILING_TOL
    for jid in DEX3_JOINTS:
        if _close_dir(side, jid) == 0.0:
            continue
        try:
            if abs(float(target[jid]) - float(preset[jid])) > tol:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _watch_hand(client, side, target, eps, budget_s=None, poll_s=None,
                where='hand_close'):
    """Pace ONE press: poll q_meas until the fingers arrive, or stop moving.

    Returns (arrived, moved, lost):
      arrived — every moving joint is within (1 - HAND_CLOSE_ARRIVE_FRAC)*eps of
                its target, so the next press would be the same command.
      moved   — the LARGEST per-joint displacement from the window's first read,
                so a press that produced no motion at all reads 0.0 (a stall =
                an obstruction) while a slow finger still reports its motion.
      lost    — the longest run of consecutive empty state reads at the end, so
                a mid-press DDS dropout is reported as STATE LOST rather than
                being mislabelled as "the fingers stopped" (the one log line an
                operator would use to tell a rail from a dropout).

    This is a PACING device only.  Safety comes from the lead bound, not from
    this function: ending the watch early can only make the walk press again
    sooner, never command a larger lead.
    """
    if budget_s is None:
        budget_s = HAND_CLOSE_WATCH_BUDGET_S
    if poll_s is None:
        poll_s = HAND_CLOSE_POLL_S
    q_last = _read_hand_actual(client, side)
    if not q_last:
        return False, 0.0, HAND_CLOSE_LOST_READS
    q0 = dict(q_last)
    moved = 0.0
    quiet = 0.0
    lost = 0
    deadline = time.time() + float(budget_s)
    while time.time() < deadline:
        # cancel probe: the loop can run for seconds while the fingers press.
        _check_cancel(where)
        time.sleep(poll_s)
        qn = _read_hand_actual(client, side)
        if not qn:
            lost += 1
            if lost >= HAND_CLOSE_LOST_READS:
                return False, moved, lost
            continue
        lost = 0
        step = 0.0
        for jid in DEX3_JOINTS:
            a = q_last.get(str(jid))
            b = qn.get(str(jid))
            m0 = q0.get(str(jid))
            if a is None or b is None or m0 is None:
                continue
            try:
                b = float(b)
                a = float(a)
                m0 = float(m0)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(b):
                continue
            step = max(step, abs(b - a))
            moved = max(moved, abs(b - m0))
        q_last = qn
        if step <= HAND_CLOSE_STILL_TOL:
            quiet += poll_s
            if quiet >= HAND_CLOSE_STILL_HOLD_S:
                break
        else:
            quiet = 0.0
        # arrival: even the farthest joint is within (1-ARRIVE_FRAC)*eps of
        # its target, so the next press would recompute the same command. stop
        # waiting.
        _worst = 0.0
        for jid in DEX3_JOINTS:
            cdir = _close_dir(side, jid)
            if cdir == 0.0:
                continue
            try:
                _worst = max(_worst, cdir * (float(target[jid]) - float(qn[str(jid)])))
            except (KeyError, TypeError, ValueError):
                continue
        if _worst <= (1.0 - HAND_CLOSE_ARRIVE_FRAC) * float(eps):
            return True, moved, lost
    return False, moved, lost


def _worst_closing_lead(client, side, target):
    """max over joints of cdir*(target - q_meas) — the HELD closing lead.

    Read live, so callers must only LOG it.  This is the quantity the measured
    torque law keys on: tau_est ~= bias + 16.6*lead.  Returns None with no
    state to read.
    """
    q = _read_hand_actual(client, side)
    if not q:
        return None
    worst = 0.0
    for jid in DEX3_JOINTS:
        try:
            qm = float(q[str(jid)])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(qm):
            continue
        cdir = _close_dir(side, jid)
        if cdir == 0.0:
            continue
        try:
            worst = max(worst, cdir * (float(target[jid]) - qm))
        except (KeyError, TypeError, ValueError):
            continue
    return worst


def _bounded_lead_close(client, side, preset, eps=None, max_presses=None,
                        dt=None, watch_budget_s=None, poll_s=None,
                        deadline_s=None):
    """Close the hand toward `preset` with the per-write LEAD bounded by eps.
    Drop-in for close_until_contact (:4114 / :5488) and for the re-press inside
    _squeeze_hold (:2219).

    WHY THIS REPLACES THE RAMP AND NOTHING SMALLER.  Per measured fact 5, the
    server's cosine ramp is BACK-LOADED and in M2 the fingers met the seated
    bottle at step 40 with the commanded lead ALREADY ~0.76 rad — ~92% of the
    rail — so every stop-threshold scheme fires at the same instant the rail
    begins.  And only the engine can bound this: the deploy build's grip_server
    has NO lead bound (its ramp writes `self._write(q)` unclamped), so the
    engine is the only place the lead can be capped.

    THE WALK, and the three ways it ends:
      (1) THE COMMAND REACHES THE PRESET on every moving joint AND the finger
          has caught up to it (or this press produced no motion at all).  The
          walk does NOT stop at the first ceiling press: with the command fixed
          AT the preset every further press re-commands the same pose, so the
          lead can only SHRINK, and waiting is free of safety cost.  This is
          what makes the C2 guarantee STRUCTURAL and independent of the CALLER's
          hold time — a free-air close hands off with the fingers ARRIVED, not
          merely commanded.  Note `_at_close_ceiling` is only true when EVERY
          moving joint is within eps of the preset, so the lead at that moment is
          <= eps by construction and can never be the rail.
      (2) THE COMMAND STOPS ADVANCING before the ceiling (this press's target
          equals the last written target) — the finger physically stopped
          short of the preset.  The hand is then held at a bounded lead of eps
          by the server's 20 Hz re-assert.
      (3) the press cap / the wall-clock deadline, logged LOUDLY.  Note the
          physical reading: under a sustained eps-lead the closing torque is
          ~5 N.m, which moves a FREE finger, so a finger that creeps under it is
          obstructed.  This corner is logged rather than hidden, and the
          closure-block margin — not this verdict — decides the keep.

    C3 INVARIANT, stated once: at EVERY commanded write,
        cdir_jid * (q_cmd[jid] - q_meas[jid]) <= eps
    enforced three independent ways — (a) _bounded_lead_target is the ONLY
    builder of a command and builds the lead by arithmetic; (b) steps=1 means
    the only write is the endpoint, so no intermediate ramp point can carry a
    stale last_q; (c) the reference is a fresh q_meas each press, never the
    previous command, so the lead cannot accumulate.  The walk is finite because
    the target is CLAMPED to the close preset on every press.

    VERDICT: 'full' | 'no_data'.  Deliberately NOT the old trichotomy.
    'contact' existed only for a pressure stop crossing GRIP_CLOSE_STOP_DELTA
    (200000, :517), which is INERT on this hardware (rest floor 104-110k, a hard
    pad press 223272), so a blocked close has always come back 'full' in normal
    use.  Returning 'contact' now would newly DISABLE the light-grip promotion
    at :4302 (which requires `_contact_closure == 'full'`) — trading a false
    success for a DROP — and the keep evidence was never this channel (:527-542)
    but the closure-block margin the CALLER scores.  So the verdict keeps its
    historical value and the LOG lines carry the truth instead.
    'no_data' means the hand state was unreadable and NOTHING WAS WRITTEN: a
    close whose lead cannot be observed IS the rail path, so the walk refuses
    rather than pressing blind (and says so at the call site).

    Never raises.
    """
    if eps is None:
        eps = HAND_CLOSE_LEAD_EPS
    if max_presses is None:
        max_presses = HAND_CLOSE_MAX_PRESSES
    if dt is None:
        dt = HAND_CLOSE_PRESS_DT
    if watch_budget_s is None:
        watch_budget_s = HAND_CLOSE_WATCH_BUDGET_S
    if poll_s is None:
        poll_s = HAND_CLOSE_POLL_S
    if deadline_s is None:
        deadline_s = HAND_CLOSE_DEADLINE_S

    def _log(msg):
        sys.stderr.write(f'[pick_daemon] bounded-lead close ({side}): {msg}\n')
        sys.stderr.flush()

    try:
        ceiling = {jid: float(preset[jid]) for jid in DEX3_JOINTS}
    except (KeyError, TypeError, ValueError):
        # an incomplete ceiling would let the commanded and scored poses drift
        # apart. refuse instead.
        _log('preset incomplete — writing NOTHING')
        return 'no_data'

    # the state channel can flicker; a transient empty read must not count as
    # blind, which would leave the hand open where it is closed.
    q = {}
    for _try in range(max(1, int(HAND_CLOSE_STATE_RETRIES))):
        q = _read_hand_actual(client, side)
        if q:
            break
        time.sleep(HAND_CLOSE_STATE_RETRY_S)
    if not q:
        _log(f'NO HAND STATE after {int(HAND_CLOSE_STATE_RETRIES)} reads — '
             f'writing NOTHING.  The lead over q_meas is what the motor sees as '
             f'torque, so with no q_meas there is nothing to bound; commanding '
             f'anyway IS the rail that latches the motor')
        return 'no_data'

    written = None
    presses = 0
    how = 'cap'
    _ever_ceil = False
    deadline = time.time() + float(deadline_s)
    while presses < int(max_presses) and time.time() < deadline:
        target = _bounded_lead_target(client, side, ceiling, eps=eps)
        if target is None:
            _log('hand state lost mid-close — stopping the walk, holding the '
                 'last bounded target (no further writes)')
            return 'no_data'
        _at_ceil = _at_close_ceiling(target, side, ceiling)
        # fixed-point exit only makes sense while the command can still advance.
        # at the ceiling the command is already the preset and repeats, so it
        # must not read as the fingers stopping.
        if (not _at_ceil) and written is not None and _targets_within(
                target, written, HAND_CLOSE_TARGET_TOL):
            how = 'stopped'
            break
        try:
            client.set(target, steps=HAND_CLOSE_PRESS_WRITES, dt=dt)
        except Exception as _exc:
            _log(f'press REFUSED ({_exc}) — holding the last bounded target, '
                 f'not pressing blind')
            return 'no_data'
        written = target
        presses += 1
        _arrived, _moved, _lost = _watch_hand(client, side, target, eps,
                                              watch_budget_s, poll_s)
        if _lost >= HAND_CLOSE_LOST_READS:
            _log('hand state LOST while watching the press — stopping the walk '
                 '(this is a DDS dropout, NOT a stall)')
            return 'no_data'
        if _at_ceil:
            _ever_ceil = True
            # the command is the preset, so further presses re-command the same
            # pose and the lead can only shrink, which costs nothing safety-wise.
            # this makes a free-air close independent of the caller's hold time:
            # _at_close_ceiling is true only when every moving joint is within eps
            # of the preset, so lead <= eps by construction and the finger still
            # has up to eps of travel left. stop on arrival, or when this press
            # produced no motion (a finger blocked at depth <= eps, logged as
            # 'ceiling' since the command really is home).
            if _arrived or _moved <= HAND_CLOSE_STILL_TOL:
                break

    if written is None:
        _log('no press issued (zero presses)')
        return 'no_data'

    _held = _worst_closing_lead(client, side, written)
    _held_s = ('n/a' if _held is None else f'{_held:.3f}')
    _tau_s = ('n/a' if _held is None
              else f'{1.09 + 16.6 * _held:.1f}')
    if how == 'stopped':
        _log(f'BLOCKED: the fingers stopped after {presses} press(es).  HOLDING '
             f'at a bounded lead of {_held_s} rad — tau_est ~= bias + 16.6*lead '
             f'~= {_tau_s} N.m, vs the ~15 N.m rail at ~0.83 rad of lead')
    elif _ever_ceil:
        _log(f'the command reached the close preset after {presses} press(es) '
             f'— the server holds it home; held lead {_held_s} rad '
             f'(tau_est <= {_tau_s} N.m)')
    else:
        _log(f'*** GAVE UP after {presses} press(es) with the fingers STILL '
             f'MOVING — holding the last BOUNDED target (held lead {_held_s} '
             f'rad, tau_est <= {_tau_s} N.m).  The closure-block margin decides '
             f'the keep; this corner is logged, not hidden')
    return 'full'

def _squeeze_hold(client, side, preset, steps=20, dt=0.01, hold_s=0.5):
    """v5.2 (2026-09-07) Edit E: re-assert the close preset, settle, re-read.

    The v5.1 blocked-read ran right after the 1.0 s ramp — the stalled
    motors had not yet converged, so the margin read was noisy.  Re-asserting
    the SAME preset re-presses the grip, `hold_s` lets the stall converge,
    then the margin is re-scored.  Load-bearing part = settle + re-read.

    The preset is re-asserted AND the re-score is taken against that SAME
    preset (Task #10, 2026-09-18): the two must not drift.  Passing a capped
    target here while `_score_closure_blocked` defaulted to the uncapped
    CLOSE preset scored |q(0.600) - q(1.0)| and reported a false capture on
    every free-air close.

    Returns ((blocked, margin, worst) after the hold).
    """
    # the re-press is bounded. a full-preset re-command is an unbounded-lead
    # ramp onto a hand already blocked by the object, the same rail as the
    # initial close, repeated at the pre-lift pin and the tuck re-pin.
    # force comes from the bounded lead per joint against the measured pose:
    # tau_est ~= bias + 16.6*HAND_CLOSE_LEAD_EPS ~= 5 N.m.
    # the score stays unchanged on purpose: `preset` (the caller's
    # F_MAX_SAFE-clamped full preset) is still the margin reference, so
    # _close_blocked still means |q_actual - preset| > CLOSURE_BLOCKED_TOL.
    # scoring the bounded target would make margin == eps < tolerance and
    # report blocked=False on a blocked hand, which the pre-lift pin would turn
    # into an aborted lift.
    # `steps` and `dt` are kept for signature compatibility, not pacing: a fixed
    # step count is what produced the unbounded ramp. the press count comes from
    # convergence; writes are HAND_CLOSE_PRESS_WRITES (== 1, load-bearing) at
    # HAND_CLOSE_PRESS_DT.
    try:
        _bounded_lead_close(client, side, preset)
    except Exception:
        pass
    time.sleep(hold_s)
    return _score_closure_blocked(client, side, preset)


def freeze_idle_arm(arm, sol_q, side):
    """T5/T8: overwrite the idle arm's slice of sol_q with its KNOWN
    CARRY_TUCK pose — deterministic pose-pin (Task 8 hardware batch).

    Called after EVERY dual-arm IK solve during the active arm's waypoints
    (waypoints, wp4 guarded insert, contact-relief retract): the inactive
    arm stays at carry_tuck — no nullspace drift, regardless of how the
    solver perturbs the idle side.  The inactive arm's target is ALWAYS
    CARRY_TUCK_{LEFT,RIGHT} by design (step 2 just moved it there), so we
    pin the KNOWN pose instead of sampling get_current_dual_arm_q (which
    returns the ACTUAL lowstate q).  Sampling closed a settle race:
    2026-08-17 the sample could land while the arm was still tracking the
    step-2 sweep, freezing it at an INTERMEDIATE pose ("stopped halfway
    into carry_tuck").  The pin is deterministic and also makes the idle
    arm CONTINUE into its tuck pose during the approach if it was still
    moving.  Belt-and-braces (Task 8 diagnosis batch): a bounded settle-wait
    polls until the idle arm ACTUALLY tracks the known pose (max |cmd−meas|
    < 0.03 rad, cmd = the pinned CARRY_TUCK, meas = actual lowstate q) or
    2.5 s elapse, so the first commanded move of a pick flow starts from a
    settled arm, not a still-tracking one.  Never blocks a pick: a timeout
    just logs and proceeds (the pin still commands the known pose).
    Mutates and returns ``sol_q``.  API unchanged (arm kept for call-site
    compatibility).
    """
    if side == 'left':
        idx = slice(7, 14)
        target = _zf_idle_target('right', CARRY_TUCK_RIGHT)   # the pin, when set
    else:
        idx = slice(0, 7)
        target = _zf_idle_target('left', CARRY_TUCK_LEFT)     # the pin, when set

    _settled = False
    _meas = None
    _deadline = time.time() + 0.5  # 2.5 s would stall every waypoint; the idle arm rarely needs that long
    _tol_rad = 0.08 if _zfloor_perarm_on() else 0.03   # 0.03 leaves the settle wait too short
    while time.time() < _deadline:
        try:
            _meas = arm.get_current_dual_arm_q()
        except Exception:
            break
        if np.max(np.abs(_meas[idx] - target)) < _tol_rad:
            _settled = True
            break
        time.sleep(0.05)
    if not _settled:
        _err = (np.max(np.abs(_meas[idx] - target)) if _meas is not None
                else float('nan'))
        _idle_name = 'left' if side == 'right' else 'right'
        sys.stderr.write(
            f'[pick_daemon] freeze_idle_arm: {_idle_name} arm not settled at its '
            f'tuck pose within {"0.5 s" if _zfloor_perarm_on() else "2.5 s"} '
            f'(max err {_err:.3f} rad) — proceeding '
            f'with the pose-pin anyway\n')
        sys.stderr.flush()

    sol_q[idx] = target
    return sol_q


def _solver_fresh(arm_ik, log_prefix=""):
    """v5.2 (2026-09-07) Edit C: make the NEXT solve raw/fresh.

    The solver's WeightedMovingFilter([0.4,0.3,0.2,0.1]) has NO reset() and
    blends every solve_ik call (robot_arm_ik.py L227-228: add_data then
    sol_q = smooth_filter.filtered_data).  Once its queue is full the NEWEST
    solve gets weight 0.1 — waypoint commands become ~90% stale, and the
    queue even survives ACROSS picks (the chain of warm starts: q[i-1] ->
    filtered q[i]).  _apply_filter returns queue[-1] while the queue is
    shorter than the window, so clearing _data_queue (private attribute —
    try/except guarded) makes the NEXT solve pass through 100% fresh.
    Called ONCE before the first solve of each waypoint phase; mid-phase
    re-solves (resume/pushes) keep smoothing.

    Returns the number of entries cleared (0 when there is no filter /
    attribute — degrades to a visible no-op, never raises).
    """
    _f = getattr(arm_ik, 'smooth_filter', None)
    if _f is None:
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}solver-fresh: no smooth_filter on '
            'the IK module — skip\n')
        sys.stderr.flush()
        return 0
    try:
        _n = len(_f._data_queue)
        _f._data_queue.clear()
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}solver-fresh: cleared {_n} stale '
            f'filter entries — next solve raw/fresh\n')
        sys.stderr.flush()
        return _n
    except AttributeError:
        sys.stderr.write(
            f'[pick_daemon] {log_prefix}solver-fresh: no _data_queue '
            'attribute — skipping (sealed-solver drift?)\n')
        sys.stderr.flush()
        return 0


def solve_and_move(arm, arm_ik, target, log_prefix="", side='right', raw=False):
    """Solve IK and move the arm smoothly.

    Parameters
    ----------
    side : str
        'right' → right arm moves to target, left arm stays at current position.
        'left'  → left arm moves to target, right arm stays at current position.
    raw : bool
        v5.0 (2026-09-06): True → the ORIGINAL raw ``arm_ik.solve_ik``
        (sacred reference solve_and_move L588-616), verbatim.  Used by the
        HORIZONTAL swing (h-wp1/h-wp2, both arms) so the wrist turns /
        normalises DURING the swing exactly like the original — the
        verified-solve wrist-GN was the palm-tilt signature (h ori_err
        0.18-0.22 rad) and the "reach-then-turn" look.  False (default) →
        verified solve (wash + FK verify + wrist-GN), kept for all UPRIGHT
        approach waypoints (deviation + verified = user feature).
    """
    # pin the idle arm where it is so it does not drift
    _cur_q = arm.get_current_dual_arm_q().copy()
    if side == 'left':
        _idle_wrist = get_wrist_pose(arm_ik, _cur_q, side='right')
        _idle_target = (pin.SE3(np.eye(3), _idle_wrist).homogeneous
                        if _idle_wrist is not None else R_NEUTRAL)
        if raw:
            # arg order: (target, _idle_target), grasp first for the left arm.
            # do not swap.
            with _SuppressStdout():
                sol_q, sol_tauff = arm_ik.solve_ik(
                    target, _idle_target,
                    current_lr_arm_motor_q=_cur_q.copy(),
                    current_lr_arm_motor_dq=np.zeros(14))
        else:
            # verified solve: wash + FK verify + wrist-GN, loud on failure
            sol_q, sol_tauff = _solve_ik_verified(arm_ik, target, _idle_target,
                                                  _cur_q, 'left', log_prefix)
    else:
        _idle_wrist = get_wrist_pose(arm_ik, _cur_q, side='left')
        _idle_target = (pin.SE3(np.eye(3), _idle_wrist).homogeneous
                        if _idle_wrist is not None else L_NEUTRAL)
        if raw:
            with _SuppressStdout():
                sol_q, sol_tauff = arm_ik.solve_ik(
                    _idle_target, target,
                    current_lr_arm_motor_q=_cur_q.copy(),
                    current_lr_arm_motor_dq=np.zeros(14))
        else:
            sol_q, sol_tauff = _solve_ik_verified(arm_ik, _idle_target, target,
                                                  _cur_q, 'right', log_prefix)

    if sol_q is None:
        return False

    # no early return when solve_ik comes back with the current q (an Ipopt
    # silent failure): command what came back, a no-op move, rather than skip
    # the waypoint.
    # freeze the idle arm: hard-overwrite its slice with the current q, sampled
    # after the carry_tuck move settled, before commanding anything.
    sol_q = freeze_idle_arm(arm, sol_q, side)

    sys.stderr.write(f'[pick_daemon] {log_prefix}IK solved, moving...\n')
    sys.stderr.flush()
    move_arm_smooth(arm, sol_q, sol_tauff,
                    zf_label=(log_prefix.strip().rstrip(':').strip() or None))
    return True


def _t5_joint_move(arm, side, pose, duration=None, zf_legacy=False):
    """T5 helper: move ONE arm to a fixed joint-space pose (LIFT/CARRY_TUCK).

    The other (idle) arm's slice is kept at its CURRENT q — a full 14-vector
    is built here, so the idle arm is untouched by construction (equivalent
    to freeze_idle_arm but no IK solve is involved).  Hand state is NOT
    touched by this helper — the caller keeps the hand closed.

    duration (v5.2, 2026-09-07): pass-through to move_arm_smooth — a gentler
    CARRY_TUCK wrist roll (CARRY_TUCK_DURATION=4.0) mitigates slinging a
    light capture out of the fingers.
    """
    _t5_q = arm.get_current_dual_arm_q().copy()
    if side == 'right':
        _t5_q[7:14] = pose
    else:
        _t5_q[0:7] = pose
    # gravity FF at the target pose, not zeros: move_arm_smooth re-asserts
    # ctrl_dual_arm(target_q, target_tauff) at the end, so zeros leave the arm
    # without gravity comp after ready/lift/CARRY_TUCK/neutral and it sags on
    # hardware.
    tau_target = _arm_gravity_ff(_t5_q, np.zeros(14))
    if duration is None:
        move_arm_smooth(arm, _t5_q, tau_target, zf_legacy=zf_legacy)
    else:
        move_arm_smooth(arm, _t5_q, tau_target, duration=duration,
                        zf_legacy=zf_legacy)


def waypoint_to_arrays(wp):
    """Convert a motion sequence waypoint to arm_q[14] + hand dicts."""
    arm_data = wp.get('arm', {})
    arm_q = np.zeros(14)
    for i in range(14):
        key = str(15 + i)
        if key in arm_data:
            arm_q[i] = arm_data[key]

    rhand = {int(k): v for k, v in wp.get('rhand', {}).items()}
    lhand = {int(k): v for k, v in wp.get('lhand', {}).items()}

    return arm_q, rhand, lhand


def play_sequence(arm, rhand, lhand, seq_data):
    """Play a motion sequence with per-waypoint timing and hand control."""
    waypoints = seq_data.get('waypoints', [])
    total = len(waypoints)

    for idx, wp in enumerate(waypoints):
        name = wp.get('name', f'wp{idx+1}')
        move_time = wp.get('move_time', 1.5)
        hold_time = wp.get('hold_time', 0.5)

        sys.stderr.write(
            f'[pick_daemon] Sequence: {name} ({idx+1}/{total}) '
            f'move={move_time}s hold={hold_time}s\n')
        sys.stderr.flush()

        arm_q, rhand_targets, lhand_targets = waypoint_to_arrays(wp)

        # move the arm over the given duration
        tau_target = _arm_gravity_ff(arm_q, np.zeros(14))
        move_arm_smooth(arm, arm_q, tau_target, duration=move_time)

# set the hands at the same time (fast, does not block long)
        if rhand_targets:
            rhand.set(rhand_targets)
        if lhand_targets:
            lhand.set(lhand_targets)

        # hold there
        if hold_time > 0:
            _check_cancel('sequence_hold')  # the arm move above is already covered;
            time.sleep(hold_time)           # this bounds the wait between


def _run_diag_bounded(fn, label, timeout=5.0):
    """Run a diagnostic under a hard time bound.

    READY must never wait on sensor availability.  The daemon imports
    real_grasp_detector from /home/unitree/leon (deployed copy) — if that
    copy ever blocks inside a diagnostic (e.g. a DDS read with a long
    timeout), init stalls for 15+s while commands buffer in stdin.  This
    bounds the worst case so READY is emitted deterministically.

    Returns fn()'s value, or None if it exceeds the bound.
    """
    import threading as _t
    _result = {}

    def _run():
        try:
            _result['val'] = fn()
        except Exception as e:
            _result['err'] = e

    _th = _t.Thread(target=_run, daemon=True)
    _th.start()
    _th.join(timeout)
    if _th.is_alive():
        sys.stderr.write(
            f'[pick_daemon] WARNING: {label} exceeded {timeout:.0f}s bound — '
            'skipping diagnostic (READY not blocked)\n')
        sys.stderr.flush()
        return None
    if 'err' in _result:
        raise _result['err']
    return _result.get('val')


def _parse_pick_tail(parts):
    """Parse the optional tail of a pick / pick_left command.

    The orientation is the only non-numeric token ('upright' / 'horizontal');
    up to three floats BEFORE it are the surface normal (nx ny nz), and floats
    AFTER it are the optional trailing [height_m, x_off] (raw token strings).

    Returns (nx, ny, nz, orientation, height_m, x_off); nx/ny/nz are None when
    no normal is given, height_m/x_off are None when absent.
    """
    orientation = 'upright'
    normals = []
    trailing = []
    section = 'before'
    for tok in parts[4:]:
        if tok in ('upright', 'horizontal'):
            orientation = tok
            section = 'after'
        elif section == 'before':
            normals.append(tok)
        else:
            trailing.append(tok)
    nx = ny = nz = None
    if normals:
        if len(normals) != 3:
            raise ValueError('expected 3 normal components, got %d' % len(normals))
        nx, ny, nz = float(normals[0]), float(normals[1]), float(normals[2])
    if len(trailing) > 2:
        raise ValueError('too many trailing tokens')
    height_m = trailing[0] if len(trailing) >= 1 else None
    x_off = trailing[1] if len(trailing) >= 2 else None
    return nx, ny, nz, orientation, height_m, x_off


# ── the grasp-verdict file log (diagnostic) ─────────────────────
# the gateway spawns this engine with stdout/stderr on pipes that are thrown
# away, so the live grasp numbers (closure-block margin, pressure delta,
# support_ratio, final quadrant) would be lost. _grasp_log tees them to an
# append-only file. best-effort: never raises and never touches the decision.
_GRASP_LOG_PATH = '/tmp/pick_walk_logs/engine_grasp.log'


def _grasp_log(arm: str, attempt: int, **kv) -> None:
    try:
        import os as _os, time as _t
        _os.makedirs(_os.path.dirname(_GRASP_LOG_PATH), exist_ok=True)
        parts = ' '.join(f'{k}={v}' for k, v in kv.items())
        with open(_GRASP_LOG_PATH, 'a') as _f:
            _f.write(f'{_t.time():.3f} {arm} a{attempt} {parts}\n')
    except Exception:
        pass


def main():
    iface = sys.argv[1] if len(sys.argv) > 1 else 'eth0'
    active_arm = sys.argv[2] if len(sys.argv) > 2 else 'both'
    # 'upper' selects the arms-only controller, which streams arm targets while
    # walking and never writes legs or waist.
    upper_only = 'upper' in sys.argv[3:]
    # 'nomove' skips the init move to NEUTRAL: no arm motion at startup, arms
    # held at the current q, hands still open, diagnostics run, gate engaged.
    nomove = 'nomove' in sys.argv[3:]
    if active_arm not in ('both', 'left', 'right'):
        sys.stderr.write(f'[pick_daemon] Invalid arm "{active_arm}", using "both"\n')
        sys.stderr.flush()
        active_arm = 'both'

    sys.stderr.write('[pick_daemon] Loading pinocchio model...\n')
    sys.stderr.flush()
    # fork guard: a native library (DDS/CasADi) forks during init, and the child
    # would become a second arm writer, making the arm oscillate. exit any forked
    # child instantly via os.register_at_fork.
    def _exit_forked_child():
        os._exit(0)
    if hasattr(os, 'register_at_fork'):
        os.register_at_fork(after_in_child=_exit_forked_child)
    with _SuppressStdout():
        arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
    global _ARM_IK  # per-step gravity FF inside move_arm_smooth*
    _ARM_IK = arm_ik

    # diagnostic: confirm the per-step gravity FF returns non-zero torques on
    # this hardware. zero or near-zero means the reduced-model rnea is broken, so
    # the arms fight gravity with kp alone and barely move.
    try:
        _ff_ready = _arm_gravity_ff(READY_POSE, np.zeros(14))
        _ff_tuck = _arm_gravity_ff(
            np.concatenate([CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT]), np.zeros(14))
        sys.stderr.write(
            f'[pick_daemon] gravity_ff CHECK ready |tau|='
            f'{np.linalg.norm(_ff_ready):.3f} carry_tuck |tau|='
            f'{np.linalg.norm(_ff_tuck):.3f}\n')
        sys.stderr.flush()
    except Exception as _ffe:
        sys.stderr.write(f'[pick_daemon] gravity_ff CHECK FAILED: {_ffe}\n')
        sys.stderr.flush()

    sys.stderr.write(f'[pick_daemon] Connecting on {iface}...\n')
    sys.stderr.flush()
    ChannelFactoryInitialize(0, iface)
    # arm transport mode is picked before construction. in writer mode the merged
    # writer owns rt/lowcmd and the arms and this daemon is a non-owning socket
    # client with no arm_sdk dds writer. DDS stays up only for the poll-only dex3
    # reads in RealGraspDetector.
    global ARM_WRITER_ENABLED
    ARM_WRITER_ENABLED = _detect_writer_mode()
    if ARM_WRITER_ENABLED:
        sys.stderr.write(
            f'[pick_daemon] ARM TRANSPORT: merged writer socket '
            f'({WRITER_SOCK_DEFAULT}) — NO arm_sdk writer; commands via '
            'WriterArmClient (engine lane)\n')
        sys.stderr.flush()
        try:
            arm = WriterArmClient()
            arm.q()  # fail-fast ping: the writer has to be up before taking commands
        except WriterError as _we:
            sys.stderr.write(
                f'[pick_daemon] FATAL: writer unreachable over the socket: '
                f'{_we}\n')
            sys.stderr.flush()
            raise SystemExit(1)
    elif upper_only:
        # arms-only controller: legs and waist are never written.
        from upper_arm_controller import G1_29_ArmControllerUpperOnly
        arm = G1_29_ArmControllerUpperOnly(motion_mode=True,
                                           simulation_mode=False)
        sys.stderr.write('[pick_daemon] UPPER-ONLY controller active '
                         '(release/acquire/hold available)\n')
    else:
        arm = G1_29_ArmController(motion_mode=True, simulation_mode=False)
    sys.stderr.flush()
    # persistent grip server owns the hands over a UNIX socket; the daemon routes
    # via GripClient. the server survives the phase-split quit and keeps the last
    # commanded q asserted at 20 Hz, since one HandCmd_ write does not persist on
    # the dex3 MCU.
    rhand = GripClient('right')
    lhand = GripClient('left')

    # hand-state DDS listeners are absent on purpose: ChannelSubscriber callbacks
    # fire on the cyclonedds receive thread at 200-1000 Hz and starve the 250 Hz
    # loop via the GIL, causing slow back-and-forth jitter. do not re-add them.
    

    # ── phase 3: the real-hardware grasp detector (per-side singletons) ────────
    _grasp_detectors = {}

    def _get_grasp_detector(side='right'):
        if side not in _grasp_detectors:
            try:
                _grasp_detectors[side] = RealGraspDetector(side=side, object_mass=0.30)
            except TypeError:
                # fallback for a detector with no 'side' parameter
                sys.stderr.write(
                    '[pick_daemon] WARNING: RealGraspDetector lacks side param — '
                    'using default (right-hand topic). Deploy updated '
                    'real_grasp_detector.py to Stuart.\n')
                sys.stderr.flush()
                _grasp_detectors[side] = RealGraspDetector(object_mass=0.30)
        return _grasp_detectors[side]

    # writer lane: the merged writer owns the arm gain and velocity-limit config.
    # speed_gradual_max and arm_velocity_limit are arm_sdk-controller API only, so
    # the writer's own regime stands.
    if ARM_WRITER_ENABLED:
        sys.stderr.write('[pick_daemon][writer] arm gains + velocity limit '
                         'left to the merged writer (no arm_sdk controller)\n')
        sys.stderr.flush()
    else:
        arm.speed_gradual_max(t=1.0)
        time.sleep(1.2)
        # velocity-limit handling: set_velocity_limit marks the limit owner-set so the
        # controller's stock gradual ramp stops stomping the cap. the controller
        # clips commanded position error to V*dt, so a low limit damps the arm to
        # ~0.16 rad/s and it never reaches poses. the daemon rate-limits its own
        # moves (cosine at MOVE_SPEED), so 20.0 never binds.
        arm.arm_velocity_limit = 1.5

    # init = NEUTRAL (both arms at the lowstate-sourced hanging rest). --arm is
    # still parsed for protocol compatibility but does not select a boot pose.
    _init_pose = np.concatenate([NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
    if not nomove:
        sys.stderr.write(
            '[pick_daemon] Moving BOTH arms to NEUTRAL (hanging rest)...\n')
        sys.stderr.flush()
        move_arm_smooth(arm, _init_pose, _arm_gravity_ff(_init_pose, np.zeros(14)))
    else:
        sys.stderr.write(
            '[pick_daemon] nomove mode: SKIPPING init move to NEUTRAL — '
            'arms held at current q (no arm motion at startup)\n')
        sys.stderr.flush()

    # fingers closed on daemon launch; the hands stay shut until a 'ready' or the
    # pick flow's ready phase opens them. ready is the only pose where the fingers
    # open on their own.
    rhand.close()
    lhand.close()

    # ── palm sensor diagnostic (check the DDS subscription is live) ───────────
    sys.stderr.write('[pick_daemon] Running palm sensor diagnostic...\n')
    sys.stderr.flush()
    _diag_right = _get_grasp_detector('right')
    _diag_right_sensors_ok = _run_diag_bounded(
        _diag_right.diagnose_palm_sensors, 'palm diagnostic (right)')
    _diag_left = _get_grasp_detector('left')
    _diag_left_sensors_ok = _run_diag_bounded(
        _diag_left.diagnose_palm_sensors, 'palm diagnostic (left)')
    _palm_data_available = _diag_right_sensors_ok and _diag_left_sensors_ok
    # right-side flag: the left Dex3 never powers on, so nothing publishes
    # rt/dex3/left/state and the grip server's `state left` reply has no `actual`
    # key. diagnose_palm_sensors('left') then blocks and times out, so the
    # combined flag stays falsy, and a falsy flag would turn every
    # `grasped_nothing` verdict into a success. right picks therefore key off the
    # right sensor alone. left picks keep the both-sides flag: a left pick on a
    # dead hand has no verifiable data anyway.
    _palm_data_available_right = bool(_diag_right_sensors_ok)
    if not _palm_data_available:
        sys.stderr.write(
            '[pick_daemon] WARNING: No DDS hand-state data — '
            'dex3_server may not be running.\n'
            '[pick_daemon]   palm diagnostic right=%r left=%r\n'
            % (_diag_right_sensors_ok, _diag_left_sensors_ok))
        sys.stderr.write(
            '[pick_daemon]   Palm-pressure checks DISABLED. '
            'Grasp verification will use BLIND mode.\n')
        sys.stderr.flush()

    # ── finger sensor diagnostic (check finger pad + fingertip sensors) ───────
    sys.stderr.write('[pick_daemon] Running finger sensor diagnostic...\n')
    sys.stderr.flush()
    _run_diag_bounded(
        _diag_right.diagnose_finger_sensors, 'finger diagnostic (right)')
    _run_diag_bounded(
        _diag_left.diagnose_finger_sensors, 'finger diagnostic (left)')

    sys.stdout.write('READY\n')
    sys.stdout.flush()
    sys.stderr.write('[pick_daemon] READY\n')
    sys.stderr.flush()

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue

        # one stdout line per command: the gateway reads exactly one reply
        # line. the flag is re-initialised at the top of every iteration, so
        # it cannot outlive its command; a stale True would swallow the next
        # reply and fake success. reply sites set it after their single write;
        # guards turn a second line into a stderr note.
        _reply_sent = False
        _zf_cmd_begin()   # flow context + per-command tallies start empty

        sys.stderr.write(f'[pick_daemon] CMD: {cmd}\n')
        sys.stderr.flush()

        try:
            # ── hand commands ──────────────────────────────────────────────
            if cmd == 'open':
                rhand.open()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'close':
                rhand.close()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'open_left':
                lhand.open()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'close_left':
                lhand.close()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'open_both':
                rhand.open()
                lhand.open()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'close_both':
                rhand.close()
                lhand.close()
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            # ── arm commands ───────────────────────────────────────────────
            elif cmd == 'home':
                # both arms to HOME (hanging at the torso sides). move_arm_smooth
                # is the same call the sibling poses use: it works on both
                # transports, carries gravity FF, applies the z-floor and is
                # cancellable. the writer's slew rate is fixed by arm_slew_rate
                # (0.3 rad/s). arms only; hands are left alone.
                move_arm_smooth(arm, HOME_POSE, _arm_gravity_ff(HOME_POSE, np.zeros(14)))
                sys.stdout.write('DONE' + _zf_reply_suffix('home') + '\n')
                sys.stdout.flush()

            elif cmd == 'ready':
                # both arms to READY. the arm reaches the pose first, then the
                # fingers open: opening at NEUTRAL or tuck flails them at the
                # hanging rest. both hands open at ready.
                move_arm_smooth(arm, READY_POSE, _arm_gravity_ff(READY_POSE, np.zeros(14)))
                rhand.open()
                lhand.open()
                sys.stdout.write('DONE' + _zf_reply_suffix('ready') + '\n')
                sys.stdout.flush()

            elif cmd == 'ready_right':
                # ready, then open hands
                move_arm_smooth(arm, READY_POSE, _arm_gravity_ff(READY_POSE, np.zeros(14)))
                rhand.open()
                lhand.open()
                sys.stdout.write('DONE' + _zf_reply_suffix('ready_right') + '\n')
                sys.stdout.flush()

            elif cmd == 'ready_left':
                # ready, then open hands
                move_arm_smooth(arm, READY_POSE, _arm_gravity_ff(READY_POSE, np.zeros(14)))
                rhand.open()
                lhand.open()
                sys.stdout.write('DONE' + _zf_reply_suffix('ready_left') + '\n')
                sys.stdout.flush()

            elif cmd == 'carry':
                # right arm -> CARRY_TUCK (joint-space); left pinned at its current q
                _t5_joint_move(arm, 'right', CARRY_TUCK_RIGHT)
                sys.stdout.write('DONE' + _zf_reply_suffix('carry') + '\n')
                sys.stdout.flush()

            elif cmd == 'carry_left':
                # left arm -> CARRY_TUCK (joint-space); right pinned at its current q
                _t5_joint_move(arm, 'left', CARRY_TUCK_LEFT)
                sys.stdout.write('DONE' + _zf_reply_suffix('carry_left') + '\n')
                sys.stdout.flush()

            elif cmd == 'neutral':
                # both arms -> NEUTRAL (lowstate-sourced hanging rest)
                _neutral_q = np.concatenate(
                    [NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
                move_arm_smooth(arm, _neutral_q, _arm_gravity_ff(_neutral_q, np.zeros(14)))
                # close both hands after the arms settle, including at NEUTRAL.
                # best-effort: never fails the verb.
                _close_hand_at_closeout(rhand, 'right', 'neutral')
                _close_hand_at_closeout(lhand, 'left', 'neutral')
                sys.stdout.write('DONE' + _zf_reply_suffix('neutral') + '\n')
                sys.stdout.flush()

            elif cmd in ('release', 'acquire', 'hold'):
                # gate control, valid only with the upper-only controller; the stock
                # 35-joint controller keeps the arm_sdk gate for itself, so an error
                # is returned there.
                if not hasattr(arm, 'release_arms'):
                    sys.stdout.write(
                        'ERROR: release/acquire/hold require the upper-only '
                        "controller (daemon argv 'upper')\n")
                    sys.stdout.flush()
                else:
                    if cmd == 'release':
                        arm.release_arms()
                    elif cmd == 'acquire':
                        arm.acquire_arms()
                    else:
                        arm.hold_only()
                    sys.stdout.write('DONE\n')
                    sys.stdout.flush()

            # ── motion sequence ────────────────────────────────────────────
            elif cmd == 'floor_clear':
                # drop the latched table z-floor. the engine has no navigate hook, so
                # a floor latched at a high surface would keep clamping NEUTRAL in
                # free space until the next pick or place re-latches it. this verb is
                # the reset.
                _clear_table_floor('operator floor_clear')
                _zf_clear_pins()   # the pin goes with the floor
                sys.stdout.write('DONE\n')
                sys.stdout.flush()

            elif cmd == 'list':
                files = sorted(glob.glob(os.path.join(SEQUENCES_DIR, '*.json')))
                names = [os.path.splitext(os.path.basename(f))[0] for f in files]
                # list as one comma-separated line
                sys.stdout.write('DONE:' + ','.join(names) + '\n')
                sys.stdout.flush()

            elif cmd == 'quit':
                # release the RC interlock first; the writer keeps owning the arms
                # and RC is live on exit.
                _hold_rc_writer(arm, False, 'quit')
                # graceful daemon exit, required between the pick and walk phases:
                # any live arm_sdk writer trips the FSM. ramp the arm_sdk gate off
                # first (upper-only controller); the stock controller never took the
                # gate. print DONE, break the stdin loop and exit.
                # posture handover: record the dual-arm q before the low-carry
                # down-move and release_arms, so the walk phase can restore the carry
                # posture. order: write the q, joint-space down-move to LOW_CARRY
                # (~1.8 s), then release_arms so the arms fall to hanging clear of
                # the legs. failures never block the quit.
                _last_q = None
                try:
                    _last_q = arm.get_current_dual_arm_q().tolist()
                except Exception as _q_exc:
                    sys.stderr.write(
                        f'[pick_daemon] quit: get_current_dual_arm_q failed '
                        f'({_q_exc}) — no posture handover json\n')
                    sys.stderr.flush()
                if _last_q is not None and len(_last_q) == 14:
                    try:
                        with open('/tmp/last_arm_q.json', 'w') as _qf:
                            json.dump({
                                'left': _last_q[0:7],
                                'right': _last_q[7:14],
                                'ts': time.time(),
                            }, _qf)
                        sys.stderr.write(
                            '[pick_daemon] quit: posture handover json '
                            'written to /tmp/last_arm_q.json\n')
                        sys.stderr.flush()
                    except Exception as _q_exc:
                        sys.stderr.write(
                            f'[pick_daemon] quit: posture json write failed '
                            f'({_q_exc}) — continuing quit\n')
                        sys.stderr.flush()
                # leg-hit mitigation: joint-space down-move of both arms to the low
                # hanging-carry pose before release_arms. release lets the arms fall
                # to hanging, which can swing them into the legs and knock the object
                # off the grip. cosine ramp of ~1.8 s; failure is non-blocking.
                try:
                    _down_q = np.concatenate(
                        [LOW_CARRY_LEFT, LOW_CARRY_RIGHT]).astype(np.float64)
                    move_arm_smooth(arm, _down_q,
                                    _arm_gravity_ff(_down_q, np.zeros(14)),
                                    duration=1.8)
                    sys.stderr.write(
                        '[pick_daemon] quit: down-moved both arms to the '
                        'low carry pose\n')
                    sys.stderr.flush()
                except Exception as _q_exc:
                    sys.stderr.write(
                        f'[pick_daemon] quit: low-carry down-move failed '
                        f'({_q_exc}) — continuing to release\n')
                    sys.stderr.flush()
                if hasattr(arm, 'release_arms'):
                    try:
                        arm.release_arms()
                    except Exception as _q_exc:
                        sys.stderr.write(
                            f'[pick_daemon] quit: release_arms failed '
                            f'({_q_exc}) — exiting anyway\n')
                        sys.stderr.flush()
                sys.stdout.write('DONE' + _zf_reply_suffix('quit') + '\n')
                sys.stdout.flush()
                sys.stderr.write(
                    '[pick_daemon] Quit (phase split) — daemon exiting, '
                    'no arm_sdk writer remains\n')
                sys.stderr.flush()
                break

            elif cmd.startswith('play '):
                seq_name = cmd[5:].strip()
                seq_path = os.path.join(SEQUENCES_DIR, f'{seq_name}.json')

                if not os.path.exists(seq_path):
                    sys.stdout.write(f'ERROR:sequence not found: {seq_name}\n')
                    sys.stdout.flush()
                    continue

                with open(seq_path, 'r') as f:
                    seq_data = json.load(f)

                wp_count = len(seq_data.get('waypoints', []))
                sys.stderr.write(
                    f'[pick_daemon] Playing "{seq_name}" ({wp_count} waypoints)\n')
                sys.stderr.flush()

                play_sequence(arm, rhand, lhand, seq_data)

                sys.stderr.write(f'[pick_daemon] Sequence "{seq_name}" complete\n')
                sys.stderr.flush()
                sys.stdout.write('DONE' + _zf_reply_suffix('play') + '\n')
                sys.stdout.flush()

            # ── pick ───────────────────────────────────────────────────────
            elif cmd.startswith('pick ') and not cmd.startswith('pick_left '):
                _zf_pick_begin('right', arm)
                parts = cmd.split()
                try:
                    if len(parts) < 4:
                        raise ValueError('pick requires x y z')
                    tx = float(parts[1])
                    ty = float(parts[2])
                    tz = float(parts[3]) + GRASP_Z_OFFSET
                    nx, ny, nz, orientation, height_m, x_off = _parse_pick_tail(parts)
                except (ValueError, IndexError) as e:
                    sys.stdout.write(f'ERROR:pick parse: {e}\n')
                    sys.stdout.flush()
                    _zf_restore_floor()   # no motion yet
                    continue
                # reference log and z-frame diagnostic only, no commanded change (see
                # SOLVER_Z_TORSO_OFFSET).
                
                sys.stderr.write(
                    '[pick_daemon] v5.2 lateral mirror: ik_y := -torso_x '
                    '(torso +x = RIGHT; URDF +Y = LEFT; both arms); '
                    f'z-frame DIAG: SOLVER_Z_TORSO_OFFSET='
                    f'{SOLVER_Z_TORSO_OFFSET:.3f} m (pelvis vs torso_link) '
                    f'- z channel unchanged\n')
                sys.stderr.flush()
                use_oriented = nx is not None

                if height_m is not None:
                    try:
                        height_m = float(height_m)
                    except ValueError:
                        sys.stdout.write('ERROR:bad height_m\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                    # <=0 or > max: height is not used
                    if height_m <= 0.0 or height_m > HEIGHT_M_MAX:
                        sys.stderr.write(
                            f'[pick_daemon] height_m={height_m} implausible '
                            f'(valid (0, {HEIGHT_M_MAX}]) — using legacy lift\n')
                        sys.stderr.flush()
                        height_m = None
                if x_off is not None:
                    try:
                        x_off = float(x_off)
                    except ValueError:
                        sys.stdout.write('ERROR:bad x_off\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                # point-cloud approach offset; fall back to the fixed standoff when
                # the depth projector sent no x_off.
                approach_offset = x_off if x_off is not None else APPROACH_FORWARD_OFFSET

                if orientation not in ('upright', 'horizontal'):
                    sys.stdout.write(f'ERROR:unknown orientation "{orientation}", expected "upright" or "horizontal"\n')
                    sys.stdout.flush()
                    _zf_restore_floor()   # no motion yet
                    continue

                # cross-verb carrier for the place-time extra lift: object height is
                # only measurable at pick time, before the object is occluded in the gripper.
                # the write sits outside the height_m gate: nested inside it, a
                # height-less pick would skip it, the holder would keep the previous
                # object's height and a later place would raise by the wrong value.
                # it also sits after every validation gate that refuses a malformed
                # command, so a command that never runs cannot clobber it. runtime
                # failures after this point leave the height in place, since the
                # object is the one requested.
                # two limits accepted: the slot is not keyed by side and not cleared
                # when a place consumes it. pick/place pairs are serial, and not
                # clearing lets a place be resent. every wrong-holder outcome is
                # bounded by PLACE_EXTRA_LIFT_MAX_M, handed back during
                # ASCEND/DESCEND, and never refuses a place: worst case is a raise of
                # the wrong size.
                _PLACE_LAST_OBJECT_HEIGHT_M[0] = height_m
                # table z-floor latch: same placement rule as the height holder,
                # after every validation gate, so a refused command cannot latch a
                # floor it never ran with.
                _latch_table_floor(tz, height_m, orientation)

                sys.stderr.write(
                    f'[pick_daemon] Pick target torso frame: '
                    f'x={tx:.3f} y={ty:.3f} z={tz:.3f}'
                    + (f' normal=({nx:.3f},{ny:.3f},{nz:.3f})' if use_oriented else '')
                    + f' orientation={orientation}'
                    + (f' height_m={height_m:.3f}' if height_m is not None
                       else ' height_m=unknown')
                    + '\n')
                sys.stderr.flush()

                # horizontal-pick ladder state, or None for the upright or
                # kill-switch flow. read at pick start, before any arm motion, to
                # capture the target label for re-detection.
                _hl = _hl_begin(orientation, tx, ty, tz, height_m)

                # reach precheck, before any arm motion. without it an out-of-reach
                # target is only refused at wp4, after the wp2/wp3 poses are
                # commanded and the arm has heated. the precheck solves the same
                # chain without moving the arm. refusal means no hold_rc, no arm move
                # and no hand move. 'unavailable' never refuses; the in-flow wp4 gate
                # is the second line, with the same knob and strict >.
                # this pick's grip evidence starts empty, so a pick ending by cancel
                # or exception cannot leave the previous pick's evidence answering
                # `grip_evidence` (it answers DONE:{}).
                _LAST_GRIP_EVIDENCE[0] = None
                try:
                    _pc_res, _pc_stage = _reach_precheck(
                        arm_ik, orientation, tx, ty, tz, x_off=x_off)
                except Exception as _pce:
                    sys.stderr.write(
                        f'[pick_daemon] REACH PRECHECK unavailable ({_pce!r}) '
                        '— not refusing\n')
                    sys.stderr.flush()
                    _pc_res, _pc_stage = None, 'unavailable'
                if _pc_res is not None and _pc_res > REACH_REFUSE_POS:
                    sys.stderr.write(
                        f'OUT OF REACH (precheck, {_pc_stage}): simulated '
                        f'residual {_pc_res:.3f} m > {REACH_REFUSE_POS:.3f} m '
                        f'for target x={tx:.3f} y={ty:.3f} z={tz:.3f} '
                        f'({orientation}) — the robot is standing too far '
                        'from the object.  Pick REFUSED before any arm '
                        'motion; move closer and retry.\n')
                    sys.stderr.flush()
                    _set_grip_evidence('right', 'failed')
                    sys.stdout.write(
                        f'ERROR:reach_ceiling residual={_pc_res:.3f}m stage=precheck\n')
                    sys.stdout.flush()
                    _zf_restore_floor()   # refused before any motion
                    continue

                # self-collision precheck: the right arm must not reach across the
                # body. a target left of the midline can drive the arm into the
                # torso, where the writer's PD holds it and the shoulder heats.
                # refused here, before hold_rc and any arm motion.
                _sc_pre = _self_collision_precheck(
                    arm, arm_ik, 'pick', orientation, tx, ty, tz)
                if _sc_pre is not None:
                    _set_grip_evidence('right', 'failed')
                    sys.stdout.write(_sc_line(_sc_pre))
                    sys.stdout.flush()
                    _zf_restore_floor()   # no motion yet
                    continue

                # state machine, steps 1-2: both arms to ready from
                # anywhere; step 2 sends the left (inactive) arm to
                # CARRY_TUCK while the right stays ready. the retry
                # loop below opens the right hand.
                # RC interlock on before any arm move: a held stick is
                # live input and norm(cmd) >= 0.05 flips Balance to
                # Walk, so the legs could step out mid-insert. off
                # after CARRY_TUCK and on failure.
                if ARM_WRITER_ENABLED:
                    try:
                        arm.hold_rc(True)
                    except Exception as _he:
                        sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                sys.stderr.write('[pick_daemon] T5 step 1: both arms → ready...\n')
                sys.stderr.flush()
                # the arm reaches READY first, then the fingers open: opening at
                # NEUTRAL flails them during the ready move.
                move_arm_smooth(arm, READY_POSE, _arm_gravity_ff(READY_POSE, np.zeros(14)))
                rhand.open()
                lhand.open()
                sys.stderr.write(
                    '[pick_daemon] T5 step 2: inactive (left) arm → carry_tuck...\n')
                sys.stderr.flush()
                _t5_prologue_q = arm.get_current_dual_arm_q().copy()
                _t5_prologue_q[0:7] = CARRY_TUCK_LEFT
                move_arm_smooth(arm, _t5_prologue_q, _arm_gravity_ff(_t5_prologue_q, np.zeros(14)))
                _zf_pin_idle('left')   # slice = pin
                # fresh retract trail for this pick, and grip-evidence slots that
                # cannot carry a previous pick (main()'s locals outlive the command).
                _trail_reset()
                _ge_close_blocked = None
                _ge_close_margin = None
                _ge_pin_margin = None

                # ── phase 3: the retry loop with grasp verification ────────────
                sys.stderr.write('[pick_daemon] DEBUG: before _get_grasp_detector...\n')
                sys.stderr.flush()
                detector = _get_grasp_detector()
                sys.stderr.write('[pick_daemon] DEBUG: detector ready\n')
                sys.stderr.flush()
                grasp_ok = False
                ty_use = ty  # may get shifted on retry
                # keep-closed handoff: attempt-1 capture evidence (close-block,
                # contact, pressure delta, pad-band or insert-press) keeps the hand
                # closed into attempt 2. opening on the verdict would drop a possibly
                # held object.
                _attempt_keep_closed = False

                for attempt in range(_upright_attempts() if _hl is None else _hl['n']):
                    detector.reset()
                    # wp4 insert-press evidence (per-attempt; upright wp4 guarded stop sets it)
                    _insert_pressed = False

                    if attempt > 0 and _hl is not None:
                        # retry line and the attempt's aim. a keep-closed re-press
                        # uses the capturing aim +2 cm forward; a fresh approach is
                        # provisional and decided after h-wp1 by _hl_aim.
                        if not _hl_next_attempt(_hl, attempt, _attempt_keep_closed):
                            break
                        tx, ty_use = _hl['aim']
                        if not _attempt_keep_closed:
                            rhand.open()
                            time.sleep(0.3)
                    elif attempt > 0:
                        sys.stderr.write(
                            f'[pick_daemon] Retry attempt {attempt+1}/{_upright_attempts()}: '
                            f'shifting grasp +2cm forward\n')
                        sys.stderr.flush()
                        # keep-closed retry: the hand stayed closed out of attempt 1,
                        # so no re-open, no open-verify (a closed hand must not fail
                        # the open check) and no waypoint re-approach that would drag
                        # the captured object back through wp1-wp3. the branch below
                        # does one guarded re-press.
                        if not _attempt_keep_closed:
                            rhand.open()
                            time.sleep(0.3)
                        ty_use = ty + 0.02  # shift forward

                    # open the hand. the keep-closed retry skips open and settle
                    # because the hand is closed on purpose.
                    sys.stderr.write('[pick_daemon] DEBUG: opening hand...\n')
                    sys.stderr.flush()
                    if not _attempt_keep_closed:
                        rhand.open()
                    sys.stderr.write('[pick_daemon] DEBUG: hand open done, sleeping...\n')
                    sys.stderr.flush()
                    if not _attempt_keep_closed:
                        time.sleep(0.3)
                    sys.stderr.write('[pick_daemon] DEBUG: sleep done\n')
                    sys.stderr.flush()

                    # check the hand is actually open before inserting: a closed
                    # hand inserts as a fist, pinches, or pushes. the keep-closed
                    # retry skips this, since a closed hand must not fail the open
                    # check.
                    if not _attempt_keep_closed:
                        _hand_open_ok, _hand_open_reason = _verify_hand_open(
                            rhand, 'right', R_HAND_OPEN)
                        if not _hand_open_ok:
                            sys.stderr.write(
                                f'[pick_daemon] hand right did not verify open '
                                f'({_hand_open_reason}) — aborting insert\n')
                            sys.stderr.flush()
                            sys.stdout.write('ERROR:hand not verified open\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        sys.stderr.write('[pick_daemon] hand right verified open\n')
                        sys.stderr.flush()

                    # fresh tactile baseline per attempt. the raw 100000 threshold
                    # sits below the open-hand rest floor (~105600), so the score
                    # uses a delta over that floor; an absolute threshold would
                    # report tactile on every air close. a flaky read (ok:false, DDS
                    # gap, max=0.0) must never arm 0.0: the detector treats 0.0/None
                    # as unavailable, so reuse the last good floor.
                    try:
                        _p_now = float(rhand.pressure('right')
                                       .get('right', {}).get('max', 0.0))
                    except Exception as _e:
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline '
                            f'(right) read failed: {_e}\n')
                        sys.stderr.flush()
                        _p_now = 0.0
                    if _p_now > 0.0:
                        _TACTILE_BASELINE_LAST_GOOD[0] = _p_now
                        detector.set_tactile_baseline(_p_now)
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (right): '
                            f'{_p_now:.0f}\n')
                        sys.stderr.flush()
                    elif _TACTILE_BASELINE_LAST_GOOD[0] is not None:
                        detector.set_tactile_baseline(_TACTILE_BASELINE_LAST_GOOD[0])
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (right): '
                            f'read={_p_now:.0f} (ok:false/DDS gap) — reusing '
                            f'last known good '
                            f'{_TACTILE_BASELINE_LAST_GOOD[0]:.0f}\n')
                        sys.stderr.flush()
                    else:
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (right): '
                            f'UNAVAILABLE ({_p_now:.0f}) — tactile '
                            f'verification skipped; failures will be loud, '
                            f'never false success\n')
                        sys.stderr.flush()

                    # ── waypoint approach: ascend → approach-xy → descend ──
                    sys.stderr.write('[pick_daemon] DEBUG: reading current arm q...\n')
                    sys.stderr.flush()
                    try:
                        _cur_q = arm.get_current_dual_arm_q().copy()
                        sys.stderr.write(
                            f'[pick_daemon] DEBUG: got arm q, shape={_cur_q.shape}\n')
                        sys.stderr.flush()
                    except Exception as e:
                        sys.stderr.write(
                            f'[pick_daemon] DEBUG: get_current_dual_arm_q FAILED: {e}\n')
                        sys.stderr.flush()
                        _cur_q = None

                    if _cur_q is not None:
                        _cur_wrist = get_wrist_pose(arm_ik, _cur_q, side='right')
                    else:
                        _cur_wrist = None
                    # grasp height is the interface tz plus GRASP_Z_UP_BIAS (+1 cm)
                    # and the per-attempt tilt. the interface tz already includes
                    # GRASP_Z_OFFSET. a grip-centre park that lands the wrist low
                    # puts it outside the reach shell at approach depth and the
                    # solver bakes the miss into the wrist, so keep this to one knob
                    # rather than stacked offsets.
                    _grasp_z = tz + GRASP_Z_UP_BIAS + _upright_tilt_dz(attempt)
                    # this attempt's live self-collision refusal (None means none)
                    _sc_rec = None

                    if attempt > 0 and _attempt_keep_closed:
                        # keep-closed re-press. the hand is closed from attempt 1,
                        # so do not re-approach and drag a captured object back
                        # through wp1-wp3. instead one guarded re-press from the grasp
                        # pose: parked target rebuilt at ty_use (ty+0.02 on retry),
                        # fresh raw solve (filter cleared), position-error backstop
                        # only, no stop_fn since a closed hand must not trip the pad
                        # probe. then flow into the shared close and verdict below.
                        _solver_fresh(arm_ik, 'keep-repress: ')
                        _kp_idle_wrist = get_wrist_pose(
                            arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                        _kp_idle = (pin.SE3(np.eye(3), _kp_idle_wrist).homogeneous
                                    if _kp_idle_wrist is not None else L_NEUTRAL)
                        if orientation == 'horizontal':
                            R_kp = make_wrist_target_middle_down(
                                tx, ty_use, tz,
                                fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        else:
                            R_kp = make_wrist_target_upright_v4(
                                tx, ty_use, tz + GRASP_Z_UP_BIAS, +1,
                                fwd_offset=-wp4_park_offset(x_off))   # parked target
                        with _SuppressStdout():
                            _kp_q, _kp_tauff = arm_ik.solve_ik(
                                _kp_idle, R_kp,
                                current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                current_lr_arm_motor_dq=np.zeros(14))
                        if _kp_q is None:
                            sys.stdout.write('ERROR:IK failed for keep-closed re-press\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        _kp_q = freeze_idle_arm(arm, _kp_q, 'right')
                        # live guard: the pinned re-press pose, before anything is
                        # commanded
                        _sc_rec = _sc_live_check(arm, _kp_q, 'keep-repress', 'pick', tx, ty)
                        if _sc_rec is not None:
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        sys.stderr.write(
                            '[pick_daemon] keep-closed re-press: moving '
                            '(guarded, position-error backstop only)\n')
                        sys.stderr.flush()
                        _kp_c, _kp_r = move_arm_smooth_guarded(
                            arm, _kp_q, _kp_tauff,
                            PALM_CONTACT_MONITOR_JOINTS_RIGHT,
                            PALM_CONTACT_POSITION_THRESHOLD,
                            PALM_CONTACT_CONSECUTIVE_STEPS,
                            zf_label='keep-closed-repress')
                        if not _kp_c:
                            sys.stderr.write(
                                f'[pick_daemon] keep-closed re-press guarded '
                                f'stop: {_kp_r}\n')
                            sys.stderr.flush()
                    elif orientation == UPRIGHT_GRASP:
                        # ── upright approach: ascend, behind, descend, insert ──
                        # identity orientation at every waypoint, deviation add-on
                        # off unless DEVIATION_ENABLED, positions at lateral tx,
                        # depth ty_use, height tz. the insert guard and retract are
                        # unchanged. the wall triad below stays diagnostic only.
                        use_oriented = False  # force identity: surface normals are unreliable for upright objects
                        # retract trail holds this approach only. attempt 2 starts
                        # at attempt 1's grasp pose, and replaying attempt 1's
                        # entries would drive the arm back down to that pose before
                        # READY.
                        
                        _trail_reset()

                        # ascend only, never down
                        _min_approach_z = tz + APPROACH_HEIGHT_OFFSET
                        if _cur_wrist is not None:
                            _approach_z = max(_cur_wrist[2], _min_approach_z)
                            if _approach_z > _cur_wrist[2] + 0.02:
                                sys.stderr.write(
                                    f'[pick_daemon] Phase 1a: ascend to z={_approach_z:.3f} '
                                    f'(current={_cur_wrist[2]:.3f})\n')
                                sys.stderr.flush()
                                R_wp1 = pin.SE3(np.eye(3), np.array([
                                    _cur_wrist[0], _cur_wrist[1], _approach_z])).homogeneous
                                _solver_fresh(arm_ik, 'wp1-ascend: ')
                                if not solve_and_move(arm, arm_ik, R_wp1, "wp1-ascend: "):
                                    sys.stdout.write('ERROR:IK failed for ascend\n')
                                    sys.stdout.flush()
                                    _reply_sent = True
                                    break
                                _trail_note(arm, 'wp1')  # trail
                            else:
                                sys.stderr.write(
                                    f'[pick_daemon] Phase 1a: SKIP ascend '
                                    f'(already at z={_cur_wrist[2]:.3f} >= {_min_approach_z:.3f})\n')
                                sys.stderr.flush()
                                _approach_z = _cur_wrist[2]
                        else:
                            _approach_z = _min_approach_z
                            sys.stderr.write(
                                '[pick_daemon] Phase 1a: SKIP ascend '
                                '(FK unavailable), going directly to approach\n')
                            sys.stderr.flush()

                        # behind target, at height
                        sys.stderr.write(
                            '[pick_daemon] Phase 1b: approach xy at height\n')
                        sys.stderr.flush()
                        # the forward standoff is applied here (ty_use +
                        # APPROACH_FORWARD_OFFSET): wp2 and wp3 park behind the
                        # contact depth, approach verifies inside the reachable
                        # shell, and the guarded wp4 push walks the final 8 cm. the
                        # guard, not the solve, is contact safety. lateral stays tx;
                        # a lateral offset outside the right-arm reach at this depth
                        # gets baked into the wrist by the solver. the triad below
                        # stays diagnostic only.
                        if x_off is not None and abs(x_off) > 1e-4:
                            _r_obj = -x_off - H_X_OFF_RADIUS_INV
                        else:
                            _r_obj = WALL_GRASP_R_DEFAULT
                        _r_obj = float(np.clip(_r_obj, H_RADIUS_MIN, H_RADIUS_MAX))
                        _x_wall = tx - _r_obj
                        _x_standoff = _x_wall - WALL_GRASP_STANDOFF
                        _x_contact = _x_wall - WALL_GRASP_PAD_GAP
                        sys.stderr.write(
                            f'[pick_daemon] wall grip (right) DIAG: r={_r_obj:.3f} '
                            f'wall_x={_x_wall:.3f} standoff_x={_x_standoff:.3f} '
                            f'contact_x={_x_contact:.3f} '
                            f'deviation wp3={_deviation_ramped(1):.3f} '
                            f'wp4={_deviation_ramped(2):.3f} rad '
                            f'(max {DEV_MAX:.3f}) — v4.0 ORIGINAL design '
                            f'RESTORED (identity R, lateral=tx={tx:.3f}, '
                            f'z=tz); the triad numbers are v3 diagnostics '
                            f'only\n')
                        sys.stderr.flush()
                        R_wp2 = make_wrist_target_upright_v4(
                            tx, ty_use, _approach_z, +1,
                            fwd_offset=APPROACH_FORWARD_OFFSET)
                        _solver_fresh(arm_ik, 'wp2-approach: ')
                        # live guard: the solved wp2 pose, before anything is
                        # commanded
                        _sc_rec = _sc_live_presolve(
                            arm, arm_ik, R_wp2, 'wp2', 'pick', tx, ty,
                            log_prefix='wp2-approach: ')
                        if _sc_rec is not None:
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        if not solve_and_move(arm, arm_ik, R_wp2, "wp2-approach: "):
                            sys.stdout.write('ERROR:IK failed for approach xy\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        _trail_note(arm, 'wp2')  # retract trail

                        # descend to grasp z
                        sys.stderr.write(
                            f'[pick_daemon] Phase 1c: descend to z={_grasp_z:.3f}\n')
                        sys.stderr.flush()
                        # descend straight down behind the object at the fixed
                        # forward standoff (ty_use + APPROACH_FORWARD_OFFSET) with
                        # identity orientation. deviation is an add-on (off unless
                        # DEVIATION_ENABLED): straight at wp2, one DEV_STEP_MAX step
                        # per waypoint toward contact, frozen at contact.
                        _tilt_sign = _upright_tilt_sign(attempt)
                        sys.stderr.write(
                            f'[pick_daemon][tilt] attempt {attempt+1}: upright '
                            f'deviation sign {_tilt_sign:+.0f} '
                            f'dz={_upright_tilt_dz(attempt):+.3f} '
                            f'({"up" if _tilt_sign > 0 else ("straight" if _tilt_sign == 0 else "down")})\n')
                        sys.stderr.flush()
                        R_wp3 = make_wrist_target_upright_v4(
                            tx, ty_use, _grasp_z, +1,
                            deviation=(_tilt_sign * _deviation_ramped(1)
                                       if DEVIATION_ENABLED else 0.0),
                            fwd_offset=APPROACH_FORWARD_OFFSET)
                        _solver_fresh(arm_ik, 'wp3-descend: ')
                        # live guard: the solved wp3 pose, before anything is
                        # commanded
                        _sc_rec = _sc_live_presolve(
                            arm, arm_ik, R_wp3, 'wp3', 'pick', tx, ty,
                            log_prefix='wp3-descend: ')
                        if _sc_rec is not None:
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        if not solve_and_move(arm, arm_ik, R_wp3, "wp3-descend: "):
                            sys.stdout.write('ERROR:IK failed for descend\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        _trail_note(arm, 'wp3')  # retract trail

                        time.sleep(0.5)
                        _rim_diag(arm, arm_ik, 'wp3', R_wp3)   # measured vs commanded

                        # ── phase 4.2: tracking stability check ──
                        servo_status = 'stable'
                        try:
                            with open('/tmp/pick_servo_status', 'r') as f:
                                servo_status = f.read().strip()
                        except Exception:
                            pass  # file may not exist yet: assume stable

                        if servo_status == 'lost':
                            sys.stderr.write('[pick_daemon] WARNING: tracking lost, but continuing — guarded insert will handle safety\n')
                            sys.stderr.flush()
                            # carry on to wp4: the palm contact guard handles safety

                        if servo_status == 'shifted':
                            sys.stderr.write(
                                '[pick_daemon] Tracking shifted, '
                                'recomputing grasp from latest detection\n')
                            sys.stderr.flush()
                            try:
                                with open('/tmp/pick_latest_detection.json', 'r') as f:
                                    latest = json.load(f)
                                tx = latest.get('x', tx)
                                ty_use = latest.get('y', ty_use)
                                tz = latest.get('z', tz) + GRASP_Z_OFFSET
                                # re-latch floor at new tz
                                _latch_table_floor(tz, height_m, orientation)
                                if use_oriented and 'nx' in latest:
                                    nx = latest['nx']
                                    ny = latest['ny']
                                    nz = latest['nz']
                                if 'height_m' in latest:
                                    _h = float(latest['height_m'])
                                    if 0.0 < _h <= HEIGHT_M_MAX:
                                        height_m = _h
                                    else:
                                        height_m = None
                                sys.stderr.write(
                                    f'[pick_daemon] Updated target: '
                                    f'x={tx:.3f} y={ty_use:.3f} z={tz:.3f}'
                                    + (f' height_m={height_m:.3f}'
                                       if height_m is not None
                                       else ' height_m=unknown')
                                    + '\n')
                                sys.stderr.flush()
                            except Exception as e:
                                sys.stderr.write(
                                    f'[pick_daemon] WARNING: '
                                    f'failed to read latest detection: {e}\n')
                                sys.stderr.flush()

                        # wp4: guarded descend to the grip from above
                        sys.stderr.write('[pick_daemon] Phase 2: descend to grasp (guarded)\n')
                        sys.stderr.flush()
                        # wp4 = guarded insert at full contact depth: lateral tx,
                        # depth ty_use, height tz, identity orientation. in the
                        # servo-shifted branch _grasp_z=tz so the insert follows the
                        # retarget. the insert guard keeps it safe; wp4 FK logs where
                        # the palm landed (diagnostic only). wp4 is the last ramping
                        # deviation waypoint, with the full target frozen from
                        # contact plus retract.
                        _grasp_z = (tz + GRASP_Z_UP_BIAS   # +1 cm
                                    + _upright_tilt_dz(attempt))
                        _dev_tgt = (_upright_tilt_sign(attempt)
                                    * _deviation_ramped(2)
                                    if DEVIATION_ENABLED else 0.0)
                        # upright wp4 park. the URDF grip zone rides +17..+69 mm
                        # forward of R_ee (mid +43 mm), so a wrist at the centroid
                        # puts the grip zone ~45 mm past the object and the solver
                        # retracts backwards. the commanded target is parked at
                        # fwd_offset=-HAND_GRIP_CENTER_OFFSET: wrist 4.5 cm behind
                        # the centroid, grip zone on it. REFUSE/CEILING residual is
                        # measured against the unparked R_grasp_centroid; a parked
                        # residual would loosen the refusal by ~4.5 cm.
                        # radius-based park.
                        _wp4_park = wp4_park_offset(x_off)
                        R_grasp = make_wrist_target_upright_v4(
                            tx, ty_use, _grasp_z, +1, deviation=_dev_tgt,
                            fwd_offset=-_wp4_park)
                        if x_off is None:
                            R_grasp_centroid = make_wrist_target_upright_v4(
                                tx, ty_use, _grasp_z, +1, deviation=_dev_tgt)
                        else:   # score target shifted by the extra park only
                            R_grasp_centroid = make_wrist_target_upright_v4(
                                tx, ty_use, _grasp_z, +1, deviation=_dev_tgt,
                                fwd_offset=wp4_score_offset(x_off))
                        if x_off is not None:
                            # log the park offset when x_off is provided
                            try:
                                x_off_val = float(x_off)
                                if x_off_val != 0 and math.isfinite(x_off_val):
                                    r_eff = -x_off_val - H_X_OFF_RADIUS_INV
                                    # log a clamped park as park_clamped=<requested>-><used>.
                                    sys.stderr.write(
                                        f'[pick_daemon] wp4 park: {_wp4_park:.3f} m '
                                        f'behind centroid (x_off={x_off_val:.4f}, '
                                        f'r_eff={r_eff:.3f})'
                                        f'{wp4_park_clamp_note(x_off)}\n')
                                    sys.stderr.flush()
                            except (ValueError, TypeError):
                                pass
                        else:
                            # log the default park for diagnostics
                            sys.stderr.write(
                                '[pick_daemon] v5.2 wp4 park: '
                                f'fwd_offset=-HAND_GRIP_CENTER_OFFSET '
                                f'({HAND_GRIP_CENTER_OFFSET*100:.2f} cm behind '
                                f'centroid — grip zone centred)\n')
                            sys.stderr.flush()

                        # pin idle arm
                        _wp4_idle_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                        _wp4_idle_target = (pin.SE3(np.eye(3), _wp4_idle_wrist).homogeneous
                                            if _wp4_idle_wrist is not None else L_NEUTRAL)
                        # raw solve. a soft-fail best-effort command would polish the
                        # pose with wrist GN and freeze the deviation; at the
                        # forward-reach shell the GN chases an unreachable deviated
                        # orientation and tilts the hand. raw solve_ik commands no GN
                        # chase, so the hand arrives flat and stops once. solve_ik
                        # never returns None on a bad solve; it returns the warm-q
                        # blend, and the guarded move plus contact stop is the safety.
                        _solver_fresh(arm_ik, 'wp4-insert: ')
                        with _SuppressStdout():
                            sol_q, sol_tauff = arm_ik.solve_ik(
                                _wp4_idle_target, R_grasp,
                                current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                current_lr_arm_motor_dq=np.zeros(14))
                        if sol_q is None:
                            sys.stdout.write('ERROR:IK failed for insert\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break

                        # reach-ceiling gate. the residual is the forward-reach
                        # shortfall: 0.068-0.092 m for an in-reach grasp against
                        # 0.17-0.32 m truly out. the threshold sits above
                        # SOLVE_VERIFY_POS_TOL because a reachable grasp can still
                        # show a residual. the guarded insert and middle-finger stop
                        # are the designed safety and the closure's pressure-delta
                        # evidence adjudicates rim against palm downstream, so a short
                        # residual is reported loudly and the pick continues.
                        # REACH_REFUSE_POS is tunable via OPENBOT_REACH_REFUSE_POS
                        # (0.03 < v <= 0.50; default 0.15).
                        _gpe, _goe, _gRe, _gde = verify_ik_target(
                            arm_ik, sol_q, 'right', R_grasp_centroid)
                        if _gpe is not None and _gpe > SOLVE_VERIFY_POS_TOL:
                            if _gpe > REACH_REFUSE_POS:
                                # a diagnostic-only gate is too permissive: an
                                # insert that solves far short closes on air and then
                                # fakes a grip through the promote gates. an
                                # unreachable insert is a refused pick.
                                sys.stderr.write(
                                    f'OUT OF REACH: solved insert pose '
                                    f'residual {_gpe:.3f} m > '
                                    f'{REACH_REFUSE_POS:.3f} m at grasp '
                                    f'z={_grasp_z:.3f} m — the arm cannot '
                                    f'reach the commanded insert pose; the '
                                    f'robot is standing too far from the '
                                    f'object.  Pick REFUSED; move closer and '
                                    f'retry.  If the miss is predominantly '
                                    f'lateral, check the known T1.12 '
                                    f'lateral-sign-mirror item.\n')
                                sys.stderr.flush()
                                sys.stdout.write(f'ERROR:reach_ceiling residual={_gpe:.3f}m\n')
                                sys.stdout.flush()
                                _reply_sent = True
                                break
                            sys.stderr.write(
                                f'REACH_CEILING(diag): solved insert pose '
                                f'residual {_gpe:.3f} m > '
                                f'{SOLVE_VERIFY_POS_TOL:.3f} m at grasp '
                                f'z={_grasp_z:.3f} m — best-effort insert '
                                f'CONTINUES (guarded insert + stop is the '
                                f'safety).  The residual is vs the UNPARKED '
                                f'object-centroid target: the parked insert '
                                f'rides -HAND_GRIP_CENTER_OFFSET '
                                f'({HAND_GRIP_CENTER_OFFSET*100:.2f} cm) so the '
                                f'grip zone itself is '
                                f'{(abs(_gpe)-HAND_GRIP_CENTER_OFFSET)*100:.0f} '
                                f'cm short of the centroid at this reach cap; '
                                f'the real fix is the robot standing closer.  '
                                f'If the miss is predominantly lateral, check '
                                f'the known T1.12 lateral-sign-mirror item.\n')
                            sys.stderr.flush()

                        # freeze the idle (left) arm at its current q to stop
                        # nullspace drift
                        sol_q = freeze_idle_arm(arm, sol_q, 'right')
                        # live guard: the pinned insert pose, before anything is
                        # commanded
                        _sc_rec = _sc_live_check(arm, sol_q, 'wp4', 'pick', tx, ty)
                        if _sc_rec is not None:
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        sys.stderr.write('[pick_daemon] wp4-insert (guarded): moving...\n')
                        sys.stderr.flush()

                        # insert guard: position-error backstop on joints [7,10]
                        # (PALM_CONTACT_MONITOR_JOINTS_RIGHT) plus the middle-finger
                        # stop only. pressure-stop-first and the wall-push monitor
                        # [7,8,9,10] remain only in the horizontal flows.

                        def _middle_check():
                            return 'middle_finger' if detector.check_middle_finger_contact() else None

                        # wp4 guard uses the automatic duration
                        # (max_delta/MOVE_SPEED), not a fixed 1.5 s.
                        completed, reason = move_arm_smooth_guarded(
                            arm, sol_q, sol_tauff,
                            PALM_CONTACT_MONITOR_JOINTS_RIGHT,
                            PALM_CONTACT_POSITION_THRESHOLD,
                            PALM_CONTACT_CONSECUTIVE_STEPS,
                            stop_fn=_middle_check, zf_label='wp4-insert')
                        # wp4 insert-press evidence: a guard stop via position_error
                        # or the middle-finger probe means the palm or fingers
                        # pressed something before the planned insert target. this is
                        # the vertical keep-evidence term; the z-band is vacuous
                        # vertically since design z = tz+GRASP_Z_UP_BIAS <= 0.035.
                        _insert_pressed = (not completed) and (
                            reason is not None and (
                                reason.startswith('position_error')
                                or reason == 'middle_finger'))
                        if not completed:
                            sys.stderr.write(
                                f'[pick_daemon] wp4-insert guarded stop: {reason} '
                                f'(retracting {RETRACT_AFTER_CONTACT:.2f}m to relieve pressure)\n')
                            sys.stderr.flush()
                            # active retract: move backward to take contact pressure
                            # off instead of freezing, which lets the G1 controller
                            # drift forward
                            _contact_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                            if _contact_wrist is not None:
                                # identity-base retract (ty_use +
                                # RETRACT_AFTER_CONTACT on top of
                                # fwd_offset=approach_offset), with the deviation
                                # frozen at wp4.
                                R_retract = make_wrist_target_upright_v4(
                                    tx, ty_use + RETRACT_AFTER_CONTACT,
                                    _grasp_z, +1, deviation=_dev_tgt,
                                    fwd_offset=approach_offset)
                                # raw solve.
                                with _SuppressStdout():
                                    _retract_q, _retract_tauff = arm_ik.solve_ik(
                                        _wp4_idle_target, R_retract,
                                        current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                        current_lr_arm_motor_dq=np.zeros(14))
                                if _retract_q is not None:
                                    # idle (left) arm untouched
                                    _retract_q = freeze_idle_arm(arm, _retract_q, 'right')
                                    # live guard: the contact retract, before
                                    # anything is commanded; the failure block
                                    # relaxes the stall and parks.
                                    _sc_rec = _sc_live_check(arm, _retract_q, 'wp4-retract', 'pick', tx, ty)
                                    if _sc_rec is not None:
                                        sys.stdout.write(_sc_line(_sc_rec))
                                        sys.stdout.flush()
                                        _reply_sent = True
                                        break
                                    move_arm_smooth(arm, _retract_q, _retract_tauff, duration=0.3, zf_legacy=True)
                                    _trail_note(arm, 'wp4-retract')  # retract trail
                            time.sleep(INSERT_GUARD_HOLD_TIME)

                        time.sleep(0.3)

                        # ── wrist orientation FK verification ──
                        # FK-log-only check with no control change: log the
                        # achieved palm normal against the commanded identity
                        # target. flexion dominance is a claw-frame property, so
                        # the check is diagnostic only.
                        _rim_diag(arm, arm_ik, 'wp4', R_grasp)   # measured vs commanded
                        _q_now = arm.get_current_dual_arm_q().copy()
                        _pe, _oe, _Re, _de = verify_ik_target(arm_ik, _q_now, 'right', R_grasp)
                        if _Re is not None:
                            _palm_cmd = np.array(R_grasp[:3, 1], dtype=np.float64)  # right: the physical palm = +Y_local
                            _palm_act = np.array(_Re[:, 1], dtype=np.float64)
                            _wq_cmd = sol_q[RIGHT_WRIST_Q_IDX]
                            _wq_act = _q_now[RIGHT_WRIST_Q_IDX]
                            _wq_d = _wq_act - _wq_cmd
                            sys.stderr.write(
                                f'[pick_daemon] wp4-FK orientation check: '
                                f'palm normal cmd=({_palm_cmd[0]:.2f},{_palm_cmd[1]:.2f},{_palm_cmd[2]:.2f}) '
                                f'act=({_palm_act[0]:.2f},{_palm_act[1]:.2f},{_palm_act[2]:.2f}) '
                                f'ori err={_oe:.3f} rad (tol {SOLVE_VERIFY_ORI_TOL:.2f}) '
                                f'det={_de:+.3f}; wrist joints [11,12,13] '
                                f'cmd=({_wq_cmd[0]:.3f},{_wq_cmd[1]:.3f},{_wq_cmd[2]:.3f}) '
                                f'act=({_wq_act[0]:.3f},{_wq_act[1]:.3f},{_wq_act[2]:.3f}) '
                                f'delta=({_wq_d[0]:+.3f},{_wq_d[1]:+.3f},{_wq_d[2]:+.3f})\n')
                            sys.stderr.flush()
                            # body-axis decomposition against the straight frame
                            # (deviation=0): pitch is about the fingers axis and has
                            # to be ~0 at contact, deviation is the ulnar roll,
                            # flexion is the in/out hinge. log only, loud on a pitch
                            # violation. the target is printed signed (s_dev * delta;
                            # right s_dev = DEV_ROT_SIGN = -1), so a perfect
                            # execution reads deviation == target; an unsigned target
                            # would show -0.524 next to +0.524 and hide a
                            # wrong-signed failure.
                            _R_straight = make_wrist_target(
                                tx, ty_use, _grasp_z)
                            _pitch, _dev_, _flex = wrist_rotation_decomposition(
                                arm_ik, _q_now, 'right', _R_straight)
                            if _pitch is not None:
                                sys.stderr.write(
                                    f'[pick_daemon] wp4 body-axis decomp: '
                                    f'pitch={_pitch:+.3f} deviation={_dev_:+.3f} '
                                    f'flexion={_flex:+.3f} rad (dev target '
                                    f'{DEV_ROT_SIGN*_dev_tgt:+.3f} '
                                    f'signed, pitch DIAG tol '
                                    f'{PITCH_FROZEN_TOL:.2f})\n')
                                sys.stderr.flush()
                                if abs(_pitch) > PITCH_FROZEN_TOL:
                                    sys.stderr.write(
                                        '[pick_daemon] WARNING: NO-PITCH rule '
                                        f'VIOLATED at wall contact: '
                                        f'|pitch|={abs(_pitch):.3f} > '
                                        f'{PITCH_FROZEN_TOL:.2f} rad (user rule '
                                        f'2026-08-30 — wrist up/down must not '
                                        f'move during upright contact)\n')
                                    sys.stderr.flush()
                        else:
                            sys.stderr.write(
                                '[pick_daemon] wp4-FK orientation check skipped (FK failed)\n')
                            sys.stderr.flush()

                    else:  # orientation == 'horizontal'
                        # ── horizontal (top-down) approach: ascend, above,
                        # descend onto ──
                        # retract trail holds this approach only (see the upright
                        # branch above).
                        _trail_reset()
                        if attempt == 0:
                            _hrule_h_cmd = height_m   # the command's height ('shifted' re-reads height_m)

                        # ascend if needed
                        _h_min_approach_z = tz + 0.20
                        if _cur_wrist is not None:
                            if _cur_wrist[2] < _h_min_approach_z:
                                _h_approach_z = _h_min_approach_z
                                sys.stderr.write(
                                    f'[pick_daemon] Horizontal wp1: ascend to z={_h_approach_z:.3f} '
                                    f'(current={_cur_wrist[2]:.3f}, bottle_z={tz:.3f})\n')
                                sys.stderr.flush()
                                R_h_wp1 = pin.SE3(np.eye(3), np.array([
                                    _cur_wrist[0], _cur_wrist[1], _h_approach_z])).homogeneous
                                # raw swing: no verified GN polish on the horizontal
                                # swing, which tilted the palm while turning the
                                # wrist from far inwards.
                                if not solve_and_move(arm, arm_ik, R_h_wp1, "h-wp1-ascend: ", raw=True):
                                    sys.stdout.write('ERROR:IK failed for horizontal ascend\n')
                                    sys.stdout.flush()
                                    _reply_sent = True
                                    break
                                _trail_note(arm, 'h-wp1')  # retract trail
                            else:
                                _h_approach_z = _cur_wrist[2]
                                sys.stderr.write(
                                    f'[pick_daemon] Horizontal wp1: SKIP ascend '
                                    f'(already at z={_cur_wrist[2]:.3f} >= {_h_min_approach_z:.3f})\n')
                                sys.stderr.flush()
                        else:
                            _h_approach_z = _h_min_approach_z
                            sys.stderr.write(
                                '[pick_daemon] Horizontal wp1: SKIP ascend '
                                '(FK unavailable), going directly to approach\n')
                            sys.stderr.flush()

                        if _hl is not None:
                            # attempt 1 uses the command's aim; 2-3 use the
                            # re-detected base (clean rows taken with the hand up
                            # here) plus the rung, reach-prechecked. False stops the
                            # ladder.
                            if not _hl_aim(_hl, attempt, arm_ik, tz, x_off):
                                break
                            tx, ty_use = _hl['aim']

                        # wp2: directly above the object's grip centre. the grip
                        # zone (+17..+69 mm forward of R_ee) sits ~45 mm past the
                        # centroid at zero offset, so the wrist parks
                        # HAND_GRIP_CENTER_OFFSET behind it.
                        sys.stderr.write(
                            '[pick_daemon] Horizontal wp2: approach xy above bottle\n')
                        sys.stderr.flush()
                        R_h_wp2 = make_wrist_target_middle_down(
                            tx, ty_use, _h_approach_z,
                            fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        # raw swing: the wrist rotates palm-down during the
                        # approach, one move straight to the position above the
                        # object.
                        _solver_fresh(arm_ik, 'h-wp2-approach: ')
                        # live guard: the solved h-wp2 pose, before anything is
                        # commanded
                        _sc_rec = _sc_live_presolve(
                            arm, arm_ik, R_h_wp2, 'h-wp2', 'pick', tx, ty,
                            raw=True, log_prefix='h-wp2-approach: ')
                        if _sc_rec is not None:
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        if not solve_and_move(arm, arm_ik, R_h_wp2, "h-wp2-approach: ", raw=True):
                            sys.stdout.write('ERROR:IK failed for horizontal approach xy\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        _trail_note(arm, 'h-wp2')  # retract trail

                        # ── tracking stability check (between wp2 and wp3) ──
                        servo_status = 'stable'
                        try:
                            with open('/tmp/pick_servo_status', 'r') as f:
                                servo_status = f.read().strip()
                        except Exception:
                            pass

                        if _hl is not None and servo_status == 'shifted':
                            # the ladder owns the aim; the block below would
                            # overwrite ty_use with an unlabelled, unstamped row.
                            _hl_log('tracking shifted ignored: the ladder owns the '
                                    f'aim (x={tx:.3f} y={ty_use:.3f})')
                            servo_status = 'shifted-ignored'

                        if servo_status == 'lost':
                            sys.stderr.write('[pick_daemon] WARNING: tracking lost, but continuing — guarded descent will handle safety\n')
                            sys.stderr.flush()

                        if servo_status == 'shifted':
                            sys.stderr.write(
                                '[pick_daemon] Tracking shifted, '
                                'recomputing grasp from latest detection\n')
                            sys.stderr.flush()
                            try:
                                with open('/tmp/pick_latest_detection.json', 'r') as f:
                                    latest = json.load(f)
                                tx = latest.get('x', tx)
                                ty_use = latest.get('y', ty_use)
                                tz = latest.get('z', tz) + GRASP_Z_OFFSET
                                # re-latch floor at new tz
                                _latch_table_floor(tz, height_m, orientation)
                                if use_oriented and 'nx' in latest:
                                    nx = latest['nx']
                                    ny = latest['ny']
                                    nz = latest['nz']
                                if 'height_m' in latest:
                                    _h = float(latest['height_m'])
                                    if 0.0 < _h <= HEIGHT_M_MAX:
                                        height_m = _h
                                    else:
                                        height_m = None
                                sys.stderr.write(
                                    f'[pick_daemon] Updated target: '
                                    f'x={tx:.3f} y={ty_use:.3f} z={tz:.3f}'
                                    + (f' height_m={height_m:.3f}'
                                       if height_m is not None
                                       else ' height_m=unknown')
                                    + '\n')
                                sys.stderr.flush()
                            except Exception as e:
                                sys.stderr.write(
                                    f'[pick_daemon] WARNING: '
                                    f'failed to read latest detection: {e}\n')
                                sys.stderr.flush()

                        # descend onto the object (guarded): straight to the
                        # interface target at y=ty_use, z=tz. the stop is middle
                        # pad (pkg 5) only; a fingertip graze (pkg 8) does not end
                        # the drop while the palm is still in the air. a lag/block
                        # stop (position_error) or a completed ramp above tz gets one
                        # resumed guarded push (single _h_resumed slot, max 2
                        # passes).
                        sys.stderr.write(
                            f'[pick_daemon] Horizontal wp3: descend onto bottle to z={tz:.3f} (guarded, pad-only stop)\n')
                        sys.stderr.flush()
                        _h_tz = _h_rule_tz(tz, _hrule_h_cmd)   # == tz unless the rule applies
                        R_h_grasp = make_wrist_target_middle_down(
                            tx, ty_use, _h_tz,
                            fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        if _hl is not None:
                            # attempts 2-3 descend dz below attempt 1's target; only
                            # the target moves, so where the z-floor clamp binds the
                            # lowering is a no-op (logged after the descent).
                            R_h_grasp = _hl_descent_target(_hl, attempt, _h_tz, R_h_grasp)

                        # pin the idle (left) arm for the retract solve below;
                        # each descent pass re-pins it.
                        _h_wp3_idle_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                        _h_wp3_idle_target = (pin.SE3(np.eye(3), _h_wp3_idle_wrist).homogeneous
                                              if _h_wp3_idle_wrist is not None else L_NEUTRAL)

                        def _h_middle_check():
                            return 'middle_finger' if detector.check_middle_finger_contact(pad_only=True) else None

                        def _h_descend_once(duration):
                            nonlocal _sc_rec  # the h-wp3 live guard's record
                            # fresh idle (left) pin and warm start from the current q
                            # each pass; raw solve, no wrist-GN chase.
                            _i = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                            _it = (pin.SE3(np.eye(3), _i).homogeneous
                                   if _i is not None else L_NEUTRAL)
                            with _SuppressStdout():
                                _s, _t = arm_ik.solve_ik(
                                    _it, R_h_grasp,
                                    current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                    current_lr_arm_motor_dq=np.zeros(14))
                            if _s is None:
                                return None, False, 'ik_fail'
                            _s = freeze_idle_arm(arm, _s, 'right')
                            # live guard: refuse this pass before the guarded move;
                            # the caller loop sees 'self_collision' and replies once.
                            _sc_rec = _sc_live_check(arm, _s, 'h-wp3', 'pick', tx, ty)
                            if _sc_rec is not None:
                                return _s, False, 'self_collision'
                            sys.stderr.write('[pick_daemon] h-wp3-descent (guarded): moving...\n')
                            sys.stderr.flush()
                            _c, _r = move_arm_smooth_guarded(
                                arm, _s, _t,
                                PALM_CONTACT_MONITOR_JOINTS_RIGHT,
                                PALM_CONTACT_POSITION_THRESHOLD,
                                PALM_CONTACT_CONSECUTIVE_STEPS,
                                stop_fn=_h_middle_check,
                                duration=duration, zf_label='h-wp3-descent')
                            return _s, _c, _r

                        _solver_fresh(arm_ik, 'h-wp3-descend: ')
                        # explicit descend pass loop: a reach stall is not contact.
                        # per pass: descend (DESCENT_DURATION on pass 0, then none),
                        # settle 0.3 s, read wrist z, compute gap = wrist_z - tz.
                        #   not completed + position_error + gap > STALL_GAP: reach
                        #   stall, not contact, so no retract and no immediate close;
                        #   take the not-bottomed path (push if budget and progress
                        #   remain, else diagnostic close at the current height, with
                        #   closure evidence deciding).
                        #   not completed + anything else (middle-finger stop,
                        #   position_error in band): true contact, retract block below.
                        #   completed: not-bottomed if gap > DEPTH_TOL.
                        # not-bottomed: gap <= DEPTH_TOL (or FK unreadable) ends the
                        # loop; _h_pushes >= H_DESCEND_MAX_PUSHES breaks with a
                        # diagnostic; progress < H_DESCEND_MIN_PROGRESS (1 cm) breaks
                        # with "no more pushes"; else push again.
                        _h_pushes = 0
                        _h_prev_gap = None
                        _h_contact = False
                        while True:
                            sol_q, completed, reason = _h_descend_once(
                                DESCENT_DURATION if _h_pushes == 0 else None)
                            if reason == 'self_collision':
                                break  # refused before moving; one reply below
                            if sol_q is None:
                                sys.stdout.write('ERROR:IK failed for horizontal descent\n')
                                sys.stdout.flush()
                                _reply_sent = True
                                break
                            time.sleep(0.3)  # settle before the depth read (post-ramp)
                            _hwz = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                            _gap = (_hwz[2] - _h_tz) if _hwz is not None else None
                            if not completed:
                                if (reason is not None and reason.startswith('position_error')
                                        and _gap is not None and _gap > H_DESCEND_STALL_GAP):
                                    sys.stderr.write(
                                        f'[pick_daemon] REACH STALL at +{_gap:.3f} m '
                                        f'({_gap*100:.1f} cm above target z={_h_tz:.3f}: {reason}) '
                                        f'— NOT contact (pads hang 5-17 mm under the wrist, '
                                        f'guard stop hit with the palm still airborne).  No '
                                        f'contact-retract, no immediate close.\n')
                                    sys.stderr.flush()
                                else:
                                    _h_contact = True
                                    break  # true contact, retract block below
                            if _gap is None or _gap <= H_DESCEND_DEPTH_TOL:
                                break  # reached the depth band (or FK unreadable)
                            if _h_pushes >= H_DESCEND_MAX_PUSHES:
                                sys.stderr.write(
                                    f'[pick_daemon] h-wp3 descend budget spent: {_h_pushes} '
                                    f'push(es), gap +{_gap:.3f} m above target — DIAG, '
                                    f'close-at-current-height; closure evidence adjudicates\n')
                                sys.stderr.flush()
                                break
                            if (_h_prev_gap is not None
                                    and (_h_prev_gap - _gap) < H_DESCEND_MIN_PROGRESS):
                                sys.stderr.write(
                                    f'[pick_daemon] h-wp3 push made less than '
                                    f'{H_DESCEND_MIN_PROGRESS:.3f} m '
                                    f'(+{_h_prev_gap:.3f} → +{_gap:.3f}) — no more pushes, '
                                    f'close-at-current-height; closure evidence adjudicates\n')
                                sys.stderr.flush()
                                break
                            _h_prev_gap = _gap
                            _h_pushes += 1
                            sys.stderr.write(
                                f'[pick_daemon] h-wp3 push {_h_pushes}: gap +{_gap:.3f} m '
                                f'(>{H_DESCEND_DEPTH_TOL:.3f}) — one more guarded push\n')
                            sys.stderr.flush()

                        if _sc_rec is not None:
                            # the h-wp3 live guard refused a descent pass, returning
                            # 'self_collision' without moving.
                            sys.stdout.write(_sc_line(_sc_rec))
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        if _hl is not None:
                            # dz: wrist z reached, lower than attempt 1's and clamped
                            _hl_descent_note(_hl, attempt, _h_tz, sol_q, arm, arm_ik)
                        _h_rule_close(arm, arm_ik, _h_tz, tz, _hrule_h_cmd)   # telemetry
                        if _h_contact:
                            sys.stderr.write(
                                f'[pick_daemon] h-wp3-descent guarded stop: {reason} '
                                f'(retracting {RETRACT_AFTER_CONTACT:.2f}m to relieve pressure)\n')
                            sys.stderr.flush()
                            # active retract: move backward to take the contact
                            # pressure off; unconditional, always pulling back
                            # RETRACT_AFTER_CONTACT at z=tz.
                            R_h_retract = make_wrist_target_middle_down(
                                tx, ty_use + RETRACT_AFTER_CONTACT, _h_tz,
                                fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                            # raw solve, no wrist-GN chase; retract channel mirror.
                            with _SuppressStdout():
                                _h_retract_q, _h_retract_tauff = arm_ik.solve_ik(
                                    _h_wp3_idle_target, R_h_retract,
                                    current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                    current_lr_arm_motor_dq=np.zeros(14))
                            if _h_retract_q is not None:
                                # idle (left) arm untouched
                                _h_retract_q = freeze_idle_arm(arm, _h_retract_q, 'right')
                                # live guard: the contact retract, before anything is commanded
                                _sc_rec = _sc_live_check(arm, _h_retract_q, 'h-wp3-retract', 'pick', tx, ty)
                                if _sc_rec is not None:
                                    sys.stdout.write(_sc_line(_sc_rec))
                                    sys.stdout.flush()
                                    _reply_sent = True
                                    break
                                move_arm_smooth(arm, _h_retract_q, _h_retract_tauff, duration=0.3, zf_legacy=True)
                                _trail_note(arm, 'h-wp3-retract')  # retract trail
                            time.sleep(INSERT_GUARD_HOLD_TIME)

                        time.sleep(0.3)

                        # FK orientation check.
                        _q_now = arm.get_current_dual_arm_q().copy()
                        _pe, _oe, _Re, _de = verify_ik_target(arm_ik, _q_now, 'right', R_h_grasp)
                        if _Re is not None:
                            _palm_cmd = np.array(R_h_grasp[:3, 1], dtype=np.float64)  # right: physical palm = +Y_local
                            _palm_act = np.array(_Re[:, 1], dtype=np.float64)
                            _wq_cmd = sol_q[RIGHT_WRIST_Q_IDX]
                            _wq_act = _q_now[RIGHT_WRIST_Q_IDX]
                            sys.stderr.write(
                                f'[pick_daemon] h-wp3-FK orientation check: '
                                f'palm normal cmd=({_palm_cmd[0]:.2f},{_palm_cmd[1]:.2f},{_palm_cmd[2]:.2f}) '
                                f'act=({_palm_act[0]:.2f},{_palm_act[1]:.2f},{_palm_act[2]:.2f}) '
                                f'ori err={_oe:.3f} rad (tol {SOLVE_VERIFY_ORI_TOL:.2f}) '
                                f'det={_de:+.3f}; wrist joints [11,12,13] '
                                f'cmd=({_wq_cmd[0]:.3f},{_wq_cmd[1]:.3f},{_wq_cmd[2]:.3f}) '
                                f'act=({_wq_act[0]:.3f},{_wq_act[1]:.3f},{_wq_act[2]:.3f})\n')
                            sys.stderr.flush()
                        else:
                            sys.stderr.write(
                                '[pick_daemon] h-wp3-FK orientation check skipped (FK failed)\n')
                            sys.stderr.flush()

                    # 4. close the hand (adaptive, stops when finger pressure passes
                    # the threshold). stop_threshold is GRIP_CLOSE_STOP_DELTA
                    # (100000), passed per call in the verb so no grip-server restart
                    # is needed. shared by the vertical and horizontal closes.
                    sys.stderr.write(
                        f'[pick_daemon] Phase 3: closing hand (bounded lead, '
                        f'eps={HAND_CLOSE_LEAD_EPS:.2f} rad)\n')
                    sys.stderr.flush()
                    _close_baseline_p = 0.0
                    try:
                        _close_baseline_p = float(
                            rhand.pressure('right').get('right', {}).get('max', 0.0))
                    except Exception:
                        pass
                    # record the wrist height at the close: wrist_z - tz <=
                    # H_PALM_TOUCH_BAND means the palm is at the object when the
                    # fingers closed.
                    _wrist_z_at_close = get_wrist_pose(
                        arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                    # the closure is capped at F_MAX_SAFE (the grip server) and the cap
                    # is honoured here: the commanded and scored pose both come from
                    # clamped_close_target. the ceiling is computed first and the
                    # close is a bounded-lead walk. a full-preset close lets the
                    # server ramp the command the whole travel ahead of the fingers,
                    # which can rail joints before contact. the ceiling is unchanged
                    # and is still the pose the closure-block score is taken against;
                    # only each write's lead is capped, so a free-air close ends at
                    # the ceiling and a real block still reads blocked=True.
                    
                    _r_close_target, _r_close_eff = clamped_close_target('right', 1.0)
                    _contact_closure = _bounded_lead_close(
                        rhand, 'right', _r_close_target)
                    if _contact_closure == 'contact':
                        # grip detected early: hold where it stopped, never tighten
                        # to the cap, since stopping early means something is already
                        # in the hand. the server's 20 Hz re-assert thread holds
                        # last_q, so logging is all this branch needs.
                        sys.stderr.write(
                            '[pick_daemon] Adaptive closure: grip detected, holding at the '
                            f'aperture reached (cap {_r_close_eff:.3f})\n')
                        sys.stderr.flush()
                    elif _contact_closure == 'no_data':
                        # this path writes nothing: a close whose lead cannot be
                        # observed is the rail path.
                        sys.stderr.write('[pick_daemon] Bounded-lead close: NO HAND STATE -- nothing written (refusing to press blind)\n')
                        sys.stderr.flush()
                    else:
                        sys.stderr.write('[pick_daemon] Bounded-lead close: walk complete -- lead held at eps (see the bounded-lead line above)\n')
                        sys.stderr.flush()

                    sys.stderr.write(
                        f'[pick_daemon] right hand last commanded q after bounded-lead close: '
                        f'{[rhand.last_q[id] for id in DEX3_JOINTS]}\n')
                    sys.stderr.flush()
                    _close_final_p = 0.0
                    try:
                        _close_final_p = float(
                            rhand.pressure('right').get('right', {}).get('max', 0.0))
                    except Exception:
                        pass
                    sys.stderr.write(
                        f'[pick_daemon] right close pressure: baseline='
                        f'{_close_baseline_p:.0f} final={_close_final_p:.0f} '
                        f'delta={_close_final_p - _close_baseline_p:.0f} '
                        f'(closure={_contact_closure})\n')
                    sys.stderr.flush()

                    # closure-block read and squeeze-hold. the grip server state
                    # reply is side-keyed as {side: {joint: q}}, so the score must
                    # unwrap the side key (same shape _verify_hand_open uses).
                    # _squeeze_hold re-asserts the close preset after the 1.0 s ramp,
                    # settles 0.5 s and re-scores; this is the primary keep evidence,
                    # since the pressure delta stays <= 24 even at contact and tactile
                    # reads False.
                    _close_delta = _close_final_p - _close_baseline_p
                    _close_blk_pre, _close_margin_pre, _close_worst_pre = _score_closure_blocked(
                        rhand, 'right', _r_close_target)
                    _close_blk_post, _close_margin_post, _close_worst_post = _squeeze_hold(
                        rhand, 'right', _r_close_target)
                    _close_blocked = _close_blk_post
                    # this command's latest close
                    _ge_close_blocked = _close_blocked
                    _ge_close_margin = _close_margin_post
                    sys.stderr.write(
                        f'[pick_daemon] right squeeze-hold: re-assert close '
                        f'preset, settle 0.5s — margin pre='
                        f'{_close_margin_pre:.3f} post={_close_margin_post:.3f} '
                        f'blocked={_close_blocked} '
                        f'(tol {CLOSURE_BLOCKED_TOL:.2f} rad)\n')
                    sys.stderr.flush()
                    # capture-evidence set for the no-reopen and attempt-1
                    # keep-closed decisions. any term set means the hand probably
                    # closed on something.
                    _any_capture = _close_blocked
                    if _contact_closure == 'contact':
                        _any_capture = True
                    if _close_delta >= _CLOSE_DELTA_TOUCH_MIN:
                        _any_capture = True
                    if (orientation == 'horizontal' and _wrist_z_at_close is not None
                            and _wrist_z_at_close[2] - _h_tz <= H_PALM_TOUCH_BAND):
                        _any_capture = True
                    if _insert_pressed:
                        _any_capture = True

                    # 4.5. verify the grasp (Phase 3 closed loop)
                    sys.stderr.write('[pick_daemon] Phase 3.5: verifying grasp...\n')
                    sys.stderr.flush()
                    time.sleep(0.3)  # let the torque build

                    deadline = time.time() + 0.5
                    state = None
                    while time.time() < deadline:
                        state = detector.update()
                        if state.is_stable:
                            break
                        time.sleep(0.02)

                    if state is None:
                        state = detector.update()

                    quadrant = state.gating_quadrant if state else "grasped_nothing"
                    # no extra veto: a light or soft object squeezed to full never
                    # crosses the close stop (100000 delta), so a real grip can read
                    # 'full'. the empty-hand false solid fires via 'contact', never
                    # 'full'. the detector verdict stands and the close's pressure
                    # delta supplies the missing light-grip evidence.
                    if _contact_closure == 'full' and quadrant == 'solid_grasp':
                        sys.stderr.write(
                            '[pick_daemon] NOTE: closure ended full (delta '
                            'below the 100000 close-stop) but the detector '
                            'says solid_grasp — light object grips read '
                            'lower.  Trusting the detector; no veto.\n')
                        sys.stderr.flush()
                    sys.stderr.write(
                        f'[pick_daemon] Grasp verdict: {quadrant} '
                        f'(motor_high={state.motor_current_high if state else "N/A"}, '
                        f'tactile={state.tactile_contact if state else "N/A"}, '
                        f'support_ratio={state.support_ratio:.2f}'
                        + (f', stable={state.is_stable}' if state else '')
                        + ')\n')
                    sys.stderr.flush()

                    _grasp_log(
                        'R', attempt,
                        closure=_contact_closure,
                        delta=round(_close_delta, 1),
                        blocked=_close_blocked,
                        m_pre=round(_close_margin_pre, 3),
                        m_post=round(_close_margin_post, 3),
                        sup=round(state.support_ratio, 3) if state is not None else None,
                        quad=quadrant,
                        anycap=_any_capture,
                        palm_avail=_palm_data_available_right,
                        final_p=round(_close_final_p, 1),
                        base_p=round(_close_baseline_p, 1),
                    )

                    # the tactile threshold (100000) can also miss a light squeeze;
                    # the close's pressure delta (final - baseline) is independent
                    # evidence the fingers press something.
                    if quadrant == 'grasped_nothing' and _contact_closure != 'no_data':
                        _close_delta = _close_final_p - _close_baseline_p
                        # reopen guard: the adaptive close's own 'contact' result is
                        # independent evidence the hand closed onto something
                        # physical, so do not drop a held object when the detector
                        # reads grasped_nothing. logged.
                        if (_contact_closure == 'contact'
                                or _close_delta >= GRIP_PRESSURE_DELTA_EVIDENCE
                                or _close_blocked):
                            sys.stderr.write(
                                f'[pick_daemon] Grasp evidence upgrade: '
                                f'pressure delta {_close_delta:.0f} >= '
                                f'{GRIP_PRESSURE_DELTA_EVIDENCE:.0f}'
                                + (' or closure-blocked (physical close '
                                   'joints short of the preset)'
                                   if _close_blocked else '')
                                + f' — object in '
                                f'hand: grasped_nothing -> light_object\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                    elif quadrant == 'hit_obstacle' and _contact_closure != 'no_data':
                        # the detector maps motor_high + tactile False to
                        # hit_obstacle, and a real rim or top-edge grip reads
                        # exactly that. the discriminator is the close pressure
                        # delta magnitude: at the rest floor (~105600) the 200000
                        # ceiling caps the squeeze delta at ~94400, below the 100000
                        # stop, so a hard jam or empty self-pinch ends 'full' near
                        # 94k and goes to the support_ratio gate. the close path does
                        # not return 'contact'; a blocked close is carried by the
                        # pressure-independent closure-block margin. the only
                        # high-motor, low-pressure case is a real light rim grip,
                        # kept by promoting hit_obstacle to light_object only in the
                        # light band ('full' and floor < delta <
                        # GRIP_PRESSURE_DELTA_EVIDENCE); 'contact' or delta >= that
                        # bound stays support-gated.
                        _close_delta = _close_final_p - _close_baseline_p
                        if (_contact_closure == 'full'
                                and _CLOSE_DELTA_LIGHT_GRIP_FLOOR < _close_delta
                                < GRIP_PRESSURE_DELTA_EVIDENCE):
                            sys.stderr.write(
                                f'[pick_daemon] Grasp evidence upgrade: '
                                f'pressure delta {_close_delta:.0f} (closure='
                                f'{_contact_closure}) in light-grip band — '
                                f'object in hand: hit_obstacle -> light_object '
                                f'(rim grip)\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                        elif _close_blocked:
                            sys.stderr.write(
                                f'[pick_daemon] Grasp evidence upgrade: '
                                f'closure-blocked - fingers stopped '
                                f'{_close_margin_post:.3f} rad short of the close '
                                f'preset (> tol {CLOSURE_BLOCKED_TOL:.2f} rad): an '
                                f'object is between the fingers; NO tactile/'
                                f'pressure confirmation (pressure channel '
                                f'non-discriminating on this hand) - GRASP LIKELY: '
                                f'hit_obstacle -> light_object\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                        # closure-blocked promotion. support_ratio is not used:
                        # tau_sum/2.943 N is a units artifact (~5e5 on every close,
                        # empty ones included). the closure-block margin is the
                        # discriminator: empty closes reach the preset
                        # (0.038-0.039 rad) while real grasps stop short
                        # (0.828-1.256 rad), and 0.06 sits between. the latch
                        # confound is excluded upstream: _verify_hand_open refuses
                        # attempt 1 unless every joint is within
                        # VERIFY_OPEN_TOLERANCE_RAD of open, and the pre-lift pin
                        # still aborts if the fingers then reach the preset.

                        # blind-mode fallback: no DDS data means skip verification
                    if not _palm_data_available_right and quadrant == "grasped_nothing":
                        sys.stderr.write(
                            '[pick_daemon] BLIND MODE: the RIGHT palm sensor has no '
                            'DDS hand-state data — skipping grasp verification, '
                            'proceeding to lift.\n')
                        sys.stderr.flush()
                        grasp_ok = True
                        break

                    if quadrant == "solid_grasp" or quadrant == "light_object":
                        grasp_ok = True
                        break
                    elif quadrant == "hit_obstacle":
                        # no-reopen retry: the hit_obstacle and grasped_nothing
                        # verdicts are context-free, so opening here would drop a
                        # real capture. with any capture evidence the hand stays
                        # closed into attempt 2 (guarded re-press).
                        sys.stderr.write('[pick_daemon] Hit obstacle, aborting attempt\n')
                        sys.stderr.flush()
                        if _any_capture:
                            _attempt_keep_closed = True
                            sys.stderr.write(
                                f'[pick_daemon] RETRY keep-closed: attempt-1 '
                                f'evidence (closure={_contact_closure}, '
                                f'blocked={_close_blocked}, '
                                f'delta={_close_delta:.0f}'
                                + (f', pad={_wrist_z_at_close[2]-_h_tz:.3f} m'
                                   if orientation == 'horizontal'
                                   and _wrist_z_at_close is not None else '')
                                + (f', insert_pressed={_insert_pressed}'
                                   if orientation != 'horizontal' else '')
                                + ') — hand stays CLOSED, guarded re-press '
                                f'+2 cm forward\n')
                            sys.stderr.flush()
                        else:
                            rhand.open()
                    else:  # "grasped_nothing"
                        sys.stderr.write('[pick_daemon] Grasped nothing, retrying...\n')
                        sys.stderr.flush()
                        if _any_capture:
                            _attempt_keep_closed = True
                            sys.stderr.write(
                                f'[pick_daemon] RETRY keep-closed: attempt-1 '
                                f'evidence (closure={_contact_closure}, '
                                f'blocked={_close_blocked}, '
                                f'delta={_close_delta:.0f}'
                                + (f', pad={_wrist_z_at_close[2]-_h_tz:.3f} m'
                                   if orientation == 'horizontal'
                                   and _wrist_z_at_close is not None else '')
                                + (f', insert_pressed={_insert_pressed}'
                                   if orientation != 'horizontal' else '')
                                + ') — hand stays CLOSED, guarded re-press '
                                f'+2 cm forward\n')
                            sys.stderr.flush()
                        else:
                            rhand.open()

                if _hl is not None:
                    _hl_end(_hl, grasp_ok)
                if not grasp_ok:
                    if _hl is not None:
                        sys.stderr.write(
                            f'[pick_daemon] GRASP FAILED after {_hl["tried"]} '
                            'attempts (hladder)\n')
                    else:
                        sys.stderr.write(f'[pick_daemon] GRASP FAILED after {_upright_attempts()} attempts\n')
                    sys.stderr.flush()
                    # final keep-closed: after a keep-closed attempt 2 with still
                    # no blocked evidence, do not open; the object may still be in
                    # the hand.
                    if _attempt_keep_closed:
                        sys.stderr.write(
                            '[pick_daemon] FINAL keep-closed: OBJECT MAY BE IN '
                            'HAND — hand left CLOSED, operator can take it\n')
                        sys.stderr.flush()
                    else:
                        rhand.open()
                    if not _reply_sent:
                        sys.stdout.write('ERROR:grasp_failed\n')
                        sys.stdout.flush()
                        _reply_sent = True
                    else:
                        # a reply already went out (the hand-open precheck), so never
                        # emit a second stdout line; keep the reason on stderr.
                        sys.stderr.write(
                            '[pick_daemon] ERROR:grasp_failed suppressed — '
                            'this command already sent its one stdout reply '
                            '(an earlier reply for this command)\n')
                        sys.stderr.flush()
                    # park first, release the RC interlock last (in `finally`);
                    # releasing it before a 6.5 s READY slew leaves the legs free
                    # while the arm moves. relax onto the measured pose (stall torque
                    # off), retract along the approach, then both arms to READY.
                    # non-cancellable, since the reply already went out.
                    try:
                        _set_grip_evidence('right', 'failed', _ge_close_blocked,
                                           _ge_close_margin)
                        _relax_arm_to_measured(arm, 'grasp_failed')
                        _retract_trail(arm, 'grasp_failed')
                        _sx_q = _safe_move(arm, READY_POSE, 'grasp_failed')
                        _post_park_stall_check(arm, _sx_q, 'grasp_failed')
                    except Exception as _sx_e:
                        sys.stderr.write(
                            f'[pick_daemon] SAFE-EXIT(grasp_failed): park '
                            f'FAILED: {_sx_e!r} — releasing the RC interlock '
                            'anyway\n')
                        sys.stderr.flush()
                    finally:
                        _hold_rc_writer(arm, False, 'grasp_failed')
                    continue

                # pre-lift pin. with the pressure channel dead
                # (delta <= 24 even at contact) the only keep
                # evidence is the close-joint block. a grip can still
                # slip between close and lift, and the LIFT to
                # CARRY_TUCK wrist roll has no re-grip check. settle
                # and re-score:
                #   still blocked, margin no worse than +0.03 rad:
                #   proceed.
                #   blocked but regressed: proceed, then re-pin after
                #   the tuck.
                #   not blocked (reached preset, nothing holding):
                #   abort LIFT with the hand left closed; the arm
                #   relaxes, retracts, parks at READY.
                _pin_abort = False
                _lpp = _squeeze_hold(rhand, 'right', _capped_close_preset('right'))
                _ge_pin_margin = _lpp[1]  # grip evidence
                if _lpp[1] != _lpp[1]:  # NaN margin = no state data (blind)
                    sys.stderr.write(
                        '[pick_daemon] pre-lift pin SKIPPED (no hand state — '
                        'blind mode), proceeding\n')
                    sys.stderr.flush()
                    _pin_slack = False
                elif not _lpp[0]:
                    sys.stderr.write(
                        '[pick_daemon] pre-lift pin: fingers reached the close '
                        'preset — object may have left the hand; LIFT ABORTED, '
                        'hand left CLOSED\n')
                    sys.stderr.flush()
                    if not _reply_sent:
                        sys.stdout.write('ERROR:grasp_failed\n')
                        sys.stdout.flush()
                        _reply_sent = True
                    else:
                        sys.stderr.write(
                            '[pick_daemon] ERROR:grasp_failed suppressed '
                            '(lift-pin abort) — this command already sent its '
                            'one stdout reply\n')
                        sys.stderr.flush()
                    # this exit must release the arm: a bare continue leaves the
                    # writer PD-holding the blocked grasp target and the arm heats
                    # up. relax onto the measured pose, retract along the approach,
                    # go to READY, run the stall check, and release the RC interlock
                    # last, in `finally`. hand stays closed. the unwind is not
                    # cancellable, since the reply already went out.
                    try:
                        _set_grip_evidence('right', 'failed', _ge_close_blocked,
                                           _ge_close_margin, _lpp[1])
                        _relax_arm_to_measured(arm, 'lift-pin abort')
                        _retract_trail(arm, 'lift-pin abort')
                        _sx_q = _safe_move(arm, READY_POSE, 'lift-pin abort')
                        _post_park_stall_check(arm, _sx_q, 'lift-pin abort')
                    except Exception as _sx_e:
                        sys.stderr.write(
                            f'[pick_daemon] SAFE-EXIT(lift-pin abort): park '
                            f'FAILED: {_sx_e!r} — releasing the RC interlock '
                            'anyway\n')
                        sys.stderr.flush()
                    finally:
                        _hold_rc_writer(arm, False, 'lift-pin abort')
                    continue
                elif _lpp[1] > _close_margin_post + PRE_LIFT_SLIP_REGRESS_TOL:
                    sys.stderr.write(
                        f'[pick_daemon] pre-lift pin CAUGHT-SLACK: margin '
                        f'{_close_margin_post:.3f} → {_lpp[1]:.3f} rad (weaker '
                        f'grip) — proceeding with tuck re-pin\n')
                    sys.stderr.flush()
                    _pin_slack = True
                else:
                    sys.stderr.write(
                        f'[pick_daemon] pre-lift pin ok (margin '
                        f'{_lpp[1]:.3f} rad)\n')
                    sys.stderr.flush()
                    _pin_slack = False

                # steps 4-5: active (right) arm to LIFT, then CARRY_TUCK (fixed
                # joint-space poses; hand stays closed the whole way). height_m is
                # parsed above for protocol compatibility only.
                
                sys.stderr.write(
                    '[pick_daemon] T5 step 4: right arm → LIFT '
                    '(hand closed)...\n')
                sys.stderr.flush()
                _t5_joint_move(arm, 'right', LIFT_RIGHT, zf_legacy=True)

                sys.stderr.write(
                    '[pick_daemon] T5 step 5: right arm → CARRY_TUCK '
                    '(hand closed)...\n')
                sys.stderr.flush()
                # gentler wrist roll: the tuck's joint-space swing at default
                # speed can sling a light capture out of the fingers. 4.0 s roll
                # duration, per CARRY_TUCK_DURATION.
                _tuck_q0 = arm.get_current_dual_arm_q().copy()  # trail
                _t5_joint_move(arm, 'right', CARRY_TUCK_RIGHT,
                               duration=CARRY_TUCK_DURATION, zf_legacy=True)
                # tuck re-pin: one more squeeze-hold after the wrist roll,
                # evidence only. it runs on every pick, not only when the pin is
                # slack. a blocked close at the grasp pose can be fingers stalled
                # on something static; re-scoring after LIFT and tuck shows whether
                # the obstruction travelled with the hand (an empty hand walks to
                # the preset, blocked=False). the reply stays a bare DONE.
                _tpr = _squeeze_hold(rhand, 'right', _capped_close_preset('right'))
                sys.stderr.write(
                    ('[pick_daemon] tuck re-pin: margin ' if _pin_slack
                     else '[pick_daemon] tuck re-score: margin ')
                    + f'{_tpr[1]:.3f} rad blocked={_tpr[0]}\n')
                sys.stderr.flush()
                _grasp_log('R', attempt, phase='tuck', tuck_blocked=_tpr[0],
                           tuck_margin=round(_tpr[1], 3))
                _set_grip_evidence('right', 'picked', _ge_close_blocked,
                                   _ge_close_margin, _ge_pin_margin,
                                   _tpr[0], _tpr[1])
                # stalled short of the clamped tuck target: relax onto the
                # measured pose before the release.
                _tuck_tgt = _tuck_q0.copy()
                _tuck_tgt[7:14] = CARRY_TUCK_RIGHT
                _post_park_stall_check(
                    arm, _expected_after_move(_tuck_q0, _tuck_tgt), 'pick tuck')
                # pick complete: release the RC interlock; the writer keeps the
                # arm at CARRY_TUCK and RC is live again.
                _hold_rc_writer(arm, False, 'pick complete (step 5)')

                # one line per command: if a reply already went out, that error
                # stands as the reply.
                if not _reply_sent:
                    sys.stdout.write('DONE\n')
                    sys.stdout.flush()

            # ── phase 4.3: pick_left (left-arm pick) ─────────────────────
            elif cmd.startswith('pick_left '):
                _zf_pick_begin('left', arm)
                parts = cmd.split()
                try:
                    if len(parts) < 4:
                        raise ValueError('pick_left requires x y z')
                    tx = float(parts[1])
                    ty = float(parts[2])
                    tz = float(parts[3]) + GRASP_Z_OFFSET
                    nx, ny, nz, orientation, height_m, x_off = _parse_pick_tail(parts)
                except (ValueError, IndexError) as e:
                    sys.stdout.write(f'ERROR:pick_left parse: {e}\n')
                    sys.stdout.flush()
                    _zf_restore_floor()   # no motion yet
                    continue
                # reference log and z-frame diagnostic only, no commanded change;
                # left mirror, the lateral mirror applies to both arms.
                sys.stderr.write(
                    '[pick_daemon] v5.2 lateral mirror: ik_y := -torso_x '
                    '(torso +x = RIGHT; URDF +Y = LEFT; both arms); '
                    f'z-frame DIAG: SOLVER_Z_TORSO_OFFSET='
                    f'{SOLVER_Z_TORSO_OFFSET:.3f} m (pelvis vs torso_link) '
                    f'- z channel unchanged (left)\n')
                sys.stderr.flush()
                use_oriented = nx is not None

                if height_m is not None:
                    try:
                        height_m = float(height_m)
                    except ValueError:
                        sys.stdout.write('ERROR:bad height_m\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                    # <=0 or > max: height is not used
                    if height_m <= 0.0 or height_m > HEIGHT_M_MAX:
                        sys.stderr.write(
                            f'[pick_daemon] height_m={height_m} implausible '
                            f'(valid (0, {HEIGHT_M_MAX}]) — using legacy lift\n')
                        sys.stderr.flush()
                        height_m = None
                if x_off is not None:
                    try:
                        x_off = float(x_off)
                    except ValueError:
                        sys.stdout.write('ERROR:bad x_off\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                # point-cloud approach offset; fall back to the fixed standoff
                # when x_off is absent
                approach_offset = x_off if x_off is not None else APPROACH_FORWARD_OFFSET

                if orientation not in ('upright', 'horizontal'):
                    sys.stdout.write(f'ERROR:unknown orientation "{orientation}", expected "upright" or "horizontal"\n')
                    sys.stdout.flush()
                    _zf_restore_floor()   # no motion yet
                    continue

                # same cross-verb height holder as the right branch: filled only
                # when measurable, deliberately outside the height_m gate, and kept
                # even for runtime failures. not side-keyed and not cleared.
                _PLACE_LAST_OBJECT_HEIGHT_M[0] = height_m
                # table z-floor latch, same placement rule as the holder.
                _latch_table_floor(tz, height_m, orientation)

                sys.stderr.write(
                    f'[pick_daemon] Left pick target torso frame: '
                    f'x={tx:.3f} y={ty:.3f} z={tz:.3f}'
                    + (f' normal=({nx:.3f},{ny:.3f},{nz:.3f})'
                       if use_oriented else '')
                    + f' orientation={orientation}'
                    + (f' height_m={height_m:.3f}' if height_m is not None
                       else ' height_m=unknown')
                    + '\n')
                sys.stderr.flush()

                # state machine, steps 1-2 (left mirror): both arms
                # to ready; the inactive (right) arm goes to
                # CARRY_TUCK first and the active (left) stays
                # ready. the retry loop below opens the left hand.
                if ARM_WRITER_ENABLED:
                    try:
                        arm.hold_rc(True)
                    except Exception as _he:
                        sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                sys.stderr.write('[pick_daemon] T5 step 1: both arms → ready...\n')
                sys.stderr.flush()
                # the arm reaches READY first, then the fingers open: opening at
                # NEUTRAL flails them during the ready move.
                move_arm_smooth(arm, READY_POSE, _arm_gravity_ff(READY_POSE, np.zeros(14)))
                rhand.open()
                lhand.open()
                sys.stderr.write(
                    '[pick_daemon] T5 step 2: inactive (right) arm → '
                    'carry_tuck...\n')
                sys.stderr.flush()
                _t5_prologue_q = arm.get_current_dual_arm_q().copy()
                _t5_prologue_q[7:14] = CARRY_TUCK_RIGHT
                move_arm_smooth(arm, _t5_prologue_q, _arm_gravity_ff(_t5_prologue_q, np.zeros(14)))
                _zf_pin_idle('right')   # slice = pin

                # ── phase 3: the retry loop with grasp verification ────────────
                sys.stderr.write('[pick_daemon] DEBUG: before _get_grasp_detector(left)...\n')
                sys.stderr.flush()
                detector = _get_grasp_detector(side='left')
                sys.stderr.write('[pick_daemon] DEBUG: left detector ready\n')
                sys.stderr.flush()
                grasp_ok = False
                ty_use = ty  # may get shifted on retry
                # keep-closed handoff (left mirror)
                _l_attempt_keep_closed = False

                for attempt in range(2):
                    detector.reset()
                    # wp4 insert-press evidence (per-attempt; left upright stop sets it)
                    _l_insert_pressed = False

                    if attempt > 0:
                        sys.stderr.write(
                            f'[pick_daemon] Left retry attempt {attempt+1}/2: '
                            f'shifting grasp +2cm forward\n')
                        sys.stderr.flush()
                        # keep-closed retry (left mirror): no re-open, verify or
                        # re-approach; one guarded re-press below.
                        if not _l_attempt_keep_closed:
                            lhand.open()
                            time.sleep(0.3)
                        ty_use = ty + 0.02  # shift forward

                    # 1. open the left hand
                    sys.stderr.write('[pick_daemon] DEBUG: opening left hand...\n')
                    sys.stderr.flush()
                    if not _l_attempt_keep_closed:
                        lhand.open()
                    sys.stderr.write('[pick_daemon] DEBUG: left hand open done, sleeping...\n')
                    sys.stderr.flush()
                    if not _l_attempt_keep_closed:
                        time.sleep(0.3)
                    sys.stderr.write('[pick_daemon] DEBUG: sleep done\n')
                    sys.stderr.flush()

                    # check the left hand actually opened before inserting: a
                    # closed hand inserts as a fist, pinches, or pushes. the
                    # keep-closed retry skips this.
                    if not _l_attempt_keep_closed:
                        _l_hand_open_ok, _l_hand_open_reason = _verify_hand_open(
                            lhand, 'left', L_HAND_OPEN)
                        if not _l_hand_open_ok:
                            sys.stderr.write(
                                f'[pick_daemon] hand left did not verify open '
                                f'({_l_hand_open_reason}) — aborting insert\n')
                            sys.stderr.flush()
                            sys.stdout.write('ERROR:hand not verified open\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        sys.stderr.write('[pick_daemon] hand left verified open\n')
                        sys.stderr.flush()

                    # fresh tactile baseline per attempt (right-hand mirror)
                    try:
                        _l_p_now = float(lhand.pressure('left')
                                         .get('left', {}).get('max', 0.0))
                    except Exception as _e:
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline '
                            f'(left) read failed: {_e}\n')
                        sys.stderr.flush()
                        _l_p_now = 0.0
                    if _l_p_now > 0.0:
                        _TACTILE_BASELINE_LAST_GOOD[0] = _l_p_now
                        detector.set_tactile_baseline(_l_p_now)
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (left): '
                            f'{_l_p_now:.0f}\n')
                        sys.stderr.flush()
                    elif _TACTILE_BASELINE_LAST_GOOD[0] is not None:
                        detector.set_tactile_baseline(_TACTILE_BASELINE_LAST_GOOD[0])
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (left): '
                            f'read={_l_p_now:.0f} (ok:false/DDS gap) — reusing '
                            f'last known good '
                            f'{_TACTILE_BASELINE_LAST_GOOD[0]:.0f}\n')
                        sys.stderr.flush()
                    else:
                        sys.stderr.write(
                            f'[pick_daemon] detector tactile baseline (left): '
                            f'UNAVAILABLE ({_l_p_now:.0f}) — tactile '
                            f'verification skipped; failures will be loud, '
                            f'never false success\n')
                        sys.stderr.flush()

                    # ── left waypoint approach: ascend → approach-xy → descend ──
                    sys.stderr.write('[pick_daemon] DEBUG: reading current left arm q...\n')
                    sys.stderr.flush()
                    try:
                        _l_cur_q = arm.get_current_dual_arm_q().copy()
                        sys.stderr.write(
                            f'[pick_daemon] DEBUG: got left arm q, shape={_l_cur_q.shape}\n')
                        sys.stderr.flush()
                    except Exception as e:
                        sys.stderr.write(
                            f'[pick_daemon] DEBUG: get_current_dual_arm_q FAILED: {e}\n')
                        sys.stderr.flush()
                        _l_cur_q = None

                    if _l_cur_q is not None:
                        _l_cur_wrist = get_wrist_pose(arm_ik, _l_cur_q, side='left')
                    else:
                        _l_cur_wrist = None
                    # grasp height is the interface tz plus GRASP_Z_UP_BIAS
                    # (+1 cm, right-arm mirror). the interface tz already includes
                    # GRASP_Z_OFFSET. a grip-centre park that lands the wrist low
                    # puts it outside the reach shell, so keep this to one knob
                    # rather than stacked offsets.
                    _l_grasp_z = tz + GRASP_Z_UP_BIAS

                    if attempt > 0 and _l_attempt_keep_closed:
                        # keep-closed guarded re-press (left mirror): closed hand,
                        # no waypoint drag, one guarded re-press at ty_use, fresh
                        # raw solve, position-error backstop only.
                        _solver_fresh(arm_ik, 'left-keep-repress: ')
                        _l_kp_idle_wrist = get_wrist_pose(
                            arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                        _l_kp_idle = (pin.SE3(np.eye(3), _l_kp_idle_wrist).homogeneous
                                      if _l_kp_idle_wrist is not None else R_NEUTRAL)
                        if orientation == 'horizontal':
                            R_l_kp = make_wrist_target_middle_down_left(
                                tx, ty_use, tz,
                                fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        else:
                            R_l_kp = make_wrist_target_upright_v4(
                                tx, ty_use, tz + GRASP_Z_UP_BIAS, -1,
                                fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        with _SuppressStdout():
                            _l_kp_q, _l_kp_tauff = arm_ik.solve_ik(
                                R_l_kp, _l_kp_idle,
                                current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                current_lr_arm_motor_dq=np.zeros(14))
                        if _l_kp_q is None:
                            sys.stdout.write('ERROR:IK failed for left keep-closed re-press\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break
                        _l_kp_q = freeze_idle_arm(arm, _l_kp_q, 'left')
                        sys.stderr.write(
                            '[pick_daemon] left keep-closed re-press: moving '
                            '(guarded, position-error backstop only)\n')
                        sys.stderr.flush()
                        _l_kp_c, _l_kp_r = move_arm_smooth_guarded(
                            arm, _l_kp_q, _l_kp_tauff,
                            PALM_CONTACT_MONITOR_JOINTS_LEFT,
                            PALM_CONTACT_POSITION_THRESHOLD,
                            PALM_CONTACT_CONSECUTIVE_STEPS,
                            zf_label='left-keep-closed-repress')
                        if not _l_kp_c:
                            sys.stderr.write(
                                f'[pick_daemon] left keep-closed re-press '
                                f'guarded stop: {_l_kp_r}\n')
                            sys.stderr.flush()
                    elif orientation == UPRIGHT_GRASP:
                        # upright approach (left mirror): identity orientation at
                        # every waypoint, with the deviation add-on on the identity
                        # base (off unless DEVIATION_ENABLED).
                        use_oriented = False  # force identity: surface normals are unreliable for upright objects

                        # ascend only, never down
                        _l_min_approach_z = tz + APPROACH_HEIGHT_OFFSET
                        if _l_cur_wrist is not None:
                            _l_approach_z = max(_l_cur_wrist[2], _l_min_approach_z)
                            if _l_approach_z > _l_cur_wrist[2] + 0.02:
                                sys.stderr.write(
                                    f'[pick_daemon] Left 1a: ascend to z={_l_approach_z:.3f} '
                                    f'(current={_l_cur_wrist[2]:.3f})\n')
                                sys.stderr.flush()
                                R_l_wp1 = pin.SE3(np.eye(3), np.array([
                                    _l_cur_wrist[0], _l_cur_wrist[1], _l_approach_z])).homogeneous
                                _solver_fresh(arm_ik, 'left-wp1-ascend: ')
                                if not solve_and_move(arm, arm_ik, R_l_wp1,
                                                      "left-wp1-ascend: ", side='left'):
                                    sys.stdout.write('ERROR:IK failed for left ascend\n')
                                    sys.stdout.flush()
                                    _reply_sent = True
                                    break
                            else:
                                sys.stderr.write(
                                    f'[pick_daemon] Left 1a: SKIP ascend '
                                    f'(already at z={_l_cur_wrist[2]:.3f} >= {_l_min_approach_z:.3f})\n')
                                sys.stderr.flush()
                                _l_approach_z = _l_cur_wrist[2]
                        else:
                            _l_approach_z = _l_min_approach_z
                            sys.stderr.write(
                                '[pick_daemon] Left 1a: SKIP ascend '
                                '(FK unavailable), going directly to approach\n')
                            sys.stderr.flush()

                        # behind target, at height
                        sys.stderr.write(
                            '[pick_daemon] Left 1b: approach xy at height\n')
                        sys.stderr.flush()
                        # forward standoff is applied (ty_use +
                        # APPROACH_FORWARD_OFFSET), mirror of the right flow; the
                        # guarded wp4 push walks the final 8 cm. lateral is tx.
                        # the triad stays diagnostic only.
                        
                        if x_off is not None and abs(x_off) > 1e-4:
                            _l_r_obj = -x_off - H_X_OFF_RADIUS_INV
                        else:
                            _l_r_obj = WALL_GRASP_R_DEFAULT
                        _l_r_obj = float(np.clip(_l_r_obj, H_RADIUS_MIN, H_RADIUS_MAX))
                        _l_x_wall = tx + _l_r_obj
                        _l_x_standoff = _l_x_wall + WALL_GRASP_STANDOFF
                        _l_x_contact = _l_x_wall + WALL_GRASP_PAD_GAP
                        sys.stderr.write(
                            f'[pick_daemon] wall grip (left) DIAG: r={_l_r_obj:.3f} '
                            f'wall_x={_l_x_wall:.3f} standoff_x={_l_x_standoff:.3f} '
                            f'contact_x={_l_x_contact:.3f} '
                            f'deviation wp3={_deviation_ramped(1):.3f} '
                            f'wp4={_deviation_ramped(2):.3f} rad '
                            f'(max {DEV_MAX:.3f}) — v4.0 ORIGINAL design '
                            f'RESTORED (identity R, lateral=tx={tx:.3f}, '
                            f'z=tz); the triad numbers are v3 diagnostics '
                            f'only\n')
                        sys.stderr.flush()
                        R_l_wp2 = make_wrist_target_upright_v4(
                            tx, ty_use, _l_approach_z, -1,
                            fwd_offset=APPROACH_FORWARD_OFFSET)
                        _solver_fresh(arm_ik, 'left-wp2-approach: ')
                        if not solve_and_move(arm, arm_ik, R_l_wp2,
                                              "left-wp2-approach: ", side='left'):
                            sys.stdout.write('ERROR:IK failed for left approach xy\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break

                        # descend to grasp z
                        sys.stderr.write(
                            f'[pick_daemon] Left 1c: descend to z={_l_grasp_z:.3f}\n')
                        sys.stderr.flush()
                        # descend straight down behind the object at the fixed
                        # forward standoff with identity orientation. the deviation
                        # add-on rides along (off unless DEVIATION_ENABLED): one
                        # DEV_STEP_MAX step per waypoint toward the contact delta,
                        # frozen at contact.
                        R_l_wp3 = make_wrist_target_upright_v4(
                            tx, ty_use, _l_grasp_z, -1,
                            deviation=(_deviation_ramped(1)
                                       if DEVIATION_ENABLED else 0.0),
                            fwd_offset=APPROACH_FORWARD_OFFSET)
                        _solver_fresh(arm_ik, 'left-wp3-descend: ')
                        if not solve_and_move(arm, arm_ik, R_l_wp3,
                                              "left-wp3-descend: ", side='left'):
                            sys.stdout.write('ERROR:IK failed for left descend\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break

                        time.sleep(0.5)

                        # ── phase 4.2: tracking stability check ──
                        servo_status = 'stable'
                        try:
                            with open('/tmp/pick_servo_status', 'r') as f:
                                servo_status = f.read().strip()
                        except Exception:
                            pass  # file may not exist yet, so assume stable

                        if servo_status == 'lost':
                            sys.stderr.write('[pick_daemon] WARNING: tracking lost, but continuing — guarded insert will handle safety\n')
                            sys.stderr.flush()
                            # carry on to wp4: the palm contact guard handles safety

                        if servo_status == 'shifted':
                            sys.stderr.write(
                                '[pick_daemon] Left tracking shifted, '
                                'recomputing grasp from latest detection\n')
                            sys.stderr.flush()
                            try:
                                with open('/tmp/pick_latest_detection.json', 'r') as f:
                                    latest = json.load(f)
                                tx = latest.get('x', tx)
                                ty_use = latest.get('y', ty_use)
                                tz = latest.get('z', tz) + GRASP_Z_OFFSET
                                # re-latch floor at new tz
                                _latch_table_floor(tz, height_m, orientation)
                                if use_oriented and 'nx' in latest:
                                    nx = latest['nx']
                                    ny = latest['ny']
                                    nz = latest['nz']
                                if 'height_m' in latest:
                                    _h = float(latest['height_m'])
                                    if 0.0 < _h <= HEIGHT_M_MAX:
                                        height_m = _h
                                    else:
                                        height_m = None
                                sys.stderr.write(
                                    f'[pick_daemon] Left updated target: '
                                    f'x={tx:.3f} y={ty_use:.3f} z={tz:.3f}'
                                    + (f' height_m={height_m:.3f}'
                                       if height_m is not None
                                       else ' height_m=unknown')
                                    + '\n')
                                sys.stderr.flush()
                            except Exception as e:
                                sys.stderr.write(
                                    f'[pick_daemon] WARNING: '
                                    f'failed to read latest left detection: {e}\n')
                                sys.stderr.flush()

                        # wp4: insert forward from behind to grasp (guarded)
                        sys.stderr.write(
                            '[pick_daemon] Left Phase 2: insert to grasp (guarded)\n')
                        sys.stderr.flush()
                        # wp4 = guarded insert at full contact depth: lateral tx,
                        # depth ty_use, height tz, identity orientation. in the
                        # servo-shifted branch _l_grasp_z is re-derived from tz. the
                        # insert guard keeps it safe; wp4 FK logs where the palm
                        # landed (diagnostic only). wp4 is the last ramping deviation
                        # waypoint, with the full target frozen from contact plus
                        # retract.
                        _l_grasp_z = tz + GRASP_Z_UP_BIAS  # +1 cm
                        _l_dev_tgt = (_deviation_ramped(2)
                                      if DEVIATION_ENABLED else 0.0)
                        # upright wp4 park (left mirror). the URDF grip zone runs
                        # +17..+69 mm forward of R_ee (mid +43 mm), so a wrist at
                        # the centroid puts the grip zone ~45 mm past the object and
                        # the solver retracts backwards. the commanded target is
                        # parked at fwd_offset=-HAND_GRIP_CENTER_OFFSET: wrist 4.5 cm
                        # behind the centroid, grip zone on it. REFUSE/CEILING
                        # residual is still measured against the unparked
                        # R_grasp_centroid; a parked residual would loosen the
                        # refusal by ~4.5 cm.
                        R_grasp = make_wrist_target_upright_v4(
                            tx, ty_use, _l_grasp_z, -1, deviation=_l_dev_tgt,
                            fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        R_grasp_centroid = make_wrist_target_upright_v4(
                            tx, ty_use, _l_grasp_z, -1, deviation=_l_dev_tgt)
                        sys.stderr.write(
                            '[pick_daemon] v5.2 wp4 park: '
                            f'fwd_offset=-HAND_GRIP_CENTER_OFFSET '
                            f'({HAND_GRIP_CENTER_OFFSET*100:.2f} cm behind '
                            f'centroid — grip zone centred)\n')
                        sys.stderr.flush()

                        # pin the idle (right) arm here, as solve_and_move does
                        _l_wp4_idle_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                        _l_wp4_idle_target = (pin.SE3(np.eye(3), _l_wp4_idle_wrist).homogeneous
                                              if _l_wp4_idle_wrist is not None else R_NEUTRAL)
                        # raw solve, left arm, arg order preserved (grasp, idle):
                        # solve_ik(left, right).
                        _solver_fresh(arm_ik, 'left-wp4-insert: ')
                        with _SuppressStdout():
                            sol_q, sol_tauff = arm_ik.solve_ik(
                                R_grasp, _l_wp4_idle_target,
                                current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                current_lr_arm_motor_dq=np.zeros(14))
                        if sol_q is None:
                            sys.stdout.write('ERROR:IK failed for left insert\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break

                        # left mirror of the reach-ceiling gate. _l_grasp_z is in
                        # scope; R_grasp is the left wp4 insert target built above.
                        _lgpe, _lgoe, _lgRe, _lgde = verify_ik_target(
                            arm_ik, sol_q, 'left', R_grasp_centroid)
                        if _lgpe is not None and _lgpe > SOLVE_VERIFY_POS_TOL:
                            # right-arm two-tier gate, mirrored: refuse above
                            # REACH_REFUSE_POS, diagnostic and continue in
                            # (SOLVE_VERIFY_POS_TOL, REACH_REFUSE_POS].
                            if _lgpe > REACH_REFUSE_POS:
                                sys.stderr.write(
                                    f'OUT OF REACH: solved left insert pose '
                                    f'residual {_lgpe:.3f} m > '
                                    f'{REACH_REFUSE_POS:.3f} m at '
                                    f'grasp z={_l_grasp_z:.3f} m — the arm '
                                    f'cannot reach the commanded insert pose; '
                                    f'the robot is standing too far from the '
                                    f'object.  Left pick REFUSED; move closer '
                                    f'and retry.  If the miss is predominantly '
                                    f'lateral, check the known T1.12 '
                                    f'lateral-sign-mirror item.\n')
                                sys.stderr.flush()
                                sys.stdout.write(f'ERROR:reach_ceiling residual={_lgpe:.3f}m\n')
                                sys.stdout.flush()
                                _reply_sent = True
                                break
                            sys.stderr.write(
                                f'REACH_CEILING(diag): solved left insert pose '
                                f'residual {_lgpe:.3f} m > '
                                f'{SOLVE_VERIFY_POS_TOL:.3f} m at '
                                f'grasp z={_l_grasp_z:.3f} m — best-effort '
                                f'left insert CONTINUES (guarded insert + '
                                f'stop is the safety).  The residual is vs the '
                                f'UNPARKED object-centroid target: the parked '
                                f'insert rides -HAND_GRIP_CENTER_OFFSET '
                                f'({HAND_GRIP_CENTER_OFFSET*100:.2f} cm) so the '
                                f'grip zone itself is '
                                f'{(abs(_lgpe)-HAND_GRIP_CENTER_OFFSET)*100:.0f} '
                                f'cm short of the centroid at this reach cap; '
                                f'the real fix is the robot standing closer.\n')
                            sys.stderr.flush()

                        # freeze the idle (right) arm at its current q to stop
                        # nullspace drift during the insert
                        sol_q = freeze_idle_arm(arm, sol_q, 'left')
                        sys.stderr.write('[pick_daemon] left-wp4-insert (guarded): moving...\n')
                        sys.stderr.flush()

                        # insert guard (left mirror): position-error backstop on
                        # joints [0,3] (PALM_CONTACT_MONITOR_JOINTS_LEFT) plus the
                        # middle-finger stop only. pressure-stop-first stays in the
                        # horizontal flows.

                        def _l_middle_check():
                            return 'middle_finger' if detector.check_middle_finger_contact() else None

                        # left wp4 guard uses the automatic duration
                        # (max_delta/MOVE_SPEED), not a fixed 1.5 s.
                        completed, reason = move_arm_smooth_guarded(
                            arm, sol_q, sol_tauff,
                            PALM_CONTACT_MONITOR_JOINTS_LEFT,
                            PALM_CONTACT_POSITION_THRESHOLD,
                            PALM_CONTACT_CONSECUTIVE_STEPS,
                            stop_fn=_l_middle_check, zf_label='left-wp4-insert')
                        # wp4 insert-press evidence (left mirror)
                        _l_insert_pressed = (not completed) and (
                            reason is not None and (
                                reason.startswith('position_error')
                                or reason == 'middle_finger'))
                        if not completed:
                            sys.stderr.write(
                                f'[pick_daemon] left-wp4-insert guarded stop: {reason} '
                                f'(retracting {RETRACT_AFTER_CONTACT:.2f}m to relieve pressure)\n')
                            sys.stderr.flush()
                            # active retract: move backward to take the contact pressure off
                            _l_contact_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                            if _l_contact_wrist is not None:
                                # identity-base retract (left mirror), with the
                                # deviation add-on frozen at the wp4 value.
                                R_l_retract = make_wrist_target_upright_v4(
                                    tx, ty_use + RETRACT_AFTER_CONTACT,
                                    _l_grasp_z, -1, deviation=_l_dev_tgt,
                                    fwd_offset=approach_offset)
                                # raw solve, left mirror, arg order (grasp, idle).
                                with _SuppressStdout():
                                    _l_retract_q, _l_retract_tauff = arm_ik.solve_ik(
                                        R_l_retract, _l_wp4_idle_target,
                                        current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                        current_lr_arm_motor_dq=np.zeros(14))
                                if _l_retract_q is not None:
                                    # idle (right) arm untouched
                                    _l_retract_q = freeze_idle_arm(arm, _l_retract_q, 'left')
                                    move_arm_smooth(arm, _l_retract_q, _l_retract_tauff, duration=0.3, zf_legacy=True)
                            time.sleep(INSERT_GUARD_HOLD_TIME)

                        time.sleep(0.3)

                        # ── left wrist FK orientation verification ──
                        # left mirror of the right side: compare the achieved left
                        # palm normal with the commanded identity target.
                        # diagnostic only. left physical palm is -Y_local, so
                        # cmd = -R_grasp[:,1].
                        _l_q_now = arm.get_current_dual_arm_q().copy()
                        _l_pe, _l_oe, _l_Re, _l_de = verify_ik_target(arm_ik, _l_q_now, 'left', R_grasp)
                        if _l_Re is not None:
                            _l_palm_cmd = -np.array(R_grasp[:3, 1], dtype=np.float64)
                            _l_palm_act = -np.array(_l_Re[:, 1], dtype=np.float64)
                            _l_wq_cmd = sol_q[LEFT_WRIST_Q_IDX]
                            _l_wq_act = _l_q_now[LEFT_WRIST_Q_IDX]
                            _l_wq_d = _l_wq_act - _l_wq_cmd
                            sys.stderr.write(
                                f'[pick_daemon] left-wp4-FK orientation check: '
                                f'palm normal cmd=({_l_palm_cmd[0]:.2f},{_l_palm_cmd[1]:.2f},{_l_palm_cmd[2]:.2f}) '
                                f'act=({_l_palm_act[0]:.2f},{_l_palm_act[1]:.2f},{_l_palm_act[2]:.2f}) '
                                f'ori err={_l_oe:.3f} rad (tol {SOLVE_VERIFY_ORI_TOL:.2f}) '
                                f'det={_l_de:+.3f}; wrist joints [4,5,6] '
                                f'cmd=({_l_wq_cmd[0]:.3f},{_l_wq_cmd[1]:.3f},{_l_wq_cmd[2]:.3f}) '
                                f'act=({_l_wq_act[0]:.3f},{_l_wq_act[1]:.3f},{_l_wq_act[2]:.3f}) '
                                f'delta=({_l_wq_d[0]:+.3f},{_l_wq_d[1]:+.3f},{_l_wq_d[2]:+.3f})\n')
                            sys.stderr.flush()
                            # body-axis decomposition against straight, mirror of
                            # the right block; left palm is -Y, same builder with
                            # dir_lateral=-1. the target is signed (left s_dev = +1)
                            # for the same comparison.
                            _l_R_straight = make_wrist_target(
                                tx, ty_use, _l_grasp_z)
                            _lpitch, _ldev_, _lflex = wrist_rotation_decomposition(
                                arm_ik, _l_q_now, 'left', _l_R_straight)
                            if _lpitch is not None:
                                sys.stderr.write(
                                    f'[pick_daemon] left-wp4 body-axis decomp: '
                                    f'pitch={_lpitch:+.3f} deviation={_ldev_:+.3f} '
                                    f'flexion={_lflex:+.3f} rad (dev target '
                                    f'{-DEV_ROT_SIGN*_l_dev_tgt:+.3f} '
                                    f'signed, pitch DIAG tol '
                                    f'{PITCH_FROZEN_TOL:.2f})\n')
                                sys.stderr.flush()
                                if abs(_lpitch) > PITCH_FROZEN_TOL:
                                    sys.stderr.write(
                                        '[pick_daemon] WARNING: NO-PITCH rule '
                                        f'VIOLATED at wall contact (left): '
                                        f'|pitch|={abs(_lpitch):.3f} > '
                                        f'{PITCH_FROZEN_TOL:.2f} rad (user rule '
                                        f'2026-08-30 — wrist up/down must not '
                                        f'move during upright contact)\n')
                                    sys.stderr.flush()
                        else:
                            sys.stderr.write(
                                '[pick_daemon] left-wp4-FK orientation check skipped (FK failed)\n')
                            sys.stderr.flush()

                    else:  # orientation == 'horizontal'
                        # ── left horizontal (top-down) approach: ascend → above → descend onto ──

                        # ascend if needed
                        _lh_min_approach_z = tz + 0.20
                        if _l_cur_wrist is not None:
                            if _l_cur_wrist[2] < _lh_min_approach_z:
                                _lh_approach_z = _lh_min_approach_z
                                sys.stderr.write(
                                    f'[pick_daemon] Left Horizontal wp1: ascend to z={_lh_approach_z:.3f} '
                                    f'(current={_l_cur_wrist[2]:.3f}, bottle_z={tz:.3f})\n')
                                sys.stderr.flush()
                                R_lh_wp1 = pin.SE3(np.eye(3), np.array([
                                    _l_cur_wrist[0], _l_cur_wrist[1], _lh_approach_z])).homogeneous
                                # raw swing, left mirror
                                if not solve_and_move(arm, arm_ik, R_lh_wp1,
                                                      "left-h-wp1-ascend: ", side='left', raw=True):
                                    sys.stdout.write('ERROR:IK failed for left horizontal ascend\n')
                                    sys.stdout.flush()
                                    _reply_sent = True
                                    break
                            else:
                                _lh_approach_z = _l_cur_wrist[2]
                                sys.stderr.write(
                                    f'[pick_daemon] Left Horizontal wp1: SKIP ascend '
                                    f'(already at z={_l_cur_wrist[2]:.3f} >= {_lh_min_approach_z:.3f})\n')
                                sys.stderr.flush()
                        else:
                            _lh_approach_z = _lh_min_approach_z
                            sys.stderr.write(
                                '[pick_daemon] Left Horizontal wp1: SKIP ascend '
                                '(FK unavailable), going directly to approach\n')
                            sys.stderr.flush()

                        # wp2: directly above the object with no fwd_offset, parked
                        # over the centroid at ty_use. the left builder stays the
                        # corrected middle_down_left mirror: the unmirrored
                        # middle_down would put the left palm normal up.
                        sys.stderr.write(
                            '[pick_daemon] Left Horizontal wp2: approach xy above bottle\n')
                        sys.stderr.flush()
                        R_lh_wp2 = make_wrist_target_middle_down_left(
                            tx, ty_use, _lh_approach_z,
                            fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                        # raw swing, left mirror
                        _solver_fresh(arm_ik, 'left-h-wp2-approach: ')
                        if not solve_and_move(arm, arm_ik, R_lh_wp2,
                                              "left-h-wp2-approach: ", side='left', raw=True):
                            sys.stdout.write('ERROR:IK failed for left horizontal approach xy\n')
                            sys.stdout.flush()
                            _reply_sent = True
                            break

                        # ── tracking stability check (between wp2 and wp3) ──
                        servo_status = 'stable'
                        try:
                            with open('/tmp/pick_servo_status', 'r') as f:
                                servo_status = f.read().strip()
                        except Exception:
                            pass

                        if servo_status == 'lost':
                            sys.stderr.write('[pick_daemon] WARNING: tracking lost, but continuing — guarded descent will handle safety\n')
                            sys.stderr.flush()

                        if servo_status == 'shifted':
                            sys.stderr.write(
                                '[pick_daemon] Left tracking shifted, '
                                'recomputing grasp from latest detection\n')
                            sys.stderr.flush()
                            try:
                                with open('/tmp/pick_latest_detection.json', 'r') as f:
                                    latest = json.load(f)
                                tx = latest.get('x', tx)
                                ty_use = latest.get('y', ty_use)
                                tz = latest.get('z', tz) + GRASP_Z_OFFSET
                                # re-latch floor at new tz
                                _latch_table_floor(tz, height_m, orientation)
                                if use_oriented and 'nx' in latest:
                                    nx = latest['nx']
                                    ny = latest['ny']
                                    nz = latest['nz']
                                if 'height_m' in latest:
                                    _h = float(latest['height_m'])
                                    if 0.0 < _h <= HEIGHT_M_MAX:
                                        height_m = _h
                                    else:
                                        height_m = None
                                sys.stderr.write(
                                    f'[pick_daemon] Left updated target: '
                                    f'x={tx:.3f} y={ty_use:.3f} z={tz:.3f}'
                                    + (f' height_m={height_m:.3f}'
                                       if height_m is not None
                                       else ' height_m=unknown')
                                    + '\n')
                                sys.stderr.flush()
                            except Exception as e:
                                sys.stderr.write(
                                    f'[pick_daemon] WARNING: '
                                    f'failed to read latest left detection: {e}\n')
                                sys.stderr.flush()

                        # descend onto the object (guarded): straight to the
                        # interface target at y=ty_use, z=tz. pad-only stop (pkg 5)
                        # plus one resume/depth-push slot, left mirror of the right
                        # branch.
                        sys.stderr.write(
                            f'[pick_daemon] Left Horizontal wp3: descend onto bottle to z={tz:.3f} (guarded, pad-only stop)\n')
                        sys.stderr.flush()
                        R_lh_grasp = make_wrist_target_middle_down_left(
                            tx, ty_use, tz,
                            fwd_offset=-HAND_GRIP_CENTER_OFFSET)

                        # pin the idle (right) arm for the retract solve below; each
                        # descent pass re-pins it.
                        _lh_wp3_idle_wrist = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                        _lh_wp3_idle_target = (pin.SE3(np.eye(3), _lh_wp3_idle_wrist).homogeneous
                                               if _lh_wp3_idle_wrist is not None else R_NEUTRAL)

                        def _lh_middle_check():
                            return 'middle_finger' if detector.check_middle_finger_contact(pad_only=True) else None

                        def _lh_descend_once(duration):
                            # fresh idle (right) pin and warm start from the current
                            # q; raw left arg order (grasp, idle).
                            _li = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='right')
                            _lit = (pin.SE3(np.eye(3), _li).homogeneous
                                    if _li is not None else R_NEUTRAL)
                            with _SuppressStdout():
                                _ls, _lt = arm_ik.solve_ik(
                                    R_lh_grasp, _lit,
                                    current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                    current_lr_arm_motor_dq=np.zeros(14))
                            if _ls is None:
                                return None, False, 'ik_fail'
                            _ls = freeze_idle_arm(arm, _ls, 'left')
                            sys.stderr.write('[pick_daemon] left-h-wp3-descent (guarded): moving...\n')
                            sys.stderr.flush()
                            _lc, _lr = move_arm_smooth_guarded(
                                arm, _ls, _lt,
                                PALM_CONTACT_MONITOR_JOINTS_LEFT,
                                PALM_CONTACT_POSITION_THRESHOLD,
                                PALM_CONTACT_CONSECUTIVE_STEPS,
                                stop_fn=_lh_middle_check,
                                duration=duration, zf_label='left-h-wp3-descent')
                            return _ls, _lc, _lr

                        # explicit descend pass loop (left mirror): a reach stall is
                        # not contact. pads hang 5-17 mm under the wrist, so a joint
                        # stall well above the target does not mean the palm touched
                        # anything.
                        _solver_fresh(arm_ik, 'left-h-wp3-descend: ')
                        _lh_pushes = 0
                        _lh_prev_gap = None
                        _lh_contact = False
                        while True:
                            sol_q, completed, reason = _lh_descend_once(
                                DESCENT_DURATION if _lh_pushes == 0 else None)
                            if sol_q is None:
                                sys.stdout.write('ERROR:IK failed for left horizontal descent\n')
                                sys.stdout.flush()
                                _reply_sent = True
                                break
                            time.sleep(0.3)  # settle before the depth read (post-ramp)
                            _lhwz = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                            _lgap = (_lhwz[2] - tz) if _lhwz is not None else None
                            if not completed:
                                if (reason is not None and reason.startswith('position_error')
                                        and _lgap is not None and _lgap > H_DESCEND_STALL_GAP):
                                    sys.stderr.write(
                                        f'[pick_daemon] REACH STALL at +{_lgap:.3f} m '
                                        f'({_lgap*100:.1f} cm above target z={tz:.3f}: {reason}) '
                                        f'— NOT contact (pads hang 5-17 mm under the wrist, '
                                        f'guard stop hit with the palm still airborne).  No '
                                        f'contact-retract, no immediate close.\n')
                                    sys.stderr.flush()
                                else:
                                    _lh_contact = True
                                    break  # true contact, retract block below
                            if _lgap is None or _lgap <= H_DESCEND_DEPTH_TOL:
                                break  # reached the depth band (or FK unreadable)
                            if _lh_pushes >= H_DESCEND_MAX_PUSHES:
                                sys.stderr.write(
                                    f'[pick_daemon] left-h-wp3 descend budget spent: '
                                    f'{_lh_pushes} push(es), gap +{_lgap:.3f} m above target — '
                                    f'DIAG, close-at-current-height; closure evidence '
                                    f'adjudicates\n')
                                sys.stderr.flush()
                                break
                            if (_lh_prev_gap is not None
                                    and (_lh_prev_gap - _lgap) < H_DESCEND_MIN_PROGRESS):
                                sys.stderr.write(
                                    f'[pick_daemon] left-h-wp3 push made less than '
                                    f'{H_DESCEND_MIN_PROGRESS:.3f} m '
                                    f'(+{_lh_prev_gap:.3f} → +{_lgap:.3f}) — no more pushes, '
                                    f'close-at-current-height; closure evidence adjudicates\n')
                                sys.stderr.flush()
                                break
                            _lh_prev_gap = _lgap
                            _lh_pushes += 1
                            sys.stderr.write(
                                f'[pick_daemon] left-h-wp3 push {_lh_pushes}: gap +{_lgap:.3f} m '
                                f'(>{H_DESCEND_DEPTH_TOL:.3f}) — one more guarded push\n')
                            sys.stderr.flush()

                        if _lh_contact:
                            sys.stderr.write(
                                f'[pick_daemon] left-h-wp3-descent guarded stop: {reason} '
                                f'(retracting {RETRACT_AFTER_CONTACT:.2f}m to relieve pressure)\n')
                            sys.stderr.flush()
                            # active retract: move backward to take the contact
                            # pressure off; unconditional at z=tz.
                            R_lh_retract = make_wrist_target_middle_down_left(
                                tx, ty_use + RETRACT_AFTER_CONTACT, tz,
                                fwd_offset=-HAND_GRIP_CENTER_OFFSET)
                            # raw solve, left mirror, arg order (grasp, idle).
                            
                            with _SuppressStdout():
                                _lh_retract_q, _lh_retract_tauff = arm_ik.solve_ik(
                                    R_lh_retract, _lh_wp3_idle_target,
                                    current_lr_arm_motor_q=arm.get_current_dual_arm_q().copy(),
                                    current_lr_arm_motor_dq=np.zeros(14))
                            if _lh_retract_q is not None:
                                # idle (right) arm untouched.
                                _lh_retract_q = freeze_idle_arm(arm, _lh_retract_q, 'left')
                                move_arm_smooth(arm, _lh_retract_q, _lh_retract_tauff, duration=0.3, zf_legacy=True)
                            time.sleep(INSERT_GUARD_HOLD_TIME)

                        time.sleep(0.3)

                        # ── left wrist FK orientation verification (horizontal) ──
                        # left physical palm is -Y_local, so cmd = -R_lh_grasp[:,1]
                        # (palm down); joints [4,5,6] are forearm yaw, up-down and
                        # flexion.
                        _lqh_q_now = arm.get_current_dual_arm_q().copy()
                        _lqh_pe, _lqh_oe, _lqh_Re, _lqh_de = verify_ik_target(arm_ik, _lqh_q_now, 'left', R_lh_grasp)
                        if _lqh_Re is not None:
                            _lqh_palm_cmd = -np.array(R_lh_grasp[:3, 1], dtype=np.float64)  # left palm = -Y_local
                            _lqh_palm_act = -np.array(_lqh_Re[:, 1], dtype=np.float64)
                            _lqh_wq_cmd = sol_q[LEFT_WRIST_Q_IDX]
                            _lqh_wq_act = _lqh_q_now[LEFT_WRIST_Q_IDX]
                            _lqh_wq_d = _lqh_wq_act - _lqh_wq_cmd
                            sys.stderr.write(
                                f'[pick_daemon] left-h-wp3-FK orientation check: '
                                f'palm normal cmd=({_lqh_palm_cmd[0]:.2f},{_lqh_palm_cmd[1]:.2f},{_lqh_palm_cmd[2]:.2f}) '
                                f'act=({_lqh_palm_act[0]:.2f},{_lqh_palm_act[1]:.2f},{_lqh_palm_act[2]:.2f}) '
                                f'ori err={_lqh_oe:.3f} rad (tol {SOLVE_VERIFY_ORI_TOL:.2f}) '
                                f'det={_lqh_de:+.3f}; wrist joints [4,5,6] '
                                f'cmd=({_lqh_wq_cmd[0]:.3f},{_lqh_wq_cmd[1]:.3f},{_lqh_wq_cmd[2]:.3f}) '
                                f'act=({_lqh_wq_act[0]:.3f},{_lqh_wq_act[1]:.3f},{_lqh_wq_act[2]:.3f}) '
                                f'delta=({_lqh_wq_d[0]:+.3f},{_lqh_wq_d[1]:+.3f},{_lqh_wq_d[2]:+.3f})\n')
                            sys.stderr.flush()
                        else:
                            sys.stderr.write(
                                '[pick_daemon] left-h-wp3-FK orientation check skipped (FK failed)\n')
                            sys.stderr.flush()

                    # 4. close the left hand (adaptive, stops on finger pressure);
                    # stop_threshold = GRIP_CLOSE_STOP_DELTA (100000), right-arm
                    # mirror.
                    sys.stderr.write(
                        f'[pick_daemon] Left Phase 3: closing hand (bounded lead, '
                        f'eps={HAND_CLOSE_LEAD_EPS:.2f} rad)\n')
                    sys.stderr.flush()
                    _l_close_baseline_p = 0.0
                    try:
                        _l_close_baseline_p = float(
                            lhand.pressure('left').get('left', {}).get('max', 0.0))
                    except Exception:
                        pass
                    # wrist height at the close (left mirror; pad-band evidence)
                    _l_wrist_z_at_close = get_wrist_pose(
                        arm_ik, arm.get_current_dual_arm_q().copy(), side='left')
                    # left mirror of the right close block: closure is capped at
                    # F_MAX_SAFE and the cap is honoured, with a bounded lead.
                    # `_l_target` is the unchanged capped pose shared by the hold
                    # and the closure-block score.
                    _l_target, _l_eff = clamped_close_target('left', 1.0)
                    _l_contact_closure = _bounded_lead_close(
                        lhand, 'left', _l_target)
                    if _l_contact_closure == 'contact':
                        # grip detected early: hold where it stopped, never tighten
                        # to the cap.
                        sys.stderr.write(
                            '[pick_daemon] Left adaptive closure: grip detected, holding at '
                            f'the aperture reached (cap {_l_eff:.3f})\n')
                        sys.stderr.flush()
                    elif _l_contact_closure == 'no_data':
                        sys.stderr.write('[pick_daemon] Left bounded-lead close: NO HAND STATE -- nothing written (refusing to press blind)\n')
                        sys.stderr.flush()
                    else:
                        sys.stderr.write('[pick_daemon] Left bounded-lead close: walk complete -- lead held at eps (see the bounded-lead line above)\n')
                        sys.stderr.flush()

                    sys.stderr.write(
                        f'[pick_daemon] left hand last commanded q after bounded-lead close: '
                        f'{[lhand.last_q[id] for id in DEX3_JOINTS]}\n')
                    sys.stderr.flush()
                    _l_close_final_p = 0.0
                    try:
                        _l_close_final_p = float(
                            lhand.pressure('left').get('left', {}).get('max', 0.0))
                    except Exception:
                        pass
                    sys.stderr.write(
                        f'[pick_daemon] left close pressure: baseline='
                        f'{_l_close_baseline_p:.0f} final={_l_close_final_p:.0f} '
                        f'delta={_l_close_final_p - _l_close_baseline_p:.0f} '
                        f'(closure={_l_contact_closure})\n')
                    sys.stderr.flush()

                    # closure-block read and squeeze-hold (left mirror); the state
                    # read is side-keyed, so the left side scores its own state.
                    _l_close_delta = _l_close_final_p - _l_close_baseline_p
                    _l_close_blk_pre, _l_close_margin_pre, _l_close_worst_pre = _score_closure_blocked(
                        lhand, 'left', _l_target)
                    _l_close_blk_post, _l_close_margin_post, _l_close_worst_post = _squeeze_hold(
                        lhand, 'left', _l_target)
                    _l_close_blocked = _l_close_blk_post
                    sys.stderr.write(
                        f'[pick_daemon] left squeeze-hold: re-assert close '
                        f'preset, settle 0.5s — margin pre='
                        f'{_l_close_margin_pre:.3f} post={_l_close_margin_post:.3f} '
                        f'blocked={_l_close_blocked} '
                        f'(tol {CLOSURE_BLOCKED_TOL:.2f} rad)\n')
                    sys.stderr.flush()
                    # capture-evidence set (left mirror of the right-arm set)
                    _l_any_capture = _l_close_blocked
                    if _l_contact_closure == 'contact':
                        _l_any_capture = True
                    if _l_close_delta >= _CLOSE_DELTA_TOUCH_MIN:
                        _l_any_capture = True
                    if (orientation == 'horizontal' and _l_wrist_z_at_close is not None
                            and _l_wrist_z_at_close[2] - tz <= H_PALM_TOUCH_BAND):
                        _l_any_capture = True
                    if _l_insert_pressed:
                        _l_any_capture = True

                    # 4.5. verify the grasp (Phase 3 closed loop)
                    sys.stderr.write('[pick_daemon] Left Phase 3.5: verifying grasp...\n')
                    sys.stderr.flush()
                    time.sleep(0.3)  # let the torque build

                    deadline = time.time() + 0.5
                    state = None
                    while time.time() < deadline:
                        state = detector.update()
                        if state.is_stable:
                            break
                        time.sleep(0.02)

                    if state is None:
                        state = detector.update()

                    quadrant = state.gating_quadrant if state else "grasped_nothing"
                    # no extra veto on the detector's verdict; the close's pressure
                    # delta supplies light-grip evidence the detector can miss.
                    if _l_contact_closure == 'full' and quadrant == 'solid_grasp':
                        sys.stderr.write(
                            '[pick_daemon] NOTE: left closure ended full '
                            '(delta below the 100000 close-stop) but the '
                            'detector says solid_grasp — light object grips '
                            'read lower.  Trusting the detector; no veto.\n')
                        sys.stderr.flush()
                    sys.stderr.write(
                        f'[pick_daemon] Left grasp verdict: {quadrant} '
                        f'(motor_high={state.motor_current_high if state else "N/A"}, '
                        f'tactile={state.tactile_contact if state else "N/A"}, '
                        f'support_ratio={state.support_ratio:.2f}'
                        + (f', stable={state.is_stable}' if state else '')
                        + ')\n')
                    sys.stderr.flush()

                    _grasp_log(
                        'L', attempt,
                        closure=_l_contact_closure,
                        delta=round(_l_close_delta, 1),
                        blocked=_l_close_blocked,
                        m_pre=round(_l_close_margin_pre, 3),
                        m_post=round(_l_close_margin_post, 3),
                        sup=round(state.support_ratio, 3) if state is not None else None,
                        quad=quadrant,
                        anycap=_l_any_capture,
                        palm_avail=_palm_data_available,
                        final_p=round(_l_close_final_p, 1),
                        base_p=round(_l_close_baseline_p, 1),
                    )

                    # evidence upgrade (left mirror)
                    if quadrant == 'grasped_nothing' and _l_contact_closure != 'no_data':
                        _l_close_delta = _l_close_final_p - _l_close_baseline_p
                        # reopen guard, left mirror: do not drop a held object on
                        # grasped_nothing when the adaptive close already detected a
                        # physical grip ('contact').
                        if (_l_contact_closure == 'contact'
                                or _l_close_delta >= GRIP_PRESSURE_DELTA_EVIDENCE
                                or _l_close_blocked):
                            sys.stderr.write(
                                f'[pick_daemon] Left grasp evidence upgrade: '
                                f'pressure delta {_l_close_delta:.0f} >= '
                                f'{GRIP_PRESSURE_DELTA_EVIDENCE:.0f}'
                                + (' or closure-blocked (physical close '
                                   'joints short of the preset)'
                                   if _l_close_blocked else '')
                                + f' — object in '
                                f'hand: grasped_nothing -> light_object\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                    elif quadrant == 'hit_obstacle' and _l_contact_closure != 'no_data':
                        # left mirror of the light-grip band. hit_obstacle is
                        # promoted only in the light band ('full' and
                        # floor < delta < 40000); a 'full' or delta >= 40000 grip
                        # (the 100000 stop sits above the ~94400 natural ceiling, so
                        # hard jams land 'full' near 94k) and any 'contact' or
                        # delta >= 40000 jam go to the support_ratio >= 0.5 gate,
                        # never un-gated.
                        _l_close_delta = _l_close_final_p - _l_close_baseline_p
                        if (_l_contact_closure == 'full'
                                and _CLOSE_DELTA_LIGHT_GRIP_FLOOR < _l_close_delta
                                < GRIP_PRESSURE_DELTA_EVIDENCE):
                            sys.stderr.write(
                                f'[pick_daemon] Left grasp evidence upgrade: '
                                f'pressure delta {_l_close_delta:.0f} (closure='
                                f'{_l_contact_closure}) in light-grip band — '
                                f'object in hand: hit_obstacle -> light_object '
                                f'(rim grip)\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                        elif _l_close_blocked:
                            sys.stderr.write(
                                f'[pick_daemon] Left grasp evidence upgrade: '
                                f'closure-blocked - fingers stopped '
                                f'{_l_close_margin_post:.3f} rad short of the close '
                                f'preset (> tol {CLOSURE_BLOCKED_TOL:.2f} rad): an '
                                f'object is between the fingers; NO tactile/'
                                f'pressure confirmation (pressure channel '
                                f'non-discriminating on this hand) - GRASP LIKELY: '
                                f'hit_obstacle -> light_object\n')
                            sys.stderr.flush()
                            quadrant = 'light_object'
                        # closure-blocked promotion, left mirror: the support_ratio
                        # closure-block margin is the discriminator and
                        # _verify_hand_open excludes the latch confound. left Dex3
                        # reports no `actual` state, so _l_close_blocked stays False
                        # and this promotion is inert on this arm.

                    # blind-mode fallback: no DDS data means skip verification
                    if not _palm_data_available and quadrant == "grasped_nothing":
                        sys.stderr.write(
                            '[pick_daemon] BLIND MODE: No DDS hand-state data — '
                            'skipping grasp verification, proceeding to lift.\n')
                        sys.stderr.flush()
                        grasp_ok = True
                        break

                    if quadrant == "solid_grasp" or quadrant == "light_object":
                        grasp_ok = True
                        break
                    elif quadrant == "hit_obstacle":
                        # no-reopen retry (left mirror): a capture keeps the hand
                        # closed even when the verdict has no context.
                        sys.stderr.write('[pick_daemon] Left hit obstacle, aborting attempt\n')
                        sys.stderr.flush()
                        if _l_any_capture:
                            _l_attempt_keep_closed = True
                            sys.stderr.write(
                                f'[pick_daemon] RETRY keep-closed (left): '
                                f'attempt-1 evidence '
                                f'(closure={_l_contact_closure}, '
                                f'blocked={_l_close_blocked}, '
                                f'delta={_l_close_delta:.0f}'
                                + (f', pad={_l_wrist_z_at_close[2]-tz:.3f} m'
                                   if orientation == 'horizontal'
                                   and _l_wrist_z_at_close is not None else '')
                                + (f', insert_pressed={_l_insert_pressed}'
                                   if orientation != 'horizontal' else '')
                                + ') — hand stays CLOSED, guarded re-press '
                                f'+2 cm forward\n')
                            sys.stderr.flush()
                        else:
                            lhand.open()
                    else:  # "grasped_nothing"
                        sys.stderr.write('[pick_daemon] Left grasped nothing, retrying...\n')
                        sys.stderr.flush()
                        if _l_any_capture:
                            _l_attempt_keep_closed = True
                            sys.stderr.write(
                                f'[pick_daemon] RETRY keep-closed (left): '
                                f'attempt-1 evidence '
                                f'(closure={_l_contact_closure}, '
                                f'blocked={_l_close_blocked}, '
                                f'delta={_l_close_delta:.0f}'
                                + (f', pad={_l_wrist_z_at_close[2]-tz:.3f} m'
                                   if orientation == 'horizontal'
                                   and _l_wrist_z_at_close is not None else '')
                                + (f', insert_pressed={_l_insert_pressed}'
                                   if orientation != 'horizontal' else '')
                                + ') — hand stays CLOSED, guarded re-press '
                                f'+2 cm forward\n')
                            sys.stderr.flush()
                        else:
                            lhand.open()

                if not grasp_ok:
                    sys.stderr.write('[pick_daemon] LEFT GRASP FAILED after 2 attempts\n')
                    sys.stderr.flush()
                    # final keep-closed (left)
                    if _l_attempt_keep_closed:
                        sys.stderr.write(
                            '[pick_daemon] FINAL keep-closed (left): OBJECT '
                            'MAY BE IN HAND — hand left CLOSED, operator can '
                            'take it\n')
                        sys.stderr.flush()
                    else:
                        lhand.open()
                    if not _reply_sent:
                        sys.stdout.write('ERROR:grasp_failed\n')
                        sys.stdout.flush()
                        _reply_sent = True
                    else:
                        # a reply already went out: never emit a second stdout line;
                        # put the reason on stderr.
                        sys.stderr.write(
                            '[pick_daemon] ERROR:grasp_failed suppressed — '
                            'this command already sent its one stdout reply '
                            '(an earlier reply for this command)\n')
                        sys.stderr.flush()
                    # park first, release the RC interlock last (in `finally`);
                    # releasing before a 6.5 s READY slew leaves the legs free while
                    # the arm moves. left failure path: both arms to READY,
                    # non-cancellable.
                    try:
                        _relax_arm_to_measured(arm, 'grasp_failed (left)')
                        _sx_q = _safe_move(arm, READY_POSE, 'grasp_failed (left)')
                        _post_park_stall_check(arm, _sx_q, 'grasp_failed (left)')
                    except Exception as _sx_e:
                        sys.stderr.write(
                            f'[pick_daemon] SAFE-EXIT(grasp_failed (left)): '
                            f'park FAILED: {_sx_e!r} — releasing the RC '
                            'interlock anyway\n')
                        sys.stderr.flush()
                    finally:
                        _hold_rc_writer(arm, False, 'grasp_failed (left)')
                    continue

                # pre-lift pin (left mirror). with no pressure the
                # close-block is keep evidence; the LIFT to
                # CARRY_TUCK roll has no re-grip check. outcomes: ok,
                # caught-slack (tuck re-pin), aborted (hand left
                # closed; arm relaxes, parks at READY, then the RC
                # interlock is released).
                _pin_abort = False
                _llpp = _squeeze_hold(lhand, 'left', _capped_close_preset('left'))
                if _llpp[1] != _llpp[1]:  # NaN margin = no state data (blind)
                    sys.stderr.write(
                        '[pick_daemon] pre-lift pin SKIPPED (no hand state — '
                        'blind mode), proceeding\n')
                    sys.stderr.flush()
                    _pin_slack = False
                elif not _llpp[0]:
                    sys.stderr.write(
                        '[pick_daemon] pre-lift pin (left): fingers reached '
                        'the close preset — object may have left the hand; '
                        'LIFT ABORTED, hand left CLOSED\n')
                    sys.stderr.flush()
                    if not _reply_sent:
                        sys.stdout.write('ERROR:grasp_failed\n')
                        sys.stdout.flush()
                        _reply_sent = True
                    else:
                        sys.stderr.write(
                            '[pick_daemon] ERROR:grasp_failed suppressed '
                            '(lift-pin abort) — this command already sent its '
                            'one stdout reply\n')
                        sys.stderr.flush()
                    # park first, release last; the unwind is not cancellable
                    # (reply already out).
                    try:
                        _relax_arm_to_measured(arm, 'lift-pin abort (left)')
                        _sx_q = _safe_move(arm, READY_POSE,
                                           'lift-pin abort (left)')
                        _post_park_stall_check(arm, _sx_q,
                                               'lift-pin abort (left)')
                    except Exception as _sx_e:
                        sys.stderr.write(
                            f'[pick_daemon] SAFE-EXIT(lift-pin abort (left)): '
                            f'park FAILED: {_sx_e!r} — releasing the RC '
                            'interlock anyway\n')
                        sys.stderr.flush()
                    finally:
                        _hold_rc_writer(arm, False, 'lift-pin abort (left)')
                    continue
                elif _llpp[1] > _l_close_margin_post + PRE_LIFT_SLIP_REGRESS_TOL:
                    sys.stderr.write(
                        f'[pick_daemon] pre-lift pin (left) CAUGHT-SLACK: '
                        f'margin {_l_close_margin_post:.3f} → {_llpp[1]:.3f} '
                        f'rad (weaker grip) — proceeding with tuck re-pin\n')
                    sys.stderr.flush()
                    _pin_slack = True
                else:
                    sys.stderr.write(
                        f'[pick_daemon] pre-lift pin ok (left, margin '
                        f'{_llpp[1]:.3f} rad)\n')
                    sys.stderr.flush()
                    _pin_slack = False

                # steps 4-5: active (left) arm to LIFT, then CARRY_TUCK (fixed
                # joint-space poses; hand stays closed). height_m is parsed above
                # for protocol compatibility only.
                sys.stderr.write(
                    '[pick_daemon] T5 step 4: left arm → LIFT '
                    '(hand closed)...\n')
                sys.stderr.flush()
                _t5_joint_move(arm, 'left', LIFT_LEFT, zf_legacy=True)

                sys.stderr.write(
                    '[pick_daemon] T5 step 5: left arm → CARRY_TUCK '
                    '(hand closed)...\n')
                sys.stderr.flush()
                # gentler wrist roll (left mirror)
                _t5_joint_move(arm, 'left', CARRY_TUCK_LEFT,
                               duration=CARRY_TUCK_DURATION, zf_legacy=True)
                if _pin_slack:
                    # tuck re-pin (left mirror)
                    _ltpr = _squeeze_hold(lhand, 'left', _capped_close_preset('left'))
                    sys.stderr.write(
                        '[pick_daemon] tuck re-pin (left): margin '
                        f'{_ltpr[1]:.3f} rad blocked={_ltpr[0]}\n')
                    sys.stderr.flush()
                # pick complete: release the RC interlock (the writer keeps the arm
                # at CARRY_TUCK and RC is live again).
                _hold_rc_writer(arm, False, 'pick complete (step 5)')

                # one line per command: if a reply already went out, that error
                # stands.
                if not _reply_sent:
                    sys.stdout.write('DONE\n')
                    sys.stdout.flush()

            # place = drop-in-place. "place x y z" carries the held
            # object to an upright wrist drop pose at [x, y, z] (torso
            # frame; z is wrist height and the standoff is the caller's
            # concern), then releases. the coords form reverses pick:
            # RC on, idle arm to CARRY_TUCK (freeze_idle_arm pin in
            # every solve), ASCEND to hover z = pz +
            # APPROACH_HEIGHT_OFFSET over [px, py], APPROACH-XY, DESCEND
            # to pz, open the holding hand (right; log-only verify),
            # then both arms to LIFT and NEUTRAL, payload off, RC off,
            # DONE. the no-coords form releases at the current carry
            # pose and opens only the holding hand.
            elif cmd == 'place' or cmd.startswith('place '):
                _zf_pick_begin('right', arm)
                parts = cmd.split()
                hold_side = 'right'
                idle_side = 'left'
                _phand = rhand
                _popen_preset = R_HAND_OPEN
                _pdir = +1
                _pcarry_idle = CARRY_TUCK_LEFT
                _verb = 'place'
                if len(parts) == 1:
                    # no coords: no extra lift. the raise is clearance for an
                    # approach and this exit has none: the release happens at the
                    # carry pose, so raising first would move the release point up
                    # by up to PLACE_EXTRA_LIFT_MAX_M and change the drop height.
                    # the raise lives only on the coords exits, where the
                    # ASCEND/DESCEND pair returns to pz.
                    if ARM_WRITER_ENABLED:
                        try:
                            arm.hold_rc(True)
                        except Exception as _he:
                            sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                            sys.stdout.flush()
                            _zf_restore_floor()   # no motion yet
                            continue
                    sys.stderr.write(
                        f'[pick_daemon] {_verb}: drop-in-place at the current '
                        f'carry pose — open {hold_side} hand ONLY, payload '
                        'off, then TWO-ARM closeout to NEUTRAL\n')
                    sys.stderr.flush()
                    # extra lift deliberately absent here; see the note above
                    _phand.open()
                    _p_open_ok, _p_open_reason = _verify_hand_open(
                        _phand, hold_side, _popen_preset)
                    _note_release_open(hold_side, _p_open_ok)
                    if not _p_open_ok:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: {hold_side} hand open '
                            f'verify failed ({_p_open_reason}) — continuing\n')
                        sys.stderr.flush()
                    time.sleep(0.5)
                    try:
                        if hasattr(arm, 'payload'):
                            arm.payload(False)
                    except Exception as _pe:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: payload off failed: {_pe} '
                            '— continuing\n')
                        sys.stderr.flush()
                    # two-arm closeout (RC stays on): the empty holding arm lifts
                    # off the release point and the idle arm leaves CARRY_TUCK on the
                    # same beat, both to LIFT, then both to NEUTRAL.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} closeout 1/2: both arms → '
                        'LIFT (holding arm clears; idle arm off CARRY_TUCK)'
                        '...\n')
                    sys.stderr.flush()
                    _p_lift = np.concatenate(
                        [LIFT_LEFT, LIFT_RIGHT]).astype(np.float64)
                    move_arm_smooth(arm, _p_lift,
                                    _arm_gravity_ff(_p_lift, np.zeros(14)), zf_legacy=True)
                    # park the holding hand closed here: arm at LIFT, fingers clear
                    # of the drop point. gated by PLACE_PARK_HAND_CLOSED.
                    _park_hand_after_lift(_phand, hold_side, _verb, _p_open_ok)
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} closeout 2/2: both arms → '
                        'NEUTRAL together...\n')
                    sys.stderr.flush()
                    _p_neutral = np.concatenate(
                        [NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
                    _pn_q0 = arm.get_current_dual_arm_q().copy()  # stall check
                    move_arm_smooth(arm, _p_neutral,
                                    _arm_gravity_ff(_p_neutral, np.zeros(14)), zf_legacy=True)
                    # stall -> relax
                    
                    _post_park_stall_check(
                        arm, _expected_after_move(_pn_q0, _p_neutral),
                        f'{_verb} closeout')
                    _hold_rc_writer(arm, False,
                                    f'{_verb} complete (drop-in-place)')
                    sys.stdout.write('DONE%s%s\n' % (_release_open_token(hold_side),
                                                   _zf_reply_suffix(_verb)))
                    sys.stdout.flush()
                else:
                    # ── coords "place x y z": reverse-pick waypoints ───────
                    # parse before RC on: a bad parse must never
                    # engage the arm.
                    try:
                        if len(parts) < 4:
                            raise ValueError(f'{_verb} requires x y z')
                        px = float(parts[1])
                        py = float(parts[2])
                        pz = float(parts[3])
                    except (ValueError, IndexError) as e:
                        sys.stdout.write(f'ERROR:{_verb} parse: {e}\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                    # optional height token: the object height for the extra lift.
                    # it is the durable height source; the in-process holder below
                    # is lost across a daemon restart.
                    # not part of the raise-on-bad-parse contract: an unparseable
                    # height means "unknown", not a malformed command, so the place
                    # still runs (warn-and-degrade); an absent token falls back to
                    # the holder.
                    _el_h_token = None
                    if len(parts) >= 5:
                        try:
                            _el_h_token = float(parts[4])
                        except (ValueError, IndexError):
                            sys.stderr.write(
                                f'[pick_daemon] {_verb}: height token '
                                f'{parts[4]!r} is not a number — treating the '
                                f'object height as unknown\n')
                            sys.stderr.flush()
                            _el_h_token = None
                    _el_h_m = (_el_h_token if _el_h_token is not None
                               else _PLACE_LAST_OBJECT_HEIGHT_M[0])
                    # self-collision precheck: refuse a drop point whose
                    # ASCEND/DESCEND would press the right arm into the torso or
                    # tucked left arm. runs before hold_rc, the extra lift and any
                    # arm motion; the object stays held.
                    _sc_pre = _self_collision_precheck(
                        arm, arm_ik, _verb, 'upright', px, py, pz)
                    if _sc_pre is not None:
                        sys.stdout.write(_sc_line(_sc_pre))
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                    if ARM_WRITER_ENABLED:
                        try:
                            arm.hold_rc(True)
                        except Exception as _he:
                            sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                            sys.stdout.flush()
                            _zf_restore_floor()   # no motion yet
                            continue
                    hover_z = pz + APPROACH_HEIGHT_OFFSET
                    sys.stderr.write(
                        f'[pick_daemon] {_verb}: right hand holds — drop at '
                        f'x={px:.3f} y={py:.3f} z={pz:.3f} '
                        f'(hover z={hover_z:.3f})...\n')
                    sys.stderr.flush()
                    # 2. idle (left) arm out of the way at its CARRY_TUCK; move it
                    # only when it is not already settled there (the pair normally
                    # just finished a pick).
                    _pcur = arm.get_current_dual_arm_q().copy()
                    _pidle_tuck = _zf_idle_target(idle_side, _pcarry_idle)   # measured tuck target
                    _pidle_slice = (slice(0, 7) if idle_side == 'left'
                                    else slice(7, 14))
                    if np.max(np.abs(_pcur[_pidle_slice] -
                                     _pidle_tuck)) > 0.03:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb} step 2: park idle '
                            f'({idle_side}) arm at CARRY_TUCK...\n')
                        sys.stderr.flush()
                        _t5_joint_move(arm, idle_side, _pidle_tuck)
                        _zf_pin_idle(idle_side)   # the commanded park is the pin
                    else:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb} step 2: idle ({idle_side}) '
                            'arm already at CARRY_TUCK — skip park\n')
                        sys.stderr.flush()
                        _zf_pin_idle(idle_side, _pcur)   # measured = pin
                    # extra lift: the place's first motion raises the holding arm
                    # alone by min(2.5*H, PLACE_EXTRA_LIFT_MAX_M), object still
                    # held. runs after step 2 and before the ascend. it is transient
                    # clearance: pz and hover_z are not raised, and the arm gives
                    # the extra z back during ASCEND/DESCEND. never refuses: an
                    # unknown height or no verified solve means no raise.
                    _el_req, _el_got, _el_ok = _place_extra_lift(
                        arm, arm_ik, hold_side, _el_h_m, _verb,
                        log_prefix=f'{_verb}-extra-lift: ')
                    # 3. ASCEND: carry pose to upright wrist at [px, py, hover_z]
                    # (de-tuck up and over the drop point).
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 3: ASCEND — upright '
                        f'{hold_side} wrist to [{px:.3f}, {py:.3f}] at '
                        f'z={hover_z:.3f}...\n')
                    sys.stderr.flush()
                    _asc_t = make_wrist_target_upright_v4(
                        px, py, hover_z, _pdir)
                    if not solve_and_move(arm, arm_ik, _asc_t,
                                          f'{_verb}-ascend: ',
                                          side=hold_side):
                        sys.stdout.write(
                            f'ERROR:IK failed for {_verb} ascend{_zf_reply_suffix(_verb)}\n')
                        sys.stdout.flush()
                        # park the holding arm at CARRY_TUCK before the release
                        # (hand untouched, object still held); never raises.
                        _e6_unmoved = _ZF_CMD['moves'] == 0   # no motion yet
                        _safe_park_holding(arm, hold_side, f'{_verb} IK failure (ascend)')
                        _hold_rc_writer(arm, False,
                                        f'{_verb} IK failure (ascend)')
                        if _e6_unmoved:
                            _zf_restore_floor()   # keeps this clamp
                        continue
                    # 4. APPROACH-XY: the ascend target is already [px, py] at
                    # hover height, so the wrist is over the drop point; log and
                    # descend in place.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 4: APPROACH-XY — wrist '
                        f'over drop at [{px:.3f}, {py:.3f}], z={hover_z:.3f}\n')
                    sys.stderr.flush()
                    # 5. DESCEND straight down to the caller drop height pz (hand
                    # stays closed)
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 5: DESCEND — upright '
                        f'{hold_side} wrist to z={pz:.3f} (hand closed, '
                        'object held)...\n')
                    sys.stderr.flush()
                    _desc_t = make_wrist_target_upright_v4(px, py, pz, _pdir)
                    if not solve_and_move(arm, arm_ik, _desc_t,
                                          f'{_verb}-descend: ',
                                          side=hold_side):
                        sys.stdout.write(
                            f'ERROR:IK failed for {_verb} descend{_zf_reply_suffix(_verb)}\n')
                        sys.stdout.flush()
                        # park before release
                        _e6_unmoved = _ZF_CMD['moves'] == 0   # no motion yet
                        _safe_park_holding(arm, hold_side, f'{_verb} IK failure (descend)')
                        _hold_rc_writer(arm, False,
                                        f'{_verb} IK failure (descend)')
                        if _e6_unmoved:
                            _zf_restore_floor()   # keeps this clamp
                        continue
                    time.sleep(0.5)  # settle before the release
                    # 6. open the holding hand (right); verify is log-only since the
                    # release already happened, so never hard-fail after it.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 6: open {hold_side} '
                        'hand (release at the drop point)...\n')
                    sys.stderr.flush()
                    _phand.open()
                    _p_open_ok, _p_open_reason = _verify_hand_open(
                        _phand, hold_side, _popen_preset)
                    _note_release_open(hold_side, _p_open_ok)
                    if not _p_open_ok:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: {hold_side} hand open '
                            f'verify failed ({_p_open_reason}) — continuing\n')
                        sys.stderr.flush()
                    # 7. closeout 1/2 (RC stays on): both arms to LIFT. the empty
                    # holding arm rises clear and the idle arm leaves CARRY_TUCK on
                    # the same beat, one synchronized full-14-DOF joint move.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 7: closeout 1/2 — both '
                        'arms → LIFT (holding arm clears the drop point; idle '
                        'arm off CARRY_TUCK)...\n')
                    sys.stderr.flush()
                    _p_lift = np.concatenate(
                        [LIFT_LEFT, LIFT_RIGHT]).astype(np.float64)
                    move_arm_smooth(arm, _p_lift,
                                    _arm_gravity_ff(_p_lift, np.zeros(14)), zf_legacy=True)
                    # park the holding hand closed here: arm at LIFT, fingers clear
                    # of the drop point. gated by PLACE_PARK_HAND_CLOSED.
                    _park_hand_after_lift(_phand, hold_side, _verb, _p_open_ok)
                    # 8. two-arm closeout 2/2: both arms to NEUTRAL together.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 8: closeout 2/2 — both '
                        'arms → NEUTRAL together...\n')
                    sys.stderr.flush()
                    _p_neutral = np.concatenate(
                        [NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
                    _pn_q0 = arm.get_current_dual_arm_q().copy()  # stall check
                    move_arm_smooth(arm, _p_neutral,
                                    _arm_gravity_ff(_p_neutral, np.zeros(14)), zf_legacy=True)
                    # stall -> relax
                    _post_park_stall_check(
                        arm, _expected_after_move(_pn_q0, _p_neutral),
                        f'{_verb} closeout')
                    # 9. payload off, RC off, DONE.
                    try:
                        if hasattr(arm, 'payload'):
                            arm.payload(False)
                    except Exception as _pe:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: payload off failed: {_pe} '
                            '— continuing\n')
                        sys.stderr.flush()
                    # the holding hand is already parked closed at LIFT
                    # (_park_hand_after_lift), so do not close it a second time: it
                    # is at the close preset and a repeat costs 0.8 s of blocking
                    # ramp. PLACE_PARK_HAND_CLOSED covers why it is at LIFT.
                    _hold_rc_writer(arm, False,
                                    f'{_verb} complete (drop point)')
                    sys.stdout.write('DONE%s%s\n' % (_release_open_token(hold_side),
                                                   _zf_reply_suffix(_verb)))
                    sys.stdout.flush()

            # mirror of the place handler with the holding side left: same
            # reverse-pick waypoints (RC on, park idle right arm, ASCEND to hover z
            # over [px, py], APPROACH-XY log, DESCEND to pz, open left hand,
            # closeout both to LIFT then NEUTRAL, payload off, RC off, DONE). no
            # coords releases at the current carry pose, opening only the left hand.
            elif cmd == 'place_left' or cmd.startswith('place_left '):
                _zf_pick_begin('left', arm)
                parts = cmd.split()
                hold_side = 'left'
                idle_side = 'right'
                _phand = lhand
                _popen_preset = L_HAND_OPEN
                _pdir = -1
                _pcarry_idle = CARRY_TUCK_RIGHT
                _verb = 'place_left'
                if len(parts) == 1:
                    # no coords: no extra lift. the raise is approach clearance, and
                    # raising here would move the release point up by up to
                    # PLACE_EXTRA_LIFT_MAX_M. the raise lives only on the coords
                    # exits.
                    if ARM_WRITER_ENABLED:
                        try:
                            arm.hold_rc(True)
                        except Exception as _he:
                            sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                            sys.stdout.flush()
                            _zf_restore_floor()   # no motion yet
                            continue
                    sys.stderr.write(
                        f'[pick_daemon] {_verb}: drop-in-place at the current '
                        f'carry pose — open {hold_side} hand ONLY, payload '
                        'off, then TWO-ARM closeout to NEUTRAL\n')
                    sys.stderr.flush()
                    # extra lift deliberately absent here; see the note above
                    _phand.open()
                    _p_open_ok, _p_open_reason = _verify_hand_open(
                        _phand, hold_side, _popen_preset)
                    _note_release_open(hold_side, _p_open_ok)
                    if not _p_open_ok:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: {hold_side} hand open '
                            f'verify failed ({_p_open_reason}) — continuing\n')
                        sys.stderr.flush()
                    time.sleep(0.5)
                    try:
                        if hasattr(arm, 'payload'):
                            arm.payload(False)
                    except Exception as _pe:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: payload off failed: {_pe} '
                            '— continuing\n')
                        sys.stderr.flush()
                    # two-arm closeout (RC stays on): the holding arm lifts off the
                    # release point and the idle arm leaves CARRY_TUCK on the same
                    # beat, both to LIFT, then both to NEUTRAL.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} closeout 1/2: both arms → '
                        'LIFT (holding arm clears; idle arm off CARRY_TUCK)'
                        '...\n')
                    sys.stderr.flush()
                    _p_lift = np.concatenate(
                        [LIFT_LEFT, LIFT_RIGHT]).astype(np.float64)
                    move_arm_smooth(arm, _p_lift,
                                    _arm_gravity_ff(_p_lift, np.zeros(14)), zf_legacy=True)
                    # park the holding hand closed here (arm at LIFT, fingers
                    # clear). gated by PLACE_PARK_HAND_CLOSED.
                    _park_hand_after_lift(_phand, hold_side, _verb, _p_open_ok)
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} closeout 2/2: both arms → '
                        'NEUTRAL together...\n')
                    sys.stderr.flush()
                    _p_neutral = np.concatenate(
                        [NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
                    _pn_q0 = arm.get_current_dual_arm_q().copy()  # stall check
                    move_arm_smooth(arm, _p_neutral,
                                    _arm_gravity_ff(_p_neutral, np.zeros(14)), zf_legacy=True)
                    # stall -> relax
                    _post_park_stall_check(
                        arm, _expected_after_move(_pn_q0, _p_neutral),
                        f'{_verb} closeout')
                    _hold_rc_writer(arm, False,
                                    f'{_verb} complete (drop-in-place)')
                    sys.stdout.write('DONE%s%s\n' % (_release_open_token(hold_side),
                                                   _zf_reply_suffix(_verb)))
                    sys.stdout.flush()
                else:
                    # coords: reverse-pick waypoints. parse before RC on so a bad
                                        
                    # command never engages the arm.
                    try:
                        if len(parts) < 4:
                            raise ValueError(f'{_verb} requires x y z')
                        px = float(parts[1])
                        py = float(parts[2])
                        pz = float(parts[3])
                    except (ValueError, IndexError) as e:
                        sys.stdout.write(f'ERROR:{_verb} parse: {e}\n')
                        sys.stdout.flush()
                        _zf_restore_floor()   # no motion yet
                        continue
                    # optional height token: the object height for the extra lift.
                    # it is the durable height source; the in-process holder below
                    # is lost across a daemon restart.
                    # not part of the raise-on-bad-parse contract: an unparseable
                    # height means "unknown", not a malformed command, so the place
                    # still runs (warn-and-degrade); an absent token falls back to
                    # the holder.
                    _el_h_token = None
                    if len(parts) >= 5:
                        try:
                            _el_h_token = float(parts[4])
                        except (ValueError, IndexError):
                            sys.stderr.write(
                                f'[pick_daemon] {_verb}: height token '
                                f'{parts[4]!r} is not a number — treating the '
                                f'object height as unknown\n')
                            sys.stderr.flush()
                            _el_h_token = None
                    _el_h_m = (_el_h_token if _el_h_token is not None
                               else _PLACE_LAST_OBJECT_HEIGHT_M[0])
                    if ARM_WRITER_ENABLED:
                        try:
                            arm.hold_rc(True)
                        except Exception as _he:
                            sys.stdout.write(f'ERROR:hold_rc on: {_he}\n')
                            sys.stdout.flush()
                            _zf_restore_floor()   # no motion yet
                            continue
                    hover_z = pz + APPROACH_HEIGHT_OFFSET
                    sys.stderr.write(
                        f'[pick_daemon] {_verb}: left hand holds — drop at '
                        f'x={px:.3f} y={py:.3f} z={pz:.3f} '
                        f'(hover z={hover_z:.3f})...\n')
                    sys.stderr.flush()
                    # 2. idle (right) arm out of the way at its CARRY_TUCK; move it
                    # only when it is not already settled there.
                    _pcur = arm.get_current_dual_arm_q().copy()
                    _pidle_tuck = _zf_idle_target(idle_side, _pcarry_idle)   # measured tuck target
                    _pidle_slice = (slice(0, 7) if idle_side == 'left'
                                    else slice(7, 14))
                    if np.max(np.abs(_pcur[_pidle_slice] -
                                     _pidle_tuck)) > 0.03:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb} step 2: park idle '
                            f'({idle_side}) arm at CARRY_TUCK...\n')
                        sys.stderr.flush()
                        _t5_joint_move(arm, idle_side, _pidle_tuck)
                        _zf_pin_idle(idle_side)   # the commanded park is the pin
                    else:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb} step 2: idle ({idle_side}) '
                            'arm already at CARRY_TUCK — skip park\n')
                        sys.stderr.flush()
                        _zf_pin_idle(idle_side, _pcur)   # measured = pin
                    # extra lift: the place's first motion, holding arm alone,
                    # min(2.5*H, PLACE_EXTRA_LIFT_MAX_M), object still held. after step
                    # 2, before ascend; pz and hover_z are unchanged and it never
                    # refuses.
                    _el_req, _el_got, _el_ok = _place_extra_lift(
                        arm, arm_ik, hold_side, _el_h_m, _verb,
                        log_prefix=f'{_verb}-extra-lift: ')
                    # 3. ASCEND: carry pose to upright wrist at [px, py, hover_z].
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 3: ASCEND — upright '
                        f'{hold_side} wrist to [{px:.3f}, {py:.3f}] at '
                        f'z={hover_z:.3f}...\n')
                    sys.stderr.flush()
                    _asc_t = make_wrist_target_upright_v4(
                        px, py, hover_z, _pdir)
                    if not solve_and_move(arm, arm_ik, _asc_t,
                                          f'{_verb}-ascend: ',
                                          side=hold_side):
                        sys.stdout.write(
                            f'ERROR:IK failed for {_verb} ascend{_zf_reply_suffix(_verb)}\n')
                        sys.stdout.flush()
                        # park before release
                        _e6_unmoved = _ZF_CMD['moves'] == 0   # no motion yet
                        _safe_park_holding(arm, hold_side, f'{_verb} IK failure (ascend)')
                        _hold_rc_writer(arm, False,
                                        f'{_verb} IK failure (ascend)')
                        if _e6_unmoved:
                            _zf_restore_floor()   # keeps this clamp
                        continue
                    # 4. APPROACH-XY: the ascend target is already [px, py] at hover
                    # height, so the wrist is over the drop point; log and descend in
                    # place.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 4: APPROACH-XY — wrist '
                        f'over drop at [{px:.3f}, {py:.3f}], z={hover_z:.3f}\n')
                    sys.stderr.flush()
                    # 5. DESCEND straight down to the caller drop height pz (hand stays
                    # closed)
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 5: DESCEND — upright '
                        f'{hold_side} wrist to z={pz:.3f} (hand closed, '
                        'object held)...\n')
                    sys.stderr.flush()
                    _desc_t = make_wrist_target_upright_v4(px, py, pz, _pdir)
                    if not solve_and_move(arm, arm_ik, _desc_t,
                                          f'{_verb}-descend: ',
                                          side=hold_side):
                        sys.stdout.write(
                            f'ERROR:IK failed for {_verb} descend{_zf_reply_suffix(_verb)}\n')
                        sys.stdout.flush()
                        # park before release
                        _e6_unmoved = _ZF_CMD['moves'] == 0   # no motion yet
                        _safe_park_holding(arm, hold_side, f'{_verb} IK failure (descend)')
                        _hold_rc_writer(arm, False,
                                        f'{_verb} IK failure (descend)')
                        if _e6_unmoved:
                            _zf_restore_floor()   # keeps this clamp
                        continue
                    time.sleep(0.5)  # settle before the release
                    # 6. open the holding hand (left); verify is log-only.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 6: open {hold_side} '
                        'hand (release at the drop point)...\n')
                    sys.stderr.flush()
                    _phand.open()
                    _p_open_ok, _p_open_reason = _verify_hand_open(
                        _phand, hold_side, _popen_preset)
                    _note_release_open(hold_side, _p_open_ok)
                    if not _p_open_ok:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: {hold_side} hand open '
                            f'verify failed ({_p_open_reason}) — continuing\n')
                        sys.stderr.flush()
                    # 7. closeout 1/2: both arms to LIFT; the holding arm rises clear
                    # and the idle arm leaves CARRY_TUCK on the same beat, one
                    # synchronized 14-DOF joint move.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 7: closeout 1/2 — both '
                        'arms → LIFT (holding arm clears the drop point; idle '
                        'arm off CARRY_TUCK)...\n')
                    sys.stderr.flush()
                    _p_lift = np.concatenate(
                        [LIFT_LEFT, LIFT_RIGHT]).astype(np.float64)
                    move_arm_smooth(arm, _p_lift,
                                    _arm_gravity_ff(_p_lift, np.zeros(14)), zf_legacy=True)
                    # park the holding hand closed here (arm at LIFT, fingers clear).
                    # gated by PLACE_PARK_HAND_CLOSED.
                    _park_hand_after_lift(_phand, hold_side, _verb, _p_open_ok)
                    # 8. two-arm closeout 2/2: both arms to NEUTRAL together.
                    sys.stderr.write(
                        f'[pick_daemon] {_verb} step 8: closeout 2/2 — both '
                        'arms → NEUTRAL together...\n')
                    sys.stderr.flush()
                    _p_neutral = np.concatenate(
                        [NEUTRAL_LEFT, NEUTRAL_RIGHT]).astype(np.float64)
                    _pn_q0 = arm.get_current_dual_arm_q().copy()  # stall check
                    move_arm_smooth(arm, _p_neutral,
                                    _arm_gravity_ff(_p_neutral, np.zeros(14)), zf_legacy=True)
                    # stall -> relax
                    _post_park_stall_check(
                        arm, _expected_after_move(_pn_q0, _p_neutral),
                        f'{_verb} closeout')
                    # 9. payload off, RC off, DONE.
                    try:
                        if hasattr(arm, 'payload'):
                            arm.payload(False)
                    except Exception as _pe:
                        sys.stderr.write(
                            f'[pick_daemon] {_verb}: payload off failed: {_pe} '
                            '— continuing\n')
                        sys.stderr.flush()
                    # the holding hand is already parked closed at LIFT
                    # (_park_hand_after_lift). do not close again: it is at the close
                    # preset and a repeat costs 0.8 s of ramp.
                    _hold_rc_writer(arm, False,
                                    f'{_verb} complete (drop point)')
                    sys.stdout.write('DONE%s%s\n' % (_release_open_token(hold_side),
                                                   _zf_reply_suffix(_verb)))
                    sys.stdout.flush()

            # relax_arm: release a held or stalled arm for the gateway's idle stop
            # and watchdog. `hold_rc off` alone does not release the arm; it only
            # re-enables the RC sticks. this collapses the commanded target onto the
            # measured q (nothing moves except the setpoint, over slew_min), then
            # releases the RC interlock. hands untouched. one reply line:
            #   DONE:relax_arm relaxed=1 err_max=<%.3f|unknown> hold_rc=off
                        
            #   ERROR:relax_arm <reason>      (hold_rc off still attempted)
            # err_max = how far the measured arm sat from the last recorded
            # target (the stall the relax released); `unknown` when there is no
            # record, or the arm has moved since the record.
            elif cmd == 'relax_arm':
                try:
                    _ra_q = np.array(arm.get_current_dual_arm_q(),
                                     dtype=np.float64)
                    _ra_err = _last_cmd_err(_ra_q)
                    if _relax_arm_to_measured(arm, 'relax_arm', q=_ra_q) is None:
                        _ra_reply = ('ERROR:relax_arm relax failed: %s'
                                     % _LAST_RELAX_ERROR[0])
                    else:
                        _ra_reply = ('DONE:relax_arm relaxed=1 err_max=%s '
                                     'hold_rc=off' % _ra_err)
                except Exception as _rae:
                    _ra_reply = 'ERROR:relax_arm %s' % (_rae,)
                if ARM_WRITER_ENABLED:
                    try:
                        arm.hold_rc(False)
                    except Exception as _rce:
                        sys.stderr.write(
                            f'[pick_daemon] relax_arm: hold_rc off FAILED: '
                            f'{_rce}\n')
                        sys.stderr.flush()
                        if _ra_reply.startswith('DONE'):
                            _ra_reply = ('ERROR:relax_arm hold_rc off failed: '
                                         '%s' % (_rce,))
                sys.stdout.write(_ra_reply + '\n')
                sys.stdout.flush()

            # ── grip_evidence: hand evidence from the last right pick ───
            # evidence only; the pick reply itself stays a bare DONE, which the
            # gateway compares exactly.
            elif cmd == 'grip_evidence':
                _ge_last = _LAST_GRIP_EVIDENCE[0]
                sys.stdout.write('DONE:' + json.dumps(
                    _ge_last if _ge_last is not None else {},
                    separators=(',', ':')) + '\n')
                sys.stdout.flush()

            else:
                sys.stdout.write(f'ERROR:unknown command "{cmd}"\n')
                sys.stdout.flush()

        except engine_cancel.Cancelled as e:
            # ── mid-task cancel unwind ───────────────────────────────────
            # raised by _check_cancel at a motion choke point. the stop is
            # freeze-in-place: the writer halts the in-flight slew at its current
            # interpolated position and holds it.
            # not a retreat to a park pose: the hand may be loaded and moving a
            # grasped object is riskier than holding still; it may be mid-press,
            # where a retreat would drag; and this hardware has no empty/loaded
            # discriminator, so "retreat only if empty" cannot be decided.
            # freeze-then-release is also the safe state for the RC interlock:
            # manual control returns with the arm at a pose it already holds.
            if ARM_WRITER_ENABLED:
                try:
                    arm.freeze()
                except Exception as _e:
                    sys.stderr.write(f'[pick_daemon] cancel freeze failed: {_e}\n')
                    sys.stderr.flush()
            # the freeze leaves the writer PD-holding its interpolated target,
            # which can sit short of the measured pose (blocked arm means stall
            # torque). collapse the target onto the measured q. no retreat.
            _relax_arm_to_measured(arm, 'cancel')
            _hold_rc_writer(arm, False, 'cancel')
            # cleared after the arm is stopped and the interlock released: clearing
            # earlier would drop a second cancel that arrived during the unwind,
            # and pressing twice is reasonable.
            engine_cancel.clear()
            sys.stderr.write(
                f'[pick_daemon] CANCELLED at phase "{e.where}" — arm frozen in '
                'place and relaxed to the measured pose, RC interlock released '
                '(manual control restored)\n')
            sys.stderr.flush()
            # not an ERROR: _shape_reply flips the envelope to ok:false on the
            # substring "ERROR:" anywhere, and the verifier classifies that as
            # failure. a cancel is a successful outcome.
            sys.stdout.write(f'CANCELLED:{e.where}\n')
            sys.stdout.flush()

        except Exception as e:
            # best-effort safety on any unhandled error: freeze an in-flight writer
            # slew, then release the RC interlock (manual recovery). a dead or
            # clamped writer is recoverable with `hold_rc off`.
            if ARM_WRITER_ENABLED:
                try:
                    arm.freeze()
                except Exception as _e:
                    sys.stderr.write(f'[pick_daemon] freeze failed: {_e}\n')
                    sys.stderr.flush()
            # frozen short of target means stall torque; relax onto the measured q
            # (never raises; a dead writer just logs).
            _relax_arm_to_measured(arm, 'exception')
            _hold_rc_writer(arm, False, 'exception')
            sys.stderr.write(f'[pick_daemon] Exception: {e}\n')
            sys.stderr.flush()
            if _reply_sent:
                sys.stderr.write(
                    '[pick_daemon] ERROR reply suppressed — this command '
                    'already sent its one stdout reply\n')
                sys.stderr.flush()
            else:
                if _ZF_CMD['moves'] == 0:
                    _zf_restore_floor()   # an exception before any arm motion
                _zf_verb = (cmd.split() or [''])[0]
                sys.stdout.write(f'ERROR:{e}{_zf_reply_suffix(_zf_verb)}\n')
                sys.stdout.flush()
        finally:
            _zf_cmd_end(cmd)   # one summary line per command


# ═════════════════════════════════════════════════════════════════════════════
# safety on every exit, reach precheck, post-park stall check and grip
# evidence.
# defined after main() because main() resolves these names at call time, once
# the whole module has executed.
# an abort that writes ERROR and continues without releasing RC or moving the
# arm leaves the writer PD-holding its target, so the arm stays pressed and
# heats up. releasing RC before a long READY slew has the same problem.
# rules: park first, release last (interlock in `finally` after the park); the
# unwind is not cancellable, since a Cancelled after the reply would emit a
# second stdout line; relax-to-measured is the only way to remove stall torque
# without new motion.
# ═════════════════════════════════════════════════════════════════════════════

# relax slew length: the writer clamps every slew to at least arm_slew_min_s,
# 1.5 s.
ARM_RELAX_DURATION_S = 1.5
# post-park stall check. static tracking error is 0.052-0.057 rad and stall
# error +0.205/+0.224 rad, so 0.10 sits between the two.
ARM_STALL_SETTLE_S = 1.0
ARM_STALL_ERR_RAD = 0.10
# retract trail: measured q after each reached approach waypoint of the current
# right pick, replayed in reverse on failure or a lift-pin-abort park.
SAFE_TRAIL_MAX = 4
SAFE_TRAIL_SKIP_RAD = 0.05
_SAFE_TRAIL = []
# the last commanded 14-DOF target known to this block: None or
# (target_q, measured_q_at_record|None). recorded by _relax_arm_to_measured,
# _safe_move and _post_park_stall_check. move_arm_smooth records nothing, so
# the pose verbs above move the arm without updating the record; the stored
# measured q is what exposes that (_last_cmd_err).
_LAST_CMD_Q = [None]
_LAST_RELAX_ERROR = [None]
# the record is trusted only while the arm is still at the measured pose.
# static servo error of a held arm is 0.052-0.057 rad while a resting arm
# moves far less between reads. more than this means motion the record does
# not describe, so relax_arm reports err_max=unknown rather than a number
# against a stale target.
ARM_CMD_RECORD_DRIFT_RAD = 0.05
# grip evidence from the last right pick (`grip_evidence` verb)
_LAST_GRIP_EVIDENCE = [None]
_GRIP_EVIDENCE_SEQ = [0]


def _safe_log(msg):
    try:
        sys.stderr.write(msg if msg.endswith('\n') else msg + '\n')
        sys.stderr.flush()
    except Exception:
        pass


def _note_cmd(target_q, meas_q=None):
    """Record the last commanded target (and the q measured with it)."""
    try:
        _LAST_CMD_Q[0] = (
            np.array(target_q, dtype=np.float64).copy(),
            None if meas_q is None else np.array(meas_q, dtype=np.float64).copy())
    except Exception:
        _LAST_CMD_Q[0] = None


def _last_cmd_err(q_now):
    """'%.3f' = max|recorded target - q_now| (the stall a relax releases), or
    'unknown' when there is no record, or when the arm has moved more than
    ARM_CMD_RECORD_DRIFT_RAD since it was taken (the record is stale).
    Never raises."""
    rec = _LAST_CMD_Q[0]
    if rec is None:
        return 'unknown'
    try:
        tgt, meas = rec
        q_now = np.asarray(q_now, dtype=np.float64)
        if meas is not None and float(np.max(np.abs(
                q_now - np.asarray(meas, dtype=np.float64)))) > ARM_CMD_RECORD_DRIFT_RAD:
            return 'unknown'
        return '%.3f' % float(np.max(np.abs(
            np.asarray(tgt, dtype=np.float64) - q_now)))
    except Exception:
        return 'unknown'


def _relax_arm_to_measured(arm, ctx, q=None):
    """Collapse the commanded arm target onto the MEASURED q.

    The writer PD-holds `arm.current` forever, so an arm stopped short of its
    target (blocked by the table, the object, or its own body) keeps pushing
    with the full position-error torque.  Re-commanding the measured pose is
    the only way to remove that torque without moving anywhere new.  Writer
    mode: a slew_start (non-blocking) over ARM_RELAX_DURATION_S with the
    gravity FF of the measured pose.  Standalone: one ctrl_dual_arm.

    Never raises.  Returns the q it commanded, or None on failure (the reason
    is kept in _LAST_RELAX_ERROR[0] and logged).
    """
    _LAST_RELAX_ERROR[0] = None
    try:
        if q is None:
            q = arm.get_current_dual_arm_q()
        q = np.array(q, dtype=np.float64)
        tau = _arm_gravity_ff(q, np.zeros(14))
        if ARM_WRITER_ENABLED:
            arm.slew_start(q, duration=ARM_RELAX_DURATION_S, tau=tau)
        else:
            arm.ctrl_dual_arm(q, tau)
        _note_cmd(q, q)
        _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): relaxed to measured q '
                  f'(commanded target := measured, '
                  f'{ARM_RELAX_DURATION_S:.1f} s)')
        return q
    except Exception as _e:
        _LAST_RELAX_ERROR[0] = repr(_e)
        _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): relax to measured q '
                  f'FAILED: {_e!r}')
        return None


def _safe_move(arm, target_q, ctx, duration=None):
    """Blocking, NON-CANCELLABLE move used only by the safe-exit unwinds.

    Same z-floor clamp as move_arm_smooth, then the writer slew with
    abort=None (or an interpolation loop with no cancel probe in standalone
    mode).  It never polls the cancel flag on purpose: these moves run AFTER
    the command's reply was written, and a Cancelled raised here would make
    the command loop's handler write a second stdout line.

    May raise (writer failure).  Returns the commanded (clamped) target.
    """
    start_q = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
    tgt = np.array(target_q, dtype=np.float64)
    tgt, _clamped = _apply_zfloor(start_q, tgt, 'safe_exit:' + ctx, zf_legacy=True)
    tgt = np.array(tgt, dtype=np.float64)
    tau = _arm_gravity_ff(tgt, np.zeros(14))
    max_delta = float(np.max(np.abs(tgt - start_q)))
    if duration is None:
        duration = max(1.5, max_delta / MOVE_SPEED)
    _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): non-cancellable move '
              f'max_delta={max_delta:.3f} rad duration={duration:.1f}s')
    # clearing the tracked target first: the writer's target is about to
    # change, and a move that raises leaves the target unknown.
    _LAST_CMD_Q[0] = None
    if ARM_WRITER_ENABLED:
        arm.slew(tgt, duration=duration, tau=tau, abort=None)
    else:
        steps = max(1, int(duration * MOVE_HZ))
        dt = duration / steps
        for i in range(1, steps + 1):
            r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
            arm.ctrl_dual_arm(start_q + r * (tgt - start_q), tau)
            time.sleep(dt)
        arm.ctrl_dual_arm(tgt, tau)
    _note_cmd(tgt)
    return tgt


def _trail_reset():
    del _SAFE_TRAIL[:]


def _trail_note(arm, tag):
    """Record the MEASURED q after a reached waypoint (never raises)."""
    try:
        q = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
        _SAFE_TRAIL.append((tag, q))
        while len(_SAFE_TRAIL) > SAFE_TRAIL_MAX:
            del _SAFE_TRAIL[0]
    except Exception as _e:
        _safe_log(f'[pick_daemon] SAFE-EXIT trail note {tag} skipped: {_e!r}')


def _retract_trail(arm, ctx):
    """Replay the recorded waypoints in REVERSE (retract along the approach).

    Entries within SAFE_TRAIL_SKIP_RAD of the current q are skipped.  The
    trail is cleared first, so a failure part-way can never leave a stale
    trail for a later command.  May raise (from _safe_move).  Returns the
    number of retract moves commanded.
    """
    trail = list(_SAFE_TRAIL)
    del _SAFE_TRAIL[:]
    n = 0
    for tag, q in reversed(trail):
        cur = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
        if float(np.max(np.abs(np.asarray(q) - cur))) <= SAFE_TRAIL_SKIP_RAD:
            _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): retract {tag} '
                      f'skipped (already within {SAFE_TRAIL_SKIP_RAD:.2f} rad)')
            continue
        _safe_move(arm, q, f'{ctx}:retract {tag}')
        n += 1
    return n


def _post_park_stall_check(arm, expected_q, ctx):
    """After a park/tuck: if the arm sits > ARM_STALL_ERR_RAD from where it
    was commanded, it is stalled against something — relax to measured.

    The software stall watchdog that needs no writer change.  Never raises.
    Returns the max joint error (rad), or None when it could not be read.
    `expected_q` is the target the caller just commanded, so it is also
    recorded (with the q measured here) as the engine's last commanded target.
    """
    try:
        if expected_q is None:
            return None
        time.sleep(ARM_STALL_SETTLE_S)
        q = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
        d = np.abs(q - np.array(expected_q, dtype=np.float64))
        j = int(np.argmax(d))
        e = float(d[j])
        _note_cmd(expected_q, q)
        if e > ARM_STALL_ERR_RAD:
            _relax_arm_to_measured(arm, ctx, q=q)
            _safe_log(f'[pick_daemon] ARM STALL after {ctx}: joint {j} err '
                      f'{e:.3f} rad — relaxed to measured')
        return e
    except Exception as _e:
        _safe_log(f'[pick_daemon] stall check after {ctx} skipped: {_e!r}')
        return None


def _safe_park_holding(arm, side, ctx):
    """Place IK failure while HOLDING: relax, then the holding side's
    CARRY_TUCK (the idle side stays at its measured q).  The hand is NOT
    touched.  Never raises (the caller's ASCEND_TAIL release must run).
    """
    try:
        q = _relax_arm_to_measured(arm, ctx)
        if q is None:
            q = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
        tgt = np.array(q, dtype=np.float64).copy()
        if side == 'left':
            tgt[0:7] = CARRY_TUCK_LEFT
        else:
            tgt[7:14] = CARRY_TUCK_RIGHT
        got = _safe_move(arm, tgt, ctx)
        _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): holding ({side}) arm '
                  'parked at CARRY_TUCK, hand untouched')
        _post_park_stall_check(arm, got, ctx)
        return got
    except Exception as _e:
        _safe_log(f'[pick_daemon] SAFE-EXIT({ctx}): park of the holding arm '
                  f'FAILED: {_e!r}')
        return None


def _precheck_solve(arm_ik, idle_T, target_T, warm):
    """3 wash solves (warm = previous output), no wrist-GN, stdout muted."""
    q = np.array(warm, dtype=np.float64)
    for _ in range(3):
        with _SuppressStdout():
            sol, _tau = arm_ik.solve_ik(
                idle_T, target_T,
                current_lr_arm_motor_q=q.copy(),
                current_lr_arm_motor_dq=np.zeros(14))
        if sol is None:
            raise RuntimeError('solve_ik returned None')
        q = np.array(sol, dtype=np.float64)
    return q


def _precheck_residual(arm_ik, q, score_T):
    r = verify_ik_target(arm_ik, q, 'right', score_T)[0]
    if r is None:
        raise RuntimeError('FK verify failed')
    return float(r)


# horizontal palm rule + upright rim diagnostics; module-level, so the pick
# handler resolves these names at call time.
# the palm ends 2 cm below the object's top with every digit 2 cm off the table.
# URDF geometry, middle-down relative to R_ee: palm underside 1.73 cm below
# R_ee, open thumb tip 11.41 cm below. height_m of a lying object is its
# vertical thickness. every digit 2 cm off the table needs h >= ~0.137 m, so
# flatter objects are infeasible and the descent target is left unchanged.
# a module constant rather than an env switch: the engine runs with a filtered
# environment.
H_RULE_MODE = 'on'            # 'on' | 'log' (plan logged, never applied) | 'off'
H_PALM_BELOW_REE = 0.0173     # m, palm underside below R_ee (middle-down)
H_THUMB_BELOW_REE = 0.1141    # m, open-thumb hull below R_ee (middle-down)
H_PALM_BELOW_TOP = 0.02       # m, palm 2 cm below the object's top
H_DIGIT_CLEAR = 0.02          # m, digits 2 cm off the table
H_RULE_H_MIN = 0.008          # m, plausible lying-object thickness band
H_RULE_H_MAX = 0.20


def _h_palm_plan(tz, height_m):
    """B8i-H: the palm-rule plan for one right horizontal h-wp3 (pelvis-frame
    metres) or None (height unknown / outside [H_RULE_H_MIN, H_RULE_H_MAX])."""
    try:
        h = float(height_m)
        tz = float(tz)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(h) and math.isfinite(tz)):
        return None
    if not (H_RULE_H_MIN <= h <= H_RULE_H_MAX):
        return None
    zc = tz - GRASP_Z_OFFSET + SOLVER_Z_TORSO_OFFSET   # torso_link 0.044 m above pelvis (URDF)
    top = zc + h / 2.0
    table = zc - h / 2.0
    ee_rule = top - H_PALM_BELOW_TOP + H_PALM_BELOW_REE
    floor = table + H_DIGIT_CLEAR
    ee_guard = floor + H_THUMB_BELOW_REE
    return {'zc': zc, 'top': top, 'table': table, 'ee_rule': ee_rule,
            'floor': floor, 'ee_guard': ee_guard, 'feasible': ee_rule >= ee_guard}


def _h_rule_tz(tz, height_m, mode=None):
    """B8i-H: the h-wp3 wrist target z - the plan's ee_rule when the mode is
    'on' and the plan is feasible, else tz (legacy).  One '[hrule] plan' line
    unless the mode is 'off'.  Never raises."""
    mode = H_RULE_MODE if mode is None else mode
    if mode == 'off':
        return tz
    try:
        plan = _h_palm_plan(tz, height_m)
        if plan is None:
            sys.stderr.write(
                f'[pick_daemon][hrule] plan mode={mode} h={height_m!r} '
                f'tz={float(tz):+.4f} -> legacy (height unknown or outside '
                f'[{H_RULE_H_MIN:.3f}, {H_RULE_H_MAX:.2f}] m)\n')
            sys.stderr.flush()
            return tz
        use = (mode == 'on' and plan['feasible'])
        out = plan['ee_rule'] if use else tz
        sys.stderr.write(
            f'[pick_daemon][hrule] plan mode={mode} h={float(height_m):.4f} '
            f'tz={float(tz):+.4f} zc={plan["zc"]:+.4f} top={plan["top"]:+.4f} '
            f'table={plan["table"]:+.4f} ee_rule={plan["ee_rule"]:+.4f} '
            f'ee_guard={plan["ee_guard"]:+.4f} feasible={plan["feasible"]} -> '
            f'{"rule" if use else "legacy"} target z={float(out):+.4f}\n')
        sys.stderr.flush()
        return out
    except Exception as _e:                                # noqa: BLE001
        sys.stderr.write(f'[pick_daemon][hrule] plan failed ({_e}) -> legacy\n')
        sys.stderr.flush()
        return tz


def _h_rule_close(arm, arm_ik, h_tz, tz, height_m):
    """B8i-H telemetry after the right h-wp3 descent: the MEASURED wrist z, the
    lowest hand/wrist hull point (_lowest_limb_z) and the estimated table.
    Diagnostic only; never raises, never moves."""
    if H_RULE_MODE == 'off':
        return
    try:
        _q = arm.get_current_dual_arm_q().copy()
        _w = get_wrist_pose(arm_ik, _q, side='right')
        _lo = _lowest_limb_z(arm_ik, _q, 'right')
        _plan = _h_palm_plan(tz, height_m)
        _fmt = lambda v: ('%+.4f' % float(v)) if v is not None else 'none'
        sys.stderr.write(
            f'[pick_daemon][hrule] close wrist_z={_fmt(_w[2] if _w is not None else None)} '
            f'target={_fmt(h_tz)} legacy_tz={_fmt(tz)} lowest_limb={_fmt(_lo)} '
            f'table_est={_fmt(_plan["table"] if _plan is not None else None)}\n')
        sys.stderr.flush()
    except Exception as _e:                                # noqa: BLE001
        sys.stderr.write(f'[pick_daemon][hrule] close skipped ({_e})\n')
        sys.stderr.flush()


def _rim_diag(arm, arm_ik, stage, target_T):
    """B8i (10-09 cup rim): the MEASURED right wrist (get_ee_pose of the measured
    q, the frame verify_ik_target compares) against the stage's commanded
    target, so the next cup pick shows whether wp3 / wp4 was held off its
    target (rim or wall contact).  Diagnostic only; never raises, never moves."""
    try:
        _q = arm.get_current_dual_arm_q().copy()
        _T = get_ee_pose(arm_ik, _q, 'right')
        if _T is None:
            raise RuntimeError('FK failed')
        _p = np.asarray(_T.translation, dtype=np.float64)
        _t = np.asarray(target_T, dtype=np.float64)[:3, 3]
        _d = _p - _t
        sys.stderr.write(
            f'[pick_daemon][rimdiag] {stage} measured R_ee=({_p[0]:.4f},{_p[1]:.4f},{_p[2]:.4f}) '
            f'target=({_t[0]:.4f},{_t[1]:.4f},{_t[2]:.4f}) '
            f'd=({_d[0]:+.4f},{_d[1]:+.4f},{_d[2]:+.4f}) |d|={float(np.linalg.norm(_d)):.4f}\n')
        sys.stderr.flush()
    except Exception as _e:                                # noqa: BLE001
        sys.stderr.write(f'[pick_daemon][rimdiag] {stage} skipped ({_e})\n')
        sys.stderr.flush()


# horizontal pick ladder ("hladder"), right-arm horizontal picks only; the
# pick handler resolves these names at call time.
# 3 attempts: 1 aims at the command; 2-3 aim at the re-detected base offset
# by H_LADDER_DXYZ and descend below attempt 1's h-wp3 target by dz
# (cumulative 2.0 + 1.5 cm lower). the base is the median of clean rows taken
# while the hand is up at h-wp1, else the command aim; every re-aimed target
# is reach-prechecked at its lowered z.
# only the descent target moves: the floor latch and z-floor clamp are
# untouched, so on a flat object the clamp binds and lowering is a no-op
# (the [hladder] descent line reports clamp=bound, lower ~0).
# a keep-closed re-press is +2 cm forward of the capturing aim, at the
# command's tz, and is always the last attempt. upright picks keep 2
# attempts, +2 cm; the left arm is untouched. flip the sign or axis of the
# rungs in H_LADDER_DXYZ only.
# kill switch, read at every pick: the H_LADDER_OFF_FLAG file exists, or
# OPENBOT_H_LADDER=0 in the engine's environment. the engine runs with a
# fixed environment, so a shell export does not reach it and the flag file
# is the operative switch (next pick, no restart).

# re-detection sources, the first fresh one wins:
#   H_LADDER_ROWS_FILE  {"t": <write epoch>, "rows": [body rows]}, one row
#     per detection; absent or stale is skipped;
#   H_LADDER_DET_FILE  one max-confidence row per message, written only for
#     the pickable classes; no stamp, so the file mtime is the row time.
# a row is clean when it carries the pick-start target's label (the instance
# id is not a gate: it changes when the hand hides the object for more than
# 2 s), arrives at least H_LADDER_SETTLE_S after the hand reached h-wp1
# and at most H_LADDER_MAX_AGE_S ago, says horizontal if it says, lies within
# H_LADDER_MAX_DZ_M of the command z and inside the command height band, and
# is within H_LADDER_MAX_JUMP_M of the last clean pose.

H_LADDER_ATTEMPTS = 3
H_LADDER_DXYZ = ((0.000, 0.000, -0.020), (0.000, 0.000, -0.035))   # attempts 2, 3: (dx, dy, dz) m, dz below attempt 1's target
H_LADDER_REPRESS_DY = 0.02         # keep-closed re-press: +2 cm forward
H_LADDER_ENV = 'OPENBOT_H_LADDER'
H_LADDER_OFF_FLAG = '/tmp/openbot_h_ladder_off'
H_LADDER_ROWS_FILE = '/tmp/openbot_body_rows.json'
H_LADDER_DET_FILE = '/tmp/pick_latest_detection.json'
H_LADDER_REF_WAIT_S = 0.3          # pick start: look this long for the target's row
H_LADDER_REF_MATCH_M = 0.03        # ... within this distance of the command (x, y)
H_LADDER_SETTLE_S = 0.5            # rows written this long after the hand left the view
H_LADDER_WINDOW_S = 1.0            # ... collected for at most this long after that
H_LADDER_POLL_S = 0.1
H_LADDER_MAX_AGE_S = 1.0
H_LADDER_MIN_ROWS = 3              # distinct clean rows (packets) for a fresh base
H_LADDER_MAX_DZ_M = 0.03
H_LADDER_MAX_DH_M = 0.015          # |h - command h| <= max(this, FRAC * command h)
H_LADDER_MAX_DH_FRAC = 0.35
H_LADDER_MAX_JUMP_M = 0.08
H_LADDER_MAX_SPREAD_M = 0.02


def _hl_log(msg):
    sys.stderr.write('[pick_daemon][hladder] ' + msg + '\n')
    sys.stderr.flush()


def _hl_switch(_environ=None, _exists=None):
    """(on, why) -- the hladder kill switch.  Never raises (an unreadable
    switch is 'on', the shipped default)."""
    try:
        env = os.environ if _environ is None else _environ
        v = str(env.get(H_LADDER_ENV, '')).strip().lower()
        if v in ('0', 'off', 'false', 'no'):
            return False, 'env %s=%s' % (H_LADDER_ENV, v)
    except Exception:
        pass
    try:
        if (os.path.exists if _exists is None else _exists)(H_LADDER_OFF_FLAG):
            return False, 'flag %s' % H_LADDER_OFF_FLAG
    except Exception:
        pass
    return True, 'default'


def _hl_rung(k):
    """(dx, dy, dz) of attempt k (2 or 3): (dx, dy) on its base, dz on attempt 1's
    descent target."""
    return H_LADDER_DXYZ[min(max(k - 2, 0), len(H_LADDER_DXYZ) - 1)]


def _hl_rows(now, _open=None, _mtime=None):
    """([(row, t)], src) from the first fresh source: 'rows', else 'det', else
    ([], 'none').  Never raises (pick_node writes 'det' non-atomically)."""
    _open = open if _open is None else _open
    _mtime = (lambda p: os.stat(p).st_mtime) if _mtime is None else _mtime
    try:
        with _open(H_LADDER_ROWS_FILE) as f:
            doc = json.load(f)
        t = float(doc.get('t'))
        rows = doc.get('rows')
        if (math.isfinite(t) and isinstance(rows, list)
                and -0.5 <= now - t <= H_LADDER_MAX_AGE_S):
            return [(r, t) for r in rows if isinstance(r, dict)], 'rows'
    except Exception:
        pass
    try:
        t = float(_mtime(H_LADDER_DET_FILE))
        with _open(H_LADDER_DET_FILE) as f:
            row = json.load(f)
        if isinstance(row, dict) and math.isfinite(t):
            return [(row, t)], 'det'
    except Exception:
        pass
    return [], 'none'


def _hl_xyz(row):
    """(x, y, z) floats of a row, or None (missing / non-finite)."""
    try:
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return x, y, z


def _hl_row_problem(row, t, now, t_min, st):
    """None when the row is CLEAN for st, else the first failing gate's name
    (shape, stale, early, orient, dz, dh, jump).  The label / instance match
    is the caller's (_hl_pick_row)."""
    p = _hl_xyz(row)
    if p is None:
        return 'shape'
    x, y, z = p
    if now - t > H_LADDER_MAX_AGE_S:
        return 'stale'
    if t < t_min:
        return 'early'
    o = row.get('orientation')
    if o is not None and o != 'horizontal':
        return 'orient'
    if abs(z - st['z_cmd']) > H_LADDER_MAX_DZ_M:
        return 'dz'
    h, hc = row.get('height_m'), st.get('h_cmd')
    if h is not None and hc is not None:
        try:
            hv = float(h)
        except (TypeError, ValueError):
            return 'dh'
        if (not math.isfinite(hv)
                or abs(hv - hc) > max(H_LADDER_MAX_DH_M, H_LADDER_MAX_DH_FRAC * hc)):
            return 'dh'
    lx, ly = st['last_xy']
    if math.hypot(x - lx, y - ly) > H_LADDER_MAX_JUMP_M:
        return 'jump'
    return None


def _hl_is_target(row, st):
    """The pick-start label.  The instance id is logged, NOT gated: s3 10-10 the
    banana was id 15 at the pick start and id 24 after the failed pick
    (gateway.log:62/68) -- the hand hid it 2.76 s (depth.log:6077-6078) and the
    tracker evicts after 2.0 s (instance_tracker.py evict_s)."""
    return row.get('label') == st.get('ref_label')


def _hl_pick_row(rows, t, now, t_min, st):
    """One packet -> ((x, y), None) for its clean target row nearest the last
    clean pose, or (None, reason): 'absent' (no row of the target) or the
    failing gate of the nearest target row."""
    lx, ly = st['last_xy']
    best = None
    worst = None
    for row in rows:
        if not _hl_is_target(row, st):
            continue
        p = _hl_xyz(row)
        d = math.hypot(p[0] - lx, p[1] - ly) if p is not None else float('inf')
        why = _hl_row_problem(row, t, now, t_min, st)
        if why is None:
            if best is None or d < best[0]:
                best = (d, p[0], p[1])
        elif worst is None or d < worst[0]:
            worst = (d, why)
    if best is not None:
        return (best[1], best[2]), None
    return None, (worst[1] if worst is not None else 'absent')


def _hl_median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _hl_redetect(st, t_view, _now=None, _sleep=None, _rows=None, _cancel=None):
    """((bx, by), info) -- the median of >= H_LADDER_MIN_ROWS clean rows of the
    target written in [t_view + SETTLE, t_view + SETTLE + WINDOW] -- or
    (None, why).  Waits at most SETTLE + WINDOW + one poll (also counted, not
    only timed); a stop is honoured every poll (_check_cancel raises
    engine_cancel.Cancelled)."""
    _now = time.time if _now is None else _now
    _sleep = time.sleep if _sleep is None else _sleep
    _rows = _hl_rows if _rows is None else _rows
    _cancel = _check_cancel if _cancel is None else _cancel
    if st.get('ref_label') is None:
        return None, 'no_ref'
    t_min = t_view + H_LADDER_SETTLE_S
    deadline = t_min + H_LADDER_WINDOW_S
    seen = set()
    clean = []
    rej = {}
    src = 'none'
    # bounded by count as well as the clock: a wall clock that stalls or steps
    # back (NTP) can never hold the arm at h-wp1 past SETTLE + WINDOW + one poll.
    
    for _poll in range(int(round((H_LADDER_SETTLE_S + H_LADDER_WINDOW_S)
                                 / H_LADDER_POLL_S)) + 1):
        _cancel('hladder_redetect')
        now = _now()
        if now >= t_min:
            samples, src = _rows(now)
            packets = {}
            for row, t in samples:
                packets.setdefault(t, []).append(row)
            for t in sorted(packets):
                if (src, t) in seen:
                    continue
                seen.add((src, t))
                xy, why = _hl_pick_row(packets[t], t, now, t_min, st)
                if xy is not None:
                    clean.append(xy)
                else:
                    rej[why] = rej.get(why, 0) + 1
            if len(clean) >= H_LADDER_MIN_ROWS or now >= deadline:
                break
        _sleep(H_LADDER_POLL_S)
    _rj = ','.join('%s:%d' % kv for kv in sorted(rej.items())) or '-'
    if len(clean) < H_LADDER_MIN_ROWS:
        return None, 'src=%s clean=%d/%d rej=%s' % (src, len(clean),
                                                     H_LADDER_MIN_ROWS, _rj)
    bx = _hl_median([c[0] for c in clean])
    by = _hl_median([c[1] for c in clean])
    spread = max(math.hypot(c[0] - bx, c[1] - by) for c in clean)
    if spread > H_LADDER_MAX_SPREAD_M:
        return None, 'src=%s clean=%d spread=%.3f>%.3f' % (
            src, len(clean), spread, H_LADDER_MAX_SPREAD_M)
    lx, ly = st['last_xy']
    return (bx, by), 'src=%s n=%d bx=%.3f by=%.3f spread=%.3f jump=%.3f rej=%s' % (
        src, len(clean), bx, by, spread, math.hypot(bx - lx, by - ly), _rj)


def _hl_log_attempt(st, k, base, dx, dy, dz, why):
    x, y = st['aim']
    _hl_log('attempt %d/%d base=%s x=%.3f y=%.3f dy=%+.3f dx=%+.3f dz=%+.3f why=%s'
            % (k, st['n'], base, x, y, dy, dx, dz, why))


def _hl_find_ref(st, _now, _sleep, _rows):
    """The target's row at pick start (the command's (x, y) within
    H_LADDER_REF_MATCH_M, z within H_LADDER_MAX_DZ_M, fresh, horizontal if it
    says): (label, instance_id or None, why) or (None, None, why)."""
    deadline = _now() + H_LADDER_REF_WAIT_S
    src = 'none'
    n = int(round(H_LADDER_REF_WAIT_S / H_LADDER_POLL_S)) + 1
    for i in range(n):
        now = _now()
        samples, src = _rows(now)
        best = None
        for row, t in samples:
            p = _hl_xyz(row)
            lab = row.get('label')
            if p is None or not isinstance(lab, str) or not lab:
                continue
            if now - t > H_LADDER_MAX_AGE_S or abs(p[2] - st['z_cmd']) > H_LADDER_MAX_DZ_M:
                continue
            o = row.get('orientation')
            if o is not None and o != 'horizontal':
                continue
            d = math.hypot(p[0] - st['x0'], p[1] - st['y0'])
            if d <= H_LADDER_REF_MATCH_M and (best is None or d < best[0]):
                best = (d, lab, row.get('instance_id'), now - t)
        if best is not None:
            return best[1], best[2], 'src=%s d=%.3f age=%.2fs' % (src, best[0], best[3])
        if now >= deadline or i == n - 1:
            break
        _sleep(H_LADDER_POLL_S)
    return None, None, 'no fresh target row within %.2f m (src=%s)' % (
        H_LADDER_REF_MATCH_M, src)


def _hl_begin(orientation, tx, ty, tz, height_m, _now=None, _sleep=None,
              _rows=None, _environ=None, _exists=None):
    """The ladder state for THIS right pick, or None = the B8i flow.  None
    (upright, or the kill switch) is returned before any log line or row read.
    Never raises: a setup error logs one line and returns None (B8i)."""
    if orientation != 'horizontal':
        return None
    if not _hl_switch(_environ, _exists)[0]:
        return None
    try:
        _now = time.time if _now is None else _now
        _sleep = time.sleep if _sleep is None else _sleep
        _rows = _hl_rows if _rows is None else _rows
        x0, y0 = float(tx), float(ty)
        st = {'n': H_LADDER_ATTEMPTS, 'tried': 1, 'x0': x0, 'y0': y0,
              'z_cmd': float(tz) - GRASP_Z_OFFSET,
              'h_cmd': float(height_m) if height_m is not None else None,
              'aim': (x0, y0), 'last_xy': (x0, y0), 'kc_used': False,
              'ref_label': None, 'ref_iid': None, 'dz': 0.0, 'wz1': None}
        lab, iid, why = _hl_find_ref(st, _now, _sleep, _rows)
        st['ref_label'], st['ref_iid'] = lab, iid
        _r = ('ref=%s%s (%s)' % (lab, '#%s' % iid if iid is not None else '', why)
              if lab is not None else
              'ref=none (%s): the rungs apply to the original aim' % why)
        _hl_log('on: horizontal right pick, %d attempts, rungs (dx,dy,dz) '
                'a2=(%+.3f,%+.3f,%+.3f) a3=(%+.3f,%+.3f,%+.3f) m; %s'
                % ((st['n'],) + tuple(_hl_rung(2)) + tuple(_hl_rung(3)) + (_r,)))
        return st
    except Exception as _e:                                # noqa: BLE001
        _hl_log('off: setup failed (%r) -> B8i flow' % (_e,))
        return None


def _hl_next_attempt(st, attempt, keep_closed):
    """Top of attempt k = attempt + 1 >= 2.  False = stop the ladder (the
    keep-closed re-press was the last attempt, or an error).  A re-press aims
    at the capturing attempt's aim + H_LADDER_REPRESS_DY (B8i: ty + 0.02, at
    the command's tz: dz 0); a fresh approach gets the provisional aim
    original + (dx, dy) and the rung's dz (_hl_aim decides after h-wp1)."""
    k = attempt + 1
    try:
        if st['kc_used']:
            _hl_log('stop before attempt %d/%d: the keep-closed re-press was the '
                    'last attempt (B8i: one re-press, then the final keep-closed)'
                    % (k, st['n']))
            return False
        if keep_closed:
            st['kc_used'] = True
            ax, ay = st['aim']
            st['aim'] = (ax, ay + H_LADDER_REPRESS_DY)
            st['dz'] = 0.0
            st['tried'] = k
            sys.stderr.write('[pick_daemon] Retry attempt %d/%d: keep-closed '
                             're-press +2cm forward (hladder, B8i re-press)\n'
                             % (k, st['n']))
            sys.stderr.flush()
            _hl_log_attempt(st, k, 'repress', 0.0, H_LADDER_REPRESS_DY, 0.0,
                            'keep-closed: attempt-%d capture evidence' % attempt)
            return True
        dx, dy, dz = _hl_rung(k)
        st['aim'] = (st['x0'] + dx, st['y0'] + dy)
        st['dz'] = dz
        st['tried'] = k
        sys.stderr.write('[pick_daemon] Retry attempt %d/%d: hladder rung '
                         'dx=%+.3f dy=%+.3f dz=%+.3f (base decided after h-wp1)\n'
                         % (k, st['n'], dx, dy, dz))
        sys.stderr.flush()
        return True
    except Exception as _e:                                # noqa: BLE001
        _hl_log('stop before attempt %d: ladder error (%r)' % (k, _e))
        return False


def _hl_aim(st, attempt, arm_ik, tz, x_off, _now=None, _sleep=None,
            _rows=None, _cancel=None, _precheck=None, _limit=None):
    """After h-wp1 of a fresh horizontal approach: the aim of attempt
    k = attempt + 1 into st['aim'] (and its dz into st['dz']).  k = 1: the
    command's aim.  k >= 2: the re-detected base + (dx, dy), else the original
    + (dx, dy), each reach-prechecked at the lowered z tz + dz (the pick-start
    gate); False = stop (no reachable aim, or an error)."""
    k = attempt + 1
    try:
        if attempt == 0:
            st['aim'] = (st['x0'], st['y0'])
            st['dz'] = 0.0
            _hl_log_attempt(st, 1, 'orig', 0.0, 0.0, 0.0, 'first')
            return True
        _now = time.time if _now is None else _now
        _precheck = _reach_precheck if _precheck is None else _precheck
        limit = REACH_REFUSE_POS if _limit is None else _limit
        dx, dy, dz = _hl_rung(k)
        st['dz'] = dz
        base, info = _hl_redetect(st, _now(), _now, _sleep, _rows, _cancel)
        cands = []
        if base is not None:
            st['last_xy'] = base          # last clean pose, used by the next jump gate
            cands.append(('fresh', base, info))
        cands.append(('orig', (st['x0'], st['y0']),
                      info if base is None else 'fallback'))
        notes = []
        for kind, (bx, by), why in cands:
            cx, cy = bx + dx, by + dy
            try:
                res = _precheck(arm_ik, 'horizontal', cx, cy, tz + dz, x_off=x_off)[0]
            except Exception:
                res = None
            if res is not None and res > limit:
                notes.append('%s(%.3f,%.3f,z%+.3f) reach %.3f>%.3f'
                             % (kind, cx, cy, tz + dz, res, limit))
                continue
            st['aim'] = (cx, cy)
            _hl_log_attempt(st, k, kind, dx, dy, dz,
                            why + (' refused=' + ';'.join(notes) if notes else ''))
            return True
        st['tried'] = k - 1
        _hl_log('stop before attempt %d/%d: no reachable aim (%s)'
                % (k, st['n'], '; '.join(notes)))
        return False
    except Exception as _e:                                # noqa: BLE001
        if attempt:
            st['tried'] = k - 1
        _hl_log('stop before attempt %d/%d: ladder error (%r)' % (k, st.get('n'), _e))
        return False


def _hl_descent_target(st, attempt, h_tz, R):
    """The h-wp3 descent target of attempt k = attempt + 1: the B8i target ``R``
    (make_wrist_target_middle_down(tx, ty_use, h_tz, ...), whose translation z is
    h_tz itself) moved dz in z (H_LADDER_DXYZ; 0 on attempt 1 and on a
    re-press): same x, y, orientation and park offset, attempt 1's height + dz.
    Returns ``R`` itself when dz is 0.  Only the TARGET moves: the B8i
    bookkeeping (gap, stall, retract, the pad band) stays on h_tz, the floor
    latched at the pick start is not touched and the z-floor clamp cuts the
    lowered move like any other.  Never raises: an error keeps ``R`` (the
    unlowered B8i target) and logs one line."""
    try:
        dz = float(st.get('dz', 0.0)) if attempt > 0 else 0.0
        st['tz_target'] = float(h_tz) + dz
        if dz == 0.0:
            return R
        tgt = np.array(R, dtype=np.float64, copy=True)
        if tgt.shape != (4, 4):
            raise ValueError('target shape %r' % (tgt.shape,))
        tgt[2, 3] = tgt[2, 3] + dz
        return tgt
    except Exception as _e:                                # noqa: BLE001
        st['tz_target'] = h_tz
        _hl_log('descent %d: lowering skipped (%r) -> the unlowered target'
                % (attempt + 1, _e))
        return R


def _hl_descent_note(st, attempt, h_tz, sol_q, arm, arm_ik):
    """After the h-wp3 descent of attempt k = attempt + 1: ONE line with the
    commanded dz, the descent target z, the MEASURED wrist z (FK of the measured
    q, the frame of the B8i 'gap' lines), how much LOWER it ended than attempt 1
    did, and whether the z-floor clamp bound: 'bound' = the right arm's
    commanded slice of the last descent pass (_ZF_LAST_SENT, post-clamp)
    differs from its IK target (pre-clamp); 'free' = equal; 'none' = no floor
    latched or PICK_ZFLOOR=0; 'n/a' = unknown.  Log only: never raises, never
    moves, reads no switch and writes no floor."""
    k = attempt + 1
    try:
        dz = float(st.get('dz', 0.0)) if attempt > 0 else 0.0
        tgt = st.get('tz_target')
        tgt = float(h_tz) + dz if tgt is None else float(tgt)
        try:
            _w = get_wrist_pose(arm_ik, arm.get_current_dual_arm_q().copy(),
                                side='right')
            wz = float(_w[2]) if _w is not None else None
        except Exception:                                  # noqa: BLE001
            wz = None
        if k == 1:
            st['wz1'] = wz
        wz1 = st.get('wz1')
        lower = (wz1 - wz) if (k > 1 and wz is not None and wz1 is not None) else None
        floor = _TABLE_FLOOR_PELVIS
        sent = _ZF_LAST_SENT[0]
        if not _ZFLOOR_ENABLED or floor is None:
            clamp = 'none'
        elif sol_q is None or sent is None:
            clamp = 'n/a'
        else:
            dev = max(abs(float(a) - float(b))
                      for a, b in zip(list(sent)[7:14], list(sol_q)[7:14]))
            clamp = 'bound' if dev > 1e-6 else 'free'
        _hl_log('descent %d/%d: dz=%+.3f target_z=%+.4f wrist_z=%s lower=%s clamp=%s '
                'floor=%s' % (k, st['n'], dz, tgt,
                              ('%+.4f' % wz) if wz is not None else 'none',
                              ('%+.4f' % lower) if lower is not None else 'n/a',
                              clamp, ('%+.4f' % floor) if floor is not None else 'none'))
    except Exception as _e:                                # noqa: BLE001
        _hl_log('descent %d: note skipped (%r)' % (k, _e))


def _hl_end(st, grasp_ok):
    try:
        _hl_log('end: grasp_ok=%s after %d/%d attempts' % (bool(grasp_ok),
                                                            st['tried'], st['n']))
    except Exception:                                      # noqa: BLE001
        pass


# ── park the upright wp4 insert by the measured radius ───────────────────────
# module-level so the pick handler above resolves these names at call time.
# GRIP_ZONE_NEAR_M: URDF grip-zone near edge, forward of R_ee. wp4_park_offset
# uses it to place the park distance behind the centroid when x_off is given.
GRIP_ZONE_NEAR_M = 0.017


def wp4_park_offset(x_off):
    """Calculate the park offset for upright right wp4 target based on x_off.

    The upright wp4 target's fwd_offset parameter determines how far behind the
    centroid the wrist parks. When x_off is provided by the gateway (from measured
    object radius), we calculate r_eff and park distance. When x_off is absent/None,
    we use the default HAND_GRIP_CENTER_OFFSET (0.045 m).

    Args:
        x_off: float or None. The x_off token from the gateway's pick command,
               or None if not provided.

    Returns:
        float: The park offset in metres (positive value to use as fwd_offset
               argument with negation: fwd_offset = -wp4_park_offset(x_off)).
    """
    # absent, None or 0: fall back to the default offset
    if x_off is None or x_off == 0:
        return HAND_GRIP_CENTER_OFFSET

    # non-finite or non-numeric input: fall back to the default offset
    try:
        x_off_val = float(x_off)
        if not math.isfinite(x_off_val):
            return HAND_GRIP_CENTER_OFFSET
    except (ValueError, TypeError):
        return HAND_GRIP_CENTER_OFFSET

    # effective object radius: the negated x_off minus the radius inverse
    r_eff = -x_off_val - H_X_OFF_RADIUS_INV
    # park distance: effective radius plus the URDF grip-zone near edge
    park = r_eff + GRIP_ZONE_NEAR_M
    # clamp to [0.030, 0.115]; 0.115 must stay below the fixed 0.13 m approach
    # offset (wp2/wp3, APPROACH_FORWARD_OFFSET) or wp4 would sit behind wp3.
    # the caller logs a clamp as park_clamped=<requested>-><used>.
    park = max(0.030, min(0.115, park))
    return park


def wp4_park_clamp_note(x_off):
    """B8h: ' park_clamped=<requested>-><used>' when wp4_park_offset clamped.

    '' when x_off is absent/non-finite/zero or the park was not clamped, so a
    log line without a clamp is byte-identical to the B8d one.  Metres, 3 dp.
    """
    try:
        x_off_val = float(x_off)
        if x_off_val == 0 or not math.isfinite(x_off_val):
            return ''
    except (ValueError, TypeError):
        return ''
    requested = -x_off_val - H_X_OFF_RADIUS_INV + GRIP_ZONE_NEAR_M
    used = wp4_park_offset(x_off)
    if abs(requested - used) <= 5e-7:
        return ''
    return ' park_clamped=%.3f->%.3f' % (requested, used)


def wp4_score_offset(x_off):
    """B8d: the fwd_offset the wp4 reach SCORE target takes.

    The insert is scored against the UNPARKED centroid on purpose (a parked
    residual would loosen REACH_REFUSE_POS by the 4.5 cm park).  A radius
    park deeper than 4.5 cm would add its extra depth to every residual and
    refuse picks that pass today, so the score target moves back by exactly
    the EXTRA park: 0.0 when x_off is absent (today's target, byte-identical).
    """
    return -(wp4_park_offset(x_off) - HAND_GRIP_CENTER_OFFSET)


def _reach_precheck(arm_ik, orientation, tx, ty, tz, x_off=None):
    """S1(b): would the RIGHT pick's waypoint chain reach — decided BEFORE
    any arm motion.

    Simulates the commanded chain without the arm.  q0 = READY with the left
    arm at its predicted T5-2 stop (_zf_idle_pre: the pin when set, else where
    the tuck move leaves the arm under the floor, CARRY_TUCK_LEFT when nothing
    clamps).  T5 step 1 lands near READY rather than exactly on it, so q0 is an
    approximation of the measured post-step-1 pose; the idle target is the FK
    left wrist at q0 (solve_and_move's idiom).
      upright:    wp2 (approach @height, fwd APPROACH_FORWARD_OFFSET), wp3
                  (descend to tz + GRASP_Z_UP_BIAS), wp4 (parked insert,
                  scored against the UNPARKED centroid like the in-flow gate);
                  wp3 / wp4 / the wp4 score carry the same wrist deviation
                  add-on the flow commands (_deviation_ramped(1) / (2) while
                  DEVIATION_ENABLED), so the solver trades position against
                  the SAME orientation targets the real chain does;
      horizontal: h-wp2 (middle-down above, parked) and h-wp3 (parked at tz,
                  scored against the unparked target).
    The final residual is the MIN over two warm starts (the chain's previous
    solution, and q0): biased against a false refusal.

    Returns (max residual over the stages, stage name), or
    (None, 'unavailable') on any failure — the caller never refuses then and
    the in-flow wp4 gate stays the second line.  Leaves the solver's smoothing
    filter FRESH so the real first solve is unaffected.
    """
    try:
        q0 = np.array(READY_POSE, dtype=np.float64).copy()
        q0[0:7] = _zf_idle_pre('left', CARRY_TUCK_LEFT, q0)   # left arm where the tuck step leaves it
        idle_w = get_wrist_pose(arm_ik, q0, side='left')
        right_w = get_wrist_pose(arm_ik, q0, side='right')
        if idle_w is None or right_w is None:
            raise RuntimeError('FK of the start pose unavailable')
        idle_T = pin.SE3(np.eye(3), idle_w).homogeneous
        if orientation == 'horizontal':
            h_z = max(float(right_w[2]), tz + 0.20)
            chain = [('h-wp2', make_wrist_target_middle_down(
                tx, ty, h_z, fwd_offset=-HAND_GRIP_CENTER_OFFSET))]
            grasp_cmd = make_wrist_target_middle_down(
                tx, ty, tz, fwd_offset=-HAND_GRIP_CENTER_OFFSET)
            grasp_score = make_wrist_target_middle_down(tx, ty, tz)
            grasp_name = 'h-wp3'
        else:
            a_z = max(float(right_w[2]), tz + APPROACH_HEIGHT_OFFSET)
            g_z = tz + GRASP_Z_UP_BIAS
            dv3 = _deviation_ramped(1) if DEVIATION_ENABLED else 0.0
            dv4 = _deviation_ramped(2) if DEVIATION_ENABLED else 0.0
            chain = [
                ('wp2', make_wrist_target_upright_v4(
                    tx, ty, a_z, +1, fwd_offset=APPROACH_FORWARD_OFFSET)),
                ('wp3', make_wrist_target_upright_v4(
                    tx, ty, g_z, +1, deviation=dv3,
                    fwd_offset=APPROACH_FORWARD_OFFSET)),
            ]
            grasp_cmd = make_wrist_target_upright_v4(
                tx, ty, g_z, +1, deviation=dv4,
                fwd_offset=-wp4_park_offset(x_off))   # same park offset as the grasp flow
            if x_off is None:
                grasp_score = make_wrist_target_upright_v4(
                    tx, ty, g_z, +1, deviation=dv4)
            else:   # same target as the flow's score
                grasp_score = make_wrist_target_upright_v4(
                    tx, ty, g_z, +1, deviation=dv4,
                    fwd_offset=wp4_score_offset(x_off))
            grasp_name = 'wp4'
        stages = []
        warm = q0
        for name, target_T in chain:
            _solver_fresh(arm_ik, 'precheck: ')
            warm = _precheck_solve(arm_ik, idle_T, target_T, warm)
            stages.append((name, _precheck_residual(arm_ik, warm, target_T)))
        g_res = []
        for g_warm in (warm, q0):
            _solver_fresh(arm_ik, 'precheck: ')
            g_q = _precheck_solve(arm_ik, idle_T, grasp_cmd, g_warm)
            g_res.append(_precheck_residual(arm_ik, g_q, grasp_score))
        stages.append((grasp_name, min(g_res)))
        _solver_fresh(arm_ik, 'precheck: ')
        worst_name, worst = max(stages, key=lambda s: s[1])
        _safe_log('[pick_daemon] reach precheck (%s): %s (%s = min of 2 warm '
                  'starts %s) -> max %.3f m at %s vs refuse > %.3f m'
                  % (orientation,
                     ' '.join('%s=%.3f' % s for s in stages), grasp_name,
                     '/'.join('%.3f' % r for r in g_res),
                     worst, worst_name, REACH_REFUSE_POS))
        return worst, worst_name
    except Exception as _e:
        _safe_log(f'[pick_daemon] REACH PRECHECK unavailable ({_e!r}) — not '
                  'refusing; the in-flow wp4 gate stays the second line')
        try:
            _solver_fresh(arm_ik, 'precheck: ')
        except Exception:
            pass
        return None, 'unavailable'


def _expected_after_move(start_q, target_q):
    """The target move_arm_smooth actually commands from start_q (the same
    no-log z-floor clamp), so a clamped move is not read as a stall."""
    try:
        return _zfloor_target(_ARM_IK, start_q, target_q)[0]
    except Exception:
        return target_q


def _ev_num(v):
    try:
        if v is None:
            return None
        f = float(v)
        return round(f, 4) if math.isfinite(f) else None
    except Exception:
        return None


def _ev_bool(v):
    return None if v is None else bool(v)


def _set_grip_evidence(side, verdict, close_blocked=None, close_margin=None,
                       pin_margin=None, tuck_blocked=None, tuck_margin=None):
    """S2: record what the hand reported on this pick's exit (never raises).
    Evidence only: no reply and no decision depends on it."""
    try:
        _GRIP_EVIDENCE_SEQ[0] += 1
        _LAST_GRIP_EVIDENCE[0] = {
            'seq': _GRIP_EVIDENCE_SEQ[0],
            'side': side,
            'verdict': verdict,
            'close_blocked': _ev_bool(close_blocked),
            'close_margin': _ev_num(close_margin),
            'pin_margin': _ev_num(pin_margin),
            'tuck_blocked': _ev_bool(tuck_blocked),
            'tuck_margin': _ev_num(tuck_margin),
            't': round(time.time(), 3),
        }
    except Exception as _e:
        _safe_log(f'[pick_daemon] grip evidence not recorded: {_e!r}')


# self-collision guard for cross-body reaches: check one commanded move
# prev -> pose (left-arm clearance along the joint path, right palm side of
# the pelvis midline; torso depth is diagnostic only, never a refusal).
# consulted at precheck, before hold_rc and any arm motion, and live on every
# pick move except the straight-up wp1 ascents: wp2, wp3, wp4, h-wp2, h-wp3,
# both contact retracts and the keep-closed re-press.
# a refusal prints ERROR:self_collision and leaves to the failure block.
# guard failure never refuses; OPENBOT_SELF_COLLISION_GUARD=0 disables it.
# the startup line prints at import time, since the engine imports this module.
try:
    import self_collision as _sc_mod
    _SC_IMPORT_EXC = None
except Exception as _sc_import_e:
    _sc_mod = None
    _SC_IMPORT_EXC = _sc_import_e

_SC_ENV = 'OPENBOT_SELF_COLLISION_GUARD'
# minimum left-arm clearance from the last precheck, quoted by a live
# refusal; reset by every precheck.
_SC_PRECHECK_C0 = [None]


def _sc_enabled():
    return os.environ.get(_SC_ENV, '1').strip() != '0'


def _sc_unavailable(exc):
    _safe_log(f'SELF-COLLISION GUARD unavailable ({exc!r}) — not refusing')


def _sc_ready():
    """True when the guard runs: enabled and the module imported (a missing
    module is logged as unavailable at every check, never silently)."""
    if not _sc_enabled():
        return False
    if _sc_mod is None:
        _sc_unavailable(_SC_IMPORT_EXC)
        return False
    return True


def _sc_banner():
    """The ONE startup line (deploy check C3: 'SELF-COLLISION GUARD armed').
    Runs at IMPORT time, so it must never raise: a module whose attributes
    do not match (a stale copy on the box) is reported 'unavailable'."""
    try:
        if not _sc_enabled():
            _safe_log(f'[pick_daemon] SELF-COLLISION GUARD DISABLED by env '
                      f'({_SC_ENV}=0) — cross-body picks are NOT refused')
        elif _sc_mod is None:
            _sc_unavailable(_SC_IMPORT_EXC)
        else:
            _safe_log(f'[pick_daemon] SELF-COLLISION GUARD armed: '
                      f'thr={_sc_mod.LEFT_CLEAR_MIN_M:.3f} m, '
                      f'links={len(_sc_mod.SAMPLES)} (right-arm sampled links vs '
                      f'{len(_sc_mod.LEFT_CAP_NAMES)} left-arm capsules; refuse '
                      f'also when the right palm y > {_sc_mod.HAND_Y_MAX_M:+.3f} '
                      f'm for a pick / > {_sc_mod.HAND_Y_MAX_PLACE_M:+.3f} m for '
                      f'a place)')
    except Exception as _e:
        _sc_unavailable(_e)


class _ScRefusal(object):
    """One self-collision refusal: the ERROR line's fields."""

    def __init__(self, verb, x, y, rep, stage, wp):
        self.verb = verb
        self.x = float(x)
        self.y = float(y)
        self.term = rep['term']
        self.value = float(rep['value'])
        self.right_link = rep['right_link']
        self.other_link = rep['other_link']
        self.move_right_m = float(rep['move_right_m'])
        self.stage = stage   # stdout stage: 'precheck' or the live waypoint
        self.wp = wp         # waypoint whose commanded move tripped the rule
        # side of the torso midline the target sits on (torso +x = right)
        self.side = 'left' if self.x < 0.0 else 'right'


def _sc_line(rec):
    """The ONE stdout reply of a self-collision refusal.  brain/agent/tools.py
    parses it (_SELF_COLLISION_RE / _SELF_COLLISION_FIELDS_RE): x/y are the
    TARGET (torso frame, x right-positive), side= the side of the midline the
    target sits on, 'clearance=' carries the refuting term's value (metres;
    the term is named in the stderr line), link= the pair that tripped it,
    move_right_m how far the target must move right to clear the rule."""
    return (f'ERROR:self_collision verb={rec.verb} side={rec.side} x={rec.x:.3f}m '
            f'y={rec.y:.3f}m clearance={rec.value:.3f}m '
            f'link={rec.right_link}~{rec.other_link} '
            f'move_right_m={rec.move_right_m:.2f} stage={rec.stage}\n')


def _sc_bar(verb):
    """The endpoint midline bar: the pick bar, or the place bar."""
    return (_sc_mod.HAND_Y_MAX_PLACE_M if verb == 'place'
            else _sc_mod.HAND_Y_MAX_M)


def _sc_cmp(rec):
    """Term-aware comparator for the stderr line ('clearance <c> m < <thr> m'
    only fits the left-arm term)."""
    _pair = f'right {rec.right_link} vs {rec.other_link}'
    if rec.term == 'left_arm_clearance':
        return (f'{_pair} clearance {rec.value:.3f} m < '
                f'{_sc_mod.LEFT_CLEAR_MIN_M:.3f} m')
    return (f'{_pair} hand y {rec.value:+.3f} m > '
            f'{_sc_bar(rec.verb):+.3f} m')


def _sc_terms(rep):
    return (f"clear={rep['left_clear_m']:.3f} hand_y={rep['hand_y_m']:+.3f} "
            f"torso_diag={rep['torso_gross_m']:+.3f}")


def _sc_check(q_from, q_to, verb, x, y, stage, wp):
    """self_collision.report() on ONE commanded move q_from -> q_to.
    Returns (_ScRefusal or None, report or None).  Never raises: a raising
    check is 'unavailable', never a refusal."""
    try:
        rep = _sc_mod.report(np.array(q_from, dtype=np.float64),
                             np.array(q_to, dtype=np.float64))
        if not rep['refuse']:
            return None, rep
        return _ScRefusal(verb, x, y, rep, stage, wp), rep
    except Exception as _e:
        _sc_unavailable(_e)
        return None, None


def _sc_live_verdict(rec, rep, wp):
    if rec is None:
        if rep is not None:
            _safe_log(f'[pick_daemon] {wp} self-collision guard: '
                      f'{_sc_terms(rep)} -> pass')
        return
    c0 = _SC_PRECHECK_C0[0]
    _c0 = f'{c0:.3f} m' if c0 is not None else 'n/a'
    _safe_log(f'SELF-COLLISION (live, {wp}): solved {wp} pose {_sc_cmp(rec)} '
              f'(precheck passed with {_c0}) — REFUSED before commanding '
              f'{wp}; safe exit.')


def _sc_live_check(arm, sol_q, stage, verb, x, y):
    """LIVE guard on an already-solved, idle-arm-pinned pose (wp4: sol_q after
    freeze_idle_arm; h-wp3: each descent pass's _s): the move from the
    MEASURED q to sol_q.  Returns a _ScRefusal -- the caller writes the ONE
    reply, sets _reply_sent and leaves the attempt loop to the failure block
    -- or None (pass / off / unavailable)."""
    if not _sc_ready():
        return None
    try:
        q_from = np.array(arm.get_current_dual_arm_q(), dtype=np.float64)
    except Exception as _e:
        _sc_unavailable(_e)
        return None
    rec, rep = _sc_check(q_from, sol_q, verb, x, y, stage, stage)
    _sc_live_verdict(rec, rep, stage)
    return rec


def _sc_live_presolve(arm, arm_ik, target, stage, verb, x, y, raw=False,
                      log_prefix=''):
    """LIVE guard for a waypoint that solve_and_move() solves AND moves in one
    call (wp2, wp3, h-wp2).  Solves the SAME target with solve_and_move's exact
    idiom for side='right' (measured q, FK idle-left target, the verified
    solve -- or the raw solve for raw=True), pins the idle arm the way
    freeze_idle_arm does (q[0:7] = CARRY_TUCK_LEFT, without its settle wait)
    and checks the move from the measured q.  The caller has just called
    _solver_fresh(), so this is the 1st-after-fresh solve; the filter is
    cleared again afterwards, so solve_and_move's own solve is again the
    1st-after-fresh solve from the same measured q (robot_arm_ik seeds
    init_data from current_lr_arm_motor_q) -- the commanded pose is the pose
    this guard checked.  An unsolvable target returns None: that is
    solve_and_move's own refusal, not this guard's."""
    if not _sc_ready():
        return None
    _q = None
    _cur_q = None
    try:
        _cur_q = arm.get_current_dual_arm_q().copy()
        _idle_wrist = get_wrist_pose(arm_ik, _cur_q, side='left')
        _idle_target = (pin.SE3(np.eye(3), _idle_wrist).homogeneous
                        if _idle_wrist is not None else L_NEUTRAL)
        if raw:
            with _SuppressStdout():
                _q, _tau = arm_ik.solve_ik(
                    _idle_target, target,
                    current_lr_arm_motor_q=_cur_q.copy(),
                    current_lr_arm_motor_dq=np.zeros(14))
        else:
            _q, _tau = _solve_ik_verified(arm_ik, _idle_target, target, _cur_q,
                                          'right', log_prefix + '[sc-guard] ')
    except Exception as _e:
        _sc_unavailable(_e)
        _q = None
    finally:
        _solver_fresh(arm_ik, log_prefix + '[sc-guard] ')
    if _q is None:
        return None
    _q = np.array(_q, dtype=np.float64).copy()
    _q[0:7] = _zf_idle_target('left', CARRY_TUCK_LEFT)   # left arm at the pose the preceding move leaves it
    rec, rep = _sc_check(_cur_q, _q, verb, x, y, stage, stage)
    _sc_live_verdict(rec, rep, stage)
    return rec


def _sc_ik_snapshot(arm_ik):
    """The solver state a precheck's solves touch: the smoothing filter's queue
    and last output, and the warm start.  None when not introspectable."""
    try:
        f = arm_ik.smooth_filter
        return (list(f._data_queue), np.array(f._filtered_data).copy(),
                np.array(arm_ik.init_data).copy())
    except Exception:
        return None


