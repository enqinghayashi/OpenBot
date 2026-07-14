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
