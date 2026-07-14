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
