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
