#!/usr/bin/env python3
"""
walk_node.py — ROS2 locomotion FSM for walking toward a named object (G1).

Drives the 4-state FSM via the sibling wrapper `G1Loco` (which never writes
leg joints — the on-board balance FSM stays in control):

    IDLE → SEARCHING → APPROACHING → REACHED → IDLE

    1. IDLE      — waiting for a `/walk_target` label. No DDS lease held.
    2. SEARCHING — 5 s detection window standing still; on expiry, a 30°
                   sweep turn (one_way → cumulative 360°) and a fresh window.
                   The sweep is INTERRUPTIBLE (T10): the first detection
                   (presence via /detections) stops the turn immediately and
                   enters the DETECTED hold — stand still, no turn, no walk —
                   until `/walk_go` (matching label, GO_STALE_S-fresh), a
                   stop, a retarget, or detection loss (re-enters the sweep).
                   DETECTED is a /walk_status vocabulary event inside
                   SEARCHING (not an FSM state), published EXACTLY ONCE per
                   hold entry. `/walk_target` is NEVER a trigger — `/walk_go`
                   is the only trigger channel.
    3. APPROACHING — 10 Hz closed loop: yaw-align, then walk at VX_WALK until
                   the body-frame distance to the target ≤ STOP_DISTANCE_M.
                   Walking starts ONLY after the `/walk_go` trigger (the pick
                   coordinator's 3-consecutive-confirmation gate + table).
    4. REACHED   — stop, publish status once, return to IDLE standing in
                   walk mode (leaves reach room for the pick arm).

Trigger (topic-driven, mirroring floor_pick_node):
    ros2 topic pub /walk_target std_msgs/msg/String "{data: 'chair'}" --once
    ros2 topic pub /walk_target std_msgs/msg/String "{data: 'stop'}" --once
    ros2 topic pub /walk_go std_msgs/msg/String "{data: 'chair'}" --once
      — approach trigger (T10): consumed ONLY by the DETECTED hold, ONLY
        when the label matches the active target and the trigger is
        GO_STALE_S-fresh; stale/wrong-label triggers are logged + dropped.

Frame notes (LOCO_PICKUP_BRIEF.md §5 / CONTRACT §7):
    - `/object_detections_body` is the camera/body frame (x lateral, y forward,
      z up, meters). Do NOT subscribe to `/object_detections` (world frame) —
      double −π/2 CAM_YAW_OFFSET mirror quirk. All world math is body-frame
      + own odom yaw (floor_pick_node pattern; SDK API signatures are
      source-verified, but control constants remain starting-point placeholders
      pending hardware calibration).

Integration note (T7, owner-directed): walk_node is a STANDALONE post-pick
    pipeline — the robot has ALREADY picked the object (object in hand); it is
    launched via its own launcher (`launch_walk.sh`, camera + yolo +
    depth_projector + walk_node only) to walk to ANY named target and hand off
    for the drop. It is NOT a pick-launcher flag: launch_pick_pipeline.sh does
    not (and must not) host walk_node — the pick daemon locks the lower-body
    legs and moves the limbs to a ready pose, both forbidden during walking.

Safety invariants:
    - Legs are NEVER written: only G1Loco verbs (SetVelocity / StopMove /
      Start), so the on-board balance FSM stays in control at all times.
    - NEVER run alongside floor_pick_node (launch_loco_pick.sh) or the pick
      pipeline: each locomotion owner uses the sport RPC endpoint, and the
      pick daemon locks legs + ready-pose limbs — both incompatible with
      walking. launch_walk.sh is safe by construction (no such nodes).
    - Backend selection (T6): `LOCO_BACKEND=mujoco` → walk-node/sim/mujoco_loco.py
      (MuJoCo sim eval, publishes /sim_loco_cmd instead of DDS). `FAKE_LOCO=1`
      → dev stub. `LOCO_BACKEND=g1_inproc` → in-process G1Loco (REFERENCE ONLY
      — rclpy + raw-DDS in one process crashes on Stuart). `FAKE_DAEMON=1` →
      dev-box dry-run against fake_loco_daemon.py (protocol-logging stand-in).
      DEFAULT: `LOCO_BACKEND=daemon` — spawns walk_loco_daemon.py as a
      subprocess (plain login-shell python3, scrubbed env) and speaks the
      stdin protocol (READY / DONE / ERROR). No unitree_sdk2py import is
      reachable on this default path — the SDK lives only inside the daemon.
      FAKE_LOCO / mujoco behavior unchanged.

Bounded hunt (Phase 1 — docs/NAV-DESIGN.md §3/§10, gates G5-G8, G15, G17):
    The hunt is NO LONGER an unbounded `while True:`. Per position: 12
    completed 30° sweep turns = one round (G6, counters in hunt_law.py);
    the approach carries a 15 s / 0.05 m stall watchdog (G5b); two
    no-progress events per epoch (stall / collision freeze / hold-escape)
    exhaust the budget (D1 graft); NOT_FOUND is a REAL terminal that stops
    the robot and publishes proof counters — the honest "searched, not
    found". WALK-HOP is design-of-record only (decision 2026-09-04 #1: held
    until the Phase-4 vision-range measurement). G17a freezes the approach
    on an obstruction between robot and target; G17b refines the stop gate
    with the published-extent near-edge law. The pure arithmetic / law code
    lives in hunt_law.py (unit-tested without rclpy).
"""

import json
import math
import os
import select
import subprocess
import sys
import threading
import time

import hunt_law        # hunt budget and collision math
import walk_lock       # approach lock and turn-back
                       # helpers (pure)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# module directory on the path, with sim/ searched
# first so the mujoco backend resolves
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, 'sim'))

# scene-graph refs can carry a suffix the detector never
# publishes, so every lookup keys the published label
# the instance resolver maps a ref to that label; a missing
# module raises at import time instead of failing silently
try:
    from instance_labels import ref_to_label          # installed flat layout
except ImportError:
    from skills.perception.instance_labels import ref_to_label   # package tree layout

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry

# the in-process backend imports the vendor SDK lazily;
# the daemon backend keeps the SDK out of this process

# some ROS 2 client releases raise a dedicated shutdown
# exception from spin(); the fallback is KeyboardInterrupt
try:
    from rclpy.executors import ExternalShutdownException  # optional exception
except ImportError:
    ExternalShutdownException = KeyboardInterrupt          # fallback

# -- sweep motion -----------------------------------------------------------
SWEEP_STRATEGY   = 'one_way'   # 'one_way' or 'alternate'
SWEEP_SIGN       = 1.0         # +1 turns left; either sign works
TURN_ANGLE_RAD   = math.pi / 6.0
YAW_RATE_RAD_S   = 0.5
TURN_DURATION_S  = TURN_ANGLE_RAD / YAW_RATE_RAD_S   # 30 deg at 0.5 rad/s
TURN_SETTLE_S    = 0.3
# continuous sweep: a short look, then turns in segments
# a velocity command is renewed every CONT_RENEW_S and
# lasts CONT_CMD_DURATION_S, so a dead node stops within 1 s
# presence is polled every DETECT_POLL_S and the first
# sighting stops the turn; 12 segments make one round
# OPENBOT_SWEEP_STEPPED=1 selects the stepped sweep
SWEEP_CONTINUOUS    = os.environ.get('OPENBOT_SWEEP_STEPPED', '').strip() != '1'
CONT_FIRST_LOOK_S   = 1.5
CONT_SEGMENT_S      = 2.0 * TURN_DURATION_S          # two 30 deg turns = 60 deg
CONT_RENEW_S        = 0.3
CONT_CMD_DURATION_S = 1.0

# -- detection windows -------------------------------------------------------
DETECT_WINDOW_S  = 5.0
DETECT_POLL_S    = 0.1
HOLD_MAX_S       = 10.0        # no position for two windows

# -- detection gating --------------------------------------------------------
MIN_CONF          = 0.20       # detection floor shared with the pick gate and the
                               # corridor check; a target's confidence runs 0.20-0.28,
                               # so 0.25 rejects most samples
STALE_DETECTION_S = 2.0        # rows past this age are untrusted
# open-vocabulary detectors put two labels on one object
SAME_OBJECT_M     = 0.20       # another label this close in body x/y is the same
                               # object, not an obstacle
# rows closer than this are the robot's own arm, hand or
# payload: never a target, never a corridor obstacle, and
# safely inside the 0.55 m stop distance
SELF_ZONE_M       = 0.30
TIE_CONF_BAND     = 0.15       # rows within this band of the best conf tie; the tie
                               # goes to the row nearest the last target
LOST_TIMEOUT_S    = 2.0        # no fresh target this long: back to SEARCHING
GO_STALE_S        = 5.0        # a /walk_go past this age is dropped, never re-fired

# -- drive constants ---------------------------------------------------------
# starting points, not validated on hardware; recalibrate on
# the robot before any untethered run
VX_WALK        = 0.30    # aligned forward speed (m/s)
VX_MIN         = 0.20    # gait floor: 0.15 does not step and 0.20 does;
                         # keep vx at zero below it
YAW_VX_DEADBAND = 0.35   # about 20 deg: fade vx to zero
K_THETA        = 1.5     # yaw gain
OMEGA_MAX      = 0.40    # yaw rate cap (rad/s)
OMEGA_SIGN     = -1.0    # +omega turns left while body +x is right, so a
                         # positive bearing error needs this sign
CTRL_DT_S      = 0.10    # loop period (s)

# -- status and stop gate -----------------------------------------------------
APPROACH_HEARTBEAT_S = 5.0
REACHED_HOLD_S   = 5.0    # republish REACHED at 1 Hz so late joiners see it
                          # before the pick side starts
STOP_DISTANCE_M = 0.55    # tabletop reach in m to the target centroid, from the
                          # body, not the toes; calibrate on the robot. the label
                          # leaves the frame near 0.43 m, so 0.55 leaves about
                          # 0.12 m margin and stays within arm reach; 0.60 is the
                          # upper limit
ARRIVAL_LATCH_S = 5.0     # grace after the last in-zone reading: commit REACHED
                          # even as the label goes stale, since the projector
                          # drops it up close

# -- stop law, blind cap and label continuity ---------------------------------
# three startup switches, read once; each is on by default and
# '0' disables it. 'legacy' for the stop law selects the
# 0.55 m centroid gate and logs no settle evidence
# OPENBOT_WALK_BLIND_CAP=0 trusts a row for STALE_DETECTION_S
# OPENBOT_WALK_CONTINUITY=0 treats a label flip as a loss
STOP_LAW_LEGACY = (os.environ.get('OPENBOT_STOP_LAW', '').strip().lower()
                   in ('legacy', '0'))
WALK_BLIND_CAP = os.environ.get('OPENBOT_WALK_BLIND_CAP', '').strip() != '0'
WALK_CONTINUITY = os.environ.get('OPENBOT_WALK_CONTINUITY', '').strip() != '0'
# steering on the odometry-tracked target between detections
# rests on leg forward-kinematics dead reckoning, not a
# measurement; walking on that estimate can hit obstacles
# OPENBOT_WALK_ODOM_LATCH=1 enables it, unset leaves it off
WALK_ODOM_LATCH = os.environ.get('OPENBOT_WALK_ODOM_LATCH', '').strip() == '1'
# four startup switches, read once; each is off with '0':
# OPENBOT_WALK_LOCK=0 follows the best row without the
# approach lock
# OPENBOT_WALK_REACQUIRE=0 sends a lost approach straight
# back to the one-way sweep
# OPENBOT_WALK_TURN_CAP=<rad/s> caps the turn rate while
# approaching: default 0.50, clamped to 0.10-0.60; 0.30 is
# too slow to turn in place and '0' disables the cap
# OPENBOT_WALK_POSE=0 refuses /walk_pose
WALK_LOCK = os.environ.get('OPENBOT_WALK_LOCK', '').strip() != '0'
WALK_REACQUIRE = os.environ.get('OPENBOT_WALK_REACQUIRE', '').strip() != '0'
WALK_POSE = os.environ.get('OPENBOT_WALK_POSE', '').strip() != '0'


def _turn_cap(raw):
    """OPENBOT_WALK_TURN_CAP -> the cap in rad/s (None = no cap)."""
    if raw is None or not str(raw).strip():
        return 0.50
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.50
    if not math.isfinite(v) or v <= 0.0:
        return None
    return max(0.10, min(0.60, v))


WALK_TURN_CAP = _turn_cap(os.environ.get('OPENBOT_WALK_TURN_CAP'))
# OPENBOT_POSE_BACKOFF_M: distance to step back before turning home
POSE_BACKOFF_ENV = os.environ.get('OPENBOT_POSE_BACKOFF_M')
BLIND_MAX_S = 0.5         # no forward motion on a target row past this age
                          # (receive time); a healthy approach sees much shorter gaps
SETTLE_LOG_S = 1.0        # log where the robot actually settled, this long
                          # after REACHED

# -- corridor escape ---------------------------------------------------------
# a corridor obstruction normally holds the approach; when
# the hold persists, walk an offset bearing around it instead
# a detour is a forward crawl, not a strafe: lateral velocity
# is unused here and its sign is unverified, while yaw is known
DETOUR_AFTER_HOLD_S = 6.0  # hold this long before detouring: a passer-by clears
                           # on its own, so the passive hold stays the default
                           # shorter than the 45 s corridor hold, so two detours
                           # plus holds fit the 120 s approach deadline
DETOUR_RAD        = 0.45   # aim offset from the target bearing (rad, about 26 deg)
DETOUR_DURATION_S = 3.0    # length of one detour walk (s)
DETOUR_VX         = VX_MIN * 1.1   # fixed crawl at 0.22 m/s, just above the gait floor;
                                   # the approach speed fades to zero near the yaw
                                   # deadband, which a detour deliberately crosses
DETOUR_MAX        = 2      # detours per approach, alternating sides; then fall
                           # back to the hold

# -- arrival alignment -------------------------------------------------------
# the stop law tests planar distance alone, so the robot
# can stop off-axis; turn in place until the bearing is
# centred before committing REACHED
ALIGN_TOL_RAD   = 0.09     # about 5 deg: centred enough for the arm handoff
ALIGN_TIMEOUT_S = 1.5      # bounded centring turn; at 0.40 rad/s it covers about 34 deg
                           # stays below LOST_TIMEOUT_S (2.0 s): the turn moves the
                           # object and can stale the label, and a longer wait would let
                           # the stale rollback undo the arrival
                           # alignment gates the stop-law branch only, so a close-range
                           # dropout reaches REACHED through the arrival latch

# -- stall watchdog -----------------------------------------------------------
STALL_WINDOW_S     = 15.0    # rolling window (s): distance must drop
STALL_MIN_PROGRESS = 0.05    # by this many metres, else stalled: stop and
                             # re-sweep; two stalls per budget end in NOT_FOUND
# -- target-stability gate ---------------------------------------------------
# a target position that jumps between consecutive fresh
# detections is not self-consistent; refuse it instead of
# walking to it, and never let it reach the stop law or the
# corridor gate
# dt comes from capture stamps, not receipt times: under load
# two samples can arrive close together after a longer capture
# gap, which would make honest motion look impossible
# the bound adds the noise floor to the physical ceiling; a
# range factor would only loosen the guard further out
YAW_RATE_MAX       = abs(K_THETA * OMEGA_MAX)   # actual yaw ceiling in rad/s: the error is
                                                # clamped, then the gain scales it to 0.60
TARGET_JUMP_MIN_M  = 0.30    # noise floor in m, between honest motion (<= 0.20) and
                             # garbage (>= 0.45); a false refusal breaks a healthy
                             # walk, so the margin is not spent on the garbage side
TARGET_JUMP_DT_S   = 5.0    # compare only detections this close in time; a longer
                            # gap can hide real motion, so it is not judged
TARGET_UNSTABLE_N  = 3       # reject after three impossible samples in a row
# -- corridor hold -----------------------------------------------------------
# a corridor freeze keeps the target and re-enters the hold
# instead of spending the give-up budget: a passer-by is
# transient, and dropping the object over it is wrong
# the hold budget and clock live in the hunt law, which is
# unit-tested separately
# the edge stop margin and max reachable height are calibrated
# against a box of known size, like STOP_DISTANCE_M

# -- ROS topics (default QoS depth 10) ----------------------------------------
DETECTIONS_TOPIC      = '/detections'
BODY_DETECTIONS_TOPIC = '/object_detections_body'
ODOM_TOPIC            = '/unitree/slam_mapping/odom'
WALK_TARGET_TOPIC     = '/walk_target'
WALK_STATUS_TOPIC     = '/walk_status'
WALK_GO_TOPIC         = '/walk_go'
# '<label>#<iid>@<hue>' names one object;
# a bare label clears the instance
WALK_INSTANCE_TOPIC   = '/walk_instance'
# a JSON object with x and y, optional yaw or face_x/face_y,
# and an optional label, in the odometry frame; 'stop' ends
# the walk
WALK_POSE_TOPIC       = '/walk_pose'
#: an instance past this age is dropped at run end,
#: so a later walk does not inherit it
INSTANCE_KEEP_S       = 10.0

# -- the locomotion plumbing --------------------------------------------------
NETWORK_IFACE  = 'eth0'
FAKE_LOCO      = os.environ.get('FAKE_LOCO', '') == '1'
FAKE_DAEMON    = os.environ.get('FAKE_DAEMON', '') == '1'
LOCO_BACKEND   = os.environ.get('LOCO_BACKEND', 'daemon').strip().lower()
                # 'mujoco' | 'g1_inproc' | 'daemon'
                # (daemon is the default)


def _wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _cap_turn(omega):
    """B8l: clamp an approach / alignment yaw rate to WALK_TURN_CAP."""
    if WALK_TURN_CAP is None:
        return omega
    return max(-WALK_TURN_CAP, min(WALK_TURN_CAP, omega))


class _FakeLoco:
    """FAKE_LOCO=1 dev stub: all verbs print `[FAKE_LOCO] ...` lines only.

    get_state() returns a static dict so callers exercise the same code paths
    as on the robot. Never touches the SDK or the network.
    """

