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

    def connect(self):
        print('[FAKE_LOCO] connect()')
        return self

    def start(self):
        print('[FAKE_LOCO] start() (skipped on hardware — robot SDK sends '
              'SetFsmId(200); R2+A is already FSM 801)')
        return None

    def set_velocity(self, vx, vy, omega, duration=1.0):
        print('[FAKE_LOCO] set_velocity '
              f'vx={vx:+.2f} vy={vy:+.2f} omega={omega:+.2f} '
              f'duration={duration:.2f}s')
        return None

    def stop(self):
        print('[FAKE_LOCO] stop()')
        return None

    def disconnect(self):
        print('[FAKE_LOCO] disconnect()')
        return None

    def get_state(self):
        return {
            'velocity': [0.0, 0.0, 0.0],
            'yaw_speed': 0.0,
            'position': [0.0, 0.0, 0.0],
            'mode': 500,
            'gait_type': 0,
        }


class _DaemonLoco:
    """LOCO_BACKEND=daemon (T6): locomotion lives in a subprocess daemon.

    Spawns walk_loco_daemon.py (or fake_loco_daemon.py under FAKE_DAEMON=1)
    with a scrubbed env (PYTHONPATH cleared — the daemon inserts the
    known-good SDK copy itself; backend switches never leak into the
    daemon), waits for the READY line, then speaks the stdin protocol:
    `walk <vx> <vy> <omega> [duration]` -> `DONE:<sport code>` (0 = OK);
    `stop` -> `DONE`; `quit` -> `DONE` + os._exit(0). One response line per
    command; `ERROR:<msg>` responses raise RuntimeError.

    Interpreter: plain login-shell python3 (sys.executable, WALK_DAEMON_PYTHON
    override) — the daemon must NOT run under the tv conda env (documented
    cyclonedds lib clash); the sport RPC works under the system 3.8.
    """

    _READY_TIMEOUT_S = 15.0

    def __init__(self, iface='eth0', daemon_script=None):
        self._iface = iface
        self._daemon_script = daemon_script or os.path.join(
            _SCRIPT_DIR, 'walk_loco_daemon.py')
        self._proc = None
        self._ready = False
        self._lock = threading.Lock()       # one command at a time

    # -- set up and tear down ---

    def connect(self):
        """Spawn the daemon and wait for its READY line."""
        env = dict(os.environ)
        env['PYTHONPATH'] = ''              # cleared so the daemon uses its own SDK path
        for key in ('LOCO_BACKEND', 'FAKE_LOCO', 'FAKE_DAEMON'):
            env.pop(key, None)              # backend switches never reach the daemon
        python = os.environ.get('WALK_DAEMON_PYTHON', sys.executable)
        self._proc = subprocess.Popen(
            [python, self._daemon_script, self._iface],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env=env, universal_newlines=True, bufsize=1)
        deadline = time.time() + self._READY_TIMEOUT_S
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f'daemon exited before READY (rc={self._proc.returncode})')
            ready, _, _ = select.select([self._proc.stdout], [], [], 0.2)
            if ready:
                line = self._proc.stdout.readline().strip()
                if line == 'READY':
                    self._ready = True
                    return self
                raise RuntimeError(
                    f'unexpected daemon first line: {line!r}')
        raise RuntimeError(
            f'daemon not READY within {self._READY_TIMEOUT_S}s')

    def disconnect(self):
        """quit + terminate backstop: ask the daemon to exit, then reap it."""
        proc, self._proc = self._proc, None
        self._ready = False
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write('quit\n')
            proc.stdin.flush()
            proc.stdout.readline()          # daemon replies DONE, then exits
        except (BrokenPipeError, OSError):
            pass
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)

    # -- the stdin protocol ---

    def _send_line(self, line):
        """One command in, one response line out. ERROR:<msg> raises."""
        with self._lock:
            if self._proc is None or not self._ready:
                raise RuntimeError('daemon not connected')
            try:
                self._proc.stdin.write(line + '\n')
                self._proc.stdin.flush()
                resp = self._proc.stdout.readline()
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f'daemon pipe broken: {e}') from e
        resp = (resp or '').strip()
        if resp.startswith('ERROR:'):
            raise RuntimeError(resp[len('ERROR:'):])
        return resp

    # -- the movement verbs ---

    def set_velocity(self, vx, vy, omega, duration=1.0):
        """`walk <vx> <vy> <omega> <duration>` -> DONE:<sport code> (0 = OK)."""
        resp = self._send_line(f'walk {vx} {vy} {omega} {duration}')
        if resp == 'DONE':
            return None
        if resp.startswith('DONE:'):
            try:
                return int(resp[len('DONE:'):])
            except ValueError:
                return resp[len('DONE:'):]
        raise RuntimeError(f'unexpected daemon response: {resp!r}')

    def stop(self):
        self._send_line('stop')
        return None

    def get_fsm_mode(self):
        """`mode` -> DONE:<fsm mode> (GET 7002). None on malformed reply.

        Hardware-day finding (2026-08-16): fsm_mode=3 = gait-BLOCKED post-boot
        state; fsm_mode=0 = walking-capable. Pre-flight must warn on != 0."""
        resp = self._send_line('mode')
        if resp.startswith('DONE:'):
            try:
                return int(resp[len('DONE:'):])
            except ValueError:
                return None
        return None

    def start(self):
        # deliberate no-op: at R2+A the robot is already in sport
        # FSM 801, and the vendor start() sends SetFsmId(200); kept
        # so every backend shares one API
        print('[DAEMON_LOCO] start() no-op — skipped on hardware '
              '(R2+A = FSM 801; SDK Start() = SetFsmId(200)).')
        return None

    def get_state(self):
        return None     # unused, kept for the backend API


class WalkNode(Node):
    """ROS2 node driving the walk FSM (see module docstring).

    rclpy callbacks only mutate guarded state; one daemon FSM thread drives
    everything with time.time() deadlines + time.sleep() loops (no ROS timers).
    """

    def __init__(self):
        super().__init__('walk_node')

        self._loco = None                # backend, connected lazily
        self._busy = False               # FSM thread running?
        # why the FSM left: 'stop', 'not_found', 'crash' or None,
        # recorded where the exit is decided and published on the
        # closing IDLE frame; None means the loop fell through ->
        # 'finished'
        self._fsm_exit_reason = None

        # one pending command, ('stop', None) or ('target', label),
        # consumed only by _poll_command
        self._cmd_lock = threading.Lock()
        self._pending = None

        # one /walk_go slot, (label, ts), under _cmd_lock; only the
        # detected hold consumes it, and it is never re-fired
        self._go_pending = None
        self._target_label = None
        # the one object named by '<label>#<iid>@<hue>', else None;
        # the FSM only sees the bare label
        # {'label','iid0','iid','hue','last_xy','match','t'}, under
        # _inst_lock; /walk_instance sets it, and the pick side
        # relays the bare label
        self._inst = None
        self._inst_lock = threading.Lock()
        self._sweep_dir = SWEEP_SIGN

        # hunt budget: proof counters for rounds, turns, windows and
        # stalls, reset per hunt; one position only
        self._hunt = hunt_law.HuntBudget()
        # corridor clock: one unbroken run of freezes, cleared by any
        # approach tick that gets through
        self._corridor = hunt_law.CorridorHold()
        # stop-law evidence from the last REACHED decision (near-edge
        # numbers, the height flag); rides the REACHED detail
        self._reach_evidence = {}
        # continuity, armed only inside the approach: the instance id
        # of each chosen row, and the last '[cont]' line, one per
        # episode
        self._cont_armed = False
        self._row_iids = {}
        self._cont_note = None
        # approach lock (under _targets_lock, alive only during a
        # label run); the sweep side a lost target asks for; a command
        # handed over by a preempted run; latest body rows for pose
        # obstacles
        self._alock = walk_lock.ApproachLock()
        self._alock_live = False
        self._sweep_pref = None
        self._deferred = None
        self._body_rows = (0.0, [])

        # sightings: label -> last-seen ts
        self._visible_lock = threading.Lock()
        self._visible = {}

        # best body row per label
        self._targets_lock = threading.Lock()
        self._targets = {}

        # latched world target
        self._twx = self._twy = None
        self._twz = 0.0
        self._tw_label = 'object'

        # odom pose: x, y, yaw
        self._odom_lock = threading.Lock()
        self._robot_x = 0.0
        self._robot_y = 0.0
        self._robot_yaw = 0.0
        self._odom_t = 0.0        # receive time of the last odom message (0 = never)

        self.create_subscription(
            String, WALK_TARGET_TOPIC, self._on_walk_target, 10)
        self.create_subscription(
            String, WALK_GO_TOPIC, self._on_walk_go, 10)
        self.create_subscription(
            String, WALK_INSTANCE_TOPIC, self._on_walk_instance, 10)
        self.create_subscription(
            String, WALK_POSE_TOPIC, self._on_walk_pose, 10)     # walk to a pose
        self.create_subscription(
            String, DETECTIONS_TOPIC, self._on_detections, 10)
        self.create_subscription(
            String, BODY_DETECTIONS_TOPIC, self._on_body_detections, 10)
        self.create_subscription(
            Odometry, ODOM_TOPIC, self._on_odom, 10)

        self._status_pub = self.create_publisher(String, WALK_STATUS_TOPIC, 10)

        self._publish_status('IDLE', detail={'strategy': SWEEP_STRATEGY})
        self.get_logger().info(
            f'WalkNode ready. Publish a label to {WALK_TARGET_TOPIC} to walk, '
            f'"stop"/"abort" to halt. Stop gate = {STOP_DISTANCE_M:.2f} m '
            f'(body frame; CALIBRATE on hardware).')
        self.get_logger().info(
            '[b8j] stop law=%s (pick edge %.2f m, large edge %.2f m + lead '
            '%.2f m); blind cap=%s; continuity=%s' % (
                'legacy' if STOP_LAW_LEGACY else 'near_edge',
                STOP_DISTANCE_M - hunt_law.PICK_HALF_DEFAULT_M,
                hunt_law.LARGE_EDGE_M, hunt_law.LARGE_LEAD_M,
                ('%.2fs' % BLIND_MAX_S) if WALK_BLIND_CAP else 'off',
                'on' if WALK_CONTINUITY else 'off'))
        self.get_logger().info(
            '[b8l] approach lock=%s; reacquire=%s; turn cap=%s; walk to pose=%s'
            % ('on' if WALK_LOCK else 'off',
               'on' if WALK_REACQUIRE else 'off',
               ('%.2f rad/s' % WALK_TURN_CAP) if WALK_TURN_CAP else 'off',
               'on' if WALK_POSE else 'off'))

    # -- ROS callbacks (guarded state only) --------------------------------------

    def _on_walk_target(self, msg):
        """Queue a command; spawn the FSM thread only when not already busy."""
        cmd = msg.data.strip().lower()
        if cmd in ('stop', 'abort'):
            with self._cmd_lock:
                if self._busy:
                    self._pending = ('stop', None)
                    self.get_logger().info('Stop requested — halting.')
                elif getattr(self, '_deferred', None) is not None:
                    self._deferred = None       # cancel a handed-over walk
                    self.get_logger().info('Stop requested — the deferred walk is dropped.')
                else:
                    self.get_logger().info('Walk not running — ignoring "stop".')
            return
        # resolve the scene-graph ref to its published label, then
        # split '<label>#<iid>@<hue>' before anything reads the
        # label; a bare label is unchanged and clears any instance,
        # and the resolver only rewrites a trailing '_<digits>'
        cmd = self._set_instance(cmd)
        cmd = ref_to_label(cmd)
        if not cmd:
            return
        with self._cmd_lock:
            if (getattr(self, '_pose_running', False) and self._pending is not None
                    and self._pending[0] == 'stop'):
                # a stop queued during the return walk is never overwritten
                # by a /walk_target in the same tick
                self.get_logger().warn(f'/walk_target "{cmd}" dropped: a stop is pending.')
                return
            self._pending = ('target', cmd)
            if not self._busy:
                self._busy = True
                threading.Thread(
                    target=self._run_fsm, args=(cmd,), daemon=True).start()

    def _on_walk_go(self, msg):
        """Queue an approach trigger (label, ts) into the 1-slot go queue.

        Consumed ONLY by the detected-hold via _poll_go(). Freshness is
        judged at consumption time against GO_STALE_S — the pick coordinator
        may edge the trigger a few hundred ms ahead of our own detect.
        Never re-fired; ignored in SEARCHING/APPROACHING by construction
        (only the hold polls it).
        """
        label = msg.data.strip().lower()
        if not label:
            self.get_logger().warn('Empty /walk_go label — ignored.')
            return
        # the trigger carries the same target string; the queue and
        # every comparison keep the bare label
        label = self._set_instance(label)
        with self._cmd_lock:
            self._go_pending = (label, time.time())
        self.get_logger().info(f'/walk_go queued: "{label}"')

    def _on_walk_instance(self, msg):
        """B8: '<label>#<iid>@<hue>' selects ONE object; a bare label clears it."""
        self._set_instance(msg.data.strip().lower(), clear_on_bare=True)

    def _set_instance(self, text, clear_on_bare=False):
        """B8: remember the instance a target string names; return its bare label.

        A string with '#'/'@' sets the instance.  A BARE label on
        /walk_target or /walk_go is what the coordinator relays for every
        navigate, so there it only clears an instance of a DIFFERENT label;
        on /walk_instance (clear_on_bare) it always clears.  Re-sending the
        SAME instance keeps the tracking state (adopted id, last position).
        """
        label, iid, hue = hunt_law.parse_target(text)
        with self._inst_lock:
            cur = self._inst
            if iid is None and hue is None:
                if cur is not None and (
                        clear_on_bare or cur['label'] != ref_to_label(text)):
                    self._inst = None
                return text
            if cur is not None and cur['label'] == label \
                    and cur['iid0'] == iid and cur['hue'] == hue:
                cur['t'] = time.time()
            else:
                self._inst = {'label': label, 'iid0': iid, 'iid': iid,
                              'hue': hue, 'last_xy': None, 'match': 'lost',
                              't': time.time()}
        return label

    def _on_detections(self, msg):
        """Presence-only sightings: refresh the label's last-seen timestamp.

        Empty scenes publish nothing, so the search window timer itself
        detects absence. Body-frame x/y/z comes from /object_detections_body.
        """
        try:
            detections = json.loads(msg.data)
            now = time.time()
            with self._inst_lock:
                inst_label = self._inst['label'] if self._inst else None
            lock_label = self._lock_label()      # a locked label's presence comes from the lock
            with self._visible_lock:
                for d in detections:
                    label = d.get('label')
                    # for an instance target, seeing "a cup" is not seeing the
                    # chosen cup; presence comes from the body topic, which
                    # picks the row
                    if label and str(label) == inst_label:
                        continue
                    if label and lock_label is not None \
                            and str(label) == lock_label:
                        continue
                    if label:
                        self._visible[str(label)] = now
        except Exception as e:
            self.get_logger().warn(f'Bad detection msg: {e}')

    def _on_body_detections(self, msg):
        """Per-label best target from /object_detections_body (body frame).

        Within one message keep ONE entry per label: the highest-conf row,
        except that rows within TIE_CONF_BAND of the best go to the one
        nearest the label's fresh previous target (B8h, _choose_row).
        """
        try:
            detections = json.loads(msg.data)
            if isinstance(detections, list):    # drop the robot's own body rows
                detections = [d for d in detections if not self._self_row(d)]
            now = time.time()
            if isinstance(detections, list):    # obstacles for a pose walk
                self._body_rows = (now, detections)
            best = {}
            # for an instance target only that label's selected row
            # can be the target (none while lost)
            inst_label, inst_row = self._select_instance_row(detections, now)
            # follows the one locked object
            tgt = getattr(self, '_target_label', None)
            lock_on = bool(WALK_LOCK and inst_label is None and tgt
                           and getattr(self, '_alock_live', False)
                           and getattr(self, '_alock', None) is not None)
            lock_armed, lock_row, lock_how = False, None, None
            if lock_on:
                pose = self._odom_pose()
                with self._targets_lock:
                    lock_armed = self._alock.armed()
                    if lock_armed:
                        lock_row, lock_how, lev = self._alock.select(
                            detections, now, pose, MIN_CONF)
                    else:
                        lev = []
                for ev in lev:
                    self.get_logger().info(ev)
            rows_by_label = {}
            for d in detections:
                if d.get('conf', 0) < MIN_CONF:
                    continue
                label = d.get('label')
                if not label:
                    continue
                if inst_label is not None and str(label) == inst_label \
                        and d is not inst_row:
                    continue
                if lock_armed and str(label) == tgt:
                    continue                    # the lock supplies this label
                rows_by_label.setdefault(label, []).append(d)
            # last target per label (fresh only), for the tie-break
            # below
            with self._targets_lock:
                prev = dict((k, v) for k, v in self._targets.items()
                            if now - v['t'] <= STALE_DETECTION_S)
            iids = {}                           # instance id of the chosen row
            for label, rows in rows_by_label.items():
                d = self._choose_row(rows, prev.get(label))
                iids[label] = hunt_law._row_iid(d)
                conf = d.get('conf', 0)
                best[label] = {
                    'label': str(label),
                    'x': float(d['x']),
                    'y': float(d['y']),
                    'z': float(d['z']),
                    'conf': conf,
                    't': now,
                    # published cloud extent (p95-p5); None when it could not be
                    # computed, and the centroid law takes over
                    'width_m': (float(d['width_m'])
                                if d.get('width_m') is not None else None),
                    'depth_m': (float(d['depth_m'])
                                if d.get('depth_m') is not None else None),
                    # capture stamp forwarded by the projector, so the stability
                    # gate measures the frame gap rather than the receipt gap;
                    # None when the projector sends none, and receipt time is used
                    'stamp': self._det_stamp(d),
                }
            # the lock's row is the target, whatever label it carries;
            # armed by the FSM at approach start, never from one lone
            # sweep packet
            if lock_armed:
                if lock_row is not None:
                    best[tgt] = self._lock_entry(tgt, lock_row, lock_how, now)
                    iids[tgt] = hunt_law._row_iid(lock_row)
                    with self._visible_lock:
                        self._visible[tgt] = now
            # continuity: mid-approach, a missing label is continued by
            # a co-located row of another label (the class flipped, the
            # object did not move)
            # not while the lock is armed, since the lock follows the
            # object across labels
            if (WALK_CONTINUITY and getattr(self, '_cont_armed', False)
                    and tgt and not lock_armed):
                if tgt not in best:
                    entry, iid = self._continue_target(detections, tgt, now)
                    if entry is not None:
                        best[tgt] = entry
                        iids[tgt] = iid
                elif getattr(self, '_cont_note', None) is not None:
                    self._cont_note = None      # the target's own label is back
            with self._targets_lock:
                for label, entry in best.items():
                    self._targets[label] = entry
                if getattr(self, '_row_iids', None) is None:
                    self._row_iids = {}
                self._row_iids.update(iids)
        except Exception as e:
            self.get_logger().warn(f'Bad body-detection msg: {e}')

    def _continue_target(self, detections, label, now):
        """B8j: (entry, iid) continuing `label` from a row of ANOTHER label at
        the target's last position (hunt_law.continuation), else (None, None).

        Only within hunt_law.CONT_WINDOW_S of the target's last row, only a
        row within CONT_RADIUS_M (planar) and CONT_DZ_M (height) of it, the
        row carrying the target's instance id first; a row WITHOUT that id
        only for a large-tier target (a pickable can have a different small
        object 0.16 m away; a bin cannot).  The entry is stored
        under the TARGET label (the stop law keeps the target's tier) and is
        marked cont='iid'|'near' + cont_label.  An object elsewhere never
        becomes the target.
        """
        with self._targets_lock:
            last = self._targets.get(label)
            last_iid = (getattr(self, '_row_iids', None) or {}).get(label)
        if last is None or now - last['t'] > hunt_law.CONT_WINDOW_S:
            return None, None
        row, how = hunt_law.continuation(
            detections, label, last, last_iid, MIN_CONF,
            allow_near=hunt_law.stop_tier(label) == 'large')
        if row is None:
            return None, None
        entry = {
            'label': str(label),
            'x': float(row['x']),
            'y': float(row['y']),
            'z': float(row['z']),
            'conf': row.get('conf', 0),
            't': now,
            'width_m': (float(row['width_m'])
                        if row.get('width_m') is not None else None),
            'depth_m': (float(row['depth_m'])
                        if row.get('depth_m') is not None else None),
            'stamp': self._det_stamp(row),
            'cont': how,
            'cont_label': str(row.get('label')),
        }
        iid = last_iid if last_iid is not None else hunt_law._row_iid(row)
        note = (entry['cont_label'], how)
        if note != getattr(self, '_cont_note', None):
            self._cont_note = note
            self.get_logger().info(
                f'[cont] "{label}" continued by "{entry["cont_label"]}" '
                f'({how}{"" if iid is None else " iid=%d" % iid}, '
                f'dxy={math.hypot(entry["x"] - last["x"], entry["y"] - last["y"]):.3f} m, '
                f'dz={abs(entry["z"] - last["z"]):.3f} m, '
                f'{now - last["t"]:.2f}s after the last "{label}" row) '
                f'at d={math.hypot(entry["x"], entry["y"]):.3f}')
        return entry, iid

    # -- approach-lock helpers ----------------------------------------------------

