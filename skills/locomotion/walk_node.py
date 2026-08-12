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

    def _lock_label(self):
        """The target label whose presence comes ONLY from the lock (armed
        lock in a live bare-label run), else None."""
        if not WALK_LOCK or not getattr(self, '_alock_live', False):
            return None
        lock = getattr(self, '_alock', None)
        tlock = getattr(self, '_targets_lock', None)
        if lock is None or tlock is None:
            return None
        with tlock:
            st = lock.st
            return st['label'] if st is not None else None

    def _odom_pose(self):
        """(x, y, yaw) of a FRESH odometry pose (<= walk_lock.LOCK_ODOM_FRESH_S
        old), else None."""
        olock = getattr(self, '_odom_lock', None)
        if olock is None:
            return None
        with olock:
            t = getattr(self, '_odom_t', 0.0)
            if not t or time.time() - t > walk_lock.LOCK_ODOM_FRESH_S:
                return None
            return (self._robot_x, self._robot_y, self._robot_yaw)

    def _lock_entry(self, label, row, how, now):
        """B8l: a _targets entry for the lock's row, stored under the TARGET
        label (the stop law keeps the target's tier); a row of another label
        is marked cont='lock' + cont_label like a B8j continuation."""
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
            'lock': how,
        }
        if str(row.get('label')) != str(label):
            entry['cont'] = 'lock'
            entry['cont_label'] = str(row.get('label'))
        return entry

    def _lock_arm_from(self, pos):
        """B8l F1: arm the approach lock on the row the approach starts on
        (the coordinator confirmed it: /walk_go after its beats).  Not for an
        explicit instance target (the B8 path) and not for 'person' (a person
        moves: the B8k best-row walk follows them, review r1 L8)."""
        if not WALK_LOCK or not getattr(self, '_alock_live', False):
            return
        lock = getattr(self, '_alock', None)
        label = self._target_label
        if lock is None or not label or str(label) == 'person':
            return
        with self._inst_lock:
            if self._inst is not None:
                return
        row = {'label': pos.get('cont_label') or label, 'x': pos['x'],
               'y': pos['y'], 'z': pos.get('z', 0.0),
               'instance_id': (getattr(self, '_row_iids', None) or {}).get(label)}
        pose = self._odom_pose()
        large = hunt_law.stop_tier(label) == 'large'
        with self._targets_lock:
            ev = lock.arm(label, row, time.time(), pose, large=large)
        if ev:
            self.get_logger().info(ev)

    def _lock_reset(self, why, live=None):
        """Clear the approach lock (logged when one was armed); ``live`` sets
        whether a new one may arm."""
        lock = getattr(self, '_alock', None)
        tlock = getattr(self, '_targets_lock', None)
        if lock is None or tlock is None:
            return
        with tlock:
            ev = lock.clear(why)
            if live is not None:
                self._alock_live = bool(live)
        if ev:
            self.get_logger().info(ev)

    @staticmethod
    def _self_row(d):
        """B8i: True for a row inside SELF_ZONE_M (planar body-frame range) --
        the robot's own arm, hand or carried object.  A row without a usable
        x/y is NOT filtered here (the code below treats it as before)."""
        try:
            return math.hypot(float(d['x']), float(d['y'])) < SELF_ZONE_M
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    @staticmethod
    def _choose_row(rows, prev):
        """B8h: the one row of a label that becomes its target.

        Highest conf wins (first row on an exact tie).  With several rows
        inside TIE_CONF_BAND of the best, and a FRESH previous target for the
        label, the tie goes to the row nearest the previous target (body-frame
        x/y), so two same-label objects stop trading the target as their
        confidences cross (s1 2026-10-08 21:56:07: two boxes, yaw_err swung
        -5.7 / +18.6 / -6.1 / +20.2 deg).  A clearly better row (more than
        the band above the rest) still wins, and with no fresh previous
        target the old highest-conf rule applies unchanged."""
        top = rows[0]
        for d in rows[1:]:
            if d.get('conf', 0) > top.get('conf', 0):
                top = d
        if prev is None or len(rows) < 2:
            return top
        floor = top.get('conf', 0) - TIE_CONF_BAND
        tied = [d for d in rows if d.get('conf', 0) >= floor]
        if len(tied) < 2:
            return top
        best, best_d = top, None
        for d in tied:
            try:
                dist = math.hypot(float(d['x']) - prev['x'],
                                  float(d['y']) - prev['y'])
            except (KeyError, TypeError, ValueError):
                continue
            if best_d is None or dist < best_d:
                best, best_d = d, dist
        return best

    def _select_instance_row(self, detections, now):
        """B8: (label, row|None) for the active instance target, else (None, None).

        Runs hunt_law.select_instance over this message's rows of the target
        label (conf >= MIN_CONF), adopts the id after a colour/position
        match, remembers the position, and refreshes the label's presence
        ONLY when the instance itself was found.
        """
        with self._inst_lock:
            inst = self._inst
            if inst is None:
                return None, None
            rows = [d for d in detections
                    if isinstance(d, dict) and str(d.get('label')) == inst['label']
                    and d.get('conf', 0) >= MIN_CONF]
            row, match = hunt_law.select_instance(
                rows, inst['iid'], inst['hue'], inst['last_xy'])
            inst['match'] = match
            if row is not None:
                rid = row.get('instance_id')
                if match != 'iid' and isinstance(rid, int) \
                        and not isinstance(rid, bool) and rid >= 1:
                    inst['iid'] = rid
                inst['last_xy'] = (float(row['x']), float(row['y']))
            label = inst['label']
        if row is not None:
            with self._visible_lock:
                self._visible[label] = now
        return label, row

    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # yaw from the quaternion (ZYX euler)
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._odom_lock:
            self._robot_x = p.x
            self._robot_y = p.y
            self._robot_yaw = yaw
            self._odom_t = time.time()

    # -- publishing status --------------------------------------------------------

    def _publish_status(self, state, label=None, detail=None):
        payload = {'state': state}
        if label is not None:
            payload['label'] = label
        if detail is not None:
            payload['detail'] = detail
        # name the object when the target is an instance
        with self._inst_lock:
            inst = dict(self._inst) if self._inst else None
        if inst is not None and label == inst['label']:
            payload['detail'] = dict(payload.get('detail') or {})
            payload['detail']['instance'] = {
                'iid': inst['iid'], 'hue': inst['hue'], 'match': inst['match']}
        msg = String()
        msg.data = json.dumps(payload)
        self._status_pub.publish(msg)
        self.get_logger().info(f'[STATUS] {msg.data}')

    # -- the one place commands get drained ---------------------------------------

    def _poll_command(self):
        """Single command choke point, called every loop iteration in both phases."""
        with self._cmd_lock:
            p, self._pending = self._pending, None
        if p is None:
            return 'none'
        kind, val = p
        if kind == 'target':                    # applied here, in one place
            if val != self._target_label:       # a new label resets the sweep side
                self._sweep_pref = None
            self._target_label = val
            self._sweep_dir = SWEEP_SIGN
            self._hunt.reset()                  # reset hunt counters
            self._lock_reset('retarget')        # new hunt, new lock
            self.get_logger().info(f'Re-targeting: "{val}"')
            return 'retarget'
        if kind == 'pose':                      # a pose walk takes priority
            self._deferred = ('pose', val)      # started after this run ends
            self._fsm_exit_reason = 'preempted'
            self.get_logger().info('Walk-to-pose requested — ending the '
                                   f'"{self._target_label}" run first.')
            return 'stop'
        # record why the FSM is leaving at the one point that decides
        # it; every stop the FSM acts on comes from here, so the
        # closing IDLE frame cannot miss or invent one
        self._fsm_exit_reason = 'stop'
        return 'stop'

    def _poll_go(self):
        """Consume the /walk_go 1-slot queue; returns (label, ts) or None."""
        with self._cmd_lock:
            go, self._go_pending = self._go_pending, None
        return go

    # -- driving the FSM ----------------------------------------------------------

    def _run_fsm(self, label):
        self._target_label = label
        self._sweep_dir = SWEEP_SIGN
        self._sweep_pref = None                 # no side asked yet
        self._lock_reset('new run', live=WALK_LOCK)   # arm a new lock
        self._hunt.reset()                      # reset hunt counters
        # fresh per run: an exit reason from a finished run must
        # never leak in
        self._fsm_exit_reason = None
        try:
            self._connect_loco()
            # pre-flight: warn on stale odometry; the fresh path is
            # body-frame, but stale odometry degrades the world-latch
            # fallback
            with self._odom_lock:
                odom_age = time.time() - self._odom_t if self._odom_t else -1.0
            if odom_age < 0 or odom_age > 3.0:
                self.get_logger().warn(
                    'No fresh /unitree/slam_mapping/odom '
                    f'(age={odom_age:.1f}s) — running body-frame-only; the '
                    'world-latch fallback is DEGRADED (verify motion visually).')
            # pre-flight FSM gate: 0 = walking-capable, 3 = gait blocked
            # after boot; code-0 RPCs would otherwise hide the block
            fsm_mode = None
            if hasattr(self._loco, 'get_fsm_mode'):
                try:
                    fsm_mode = self._loco.get_fsm_mode()
                except Exception as e:
                    self.get_logger().warn(f'fsm_mode read failed: {e!r}')
            if fsm_mode not in (None, 0):
                self.get_logger().warn(
                    f'fsm_mode={fsm_mode} (expected 0) — gait may be BLOCKED. '
                    'Redo the boot sequence (L1+B → L2+UP → R2+A) and retry.')
            state = 'SEARCHING'
            self._publish_status('SEARCHING', label=self._target_label,
                                 detail={'strategy': SWEEP_STRATEGY})
            while state != 'IDLE':
                if state == 'SEARCHING':
                    state = self._phase_search()
                elif state == 'APPROACHING':
                    state = self._phase_approach()
                    self._cont_armed = False    # continuity ends here
                    if state != 'REACHED':      # REACHED keeps the lock
                        self._lock_reset('approach ended (%s)' % state)
                elif state == 'NOT_FOUND':
                    # hunt budget exhausted: stop, publish proof counters and
                    # return IDLE; no success state is reported
                    self._stop_motion()
                    proof = self._hunt.proof()
                    self._publish_status('NOT_FOUND', label=self._target_label,
                                         detail=proof)
                    self.get_logger().warn(
                        f'NOT_FOUND "{self._target_label}" — hunt exhausted: '
                        f'{proof}')
                    self._fsm_exit_reason = 'not_found'
                    self._publish_status('IDLE',
                                         detail={'not_found': True})
                    state = 'IDLE'
                elif state == 'REACHED':
                    # republish REACHED at 1 Hz for REACHED_HOLD_S so late
                    # joiners catch it, while honouring stop and retarget;
                    # stop-law evidence rides along
                    detail = {'standby': True}
                    if self._reach_evidence:
                        detail['stop_law'] = self._reach_evidence
                    self._publish_status('REACHED', label=self._target_label,
                                         detail=detail)
                    hold_until = time.time() + REACHED_HOLD_S
                    settle_at = time.time() + SETTLE_LOG_S   # schedule the settle log
                    next_state = 'IDLE'
                    while time.time() < hold_until:
                        cmd = self._poll_command()
                        if cmd == 'stop':
                            break
                        if cmd == 'retarget':
                            # the new label is applied
                            next_state = 'SEARCHING'
                            break
                        self._publish_status('REACHED',
                                             label=self._target_label,
                                             detail={'standby': True})
                        if settle_at and time.time() >= settle_at:
                            settle_at = 0.0
                            self._log_settled()
                        time.sleep(1.0)
                    state = next_state
        except Exception as e:
            self.get_logger().error(f'FSM crashed: {e!r}')
            self._fsm_exit_reason = 'crash'
            self._stop_motion()
            self._publish_status('IDLE', detail={'error': str(e)})
        finally:
            self._cont_armed = False            # continuity off after a run
            self._lock_reset('run ended', live=False)   # no lock outlives the run
            self._stop_motion()
            with self._cmd_lock:
                self._busy = False
            # drop the instance at run end unless a new navigate armed
            # one (its /walk_target may still be in flight)
            with self._inst_lock:
                if self._inst is not None and \
                        time.time() - self._inst['t'] > INSTANCE_KEEP_S:
                    self._inst = None
            # single exit for every path: publish one IDLE frame with the
            # reason recorded at the decision point, so a dead FSM never
            # leaves its last frame standing
            # NOT_FOUND and crash publish their own IDLE first; this
            # second frame is additive, and publishing it after _busy
            # clears lets a consumer dispatch a new target at once
            self._publish_status(
                'IDLE', detail={'exit': self._fsm_exit_reason or 'finished'})
            self._start_deferred()              # start the command handed over

    def _phase_search(self):
        look = True                              # look before turning
        while True:
            # hunt budget check when a window opens: 12 completed turns
            # make a round, and the FSM gives up at round end rather
            # than moving to another position; stalls count too
            if self._hunt.exhausted:
                if SWEEP_CONTINUOUS:
                    self._stop_motion()          # stop the turn
                return 'NOT_FOUND'
            if SWEEP_CONTINUOUS:                 # take a look on entry
                window_end = time.time() + (CONT_FIRST_LOOK_S if look else 0.0)
            else:
                window_end = time.time() + DETECT_WINDOW_S
            while time.time() < window_end:
                cmd = self._poll_command()
                if cmd == 'stop':
                    return 'IDLE'
                if cmd == 'retarget':
                    self._publish_status('SEARCHING', label=self._target_label,
                                         detail={'retarget': True})
                    return 'SEARCHING'           # restart with the new label
                label = self._target_label
                pos = self._fresh_target(label)
                if pos is not None:
                    in_zone, ev = self._stop_law(
                        pos['label'], pos['x'], pos['y'], pos.get('z', 0.0),
                        pos.get('width_m'), pos.get('depth_m'), STOP_DISTANCE_M)
                    if in_zone:
                        self._reach_evidence = ev
                        self._stop_motion()
                        return 'REACHED'         # inside the gate
                if self._label_visible(label):
                    outcome = self._phase_hold()   # the detected hold
                    if outcome != 'SEARCHING':
                        return outcome
                    # the hold recycled; the budget may be spent, so give up
                    # at once: a visible label never expires a window, and
                    # another cycle would be unbounded
                    if self._hunt.exhausted:
                        return 'NOT_FOUND'
                    window_end = time.time() + (CONT_FIRST_LOOK_S if SWEEP_CONTINUOUS
                                                else DETECT_WINDOW_S)  # new window after the hold
                time.sleep(DETECT_POLL_S)
            # window expired: one 30 deg turn, then a fresh window
            self._hunt.add_window()
            look = False
            result = (self._sweep_continuous() if SWEEP_CONTINUOUS
                      else self._sweep_turn())
            if result == 'stop':                 # continuous mode only
                return 'IDLE'
            if result == 'retarget':             # continuous mode only
                self._publish_status('SEARCHING', label=self._target_label,
                                     detail={'retarget': True})
                return 'SEARCHING'
            if result == 'interrupted':
                look = True
                # first detection mid-turn: stop the sweep and hold; an
                # incomplete turn does not count toward the round
                outcome = self._phase_hold()
                if outcome != 'SEARCHING':
                    return outcome
                if self._hunt.exhausted:
                    return 'NOT_FOUND'
            else:
                self._hunt.add_turn()            # completed 30 deg turn
                self._publish_status('SEARCHING', label=self._target_label,
                                     detail={'window_expired': True})
                if self._hunt.round_complete:
                    self.get_logger().info(
                        f'Search round complete at this position '
                        f'(turns={self._hunt.turns}); giving up (decision 1).')

    def _sweep_continuous(self):
        """hf15: one CONTINUOUS sweep segment (CONT_SEGMENT_S).  Returns
        'interrupted' (first sighting: motion stopped), 'stop' / 'retarget'
        (an operator command: motion stopped) or 'done' (segment completed,
        motion NOT stopped: the next segment continues the same turn)."""
        if SWEEP_STRATEGY == 'alternate':
            self._sweep_dir = -self._sweep_dir
        else:
            self._sweep_dir = getattr(self, '_sweep_pref', None) or SWEEP_SIGN
        omega = self._sweep_dir * YAW_RATE_RAD_S
        self.get_logger().info(
            f'Sweep segment (continuous): omega={omega:+.2f} rad/s for '
            f'{CONT_SEGMENT_S:.2f}s')
        end = time.time() + CONT_SEGMENT_S
        next_cmd = 0.0
        while time.time() < end:
            cmd = self._poll_command()
            if cmd in ('stop', 'retarget'):
                self._stop_motion()
                return cmd
            now = time.time()
            if now >= next_cmd:
                self._loco.set_velocity(0.0, 0.0, omega,
                                        duration=CONT_CMD_DURATION_S)
                next_cmd = now + CONT_RENEW_S
            if self._label_visible(self._target_label):
                self._stop_motion()
                self.get_logger().info(
                    'Sweep interrupted: label visible — sweep stopped.')
                return 'interrupted'
            time.sleep(DETECT_POLL_S)
        return 'done'

    def _sweep_turn(self):
        if SWEEP_STRATEGY == 'alternate':
            self._sweep_dir = -self._sweep_dir
        else:                                   # one-way: 360 deg total
            self._sweep_dir = getattr(self, '_sweep_pref', None) or SWEEP_SIGN
        omega = self._sweep_dir * YAW_RATE_RAD_S
        self.get_logger().info(
            f'Sweep turn: omega={omega:+.2f} rad/s for {TURN_DURATION_S:.2f}s')
        self._loco.set_velocity(0.0, 0.0, omega, duration=TURN_DURATION_S)
        deadline = time.time() + TURN_DURATION_S
        while time.time() < deadline:
            # stop on the first sighting; presence comes from the
            # presence-only topic, since a far target appears well
            # before its depth is usable and the hold copes with late
            # depth
            if self._label_visible(self._target_label):
                self._stop_motion()
                self.get_logger().info(
                    'Sweep interrupted: label visible — sweep stopped.')
                return 'interrupted'
            time.sleep(DETECT_POLL_S)
        time.sleep(TURN_SETTLE_S)
        self._stop_motion()
        return 'done'

    def _phase_hold(self):
        """Detected-hold (T10): stand still — no turn, no walk.

        Runs inside SEARCHING (the DETECTED status is a /walk_status
        vocabulary event, not an FSM state). Publishes DETECTED EXACTLY ONCE
        at entry, then loops at DETECT_POLL_S until /walk_go (matching label,
        GO_STALE_S-fresh), a stop, a retarget, or the detection is lost.
        /walk_target is NEVER a trigger — /walk_go is the only trigger channel.
        """
        label = self._target_label
        self._publish_status('DETECTED', label=label, detail={'once': True})
        self.get_logger().info(
            f'DETECTED "{label}" — holding (no turn, no walk); waiting for '
            f'{WALK_GO_TOPIC} with a matching label.')
        last_visible = time.time()
        no_pos_since = None
        while True:
            cmd = self._poll_command()
            if cmd == 'stop':
                return 'IDLE'
            if cmd == 'retarget':               # a target replays as a retarget,
                self._publish_status(           # never as a trigger
                    'SEARCHING', label=self._target_label,
                    detail={'re_sweep': True, 'reason': 'retarget'})
                return 'SEARCHING'
            go = self._poll_go()
            if go is not None:
                go_label, ts = go
                if time.time() - ts > GO_STALE_S:
                    self.get_logger().warn(
                        f'/walk_go "{go_label}" is {time.time() - ts:.1f}s old '
                        f'(> {GO_STALE_S:.0f}s) — dropped (stay DETECTED).')
                elif go_label != label:
                    self.get_logger().warn(
                        f'/walk_go label "{go_label}" != active "{label}" '
                        '— dropped (stay DETECTED).')
                else:
                    pos = self._fresh_target(label)   # recheck at trigger time
                    if pos is None:
                        # no-progress recycle: the trigger arrived but there is no
                        # fresh position, so count a stall; this must stay bounded
                        # because the label is usually visible, so the window never
                        # expires
                        self._hunt.add_stall('trigger_stale')
                        self.get_logger().warn(
                            f'/walk_go "{label}" but no fresh position — back '
                            'to SEARCHING (never walk on a stale trigger).')
                        self._publish_status(
                            'SEARCHING', label=label,
                            detail={'re_sweep': True, 'reason': 'stale_trigger'})
                        return 'SEARCHING'
                    in_zone, ev = self._stop_law(
                        pos['label'], pos['x'], pos['y'], pos.get('z', 0.0),
                        pos.get('width_m'), pos.get('depth_m'), STOP_DISTANCE_M)
                    if in_zone:
                        self._reach_evidence = ev
                        self._stop_motion()
                        return 'REACHED'        # inside the gate
                    self._latch_target(pos)
                    self._lock_arm_from(pos)    # the row that got confirmed
                    return 'APPROACHING'
            pos = self._fresh_target(label)
            if self._label_visible(label):
                last_visible = time.time()
                if pos is not None:
                    no_pos_since = None
                elif no_pos_since is None:
                    no_pos_since = time.time()
                elif time.time() - no_pos_since >= HOLD_MAX_S:
                    # hold escape: presence without position; the recycle spends
                    # the stall budget, so this stays bounded even though the
                    # window never expires
                    self._hunt.add_stall('hold_depth_stale')
                    self.get_logger().warn(
                        f'"{label}" visible but depth-invalid for '
                        f'{HOLD_MAX_S:.0f}s — back to SEARCHING')
                    self._publish_status(
                        'SEARCHING', label=label,
                        detail={'re_sweep': True, 'reason': 'depth_stale'})
                    return 'SEARCHING'
            elif time.time() - last_visible >= LOST_TIMEOUT_S:
                self.get_logger().warn(
                    f'"{label}" lost for {LOST_TIMEOUT_S:.0f}s during hold '
                    '— back to SEARCHING')
                self._lost_side(label)          # sweep that side
                self._lock_reset('lost during hold')
                self._publish_status(
                    'SEARCHING', label=label,
                    detail={'re_sweep': True, 'reason': 'lost'})
                return 'SEARCHING'
            time.sleep(DETECT_POLL_S)

    def _phase_approach(self):
        self._cont_armed = True     # continuity is armed here only;
                                    # _run_fsm disarms it
        last_anchor = time.time()
        last_heartbeat = 0.0
        # fresh stall watchdog per approach (15 s, >= 0.05 m)
        stall_wd = hunt_law.StallWatchdog(STALL_WINDOW_S, STALL_MIN_PROGRESS)
        # frame-to-frame displacement, reset per approach; no state
        # is inherited
        prev_meas = None
        jumps = 0
        # a tick whose fresh sample the gate judged impossible: that
        # sample, and the latched copy the loop falls back to after
        # it, must not reach the stop law, or it would commit the
        # arrival the gate exists to prevent
        last_fresh_rejected = False
        # detour budget and alignment clock, reset per approach
        detours = 0
        detour_until = 0.0
        detour_sign = 0.0
        align_since = 0.0
        blind_since = 0.0           # start of the blind hold
        reacq_used = False          # one reacquire per approach
        while True:
            cmd = self._poll_command()
            if cmd == 'stop':
                self._stop_motion()
                return 'IDLE'
            if cmd == 'retarget':
                self._stop_motion()
                return 'SEARCHING'
            fresh = self._fresh_target(self._target_label)
            if fresh is not None:               # re-anchor on every fresh row
                self._latch_target(fresh)
                last_anchor = time.time()
            else:
                # arrival latch: the stop law below needs a fresh detection,
                # but the projector drops the label at close range, so the
                # robot can sit inside the gate with no fresh measurement and
                # roll back
                # once the retained measurement lands in the zone, commit
                # REACHED for ARRIVAL_LATCH_S even as the label goes stale
                # evaluated before the lost-track rollbacks, so the loss the
                # projector causes cannot undo the arrival
                anchor = self._retained_target(self._target_label)
                if anchor is not None:
                    latched, lev = hunt_law.arrival_latch(
                        anchor, time.time(), STOP_DISTANCE_M, ARRIVAL_LATCH_S,
                        law=self._stop_law)
                    if latched:
                        self._reach_evidence = lev
                        self.get_logger().info(
                            f'Arrival latch: last-known '
                            f'"{self._target_label}" in stop zone '
                            f'({lev.get("law")}, age '
                            f'{lev.get("latch_age_s")}s) — REACHED')
                        self._stop_motion()
                        return 'REACHED'
            if time.time() - last_anchor > LOST_TIMEOUT_S:
                # target vanished: count one stall so a flickering visible
                # label cannot recycle forever; the count happens once inside
                # _after_stall, and a separate count here would burn the
                # whole stall budget
                self.get_logger().warn(
                    f'No fresh "{self._target_label}" for {LOST_TIMEOUT_S:.0f}s '
                    '— back to SEARCHING')
                self._stop_motion()
                r = self._approach_reacquire(reacq_used)
                reacq_used = reacq_used or WALK_REACQUIRE
                if r == 'stop':
                    return 'IDLE'
                if r == 'retarget':
                    return 'SEARCHING'
                if r == 'found':
                    last_anchor = time.time()
                    prev_meas, jumps, last_fresh_rejected = None, 0, False
                    blind_since, align_since = 0.0, 0.0
                    stall_wd = hunt_law.StallWatchdog(STALL_WINDOW_S,
                                                      STALL_MIN_PROGRESS)
                    continue
                nxt = self._after_stall('approach_lost')
                if nxt == 'SEARCHING' and WALK_REACQUIRE:   # announce the re-sweep
                    self._publish_status('SEARCHING', label=self._target_label,
                                         detail={'re_sweep': True,
                                                 'reason': 'approach_lost'})
                return nxt
            cur = self._current_target()
            if cur is None:
                # main loss path in practice: the row goes stale
                # (STALE_DETECTION_S) before LOST_TIMEOUT_S expires, so the
                # branch above rarely runs
                self._stop_motion()
                r = self._approach_reacquire(reacq_used)
                reacq_used = reacq_used or WALK_REACQUIRE
                if r == 'stop':
                    return 'IDLE'
                if r == 'found':
                    last_anchor = time.time()
                    prev_meas, jumps, last_fresh_rejected = None, 0, False
                    blind_since, align_since = 0.0, 0.0
                    stall_wd = hunt_law.StallWatchdog(STALL_WINDOW_S,
                                                      STALL_MIN_PROGRESS)
                    continue
                if r != 'retarget':
                    self._lock_reset('lost during approach')
                if r != 'retarget' and WALK_REACQUIRE:
                    # announce the re-sweep so the caller can re-narrow the
                    # target
                    self._publish_status('SEARCHING', label=self._target_label,
                                         detail={'re_sweep': True,
                                                 'reason': 'target_stale'})
                return 'SEARCHING'                  # no stall counted
            x_b, y_b, _z, _lab = cur
            d = math.hypot(x_b, y_b)
            now = time.time()
            # target-stability gate: runs before the stop law and the
            # corridor gate because both consume the position
            # only fresh detections are compared, and the reference sample
            # is only replaced by a fresh one; the latched value is
            # returned in between, so comparing every tick would reset the
            # streak and the gate could never fire (fresh samples are about
            # 0.1 s apart)
            # consume_ok: the position may be acted on this tick
            consume_ok = not last_fresh_rejected
            if fresh is not None:
                _st = self._det_stamp(fresh)
                if prev_meas is not None:
                    _px, _py, _pt_st, _pt_rx = prev_meas
                    # dt is the window the object had to move in: the capture
                    # gap, not the receipt gap; under load they differ by up to
                    # ~0.27 s, the whole noise budget at 0.60 rad/s
                    _dt = (_st - _pt_st
                           if (_st is not None and _pt_st is not None)
                           else now - _pt_rx)
                    if 0.0 < _dt <= TARGET_JUMP_DT_S:
                        _jump = math.hypot(x_b - _px, y_b - _py)
                        # the most the robot's own motion could have moved a
                        # stationary object in _dt: translation plus the arc swept at
                        # the real yaw ceiling, added to the noise floor
                        _ceil = (VX_WALK + YAW_RATE_MAX * d) * _dt
                        _bound = TARGET_JUMP_MIN_M + _ceil
                        if _jump > _bound:
                            jumps += 1
                            last_fresh_rejected = True
                            consume_ok = False
                            self.get_logger().warn(
                                f'G18: "{self._target_label}" jumped '
                                f'{_jump:.2f} m in {_dt:.2f}s (bound '
                                f'{_bound:.2f} m at d={d:.2f}) — not physical '
                                f'({jumps}/{TARGET_UNSTABLE_N})')
                            if jumps >= TARGET_UNSTABLE_N:
                                self.get_logger().warn(
                                    f'G18: "{self._target_label}" position is '
                                    'not self-consistent — refusing the '
                                    'approach (walking to it would stop at the '
                                    'wrong object).')
                                self._stop_motion()
                                self._publish_status(
                                    'SEARCHING', label=self._target_label,
                                    detail={'re_sweep': True,
                                            'reason': 'target_unstable'})
                                return self._after_stall('target_unstable')
                        else:
                            jumps = 0       # honest motion clears it
                            last_fresh_rejected = False
                    else:
                        # gap longer than TARGET_JUMP_DT_S (a missed detection) or a
                        # backwards stamp: no evidence either way, so neither count
                        # nor clear; the sample still becomes the reference, but the
                        # rejected flag describes the current position and must be
                        # cleared
                        last_fresh_rejected = False
                prev_meas = (x_b, y_b, _st, now)
            if not consume_ok:
                # skip the consumption, not the loop period: a bare continue
                # would spin this thread at full speed
                time.sleep(CTRL_DT_S)
                continue
            # blind cap: no forward motion on a target row past this age;
            # without it a stale row plus the approach speed can overrun
            # the target
            # past BLIND_MAX_S the robot holds with no vx and no omega
            # until a fresh row appears; a stale row inside the stop zone
            # still commits REACHED below
            _blind = self._blind_age(fresh, now)
            if _blind is not None:
                _inz = detour_until <= now and self._stop_law(
                    fresh['label'], fresh['x'], fresh['y'],
                    fresh.get('z', 0.0), fresh.get('width_m'),
                    fresh.get('depth_m'), STOP_DISTANCE_M)[0]
                if not _inz:
                    if not blind_since:
                        blind_since = now
                        self._stop_motion()
                        self.get_logger().warn(
                            f'[blind] "{self._target_label}" row is '
                            f'{_blind:.2f}s old (> {BLIND_MAX_S:.2f}s) at '
                            f'd={d:.3f} — holding, no forward motion until '
                            'a fresh row.')
                    time.sleep(CTRL_DT_S)
                    continue
            elif blind_since:
                self.get_logger().info(
                    f'[blind] "{self._target_label}" fresh row after '
                    f'{now - blind_since:.2f}s of hold — resuming.')
                blind_since = 0.0
            # detour under way: walk the offset bearing and skip both
            # laws below; the corridor gate would re-freeze on the
            # obstacle being rounded and the stop law would fire on a
            # target passed at an angle
            # the detour ends on its clock, not when the corridor clears:
            # the obstacle leaves the sector partway, so a mid-arc clear
            # is expected; the corridor is retested from the new spot
            # once the clock runs out
            if detour_until > now:
                aim = _wrap_pi(math.atan2(x_b, y_b) + detour_sign * DETOUR_RAD)
                omega = OMEGA_SIGN * K_THETA * max(-OMEGA_MAX,
                                                   min(OMEGA_MAX, aim))
                vx = DETOUR_VX      # fixed speed: the approach vx would be zero here
                self._loco.set_velocity(vx, 0.0, omega, duration=1.0)
                # the side comes from the omega sign, not detour_sign:
                # OMEGA_SIGN is -1.0, so a positive aim gives a negative
                # omega and turns right
                self.get_logger().info(
                    f'G17a detour {detours}/{DETOUR_MAX} '
                    f'({"left" if omega > 0 else "right"}): '
                    f'aim={math.degrees(aim):+.1f} vx={vx:.2f} '
                    f'omega={omega:+.2f} ({detour_until - now:.1f}s left)')
                time.sleep(CTRL_DT_S)
                continue
            # corridor gate: a person or furniture between robot and
            # target freezes the approach before the motion beat; the
            # corridor is a bearing within OBSTRUCTION_SECTOR_RAD of the
            # target bearing and closer than the target
            # the freeze holds the target, bounded by
            # COLLISION_HOLD_MAX_S
            ob = self._obstruction_between(x_b, y_b)
            if ob is not None:
                verdict, held = self._corridor.frozen(now)
                self._stop_motion()
                self._publish_status(
                    'SEARCHING', label=self._target_label,
                    detail={'re_sweep': True, 'reason': 'collision_freeze',
                            'obstacle': ob['label'],
                            'obstacle_d_m': round(ob['od'], 3),
                            'held_s': round(held, 1)})
                self.get_logger().warn(
                    f'G17a: "{ob["label"]}" at d={ob["od"]:.2f} m between '
                    f'robot and target — approach frozen '
                    f'({held:.0f}s of {hunt_law.COLLISION_HOLD_MAX_S:.0f}s '
                    'held).')
                # escape route: before spending the whole hold budget on a
                # corridor that will not clear, walk around it, aiming to the
                # side the obstacle is not on
                # a failure degrades to the hold path below: the clock is
                # cleared and the budget is spent normally
                # missing x/y must degrade instead of raising; an exception
                # in the FSM loop costs the whole approach
                _ox, _oy = ob.get('x'), ob.get('y')
                if (held >= DETOUR_AFTER_HOLD_S and detours < DETOUR_MAX
                        and _ox is not None and _oy is not None):
                    ob_bear = math.atan2(_ox, _oy)
                    tgt_bear = math.atan2(x_b, y_b)
                    delta = _wrap_pi(ob_bear - tgt_bear)
                    if detours == 0:
                        # first try: pass on the far side
                        detour_sign = -1.0 if delta > 0.0 else 1.0
                    else:
                        # blocked again: try the other side
                        detour_sign = -detour_sign
                    detours += 1
                    detour_until = now + DETOUR_DURATION_S
                    self._corridor.clear()
                    self.get_logger().warn(
                        f'G17a: detouring around "{ob["label"]}" (obstacle '
                        f'{math.degrees(delta):+.0f} deg off the target '
                        f'bearing, so aiming '
                        f'{math.degrees(detour_sign * DETOUR_RAD):+.0f} deg) '
                        f'— attempt {detours}/{DETOUR_MAX}.')
                    time.sleep(CTRL_DT_S)
                    continue
                if verdict == 'give_up':
                    self.get_logger().warn(
                        f'G17a: corridor blocked for {held:.0f}s — giving up '
                        'on this approach (hold budget spent).')
                    return self._after_stall('collision_freeze')
                return 'SEARCHING'          # keep the target and rejoin the hold
            self._corridor.clear()          # corridor clear, freeze streak broken
            # stall watchdog: no progress in the window stops the
            # robot and re-sweeps, or gives up once the budget is gone
            if stall_wd.update(d, now) == 'stall':
                self.get_logger().warn(
                    f'G5b: no progress for {STALL_WINDOW_S:.0f}s '
                    f'(< {STALL_MIN_PROGRESS:.2f} m) — approach stalls.')
                self._stop_motion()
                self._publish_status(
                    'SEARCHING', label=self._target_label,
                    detail={'re_sweep': True, 'reason': 'stall_watchdog'})
                return self._after_stall('stall_watchdog')
            # stop law: the centroid gate is the floor; wide-table and
            # tall/large targets use the near-edge gate on the published
            # extent, falling back to the centroid when the projector
            # published no extent
            if fresh is not None:
                in_zone, ev = self._stop_law(
                    fresh['label'], fresh['x'], fresh['y'],
                    fresh.get('z', 0.0), fresh.get('width_m'),
                    fresh.get('depth_m'), STOP_DISTANCE_M)
                if fresh.get('cont'):           # the decision used a continuation
                    ev['cont'] = fresh['cont']
                    ev['cont_label'] = fresh.get('cont_label')
            else:
                in_zone, ev = self._stop_law(
                    _lab, x_b, y_b, _z, None, None, STOP_DISTANCE_M)
            if in_zone:
                # line up on arrival: in_zone is a pure distance test, so
                # committing here would stop the robot wherever hypot(x,y)
                # crossed STOP_DISTANCE_M; turn in place until the bearing is
                # inside ALIGN_TOL_RAD, capped by ALIGN_TIMEOUT_S
                yaw_err = _wrap_pi(math.atan2(x_b, y_b))
                if abs(yaw_err) > ALIGN_TOL_RAD:
                    if align_since == 0.0:
                        align_since = now
                        self.get_logger().info(
                            f'Arrival alignment: {math.degrees(yaw_err):+.1f} '
                            f'deg off-axis at d={d:.3f} — centring before '
                            'REACHED.')
                    if now - align_since < ALIGN_TIMEOUT_S:
                        omega = _cap_turn(OMEGA_SIGN * K_THETA * max(
                            -OMEGA_MAX, min(OMEGA_MAX, yaw_err)))   # turn cap
                        self._loco.set_velocity(0.0, 0.0, omega, duration=1.0)
                        time.sleep(CTRL_DT_S)
                        continue
                    self.get_logger().warn(
                        f'Arrival alignment: still '
                        f'{math.degrees(yaw_err):+.1f} deg off-axis after '
                        f'{ALIGN_TIMEOUT_S:.0f}s — committing REACHED '
                        'unaligned.')
                self._reach_evidence = ev
                self.get_logger().info(
                    f'Reached stop gate ({ev.get("law")}): ' + ', '.join(
                        f'{k}={v}' for k, v in sorted(ev.items())))
                self._stop_motion()
                return 'REACHED'
            # out of the zone: reset the alignment clock, so re-entering
            # gets a fresh bounded turn
            align_since = 0.0
            yaw_err = _wrap_pi(math.atan2(x_b, y_b))     # body-frame bearing
            omega = OMEGA_SIGN * K_THETA * max(-OMEGA_MAX, min(OMEGA_MAX, yaw_err))
            omega = _cap_turn(omega)            # cap the turn rate
            vx = VX_WALK * max(0.0, 1.0 - abs(yaw_err) / YAW_VX_DEADBAND)
            if vx < VX_MIN:
                vx = 0.0
            self._loco.set_velocity(vx, 0.0, omega, duration=1.0)   # expires after 1 s
            self.get_logger().info(
                f'APPROACH d={d:.3f} yaw_err={math.degrees(yaw_err):+.1f} '
                f'vx={vx:.2f} omega={omega:+.2f}')
            if now - last_heartbeat >= APPROACH_HEARTBEAT_S:
                self._publish_status('APPROACHING', label=self._target_label,
                                     detail={'distance_m': round(d, 3)})
                last_heartbeat = now
            time.sleep(CTRL_DT_S)

    def _after_stall(self, reason):
        """Common stall-exit: count the stall; give up if the budget is spent."""
        self._lock_reset(reason)                # clear the lock; it re-arms on sight
        self._hunt.add_stall(reason)
        if self._hunt.exhausted:
            self.get_logger().warn(
                f'Epoch exhaust (stalls={self._hunt.stalls}, '
                f'turns={self._hunt.turns}) — NOT_FOUND.')
            return 'NOT_FOUND'
        return 'SEARCHING'

    # -- finding a lost target again ----------------------------------------------

    def _approach_reacquire(self, used):
        """The approach lost its target: the FIRST loss of an approach runs
        _reacquire ('found' | 'failed' | 'stop' | 'retarget'); a later one
        only turns the sweep toward the side it was lost on (None).  None
        too with OPENBOT_WALK_REACQUIRE=0 (B8k)."""
        if not WALK_REACQUIRE:
            return None
        if used:
            self._lost_side(self._target_label)
            return None
        return self._reacquire()

    def _lost_bearing(self, label, odom_only=False):
        """Bearing (walk yaw error, + = right) of where the lost ``label``
        should be NOW: the lock's / the latch's odometry-world position when
        odometry is fresh, else (unless ``odom_only``) the last body-frame
        row.  None when nothing is known."""
        pose = self._odom_pose()
        if pose is not None:
            w = None
            lock = getattr(self, '_alock', None)
            tlock = getattr(self, '_targets_lock', None)
            if lock is not None and tlock is not None:
                with tlock:
                    st = lock.st
                    if st is not None and st.get('w') is not None:
                        w = st['w']
            if w is None and getattr(self, '_twx', None) is not None:
                w = (self._twx, self._twy)
            if w is not None:
                x, y = walk_lock.world_to_body(pose, w[0], w[1])
                return walk_lock.bearing(x, y)
        if odom_only:
            return None
        row = self._retained_target(label)
        if row is None:
            return None
        try:
            return walk_lock.bearing(float(row['x']), float(row['y']))
        except (KeyError, TypeError, ValueError):
            return None

    def _lost_side(self, label, err=None):
        """Turn the rest of this hunt's sweep TOWARD the side ``label`` was
        lost on (B8k: always SWEEP_SIGN)."""
        if not WALK_REACQUIRE:
            return
        if err is None:
            err = self._lost_bearing(label)
        if err is None:
            return
        side = walk_lock.sweep_side(err, default=SWEEP_SIGN,
                                    omega_sign=OMEGA_SIGN)
        self._sweep_pref = side
        self.get_logger().info(
            f'[reacq] "{label}" was last at {math.degrees(err):+.0f} deg — '
            f'the sweep turns {"left" if side > 0 else "right"} first.')

    def _reacq_wait(self, label, secs, since):
        """Stand still up to ``secs``: 'found' (a row of the target newer
        than ``since``), 'stop' / 'retarget', or None (nothing)."""
        end = time.time() + secs
        while time.time() < end:
            cmd = self._poll_command()
            if cmd in ('stop', 'retarget'):
                return cmd
            row = self._fresh_target(label)
            if row is not None and row.get('t', 0.0) >= since:
                return 'found'
            time.sleep(DETECT_POLL_S)
        return None

    def _reacquire(self):
        """B8l F5: the target was lost mid-approach.  Stand and look
        (REACQ_DWELL_S), turn back toward where it was at REACQ_OMEGA
        (closed loop on odometry when fresh, else open loop for the bearing),
        look again (REACQ_LOOK_S).  'found' resumes the approach; 'failed'
        hands over to the sweep, turning first toward the side it was lost
        on.  Published as APPROACHING{reacquire} so the coordinator holds its
        re-sweep and the gateway keeps the full vocabulary (B8l F2)."""
        label = self._target_label
        t0 = time.time()
        err0 = self._lost_bearing(label)
        self._publish_status('APPROACHING', label=label,
                             detail={'reacquire': True,
                                     'bearing_deg': (None if err0 is None else
                                                     round(math.degrees(err0), 1))})
        self.get_logger().info(
            f'[reacq] "{label}" lost — looking {walk_lock.REACQ_DWELL_S:.1f}s, '
            'then turning back toward '
            + ('?' if err0 is None else f'{math.degrees(err0):+.0f} deg'))
        r = self._reacq_wait(label, walk_lock.REACQ_DWELL_S, t0)
        if r is None:
            err = self._lost_bearing(label)
            omega, dur = walk_lock.reacquire_turn(err, omega_sign=OMEGA_SIGN)
            if dur > 0.0:
                self.get_logger().info(
                    f'[reacq] turning back {math.degrees(err):+.0f} deg at '
                    f'omega={omega:+.2f} (<= {dur:.1f}s)')
                end = time.time() + dur
                next_cmd = 0.0
                while time.time() < end:
                    cmd = self._poll_command()
                    if cmd in ('stop', 'retarget'):
                        r = cmd
                        break
                    row = self._fresh_target(label)
                    if row is not None and row.get('t', 0.0) >= t0:
                        r = 'found'
                        break
                    e = self._lost_bearing(label, odom_only=True)
                    if e is not None and (abs(e) <= walk_lock.REACQ_TOL_RAD
                                          or (e > 0) != (err > 0)):
                        break               # odometry says it is straight ahead
                    now = time.time()
                    if now >= next_cmd:
                        self._loco.set_velocity(0.0, 0.0, omega,
                                                duration=CONT_CMD_DURATION_S)
                        next_cmd = now + CONT_RENEW_S
                    time.sleep(DETECT_POLL_S)
                self._stop_motion()
            if r is None:
                r = self._reacq_wait(label, walk_lock.REACQ_LOOK_S, t0)
        if r == 'found':
            self.get_logger().info(
                f'[reacq] "{label}" found again after {time.time() - t0:.1f}s '
                '— resuming the approach.')
            self._publish_status('APPROACHING', label=label,
                                 detail={'reacquired': True})
            return 'found'
        if r in ('stop', 'retarget'):
            return r
        self.get_logger().warn(
            f'[reacq] "{label}" not found in {time.time() - t0:.1f}s — '
            'back to the sweep.')
        self._lost_side(label, err0)
        return 'failed'

    # -- walk to an odometry pose ------------------------------------------------

    def _on_walk_pose(self, msg):
        """/walk_pose: start (or queue) a walk to an odometry pose."""
        kind, goal = walk_lock.parse_pose_goal(getattr(msg, 'data', ''))
        if kind is None:
            self.get_logger().warn(f'/walk_pose ignored: {goal}')
            self._publish_status('HOME_FAILED', label='home',
                                 detail={'reason': 'bad_goal: %s' % goal})
            return
        if kind == 'stop':
            with self._cmd_lock:
                if self._busy:
                    self._pending = ('stop', None)
            return
        if not WALK_POSE:
            self._publish_status('HOME_FAILED', label=goal['label'],
                                 detail={'reason': 'disabled '
                                         '(OPENBOT_WALK_POSE=0)'})
            return
        with self._cmd_lock:
            if self._busy:
                self._pending = ('pose', goal)
                self.get_logger().info('/walk_pose queued (a walk is running).')
                return
            self._busy = True
        threading.Thread(target=self._run_pose, args=(goal,),
                         daemon=True).start()

    def _start_deferred(self):
        """Start the command a preempted run handed over (F7), if any."""
        with self._cmd_lock:
            d, self._deferred = getattr(self, '_deferred', None), None
            if d is None:
                return
            kind, val = d
            if self._busy:
                self.get_logger().warn(
                    f'deferred {kind} dropped: another walk already started.')
                return
            self._busy = True
        target = self._run_pose if kind == 'pose' else self._run_fsm
        threading.Thread(target=target, args=(val,), daemon=True).start()

    def _run_pose(self, goal):
        """B8l F7: walk to the odometry pose ``goal`` (walk_lock.pose_command),
        then face goal face_x/face_y (the requester) or goal yaw.  Statuses:
        RETURNING (1 Hz), then HOME_REACHED or HOME_FAILED{reason}, then IDLE.
        Never walks on stale odometry; stops behind a person / obstacle
        (walk_lock.pose_blocked) and gives up after POSE_BLOCK_HOLD_S."""
        label = goal.get('label', 'home')
        self._fsm_exit_reason = None
        self._pose_running = True
        reason, done = None, None
        face_since = 0.0
        requester_ahead = None
        try:
            self._connect_loco()
            pose = self._odom_pose_age(walk_lock.POSE_ODOM_FRESH_S)
            if pose is None:
                reason = 'odom_stale'
                return
            x_b, y_b = walk_lock.world_to_body(pose, goal['x'], goal['y'])
            dist0 = math.hypot(x_b, y_b)
            if dist0 > walk_lock.POSE_MAX_M:
                reason = 'too_far (%.1f m > %.1f m)' % (dist0, walk_lock.POSE_MAX_M)
                return
            self._publish_status('RETURNING', label=label, detail={
                'distance_m': round(dist0, 3), 'goal': [round(goal['x'], 3),
                                                        round(goal['y'], 3)]})
            self.get_logger().info(
                f'[pose] walking to "{label}" ({goal["x"]:.3f}, {goal["y"]:.3f}) '
                f'from ({pose[0]:.3f}, {pose[1]:.3f}): {dist0:.2f} m')
            r = self._pose_backoff(dist0)          # clear of the table
            if r is not None:
                reason = r
                return
            stage = 'drive'
            deadline = time.time() + min(120.0, 20.0 + dist0 / 0.10)
            stall = hunt_law.StallWatchdog(walk_lock.POSE_STALL_S,
                                           walk_lock.POSE_STALL_M)
            blocked_since, last_hb, stale_since = 0.0, 0.0, 0.0
            silent_since, silent_walk_s, last_tick = 0.0, 0.0, 0.0
            turn_since, turn_yaw0, turn_boost = 0.0, 0.0, False   # stuck mid-turn?
            while True:
                with self._cmd_lock:
                    p, self._pending = self._pending, None
                if p is not None:
                    kind, val = p
                    if kind == 'stop':
                        reason = 'stop'
                        return
                    if kind == 'target':
                        # a /walk_target during the return is the pick side
                        # re-sending its navigate label, not a fresh request;
                        # only a stop or a new pose ends the return trip
                        self.get_logger().warn(
                            f'[pose] /walk_target "{val}" ignored during the '
                            'walk to the start point (stop / a new pose end it).')
                        continue
                    if kind == 'pose':
                        goal, stage = val, 'drive'
                        label = goal.get('label', 'home')
                        stall = hunt_law.StallWatchdog(walk_lock.POSE_STALL_S,
                                                       walk_lock.POSE_STALL_M)
                        self.get_logger().info('[pose] new goal')
                now = time.time()
                pose = self._odom_pose_age(walk_lock.POSE_ODOM_FRESH_S)
                if pose is None:
                    self._stop_motion()
                    if not stale_since:
                        stale_since = now
                    elif now - stale_since > 1.0:
                        reason = 'odom_stale'
                        return
                    time.sleep(CTRL_DT_S)
                    continue
                stale_since = 0.0
                cmd = walk_lock.pose_command(pose, goal, stage)
                tick_prev, last_tick = last_tick, now
                if cmd['stage'] != stage:
                    self.get_logger().info(
                        f'[pose] {stage} -> {cmd["stage"]} at '
                        f'dist={cmd["dist"]:.3f} err={math.degrees(cmd["err"]):+.1f}')
                    if cmd['stage'] == 'face':
                        face_since = now
                stage = cmd['stage']
                if stage == 'done':
                    done = {'err_m': round(cmd['dist'], 3), 'faced': True,
                            'face_err_deg': round(math.degrees(cmd['err']), 1)}
                    return
                if stage == 'face' and face_since \
                        and now - face_since > walk_lock.POSE_FACE_MAX_S:
                    done = {'err_m': round(cmd['dist'], 3), 'faced': False,
                            'face_err_deg': round(math.degrees(cmd['err']), 1)}
                    return
                if stage == 'drive':
                    rows_t, rows = getattr(self, '_body_rows', (0.0, []))
                    # the detector sends nothing for an empty view, so a long
                    # silence may mean a dead detector; walk forward blind only
                    # briefly
                    if now - rows_t > walk_lock.POSE_SILENT_S:
                        if not silent_since:
                            silent_since = now
                            self.get_logger().warn(
                                f'[pose] no detection packet for '
                                f'{now - rows_t:.1f}s (empty view, or the '
                                'detector is down) — slowing to '
                                f'{walk_lock.POSE_VX_MIN:.2f} m/s.')
                        if cmd['vx'] > 0.0 and tick_prev:   # only while walking
                            silent_walk_s += now - tick_prev
                        if silent_walk_s > walk_lock.POSE_SILENT_FAIL_S:
                            reason = 'perception_silent'
                            return
                        if cmd['vx'] > walk_lock.POSE_VX_MIN:
                            cmd['vx'] = walk_lock.POSE_VX_MIN
                    elif silent_since:
                        self.get_logger().info(
                            f'[pose] detections back after {silent_walk_s:.1f}s of '
                            'blind walking — full speed.')
                        silent_since, silent_walk_s = 0.0, 0.0
                    blk = (walk_lock.pose_blocked(rows, cmd['dist'])
                           if now - rows_t <= walk_lock.OBST_ROW_MAX_AGE_S
                           else None)
                    if blk is not None and walk_lock.is_requester_ahead(
                            blk, pose, goal):
                        # if the person in the way asked for the walk, this is
                        # arrival, not a block
                        self.get_logger().info(
                            '[pose] the person who asked is ahead '
                            f'({cmd["dist"]:.2f} m from the start point) — '
                            'stopping here, facing them.')
                        requester_ahead = round(cmd['dist'], 3)
                        stage, face_since = 'face', now
                        self._stop_motion()
                        time.sleep(CTRL_DT_S)
                        continue
                    if blk is not None:
                        if cmd['dist'] <= walk_lock.POSE_NEAR_DONE_M:
                            self.get_logger().info(
                                f'[pose] "{blk.get("label")}" ahead '
                                f'{cmd["dist"]:.2f} m from the goal — close '
                                'enough, facing now.')
                            stage, face_since = 'face', now
                            self._stop_motion()
                            time.sleep(CTRL_DT_S)
                            continue
                        if not blocked_since:
                            blocked_since = now
                            self._stop_motion()
                            self.get_logger().warn(
                                f'[pose] "{blk.get("label")}" in the way at '
                                f'd={math.hypot(float(blk["x"]), float(blk["y"])):.2f} m '
                                '— holding.')
                        if now - blocked_since > walk_lock.POSE_BLOCK_HOLD_S:
                            reason = 'blocked (%s)' % blk.get('label')
                            return
                        time.sleep(CTRL_DT_S)
                        continue
                    if blocked_since:
                        self.get_logger().info('[pose] way clear — resuming.')
                        blocked_since = 0.0
                        stall = hunt_law.StallWatchdog(walk_lock.POSE_STALL_S,
                                                       walk_lock.POSE_STALL_M)
                    if cmd['vx'] <= 0.0:
                        # spinning in place (home behind the robot takes about 10 s
                        # at 0.30 rad/s) is not a stall; the deadline bounds it
                        stall = hunt_law.StallWatchdog(walk_lock.POSE_STALL_S,
                                                       walk_lock.POSE_STALL_M)
                    elif stall.update(cmd['dist'], now) == 'stall':
                        reason = 'no_progress'
                        return
                if now > deadline:
                    reason = 'timeout'
                    return
                # stuck-turn check: a turn command with no rotation
                if cmd['vx'] == 0.0 and cmd['omega'] != 0.0:
                    if not turn_since:
                        turn_since, turn_yaw0 = now, pose[2]
                    elif now - turn_since >= walk_lock.POSE_TURN_STUCK_S:
                        turned = abs(math.degrees(walk_lock.wrap_pi(pose[2] - turn_yaw0)))
                        if turned < walk_lock.POSE_TURN_MIN_DEG:
                            if turn_boost:
                                self.get_logger().warn(
                                    f'[pose] turned only {turned:.0f} deg in '
                                    f'{now - turn_since:.1f}s even at '
                                    f'{walk_lock.POSE_TURN_BOOST:.2f} rad/s — cannot turn here.')
                                reason = 'cannot_turn'
                                return
                            turn_boost = True
                            self.get_logger().warn(
                                f'[pose] turned only {turned:.0f} deg in '
                                f'{now - turn_since:.1f}s — boosting the turn to '
                                f'{walk_lock.POSE_TURN_BOOST:.2f} rad/s.')
                        turn_since, turn_yaw0 = now, pose[2]
                    if turn_boost:
                        cmd['omega'] = math.copysign(walk_lock.POSE_TURN_BOOST, cmd['omega'])
                else:
                    turn_since = 0.0
                self._loco.set_velocity(cmd['vx'], 0.0, cmd['omega'],
                                        duration=1.0)
                if now - last_hb >= 1.0:
                    last_hb = now
                    self._publish_status('RETURNING', label=label, detail={
                        'distance_m': round(cmd['dist'], 3), 'stage': stage})
                    self.get_logger().info(
                        f'[pose] {stage} dist={cmd["dist"]:.3f} '
                        f'err={math.degrees(cmd["err"]):+.1f} vx={cmd["vx"]:.2f} '
                        f'omega={cmd["omega"]:+.2f}')
                time.sleep(CTRL_DT_S)
        except Exception as e:
            self.get_logger().error(f'[pose] crashed: {e!r}')
            reason = 'crash: %s' % e
        finally:
            self._stop_motion()
            if done is not None:
                if requester_ahead is not None:
                    done['requester_ahead_m'] = requester_ahead
                pose = self._odom_pose_age(1.0)
                if pose is not None:
                    done['pose'] = [round(pose[0], 3), round(pose[1], 3),
                                    round(math.degrees(pose[2]), 1)]
                self._publish_status('HOME_REACHED', label=label, detail=done)
                self.get_logger().info(f'[pose] "{label}" reached: {done}')
            else:
                self._publish_status('HOME_FAILED', label=label,
                                     detail={'reason': reason or 'unknown'})
                self.get_logger().warn(f'[pose] "{label}" failed: {reason}')
            self._fsm_exit_reason = ('finished' if done is not None
                                     else ('stop' if reason == 'stop'
                                           else (reason or 'unknown')))
            self._pose_running = False
            with self._cmd_lock:
                self._busy = False
            self._publish_status('IDLE', detail={'exit': self._fsm_exit_reason})
            self._start_deferred()

    def _pose_backoff(self, dist0):
        """B8l-hf4: before the return turns, step BACK
        walk_lock.backoff_distance(OPENBOT_POSE_BACKOFF_M) metres at the gait
        floor (the robot just picked at a table: turning in place there gets it
        stuck, operator 2026-10-17).  Odometry measures the distance; time
        bounds it.  Returns 'stop' / 'odom_stale' to end the walk, else None.
        A /walk_target is ignored here like in the walk itself."""
        want = walk_lock.backoff_distance(POSE_BACKOFF_ENV)
        if want <= 0.0 or dist0 <= walk_lock.POSE_BACKOFF_MIN_DIST_M:
            return None
        p0 = self._odom_pose_age(walk_lock.POSE_ODOM_FRESH_S)
        if p0 is None:
            return 'odom_stale'
        self.get_logger().info(
            f'[pose] stepping back {want:.2f} m at {walk_lock.POSE_BACKOFF_VX:.2f} '
            'm/s before turning (away from the table edge).')
        end = time.time() + want / walk_lock.POSE_BACKOFF_VX + 1.5
        moved = 0.0
        while time.time() < end:
            with self._cmd_lock:
                p = self._pending
                if p is not None and p[0] == 'stop':
                    self._pending = None
            if p is not None and p[0] == 'stop':
                self._stop_motion()
                return 'stop'
            pose = self._odom_pose_age(walk_lock.POSE_ODOM_FRESH_S)
            if pose is None:
                self._stop_motion()
                return 'odom_stale'
            moved = math.hypot(pose[0] - p0[0], pose[1] - p0[1])
            if moved >= want:
                break
            self._loco.set_velocity(-walk_lock.POSE_BACKOFF_VX, 0.0, 0.0,
                                    duration=1.0)
            time.sleep(CTRL_DT_S)
        self._stop_motion()
        self.get_logger().info(f'[pose] stepped back {moved:.2f} m.')
        time.sleep(walk_lock.POSE_BACKOFF_SETTLE_S)
        return None

    def _odom_pose_age(self, max_age_s):
        """(x, y, yaw) when the odometry pose is at most ``max_age_s`` old."""
        olock = getattr(self, '_odom_lock', None)
        if olock is None:
            return None
        with olock:
            t = getattr(self, '_odom_t', 0.0)
            if not t or time.time() - t > max_age_s:
                return None
            return (self._robot_x, self._robot_y, self._robot_yaw)

    def _obstruction_between(self, tx, ty):
        """G17a: the fresh OTHER label standing between robot and target.

        Uses body-frame detections (conf >= MIN_CONF, age <= STALE_DETECTION_S)
        with a position — presence-only labels (no depth) cannot be tested for
        the corridor and are ignored here. Returns {'label', 'od', 'x', 'y'} or
        None.

        `x`/`y` are the obstacle's body-frame position, carried so the caller
        can tell WHICH SIDE of the target bearing it sits on and aim the detour
        at the other one (2026-09-26).  Additive: the label/od consumers are
        unchanged.
        """
        now = time.time()
        best = None
        with self._targets_lock:
            for label, det in self._targets.items():
                if label == self._target_label:
                    continue
                if now - det['t'] > STALE_DETECTION_S:
                    continue
                # another label sitting on the target's spot is the same
                # object, not an obstacle
                if math.hypot(det['x'] - tx, det['y'] - ty) <= SAME_OBJECT_M:
                    continue
                if not hunt_law.obstructs(tx, ty, det['x'], det['y']):
                    continue
                od = math.hypot(det['x'], det['y'])
                if best is None or od < best['od']:
                    best = {'label': str(label), 'od': od,
                            'x': float(det['x']), 'y': float(det['y'])}
        return best

    # -- target helpers ----------------------------------------------------------

    def _label_visible(self, label):
        """True if /detections has seen the label within STALE_DETECTION_S."""
        with self._visible_lock:
            ts = self._visible.get(label)
        if ts is None:
            return False
        return time.time() - ts <= STALE_DETECTION_S

    def _fresh_target(self, label):
        """Latest body-frame target for the label if fresh, else None."""
        with self._targets_lock:
            det = self._targets.get(label)
        if det is None:
            return None
        if time.time() - det['t'] > STALE_DETECTION_S:
            return None
        return det

    @staticmethod
    def _det_stamp(det):
        """The detection's own CAPTURE stamp, or None when the projector did
        not publish one.

        MEASURED on Stuart 2026-09-26 under loadavg 10.6: the detection
        pipeline's latency runs 0.041-0.281 s (p50 0.061, p90 0.209) and its
        publish interval reaches 0.331 s, so two samples can ARRIVE 0.06 s
        apart having been CAPTURED 0.33 s apart.  Anything that reasons about
        how far a stationary object could have moved must use the capture gap;
        receive time understates it by the whole latency jitter.  None means
        "cannot be judged", and callers fall back to receive time — which is
        the old behaviour, so an unstamped projector is no worse off."""
        s = det.get('stamp')
        if s is None:
            return None
        try:
            return float(s)
        except (TypeError, ValueError):
            return None

    def _retained_target(self, label):
        """Latest body-frame measurement retained for the label, no freshness
        gate.  The dict entry survives staleness, so an in-zone geometry can
        still be re-evaluated by the arrival latch (#214) after the projector
        stops emitting the label at close range."""
        with self._targets_lock:
            return self._targets.get(label)

    # -- stop-law and blind-cap helpers ------------------------------------------

    @staticmethod
    def _stop_law(label, x, y, z, width_m, depth_m, stop_distance_m):
        """B8j: the active stop law -- hunt_law.near_edge_law, or the B8i
        hunt_law.stop_law under OPENBOT_STOP_LAW=legacy.  Same signature and
        (in_zone, evidence) return as both."""
        law = hunt_law.stop_law if STOP_LAW_LEGACY else hunt_law.near_edge_law
        return law(label, x, y, z, width_m, depth_m, stop_distance_m)

    @staticmethod
    def _blind_age(fresh, now):
        """B8j: the age (s) of the target row when the blind cap must hold
        the robot, else None (cap off, no row, no receive time, or young)."""
        if not WALK_BLIND_CAP or not isinstance(fresh, dict):
            return None
        t = fresh.get('t')
        if not isinstance(t, (int, float)) or isinstance(t, bool):
            return None
        age = now - t
        return age if age > BLIND_MAX_S else None

    def _log_settled(self):
        """B8j diagnostic (log only): where the robot ENDED, SETTLE_LOG_S
        after REACHED, against the decision -- the coast that LARGE_LEAD_M
        stands for.  Never raises; silent under OPENBOT_STOP_LAW=legacy."""
        if STOP_LAW_LEGACY:
            return
        try:
            ev = dict(self._reach_evidence or {})
            label = self._target_label
            dec = (f'decision d={ev.get("d_m")} d_edge={ev.get("d_edge_m")} '
                   f'tier={ev.get("tier")} gate={ev.get("gate_m")}')
            row = self._fresh_target(label)
            if row is not None and time.time() - row['t'] > BLIND_MAX_S:
                row = None              # a stale row has no useful data
            if row is None:
                seen = ''
                xm, ym = ev.get('x_m'), ev.get('y_m')
                if xm is not None and ym is not None:
                    now = time.time()
                    with self._targets_lock:
                        near = [(math.hypot(v['x'] - xm, v['y'] - ym), k, v)
                                for k, v in self._targets.items()
                                if k != label
                                and now - v['t'] <= STALE_DETECTION_S]
                    near = [c for c in near if c[0] <= hunt_law.CONT_RADIUS_M]
                    if near:
                        _dd, k, v = min(near, key=lambda c: c[0])
                        seen = (f'; seen as "{k}" d='
                                f'{math.hypot(v["x"], v["y"]):.3f}')
                self.get_logger().info(
                    f'[near-edge] settled "{label}": no fresh row{seen} | {dec}')
                return
            _inz, sev = self._stop_law(
                row['label'], row['x'], row['y'], row.get('z', 0.0),
                row.get('width_m'), row.get('depth_m'), STOP_DISTANCE_M)
            coast = None
            if ev.get('d_edge_m') is not None and sev.get('d_edge_m') is not None:
                coast = round(ev['d_edge_m'] - sev['d_edge_m'], 3)
            self.get_logger().info(
                f'[near-edge] settled "{label}": d={sev.get("d_m")} '
                f'd_edge={sev.get("d_edge_m")} half={sev.get("half_m")}'
                f'({sev.get("half_src")}) row_age='
                f'{time.time() - row["t"]:.2f}s'
                + (f' via "{row.get("cont_label")}"' if row.get('cont') else '')
                + f' | {dec} | coast={coast}')
        except Exception as e:          # diagnostics never cost the FSM
            self.get_logger().warn(f'[near-edge] settle log failed: {e!r}')

    def _latch_target(self, det):
        """Freeze the target's world position (body→world via odom yaw).

        B8k (2026-10-16): FRAME FIX.  Detections are body frame x = RIGHT,
        y = FORWARD; the odometry (/unitree/slam_mapping/odom, skills/odom)
        is REP-103: x = forward, y = left, yaw CCW (+omega = LEFT).  The old
        code rotated (right, forward) as if it were (forward, left): harmless
        while the odom never published (yaw 0, pose 0), but with live odom a
        target 1 m ahead read 0.5 m to the LEFT after walking 0.5 m forward
        (tests/unit/walk/test_walk_odom_latch_frame.py).  Convert first:
        forward = y_b, left = -x_b."""
        with self._odom_lock:
            rx, ry, yaw = self._robot_x, self._robot_y, self._robot_yaw
        fwd, left = det['y'], -det['x']
        self._twx = rx + math.cos(yaw) * fwd - math.sin(yaw) * left
        self._twy = ry + math.sin(yaw) * fwd + math.cos(yaw) * left
        self._twz = det['z']
        self._tw_label = det['label']
        self.get_logger().info(
            f'Latched "{self._tw_label}" world=({self._twx:.3f}, {self._twy:.3f})')

