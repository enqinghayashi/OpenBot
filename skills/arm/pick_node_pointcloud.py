#!/usr/bin/env python3
"""
pick_node.py — ROS2 node for autonomous pick + arm/hand control.

Topics:
  /pick_target  — Two-step pick flow (label → confirm/cancel)
  /arm_command  — Direct daemon commands (open, close, play, etc.)
  /pick_status  — Status output

Pick flow:
    ros2 topic pub /pick_target std_msgs/msg/String "{data: 'bottle'}" --once
    ros2 topic pub /pick_target std_msgs/msg/String "{data: 'confirm'}" --once
x
Direct commands:
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'open'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'close'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'open_left'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'close_left'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'open_both'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'close_both'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'ready'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'home'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'play wave'}" --once
    ros2 topic pub /arm_command std_msgs/msg/String "{data: 'list'}" --once
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import select
import subprocess
import threading
import time
import os


TV_PYTHON     = '/home/unitree/miniconda3/envs/OpenBot/bin/python3'  # its own python
DAEMON_SCRIPT = '/home/unitree/leon/tabletop_pick_place_engine.py'    # non-owning engine
# engine entrypoint; same
# command body as the gateway
# serves /tmp/pick_engine.sock
# one resident engine, not a
# gateway spawn (that spawn
# dropped the arms to NEUTRAL)
NETWORK_IFACE = 'eth0'

# parse --arm/--upper-only/
# --no-daemon before rclpy
# eats the ROS args
_DAEMON_ARM = 'both'
_DAEMON_UPPER_ONLY = False
_NO_DAEMON = False
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == '--arm' and _i + 1 < len(sys.argv):
        _DAEMON_ARM = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] == '--upper-only':
        # arms-only ctrl
        # no leg writes
        # walk + carry
        _DAEMON_UPPER_ONLY = True
        _i += 1
    elif sys.argv[_i] == '--no-daemon':
        # zero arm_sdk writers at boot
        # so walk right after launch
        # (801 never hit). Later:
        # 'spawn_daemon' → pick →
        # quit → walk again
        _NO_DAEMON = True
        _i += 1
    else:
        _remaining.append(sys.argv[_i])
        _i += 1
sys.argv = [sys.argv[0]] + _remaining

DAEMON_ENV = {
    'HOME':            '/home/unitree',
    'USER':            'unitree',
    'PATH':            (
        '/home/unitree/miniconda3/envs/tv/bin'
        ':/home/unitree/miniconda3/bin'
        ':/usr/local/bin:/usr/bin:/bin'
    ),
    'PYTHONPATH':      '',
    'LD_LIBRARY_PATH': (
        '/home/unitree/miniforge3/envs/pinocchio_env/lib'
        ':/home/unitree/miniconda3/envs/tv/lib'
    ),
    'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
}
# z-floor kill switches only
# reach the engine if set here:
# PERARM=0 → scalar clamp,
# FLOOR=0 → no clamp.
# Unset = nothing added
for _zf_key in ('PICK_ZFLOOR', 'PICK_ZFLOOR_PERARM'):
    if _zf_key in os.environ:
        DAEMON_ENV[_zf_key] = os.environ[_zf_key]

# must equal yolo GATE_CONF
# 0.20: bottle conf flickers
# 0.26-0.40, so at 0.25 the gate
# kept re-entering "confirming"
# (publishes 10-20 s apart;
# so MIN_CONF must match).
# Stricter MIN_CONF drops the
# 0.20-0.25 band. floor_pick_node
# locksteps at 0.20.
# COCO conf; silent on an empty
# scene on purpose
MIN_CONF = 0.20

# ── adaptive tracking constants ──────────────────────────────────────────────
STABILITY_THRESHOLD_M = 0.03    # ~15 px at range
LOST_CONF_THRESHOLD   = 0.4     # softer on purpose
# 3.0 s: yolo
# publishes once per ~1.36 s,
# so 0.3 s flagged 'lost' ~80%
LOST_TIME_THRESHOLD   = 3.0     # "lost" after 3 s
STATUS_FILE           = '/tmp/pick_servo_status'
DETECTION_FILE        = '/tmp/pick_latest_detection.json'

# no pick check
DIRECT_COMMANDS = {
    'open', 'close', 'open_left', 'close_left',
    'open_both', 'close_both', 'home', 'ready', 'list',
    'carry', 'carry_left',   # carry posture
    'neutral',               # hanging rest
    'release', 'acquire', 'hold',  # arm_sdk gate
                                   # needs --upper-only;
                                   # ERROR otherwise.
                                   # we forward;
                                   # daemon validates
    'drop',                        # drop sequence
                                   # LOW_CARRY → hand open →
                                   # payload off → NEUTRAL →
                                   # hold_rc off, then DONE
    'quit',                        # graceful exit
                                   # send after carry_tuck DONE,
                                   # before walking. Prints DONE,
                                   # frees the gate, exits;
                                   # daemon_alive false, 5 Hz
    'spawn_daemon',                # spawn on demand
                                   # after --no-daemon
                                   # guarded: only if dead
                                   # (proc None or poll() != None)
                                   # ERROR if alive
                                   # reuses _start_daemon
                                   # never sent onward
}


def is_reachable(x, y, z):
    """Return True if target (torso-frame coordinates) is within arm workspace."""
    if not (0.15 <= y <= 0.65):
        return False
    if abs(x) >= 0.45:
        return False
    if not (-0.35 <= z <= 0.45):
        return False
    return True


class PickNode(Node):
    def __init__(self):
        super().__init__('pick_node')

        self._latest_detections = {}
        self._busy = False
        self._daemon_proc = None
        self._daemon_lock = threading.Lock()

        self._pending_pick = None
        self._pending_lock = threading.Lock()

        # ── pick state for walk coordinator ──────────────────────────────────
        # state: idle/busy/holding/error
        # label: last pick
        # held: in hand?
        # alive: proc health
        # JSON on /pick_state, 5 Hz
        self._pick_state = 'idle'
        self._pick_label = None
        self._pick_held = False
        self._pick_state_lock = threading.Lock()

        self.create_subscription(
            String, '/object_detections_body', self._on_detections, 10)
        self.create_subscription(
            String, '/pick_target', self._on_pick_target, 10)
        self.create_subscription(
            String, '/arm_command', self._on_arm_command, 10)

        self._status_pub = self.create_publisher(String, '/pick_status', 10)
        self._pick_state_pub = self.create_publisher(
            String, '/pick_state', 10)

        # ── tracking stability state ─────────────────────────────────────────
        self._prev_tx = None
        self._prev_ty = None
        self._prev_tz = None
        self._last_detection_time = None
        self._tracking_label = None

        # wipe stale side files
        # (ghost guard)
        for _p in (DETECTION_FILE, STATUS_FILE):
            try:
                os.remove(_p)
            except OSError:
                pass

        if _NO_DAEMON:
            self.get_logger().info(
                '--no-daemon: starting with ZERO arm_sdk writers — '
                'robot can walk immediately. Spawn the pick daemon on '
                'demand via /arm_command "spawn_daemon".')
        else:
            self._start_daemon()

        self.get_logger().info(
            'Pick node ready. Publish label to /pick_target, '
            'then "confirm" to execute. Direct commands on /arm_command.')

        # ── tracking-status writer thread ────────────────────────────────────
        threading.Thread(target=self._write_servo_status, daemon=True).start()

        # ── /pick_state publisher, 5 Hz ──────────────────────────────────
        threading.Thread(target=self._write_pick_state, daemon=True).start()

    # ── Looking after the daemon ────────────────────────────────────────────────

    def _start_daemon(self):
        """Spawn the pick daemon and wait for READY.

        T11: returns True ONLY if the daemon printed READY within the
        60 s deadline and the process is still alive; False on ANY
        failure — READY timeout, the process dying during startup
        (poll() != None, e.g. the corrupted-transport null-byte import
        crash), or a spawn exception.  Failure paths log the daemon's
        last stderr lines (see _log_last_stderr) so the pick_node log
        carries the diagnosis.  Callers reply DONE only on True.
        """
        # clear posture json
        # or restore_arm_pose.py
        # revives an old pose
        try:
            os.remove('/tmp/last_arm_q.json')
            self.get_logger().info('Cleared stale /tmp/last_arm_q.json')
        except FileNotFoundError:
            pass
        self.get_logger().info(
            'Starting pick_daemon.py (loading pinocchio — takes ~10s)...')
        # refuse if ANY daemon alive
        # (orphan = second writer,
        # arm ping-pongs). Only our
        # child tracked; [p] keeps
        # pgrep off itself
        try:
            _chk = subprocess.run(['pgrep', '-f',
                                     '[t]abletop_pick_place_engine.py'
                                     '|[p]ick_engine_v[23].py'],
                                  capture_output=True, text=True, timeout=5)
            if _chk.returncode == 0:
                self.get_logger().error(
                    'REFUSING to spawn: pick_daemon already alive (double '
                    'arm_sdk writer). Run: pkill -f pick_daemon')
                return False
        except Exception:
            pass
        try:
            daemon_args = [TV_PYTHON, DAEMON_SCRIPT, NETWORK_IFACE]
            # argv[2]: arm selector
            # always pass it
            # no 'upper' flag: stock
            # G1_29 controller;
            # no UpperOnly subclass
            daemon_args.append(
                _DAEMON_ARM if _DAEMON_ARM in ('left', 'right') else 'both')
            self._daemon_proc = subprocess.Popen(
                daemon_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=DAEMON_ENV,
            )
            threading.Thread(
                target=self._log_daemon_stderr, daemon=True).start()

            deadline = time.time() + 60.0
            while time.time() < deadline:
                # dead proc: bail out
                # readline '' = EOF:
                # dump stderr, skip 60 s
                if self._daemon_proc.poll() is not None:
                    self._log_last_stderr()
                    self.get_logger().error(
                        'Daemon process died during startup '
                        f'(exit code {self._daemon_proc.returncode}) — '
                        'crash at startup, NOT ready')
                    return False
                line = self._daemon_proc.stdout.readline().strip()
                if not line:
                    continue
                if line == 'READY':
                    self.get_logger().info('Pick daemon ready.')
                    # DDS may fork at init,
                    # leaving a second rt/arm_sdk
                    # writer; arm oscillates
                    # kill all but ours
                    _mine_pid = self._daemon_proc.pid
                    def _watchdog(_mine_pid=_mine_pid):
                        import signal as _sig
                        while True:
                            time.sleep(2.0)
                            if (self._daemon_proc is None
                                    or self._daemon_proc.poll() is not None):
                                return
                            try:
                                _p = subprocess.run(
                                    ['pgrep', '-f',
                                     '[t]abletop_pick_place_engine.py'
                                     '|[p]ick_engine_v[23].py'],
                                    capture_output=True, text=True, timeout=5)
                                for _pid in [int(x) for x in _p.stdout.split()
                                             if x.strip().isdigit()]:
                                    if _pid != _mine_pid:
                                        os.kill(_pid, _sig.SIGKILL)
                                        self.get_logger().warn(
                                            'killed forked/extra daemon pid '
                                            f'{_pid}')
                            except Exception:
                                pass
                    threading.Thread(target=_watchdog, daemon=True).start()
                    return True
                self.get_logger().warn(
                    f'Daemon stray output during init (skipped): "{line}"')
            # no READY in 60 s
            self._log_last_stderr()
            self.get_logger().error(
                'Daemon failed to send READY within 60s — '
                'startup failed, NOT ready')
            return False

        except Exception as e:
            self.get_logger().error(f'Failed to start daemon: {e}')
            return False

    def _log_daemon_stderr(self):
        for line in self._daemon_proc.stderr:
            self.get_logger().info(f'[daemon] {line.strip()}')

    def _log_last_stderr(self):
        """T11: surface the daemon's last stderr lines in the pick_node log.

        The streaming _log_daemon_stderr thread usually forwards every
        line as it arrives, but a fast crash can outrun it.  This drains
        whatever is still buffered (bounded — never blocks on a live but
        silent daemon) and logs it as errors so the crash cause (e.g.
        "ValueError: source code string cannot contain null bytes") is
        in the pick_node log.
        """
        try:
            fd = self._daemon_proc.stderr.fileno()
        except Exception:
            return
        deadline = time.time() + 0.5
        try:
            while time.time() < deadline:
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    break        # thread drained it
                line = self._daemon_proc.stderr.readline()
                if not line:
                    break        # EOF, fully drained
                self.get_logger().error(f'[daemon stderr] {line.strip()}')
        except Exception:
            pass

    def _send_cmd(self, cmd: str, timeout: float = 30.0) -> str:
        with self._daemon_lock:
            # dead daemon: skip wait;
            # a stale _busy=True would
            # lock later commands out
            if self._daemon_proc is None or self._daemon_proc.poll() is not None:
                self._busy = False
                return 'ERROR:daemon not running'
            try:
                self._daemon_proc.stdin.write(cmd + '\n')
                self._daemon_proc.stdin.flush()

                deadline = time.time() + timeout
                while time.time() < deadline:
                    line = self._daemon_proc.stdout.readline()
                    if not line:
                        # '' = daemon dead
                        if self._daemon_proc.poll() is not None:
                            self._busy = False
                            return 'ERROR:daemon died'
                        continue
                    line = line.strip()
                    if line.startswith('DONE') or line.startswith('ERROR:'):
                        return line
                    self.get_logger().warn(
                        f'Daemon stray output (skipped): "{line}"')
                return 'ERROR:timeout waiting for daemon response'
            except BrokenPipeError:
                self._busy = False
                return 'ERROR:daemon pipe broken'
            except Exception as e:
                return f'ERROR:{e}'

    # ── What ROS calls us with ──────────────────────────────────────────────────

    def _on_detections(self, msg):
        try:
            detections = json.loads(msg.data)
            _now = time.time()
            self._latest_detections = {
                d['label']: {**d, '_t': _now} for d in detections
                if d.get('conf', 0) >= MIN_CONF
            }

            # ── refresh tracking off best detection ────────────────────────
            if detections:
                best = max(detections, key=lambda d: d.get('conf', 0))
                conf = best.get('conf', 0)
                if conf >= LOST_CONF_THRESHOLD:
                    self._prev_tx = best.get('x')
                    self._prev_ty = best.get('y')
                    self._prev_tz = best.get('z')
                    self._tracking_label = best.get('label')
                    self._last_detection_time = time.time()

                    # for "shifted"
                    _det_data = {
                        'label': self._tracking_label,
                        'x': self._prev_tx,
                        'y': self._prev_ty,
                        'z': self._prev_tz,
                        'conf': conf,
                    }
                    if 'orientation' in best:
                        _det_data['orientation'] = best['orientation']
                    if 'nx' in best:
                        _det_data['nx'] = best['nx']
                        _det_data['ny'] = best['ny']
                        _det_data['nz'] = best['nz']
                    if 'height_m' in best:
                        _det_data['height_m'] = best['height_m']
                    if 'x_off' in best:
                        _det_data['x_off'] = best['x_off']
                    try:
                        with open(DETECTION_FILE, 'w') as f:
                            json.dump(_det_data, f)
                    except Exception:
                        pass
        except Exception as e:
            self.get_logger().warn(f'Bad detection msg: {e}')

    def _write_servo_status(self):
        """Phase 4: write tracking stability status to /tmp/pick_servo_status.

        Runs as a daemon thread. Status values:
          "stable"  — centroid shift < STABILITY_THRESHOLD_M
          "shifted" — centroid shift >= STABILITY_THRESHOLD_M
          "lost"    — no recent detection or confidence too low
        """
        _last_tx = None
        _last_ty = None

        while True:
            try:
                now = time.time()
                cur_tx = self._prev_tx
                cur_ty = self._prev_ty

                if (self._last_detection_time is None
                        or (now - self._last_detection_time) > LOST_TIME_THRESHOLD):
                    status = 'lost'
                elif cur_tx is None or cur_ty is None:
                    status = 'lost'
                elif _last_tx is not None and _last_ty is not None:
                    dx = cur_tx - _last_tx
                    dy = cur_ty - _last_ty
                    if dx * dx + dy * dy >= STABILITY_THRESHOLD_M * STABILITY_THRESHOLD_M:
                        status = 'shifted'
                    else:
                        status = 'stable'
                else:
                    status = 'stable'

                _last_tx = cur_tx
                _last_ty = cur_ty

                with open(STATUS_FILE, 'w') as f:
                    f.write(status + '\n')
            except Exception:
                pass
            time.sleep(0.1)

    # ── /pick_state JSON at 5 Hz ───────────────────────────────────────────────
    def _set_pick_state(self, state, label=None, held=None):
        """Update the pick state machine (thread-safe)."""
        with self._pick_state_lock:
            self._pick_state = state
            if label is not None:
                self._pick_label = label
            if held is not None:
                self._pick_held = held

    def _pick_state_snapshot(self):
        with self._pick_state_lock:
            return (self._pick_state, self._pick_label, self._pick_held)

    def _write_pick_state(self):
        """Publish {state, label, held, daemon_alive} JSON at 5 Hz."""
        while True:
            try:
                state, label, held = self._pick_state_snapshot()
                daemon_alive = (
                    self._daemon_proc is not None
                    and self._daemon_proc.poll() is None)
                payload = json.dumps({
                    'state': state,
                    'label': label,
                    'held': held,
                    'daemon_alive': daemon_alive,
                })
                msg = String()
                msg.data = payload
                self._pick_state_pub.publish(msg)
            except Exception:
                pass
            time.sleep(0.2)

