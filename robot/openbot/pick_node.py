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


TV_PYTHON     = '/home/unitree/miniconda3/envs/tv/bin/python3'
DAEMON_SCRIPT = '/home/unitree/leon/tabletop_pick_place_engine.py'
NETWORK_IFACE = 'eth0'

# our flags, before rclpy eats argv
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
        # arms only: legs/waist untouched
        # so we can walk while holding
        _DAEMON_UPPER_ONLY = True
        _i += 1
    elif sys.argv[_i] == '--no-daemon':
        # no arm_sdk writers; never hits 801
        # walk right away
        # daemon on demand: 'spawn_daemon'
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
# z-floor kill switches
# reach the engine only if
# this node's env sets them
# PICK_ZFLOOR_PERARM=0: scalar clamp
# PICK_ZFLOOR=0: no clamp at all
# unset = spawn env untouched
for _zf_key in ('PICK_ZFLOOR', 'PICK_ZFLOOR_PERARM'):
    if _zf_key in os.environ:
        DAEMON_ENV[_zf_key] = os.environ[_zf_key]

# lockstep with yolo GATE_CONF
# bottle conf flickers 0.26-0.40
# at 0.25 the gate kept
# re-entering "confirming",
# publishes 10-20 s apart
# GATE_CONF is 0.20, so MIN_CONF is too;
# a tighter MIN_CONF (>=0.25)
# drops the 0.20-0.25 band
# floor_pick_node locksteps too
# conf = COCO detection
# empty-scene silence on purpose
# (COCO has real negatives)
MIN_CONF = 0.20

# safety whitelist
# gates ingestion + tracking
# yolo: 16 classes, these only
# an arm in frame can be
# read as person 0.25
# person: never pickable
PICKABLE_CLASSES = {
    'bottle', 'cup', 'bowl', 'wine glass', 'vase', 'remote', 'cell phone',
}

# ── tracking stability knobs ─────────────────
STABILITY_THRESHOLD_M = 0.03    # ~15 px here
LOST_CONF_THRESHOLD   = 0.4     # looser than gate
LOST_TIME_THRESHOLD   = 3.0     # timeout → "lost"
                                # 3 s spans 2-3 body msgs: SEG
                                # yolo sends one every ~1.1 s on
                                # CPU, so a shorter one marks it lost
# human confirm takes 3-6 s
# old 3.0 s gate: no
# keyed on det['_t']
# (per body msg, conf>=MIN_CONF)
# not _last_detection_time (0.4):
# the 0.4 gate starves on the
# bottle's 0.26-0.40 flicker
# while it's still in view
# 15 s clears the window,
# still refuses a gone target
CONFIRM_FRESHNESS_SEC = 15.0

STATUS_FILE           = '/tmp/pick_servo_status'
DETECTION_FILE        = '/tmp/pick_latest_detection.json'

# straight to the daemon
DIRECT_COMMANDS = {
    'open', 'close', 'open_left', 'close_left',
    'open_both', 'close_both', 'home', 'ready', 'list',
    'carry', 'carry_left',   # carry posture
    'neutral',               # arms hanging
    'release', 'acquire', 'hold',  # arm_sdk gate
                                   # ERROR unless --upper-only
                                   # forwarded anyway;
                                   # daemon validates
    'quit',                        # graceful exit
                                   # after carry_tuck DONE,
                                   # before walking
                                   # prints DONE, unlocks
                                   # daemon_alive false
    'spawn_daemon',                # on demand
                                   # after a --no-daemon launch
                                   # dead only (proc None or
                                   # poll() != None), else ERROR
                                   # reuses _start_daemon
                                   # node handles it, not the daemon
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

        # ── pick state: walk coordinator feed ────────
        # state: idle|busy|holding|error
        # label: last picked
        # held: in hand?
        # daemon health
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

        # ── tracking stability state ─────────────────
        self._prev_tx = None
        self._prev_ty = None
        self._prev_tz = None
        self._last_detection_time = None
        self._tracking_label = None
        # warn once only
        self._whitelist_warned = set()

        # wipe side-channel files
        # from old runs, so no
        # stale re-target
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

        # ── servo status writer ──────────────────────
        threading.Thread(target=self._write_servo_status, daemon=True).start()

        # ── /pick_state thread, 5 Hz ─────────────────
        threading.Thread(target=self._write_pick_state, daemon=True).start()

    # ── daemon wrangling ─────────────────────────

