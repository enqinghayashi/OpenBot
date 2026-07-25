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


