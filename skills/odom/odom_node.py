#!/usr/bin/env python3
"""odom_node.py -- ROS-only wrapper publishing leg-FK + IMU odometry.

Publishes nav_msgs/Odometry on /unitree/slam_mapping/odom, the topic that
walk_node.py (:763) and the depth projector (:175-181) already consume and that
NOTHING on Stuart publishes (walk.log `age=-1.0s`).

TWO PROCESSES by design: this process is rclpy-only and NEVER imports
unitree_sdk2py; the lowstate source is odom_sdk_leg.py (SDK-only), spawned as a
subprocess exactly like walk_node._DaemonLoco.connect (walk_node.py:471-481).
Module top level imports only stdlib + odom_core; rclpy / nav_msgs are imported
lazily inside _RosAdapter.  Publishes ONLY odometry: no actuation of any kind.

Exit codes: 0 clean SIGINT/SIGTERM | 3 REFUSE (foreign publisher present) |
4 YIELD (another publisher appeared) | 5 SDK leg failed / not READY |
6 single-instance lock held.

Status: nothing here has run on the robot (UNVERIFIABLE-FROM-HERE).
"""

import argparse
import collections
import fcntl
import json
import os
import select
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import odom_core as oc  # noqa: E402

RC_OK = 0
RC_REFUSED = 3
RC_YIELDED = 4
RC_LEG = 5
RC_LOCK = 6

LOCK_PATH = '/tmp/odom_node.lock'
DISCOVERY_SETTLE_S = 2.0       # inferred: DDS discovery
RECHECK_S = 5.0
LOG_EVERY_S = 5.0
LEG_READY_TIMEOUT_S = 15.0
LEG_READY_LINE = 'SDK_LEG_READY'
LEG_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'odom_sdk_leg.py')


class LegDied(Exception):
    pass


def log(msg):
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()


def build_leg_env(environ):
    """Scrubbed env for the SDK leg (mirrors walk_node.py:473): the leg inserts
    its own SDK path, so PYTHONPATH must be empty."""
    env = dict(environ)
    env['PYTHONPATH'] = ''
    return env


