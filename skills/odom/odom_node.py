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


def acquire_lock(path):
    """flock the single-instance lock; returns the open file or None if held."""
    fh = open(path, 'a+')
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        fh.close()
        return None
    return fh


class SdkLeg(object):
    """Spawns odom_sdk_leg.py and yields parsed sample dicts."""

    def __init__(self, iface='eth0', script=LEG_SCRIPT):
        self._iface = iface
        self._script = script
        self._proc = None
        self._buf = ''
        self._lines = collections.deque(maxlen=500)

    def start(self, timeout_s=LEG_READY_TIMEOUT_S):
        python = os.environ.get('ODOM_SDK_PYTHON', sys.executable)
        try:
            self._proc = subprocess.Popen(
                [python, self._script, self._iface],
                stdout=subprocess.PIPE, env=build_leg_env(os.environ),
                universal_newlines=True)
        except OSError as exc:
            raise LegDied('cannot spawn SDK leg (%s): %s' % (python, exc))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self._next_line(0.2)
            if line is None:
                continue
            if line.strip() == LEG_READY_LINE:
                return True
        raise LegDied('SDK leg not READY within %.0fs' % timeout_s)

    def _fill(self, timeout_s):
        """Raw os.read on the pipe fd (never the TextIOWrapper, so select()
        and the buffer cannot disagree)."""
        if self._proc.poll() is not None and not self._buf:
            raise LegDied('SDK leg exited rc=%s' % self._proc.returncode)
        fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            return
        chunk = os.read(fd, 65536)
        if not chunk:
            raise LegDied('SDK leg closed stdout')
        self._buf += chunk.decode('utf-8', 'replace')
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            self._lines.append(line)

    def _next_line(self, timeout_s):
        if not self._lines:
            self._fill(timeout_s)
        return self._lines.popleft() if self._lines else None

    def read(self, timeout_s):
        """One parsed sample dict, or None on timeout.  Raises LegDied."""
        line = self._next_line(timeout_s)
        if line is None:
            return None
        try:
            d = json.loads(line)
            if isinstance(d, dict) and 'quat' in d and 'q' in d and 't' in d:
                return d
        except ValueError:
            pass
        return self.read(0.0) if self._lines else None   # skip junk line

    def close(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(2.0)
            except Exception:
                self._proc.kill()


class _RosAdapter(object):
    """rclpy wrapper.  rclpy / nav_msgs are imported HERE, lazily."""

    def __init__(self):
        import rclpy
        from nav_msgs.msg import Odometry
        self._rclpy = rclpy
        self._Odometry = Odometry
        rclpy.init()
        self._node = rclpy.create_node('odom_node')
        self._pub = None

    def ok(self):
        return self._rclpy.ok()

    def spin_once(self, timeout_s):
        self._rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def count_publishers(self, topic):
        return self._node.count_publishers(topic)

