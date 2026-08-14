#!/usr/bin/env python3
"""detection_gate_node.py -- OPT-IN spatial gate relay (default OFF).

Subscribes /object_detections_body and /walk_target, publishes
/object_detections_body_gated.  It touches NO other topic.  Starting this
relay alone changes nothing: only a walk_node restarted with
  --ros-args -r /object_detections_body:=/object_detections_body_gated
reads the gated topic.  Relay dead => walk_node sees silence => its existing
lost-target path (same as a detector blink); nothing new is commanded.

Exit codes: 0 clean | 3 REFUSE (gated topic already has a publisher) |
6 single-instance lock held (/tmp/odom_gate.lock).

rclpy is imported lazily inside _RosAdapter.  Nothing here has run on the
robot (UNVERIFIABLE-FROM-HERE).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import odom_core as oc  # noqa: E402
import gate_core as gc  # noqa: E402
from odom_node import acquire_lock, log, _Clock  # noqa: E402

RC_OK = 0
RC_REFUSED = 3
RC_LOCK = 6

BODY_TOPIC = '/object_detections_body'
TARGET_TOPIC = '/walk_target'
GATED_TOPIC = '/object_detections_body_gated'
LOCK_PATH = '/tmp/odom_gate.lock'
SETTLE_S = 2.0
LOG_EVERY_S = 2.0


def load_ref_resolver():
    """Same try-import fallback as walk_node.py:119-121.  Returns (fn, note)."""
    try:
        from instance_labels import ref_to_label            # box layout
        return ref_to_label, 'instance_labels (box layout)'
    except ImportError:
        pass
    try:
        from skills.perception.instance_labels import ref_to_label   # repo layout
        return ref_to_label, 'skills.perception.instance_labels (repo layout)'
    except ImportError:
        return (lambda s: s), ('instance_labels NOT importable -- using the '
                               'walk target label verbatim')


class _RosAdapter(object):
    def __init__(self):
        import rclpy
        from std_msgs.msg import String
        self._rclpy = rclpy
        self._String = String
        rclpy.init()
        self._node = rclpy.create_node('detection_gate')
        self._pub = None

    def ok(self):
        return self._rclpy.ok()

    def spin_once(self, timeout_s):
        self._rclpy.spin_once(self._node, timeout_sec=timeout_s)

    def count_publishers(self, topic):
        return self._node.count_publishers(topic)

    def make_publisher(self):
        self._pub = self._node.create_publisher(self._String, GATED_TOPIC, 10)

    def subscribe(self, on_target, on_detections):
        self._node.create_subscription(
            self._String, TARGET_TOPIC, lambda m: on_target(m.data), 10)
        self._node.create_subscription(
            self._String, BODY_TOPIC, lambda m: on_detections(m.data), 10)

    def publish(self, text):
        msg = self._String()
        msg.data = text
        self._pub.publish(msg)

    def shutdown(self):
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


def run(ros, clock, args, resolver=None):
    lock = acquire_lock(args.lock_path)
    if lock is None:
        log('LOCK: %s is held by another detection_gate -- exiting' % args.lock_path)
        return RC_LOCK
    try:
        t0 = clock.now()
        while clock.now() - t0 < args.settle_s:
            ros.spin_once(0.1)
        n = ros.count_publishers(GATED_TOPIC)
        if oc.decide_startup(n) == 'REFUSE':
            log('REFUSE: %d publisher(s) already on %s -- not publishing'
                % (n, GATED_TOPIC))
            return RC_REFUSED
        ros.make_publisher()
        if resolver is None:
            resolver, note = load_ref_resolver()
            log('gate: ref resolver = %s' % note)
        gate = gc.TargetGate(noise_floor_m=args.noise_floor_m)
        core = gc.RelayCore(gate, resolver, passthrough=args.passthrough)
        state = {'last_log': clock.now()}

        def on_target(text):
            core.on_target(text)

        def on_det(raw):
            now = clock.now()
            out = core.on_detections(raw, now)
            if out is not None:
                ros.publish(out)
            if core.label is not None and now - state['last_log'] >= LOG_EVERY_S:
                state['last_log'] = now
                log('gate: label=%s accepted=%d dropped=%d R=%.3f' % (
                    core.label, gate.accepted, gate.dropped, gate.last_radius))

        ros.subscribe(on_target, on_det)
        while ros.ok():
            ros.spin_once(0.1)
        return RC_OK
    finally:
        lock.close()


