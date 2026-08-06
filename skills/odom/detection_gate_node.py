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

