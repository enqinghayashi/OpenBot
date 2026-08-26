#!/usr/bin/env python3
"""pc_consumer_node: the on-demand point-cloud consumer (batch 3).

DISARMED BY DEFAULT.  While disarmed the node holds exactly one subscription,
/pc_arm (std_msgs/String JSON).  It never subscribes to /object_pointcloud or
/object_detections_body unless armed (or 'warm'), and it destroys both
subscriptions when the task finishes or its TTL (a dead-man) expires.

Request (/pc_arm, JSON):
  {"rid": str, "task": hold_baseline|hold_check|place_rim|split|grasp|person
                       |warm|disarm,
   "label": str, "ref": [x, y]  (MANDATORY for hold_baseline),
   "pose": {"src": "measured", "torso_z", "pitch", "roll", "waist_pitch", "t"},
   "ttl_s": float,
   hold_check only: "baseline_rid": str, "phase": tucked|cancelled|error|other,
                    "t_reply_wall": float (engine reply wall time)}
Result (/pc_result, JSON): exactly ONE envelope per armed request
  (pc_core.envelope), including skip counters and the node's own CPU / RSS.

All logic is in ConsumerCore (no ROS import) so tests drive it directly.  The
rclpy glue is in main() and imports rclpy lazily: without ROS the script exits
with rc 2 and a one-line message, it does not traceback.

Python 3.8-safe.  stdlib + numpy (via pc_core).
"""
import collections
import json
import math
import os
import sys
import time

try:  # normal package import
    from . import pc_core as core
except ImportError:  # flat import: python3 pc_consumer_node.py
    import pc_core as core

TOPIC_ARM = "/pc_arm"
TOPIC_RESULT = "/pc_result"
TOPIC_CLOUD = "/object_pointcloud"
TOPIC_BODY = "/object_detections_body"


