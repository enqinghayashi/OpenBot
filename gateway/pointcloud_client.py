"""pc_client.py - the gateway's client for the point-cloud consumer node.

Batch 3b, hook H-GW-0 (docs/POINTCLOUD_HOOKS_SPEC_batch3b.md section 4).  The
consumer (skills/perception/pointcloud/pc_consumer_node.py) is a separate,
hand-started, DISARMED-by-default node.  It answers a request published on
``/pc_arm`` with exactly one envelope on ``/pc_result`` (both std_msgs/String
JSON).  This module is the gateway's side of that wire and nothing else.

WHAT IT IS NOT.  It is not a ROS module.  Every ROS object is INJECTED
(``publish_arm`` is a plain ``callable(str)``), so the whole thing is unit
testable without rclpy, and with every operator switch OFF the gateway never
imports it (gateway/gateway_node.py builds a :class:`PcClient` only when one of
``embodiment.envelope.pc_shadow`` / ``pc_hold_check`` / ``pc_place_rim`` is a
real ``True``).  stdlib ONLY: no rclpy, no numpy, no brain/ import.

THE CONTRACT THIS CLIENT KEEPS
  * :meth:`PcClient.request` NEVER raises and NEVER blocks past its deadline.
    A silent consumer, a publish error, a malformed result - every one of them
    returns an ``{"status": "UNKNOWN", ...}`` dict, and UNKNOWN changes
    nothing downstream.
  * :meth:`PcClient.on_result` is called from the ROS subscription thread.  It
    ignores malformed JSON, non-dict JSON and results for rids nobody is
    waiting for; it never raises.
  * Thread model: ONE lock guards the pending table, and each waiter blocks on
    its OWN per-rid ``threading.Event`` (a lane / tool thread calls
    ``request``; the executor thread calls ``on_result``).  No global state.
  * The pose helpers (:func:`pose_check`, :func:`pose_equal`) are a COPY of
    ``pc_core.pose_check`` / ``pc_core.pose_equal`` (src must be "measured",
    all four POSE_KEYS finite, |dz| <= 1 cm, |d angle| <= 1 degree).  The two
    tolerances are literals here because the gateway cannot import numpy-backed
    ``pc_core``; ``tests/test_pc3b_client.py`` pins them EQUAL to
    ``pc_config.DEFAULT`` and pins agreement with ``pc_core.pose_equal`` on
    random pose pairs.
  * Nothing here ever asserts that an object is HELD.  The consumer's hold
    verdicts are negative-only (pc_core.HOLD_VERDICTS); this client only
    carries them.

Python 3.8-safe (the robot's interpreter).
"""
import inspect
import json
import math
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

#: wire topics; a test pins them == pc_consumer_node
TOPIC_ARM = "/pc_arm"
TOPIC_RESULT = "/pc_result"

#: copy of pc_core.POSE_KEYS; test-pinned
POSE_KEYS = ("torso_z", "pitch", "roll", "waist_pitch")
#: odometry pose, BodyPoseFilter 10 Hz (0.5 s median)
#: usable only this fresh. consumer kills >1.0 s
#: (pc_config POSE_MAX_AGE_S); 0.6 s leaves trip room
POSE_SOURCE_MAX_AGE_S = 0.6
POSE_SOURCE_FUTURE_S = 0.2
#: pc_config.DEFAULT tolerances, copied:
#: 1 cm torso height, 1 deg (0.0175 rad)
POSE_Z_TOL_M = 0.01
POSE_ANG_TOL_RAD = 0.0175

#: place_rim budget: ttl clamped to [0.5, 15] s, 3 frames
#: at <=5 Hz -> ~1 s. ttl sits under the deadline so a
#: stuck capture is a consumer-side UNKNOWN, not ours
PLACE_RIM_DEADLINE_S = 4.0
PLACE_RIM_TTL_S = 3.5
#: rim fit only on rows this fresh
PLACE_RIM_ROW_MAX_AGE_S = 1.0
#: future stamp = clock jump, not fresh
_FUTURE_STAMP_SLACK_S = 0.5

#: clock re-read; a result wake-up is instant
#: this only bounds deadline lateness
_WAIT_SLICE_S = 0.05

_DONE_RE = re.compile(r"DONE(?![A-Za-z0-9_])")
_REF_SUFFIX_RE = re.compile(r"_\d+$")
_RANK_SUFFIX_RE = re.compile(r"#\d+$")


def _json_default(obj: Any) -> Any:
    """numpy scalars / arrays -> plain python (stdlib only: duck-typed)."""
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError("not JSON serialisable: %s" % type(obj).__name__)
