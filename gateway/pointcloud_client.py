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


# ============================================================ pose (pure)
def pose_check(pose: Any) -> Tuple[bool, str]:
    """Copy of ``pc_core.pose_check`` with ``POSE_REQUIRE_MEASURED`` True."""
    if not isinstance(pose, dict):
        return False, "no_pose"
    if pose.get("src") != "measured":
        return False, "pose_unmeasured"
    for k in POSE_KEYS:
        try:
            if not math.isfinite(float(pose[k])):
                return False, "pose_incomplete"
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, "pose_incomplete"
    return True, ""


def pose_equal(a: Any, b: Any) -> Tuple[bool, str]:
    """Same torso frame?  Copy of ``pc_core.pose_equal`` (default tolerances).

    Both poses must be valid MEASURED vectors.  Returns ``(True, "")`` or
    ``(False, reason)`` with reason one of ``no_pose``, ``pose_unmeasured``,
    ``pose_incomplete``, ``pose_changed``.  The reason of ``a`` wins over the
    reason of ``b`` (the same order as the core).
    """
    for p in (a, b):
        ok, why = pose_check(p)
        if not ok:
            return False, why
    if abs(float(a["torso_z"]) - float(b["torso_z"])) > POSE_Z_TOL_M:
        return False, "pose_changed"
    for k in ("pitch", "roll", "waist_pitch"):
        if abs(float(a[k]) - float(b[k])) > POSE_ANG_TOL_RAD:
            return False, "pose_changed"
    return True, ""


# ============================================================ replies (pure)
def phase_of(reply: Any) -> str:
    """The consumer's ``phase`` for an engine reply: tucked | cancelled | error.

    ``tucked`` is claimed ONLY for a reply that starts with the word ``DONE``
    and carries no cancel and no ``ERROR:`` token anywhere; every doubt reads
    ``error`` (a false 'tucked' is the direction this must never err in).

      * ``cancelled``: the reply starts with ``CANCELLED:`` or carries
        ``cancelled_by_operator``.  The substring ``CANCELLED:`` ANYWHERE
        counts too: the gateway has wrapped a cancelled engine reply as
        ``DONE pick executed; ...; engine_reply=CANCELLED:<where>`` (brain/
        verifier.py ``_task_was_cancelled``), a reply that LEADS with DONE.
        The cancel check runs BEFORE the error check because a cancelled pick
        reply leads with ``ERROR:cancelled_by_operator`` (gateway_node
        ``_do_pick``).
      * ``error``: the reply starts with ``ERROR``, carries an ``ERROR:``
        token anywhere (the house rule of ``_shape_reply`` / the verifier's
        ``_classify_reply``), is empty / not a string, or is anything else.
    """
    if not isinstance(reply, str):
        return "error"
    text = reply.strip()
    if not text:
        return "error"
    if "CANCELLED:" in text or "cancelled_by_operator" in text:
        return "cancelled"
    if text.startswith("ERROR") or "ERROR:" in text:
        return "error"
    if _DONE_RE.match(text):
        return "tucked"
    return "error"


def class_label(hint: Any) -> Optional[str]:
    """The CLASS noun phrase the consumer can match, from a gateway-side label.

    The consumer selects a row by ``pc_core.label_matches``: the LAST word of
    the requested label must be one of the words of the row's class label
    ('bottle', 'cell phone').  The brain's raw pick ``label`` is often the
    scene-graph REF ('bottle_1', 'cell_phone_2': gateway_node G4 label-match
    note, MEASURED run3/run4 voice_run.log), and a perception row may carry a
    ``class#<n>`` rank suffix; neither matches a row word.  So a trailing
    ``#<n>`` and a trailing ``_<n>`` are dropped and ``_`` is read as a space
    (the same class-level reading ``gateway_node._b5_hint_class_keys`` applies
    to the same hint); a plain class label passes through unchanged
    (lower-cased).  ``None`` for anything that is not a non-empty string.
    """
    if not isinstance(hint, str):
        return None
    text = _RANK_SUFFIX_RE.sub("", hint.strip().lower())
    text = _REF_SUFFIX_RE.sub("", text)
    text = " ".join(text.replace("_", " ").split())
    return text or None


def supported_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of ``kwargs`` that ``fn`` declares as parameters.

    Everything when ``fn`` takes ``**kwargs``; NOTHING when its signature cannot
    be read.  The gateway uses it to hand the optional point-cloud keywords
    (``pc`` / ``pc_label`` to ``pick_sequence``, ``pc_rim`` to
    ``place_target_from_row``) to a callee only when that callee DECLARES them:
    ``gateway_node.py``, ``descent_lane.py`` and ``place_target.py`` are copied
    to the robot one file at a time, so a switch turned on while one of them is
    still the older file must degrade to the call that file always took, never
    to a TypeError in the middle of a pick.  (The same reasoning as
    ``gateway_node._lane_law``.)
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    keyword_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                     inspect.Parameter.KEYWORD_ONLY)
    return dict((k, v) for k, v in kwargs.items()
                if k in params and params[k].kind in keyword_kinds)


# ============================================================ the client
class _Slot(object):
    """One waiter: its own Event plus the result that wakes it."""

    __slots__ = ("event", "result")

