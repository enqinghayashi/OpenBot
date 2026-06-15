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

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result = None


class PcClient(object):
    """Publish a request to the consumer, wait (bounded) for its one result.

    ``publish_arm(text)``  publishes one /pc_arm String (injected; may raise).
    ``clock_wall()``       wall clock (injected; exposed as ``self.clock_wall``).
    ``clock_mono()``       monotonic clock; the request deadline reads it.
    ``writer_verb(line)``  the gateway's writer-socket call (``None`` = no
                           pose source: :meth:`measured_pose` answers None).
    ``pose_source()``      B8k: the latest MEASURED body pose from the odometry
                           process (``/openbot/body_pose``), or None.  Used
                           first when it is measured, complete and fresh
                           (:data:`POSE_SOURCE_MAX_AGE_S`); the writer verb is
                           the fallback.
    ``decisive``           True only with ``pc_hold_check``: the lane may let
                           a negative hold verdict downgrade its pick verdict.
    ``log(message)``       optional ``callable(str)``; never allowed to raise
                           into a caller.

    ``t_reply_wall`` is set by the lane (the wall time of the engine reply) and
    carried into a hold_check request.
    """

    def __init__(self, publish_arm: Callable[[str], Any],
                 clock_wall: Callable[[], float],
                 clock_mono: Callable[[], float],
                 writer_verb: Optional[Callable[[str], str]] = None,
                 decisive: bool = False,
                 log: Optional[Callable[[str], Any]] = None,
                 pose_source: Optional[Callable[[], Any]] = None) -> None:
        if not callable(publish_arm):
            raise TypeError("publish_arm must be callable")
        self._publish_arm = publish_arm
        self.clock_wall = clock_wall
        self.clock_mono = clock_mono
        self._writer_verb = writer_verb
        self._pose_source = pose_source
        self._log = log
        self.decisive = bool(decisive)
        self.t_reply_wall = None
        self._lock = threading.Lock()
        self._pending = {}          # rid -> _Slot
        self._seq = 0
        self._tag = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------ helpers
    def _say(self, message: str) -> None:
        log = self._log
        if log is None:
            return
        try:
            log(message)
        except Exception:  # noqa: BLE001 - a log hook must never take out a caller
            pass

    def new_rid(self, prefix: str) -> str:
        """A unique, non-empty request id: ``<prefix>-<run tag>-<n>``."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        head = str(prefix) if prefix else "pc"
        return "%s-%s-%d" % (head, self._tag, seq)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ------------------------------------------------------------ request
    def request(self, req: Dict[str, Any], deadline_s: float) -> Dict[str, Any]:
        """Publish ``req`` and block THIS thread for its result, or the deadline.

        Adds ``rid`` when missing.  The request goes out as given, except that
        a string ``label`` is put through :func:`class_label` first (the wire
        adapter: the consumer matches class nouns, the gateway holds the brain's
        refs - 'bottle_1' goes out as 'bottle'; a plain class label is
        unchanged, and the caller's dict is never modified).  The pending entry
        is registered BEFORE the publish (a result that arrives synchronously
        must find it) and removed on EVERY path.  Returns the consumer's
        envelope dict, or ``{"status": "UNKNOWN", "reason": "deadline", "rid":
        <rid>}`` when no matching result arrived in ``deadline_s`` seconds of
        the monotonic clock (an extra real-time cap stops a frozen injected
        clock from blocking forever), or ``{"status": "UNKNOWN", "reason":
        "client_error: <Type>: <msg>"}`` (plus ``rid`` once known) when
        anything inside raised.  NEVER raises.
        """
        rid = None
        slot = None
        try:
            if not isinstance(req, dict):
                raise TypeError("request must be a dict, got %s" % type(req).__name__)
            wait_s = float(deadline_s)
            if not math.isfinite(wait_s):
                raise ValueError("deadline_s must be finite, got %r" % (deadline_s,))
            wait_s = max(0.0, wait_s)
            body = dict(req)
            label = body.get("label")
            if isinstance(label, str):
                wire_label = class_label(label)
                if wire_label is not None:
                    body["label"] = wire_label
            rid = body.get("rid")
            if not isinstance(rid, str) or not rid:
                rid = self.new_rid(str(body.get("task") or "pc"))
                body["rid"] = rid
            slot = _Slot()
            with self._lock:
                if rid in self._pending:
                    slot = None         # not ours: finally mustn't drop
                    raise ValueError("a request with rid %r is already pending" % rid)
                self._pending[rid] = slot
            self._publish_arm(json.dumps(body, default=_json_default))
            t_end = self.clock_mono() + wait_s
            real_end = time.monotonic() + wait_s
            while not slot.event.is_set():
                left = min(t_end - self.clock_mono(), real_end - time.monotonic())
                if left <= 0.0:
                    break
                slot.event.wait(min(left, _WAIT_SLICE_S))
            with self._lock:
                result = slot.result
            if result is not None:
                return result
            self._say("request rid=%s task=%s: no result in %.2fs (deadline)"
                      % (rid, body.get("task"), wait_s))
            return {"status": "UNKNOWN", "reason": "deadline", "rid": rid}
        except Exception as exc:  # noqa: BLE001 - request NEVER raises, no matter what
            self._say("request rid=%s failed: %s: %s" % (rid, type(exc).__name__, exc))
            out = {"status": "UNKNOWN",
                   "reason": "client_error: %s: %s" % (type(exc).__name__, exc)}
            if isinstance(rid, str) and rid:
                out["rid"] = rid
            return out
        finally:
            if slot is not None and rid is not None:
                with self._lock:
                    if self._pending.get(rid) is slot:
                        del self._pending[rid]

    def on_result(self, text: Any) -> None:
        """The /pc_result subscription callback (ROS thread).  Never raises.

        Malformed JSON, a non-object value, a missing / non-string rid and a
        rid nobody is waiting for are all ignored.  The FIRST result for a
        pending rid wins; it wakes that waiter only.
        """
        try:
            res = json.loads(text)
            if not isinstance(res, dict):
                return
            rid = res.get("rid")
            if not isinstance(rid, str) or not rid:
                return
            with self._lock:
                slot = self._pending.get(rid)
                if slot is None or slot.result is not None:
                    return
                slot.result = res
                slot.event.set()
        except Exception:  # noqa: BLE001 - a result handler must never raise into rclpy
            return

    def warm(self, ttl_s: float) -> None:
        """Fire-and-forget pre-subscribe of the consumer (it answers nothing)."""
        try:
            self._publish_arm(json.dumps(
                {"task": "warm", "rid": self.new_rid("warm"), "ttl_s": ttl_s},
                default=_json_default))
        except Exception as exc:  # noqa: BLE001 - warm doesn't raise either
            self._say("warm failed: %s: %s" % (type(exc).__name__, exc))

    # ------------------------------------------------------------ pose
    def _source_pose(self) -> Optional[Dict[str, Any]]:
        """B8k: the odometry body pose when it is usable, else None.

        Usable = a dict that passes :func:`pose_check` (src "measured", the
        four POSE_KEYS finite) and carries a finite ``t`` no older than
        :data:`POSE_SOURCE_MAX_AGE_S` and not ahead of the wall clock by more
        than :data:`POSE_SOURCE_FUTURE_S`.  A copy is returned (the caller
        may keep it).  Never raises."""
        src = self._pose_source
        if src is None:
            return None
        try:
            pose = src()
            if not isinstance(pose, dict) or not pose_check(pose)[0]:
                return None
            t = float(pose.get("t"))
            age = float(self.clock_wall()) - t
            if not math.isfinite(age) or age > POSE_SOURCE_MAX_AGE_S \
                    or age < -POSE_SOURCE_FUTURE_S:
                return None
            return dict(pose)
        except Exception:  # noqa: BLE001 - an unreadable pose counts as no pose
            return None

    def measured_pose(self) -> Optional[Dict[str, Any]]:
        """The odometry body pose (B8k) when usable, else ONE ``pose`` verb on
        the writer socket, else None.

        ``None`` when there is no writer verb, the call raised, the reply is
        not ONE JSON object, or the object does not carry ``src == "measured"``
        (an ``ERROR:`` reply from a writer without H-WR-1 is the common case).
        Blocks no longer than the writer verb call itself.  Never raises.
        """
        pose = self._source_pose()
        if pose is not None:
            return pose
        verb = self._writer_verb
        if verb is None:
            return None
        try:
            line = verb("pose")
            if not isinstance(line, str):
                return None
            pose = json.loads(line.strip())
            if isinstance(pose, dict) and pose.get("src") == "measured":
                return pose
            return None
        except Exception:  # noqa: BLE001 - if the pose won't read, it's no pose
            return None

    #: helpers exposed on the client
    phase_of = staticmethod(phase_of)
    pose_equal = staticmethod(pose_equal)


# ============================================================ rim policy
def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def _is_finite_number(value: Any) -> bool:
    """A finite int / float, by the SAME predicate ``place_target_from_row``
    applies to the rim z (an int / float that is not a bool and is finite).  A
    z this client let through but that gate refused would fall back to the
    formula WITHOUT a log line, so the two must not differ."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:       # too big for a float
        return False


def _row_stamp(row: Dict[str, Any]) -> Optional[float]:
    """The receive stamp of a resolver row, finite or None.

    ``t`` is the key ``place_target.resolver_rows`` stamps from
    ``Detection.ts``; ``_t`` and ``ts`` are the other two spellings in the repo
    (``instance_labels._det_t`` / the Detection attribute).
    """
    for key in ("t", "_t", "ts"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None
