"""place_target.py — the one-shot ``/place_target <label>`` lane.

A ``/place_target`` message names a scene object; the gateway resolves it
against the FRESH detections, computes a wrist target just above that object,
and dispatches the shipped coordinate place (``place x y z``) so the held
object is released ON the named target instead of at the current carry pose.

WHY THIS IS A MODULE AND NOT A NODE.  Five facts force the hosting:

  * the engine's per-phase stderr markers (``place step N: ...``) are captured
    by ``EngineBridge._drain_stderr`` — reachable only INSIDE the gateway
    process, where the engine runs as a child;
  * ``EngineBridge.PICK_REPLY_TIMEOUT_S`` is 300 s and the gateway spins a
    SingleThreadedExecutor, so a place performed inline in a DDS callback
    freezes the detection + vision lanes for up to five minutes (the ROS
    adapter therefore hands off to a worker thread);
  * the wire carries NO timestamp (the projector emits label/conf/x/y/z/…),
    so freshness must be stamped at RECEIVE — exactly what
    :func:`resolver_rows` does from ``Detection.ts``;
  * the held-side law, the unheld refusal and the post-place verifier are
    already owned by ``GatewayNode._do_place``; re-deriving any of them here
    would create a second, divergent copy;
  * ``skills/`` importing ``brain.``/``gateway.`` has zero precedent in this
    repo, so a standalone node would be the only component inverting the
    dependency direction.

So this module holds the PURE part (resolution, geometry, phase bookkeeping,
table rendering) and the gateway holds the thin wiring.  It is stdlib-only and
imports rclpy never, so it is unit-testable without ROS.

REUSE LAW — one implementation of every fact.  This module never re-derives:
  * the drop-point geometry  -> ``brain.agent.tools._drop_point``
  * label/rank/side/colour   -> ``skills.perception.instance_labels``
  * held-side + refusal + engine line + verifier -> ``GatewayNode._do_place``
                              (via ``dispatch("place", [x, y, z])``)

ONE DELIBERATE EXCEPTION — ``_drop_point``.  The robot's ``brain/agent/tools.py``
(md5 da36070b) predates the coordinate-place work and defines no ``_drop_point``
at all, and it has diverged from the repo for reasons far outside this lane, so
deploying it would change what the BRAIN sends for an ordinary place.
``_load_drop_point()`` therefore prefers the real import and falls back to a
copy that is line-for-line the same function — and
``tests/test_place_target_lane.py`` asserts the two agree on every input class
wherever both are importable, so they cannot drift apart unnoticed.

THE WRIST-vs-OBJECT SUBTLETY.  The engine reads the third number as the WRIST
height (``make_wrist_target``; the engine's own comment: "z = wrist height —
the standoff is the caller's concern"), NOT as the object's height.  With
``PLACE_WRIST_STANDOFF_M = 0.0`` the released object's bottom lands at
``top_z + PLACE_CLEARANCE_M - D`` where ``D`` is the unknown wrist-to-object-
bottom distance at the release pose.  Both residual errors are conservative
and push the release UP, never down: ``height_m`` is a 5-95 percentile span so
the true top is slightly higher, and ``D`` is positive.  The failure mode is a
short drop, never a collision with the target.  ``D`` MUST be measured on the
box before this constant is set to anything but 0.0.
"""

import importlib
import json
import math
import textwrap
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

#: inbound: one std_msgs/String, data = target label.
PLACE_TARGET_TOPIC = "/place_target"
#: outbound: one JSON per phase; the LAST adds ``table`` + ``verdict``.
PLACE_STATUS_TOPIC = "/place_status"

#: gap from the estimated object top to the commanded wrist height
#: (mirrors ``_drop_point``'s +0.02, which owns that number).
PLACE_CLEARANCE_M = 0.02
#: wrist to held-object bottom at release, metres.  0.0 reproduces the
#: drop formula; the real value must be MEASURED on the box, not guessed.
PLACE_WRIST_STANDOFF_M = 0.0

#: a place already in flight: refuse, never queue -- a queued second
#: place would release at a scene the first one has changed.
PLACE_BUSY_MSG = "ERROR:place_target busy - a place is already in flight"

#: the four measured rows.  No ascend/descend/release/closeout row:
#: ``place x y z`` is ONE blocking command with no internal boundaries,
#: so those would carry made-up times.  Engine markers (``place step N:``)
#: need an in-gateway stderr tap, off by default.
TABLE_HEADER = ("phase", "status", "detail", "elapsed")
#: detail-cell wrap width: the box adds 33 chars of chrome, so 64 keeps
#: lines inside 100 columns.  A verdict (~180 chars) must WRAP, not cut.
DETAIL_WRAP = 64

#: two layouts: the repo's skills/perception/instance_labels.py, or the
#: robot's BYTE-IDENTICAL copy (md5 965f6e34) at openbot/instance_labels.py.
#: Try both; one module serves both.
_INSTANCE_LABEL_MODULES = ("skills.perception.instance_labels",   # repo layout
                           "openbot.instance_labels")             # box layout

_IMPL: Dict[str, Any] = {}


def _load_instance_labels() -> Any:
    """The shared label->instance resolver, from whichever layout is present."""
    last_exc: Optional[Exception] = None
    for name in _INSTANCE_LABEL_MODULES:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_exc = exc
    raise ImportError("instance_labels not importable (tried %s): %s"
                      % (", ".join(_INSTANCE_LABEL_MODULES), last_exc))


def _drop_point_fallback(det: Any) -> Optional[dict]:
    """Equivalent to ``brain.agent.tools._drop_point``.  See _load_drop_point.

    This exists ONLY because the robot's ``brain/agent/tools.py`` predates the
    coordinate-place work and does not define ``_drop_point``.  Deploying the
    newer brain module would change what the BRAIN sends for a place — far
    outside this lane's blast radius — so the fallback covers that layout.
    ``tests/test_place_target_lane.py`` pins this byte-for-byte against the
    real ``_drop_point`` wherever both exist, so the two cannot drift.
    """
    def _finite(value: Any) -> Optional[float]:
        try:
            coord = float(value)
        except (TypeError, ValueError):
            return None
        return coord if math.isfinite(coord) else None

    x = _finite(getattr(det, "x", None))
    y = _finite(getattr(det, "y", None))
    z = _finite(getattr(det, "z", None))
    if x is None or y is None or z is None:
        return None
    height = _finite(getattr(det, "height_m", None))
    lift = height / 2.0 if height is not None and height > 0.0 else 0.0
    return {"x": x, "y": y, "z": z + lift + 0.02}


def _load_drop_point() -> Any:
    """The shared drop-point geometry, or the proven-equivalent fallback."""
    try:
        from brain.agent.tools import _drop_point  # noqa: PLC0415
    except ImportError:
        return _drop_point_fallback
    return _drop_point


def _impl(name: str) -> Any:
    """Lazily import one reused implementation, cached after first use.

    Lazy on purpose: the gateway imports this module at boot, and a missing
    ``skills/``/``brain/`` path must degrade to ONE honest error row — never
    an ImportError that takes the whole node down.  (The caller guarantees
    that: ``GatewayNode.place_target`` contains any failure and still emits a
    table.)
    """
    if not _IMPL:
        labels = _load_instance_labels()
        _IMPL.update({"_drop_point": _load_drop_point(),
                      "resolve_pick_target": labels.resolve_pick_target,
                      "stem": labels.stem,
                      "ref_to_label": labels.ref_to_label,
                      "PICK_STALE_S": labels.PICK_STALE_S,
                      "PICK_MIN_CONF": labels.PICK_MIN_CONF})
    return _IMPL[name]


# ── resolve ──

def resolver_rows(dets: Any) -> List[dict]:
    """Detection objects -> ``resolve_pick_target`` dicts.

    THE FRESHNESS STAMP IS THE POINT.  ``resolve_pick_target`` drops every row
    whose ``t``/``_t`` is absent, and the perception wire carries NO timestamp
    — so feeding raw wire rows to the resolver returns ``status:'none'`` for
    every label, forever, with a perfectly good object in frame.  Stamping
    ``t`` from the gateway's own ``Detection.ts`` (set at ingest) is what makes
    the resolver's staleness window real rather than permanently empty.

    ``_src`` carries the original Detection so the drop-point geometry can run
    on its native attribute object without a second, parallel representation.
    """
    rows: List[dict] = []
    for det in dets or ():
        label = getattr(det, "label", None)
        if not isinstance(label, str):
            continue
        rows.append({
            "label": label,
            "conf": getattr(det, "conf", None),
            "t": getattr(det, "ts", None),
            "x": getattr(det, "x", None),
            "y": getattr(det, "y", None),
            "z": getattr(det, "z", None),
            "height_m": getattr(det, "height_m", None),
            "hue": getattr(det, "hue", None),
            "color_name": getattr(det, "color_name", None),
            "body_side": getattr(det, "body_side", None),
            "_src": det,
        })
    return rows


def resolve(label: str, rows: List[dict], now: float) -> dict:
    """Resolve ``label`` against rows (the shipped pick-target grammar)."""
    return _impl("resolve_pick_target")(label, rows, now)


def stale_only(label: str, rows: List[dict], now: float) -> bool:
    """True when rows hold the label's stem but EVERY one is unusable.

    An old-but-present detection and an absent one both reach the resolver as
    ``status:'none'``.  Without this pre-check a stale scene masquerades as
    'not detected' — which would point the operator at the camera when the
    real problem is that nothing has been published for seconds.
    """
    stem = _impl("stem")
    # a ref-shaped target ('bottle_3') must become a label BEFORE the stem
    # compare, or no row matches, `seen` stays False, and a FROZEN scene
    # reads as 'not detected' -- the exact misdiagnosis this pre-check
    # exists to prevent.  See instance_labels.ref_to_label.
    target = stem(_impl("ref_to_label")(label))
    stale_s = _impl("PICK_STALE_S")
    min_conf = _impl("PICK_MIN_CONF")
    seen = False
    for row in rows or ():
        if stem(str(row.get("label") or "")) != target:
            continue
        seen = True
        try:
            t = float(row.get("t"))
            conf = float(row.get("conf"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(t) and math.isfinite(conf) \
                and (now - t) <= stale_s and conf >= min_conf:
            return False
    return seen


def classify_refusal(label: str, result: dict, stale: bool) -> Tuple[str, str]:
    """(status, detail) for a resolver result that is not ``'single'``.

    Every branch refuses — a failure to name the target must NEVER degrade to
    a bare current-pose release, which would drop the payload at whatever pose
    the arm happens to hold.
    """
    status = result.get("status")
    reason = str(result.get("reason") or "")
    visible = str(result.get("visible_text") or "")
    if status == "bad_shape":
        return "refuse", "ERROR:/place_target: %s" % reason
    if status == "ambiguous":
        return "refuse", \
            "ERROR:/place_target: %s - name #N or side=" % reason
    if stale:
        return "refuse", ("ERROR:/place_target: '%s' not detected "
                          "(no fresh detection; scene may be stale)" % label)
    if not visible:
        return "refuse", "ERROR:/place_target: '%s' not detected" % label
    return "refuse", \
        "ERROR:/place_target: %s (visible: %s)" % (reason, visible)


# ── target ──

def _as_source(row: dict) -> Any:
    """The attribute object ``_drop_point`` reads (Detection, or a shim).

    ``_drop_point`` uses ``getattr``, so a plain dict would read as "no
    coordinates" and refuse a perfectly good target.  The shim keeps the
    module unit-testable with plain dicts without a second geometry path.
    """
    src = row.get("_src")
    if src is not None:
        return src
    return types.SimpleNamespace(**{k: row.get(k)
                                    for k in ("x", "y", "z", "height_m")})
