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


def place_target_from_row(
        row: dict,
        pc_rim: Optional[Callable[[dict], Optional[dict]]] = None,
) -> Tuple[str, str, Optional[List[float]]]:
    """(status, detail, ``[x, y, wrist_z]`` | None) for the resolved row.

    The geometry itself is ``_drop_point`` — never re-derived here, so the
    drop formula exists exactly once in the repo.  An inestimable height is a
    WARNING, not a refusal (the caller asked to place on top of a target it
    can see; the honest response is to release at the centroid + clearance and
    say so), but an unusable POSITION is a refusal: without x/y/z there is no
    target to place on at all.

    ``pc_rim`` is the optional point-cloud rim hook (H-PT-1; default None).
    The gateway passes a callable only when the rim fit is switched on.  It
    is called with ``row`` once, and only AFTER the position check passed.  It
    returns None or ``{"drop": {"x": .., "y": .., "z": ..}, ...}``; only
    ``["drop"]["z"]`` is read, and only when it is a finite int/float (a bool,
    a string, NaN or inf is rejected).  That z replaces the formula's drop z;
    x and y stay the formula's.  A callable that raises counts as None.  With
    the hook omitted, None, returning None, raising, or returning a rejected
    shape, the returned 3-tuple (detail string included) is the one this
    function returned before the hook existed.  When the rim z is applied the
    detail ends with ``[pc rim z=... replaces formula z=...]``.
    """
    drop = _impl("_drop_point")(_as_source(row))
    if drop is None:
        return ("refuse",
                "ERROR:/place_target: '%s' has no usable position "
                "(x/y/z absent or non-finite)" % row.get("label"),
                None)
    rim_marker = ""
    if pc_rim is not None:
        try:
            rim = pc_rim(row)
            rim_drop = rim.get("drop") if isinstance(rim, dict) else None
            rim_z = rim_drop.get("z") if isinstance(rim_drop, dict) else None
            if (isinstance(rim_z, (int, float))
                    and not isinstance(rim_z, bool)
                    and math.isfinite(rim_z)):
                rim_z = float(rim_z)
            else:
                rim_z = None
        except Exception:  # hook raise = None, not fail
            rim_z = None
        if rim_z is not None:
            rim_marker = ("  [pc rim z=%.3f replaces formula z=%.3f]"
                          % (rim_z, drop["z"]))
            drop = dict(drop, z=rim_z)
    height = row.get("height_m")
    lifted = False
    try:
        hf = float(height)
        lifted = math.isfinite(hf) and hf > 0.0
    except (TypeError, ValueError):
        lifted = False
    # drop['z'] = centroid z + lift + clearance, so subtracting the
    # clearance gives the object top back without restating the formula.
    top_z = drop["z"] - PLACE_CLEARANCE_M
    wrist_z = drop["z"] + PLACE_WRIST_STANDOFF_M
    detail = ("top_z=%.3f  wrist_z=%.3f (+%.3f standoff)"
              % (top_z, wrist_z, PLACE_WRIST_STANDOFF_M))
    status = "ok"
    if not lifted:
        status = "warn"
        detail += ("  [height_m absent - lifting 0.00 (fallback z+%.2f)]"
                   % PLACE_CLEARANCE_M)
    detail += rim_marker
    return status, detail, [drop["x"], drop["y"], wrist_z]


def fmt_num(value: Any, spec: str = "%.2f") -> str:
    """Format a numeric cell, or ``'?'`` when it is absent/non-numeric.

    A confidence the perception lane never sent must read as unknown, never as
    ``0.00`` — a fabricated zero looks like a measured one.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "?"
    return "?" if not math.isfinite(number) else spec % number


# ── outcome ──

def classify_dispatch(reply: str, verdict: Optional[str]) -> Tuple[str, str]:
    """(status, detail) for the ``execute`` row, from the SHIPPED reply grammar.

    Reuses the repo's own definitive rule rather than inventing one: any
    embedded ``ERROR:`` anywhere in the reply is a failure (brain/verifier.py
    ``_classify_reply``, mirrored by ``GatewayNode._shape_reply``).  The
    unheld-place refusal is split out because it is a POLICY refusal (the
    command was well-formed and the gateway declined on evidence), not a
    broken engine — an operator reads those two very differently.
    """
    text = str(reply or "")
    if verdict is not None and verdict.startswith("VERDICT:error"):
        return "error", text
    if text.startswith("ERROR:nothing verified held"):
        return "refuse", text
    if text.startswith("ERROR") or "ERROR:" in text:
        return "error", text
    return "ok", text


def verdict_row(verdict: Optional[str]) -> Tuple[str, str]:
    """(status, detail) for the VERDICT row.

    Success is claimed ONLY on an explicit ``VERDICT:success`` from the
    verifier.  A missing verdict is an ERROR, never a silent pass — an
    absent verifier must not read as a good place.
    """
    if not verdict:
        return "error", "no verdict (verifier did not run)"
    if verdict.startswith("VERDICT:success"):
        return "success", verdict
    if verdict.startswith("VERDICT:refused"):
        return "refuse", verdict
    return "error", verdict


def side_from_reply(reply: str) -> Optional[str]:
    """The released side, read off the SHIPPED place reply annotation.

    ``_do_place`` owns the held-side law and appends
    ``; prev_is_open=<side>:...`` — the only place the gateway states which
    hand it released.  DISPLAY-ONLY: an unreadable annotation leaves the side
    unknown rather than guessing at one (re-deriving the law here would create
    the second, divergent copy this module exists to avoid).
    """
    text = str(reply or "")
    marker = "prev_is_open="
    idx = text.find(marker)
    if idx < 0:
        return None
    rest = text[idx + len(marker):]
    for side in ("right", "left"):
        if rest.startswith(side + ":"):
            return side
    return None


# ── phases ──

class PhaseLog:
    """Ordered phase rows whose elapsed times are MEASURED, never invented.

    Each row's ``elapsed_s`` is the wall time since the previous row was
    added — the only timing the gateway can actually observe.  The VERDICT
    row carries the TOTAL, so the summary reads as the whole run.
    """

    def __init__(self, label: str):
        self.label = label
        self.side: Optional[str] = None
        self.rows: List[dict] = []
        self._t0: Optional[float] = None
        self._t_last: Optional[float] = None

    def start(self, now: float) -> None:
        self._t0 = now
        self._t_last = now

    def add(self, phase: str, status: str, detail: str, now: float) -> None:
        elapsed = 0.0 if self._t_last is None else max(0.0, now - self._t_last)
        self._t_last = now
        self.rows.append({"phase": phase, "status": status,
                          "detail": detail, "elapsed_s": elapsed})

    def total_s(self, now: float) -> float:
        return 0.0 if self._t0 is None else max(0.0, now - self._t0)

    def finish(self, status: str, detail: str, now: float) -> float:
        """Append the VERDICT row (elapsed = total) and return the total."""
        total = self.total_s(now)
        self.rows.append({"phase": "VERDICT", "status": status,
                          "detail": detail, "elapsed_s": total})
        self._t_last = now
        return total

    def header(self) -> str:
        return ("PLACE  target=%s  side=%s"
                % (self.label, self.side or "?"))


def render_table(log: PhaseLog) -> str:
    """Render the phase table (box drawing, every column measured).

    The ``detail`` cell is WORD-WRAPPED, never truncated.  A verifier verdict
    is long, and a table that ran off the terminal would be as unreadable as
    one that silently dropped the reason — wrapping keeps every character
    visible at a readable width.
    """
    rows = []
    for row in log.rows:
        chunks = textwrap.wrap(str(row.get("detail", "")), DETAIL_WRAP) or [""]
        rows.append((str(row.get("phase", "")), str(row.get("status", "")),
                     chunks, "%.2f s" % float(row.get("elapsed_s") or 0.0)))

    widths = [len(TABLE_HEADER[0]), len(TABLE_HEADER[1]), 0,
              len(TABLE_HEADER[3])]
    for name, status, chunks, elapsed in rows:
        widths[0] = max(widths[0], len(name))
        widths[1] = max(widths[1], len(status))
        widths[3] = max(widths[3], len(elapsed))
        for chunk in chunks:
            widths[2] = max(widths[2], len(chunk))

    def rule(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (w + 2) for w in widths) + right

    def line(cell: List[str]) -> str:
        parts = []
        for i, text in enumerate(cell):
            # elapsed right-aligned
            body = text.rjust(widths[i]) if i == 3 else text.ljust(widths[i])
            parts.append(" %s " % body)
        return "│" + "│".join(parts) + "│"

    def block(name: str, status: str, chunks: List[str],
              elapsed: str) -> List[str]:
        out = [line([name, status, chunks[0], elapsed])]
        for chunk in chunks[1:]:
            out.append(line(["", "", chunk, ""]))
        return out

    out = [log.header(), rule("┌", "┬", "┐"), line(list(TABLE_HEADER)),
           rule("├", "┼", "┤")]
    for i, (name, status, chunks, elapsed) in enumerate(rows):
        if i == len(rows) - 1 and len(rows) > 1:
            out.append(rule("├", "┼", "┤"))
        out.extend(block(name, status, chunks, elapsed))
    out.append(rule("└", "┴", "┘"))
    return "\n".join(out)


