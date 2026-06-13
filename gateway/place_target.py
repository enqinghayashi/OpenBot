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
