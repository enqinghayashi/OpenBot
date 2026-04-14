"""Tool layer for the G1 brain: Tool, ToolRegistry, result envelope, backend
protocol.

RESULT ENVELOPE CONTRACT — every tool result is a dict, exactly one shape:
    success: {'success': True,  'value': <Any>}    (value = the raw reply)
    failure: {'success': False, 'reason': <str>}   (reason = honest detail)
``envelope_to_str()`` serializes it to a protocol string the verifier and
the LLM can parse (``OK: <value>`` / ``ERROR: <reason>``).  The agent never
rewrites a tool result; it only truncates for the context budget.

The registry exposes the brain tools: the seven motion tools (pick,
place, place_at_spot, navigate, turn_around, open_hand, close_hand —
human-confirmed via the agent's pre-hook pipeline) plus resolve_ref,
scene_brief, query_user (blocking) and notify_user (non-blocking).

TOOL LAYER HONESTY (platform facts, verified on Stuart):
- ``pick`` is TABLETOP-ONLY: it resolves the requested label against the
  freshest detections, refuses floor objects (z below TABLETOP_MIN_Z, a
  TORSO-frame height — z=0 is the torso link; see the constant) with an
  honest capability-block refusal and never attempts a floor pick.
- Detections are EVENT-DRIVEN: silence == empty scene, not 'no data'; a
  stale observation refuses motion (STALE_DETECTION).
- The tool layer does NOT verify anything: ``pick`` returning means the
  engine was commanded; whether the world reached the goal state is the
  verifier's (post-hook) judgment, never the tool's.
- ``place`` semantics: place may carry an OPTIONAL x/y/z drop point — a
  commanded target the engine drops to; when the caller names none, the
  drop stays release-only (bare ``place``, current pose).  The verifier
  confirms ONLY that the grip opened and the object left the hand — it
  does NOT verify the object landed at any coordinate.
  Verifier v2 ``into:<container_ref>`` is ADDITIVE: the prepare hook then
  resolves the container against the freshest detections exactly like the
  pick label (``_find_target_detection`` + scene_graph.resolve_ref;
  unknown/unresolved container = the same honest NOT_DETECTED refusal) and
  the run layer annotates a SUCCESSFUL drop reply with
  into_ref/into_label/placed_object so the verifier's v2 containment half
  can judge the drop INSIDE the container region.  The annotation is
  brain-side post-tool evidence (into_* is never passed as a gateway arg).
  The into: lane also COMMANDS A DROP POINT over the resolved container
  (operator decision 2026-09-21): ``_drop_point`` supplies x/y/z, so the
  arm carries the object onto the container and releases there.  A place
  with no into: and no explicit coordinate stays a release at the current
  pose.

The :class:`GatewayBackend` protocol is the brain<->gateway seam (duck-
typed; implemented by gateway/brain_client.py on the laptop/Stuart socket
path and mocked in tests).  Registry tools read the backend/scene-graph
off the registry at call time, so late wiring via :meth:`ToolRegistry.attach`
works.

EMBODIMENT CONSUMPTION (round-3 + round-5 config-seam): the pick
placeholder consumes the embodiment envelope (wired onto the registry via
``attach(embodiment=...)`` — G1Agent.from_config injects the loader's
``EmbodimentConfig``, or anything with an ``.envelope``): the commanded
pick point is clamped to ``max_reach_m`` and offset by
``grip_center_standoff_m`` on z AND x, and the tabletop floor / overhead
ceiling refusals read ``pick_min_height_m`` / ``pick_max_height_m``.
WARNING (2026-09-06 E2 wrong-pick): the shipped ``grip_center_standoff_m``
is 0.0 — the pick engines/daemon treat stdin x/y/z as the WRIST target
and consume NO standoff, so ANY non-zero value shifts the commanded point
onto air (0.37 -> cx 0.514).  Set it non-zero only for a consumer that
reads it (pick_engine_v3 / tabletop_pick_place_engine).
``place_at_spot`` consumes ``base_relative.key_positions`` (round-5
config-seam; round-6 honesty): the named spot is resolved against the
CONFIGURED base-relative map for VALIDATION — an unknown spot name is an
honest UNKNOWN_SPOT tool error, never a guessed position.  Placement stays
a RELEASE: the object is dropped at the CURRENT pose — place_at_spot calls
backend.place WITHOUT x/y/z (no spot-relative coordinate is ever sent to
any actor; the platform has no spot-relative placement), so the spot is a
capability/consistency check, not a commanded target.  Without a wired
embodiment the tool layer keeps its own independent constants
(TABLETOP_MIN_Z / TABLETOP_MAX_Z), applies no standoff or clamp, and
place_at_spot refuses (no key_positions to resolve against).
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from .. import known_labels
from ..observation import detection_matches_label
from .optional_tools import register_optional_tools

_log = logging.getLogger(__name__)

# classes pick refuses: flat, fixed or too large to grasp
NOT_PICKABLE_CLASSES = frozenset((
    "person", "chair", "couch", "bed", "dining table", "tv", "laptop",
    "keyboard", "toilet", "refrigerator", "oven", "sink",
    "microwave", "monitor", "desk", "table", "door", "window", "bench",
    "potted plant",
))   # mouse is small enough to grasp

# optional import: a missing descent law disables the low-workspace lane
# and falls back to the floor refusal instead of failing at import.
try:
    from ..descent_law import effective_pick_floor
except Exception as _exc:  # noqa: BLE001
    print("[descent] descent_law import FAILED (%r) — low-workspace lane "
          "FORCED OFF" % (_exc,), flush=True)

    def effective_pick_floor(envelope):
        """Lane-OFF floor: the envelope's own ``pick_min_height_m`` UNCHANGED.

        The fallback used only when brain/descent_law.py is absent.  It
        reproduces the law's switch-off return exactly, so the floor refusal
        below stays today's honest capability-block.
        """
        return envelope.pick_min_height_m

# --- result envelope ---------------------------------------------------------


def ok(value: Any = None) -> dict:
    """Success envelope: ``{'success': True, 'value': value}``."""
    return {"success": True, "value": value}


def fail(reason: str) -> dict:
    """Failure envelope: ``{'success': False, 'reason': reason}``."""
    return {"success": False, "reason": reason}


def envelope_to_str(env: dict) -> str:
    """Serialize an envelope to the protocol string (verifier/LLM input)."""
    if env.get("success"):
        detail = "" if env.get("value") is None else str(env["value"])
        return f"OK: {detail}".rstrip()
    return f"ERROR: {env.get('reason') or 'unknown error'}"


def _not_detected(label: str, recovery: Optional[str] = None) -> str:
    """The ONE NOT_DETECTED refusal shape.  The head is written here and
    nowhere else, so the pick lookup and the place ``into:`` lookup cannot
    drift into two different stories about the same missing object.

    ``recovery`` is MODEL-FACING, and must be passed only where the string
    can never reach the speaker.  Part E (operator directive 2026-09-28)
    wants a refusal to name the way out — call ``navigate``, then try again —
    because that is what makes the model chain instead of giving up.  But
    ``_fastpath_summary`` interpolates a place ``into:`` refusal VERBATIM
    into the spoken sentence ("I could not place the bottle into the bin:
    <reason>"), and the operator's standing directive is that speech stays
    short and reason-free.  Reading "call navigate with this label ..." aloud
    would be an instruction addressed to the model, delivered to a human.

    So the asymmetry is deliberate and is pinned by test: the pick lookup
    passes a recovery (its failure phrasing is a fixed one-line sentence and
    never echoes the reason), the container lookup does not.
    """
    head = (
        f"NOT_DETECTED: {label!r} is not in the current scene "
        "(detections are event-driven — silence means empty)"
    )
    if recovery is None:
        return head
    return (
        head + ". This is NOT the end of the task: call navigate with this "
        f"label to search the workspace and walk to it, then {recovery}."
    )


#: model-facing next step for a fetch pick, never spoken and never
#: phrased as an error.
_FETCH_NEXT_STEP = ("; NEXT_STEP=fetch_return: this pick is part of a fetch for the "
                    "person who asked - do NOT stop here and do NOT put it on a "
                    "table: the robot walks back to the start point by itself "
                    "(else call return_home), then place with into='person' "
                    "(navigate to 'person' first if they are out of reach)")


def _not_detected_held(label: str, ss: Any) -> str:
    """NOT_DETECTED for a label the self-state believes is in the hand: the
    object is absent from the scene BECAUSE it is probably held, so the
    generic 'navigate to search' recovery would send the robot off for it."""
    age = 0
    try:
        age = ss.hold_age()
    except Exception:  # noqa: BLE001
        pass
    if _hold_is_measured(ss):
        # a hold the fingers confirm: they are held open by an object after the lift.
        # do not suggest open_hand here, it would release the object.
        return (
            "NOT_DETECTED: %r is not in the current scene (detections are "
            "event-driven \u2014 silence means empty) - because you are "
            "holding it (picked %ds ago; the fingers were held off their "
            "close command by an object after the lift, MEASURED). Stay: do NOT "
            "navigate to find it and do NOT pick it again. Call place if "
            "the operator asked you to put it somewhere; call open_hand only "
            "if the operator asks you to let go" % (label, age)
        )
    return (
        "NOT_DETECTED: %r is not in the current scene (detections are "
        "event-driven \u2014 silence means empty) - but you may be holding it "
        "(picked %ds ago; if a release or cancel left the hand state "
        "unconfirmed it may or may not still be there). It is probably in "
        "your hand: do NOT navigate to find it now; check, or call place "
        "or open_hand first. If the operator asks for it again after "
        "this, navigate to search for it"
        % (label, age)
    )


def _hold_is_measured(ss: Any) -> bool:
    """B8h: the self-state's hold is the verifier's MEASURED one (False for a
    self-state without the method: byte-identical old wording)."""
    try:
        return bool(ss.hold_measured())
    except Exception:  # noqa: BLE001
        return False
