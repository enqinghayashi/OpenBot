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


def _not_detected_unconfirmed(label: str, ss: Any) -> str:
    """B8h: NOT_DETECTED for a label whose last pick came back UNCONFIRMED.
    MEASURED (s1): a lifted object leaves the camera's view, so its absence is
    the expected sign of a hold.  The generic reply said 'call navigate', the
    held reply said 'open_hand first'; both made the model leave or drop it.
    Neither is said here."""
    age = 0
    try:
        age = ss._age(ss.pick_unconfirmed["ts"])
    except Exception:  # noqa: BLE001
        pass
    return (
        "NOT_DETECTED: %r is not in the current scene (detections are "
        "event-driven \u2014 silence means empty) - and your pick of it %ds ago "
        "was not fully confirmed, so it is probably in your hand (the camera "
        "cannot see a held object). Stay: do NOT navigate to find it and do NOT "
        "pick it again. Tell the operator in one sentence that the pick is "
        "not fully confirmed; call place if the operator asked you to put it "
        "somewhere; call open_hand only if the operator asks you to let go"
        % (label, age)
    )


#: reach-ceiling failures from the pick engine, bare or behind engine_reply=.
_REACH_CEILING_RE = re.compile(r"\breach_ceiling\b")
#: set when the hint is added, so it is never added twice.
_REACH_HINT_MARK = "OUT OF REACH FROM HERE"


def _reach_ceiling_hint(label: str) -> str:
    """Model-facing recovery for ERROR:reach_ceiling: walk closer, re-pick."""
    return (
        ". %s: the arm cannot reach %r from where the robot stands now. This "
        "is NOT the end of the task: the next step is to call navigate with "
        "the label %r to walk closer to it, then pick it again. Do not ask "
        "the operator for permission, and do not switch arms"
        % (_REACH_HINT_MARK, label, label)
    )


#: cross-body failures from the pick engine, bare or behind engine_reply=.
#: walking closer cannot help, so the hint asks the human to act.
_SELF_COLLISION_RE = re.compile(r"\bself_collision\b")
#: appended once; presence suppresses a second hint.
_CROSS_BODY_MARK = "CROSS-BODY FROM HERE"
#: collision fields; optional, missing values fall back to plain text.
_SELF_COLLISION_FIELDS_RE = re.compile(
    r"side=(left|right) x=(-?[\d.]+)m y=(-?[\d.]+)m .*?move_right_m=([\d.]+)")
#: stage field: 'precheck' means no arm motion, any other stage means
#: the arm moved and is parked at ready.
_SELF_COLLISION_STAGE_RE = re.compile(r"\bstage=([A-Za-z0-9_-]+)")
#: arm-safety state after the failure, e.g. already_safe or parked.
_ARM_AFTER_RE = re.compile(r"\barm_after=([a-z_]+)")
_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.I)


def _self_collision_hint(label: str, reason: str) -> str:
    """Model-facing recovery for ERROR:self_collision: do NOT navigate; tell
    the human to move the object right / turn the robot.  Pick only: a pick
    failure is spoken as a fixed sentence (tool_executor._fastpath_summary),
    so this text never reaches the speaker.  It states only what the engine
    line says: the side of the midline (side=), the stage (a live stage
    means the arm HAD moved), and numbers only when the fields are sane."""
    x = cm = deg = None
    side = "left"
    m = _SELF_COLLISION_FIELDS_RE.search(reason or "")
    if m:
        try:
            side = m.group(1)
            xv, yv, mv = float(m.group(2)), float(m.group(3)), float(m.group(4))
            if all(math.isfinite(v) for v in (xv, yv, mv)):
                x = "%.3f" % xv
                if mv > 0.0:
                    cm = int(round(mv * 100.0))
                    if yv > 0.0:
                        deg = int(round(math.degrees(math.atan2(mv, yv))))
        except (TypeError, ValueError):
            pass
    st = _SELF_COLLISION_STAGE_RE.search(reason or "")
    stage = st.group(1) if st else None
    aa = _ARM_AFTER_RE.search(reason or "")
    arm_after = aa.group(1) if aa else None
    noun = _ARTICLE_RE.sub("", label) or "object"
    where = " (torso x=%s m)" % x if x is not None else ""
    if side == "right":
        pos = "%r is too close to the robot's centre line%s" % (label, where)
        say = "the %s is too close to my centre line" % noun
    else:
        pos = "%r is on the robot's LEFT side%s" % (label, where)
        say = "the %s is on my left side" % noun
    if stage == "precheck":
        when = ("so the pick was REFUSED before any arm motion (the arm did "
                "NOT move)")
        if arm_after in ("parked", "park_failed", "unknown"):
            when += ("; afterwards the gateway's arm-safety check reported "
                     "arm_after=%s" % arm_after)
    elif stage:
        if arm_after in ("park_failed", "unknown"):
            parked = ("its park back to the ready pose was NOT confirmed "
                      "(arm_after=%s)" % arm_after)
        elif arm_after in ("parked", "already_safe"):
            parked = "it was parked back at its ready pose"
        else:
            parked = "the engine's safe exit parks it back at its ready pose"
        when = ("so the pick was STOPPED before the %s waypoint was commanded "
                "(the arm had moved; %s)" % (stage, parked))
    else:
        when = "so the pick was REFUSED"
    ask = "please move it to my right side"
    if cm is not None:
        ask += " (about %d cm further right)" % cm
    ask += " or turn me to the left"
    if deg is not None:
        ask += " (about %d degrees)" % deg
    return (
        ". %s: %s: from where the robot stands the right arm would have to "
        "reach across the body and press into the torso or the tucked left "
        "arm, %s. Walking closer does NOT help: do NOT call navigate and do "
        "NOT pick %r again until the human says it has been moved or the robot "
        "has been turned. Tell the human in one sentence: %s; %s so it sits "
        "right of my centre line, then ask me again. Do not switch arms"
        % (_CROSS_BODY_MARK, pos, when, label, say, ask)
    )


def _annotate_cross_body(env, label: str):
    """Append the cross-body hint ONCE to a failed pick/place reply whose
    reason carries ERROR:self_collision (bare or lane-wrapped)."""
    try:
        reason = env.get("reason") if isinstance(env, dict) else None
        if (isinstance(env, dict) and not env.get("success")
                and isinstance(reason, str)
                and _SELF_COLLISION_RE.search(reason)
                and _CROSS_BODY_MARK not in reason):
            env = dict(env)
            env["reason"] = reason.rstrip() + _self_collision_hint(
                label or "object", reason)
    except (TypeError, ValueError):
        pass
    return env


# --- constants ---------------------------------------------------------------

DEFAULT_TIMEOUT_S = 30.0

#: cancel reply timeout: a healthy reply arrives in milliseconds, so a
#: longer bound only waits on a hang. keep it above the gateway's 5.0 s
#: cancel-reply timeout.
CANCEL_TIMEOUT_S = 10.0

# must mirror the per-tool contract timeouts
CONTRACT_TIMEOUTS: Dict[str, float] = {
    # pick needs 180 s: the engine reports done in about 3 s, then the daemon
    # spends about 33 s lifting and tucking. timeouts must nest as 120 s
    # internal < 150 s socket < 180 s tool.
    "pick": 180.0,
    "place": 60.0,
    # navigate needs 180 s: a full search takes 12 windows of 5 s plus a
    # ~1.05 s turn each (~73 s) before the FSM gives up, before any detected
    # hold of up to 10 s. a 60 s bound would cut that short.
    "navigate": 180.0,
    "open_hand": 10.0,
    "close_hand": 15.0,
    # pose needs 30 s: 'tuck' is two moves (carry and carry_left), each a
    # gravity feed-forward solve.
    "pose": 30.0,
}

#: names the tool accepts; the backend maps them to engine verbs, which
#: carry 'tuck' per arm.
POSE_NAMES: frozenset = frozenset({"ready", "tuck", "neutral", "home"})

# torso frame: the torso link is z=0, shared by the projector and the
# engine. tabletop rows read z in [0, 0.07] and ground objects near
# -0.75, so -0.35 refuses ground picks and keeps tabletops.
TABLETOP_MIN_Z = -0.35
TABLETOP_MAX_Z = 1.20  # default when no embodiment supplies a bound

#: a container detection is usable only while this fresh; only the
#: current row is checked for disambiguation.
CONTAINER_REF_FRESH_S = 10.0

# a value matching this is written bare; other labels are quoted so
# multi-word names like 'wine glass' survive the round trip.
_ANNOTATION_BARE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _annotation_value(value: str) -> str:
    """Encode a label for the legacy place-into annotation format.

    Bare when the value contains only ``[A-Za-z0-9_.-]`` (single-token
    labels — annotation bytes are byte-identical with the legacy format);
    double-quoted otherwise, so multi-word labels ('wine glass') survive
    the round trip with internal spaces intact.  The verifier's regexes
    (brain/verifier._INTO_REF_RE/_INTO_LABEL_RE/_PLACED_OBJECT_RE) accept
    the optional quote pair around the value.
    """
    if _ANNOTATION_BARE_VALUE_RE.match(value or ""):
        return value
    return '"%s"' % value


#: extra lift for place: max(pick-time h when 0 < h <= 0.5, a class floor).
#: the gate keeps junk heights out; 2.5 x 0.095 = 0.2375 m cup floor,
#: 2.5 x 0.16 = 0.40 m engine cap. never applied to a pick.
PLACE_LIFT_FLOOR_H_M = 0.095
PLACE_LIFT_TALL_H_M = 0.16
PLACE_LIFT_TALL_CLASSES = frozenset(("bottle", "vase", "wine glass"))
PLACE_LIFT_H_MAX_M = 0.5  # matches the gateway pick height limit


def place_lift_height(cls: Any, h_pick: Any) -> float:
    """B8i: the place-only height token (m) for the engine's 2.5x extra lift."""
    try:
        h = float(h_pick)
    except (TypeError, ValueError):
        h = 0.0
    if not (math.isfinite(h) and 0.0 < h <= PLACE_LIFT_H_MAX_M):
        h = 0.0
    c = str(cls or "").strip().lower()
    floor = (PLACE_LIFT_TALL_H_M if c in PLACE_LIFT_TALL_CLASSES
             else PLACE_LIFT_FLOOR_H_M)
    return round(max(h, floor), 4)


def _drop_point(det: Any) -> Optional[dict]:
    """Optional coordinate drop target just above a detection.

    Returns ``{"x": det.x, "y": det.y, "z": det.z + height_half + 0.02}``
    (``height_half`` = ``det.height_m / 2`` when the detection reports a
    positive finite height, else 0) — a commanded place target that releases
    the held object onto/into the destination detection instead of a bare
    current-pose release.  Returns None (the caller then keeps the place
    release-only) when any coordinate is missing, non-numeric, or
    non-finite: a junk/absent detection must never fabricate a target.
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
    # cap the height by container class so a tall, under-segmented mask
    # cannot push the drop point out of the lane band.
    try:
        from ..observation import detection_base_class as _dbc
        _cap = CONTAINER_H_MAX_M.get(_dbc(det), CONTAINER_H_MAX_DEFAULT_M)
    except Exception:  # noqa: BLE001
        _cap = CONTAINER_H_MAX_DEFAULT_M
    if height is not None:
        height = min(height, _cap)
    lift = height / 2.0 if height is not None and height > 0.0 else 0.0
    return {"x": x, "y": y, "z": z + lift + 0.02}


# --- placing into a container ---
# into a container: the walk stops at the near edge (0.45 m for a
# non-pickable), creeps to 0.25 m and releases. a centre aim overshoots
# reach, so aim just past the near rim, clearing it, and reach-check.
#: set to '0' to turn into-container aiming off
CONTAINER_PLACE_ENV = "OPENBOT_CONTAINER_PLACE"
#: classes treated as containers. keep this in sync with the pick
#: container classes; never person/chair, nor cup, vase or plate.
CONTAINER_CLASSES = frozenset((
    "bowl", "bin", "box", "basket", "bucket", "tray", "pot", "pan",
    "lunchbox", "toolbox", "backpack", "handbag", "suitcase"))
#: reach law: residual is max(0, |target - shoulder| - REACH_LEN_M).
REACH_SHOULDER = (0.100, 0.004, 0.248)
REACH_LEN_M = 0.386
REACH_OK_M = 0.10            # 0.10 m tolerance; a tighter value would refuse reachable containers
AIM_CLEAR_M = 0.04           # object bottom clears the rim by this much
ENGINE_Z_OFFSET_M = 0.044    # engine z offset: the wrist lands 4.4 cm below the z that is sent
AIM_INSET_FRAC = 0.35        # inset as a fraction of container depth
AIM_INSET_MIN_M = 0.05
AIM_INSET_MAX_M = 0.12
AIM_INSET_DEAD_MIN_M = 0.06  # minimum inset when the aim is dead-reckoned
AIM_FAR_MARGIN_M = 0.04      # margin kept inside the far rim
AIM_Z_STEP_M = 0.02
AIM_Z_RAISE_MAX_M = 0.40     # cap on how far the aim may be raised
AIM_Z_MAX = 0.22             # aim never goes above this z
#: a detection above this z is not a container; it is the hand or a
#: mislabelled object.
CONTAINER_Z_MAX_M = 0.20
#: height cap per class, applied to the rim estimate z + h/2; a taller
#: read means the mask ate the surface under the object.
CONTAINER_H_MAX_M = {"bowl": 0.15, "cup": 0.15, "tray": 0.10, "basket": 0.40,
                     "bucket": 0.40, "bin": 0.60, "box": 0.50}
CONTAINER_H_MAX_DEFAULT_M = 0.50
#: refuse a place-on target farther than this planar distance or with a
#: larger fit residual; navigate closer instead.
PLACE_ON_MAX_PLANAR_M = 0.55
PLACE_ON_MAX_RESIDUAL_M = 0.10
#: the descent lane's low band in the wire z frame; a rim inside it is
#: never refused here.
LOW_BAND_Z_MIN_M = -0.54
LOW_BAND_Z_MAX_M = -0.02
AIM_X_MARGIN_M = 0.06        # margin keeping the aim inside the side rims
CONTAINER_HALF_MIN_M = 0.06
CONTAINER_HALF_MAX_M = 0.35
#: clearance above the rim reported by the point-cloud bowl fit.
PC_RIM_CLEARANCE_M = 0.02
#: only round openings use a circle fit (radius 0.05-0.18 m); a box
#: shape overestimates the rim.
PC_RIM_CLASSES = frozenset(("bowl", "cup", "vase", "bin"))
#: largest near edge where the creep phase may start
CREEP_START_MAX_EDGE_M = 0.60
#: containers this tall are deep: judge them by their footprint, not
#: by the +-0.12 m box.
DEEP_CONTAINER_H_M = 0.25
#: a cached container row may be used for this long
LOCK_USE_ROW_MAX_S = 20.0
_CREEP_TOKEN_RE = re.compile(r"\b([A-Z_]+)=([^;\s]+)")


def container_place_on(environ: Any = None) -> bool:
    """OPENBOT_CONTAINER_PLACE: unset / anything but '0' = ON."""
    if environ is None:
        import os  # imported lazily; stdlib only
        environ = os.environ
    return str(environ.get(CONTAINER_PLACE_ENV, "1")).strip() != "0"


def reach_residual(x: float, y: float, z: float) -> float:
    """B8j reach law: max(0, |(x, y, z) - REACH_SHOULDER| - REACH_LEN_M)."""
    sx, sy, sz = REACH_SHOULDER
    return max(0.0, math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2)
               - REACH_LEN_M)


def container_aim(xc: float, yc: float, half: float, rim_z: float,
                  edge: float, h_obj: float, dead: bool = False
                  ) -> Optional[dict]:
    """The drop point for a container whose near rim is ``edge`` m away
    (planar) on the bearing of (xc, yc): ``inset`` past the near rim, the
    held object's bottom AIM_CLEAR_M over the rim (``h_obj`` its height),
    raised in AIM_Z_STEP_M steps and moved toward the near rim until the
    reach law accepts it (deepest inset first: containment before a low
    drop).  None = no reachable point inside the container."""
    vals = (xc, yc, half, rim_z, edge, h_obj)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in vals):
        return None
    d = math.hypot(xc, yc)
    if d < 0.05 or edge <= 0.0:
        return None
    half = max(CONTAINER_HALF_MIN_M, min(CONTAINER_HALF_MAX_M, float(half)))
    ux, uy = xc / d, yc / d
    depth = 2.0 * half
    ins_lo = AIM_INSET_DEAD_MIN_M if dead else AIM_INSET_MIN_M
    ins_hi = min(max(AIM_INSET_MIN_M, min(AIM_INSET_MAX_M,
                                          AIM_INSET_FRAC * depth)),
                 depth - AIM_FAR_MARGIN_M)
    if ins_hi < ins_lo:
        insets = [round(depth / 2.0, 3)]          # container too small for an inset: aim at its centre
    else:
        insets, k = [], 0
        while ins_hi - 0.01 * k >= ins_lo - 1e-9:
            insets.append(round(ins_hi - 0.01 * k, 3))
            k += 1
    z0 = rim_z + AIM_CLEAR_M + ENGINE_Z_OFFSET_M + max(0.0, h_obj) / 2.0
    zs, k = [round(z0, 3)], 1
    while (z0 + AIM_Z_STEP_M * k <= min(AIM_Z_MAX, z0 + AIM_Z_RAISE_MAX_M)
           + 1e-9):
        zs.append(round(z0 + AIM_Z_STEP_M * k, 3))
        k += 1
    x_lo, x_hi = xc - (half - AIM_X_MARGIN_M), xc + (half - AIM_X_MARGIN_M)
    for ins in insets:
        da = edge + ins
        x, y = ux * da, uy * da
        if x_lo <= x_hi:
            x = max(x_lo, min(x_hi, x))
            if x < 0.0 <= x_hi:
                x = 0.0            # the right arm cannot cross the centre line
        for z in zs:
            r = reach_residual(x, y, z)
            if r <= REACH_OK_M:
                return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3),
                        "inset": ins, "residual": round(r, 3),
                        "edge": round(edge, 3)}
    return None


def _low_lane_drop_point(geo: dict) -> Optional[dict]:
    """B8j-hf13: the drop point handed to the descent lane for a low container:
    container_aim's x/y rule (deepest inset past the near rim, inside the side
    rims, x >= 0 for the right arm when the opening allows) without its
    standing reach gate, at z = rim + 0.02 (the legacy release height)."""
    try:
        xc, yc = float(geo["x"]), float(geo["y"])
        half = max(CONTAINER_HALF_MIN_M, min(CONTAINER_HALF_MAX_M, float(geo["half"])))
        edge, rim = float(geo["near"]), float(geo["rim"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (xc, yc, half, edge, rim)):
        return None
    d = math.hypot(xc, yc)
    if d < 0.05:
        return None
    depth = 2.0 * half
    ins = min(max(AIM_INSET_MIN_M, min(AIM_INSET_MAX_M, AIM_INSET_FRAC * depth)),
              depth - AIM_FAR_MARGIN_M)
    if ins < AIM_INSET_MIN_M:
        ins = depth / 2.0
    da = max(0.0, edge) + ins
    x, y = xc / d * da, yc / d * da
    x_lo, x_hi = xc - (half - AIM_X_MARGIN_M), xc + (half - AIM_X_MARGIN_M)
    if x_lo <= x_hi:
        x = max(x_lo, min(x_hi, x))
        if x < 0.0 <= x_hi:
            x = 0.0
    return {"x": round(x, 3), "y": round(y, 3), "z": round(rim + 0.02, 3)}


def parse_creep_reply(text: Any) -> Dict[str, str]:
    """The KEY=value tokens of a gateway creep reply (strings)."""
    return dict(_CREEP_TOKEN_RE.findall(text if isinstance(text, str) else ""))


# motion tools the model sees; the backend maps each to a gateway verb
MOTION_TOOLS = frozenset({"pick", "place", "navigate", "open_hand", "close_hand"})

TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ToolError(Exception):
    """Base class for tool-layer errors."""


class ToolNameConflict(ToolError):
    """Raised when a tool name is registered twice."""


class UnknownTool(ToolError):
    """Raised by ToolRegistry.execute for a name that is not registered."""


class ToolNameInvalid(ValueError):
    """Raised when a Tool name violates the naming pattern."""


# --- backend protocol ---------------------------------------------------------


class GatewayBackend(Protocol):
    """Brain<->gateway seam (duck-typed; implement in gateway_node.py).

    ``observe()`` returns a brain.observation.WorldObservation (the
    normalized snapshot with ``detections`` / ``grip`` / ``walk`` /
    ``detections_ts`` / ``fresh``) or None when no observation source is
    wired.  The motion methods receive RESOLVED arguments (x/y/z in torso
    frame) and return an envelope dict; their ``value`` carries the raw
    slave reply (e.g. 'DONE:contact') for the verifier to judge.
    """

    def observe(self) -> Any:
        """Latest WorldObservation, or None."""
        ...

    def pick(self, args: dict) -> dict:
        """Command the pick engine (tabletop only); args carry x/y/z/ref."""
        ...

    def place(self, args: dict) -> dict:
        """Release-only drop of the held object."""
        ...

    def navigate(self, args: dict) -> dict:
        """Walk FSM approach; verify only via /walk_status == REACHED."""
        ...

    def open_hand(self, args: dict) -> dict:
        """Open the gripper (side: right|left|both)."""
        ...

    def close_hand(self, args: dict) -> dict:
        """Close the gripper (side: right|left|both)."""
        ...

    def query_user(self, args: dict) -> dict:
        """Blocking human question; value = the human's answer."""
        ...

    def notify_user(self, args: dict) -> dict:
        """Non-blocking notice to the human."""
        ...

    def stop(self, args: dict) -> dict:
        """Immediate halt (no confirmation required — stopping is always
        allowed; the safety-critical reverse of the confirm gate)."""
        ...

    def capture_image(self) -> Optional[dict]:
        """Latest camera frame {jpeg_b64, w, h, ts, fresh}, or None.

        The camera lane is OPTIONAL (vision.enabled): a backend with no
        camera returns None.  ``fresh`` is the frame's honesty flag —
        False means stale / no camera / encode error (with a ``reason``),
        never a stale frame under the green light.  Callers MUST
        getattr-guard this member — camera is optional and older
        backends may lack it.
        """
        ...


# --- Tool ---------------------------------------------------------------------


class Tool:
    """A callable tool exposed to the LLM.

    ``schema`` is an OpenAI function-calling JSON schema
    (``{"type": "object", "properties": {...}, "required": [...]}``).
    ``run(args: dict) -> dict`` returns the result envelope.  ``blocking``
    marks user-interaction tools (query_user blocks the turn; notify_user
    does not).  ``prepare(args)`` is the agent's PRE-hook seam: it resolves
    arguments (e.g. a pick label -> torso-frame x/y/z/ref) so the safety
    hooks can inspect them before any motion, returning
    ``(augmented_args, refusal_or_None)``.
    """

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict,
        run: Callable[[dict], dict],
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        blocking: bool = True,
        prepare: Optional[Callable[[dict], tuple[dict, Optional[dict]]]] = None,
    ) -> None:
        if not isinstance(name, str) or not TOOL_NAME_RE.match(name):
            raise ToolNameInvalid(
                f"tool name {name!r} must match ^[a-z][a-z0-9_]{{1,63}}$"
            )
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"tool {name!r} needs a non-empty description")
        self._validate_schema(schema)
        if not callable(run):
            raise TypeError(f"tool {name!r} run must be callable")
        if timeout_s <= 0:
            raise ValueError(f"tool {name!r} timeout_s must be > 0")
        if prepare is not None and not callable(prepare):
            raise TypeError(f"tool {name!r} prepare must be callable")
        self.name = name
        self.description = description
        self.schema = schema
        self.run = run
        self.timeout_s = float(timeout_s)
        self.blocking = bool(blocking)
        self._prepare_hook = prepare

    @staticmethod
    def _validate_schema(schema: dict) -> None:
        """Validate that ``schema`` is a well-formed JSON-schema object."""
        if not isinstance(schema, dict):
            raise ValueError("schema must be a dict (JSON schema object)")
        if schema.get("type") != "object":
            raise ValueError("schema['type'] must be 'object'")
        props = schema.get("properties")
        if not isinstance(props, dict):
            raise ValueError("schema['properties'] must be a dict")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError("schema['required'] must be a list")
        for field_name in required:
            if field_name not in props:
                raise ValueError(
                    f"required field {field_name!r} missing from properties"
                )

    def validate_args(self, args: dict) -> List[str]:
        """Validate ``args`` against the schema; [] means valid."""
        if not isinstance(args, dict):
            return [f"arguments (expected a dict, got {type(args).__name__})"]
        errors: List[str] = []
        props = self.schema.get("properties", {})
        for field_name in self.schema.get("required", []):
            if field_name not in args:
                errors.append(f"{field_name} (missing required argument)")
        for field_name, value in args.items():
            spec = props.get(field_name)
            if spec is None:
                continue  # extra arguments are ignored
            expected = spec.get("type")
            if expected is None:
                continue
            if not self._type_matches(value, expected):
                errors.append(
                    f"{field_name} (expected {expected}, got {type(value).__name__})"
                )
            enum = spec.get("enum")
            if enum is not None and value not in enum:
                errors.append(
                    f"{field_name} (must be one of {enum}, got {value!r})"
                )
        return errors

    @staticmethod
    def _type_matches(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected in ("number", "integer"):
            is_num = isinstance(value, (int, float)) and not isinstance(
                value, bool
            )
            return is_num and (expected == "number" or isinstance(value, int))
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "null":
            return value is None
        return True  # unknown schema types accept any value

    # -- invocation ------------------------------------------------------------

    def prepare(self, args: dict) -> tuple[dict, Optional[dict]]:
        """Resolve/prepare args for the pre-hook phase.

        Returns ``(augmented_args, refusal)``; ``refusal`` is a failure
        envelope when the tool cannot proceed at all (e.g. the pick target
        is not in fresh detections).  The agent runs the confirm gate and
        safety checks on the AUGMENTED args (so the safety hook sees
        resolved coordinates) and refuses motion when a refusal is
        returned.  Base implementation passes args through unchanged; a
        tool may override it via the constructor's ``prepare`` hook.
        """
        if self._prepare_hook is not None:
            return self._prepare_hook(args)
        return dict(args), None

    def call(self, args: dict) -> dict:
        """Validate, then execute ``run`` under the per-tool timeout.

        Never raises: every failure mode is returned as a failure envelope.
        """
        violations = self.validate_args(args)
        if violations:
            detail = "; ".join(violations)
            return fail(f"invalid arguments: {detail}")
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"tool-{self.name}"
        )
        future = pool.submit(self.run, args)
        try:
            try:
                result = future.result(timeout=self.timeout_s)
            except concurrent.futures.TimeoutError:
                return fail(
                    f"TIMEOUT: {self.name} did not finish in {self.timeout_s:g}s"
                )
        except Exception as exc:  # noqa: BLE001
            return fail(f"{self.name} raised: {exc}")
        finally:
            # do not wait: a handler that outlives its timeout must not block the
            # caller, so the single-use pool is shut down without waiting.
            pool.shutdown(wait=False)
        if isinstance(result, dict) and "success" in result:
            return result
        return fail(f"{self.name} returned a non-envelope result: {result!r}")


# --- ToolRegistry ---------------------------------------------------------------


class ToolRegistry:
    """Holds the agent's flat tool set, in registration order.

    Tools read ``backend`` and ``scene_graph`` off the registry at call
    time, so :meth:`attach` can wire them late (e.g. when the gateway
    connects after construction).
    """

    def __init__(
        self,
        backend: Any = None,
        scene_graph: Any = None,
        embodiment: Any = None,
        switches: Any = None,
    ) -> None:
        self.backend = backend
        self.scene_graph = scene_graph
        self.embodiment = embodiment
        # optional lanes, read once when the optional tools are registered;
        # None turns all of them off.
        self._optional_switches = switches
        self._tools: Dict[str, Tool] = {}
        self._register_defaults()

    def attach(
        self,
        backend: Any = None,
        scene_graph: Any = None,
        embodiment: Any = None,
    ) -> None:
        """Wire the backend/scene graph/embodiment (idempotent; None leaves
        unchanged).  ``embodiment`` is a brain.embodiment.Embodiment (or
        anything with an ``.envelope``) read by the pick placeholder."""
        if backend is not None:
            self.backend = backend
        if scene_graph is not None:
            self.scene_graph = scene_graph
        if embodiment is not None:
            self.embodiment = embodiment

    def _left_hand_in_service(self) -> bool:
        """BRAIN BR-6: embodiment.envelope.left_hand_in_service (strict
        True); an embodiment without an envelope reads False (refused)."""
        env = getattr(self.embodiment, "envelope", None)
        return getattr(env, "left_hand_in_service", False) is True

    # -- defaults ---------------------------------------------------------------

    def _register_defaults(self) -> None:
        """The brain tools (motion + query/user)."""

        def make_pick() -> Tool:
            _left_in_service_at_reg = self._left_hand_in_service()

            def prepare(args: dict) -> tuple[dict, Optional[dict]]:
                """Resolve label -> fresh detection coordinates (x/y/z/ref)
                and bind the grasp orientation to the detection.

                ORIENTATION (fix 2026-09-07): the depth projector computes
                whether the object is upright or horizontal, but the old
                lane DROPPED that field, so a lying (horizontal) bottle was
                picked with the VERTICAL approach.  This prepare hook is the
                single place that already matches the label to the SAME
                observation that produced x/y/z, so 'auto' (or an absent
                orientation) binds here to the detection's own orientation
                — a concrete 'upright'/'horizontal' is what travels on the
                wire, never 'auto'.  An explicit upright/horizontal request
                is honored; an unknown value falls back to 'auto'
                resolution; a detection with no known orientation defaults
                to 'upright' (preserving today's vertical behaviour).
                """
                from ..observation import detection_base_class  # used below for the not-pickable check

                label = (args.get("label") or "").strip()
                if not label:
                    return dict(args), fail("invalid arguments: missing label")
                # a left pick is refused before any motion unless the envelope
                # confirms the left hand is in service.
                arm = str(args.get("arm") or "auto").strip().lower()
                if arm == "left" and not self._left_hand_in_service():
                    return dict(args), fail(
                        "pick refused: the LEFT hand is out of service "
                        "(operator directive) - pick with the right arm (arm "
                        "'auto' or 'right'). A failed pick is never a reason "
                        "to switch arms: re-observe, navigate closer, or "
                        "open_hand and retry with the right arm")
                # resolve the alias first so a refusal names the search target
                # ('me' -> 'person') instead of echoing the raw input.
                label = known_labels.resolve_alias(label)
                ss = getattr(self.scene_graph, "self_state", None)
                # record the attempt before any refusal below, otherwise a
                # prepare-time refusal never reaches the self-state log.
                if ss is not None and hasattr(ss, "note_pick_attempt"):
                    ss.note_pick_attempt(label)
                det = self._find_target_detection(label)
                _find_obs = self._take_find_obs()
                # order matters: this guard runs after the lookup so a held label
                # can return its own NOT_DETECTED wording.
                if ss is not None and ss.pick_blocks():
                    if det is None and ss.holds(label):
                        return dict(args), fail(_not_detected_held(label, ss))
                    return dict(args), fail(ss.pick_refusal(label))
                # walk-only and post-place refusals run before NOT_DETECTED: the
                # task context decides whether this pick is wanted at all.
                if ss is not None and hasattr(ss, "walk_only_refusal"):
                    _msg = ss.walk_only_refusal(label)
                    if _msg:
                        return dict(args), fail(_msg)
                if ss is not None and hasattr(ss, "post_place_refusal"):
                    _msg = ss.post_place_refusal(label)
                    if _msg:
                        return dict(args), fail(_msg)
                # a pick whose outcome is unknown blocks the re-pick. its freshest
                # detection may be the object resting in the parked hand, while
                # the grip observation still reads empty.
                if (ss is not None and det is not None
                        and hasattr(ss, "repick_refusal")):
                    _grip = getattr(_find_obs, "grip", None)
                    if _grip is not None:
                        _msg = ss.repick_refusal(label, grip=_grip)
                    else:
                        _msg = ss.repick_refusal(label)
                    if _msg:
                        return dict(args), fail(_msg)
                # after a cancel only newer detections count; an older row may
                # show an object that has since moved.
                if (ss is not None and det is not None
                        and hasattr(ss, "stale_after_cancel")):
                    _msg = ss.stale_after_cancel(det)
                    if _msg:
                        return dict(args), fail(_msg)
                # a release may be unconfirmed or the hand state unknown, but an
                # absent label that might be held must not send the robot off to
                # navigate. only applies after a pick.
                if (ss is not None and det is None
                        and hasattr(ss, "held_hint_due")
                        and ss.held_hint_due(label)):
                    return dict(args), fail(_not_detected_held(label, ss))
                # absent and unsure: probably held, so neither navigate nor
                # open_hand is suggested, for every pick inside the ambiguity window.
                if (ss is not None and det is None
                        and hasattr(ss, "unconfirmed_absent")
                        and ss.unconfirmed_absent(label)):
                    # prefer the ambiguity note when one is pending
                    return dict(args), fail(self._take_find_note()
                                            or _not_detected_unconfirmed(label, ss))
                if det is None:  # prefer the ambiguity note when one is pending
                    return dict(args), fail(self._take_find_note() or _not_detected(label, "pick it again"))
                # refuse classes beyond the grasp vocabulary
                base_class = detection_base_class(det)
                if base_class.lower() in NOT_PICKABLE_CLASSES:
                    return dict(args), fail(
                        f"pick refused: {base_class} is not an object this robot "
                        f"can manipulate (not in the grasping vocabulary). "
                        f"Name a graspable object instead.")
                augmented = dict(args)
                augmented.setdefault("x", det.x)
                augmented.setdefault("y", det.y)
                augmented.setdefault("z", det.z)
                if det.ref:
                    augmented.setdefault("ref", det.ref)
                requested = str(args.get("orientation") or "auto").strip().lower()
                if requested not in ("auto", "upright", "horizontal"):
                    requested = "auto"
                det_ori = str(getattr(det, "orientation", "") or "upright").lower()
                if det_ori not in ("upright", "horizontal"):
                    det_ori = "upright"
                augmented["orientation"] = det_ori if requested == "auto" else requested
                # carry the detection's torso-frame 5-95 percentile z span as
                # height_m so the engine records the pick height and the place lift
                # has a value. setdefault keeps a caller value; None means unknown.
                augmented.setdefault("height_m", getattr(det, "height_m", None))
                # send the detector label downstream: scene names like 'green_cup_2'
                # never appear in detections and would make the gateway refuse a fresh
                # target as stale. the scene name travels as ref.
                _cls = self._walk_label(label)
                if _cls != label:
                    augmented["label"] = _cls
                    augmented.setdefault("ref", label)
                return augmented, None

            def run(args: dict) -> dict:
                label = (args.get("label") or "").strip()
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot execute pick")
                x, y, z, refusal = self._constrained_pick_point(
                    args.get("x"), args.get("y"), args.get("z"), label
                )
                if refusal is not None:
                    return refusal
                self._home_fetch_confirm()          # promote a fetch start when its words arrive after the walk

                def _send(px, py, pz):
                    return backend.pick(
                    {
                        "label": label,
                        "ref": args.get("ref"),
                        "x": px,
                        "y": py,
                        "z": pz,
                        "orientation": args.get("orientation", "auto"),
                        "arm": args.get("arm", "auto"),
                        # carried through from prepare: the detection's own height. the
                        # gateway gates it and the engine records it so pick and place agree.
                        "height_m": args.get("height_m"),
                    }
                    )

                env = _send(x, y, z)
                # cross-body retry: when the refusal left the arm unmoved, turn so
                # the object sits right of centre and pick once more.
                _aim = self._cross_body_aim(env, label, x, y)
                if isinstance(_aim, tuple):
                    ax, ay, az = _aim
                    if not math.isfinite(az):
                        az = z
                    x2, y2, z2, refusal = self._constrained_pick_point(
                        ax, ay, az, label)
                    if refusal is not None:
                        return refusal
                    x, y, z = x2, y2, z2
                    env = _send(x, y, z)
                elif isinstance(_aim, str) and isinstance(env, dict) \
                        and isinstance(env.get("reason"), str):
                    env = dict(env)
                    env["reason"] = env["reason"].rstrip() + "; AUTO_AIM=" + _aim
                # a fetch reached its object: tell the home sequence the pick
                # succeeded. spoken words can pause a fetch but never end it.
                if (isinstance(env, dict) and env.get("success")
                        and self.__dict__.get("_home_seq") is not None):
                    self._home_call({"op": "picked"})
                # tag a DONE reply with the commanded pick point so verification can
                # tell an object in the parked hand from one on the table. the
                # engine reply itself stays untouched.
                try:
                    value = env.get("value") if isinstance(env, dict) else None
                    if (isinstance(env, dict) and env.get("success")
                            and isinstance(value, str)
                            and value.lstrip().startswith("DONE")
                            and "pick_point=" not in value):
                        px, py, pz = float(x), float(y), float(z)
                        if all(math.isfinite(v) for v in (px, py, pz)):
                            env = dict(env)
                            env["value"] = "%s; pick_point=%.3f,%.3f,%.3f" % (
                                value, px, py, pz)
                except (TypeError, ValueError):
                    pass
                # out of reach from here: tell the model to navigate closer and pick
                # again. model-facing only: the spoken failure is a fixed sentence,
                # so this text never reaches the speaker.
                try:
                    reason = env.get("reason") if isinstance(env, dict) else None
                    if (isinstance(env, dict) and not env.get("success")
                            and isinstance(reason, str)
                            and _REACH_CEILING_RE.search(reason)
                            and _REACH_HINT_MARK not in reason):
                        env = dict(env)
                        env["reason"] = reason.rstrip() + _reach_ceiling_hint(
                            label)
                except (TypeError, ValueError):
                    pass
                # a self-collision hint tells the model not to navigate
                env = _annotate_cross_body(env, label)
                # a pick that completes a fetch names the next step on the reply so
                # the model continues instead of stopping after the pick.
                try:
                    value = env.get("value") if isinstance(env, dict) else None
                    if (isinstance(env, dict) and env.get("success")
                            and isinstance(value, str)
                            and value.lstrip().startswith("DONE")
                            and self._fetch_filter_live()
                            and "NEXT_STEP=" not in value):
                        env = dict(env)
                        env["value"] = value + _FETCH_NEXT_STEP
                except (TypeError, ValueError):
                    pass
                return env

            return Tool(
                name="pick",
                description=(
                    "Pick up an object resting on a surface the robot can reach — a "
                    "table, a chair seat, a low shelf (torso-frame z >= -0.35 m; z=0 is "
                    "the torso link, so a NEGATIVE z is normal for a low surface and does "
                    "not mean the object is off the table). The robot CANNOT pick "
                    "floor/ground objects (they read z ~ -0.75) — no ground-pick skill "
                    "exists, so it refuses honestly. If the object is not in view, or its "
                    "scene line says OUT-OF-REACH, call navigate first, then pick. The "
                    "tool resolves the label against the "
                    "freshest detections for torso-frame coordinates."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Object label from CURRENT SCENE (e.g. "
                            "'bottle'). Say it the way the human said it — the "
                            "lookup matches articles, case and the bare class "
                            "('cup' finds a 'red cup' row). It can ONLY match a "
                            "class perception is able to report; scene_brief "
                            "lists them.",
                        },
                        "orientation": {
                            "type": "string",
                            "enum": ["auto", "upright", "horizontal"],
                            "description": "Grasp orientation: 'upright' for a "
                            "vertical object, 'horizontal' for a lying object. "
                            "'auto' (default) binds to the detected object's own "
                            "orientation so a lying object is picked with the "
                            "horizontal approach, never the vertical one.",
                        },
                        "arm": {
                            "type": "string",
                            "enum": ["auto", "right", "left"],
                            "description": (
                                "Arm to pick with; 'auto' picks the reachable side."
                                if _left_in_service_at_reg else
                                "Arm to pick with: 'auto' (the right arm) or "
                                "'right'. The LEFT hand is out of service and "
                                "'left' is refused."),
                        },
                    },
                    "required": ["label"],
                },
                run=run,
                prepare=prepare,
                timeout_s=CONTRACT_TIMEOUTS["pick"],
            )

        def make_place() -> Tool:
            def _prepare_core(args: dict) -> tuple[dict, Optional[dict]]:
                """Resolve the optional into:<container_ref> argument against
                the freshest detections — the same resolution the pick label
                gets (``_find_target_detection`` + scene_graph.resolve_ref),
                so an unknown or unresolvable container fails with the SAME
                honest NOT_DETECTED phrasing unknown args get today.  No
                into: argument -> the args pass through untouched here
                (prepare adds nothing; the run layer's optional drop-point
                resolver may still attach a destination-derived x/y/z).
                """
                _ss = getattr(self.scene_graph, "self_state", None)
                _msg = _ss.place_refusal() if _ss is not None else None
                if _msg:
                    return dict(args), fail(_msg)
                augmented = dict(args)
                into = (args.get("into") or "").strip()
                det = None
                # the container the robot navigated to: its instance id survives a
                # label change, so it wins over the label in the args.
                _lock_rec = self._container_lock_record(args)
                if _lock_rec is not None:
                    det = self._lock_detection(_lock_rec)
                    into = str(_lock_rec["label"])
                    augmented["into"] = into
                elif not into:
                    # a bare place whose destination names an object uses that object:
                    # models often put the container in 'destination' and leave 'into'
                    # empty.
                    into, refusal = self._place_destination_target(args.get("destination"))
                    if refusal is not None:
                        # a destination naming nothing in view may be rescued by the
                        # operator's words; a beside relation never can.
                        alt = None
                        if "BESIDE" not in str(refusal.get("reason") or refusal.get("error") or ""):
                            alt = self._place_operator_target()
                        if alt is None:
                            return dict(args), refusal
                        into = alt
                    elif not into:
                        # no into and no destination: the operator's words may name the
                        # target ('put it in the bowl'), otherwise release in place.
                        into = self._place_operator_target() or ""
                        if not into:
                            # an ambiguous target that is in view with low confidence: report
                            # it rather than dropping the object.
                            _onote = self._take_operator_note()
                            if _onote:
                                return dict(args), fail(_onote)
                            return augmented, None
                    augmented["into"] = into
                else:
                    # an into label that will not resolve falls back to the
                    # destination, then the operator's target, then the first refusal.
                    # keep the first refusal for the failure path
                    det = self._find_target_detection(into)
                    if det is None:
                        first_refusal = fail(self._take_find_note("use it as the container") or _not_detected(into, "place it into it (keep holding the object meanwhile)"))  # keep the same reason wording as the pick lookup
                        alt, _alt_refusal = self._place_destination_target(args.get("destination"))
                        if (_alt_refusal is not None and "BESIDE" in str(
                                _alt_refusal.get("reason") or _alt_refusal.get("error") or "")):
                            # a beside relation is never promoted into an into target: the
                            # operator's words cannot turn 'next to' into 'inside'.
                            return dict(args), _alt_refusal
                        if not alt:
                            alt = self._place_operator_target()
                        if not alt:
                            return dict(args), first_refusal
                        into = alt
                        augmented["into"] = into
                if det is None:
                    det = self._find_target_detection(into)
                if det is None:  # prefer the ambiguity note when one is pending
                    return dict(args), fail(self._take_find_note("use it as the container") or _not_detected(into, "place it into it (keep holding the object meanwhile)"))  # reuse the same reason wording
                # a detection above the torso workspace cannot rest on a surface:
                # it is the held object or the hand mislabelled, so refuse.
                try:
                    _cz = float(getattr(det, "z", float("nan")))
                except (TypeError, ValueError):
                    _cz = float("nan")
                if math.isfinite(_cz) and _cz > CONTAINER_Z_MAX_M:
                    return dict(args), fail(
                        "NOT_DETECTED: the only %r detection sits at z %+.2f m, "
                        "above the arm's workspace (containers rest on a surface "
                        "below %+.2f m): it is the held object or the hand "
                        "mislabelled, not a container; the object is still held. "
                        "Call navigate with this label if the real %s is out of "
                        "view, then place again" % (into, _cz, CONTAINER_Z_MAX_M,
                                                    into))
                label = (det.label or "").strip()
                if not label:
                    return dict(args), fail(
                        f"NOT_DETECTED: {into!r} resolved to a detection "
                        "without a label — cannot name the container"
                    )
                ref = None
                if self.scene_graph is not None:
                    try:
                        ref = self.scene_graph.resolve_ref(into)
                    except Exception:  # noqa: BLE001
                        ref = None
                # container identity: the scene-graph ref when available, else the
                # label, which is what the verifier matches on.
                augmented["into_ref"] = ref if ref is not None else label
                augmented["into_label"] = label
                # record a fresh last-seen position so the verifier can tell
                # containers apart; it is used for disambiguation only, never for
                # the offsets.
                if ref is not None and self.scene_graph is not None:
                    try:
                        entry = self.scene_graph.refs.get(ref)
                        if (
                            entry is not None
                            and isinstance(entry, dict)
                            and (time.time() - entry.get("last_seen", 0.0))
                            <= CONTAINER_REF_FRESH_S
                        ):
                            for axis in ("x", "y", "z"):
                                v = entry.get(axis)
                                if isinstance(v, (int, float)) and not isinstance(
                                    v, bool
                                ):
                                    augmented["into_" + axis] = float(v)
                    except Exception:  # noqa: BLE001
                        pass
                # an into place commands a drop point above the resolved container;
                # without x/y/z the engine releases where it stands, next to the
                # target. a caller's x/y/z wins.
                # a place into a container aims just past the near rim inside the
                # reach law, creeping first when needed, else at the centre.
                _plan, _prefusal = self._container_plan(det, _lock_rec, args)
                if _prefusal is not None:
                    return dict(args), _prefusal
                if _plan is not None:
                    for _axis in ("x", "y", "z"):
                        augmented[_axis] = _plan["aim"][_axis]
                    augmented["_container_plan"] = _plan
                    return augmented, None
                _hint = getattr(self, "_low_lane_drop", None)   # handed over by the low lane
                self._low_lane_drop = None
                drop = _hint[1] if (_hint and _hint[0] == id(det) and _hint[1]) \
                    else _drop_point(det)
                if drop is None:
                    # no usable position: refuse instead of releasing in the wrong
                    # place. refusing leaves the object in the hand, which is recoverable.
                    return dict(args), fail(
                        f"NOT_DETECTED: {into!r} has no usable position "
                        "(x/y/z absent or non-finite) — cannot place onto it"
                    )
                # reach gate: refuse a drop point beyond reach instead of releasing
                # at the arm's limit, and name the way out.
                _pd = math.hypot(float(drop["x"]), float(drop["y"]))
                _pr = reach_residual(float(drop["x"]), float(drop["y"]),
                                     min(float(drop["z"]), AIM_Z_MAX))
                _low = LOW_BAND_Z_MIN_M <= float(drop["z"]) <= LOW_BAND_Z_MAX_M
                # handover: a person within HANDOVER_START_MAX_M is approached to
                # arm's reach, then the hand opens where it is with no arm motion.
                _ho = self._handover_plan(into, det, _pd)
                if _ho is not None:
                    augmented["_handover"] = _ho
                    for _axis in ("x", "y", "z"):
                        augmented.setdefault(_axis, drop[_axis])
                    return augmented, None
                if _pd > PLACE_ON_MAX_PLANAR_M or (_pr > PLACE_ON_MAX_RESIDUAL_M
                                                   and not _low):
                    return dict(args), fail(
                        "NOT_REACHABLE: the %s is %.2f m away (residual %.2f m "
                        "beyond the arm's reach) - too far to place onto from "
                        "here. This is NOT the end of the task: call navigate "
                        "with this label to walk to it, then place again; the "
                        "object is still held" % (into, _pd, _pr))
                for _axis in ("x", "y", "z"):
                    augmented.setdefault(_axis, drop[_axis])
                return augmented, None

            def prepare(args: dict) -> tuple[dict, Optional[dict]]:
                """B8i: _prepare_core, then (success only) the held object's
                class as 'label' and, for a coordinate place without a
                caller height, the place-only lift height token."""
                aug, refusal = _prepare_core(args)
                if refusal is not None:
                    return aug, refusal
                _pl = self._place_object_label(aug.get("label"))
                if isinstance(_pl, str) and _pl:
                    aug["label"] = _pl
                if (all(aug.get(_k) is not None for _k in ("x", "y", "z"))
                        and aug.get("height_m") is None):
                    aug["height_m"] = self._place_lift_height(aug.get("label"))
                return aug, None

            def run(args: dict) -> dict:
                # place may carry an optional x/y/z as a commanded target, or fall
                # back to a release at the current pose. no envelope key constrains
                # it: max_reach_m and grip_center_standoff_m bound pick only.
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot execute place")
                # creep to the container first when prepare planned it, then
                # recompute the drop at the new position.
                _plan = args.get("_container_plan")
                _creep_note, _travel = None, 0.0
                if isinstance(_plan, dict) and _plan.get("creep") is not None:
                    _cr = self._run_container_creep(_plan)
                    if isinstance(_cr, dict):
                        return _cr
                    _aim, _creep_note, _travel, _geo = _cr
                    args = dict(args)
                    for _axis in ("x", "y", "z"):
                        args[_axis] = _aim[_axis]
                    _plan = dict(_plan, aim=_aim, geo=_geo)
                # close in on the person first, then recompute the drop point
                _ho = args.get("_handover")
                if isinstance(_ho, dict):
                    _hr = self._run_handover_approach(_ho)
                    if isinstance(_hr, dict):
                        return _hr
                    return self._handover_open(_hr)     # open at arm's reach, no drop motion
                place_payload = {
                    "label": args.get("label"),
                    "destination": args.get("destination"),
                }
                # explicit x/y/z go through as given; the backend coerces and
                # guards them. an into place always carries one, so the arm moves
                # over the container before releasing and the verifier re-observes.
                caller_xyz = {
                    key: args[key]
                    for key in ("x", "y", "z")
                    if args.get(key) is not None
                }
                if caller_xyz:
                    place_payload.update(caller_xyz)
                # forward the engine's place-time 2.5x lift. it is accepted only
                # with coordinates (the gateway parses 'place x y z height_m'); a
                # lone token is a parse error, and the gateway re-gates the value.
                if caller_xyz and args.get("height_m") is not None:
                    place_payload["height_m"] = args.get("height_m")
                env = backend.place(place_payload)
                # the object left the hand: clear the fetch sequence unless the
                # plan changed to a handover.
                if (isinstance(env, dict) and env.get("success")
                        and self.__dict__.get("_home_seq") is not None):
                    self._home_call({"op": "clear", "why": "handover"})
                    self._home_seq = None
                # tag a successful into reply with the container ref and label so
                # the verifier can find it. brain-side evidence only: into_* never
                # rides the wire.
                into_ref = args.get("into_ref")
                into_label = args.get("into_label")
                if (
                    isinstance(env, dict)
                    and env.get("success")
                    and into_ref
                    and into_label
                ):
                    value = env.get("value")
                    label = (args.get("label") or "").strip()
                    tail = "; into=%s; into_label=%s" % (
                        _annotation_value(into_ref),
                        _annotation_value(into_label))
                    # a missing object label leaves containment un-checkable, so the
                    # verifier treats it as in progress.
                    if label:
                        tail += "; placed_object=%s" % _annotation_value(label)
                    else:
                        tail += "; placed_object=(label missing)"
                    # container position from prepare, when fresh
                    for axis in ("x", "y", "z"):
                        v = args.get("into_" + axis)
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            if axis == "y":
                                v = v - _travel    # adjust y for the creep travel
                            tail += "; into_%s=%.3f" % (axis, v)
                    if isinstance(_plan, dict):
                        tail += self._container_tail(_plan, _creep_note)
                    # execution timestamp on time.time(), the base the gateway and the
                    # verifier share; identity only, never age math. predrop_* carry the
                    # pre-drop in-region snapshot that the verifier cannot see.
                    tail += "; dropped_at=%.6f" % time.time()
                    snap = args.get("_predrop_containment")
                    if isinstance(snap, dict) and snap.get("state") == "known":
                        tail += "; predrop_state=known"
                        positions = snap.get("preexisting") or []
                        if positions:
                            tail += "; predrop_obj=" + ";".join(
                                "%.3f,%.3f,%.3f"
                                % (float(p[0]), float(p[1]) - _travel,
                                   float(p[2]))
                                for p in positions)
                    else:
                        tail += "; predrop_state=unknown"
                    if isinstance(value, str) and value:
                        return ok(value + tail)
                    # an empty drop reply is redacted so the verifier reads in progress,
                    # never a false DONE.
                    return ok("(drop reply empty)" + tail)
                # a cross-body place refusal stays bare: place reasons are spoken
                # verbatim, so a hint would be read aloud. the model sees it
                # through self-state instead.
                return env

            return Tool(
                name="place",
                description=(
                    "Release the held object by opening the gripper (engine 'place': "
                    "hold_rc on -> LOW_CARRY -> open -> payload off -> NEUTRAL -> "
                    "hold_rc off). Without coordinates this is RELEASE-ONLY: the "
                    "hand opens at the current carry pose.  "
                    "WHEN THE OPERATOR NAMES AN OBJECT TO PUT IT ON OR IN ('in the "
                    "bowl', 'on the green cup'), PASS THAT OBJECT AS 'into' — "
                    "without it the hand just opens where the arm is.  "
                    "GIVING IT TO A PERSON ('give it to me', 'hand it over'): "
                    "into='person' -- you turn to face them, step in to arm's "
                    "reach and OPEN the hand for them to take it (no drop "
                    "motion); say 'here you go' as you do.  "
                    "Verifier v2: pass an optional 'into' argument (a container "
                    "named in CURRENT SCENE) — the container resolves "
                    "like the pick label (unknown container = NOT_DETECTED), "
                    "the arm is then commanded to a drop point above it so the "
                    "object is released ONTO the container, and the verifier "
                    "only succeeds on demonstrated containment by "
                    "re-observation."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "The object you are holding.",
                        },
                        "destination": {
                            "type": "string",
                            "description": "Where you intend to put it (informational).",
                        },
                        "into": {
                            "type": "string",
                            "description": "Container to place the object INTO "
                            "(a name from CURRENT SCENE); resolved like the "
                            "pick label and gates verifier v2 containment.",
                        },
                        "height_m": {
                            "type": "number",
                            "description": "The held object's height in metres, "
                            "for the engine's place-time extra lift (it raises "
                            "the holding arm by 2.5x this, clamped, before the "
                            "place proceeds). Only used WITH x/y/z. Omit it and "
                            "the pick-time height of the last pick is used.",
                        },
                    },
                    "required": ["label", "destination"],
                },
                run=run,
                prepare=prepare,
                timeout_s=CONTRACT_TIMEOUTS["place"],
            )

        def make_place_at_spot() -> Tool:
            """Release the held object (VALIDATED against a named spot).

            Consumes ``embodiment.base_relative.key_positions`` (round-5
            config-seam) for VALIDATION only: the requested spot name is
            resolved against the CONFIGURED key_positions map — an unknown
            spot name is an honest UNKNOWN_SPOT tool error and an unwired
            embodiment is an honest refusal (never a guessed position).
            Placement itself is a RELEASE at the CURRENT pose (bare
            'place', target-agnostic — the same backend.place path as the
            place tool): the spot does NOT move the arm, no positional
            guarantee is made or claimed, and the resolved position is
            never sent anywhere — the platform has no spot-relative
            placement.
            """

            def prepare(args: dict) -> tuple[dict, Optional[dict]]:
                _ss = getattr(self.scene_graph, "self_state", None)
                _msg = _ss.place_refusal() if _ss is not None else None
                if _msg:
                    return dict(args), fail(_msg)
                return dict(args), None

            def run(args: dict) -> dict:
                spot = (args.get("spot") or "").strip()
                if not spot:
                    return fail("invalid arguments: missing spot")
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot execute place_at_spot")
                _, refusal = self._spot_position(spot)
                if refusal is not None:
                    return refusal
                return backend.place(
                    {
                        "label": args.get("label"),
                        "destination": spot,
                        "spot": spot,
                    }
                )

            return Tool(
                name="place_at_spot",
                description=(
                    "Release the held object (RELEASE-ONLY: engine 'place' opens "
                    "the gripper at the CURRENT pose — no positional guarantee, "
                    "the drop does not move the arm).  The named spot from "
                    "embodiment.base_relative.key_positions is VALIDATED for "
                    "capability/consistency checking only: an unknown spot name "
                    "is an honest UNKNOWN_SPOT error and an unwired embodiment "
                    "an honest refusal — the robot never guesses a position, and "
                    "the resolved position is NOT commanded anywhere (the "
                    "platform has no spot-relative placement; release happens "
                    "where the robot stands)."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "spot": {
                            "type": "string",
                            "description": "Named base-relative spot from "
                            "embodiment.base_relative.key_positions.",
                        },
                        "label": {
                            "type": "string",
                            "description": "The object you are holding.",
                        },
                    },
                    "required": ["spot"],
                },
                run=run,
                prepare=prepare,
                timeout_s=CONTRACT_TIMEOUTS["place"],
            )

        def make_navigate() -> Tool:
            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot navigate")
                # resolve the alias before the target leaves the tool: the FSM matches
                # by label and sweeps a class perception can publish, so 'bring it to
                # me' would otherwise hunt nothing.
                target = known_labels.resolve_alias(args.get("target"))
                refusal = self._navigate_vocab(target)          # add the word to the detector vocabulary first
                if refusal is not None:
                    return refusal
                # a fetch walk targets the requester only
                spec, refusal = self._requester_walk_target(target)
                if refusal is not None:
                    return refusal
                # the first empty-handed walk marks the fetch start
                self._home_fetch_mark(target)
                target = spec or self._walk_target_spec(target)
                # not in view: step back, sweep
                refusal = self._search_step_back(spec, target)
                if refusal is not None:
                    return refusal
                return backend.navigate({"target": target})

            return Tool(
                name="navigate",
                description=(
                    "SEARCH for a destination or object and walk to it using the "
                    "walk FSM (/walk_target, /walk_go; status on /walk_status). "
                    "It SWEEPS when the target is not in view, so call it for "
                    "anything you cannot see or cannot reach — being out of sight "
                    "is not being unavailable. Verification is ONLY via "
                    "/walk_status == REACHED — there is no odometry; a REACHED "
                    "target is the go-ahead to pick it. Call it yourself whenever "
                    "the target is out of view or out of reach — never ask the "
                    "human for permission to walk first."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Destination or object name. Name it the "
                            "way the human did — the label is matched against the "
                            "scene, not spelled exactly ('the bottle' is fine). "
                            "The walk FSM can only search for a class perception "
                            "is able to report; scene_brief lists them.",
                        },
                    },
                    "required": ["target"],
                },
                run=run,
                timeout_s=CONTRACT_TIMEOUTS["navigate"],
            )

        def make_turn_around() -> Tool:
            """Turn in place and sweep for a person, holding on first sight
            (operator directive 2026-10-03: *"just the simple turn around
            sweep when i ask it to turn around, it will turn around to look
            for person, once detected person, it stopped sweeping"*).

            The sweep primitive ALREADY EXISTS: navigate's walk FSM sweeps
            and interrupts on the first sighting (walk_node._sweep_turn ->
            _phase_hold: DETECTED, no turn, no walk).  What this tool must
            NOT do is reach that FSM through the coordinator, whose
            auto-approach (/walk_go on the 3rd detection beat) would walk the
            robot TO the person.  The design of record for the gateway-side
            verb is therefore to start the sweep by publishing the label
            directly on /walk_target, bypassing /navigate_target and the
            coordinator entirely, so the FSM reaches its DETECTED hold and
            stays there; this brain half only names the target.

            This is MOTION (hooks.MOTION_TOOLS carries 'turn_around'), so it
            takes the same confirm gate as pick/place/navigate — turning in
            place drives the walk FSM like any other locomotion.
            """

            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot turn around")
                method = getattr(backend, "turn_around", None)
                if method is None:
                    return fail("backend has no turn_around — cannot turn around")
                # same alias law: 'me' -> 'person' before it leaves the tool, and no
                # target means the operator.
                target = known_labels.resolve_alias(args.get("target") or "person")
                return method({"target": target})

            return Tool(
                name="turn_around",
                description=(
                    "TURN IN PLACE and sweep the room for a person; STOP "
                    "sweeping when one is seen, then rotate in place to "
                    "CENTRE them in the camera frame — it does NOT walk to "
                    "them. The call BLOCKS for up to about 2 minutes. Use "
                    "this when the human asks to be found or to look around: "
                    "'turn around', 'turnaround', 'look around', 'spin "
                    "around', 'find me', 'who's there'. This is NOT navigate: "
                    "navigate walks to a target, this only turns, watches and "
                    "centres. Read TURN_RESULT in the reply: centred = done, "
                    "the person is centred in front of you; not_seen = say "
                    "you swept the full circle and saw no person (never claim "
                    "you found someone); seen_not_centred or cancelled = "
                    "report exactly that, and retry at most once. Call it "
                    "yourself — never ask the human for permission first."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Who to look for; defaults to "
                            "'person'. 'me', 'myself' and 'I' mean the "
                            "person speaking.",
                        },
                    },
                    "required": [],
                },
                run=run,
                # the sweep takes as long as a navigate (12 windows of 5.0 s plus a
                # ~1.05 s turn each, ~73 s), so reuse that timeout.
                timeout_s=CONTRACT_TIMEOUTS["navigate"],
            )

        def make_return_home() -> Tool:
            """B8l F7 (operator 2026-10-17): "hey stuart, can you go get me a
            bottle of water?" -> navigate to the bottle, pick it, RETURN to
            where the walk started, give it to the person near that spot.
            The start point is recorded by the gateway (odometry) when a
            navigate starts with an empty hand."""

            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot return")
                method = getattr(backend, "return_home", None)
                if method is None:
                    return fail("backend has no return_home — cannot return")
                return method({})

            return Tool(
                name="return_home",
                description=(
                    "WALK BACK to where you were when the person asked you to "
                    "fetch something (the start point, remembered by "
                    "odometry when the first walk of that fetch began with "
                    "an empty hand), and turn to face the person who asked. Use it "
                    "for every fetch ('get me X', 'bring me X', 'can you go "
                    "get me X'): navigate to X, pick X, call return_home, then "
                    "place with into='person' — only a person near the start "
                    "point counts as the one who asked; anyone far from it is "
                    "ignored. If the person is out of reach after "
                    "return_home, navigate to 'person' (it walks to the "
                    "requester only), then place. It BLOCKS until arrival. "
                    "ERROR replies say why (no start point recorded, "
                    "odometry stale, blocked, cancelled) — report that "
                    "honestly; never claim you are back when it failed."
                ),
                schema={"type": "object", "properties": {}, "required": []},
                run=run,
                timeout_s=140.0,
            )

        def make_cancel() -> Tool:
            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot cancel")
                method = getattr(backend, "stop", None)
                if method is None:
                    return fail("backend has no stop — cannot cancel")
                return method({})

            return Tool(
                name="stop",
                description=(
                    "CANCEL the task that is running right now: stop picking "
                    "while it is picking, stop placing while it is placing, "
                    "stop sweeping/searching while it is searching. The arm "
                    "freezes in place and the operator regains manual "
                    "control; the walk stops. Call this IMMEDIATELY when the "
                    "human says stop/cancel/halt/freeze/never mind/forget it "
                    "— do not finish the current step first, and do not ask "
                    "for confirmation. Stopping is always allowed and is "
                    "never a destructive action."
                ),
                schema={"type": "object", "properties": {}},
                run=run,
                # the cancel answer comes back on the engine's connection thread, so
                # a healthy one is instant. a long bound would make the stop itself
                # wait on a wedged listener.
                timeout_s=CANCEL_TIMEOUT_S,
                # non-blocking: a cancel must not queue behind the task it cancels
                blocking=False,
            )

        def make_hand(name: str, verb: str) -> Tool:
            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail(f"gateway not wired — cannot {verb}")
                method = getattr(backend, name, None)
                if method is None:
                    return fail(f"backend has no {name} — cannot {verb}")
                return method({"side": args.get("side", "right")})

            if name == "open_hand":
                description = (
                    "Open the gripper (grip socket open|open_left|open_both). "
                    "Verify afterwards via grip 'state <side>' — the grip does "
                    "NOT auto-verify. "
                    # opening the hand drops a good pick, so it is offered only as an
                    # explicit release, never as a check.
                    "Opening the hand DROPS whatever it holds, where the arm "
                    "is now. Call it ONLY when the operator asks to open the "
                    "hand, let go or drop; never to check, correct or retry a "
                    "pick - if you are unsure the right object was picked, "
                    "say so and ask the operator."
                )
            else:
                description = (
                    "Close the gripper (grip socket close|close_left|close_both). "
                    "For grasping the gateway should use close_adaptive "
                    "(DONE:contact = the real grasp signal). Verify afterwards via "
                    "grip 'state <side>' — the grip does NOT auto-verify."
                )
            def prepare(args: dict) -> tuple[dict, Optional[dict]]:
                """B8h (s1 rows 9 and 13, MEASURED): the description guard
                above failed twice - the model opened the hand on a VERIFIED
                pick nobody asked it to release.  Code guard: while a
                verified pick is held, open_hand needs a release request from
                the operator in this turn or the previous one
                (SelfState.open_hand_refusal; inert for hand_unknown,
                pick_unconfirmed and without the voice hook)."""
                ss = getattr(self.scene_graph, "self_state", None)
                if ss is not None and hasattr(ss, "open_hand_refusal"):
                    msg = ss.open_hand_refusal(args.get("side"))
                    if msg:
                        return dict(args), fail(msg)
                return dict(args), None

            return Tool(
                name=name,
                description=description,
                schema={
                    "type": "object",
                    "properties": {
                        "side": {
                            "type": "string",
                            "enum": ["right", "left", "both"],
                            "description": "Which hand (default right).",
                        },
                    },
                    "required": [],
                },
                run=run,
                prepare=prepare if name == "open_hand" else None,
                timeout_s=CONTRACT_TIMEOUTS[name],
            )

        def make_pose() -> Tool:
            """Move the arms to a named resting pose (operator directive
            2026-09-26: *"i can ask it to go to neutral pose, tuck in pose
            and ready pose ... tuck your arms in, be neutral, go ready
            pose"*).

            ONE tool with an enum rather than four tools, because the human
            names a POSE and the four names differ only by which engine verb
            they reach.  The enum members are OUR names; the engine's own
            verbs differ (`tuck` needs TWO of them) and that mapping is the
            backend's job, not the model's — see POSE_ENGINE_VERBS in
            gateway/brain_client.py.

            This is MOTION (hooks.MOTION_TOOLS carries 'pose'), so it passes
            the same confirm gate as pick/place/navigate.
            """

            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail("gateway not wired — cannot pose")
                method = getattr(backend, "pose", None)
                if method is None:
                    return fail("backend has no pose — cannot pose")
                name = str(args.get("name") or "").strip().lower()
                if name not in POSE_NAMES:
                    return fail(
                        f"unknown pose {name!r}; valid: "
                        + ", ".join(sorted(POSE_NAMES))
                    )
                return method({"name": name})

            return Tool(
                name="pose",
                description=(
                    "Move the robot's arms to a named resting pose. Use this "
                    "whenever the human asks for a posture rather than an "
                    "object: 'tuck your arms in', 'go to neutral', 'ready "
                    "pose', 'put your arms down', 'arms down at your sides', "
                    "'go home'.\n"
                    "'ready' — the working pose, both arms tucked and BOTH "
                    "HANDS OPEN (the pose the robot starts a task from).\n"
                    "'tuck' — both arms folded in to the carry pose; hands "
                    "are left as they are (a carried object stays held).\n"
                    "'neutral' — the lowstate rest position: arms hanging "
                    "with the elbows softened, both hands CLOSED. A working "
                    "rest, NOT a full arms-down.\n"
                    "'home' — THE ARMS-DOWN POSE: both arms hanging STRAIGHT "
                    "DOWN at the sides of the torso, elbows nearly straight, "
                    "palms turned inboard. Say this one for 'put your arms "
                    "down', 'arms at your sides', 'lower your arms'.\n"
                    "'ready' and 'tuck' are DIFFERENT poses: 'get ready' is "
                    "'ready', 'tuck your arms in' is 'tuck'. 'arms down' is "
                    "'home', NOT 'neutral'."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": sorted(POSE_NAMES),
                            "description": "Which pose to move to.",
                        },
                    },
                    "required": ["name"],
                },
                run=run,
                timeout_s=CONTRACT_TIMEOUTS["pose"],
            )

        def make_resolve_ref() -> Tool:
            def run(args: dict) -> dict:
                graph = self.scene_graph
                label = (args.get("label") or "").strip()
                if graph is None:
                    return fail("scene graph not wired — cannot resolve_ref")
                ref = graph.resolve_ref(label)
                if ref is None:
                    return fail(
                        f"NOT_DETECTED: no scene reference for {label!r}"
                    )
                return ok(ref)

            return Tool(
                name="resolve_ref",
                description=(
                    "Map a label from CURRENT SCENE to its persistent object "
                    "reference (e.g. 'bottle' -> 'bottle_3'). Use it when the same "
                    "object must be tracked across steps. Honest error when the "
                    "label is not in the current scene."
                ),
                schema={
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "description": "Label as shown in CURRENT SCENE.",
                        },
                    },
                    "required": ["label"],
                },
                run=run,
                timeout_s=10.0,
            )

        def make_scene_brief() -> Tool:
            def run(args: dict) -> dict:
                graph = self.scene_graph
                if graph is None:
                    return fail("scene graph not wired — cannot scene_brief")
                # live re-check, not a cache read: the description promises a fresh
                # look, and cached refs would report an empty scene once the lane
                # died. update() only takes fresh observations, so health is judged here.
                lane_stale_s = None
                if self.backend is not None:
                    obs = self._observe()
                    if obs is None:
                        return fail(
                            "scene unavailable: the perception/gateway lane is "
                            "not answering (observe() returned no observation). "
                            "This is NOT an empty scene — nothing is currently "
                            "known about the world. Do not plan motion."
                        )
                    ts = float(getattr(obs, "detections_ts", 0.0) or 0.0)
                    if ts <= 0.0 or not getattr(obs, "fresh", True):
                        # stale lane: report last-known detections with an age flag instead
                        # of failing. the pick lookup and the 2.0 s pre-motion check refuse
                        # stale data anyway, and merging it could turn no-data into an empty scene.
                        lane_stale_s = (
                            max(0.0, time.time() - ts) if ts > 0.0 else 0.0
                        )
                    else:
                        lane_stale_s = None
                        graph.update(obs)
                # healthy path: the call shape differs, so keep the plain call for
                # graphs whose scene_brief takes no arguments.
                brief = (
                    graph.scene_brief()
                    if lane_stale_s is None
                    else graph.scene_brief(lane_stale_s=lane_stale_s)
                )
                # the detectable vocabulary is not appended here: the offline lane
                # speaks this string, so it belongs in the tool description as
                # static text.
                if getattr(graph, "_updates", 1) == 0:
                    # the graph has never had an observation: report failure, because
                    # success beside 'nothing is known' is a contradiction the model
                    # resolves as false.
                    return fail(brief)
                return ok(brief)

            return Tool(
                name="scene_brief",
                description=(
                    "Return the compact current-scene summary with persistent "
                    "references (one line per object: ref, label, pose, "
                    "holding/in-container state, freshness). Call it to re-check "
                    "what objects are visible and their refs.\n"
                    "WHAT THE ROBOT CAN EVER SEE: " + known_labels.describe()
                    + ". These 20 are the ONLY labels perception can report "
                    "(a curated set, not all of COCO). When a human names "
                    "something outside this list (a mug, a phone stand, a "
                    "door), do NOT invent a label and do NOT report it as "
                    "merely absent — say plainly that it is not something this "
                    "robot can see, and name the closest real alternative when "
                    "one exists (a mug is not visible to me; I can see cups)."
                ),
                schema={"type": "object", "properties": {}, "required": []},
                run=run,
                timeout_s=5.0,
            )

        def make_user_tool(name: str, blocking: bool) -> Tool:
            def run(args: dict) -> dict:
                backend = self.backend
                if backend is None:
                    return fail(f"gateway not wired — cannot {name}")
                method = getattr(backend, name, None)
                if method is None:
                    return fail(f"backend has no {name}")
                return method(args)

            if name == "query_user":
                description = (
                    "Ask the human operator a question and WAIT for their reply "
                    "(blocking). Returns the human's answer verbatim. Use it ONLY "
                    "when the instruction itself is ambiguous — never to ask "
                    "permission for a motion."
                )
                schema = {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question for the human.",
                        },
                    },
                    "required": ["question"],
                }
            else:
                description = (
                    "Send a short notice to the human operator WITHOUT waiting "
                    "(non-blocking). Use to report progress or completion."
                )
                schema = {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The notice text.",
                        },
                    },
                    "required": ["message"],
                }
            return Tool(
                name=name,
                description=description,
                schema=schema,
                run=run,
                # the gateway's query reply timeout is 20.0 s and its honest reply
                # goes out at that deadline, so this must be strictly greater (25.0)
                # or the agent times out first and loses an answer from the last seconds.
                timeout_s=25.0,
                blocking=blocking,
            )

        for tool in (
            make_pick(),
            make_place(),
            make_place_at_spot(),
            make_navigate(),
            make_turn_around(),
            *((make_return_home(),) if self._fetch_on() else ()),   # '0' omits the return_home tool
            make_hand("open_hand", "open"),
            make_hand("close_hand", "close"),
            make_cancel(),
            make_pose(),
            make_resolve_ref(),
            make_scene_brief(),
            make_user_tool("query_user", blocking=True),
            make_user_tool("notify_user", blocking=False),
        ):
            self.register(tool)
        register_optional_tools(self, self._optional_switches)

    # -- embodiment consumption (pick placeholder geometry) ----------------------

    def _envelope(self) -> Optional[Any]:
        """The embodiment envelope object, or None when not wired."""
        emb = self.embodiment
        if emb is None:
            return None
        return getattr(emb, "envelope", None) or None

    def _spot_position(
        self, spot_name: str
    ) -> tuple[Optional[List[float]], Optional[dict]]:
        """Resolve a named base-relative spot from the wired embodiment.

        Consumes ``embodiment.base_relative.key_positions`` (round-5
        config-seam): the position returned is the CONFIGURED one — an
        unknown spot name (or an unwired/no-key_positions embodiment) is
        an honest failure envelope, NEVER a guessed position.  Works with
        the loader's ``EmbodimentConfig`` (frozen, key_positions is a
        mappingproxy) and brain/embodiment.py's ``Embodiment`` (which also
        offers ``base_relative.position(name)``).

        Returns ``(position, None)`` or ``(None, failure_envelope)``.
        """
        emb = self.embodiment
        if emb is None:
            return None, fail(
                "embodiment not wired — no base_relative.key_positions to "
                "resolve the spot against"
            )
        br = getattr(emb, "base_relative", None)
        if br is None:
            return None, fail(
                "embodiment has no base_relative section — cannot resolve spots"
            )
        pos: Any = None
        pos_fn = getattr(br, "position", None)
        if callable(pos_fn):
            try:
                pos = pos_fn(spot_name)
            except Exception:  # noqa: BLE001 — resolution must not crash the tool
                pos = None
        if pos is None:
            kp = getattr(br, "key_positions", None)
            if kp is not None:
                pos = kp.get(spot_name)
        if pos is None:
            known = sorted(getattr(br, "key_positions", {}))
            return None, fail(
                f"UNKNOWN_SPOT: {spot_name!r} is not a configured base-relative "
                f"spot (known: {known}) — never guessing a position"
            )
        try:
            pos = [float(x) for x in pos]
        except (TypeError, ValueError):
            return None, fail(
                f"spot {spot_name!r} has a non-numeric position {pos!r}"
            )
        return pos, None

    def _constrained_pick_point(
        self, x: Any, y: Any, z: Any, label: str
    ) -> tuple[float, float, float, Optional[dict]]:
        """Placeholder pick-point computation consuming the embodiment
        envelope (round-3 config-seam; documented semantics — the per-object
        grasp geometry is the pick engine's / depth projector's job, this is
        the brain-side capability net):

          1. Tabletop floor: a target z below ``pick_min_height_m`` is a
             capability-block (FLOOR_OBJECT refusal).  The bound is a
             TORSO-frame height (z=0 at the torso link — the perception
             and engine frames); ground objects read ~-0.75 and every
             tabletop/low-workspace object passes.  No ground-pick skill
             exists.
          2. Height ceiling: z above ``pick_max_height_m`` is out of the
             workspace (OUT_OF_REACH refusal).
          3. Grip-center standoff (shipped 0.0): the commanded pick point is
             the detection position offset by ``grip_center_standoff_m`` on
             z and on x toward the object's side.  NO current target
             (pick_engine_v2 / pick daemon) consumes it — stdin x/y/z is
             the WRIST target — so a non-zero value offsets the point onto
             air (0.37 -> cx 0.514, E2 22:49).  The engine's own grip-zone
             + approach offsets do this job engine-side.
          4. Reach clamp: the horizontal (x, y) distance of the COMMANDED
             point is clamped to ``max_reach_m`` (x/y scaled together) —
             the arm is never commanded past the envelope reach.

        Without a wired embodiment the tool layer keeps its independent
        constants (TABLETOP_MIN_Z / TABLETOP_MAX_Z) and applies no standoff
        or clamp (the historical behavior).  Non-numeric / non-finite inputs
        pass through untouched — the gateway refuses them honestly.

        Returns ``(x, y, z, refusal_or_None)``; a refusal is a failure
        envelope.
        """
        if x is None or y is None or z is None:
            return x, y, z, None
        try:
            xf, yf, zf = float(x), float(y), float(z)
        except (TypeError, ValueError):
            return x, y, z, None
        if not (math.isfinite(xf) and math.isfinite(yf) and math.isfinite(zf)):
            return x, y, z, None
        env = self._envelope()
        # the floor comes from the envelope through the shared law: with the
        # lane off it is pick_min_height_m unchanged, with the lane on the
        # rescue floor. TABLETOP_MIN_Z is the fallback when none is wired.
        min_z = effective_pick_floor(env) if env is not None else TABLETOP_MIN_Z
        max_z = env.pick_max_height_m if env is not None else TABLETOP_MAX_Z
        if zf < min_z:
            if getattr(env, "low_workspace_lane", False) is True:
                return x, y, z, fail(
                    f"FLOOR_OBJECT: {label!r} is at torso-frame height "
                    f"{zf:.2f} m — below the low-workspace descent floor "
                    f"(min {min_z:.2f} m torso frame; a SIMULATED "
                    "descent-budget bound with the lane armed, not a "
                    "measured body capability). Refusing honestly."
                )
            return x, y, z, fail(
                f"FLOOR_OBJECT: {label!r} is at torso-frame height "
                f"{zf:.2f} m — below the tabletop workspace (min "
                f"{min_z:.2f} m torso frame). The robot has no floor-pick "
                "skill; refusing honestly."
            )
        if zf > max_z:
            return x, y, z, fail(
                f"OUT_OF_REACH: {label!r} is at height {zf:.2f} m — above "
                f"the workspace ceiling (max {max_z:.2f} m)."
            )
        cx, cz = xf, zf
        if env is not None:
            standoff = float(getattr(env, "grip_center_standoff_m", 0.0) or 0.0)
            cz = zf + standoff
            if xf > 0.0:
                cx = xf + standoff
            elif xf < 0.0:
                cx = xf - standoff
            max_reach = float(getattr(env, "max_reach_m", 0.0) or 0.0)
            if max_reach > 0.0:
                d = math.hypot(cx, yf)
                if d > max_reach:
                    scale = max_reach / d
                    cx, yf = cx * scale, yf * scale
        return cx, yf, cz, None

    # -- detection lookup (pick resolution) --------------------------------------

    # -- where the fetch started and who asked for it ---------------------------

    def _home_call(self, req: dict) -> Optional[str]:
        """One gateway ``home`` call; the reply text, or None (no such
        backend method / failure).  Never raises."""
        fn = getattr(self.backend, "home", None) if self.backend is not None else None
        if fn is None:
            return None
        try:
            env = fn(dict(req))
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(env, dict) or not env.get("success"):
            return None
        value = env.get("value")
        return value if isinstance(value, str) else None

    def _home_snapshot(self) -> Optional[dict]:
        """The gateway's start-point snapshot (dict) or None."""
        import json  # imported lazily to keep the module import light
        text = self._home_call({"op": "status"})
        if not text or not text.startswith("DONE:home {"):
            return None
        try:
            snap = json.loads(text[len("DONE:home "):])
        except ValueError:
            return None
        return snap if isinstance(snap, dict) else None

    #: '0' disables the fetch-return rule: no start point, no requester
    #: filtering and no return_home tool.
    FETCH_RETURN_ENV = "OPENBOT_FETCH_RETURN"

    def _fetch_on(self) -> bool:
        import os  # imported lazily; stdlib only
        return os.environ.get(self.FETCH_RETURN_ENV, "").strip() != "0"

    def _self_state(self) -> Any:
        return getattr(self.scene_graph, "self_state", None)

    def _home_fetch_mark(self, target: Any) -> None:
        """navigate with an EMPTY hand: during a fetch-to-me request ("get me a
        bottle") record the start point ONCE per request (r1 H3: never for a
        plain walk); operator decisions 2026-10-17: a retry or a repeated
        request keeps the first start point; the fetch ends on the handover /
        any place, a stop BEFORE the pick (after it a stop only pauses), or the
        gateway's 15 min expiry.  B8l-hf3: without a live fetch the walk's start
        is kept as a CANDIDATE -- the realtime voice lane calls navigate before
        the words of the request are transcribed; the pick promotes it."""
        if not self._fetch_on() or self._held_record() is not None:
            return
        if self._names_person(target):
            return
        ss = self._self_state()
        fetch_live = getattr(ss, "fetch_live", None)
        if (ss is None or fetch_live is None or not fetch_live()
                or (self.__dict__.get("_home_marked_seq") == ss.fetch_seq
                    and self._home_snapshot() is None)):
            # no live fetch: keep this walk's start as a candidate
            self._home_call({"op": "candidate"})
            return
        self._home_record(ss, "mark")

