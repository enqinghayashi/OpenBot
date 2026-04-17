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

