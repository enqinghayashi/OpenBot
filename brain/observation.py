"""
brain/observation.py — the normalized world snapshot the verifier, scene
graph, safety hooks and context assembler all read.

Owned by: worker-core.  Stdlib only, zero ROS.

SHARED CONTRACT (other modules depend on these exact names):
    Detection(label:str, x:float, y:float, z:float, conf:float,
              ref:str|None, ts:float, orientation:str='upright',
              width_m:float|None=None, height_m:float|None=None,
              hue:float|None=None, color_name:str|None=None,
              body_side:str|None=None, class_name:str|None=None,
              instance_id:int|None=None)
      - instance_id (batch 4 / G1) is the OPTIONAL perception track id
        (int >= 1, present only with OPENBOT_INSTANCE_IDS=1); advisory,
        never a ref, None when absent/malformed.
      - orientation is the depth projector's grasp-approach signal
        ('upright'|'horizontal'); it is carried end-to-end to the pick
        engine so a lying object is NOT picked upright.  Defaults to
        'upright' when absent/unknown.
      - hue/color_name/body_side are OPTIONAL perception colour + side
        semantics (degrees 0-360 / colour band / 'left'|'right'), carried
        for the model's scene prose.  All default to None when
        absent/unknown — never guessed.  Informational, not positions.
    GripState(side:str, joints:list[float], contact:bool, available:bool,
              stale_ts:float|None, contact_ts:float|None, is_open:bool|None,
              left_* folded equivalents, commanded:list[float])
      - joints is the MEASURED motor q — the grip reply's ``actual`` mapping
        in joint-id order 0..6 — and is EMPTY when the reply omits or
        malforms ``actual`` (never faked from the commanded values).
        commanded is the last COMMANDED q (the reply's top-level per-joint
        keys = grip_server ``last_q``).  is_open / the open reference are
        judged on joints (MEASURED space) only.
    WalkStatus(phase:str)          # 'SEARCHING'|'APPROACHING'|'REACHED'|'IDLE'
    WorldObservation(detections:list[Detection], grip:GripState|None,
                     walk:WalkStatus|None, detections_ts:float, fresh:bool,
                     vocab:dict|None=None)   # batch 4 / D13, gateway cache

Platform frame (STUART facts, verified on the robot):
    - Detections come from /object_detections_body (torso frame):
      x = lateral RIGHT, y = forward, z = up.
    - The topic is EVENT-DRIVEN: silence == empty scene, NOT 'no data'.
      `detections_ts` is the timestamp of the last message; `fresh` is the
      gateway's judgment (a dead node is a STALE node, and stale detections
      must never drive motion).
    - Walk FSM: /walk_status == 'REACHED' is the ONLY navigation truth.
      /unitree/slam_mapping/odom does NOT exist — never verify on odom.
    - Grip: grip_server `state [side]` is the verification signal; the grip
      does NOT auto-verify, `contact` (close_adaptive -> DONE:contact) is the
      ONLY real grasp signal — never derived from joints (world-truth
      round 4, CONTACT LAW).  The deployed grip_server state payload emits,
      per side, the last COMMANDED q as top-level per-joint keys (``last_q``
      -> GripState.commanded), the MEASURED motor q under ``actual``
      (-> GripState.joints; OMITTED — never faked from last_q — when the
      dex3 lane has no state, skills/arm/grip_server.py:1036-1048) + the
      per-side available flag (the dex3 hand-state/DDS lane health probe);
      there is NO literal hand-state contact field in the reply.
    - World-truth round 4: the state read AGES (stale_ts / available=False
      past GRIP_STATE_MAX_AGE_S); `contact_ts` is the timestamp of the
      close_adaptive verdict folded into `contact` (None when none is on
      file) so a stale fold can be aged by its own window; `is_open` is the
      measured-open judgment against the CAUSAL measured-open reference
      (world-truth round 5 — the round-4 boot-window path is DELETED): per
      side the reference forms ONLY from TWO CONSECUTIVE agreeing state
      reads taken AFTER a gateway-dispatched open command for that side
      that ACTUALLY RAN (DONE grip reply this session) — never a
      single-first-read bootstrap, and reads taken before ANY open command
      (boot reads, however agreeing) NEVER qualify.

`from_dict` classmethods parse gateway payloads (socket JSON lines or the
mock in-process payload).  Malformed numeric payloads raise ValueError —
silent 0.0 coercion would fabricate world state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# only 'REACHED' is success
WALK_PHASES = ("SEARCHING", "APPROACHING", "REACHED", "IDLE")

#: colour bands, mirrors HUE_BANDS in
#: skills/perception/instance_labels.py
#: label is '<colour> <class>'; class_name stays bare
#: a colour prefix must not break class matching
#: extra words so requested_colour() sees them
PERCEPTION_COLOUR_NAMES = (
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
    "white", "black", "grey", "gray",
    "teal", "turquoise", "aqua", "navy", "sky blue", "maroon", "crimson",
    "scarlet", "violet", "lilac", "magenta", "gold", "lime",
)
_LABEL_ARTICLES = ("the", "a", "an")


def detection_base_class(det: Any) -> str:
    """Bare COCO class of a Detection (or detection-like dict), lowercased.

    ``class_name`` — the bare class perception now emits on every row
    ('cup') — wins; when it is absent (legacy rows) a colour prefix is
    stripped off ``label``.  Never raises: a malformed/missing source
    yields '' (no match), never a guess.
    """
    cn = getattr(det, "class_name", None)
    if not isinstance(cn, str):
        try:
            if isinstance(det, dict):
                cn = det.get("class_name")
        except Exception:  # noqa: BLE001
            cn = None
    if isinstance(cn, str) and cn.strip():
        return cn.strip().lower()
    label = getattr(det, "label", None)
    if not isinstance(label, str):
        try:
            if isinstance(det, dict):
                label = det.get("label")
        except Exception:  # noqa: BLE001
            label = None
    return label_base_class(label)


def label_base_class(label: Any) -> str:
    """Map a query/detection label to its bare COCO class, lowercased.

    Strips a leading article and a leading perception colour word, so
    'red cup', 'the red cup' and 'cup' all resolve to 'cup'.  A lone colour
    token is NOT stripped: 'orange' is a COCO class (the fruit) and must
    not vanish — the colour-prefix rule only fires when a real remainder
    follows the colour word.
    """
    if not isinstance(label, str):
        return ""
    toks = label.strip().lower().split()
    while toks and toks[0] in _LABEL_ARTICLES:
        toks.pop(0)
    if toks and toks[0] in PERCEPTION_COLOUR_NAMES and len(toks) > 1:
        toks.pop(0)
    return " ".join(toks)


def requested_colour(label: Any) -> Optional[str]:
    """The colour word prefixed on a query label, or None.

    'the red cup' -> 'red'; 'cup' / 'orange' (the fruit) -> None.  Only a
    leading colour that modifies a real remainder is read as a colour
    request — a lone colour token is treated as a class name, never a
    colour qualifier.
    """
    if not isinstance(label, str):
        return None
    toks = label.strip().lower().split()
    while toks and toks[0] in _LABEL_ARTICLES:
        toks.pop(0)
    if toks and toks[0] in PERCEPTION_COLOUR_NAMES and len(toks) > 1:
        return toks[0]
    return None


def detection_matches_label(det: Any, label: Any) -> bool:
    """Does one detection match a query label (class-aware)?

    True when the query is the detection's exact display ``label``
    ('red cup' == a row labelled 'red cup') OR when the query's base COCO
    class equals the detection's base COCO class ('cup' matches a row whose
    class is 'cup' — coloured or bare).  Colour disambiguation among
    same-class detections is the caller's job via ``color_name``.
    """
    if label is None:
        return False
    q = str(label).strip().lower()
    if not q:
        return False
    det_label = getattr(det, "label", None)
    if not isinstance(det_label, str):
        try:
            det_label = det.get("label") if isinstance(det, dict) else None
        except Exception:  # noqa: BLE001
            det_label = None
    if isinstance(det_label, str) and det_label.strip().lower() == q:
        return True
    base = detection_base_class(det)
    return bool(base) and base == label_base_class(label)


def _num(value: Any, name: str) -> float:
    """Coerce a payload value to float; raise on anything non-numeric.

    None and bool are REJECTED: a detection missing x/y/z must never
    become a phantom object at the torso origin, and a bool is not a
    measurement — silent 0.0 coercion would fabricate world state.
    NaN/Inf are REJECTED too: a NaN z bypasses the tabletop floor refusal
    (NaN < -0.35 is False) so a phantom pick could proceed; a non-finite
    measurement is never a world state.
    """
    if value is None or isinstance(value, bool):
        raise ValueError(f"observation payload: {name} must be numeric, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"observation payload: {name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"observation payload: {name} must be finite, got {value!r}")
    return number


def _opt_hue(value: Any) -> Optional[float]:
    """Parse the optional perception hue (degrees, 0-360) -> float|None.

    Colour is INFORMATION for the model's scene prose (so it can tell "the
    red cup on the left" from "the green cup on the right") — never a
    position — so a malformed hue (absent, bool, non-numeric, NaN/Inf, or
    outside [0, 360]) drops to None instead of raising (unlike ``_num``,
    which guards world-state positions).  Only finite degrees in [0, 360]
    pass through.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not (0.0 <= number <= 360.0):
        return None
    return number


def _opt_color_name(value: Any) -> Optional[str]:
    """Parse the optional perception colour band -> str|None.

    A non-string value is None; surrounding whitespace is stripped and a
    blank string is None (never a guessed colour).  Case is preserved as
    emitted by perception.
    """
    if not isinstance(value, str):
        return None
    name = value.strip()
    return name or None


def _opt_instance_id(value: Any) -> Optional[int]:
    """Parse the optional perception instance id -> int >= 1 | None.

    Batch 4 / G1 (docs/YOLOE_BATCH4_ARCHITECTURE.md §3.1, §3.3): the yolo
    node emits ``instance_id`` (a session-monotonic track id >= 1) ONLY when
    ``OPENBOT_INSTANCE_IDS=1``.  Only a real int >= 1 passes; a bool, a str
    ('7'), a float (7.0), zero or a negative is None — never a guess, an id
    that is not exactly what the tracker minted binds nothing.  The id is
    ADVISORY (D6): position + gate stay the authority.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _opt_body_side(value: Any) -> Optional[str]:
    """Parse the optional perception body-side token -> 'left'|'right'|None.

    Only the two canonical tokens (case-insensitive) pass; anything else —
    a non-string, a bool, an unknown token — is None (never guessed).  This
    is which side of the torso line the object sits on (display semantics),
    never a position.
    """
    if not isinstance(value, str):
        return None
    side = value.strip().lower()
    if side in ("left", "right"):
        return side
    return None


@dataclass
class Detection:
    """One torso-frame object detection (yolo-seg + depth projection)."""

    label: str
    x: float  # torso frame: right (m)
    y: float  # forward (m)
    z: float  # up (m)
    conf: float
    ref: Optional[str]  # scene-graph id, e.g. 'bottle_3'
    ts: float  # time.time() stamp
    #: grasp approach from the depth projector
    #: 'upright' | 'horizontal' (lying object)
    #: a lying object must not be picked upright
    #: absent -> 'upright'
    orientation: str = "upright"
    #: extents in metres
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    #: colour + side, for the model's prose
    #: hue 0-360, body_side left|right
    #: info only, never a position; None if absent
    hue: Optional[float] = None
    color_name: Optional[str] = None
    body_side: Optional[str] = None
    #: bare COCO class ('cup')
    #: None -> fall back to label
    class_name: Optional[str] = None
    #: track id, OPENBOT_INSTANCE_IDS=1 only
    #: last field, positional order safe
    #: not a ref (refs stay label_N)
    #: advisory; position + gate decide
    #: observe reply drops it when None
    instance_id: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Detection":
        """Parse a gateway detection payload.

        Accepts keys: label (or 'labels'), class_name, x, y, z, conf, ref,
        ts, orientation, width_m, height_m, hue, color_name, body_side,
        instance_id.
        Optional keys default: conf 0.0, ref None, ts 0.0, orientation
        'upright', width_m None, height_m None, hue None, color_name None,
        body_side None, class_name None, instance_id None (a non-int or
        an int < 1 is None — never a guess).  An unrecognized orientation value
        is NOT passed through — it defaults to 'upright' (never a guess at
        a grasp approach).  A malformed hue/color_name/body_side value drops
        to None (colour is informational, never a position) — no raise.
        """
        label = d.get("label", d.get("labels"))
        if not isinstance(label, str):
            raise ValueError(f"detection payload missing string label, got {label!r}")
        ref = d.get("ref")
        if ref is not None and not isinstance(ref, str):
            raise ValueError(f"detection payload ref must be str|None, got {ref!r}")
        orientation = d.get("orientation")
        if not isinstance(orientation, str) or orientation not in ("upright", "horizontal"):
            orientation = "upright"
        width_m = d.get("width_m")
        if width_m is not None:
            width_m = _num(width_m, "width_m")
        height_m = d.get("height_m")
        if height_m is not None:
            height_m = _num(height_m, "height_m")
        class_name = d.get("class_name")
        if class_name is not None and not isinstance(class_name, str):
            class_name = None  # never guess; fall back to label
        if isinstance(class_name, str):
            class_name = class_name.strip() or None
        return cls(
            label=label,
            x=_num(d.get("x"), "x"),
            y=_num(d.get("y"), "y"),
            z=_num(d.get("z"), "z"),
            conf=_num(d.get("conf", 0.0), "conf"),
            ref=ref,
            ts=_num(d.get("ts", 0.0), "ts"),
            orientation=orientation,
            width_m=width_m,
            height_m=height_m,
            hue=_opt_hue(d.get("hue")),
            color_name=_opt_color_name(d.get("color_name")),
            body_side=_opt_body_side(d.get("body_side")),
            class_name=class_name,
            instance_id=_opt_instance_id(d.get("instance_id")),
        )

    def describe(self) -> str:
        ref = self.ref or "-"
        return f"{self.label}[{ref}]@({self.x:.3f},{self.y:.3f},{self.z:.3f})"


