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
