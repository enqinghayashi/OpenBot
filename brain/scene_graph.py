"""Persistent scene graph: merges event-driven detections into stable object refs that survive walks."""

import math
import re
import time
from typing import Dict, List, Optional

from brain.observation import Detection, WorldObservation
from brain.verifier import Verdict
from brain.self_state import SelfState, PLACE_RELEASED_MARK
from brain import spatial

# ref dict, shared contract.
# extra optional key: instance_id: int
# (see _claim_instance).
#   label: str          normalized, lowercase
#   x, y, z: float      last torso pose (m)
#   conf: float         newest detection confidence
#   last_seen: float    ts of newest confirmation (s)
#   holding: bool       believed in gripper
#   in_container: bool|None  post-place belief
#   color_name: str|None  perception colour, prose only
#   body_side: str|None  left/right, prose only
#   colour_vote: list   last 7 colour names
#   display_name: str|None  spatial.display_names

# '<label>_<n>' is an EXACT-REF ask.
# No substring fallthrough: a missing
# 'bottle_10' must never alias 'bottle_1'.
# Same inside a phrase: 'grab bottle_10'.
_REF_SYNTAX_RE = re.compile(r"^[a-z0-9_]+_\d+$")

#: Planar standoff (m). Past it the engine refuses.
#:
#: Residual by target y: 0.057@0.249,
#: 0.107@0.435, 0.352@0.629,
#: 0.348@0.750, 0.397@0.771 m.
#: Refusal bar 0.15 m (REACH_REFUSE_POS,
#: skills/arm/pick_engine_v3.py).
#: Slope 1.26 m/m -> 0.15 at y ~= 0.47 m.
#: 0.50 leaves ~3 cm.
#:
#: Caveat: needs the full 3D pose and stance,
#: not planar y alone. A hint, not a promise.
PICK_REACH_ADVISE_M = 0.50

#: OUT-OF-REACH uses the engine's reach LAW, not the hypot:
#: residual = d(right shoulder -> target) - 0.382 m.
#: Shoulder in torso_link: (fwd 0.00396, right 0.10021,
#: up 0.24778) = (0.10021, 0.00396, 0.24778) here.
#: Refusal above 0.15 m (pick_engine_v3.py).
#: Hypot missed 0.10 m right offset, 0.25 m height:
#: cup (0.284, 0.428, -0.010), hypot 0.514,
#: residual 0.147 -- reachable, model walked away.
#: Left Dex3 is out; right arm is the arm.
RIGHT_SHOULDER_XYZ = (0.10021, 0.00396, 0.24778)
ARM_REACH_LAW_M = 0.382
REACH_REFUSE_POS_M = 0.15


def reach_residual(x: float, y: float, z: float) -> float:
    """Predicted pick residual (m) of the right arm for a torso-frame target."""
    sx, sy, sz = RIGHT_SHOULDER_XYZ
    return math.sqrt((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) - ARM_REACH_LAW_M


class SceneGraph:
    """Tracks persistent object refs (label_<n>) across detection batches.

    update() associates each detection with the nearest ref of the same label
    within assoc_distance_m (3D euclidean, torso frame), creating label_<n>
    refs as needed. A detection below conf_floor (default 0.0 = disabled) is
    filtered in the loop: no ref, no refresh, no brief entry. Refs are NEVER
    deleted eagerly: an object unseen for
    stale_age_s is marked STALE in the brief, but its identity survives walks
    and detection silence (event-driven detections: silence == empty scene,
    not empty world). apply_verdict() mutates ref edges (holding /
    in_container) ONLY on a verifier-confirmed success verdict for pick/place.
    Pure Python, deterministic, zero ROS.
    """

    def __init__(self, assoc_distance_m: float = 0.25, stale_age_s: float = 5.0,
                 conf_floor: float = 0.0,
                 pick_reach_advise_m: float = PICK_REACH_ADVISE_M,
                 target_disambiguation: bool = False):
        self.assoc_distance_m = assoc_distance_m
        # Default off. When exactly True,
        # _bind_target narrows same-class rows
        # by spoken colour and REFUSES a bare
        # label naming several objects
        # ("ambiguous: N cups ..."). Off keeps
        # freshest-first binding, pick and place into:.
        # Read as getattr(sg, "target_disambiguation",
        # False) is True.
        self.target_disambiguation = target_disambiguation
        self.stale_age_s = stale_age_s
        # Below this: no ref, no pose refresh,
        # no brief line (#109: a conf-0.234
        # phantom became a referent). Code
        # default 0.0; shipped 0.35 from
        # PerceptionConfig.det_conf_floor.
        # Non-zero would change every existing
        # SceneGraph(). BRAIN-ONLY: the verifier
        # reads raw detections. Move this upstream
        # of its obs and a table object reads
        # 'gone', so verify_pick certifies a grasp.
        self.conf_floor = conf_floor
        # Planar standoff (m) that flags a line
        # OUT-OF-REACH. Constructor arg so tests
        # pin the boundary. SCENE-BRIEF ONLY:
        # tail text, gates no motion.
        self.pick_reach_advise_m = pick_reach_advise_m
        self.refs: Dict[str, dict] = {}
        self._counters: Dict[str, int] = {}  # per-label ref counters
        self._clock: float = 0.0  # newest observation ts
        #: Observations merged. Zero = never heard from
        #: perception, not "the scene is empty".
        #:
        #: Can't say "the lane died since": failed
        #: observations never get here. Every update()
        #: call site guards on
        #: `obs is not None and obs.fresh and detections_ts > 0`
        #: (brain/agent/tool_executor.py and
        #: brain/agent/agent.py), so update(None) is tests
        #: only. One success pins it at 1 forever.
        #: Liveness is the reader's job (brain/agent/tools.py
        #: make_scene_brief re-observes and refuses).
        self._updates: int = 0
        #: self memory per label, from verdicts only
        self.self_state = SelfState()

    # -- taking in ---------------------------------------------------------

    def update(self, obs: Optional[WorldObservation]) -> None:
        """Merge one WorldObservation's detections into the persistent refs."""
        if obs is None:
            return
        self._updates += 1
        now_ts = max([obs.detections_ts] + [d.ts for d in obs.detections] + [0.0])
        if now_ts <= 0.0:
            now_ts = time.time()
        self._clock = max(self._clock, now_ts)
        # The (label, instance_id) pairs this packet
        # carries. A live id in the same packet
        # belongs to that row, so another row can't
        # steal it by label+distance (two tracks in
        # the 0.25 m gate flipped ids every frame).
        # Empty -> plain association.
        live_ids = {(self._normalize(d.label), d.instance_id)
                    for d in obs.detections
                    if getattr(d, "instance_id", None) is not None
                    and not d.ref and d.conf >= self.conf_floor}
        for d in obs.detections:
            # CONF-FLOOR FILTER (#109): no ref, no pose,
            # no brief. In the LOOP, not _associate --
            # that's typed `-> str`, so an early bare
            # return gives None (update() drops it
            # anyway). The verifier never reads this
            # graph, so only the model's channel loses it.
            if d.conf < self.conf_floor:
                continue
            self._associate(d, d.ts if d.ts > 0.0 else self._clock,
                            live_ids=live_ids)

    def _associate(self, d: Detection, ts: float, live_ids=None) -> str:
        """Return the ref key for this detection, creating or refreshing it.

        ``live_ids`` (set of (label, instance_id), default None = none): an
        entry whose id differs from this row's and is carried by another row
        of the SAME packet is not a label+distance candidate for this row.
        """
        label = self._normalize(d.label)
        if d.ref:  # from the gateway
            key = d.ref
            if key not in self.refs:
                self.refs[key] = self._new_entry(d, ts, label)
            else:
                self._refresh(self.refs[key], d, ts)
            return key
        best, best_d2 = self._instance_match(d, label), None  # id first
        iid = getattr(d, "instance_id", None)
        for key, e in (self.refs.items() if best is None else ()):
            if e["label"] != label:
                continue
            held = e.get("instance_id")
            if (iid is not None and held is not None and held != iid
                    and live_ids and (label, held) in live_ids):
                continue  # owned elsewhere
            d2 = (e["x"] - d.x) ** 2 + (e["y"] - d.y) ** 2 + (e["z"] - d.z) ** 2
            if d2 <= self.assoc_distance_m ** 2 and (best_d2 is None or d2 < best_d2):
                best, best_d2 = key, d2
        if best is None:
            key = self._new_ref(label)
            self.refs[key] = self._claim_instance(self._new_entry(d, ts, label), d)
        else:
            self._refresh(self._claim_instance(self.refs[best], d), d, ts)
        return key if best is None else best

    # -- mutation only after a verdict ------------------------------------

