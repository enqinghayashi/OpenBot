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

    def apply_verdict(self, verdict: Optional[Verdict], tool: str,
                      ref: Optional[str], *, label: Optional[str] = None,
                      args: Optional[dict] = None) -> None:
        """Flip holding/edges ONLY on a verifier-CONFIRMED success for pick/place.

        in_progress and failure verdicts leave the graph untouched. 'pick'
        success sets holding=True (object leaves any container belief);
        'place' success sets holding=False and in_container=True as the
        graph's task-level belief -- but only when ``args`` names a container
        (into_ref/into_label); a bare release-only place mints no
        IN-CONTAINER belief.  ``args=None`` (direct callers) keeps the old
        unconditional behaviour.  Unknown refs and other tools are no-ops.

        H1 (BRAIN B3d): the real detection shape carries ref=None, so a falsy
        ``ref`` is resolved from ``label`` WITHOUT guessing across
        instances (see _resolve_for_verdict).
        """
        if verdict is None:
            return
        if verdict.status != "success":
            # A place whose release chain PASSED
            # (verifier's released note) is out of
            # hand though containment is unconfirmed;
            # no in_container belief minted.
            if (tool in ("place", "place_at_spot")
                    and verdict.status == "in_progress"
                    and PLACE_RELEASED_MARK in (verdict.evidence or "")):
                if not ref and label:
                    ref = self._resolve_for_verdict(tool, label, args)
                e = self.refs.get(ref) if ref else None
                if e is not None:
                    e["holding"] = False
            return
        if not ref and label:
            ref = self._resolve_for_verdict(tool, label, args)
        e = self.refs.get(ref) if ref else None
        if e is None:
            return
        if tool == "pick":
            e["holding"] = True
            e["in_container"] = False
        elif tool in ("place", "place_at_spot"):
            e["holding"] = False
            if args is None or args.get("into_ref") or args.get("into_label"):
                e["in_container"] = True

    def _resolve_for_verdict(self, tool: str, label: str,
                             args: Optional[dict]) -> Optional[str]:
        lab = self._normalize(str(label))
        same = [k for k, e in self.refs.items() if e["label"] == lab]
        if tool == "pick":
            if not same:
                return None
            args = args or {}
            try:
                x, y, z = (float(args["x"]), float(args["y"]), float(args["z"]))
                finite = all(math.isfinite(v) for v in (x, y, z))
            except (KeyError, TypeError, ValueError):
                finite = False
            if finite:
                best, best_d = None, None
                for k in same:
                    e = self.refs[k]
                    d = math.sqrt((e["x"] - x) ** 2 + (e["y"] - y) ** 2
                                  + (e["z"] - z) ** 2)
                    if best_d is None or d < best_d:
                        best, best_d = k, d
                if best_d is not None and best_d <= 2 * self.assoc_distance_m:
                    return best
            return same[0] if len(same) == 1 else None
        if tool in ("place", "place_at_spot"):
            held = [k for k in same if self.refs[k]["holding"]]
            return held[0] if len(held) == 1 else None
        return None

    # -- lookups -----------------------------------------------------------

    def resolve_ref(self, text: str) -> Optional[str]:
        """Resolve a phrase ('the red cup', 'grab bottle_3', 'red_cup_2') to a ref key, or None.

        Matching is deliberately simple (no embeddings): (1) an EXACT ref
        key match wins first -- the whole (trimmed, lowercased) text equal
        to a ref key resolves to it before any substring logic runs, so
        with refs {bottle_1, bottle_10} the query 'bottle_10' returns
        bottle_10 (a substring scan would match bottle_1 first and look
        ambiguous); (1b) else a query that ITSELF matches the ref syntax
        ('label_<n>', e.g. 'bottle_10') is treated as an EXACT-REF query:
        exact key match only, NO fall-through to the substring tier -- a
        nonexistent 'bottle_10' must return None, never silently resolve
        to the prefix-colliding 'bottle_1' (the substring tier would match
        'bottle_1' inside 'bottle_10'); (1c) else ANY ref-syntax TOKEN
        embedded in the phrase ('grab bottle_10') is an exact-ref
        candidate too -- a whole-token key hit wins over every fuzzy tier
        below, and a ref-syntax token with NO exact key resolves to None
        ('grab bottle_10' with refs {bottle_1} must not alias bottle_1);
        two+ distinct exact ref-syntax tokens are genuine ambiguity ->
        None; (2) else a ref key that appears in the
        text wins, with multiple key matches resolving to None (genuine
        ambiguity -- the caller asks rather than acting on the wrong
        object); (3) else an exact token match on label or display name (WP-B2);
        (4) else a substring match on label.  Multiple label candidates resolve
        deterministically: a single holding ref wins (the in-hand object
        is the most salient referent), otherwise the most recently seen
        ref.  Stale refs stay resolvable -- identity survives the walk.
        None on genuine ambiguity.
        """
        if not text:
            return None
        t = text.strip().lower()
        if t in self.refs:  # exact key wins
            return t
        # Exact DISPLAY NAME ('red_cup_2',
        # 'red cup 2') names one ref. Checked
        # before the ref-syntax tier: it looks
        # like '<label>_<n>' and would be
        # refused as a missing key.
        _tn = "_".join(re.findall(r"[a-z0-9]+", t))
        _named = [k for k, n in self._compute_display_names().items() if n == _tn]
        if len(_named) == 1:
            return _named[0]
        if _REF_SYNTAX_RE.match(t):
            # ref-syntax ask, no exact key: None.
            # A missing ref must NOT alias a prefix-
            # colliding one: resolve_ref('bottle_10')
            # with refs {bottle_1}.
            return None
        # Ref-syntax token inside a phrase counts
        # too: whole-token match; an exact hit
        # beats the fuzzy tiers; no hit -> None,
        # no substring alias.
        ref_tokens = [
            tok for tok in set(re.findall(r"[a-z0-9_]+", t))
            if _REF_SYNTAX_RE.match(tok)
        ]
        if ref_tokens:
            exact = [tok for tok in ref_tokens if tok in self.refs]
            if len(exact) == 1 and len(exact) == len(ref_tokens):
                return exact[0]
            return None  # none, or several
        keys = [k for k in self.refs if k in t]
        if len(keys) == 1:
            return keys[0]
        if len(keys) > 1:
            return None
        words = set(re.findall(r"[a-z0-9_]+", t))
        cands = [k for k, e in self.refs.items() if e["label"] in words]
        if not cands:  # substring fallback
            cands = [k for k, e in self.refs.items() if e["label"] in t]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        holding = [k for k in cands if self.refs[k]["holding"]]
        if len(holding) == 1:
            return holding[0]
        return max(cands, key=lambda k: self.refs[k]["last_seen"])

    def scene_brief(self, lane_stale_s: Optional[float] = None) -> str:
        """Compact one-line-per-ref summary with pose, state and freshness.

        The placed-state label reads 'IN-CONTAINER (task-inferred)' so the
        model context never mistakes the graph's task-level belief for a
        sensor claim (verifier v2 region-verifies into: places when the
        evidence allows, but this flag is the graph's task-level inference,
        not a sensor claim).

        ``lane_stale_s`` is the READER's wall-clock dead time since the last
        fresh detection, supplied ONLY when the lane has gone quiet (None =
        the lane is live, and the output is byte-identical to before).  It
        exists because this class cannot see its own silence: ``_clock`` is
        observation-derived and advances only inside update(), so a dead lane
        FREEZES it at the final fresh frame -- and a ref merged from that very
        frame then reads age 0.0, i.e. a position that looks current while
        being minutes old.  That is not hypothetical: run8 (2026-09-28) went
        stale at ts=1790594754.0 with the bottle merged from that exact frame,
        so the old line would have read '0.0s' beside a lane silent for 221 s.
        Adding the reader's dead time is a CORRECTION to the one clock, not a
        second clock: true age = (now - _clock) + (_clock - last_seen), and
        the reader's clock is the same one detections_ts is stamped from
        (gateway/assemble.py:412 stamps it from time.time(); brain/hooks.py:466
        subtracts the same way against a 2.0 s budget).
        """
        if not self.refs:
            if self._updates == 0:
                # Never heard from perception -> "empty"
                # would claim something from zero evidence.
                # A dead lane looks the same as an empty
                # one, so this names the lane instead.
                # Absence of evidence is not evidence
                # of absence.
                return ("SCENE: unavailable — no observation has EVER been "
                        "received (the perception/gateway lane is not "
                        "answering). This is NOT an empty scene: nothing is "
                        "known about the world. Do not plan motion.")
            return "SCENE: empty"
        # stash display names
        display_names_dict = self._compute_display_names()
        for ref, name in display_names_dict.items():
            if ref in self.refs:
                self.refs[ref]["display_name"] = name
        lines: List[str] = []
        for ref, e in sorted(
            self.refs.items(),
            key=lambda kv: (kv[1]["holding"], kv[1]["last_seen"]),
            reverse=True,
        ):
            age = self._clock - e["last_seen"]
            # A stale lane ages every ref: the graph
            # clock froze at the last frame, so add
            # the reader's dead time -- true wall age,
            # and STALE can fire on its own.
            lane_stale = bool(lane_stale_s and lane_stale_s > 0.0)
            if lane_stale:
                age += float(lane_stale_s)
            parts = []
            if e["holding"]:
                parts.append("IN-HAND")
            if e["in_container"] is True:
                parts.append("IN-CONTAINER (task-inferred)")
            parts.append(
                f"STALE {age:.0f}s"
                if (lane_stale or age > self.stale_age_s)
                else f"{age:.1f}s"
            )
            if lane_stale:
                # LAST-KNOWN rides the free-text TAIL:
                # `_SCENE_BRIEF_LINE` ends in
                # `(?P<parts>.*)$`, so the spoken lane
                # parses it. A separate note line would
                # not -- the robot would read its own
                # detection log aloud.
                parts.append("LAST-KNOWN")
            # REACH ADVISORY. The brief is the brain's
            # only object truth, so each line carries a
            # reach token: without one a far target
            # looks as good as a near one, and the model
            # may pick a target the engine refuses.
            # Standoff is hypot(x, y) of the torso pose;
            # see PICK_REACH_ADVISE_M.
            #
            # Rides the TAIL like LAST-KNOWN: prefix
            # through conf=<d.d> untouched, so
            # `_SCENE_BRIEF_LINE` still matches and
            # scene_brief_prose restructures. Tokens
            # dodge the spoken flags ('IN-HAND',
            # 'IN-CONTAINER', 'STALE <n>s').
            #
            # Non-finite x/y emits no token:
            # 'reach=nanm' would be a lie.
            _x, _y, _z = e.get("x"), e.get("y"), e.get("z")
            if (isinstance(_x, (int, float)) and isinstance(_y, (int, float))
                    and math.isfinite(_x) and math.isfinite(_y)):
                reach_m = math.hypot(_x, _y)
                parts.append(f"reach={reach_m:.2f}m")
                if isinstance(_z, (int, float)) and math.isfinite(_z):
                    # right-arm law (RIGHT_SHOULDER_XYZ)
                    out = reach_residual(_x, _y, _z) > REACH_REFUSE_POS_M
                else:
                    out = reach_m > self.pick_reach_advise_m   # no height: legacy
                if out:
                    parts.append("OUT-OF-REACH (navigate to it first)")
            # Colour + side suffix when present, so
            # the model can tell "the red cup on the
            # left" from "the green cup on the right".
            # Absent fields leave the line byte-identical.
            colour_side = ""
            if e.get("color_name"):
                colour_side += " (%s)" % e["color_name"]
            if e.get("body_side"):
                colour_side += ", %s of body" % e["body_side"]
            # WIRE FORMAT, NOT COSMETICS -- do not
            # reorder (label first), never prepend a
            # header. The OFFLINE lane reads this aloud:
            # scene_brief_prose parses each line with a
            # STRICT full-line regex (`_SCENE_BRIEF_LINE`)
            # wanting exactly `ref: label x= y= z= conf=`.
            # A miss -> byte-identical echo, robot reads
            # its own log aloud.
            # The ref is the model's handle by POSITION;
            # bad targets get fixed in the prompts.
            # name, distance, bearing, neighbours
            display_name = e.get("display_name")
            if display_name:
                parts.append(f"name={display_name}")
            # distance + bearing
            try:
                dist = spatial.distance(e)
                bearing_deg, bearing_dir = spatial.bearing(e)
                parts.append(f"distance={dist:.2f}m")
                parts.append(f"bearing={bearing_dir}")
            except Exception:
                pass  # no spatial info
            # nearest 2
            try:
                other_refs = [k for k in self.refs.keys() if k != ref]
                if other_refs:
                    others = [self.refs[k] for k in other_refs]
                    # dist to each other ref
                    dists = []
                    for other_ref, other_e in zip(other_refs, others):
                        dx = float(other_e.get("x") or 0) - float(e.get("x") or 0)
                        dy = float(other_e.get("y") or 0) - float(e.get("y") or 0)
                        pair_dist = math.hypot(dx, dy)
                        # which way it lies
                        if abs(dx) < 0.03:
                            direction = 'ahead' if dy > 0 else 'behind'
                        elif dx < 0:
                            direction = 'left'
                        else:
                            direction = 'right'
                        other_display = display_names_dict.get(other_ref, other_ref)
                        dists.append((pair_dist, other_display, direction))
                    # nearest 2
                    dists.sort(key=lambda x: x[0])
                    for pair_dist, other_name, direction in dists[:2]:
                        parts.append(f"{other_name} {pair_dist:.2f}m {direction}")
            except Exception:
                pass  # no neighbours

            lines.append(
                f"{ref}: {e['label']} x={e['x']:.2f} y={e['y']:.2f} z={e['z']:.2f} "
                f"conf={e['conf']:.2f} {' '.join(parts)}{colour_side}"
            )
        return "\n".join(lines)

    # -- machinery ---------------------------------------------------------

    def spatial_objects(self, include_stale: bool = False) -> List[dict]:
        """B7: the refs as brain.spatial object dicts -- FRESH (seen within
        stale_age_s of the graph clock) and NOT in hand, unless include_stale;
        colour = the majority vote of the last 7 measured names; hue / extents
        as last measured (None when never measured)."""
        out = []
        for ref, e in self.refs.items():
            if e.get("holding"):
                continue
            if not include_stale and (self._clock - e["last_seen"]) > self.stale_age_s:
                continue
            votes = [v for v in (e.get("colour_vote") or []) if v]
            colour = (max(set(votes), key=lambda v: (votes.count(v), -votes[::-1].index(v)))
                      if votes else None)
            out.append({"ref": ref, "label": e.get("label", "object"), "colour": colour,
                        "hue": e.get("hue"), "x": e.get("x"), "y": e.get("y"),
                        "z": e.get("z"), "width_m": e.get("width_m"),
                        "height_m": e.get("height_m"), "conf": e.get("conf"),
                        "instance_id": e.get("instance_id")})
        return out

    def _compute_display_names(self) -> Dict[str, str]:
        """ref -> colour_objectname display name (B7) over every ref that is
        not in hand (stale ones included, so a name never changes because an
        object was briefly unseen)."""
        if not self.refs:
            return {}
        objs = self.spatial_objects(include_stale=True)
        names = spatial.display_names(objs)
        # No measured colour -> keep the ref key
        # as the name: a second 'cup_2' would
        # collide with another cup's key.
        for o in objs:
            if not o.get("colour"):
                names[o["ref"]] = o["ref"]
        return names

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower()

