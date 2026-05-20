"""brain/verifier.py — deterministic post-condition judge (post-hook; conservative:
ambiguous -> in_progress + evidence 're-observe', NEVER false success).

STOP (world-truth round 3, fix E): navigate 'stop' claims NO goal
post-condition — the gateway forwards it to the FSM's real stop trigger
(/walk_target 'stop') and requests NO verdict, so this module never judges a
stop.  Documented here only; the gateway does not call the verifier for stop.

World-truth round 4 (CONTACT LAW + PICK GRASP LAW + measured-open reference):
* CONTACT is ONLY the literal ``close_adaptive`` DONE:contact verdict — never
  derived from joints, and the obs lane exposes NO derived-contact field.
  The deployed grip_server state reply emits the joints (commanded
  ``last_q`` per joint, plus the measured ``actual`` mapping when the dex3
  lane has state) + the per-side ``available`` flag (the dex3 hand-state/DDS
  lane-health probe); there is no literal hand-state contact field in it.  Folded contact confirms a close
  ONLY when its own verdict timestamp (contact_ts) is inside
  CLOSE_CONTACT_WINDOW_S AND the grip state read is fresh
  (GRIP_STATE_MAX_AGE_S) — a daemon'd grip's stale folded contact must not
  confirm (fix 2).
* PICK GRASP LAW: a real engine pick closes the grip over the ENGINE's
  socket — the gateway never sees the close_adaptive verdict, so a
  serialized close verdict can NEVER certify a pick.  When the palm DDS lane
  is UP the engine VERIFIES its own grasp (quadrant/palm-data check) and
  fails with ERROR:grasp_failed on an air-grab; in BLIND MODE (lane down) it
  skips verification and returns DONE anyway.  The gateway CANNOT see the
  engine's grip-socket traffic; its immediate post-pick ``state [side]``
  poll's available flag IS the lane probe: lane_state=up (available True,
  fresh) <=> the engine DONE for this pick was engine-verified.  verify_pick
  confirms the grasp half IFF lane_state == up; down/unknown -> in_progress
  (blind mode, no evidence available).  The literal folded-close-verdict
  contact path remains ONLY for the close_hand tool — it is NOT pick
  evidence.
* Grip state reads AGE (GRIP_STATE_MAX_AGE_S): a stale or errored read is
  available=False with stale_ts — a daemon'd grip can never confirm forever.

World-truth round 5 (CAUSAL OPEN REFERENCE + DETECTIONS-ALIVE GATE +
close_both per-side dispatch):
* 'open' is MEASURED against the CAUSAL measured-open reference: per side,
  the reference forms ONLY from state reads taken AFTER a gateway-dispatched
  open command for that side (open/open_left/open_both/drop) — two
  CONSECUTIVE agreeing reads within OPEN_JOINT_TOLERANCE (assemble-side
  constant).  The round-4 BOOT WINDOW is deleted: boot reads carry NO
  semantic guarantee (a mid-grasp boot / power-down with closed hands would
  make the closed shape the 'open' reference and a closed hand would verify
  'open' — round-5 finding).  Without an open command, NO read ever forms a
  reference; is_open stays None and open/place verification is in_progress
  're-observe' (one re-observe after the open completes the double-read).
  This module consumes assemble's is_open judgment via GripState; it holds
  no reference state of its own.
* DETECTIONS-ALIVE GATE: verify_pick's world-object (target-gone) checks use
  CHANNEL-LIVENESS, not packet freshness: target-gone is confirmable when
  detections_alive (the node ever spoke and within DETECTIONS_ALIVE_S) AND
  the latest packet lacks the target ref (or detections empty — silence ==
  empty scene).  A quiet-but-running node is alive while its observation
  ages out of freshness; a dead node (never spoke, or silent beyond the
  liveness window) -> in_progress 're-observe; detections node silent'.
  A fresh NON-EMPTY packet is never required for target-gone.  The
  ``silence_means_empty`` switch (mirror of embodiment.perception) gates the
  empty-detections == empty-scene reading: False -> an empty scene is
  unconfirmable (never success).
* close_both (sides=both) now has a REAL satisfaction path: the gateway
  dispatches close_adaptive RIGHT then close_adaptive LEFT (per-side tactile
  evidence) and the tool result lists the per-side verdicts
  (right_close=contact|full|no_data; left_close=...).  Both sides'
  verdicts contact|full on FRESH available reads -> success; any no_data /
  missing / offline side -> in_progress.  Replies without per-side fields
  fall back to the folded-contact path (both hands' literal contact).

Verifier v2 (PLACE-INTO-CONTAINER PASS, additive):
* 'place' with an ``into:<container>`` argument (AGENT-lane semantics — the
  gateway rejects place args, so only the brain tool layer can carry it)
  gains a CONTAINMENT half on top of the release-only chain.  The tool
  layer resolves the container against the freshest detections + scene
  graph and annotates the drop reply with
  ``into=<ref>; into_label=<label>; placed_object=<label>``; the verifier
  parses those annotations.  Success requires the release evidence AND the
  placed object's PROJECTED position (a detection in the SAME
  depth-projector observation the release checks read — no new perception
  node, no fabricated geometry) inside the container's containment region:
  the container's projected detection position plus/minus CONTAINER_FIT_M
  on x, y and z (torso frame).  The observation carries NO pixel box and
  NO refs (detections are depth-projected points; Detection.ref is always
  None), so the region is a configured half-extent around the projected
  container point — never a measured one — and object identity is by
  LABEL plus the temporal evidence this pass enforces.  The verifier never
  claims containment it did not measure.

  PLACE-INTO HONESTY GATES (fix round, review D1-D3):
  * D1 REGION: CONTAINER_FIT_M = 0.12 m per axis — the plausibility
    half-extent for a ~20 cm workspace bin (0.10 m half-width + ~2 cm
    depth-projection noise allowance at arm reach), and strictly tighter
    than the 0.15 m beside-the-bin / in-flight misses that must read
    in_progress.  Nothing on the detection path carries a box/size
    (parse_detections keeps only label/x/y/z/conf/ts), so the region is a
    CONFIGURED tolerance, never a measured one.
  * D2 POST-DROP FRESHNESS + STATIONARITY: the containment observation
    must be at most POST_DROP_FRESHNESS_S = 3.0 s old (the 10 s
    DETECTIONS_ALIVE_S window is channel liveness, not packet freshness —
    a packet up to 10 s old can predate the drop's completion).  And one
    instantaneous detection never certifies: the SkillVerifier holds a
    stateful per-(container, object) record (mirror of the navigate epoch
    guard) — success needs TWO CONSECUTIVE agreeing in-region
    observations, the first sighting being honest in_progress with a
    re-observe instruction.
  * FIX ROUND 2, D4 PER-TASK RECORD (containment state leak): the
    stationarity record is SCOPED TO THE TASK EXECUTION, not the
    (container, object) key forever.  The tool layer stamps the drop
    EXECUTION epoch (``dropped_at``, time.time() at drop execution — the
    same clock base as the gateway-stamped detections_ts and the
    verifier's ``now``; the verifier uses it ONLY as an identity token
    (equality across calls), never for age math) onto every into: place
    reply.  The record re-anchors whenever the epoch changes, so a new
    drop of the same key can never pair with the previous task's frames:
    a completed containment task leaves its record behind and task B's
    FIRST post-drop observation is a fresh-task first sighting
    (in_progress), never a second agreeing observation on task A's last
    frame.  ``first_ts`` is READ (fix round 2): the two agreeing
    observations must be two DISTINCT frames within RE_OBSERVE_WINDOW_S
    of each other — a re-delivered packet (equal detections_ts) or a
    pair straddling the window does not certify and re-anchors.
  * FIX ROUND 2, D5 PRE-DROP PRE-EXISTING SET (substitution blocker
    anchoring): the in-region 'preexisting' set is built from a REAL
    PRE-DROP observation — the observation the tool layer held before
    executing the drop (the executor plumbs it onto the reply as
    predrop_state/predrop_obj) — NEVER from the first post-drop frame.
    When no pre-drop observation is available the set is UNKNOWN and
    success additionally requires displacement evidence (the object
    observed OUTSIDE the region at least once in this task's sequence,
    then in-region, then agree).  A genuinely-occupied-same-position bin
    (the drop lands on a pre-existing same-label object's position) stays
    in_progress with evidence naming the ambiguity honestly.
  * D3 NO SUCCESS-BY-SUBSTITUTION: success requires the confirming
    in-region candidate to be a NEW ARRIVAL — not within
    PREEXISTING_ASSOC_M of any PRE-DROP in-region position (fix round 2,
    D5).  A same-label object that was already in the container can never
    certify a drop — with Detection.ref always None, an unchanged
    in-region object is consistent with pre-existence and reads
    in_progress-and-diagnose.
  Object detected OUTSIDE the region, an unchanged pre-existing in-region
  object, multiple same-label candidates, a missing container/object
  detection, or a silent/stale detections node -> in_progress with honest
  evidence naming the measured offsets (never success, never a fabricated
  in-bin).  A reply WITHOUT the into= annotation is the round-5
  release-only place, byte-for-byte unchanged (the gateway lane's drop
  never carries into=).
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass

from brain import self_state as _self_state
from brain.known_labels import TARGET_ALIASES as _TARGET_ALIASES
from brain.known_labels import TRACKED_CLASSES as _TRACKED_CLASSES
from brain.observation import WorldObservation
from brain.tool_contract import ToolContract

__all__ = ["Verdict", "SkillVerifier"]

# ── verdict objects ───────────────────────────────────────────────────────────

VALID_STATUSES = ("success", "failure", "in_progress")


@dataclass
class Verdict:
    """Outcome of verifying one tool invocation against the world.

    status        - one of 'success' | 'failure' | 'in_progress'
    evidence      - what the verifier actually observed (the LLM's evidence text)
    failure_reason- structured reason for 'failure' (replan input), None otherwise

    Conservative rule: any ambiguity (missing/unfresh data, indistinguishable
    target) MUST be 'in_progress' with evidence starting 're-observe'. A verifier
    that cannot confirm must never claim success.
    """

    status: str
    evidence: str
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"invalid Verdict status {self.status!r}; must be one of {VALID_STATUSES}")

    @classmethod
    def success(cls, evidence: str) -> "Verdict":
        return cls(status="success", evidence=evidence)

    @classmethod
    def failure(cls, evidence: str, reason: str) -> "Verdict":
        return cls(status="failure", evidence=evidence, failure_reason=reason)

    @classmethod
    def in_progress(cls, evidence: str) -> "Verdict":
        return cls(status="in_progress", evidence=evidence)

    def to_string(self) -> str:
        """'VERDICT:success|Evidence:...|Reason:...' (Reason omitted when None)."""
        s = f"VERDICT:{self.status}|Evidence:{self.evidence}"
        if self.failure_reason is not None:
            s += f"|Reason:{self.failure_reason}"
        return s


# ── what a tool reply looks like ─────────────────────────────────────────────
# writer socket:
#   DONE | ERROR:<msg> | Q:<14 floats> | STATE:...
# grip socket:
#   open|close|... -> DONE
#   close_adaptive -> contact | full | no_data
#   state [side] -> DONE:<json>
#   the grip never auto-verifies
#   'state' is the only world signal
# engine socket: READY, then DONE | ERROR:<msg>
# Q: / STATE: -> no world signal


# ── parsers for the fields the gateway annotates onto tool results ───────────
# gateway fields on motion results:
#   pick: lane_state=, grip_state_ts=
#   place: prev_is_open=, prev_pick_verified=, release_open= (engine open check)
#   open_hand: sides=, prev_contact=
#   navigate: NAV_EPOCH=, NAV_LABEL=, LATEST_WALK_TS=, PHASES_SEEN=
# lane_state=up is the only grasp evidence
# release_open counts because the lift parks the hand closed before DONE, so
# is_open never holds after a place
# the open_hand fields are informational

_NAV_EPOCH_RE = re.compile(r"NAV_EPOCH=([0-9]+(?:\.[0-9]+)?)")
_NAV_LABEL_RE = re.compile(r"NAV_LABEL=([^;]*)")
_LATEST_WALK_TS_RE = re.compile(r"LATEST_WALK_TS=([0-9]+(?:\.[0-9]+)?)")
_PHASES_SEEN_RE = re.compile(r"PHASES_SEEN=([A-Za-z_,]*)")
# motion-wait token: a cancel writes WALK_WAIT=cancelled_by_operator@<s>
# the value class takes letters and underscores, leaving out the @<s> suffix
_WALK_WAIT_RE = re.compile(r"WALK_WAIT=([A-Za-z_]+)")
_LANE_STATE_RE = re.compile(r"lane_state=(up|down|unknown)")
_GRIP_STATE_TS_RE = re.compile(r"grip_state_ts=([0-9]+(?:\.[0-9]+)?)")
_PREV_IS_OPEN_RE = re.compile(r"prev_is_open=([A-Za-z]+):(True|False|None)")
_PICK_VERIFIED_RE = re.compile(r"prev_pick_verified=([A-Za-z]+):(True|False)")
_RELEASE_OPEN_RE = re.compile(r"release_open=([A-Za-z]+):(True|False)")
_SIDES_RE = re.compile(r"sides=([A-Za-z]+)")
_RIGHT_CLOSE_RE = re.compile(r"right_close=([^;\s]*)")
_LEFT_CLOSE_RE = re.compile(r"left_close=([^;\s]*)")
# annotated values are written bare when they contain only [A-Za-z0-9_.-] and
# double-quoted otherwise ('wine glass')
# both forms are anchored, so a bad line yields None instead of swallowing the
# next key
_INTO_REF_RE = re.compile(r'into=(?:"([^"]*)"|([A-Za-z0-9_.\-]+))')
_INTO_LABEL_RE = re.compile(r'into_label=(?:"([^"]*)"|([A-Za-z0-9_.\-]+))')
_PLACED_OBJECT_RE = re.compile(r'placed_object=(?:"([^"]*)"|([A-Za-z0-9_.\-]+))')
_INTO_X_RE = re.compile(r"into_x=(-?[0-9]+(?:\.[0-9]+)?)")
_INTO_Y_RE = re.compile(r"into_y=(-?[0-9]+(?:\.[0-9]+)?)")
_INTO_Z_RE = re.compile(r"into_z=(-?[0-9]+(?:\.[0-9]+)?)")
# drop epoch, stamped at execution; it scopes the stationarity record to one
# task
# predrop_*: the pre-drop in-region set read from the pre-tool observation
_DROPPED_AT_RE = re.compile(r"dropped_at=([0-9]+(?:\.[0-9]+)?)")
_PREDROP_STATE_RE = re.compile(r"predrop_state=(known|unknown)")
_PREDROP_OBJ_RE = re.compile(r"predrop_obj=([^;]*)")
#: place-into tokens: container id plus the deep footprint
#: as xc, near y, far y, half x, rim z, bottom z
_INTO_IID_RE = re.compile(r"\binto_iid=(\d+)")
_INTO_FP_RE = re.compile(
    r"\binto_fp=(-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){5})")

#: grip 'state' read max age in seconds
#: a read older than this cannot confirm grip state
GRIP_STATE_MAX_AGE_S = 5.0

#: folded close verdict max age in seconds
#: a stale folded verdict cannot confirm a close
CLOSE_CONTACT_WINDOW_S = 15.0

#: detections channel liveness window in seconds
#: a quiet node inside the window is alive; never-spoke or older is dead
DETECTIONS_ALIVE_S = 10.0

#: finger hold threshold: a joint at least this far off its commanded q is held
#: joints 1..6 are checked. real holds measure 0.174-0.253 rad off command,
#: misses 0.030 and controls 0.058, so 0.10 separates a held object from an
#: empty close.
#: a threshold near this value reads 'empty' and triggers a re-pick that drops
#: the object; a test pins the constant.
FINGER_HOLD_LAG_RAD = 0.10
#: held needs this many of the six finger joints off command
#: 0 = empty, 1..3 = no claim
FINGER_HOLD_MIN_JOINTS = 4

#: containment half-extent in the torso frame, metres
#: a placed object must project within this of the container
#: detections carry no box, so this is a configured tolerance, not a measured
#: one
#: 0.12 covers a ~20 cm bin (0.10 half-width) plus ~2 cm of depth-projection
#: noise at arm reach
#: it stays tighter than the 15 cm misses (dx=0.15, dz=0.07) and the in-flight
#: frame (dz=0.15)
CONTAINER_FIT_M = 0.12

#: packet age allowed for a finished drop
#: the liveness window is not freshness: a packet older than this can predate
#: the drop's completion
POST_DROP_FRESHNESS_S = 3.0

#: association tolerance in metres
#: closer than this is the same object
PREEXISTING_ASSOC_M = 0.25

#: deep bin: an object sits 0.14-0.30 m below the container centroid, and the
#: centroid drifts to the far wall, so judge the aimed-at footprint
#: shrunk by DEEP_FIT_MARGIN_M, from bottom-DEEP_BOT_BELOW_M to
#: rim+DEEP_RIM_ABOVE_M
DEEP_FIT_MARGIN_M = 0.03
DEEP_BOT_BELOW_M = 0.05
DEEP_RIM_ABOVE_M = 0.10

#: re-observe window: the two agreeing frames must be distinct and no further
#: apart than this
#: a re-observe turn takes ~1-2 s, so 5.0 s covers a normal pair while staying
#: inside the 10.0 s liveness window
#: an older frame re-anchors the pair
RE_OBSERVE_WINDOW_S = 5.0


def _parse_float_field(pattern: re.Pattern, tool_result: str | None) -> float | None:
    """Parse one float-valued field (NAV_EPOCH / LATEST_WALK_TS)."""
    if not tool_result:
        return None
    m = pattern.search(tool_result)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_nav_epoch(tool_result: str | None) -> float | None:
    return _parse_float_field(_NAV_EPOCH_RE, tool_result)


def _parse_latest_walk_ts(tool_result: str | None) -> float | None:
    return _parse_float_field(_LATEST_WALK_TS_RE, tool_result)


def _parse_phases_seen(tool_result: str | None) -> list[str]:
    """Phases observed since the dispatch epoch (uppercased, in order)."""
    if not tool_result:
        return []
    m = _PHASES_SEEN_RE.search(tool_result)
    if not m or not m.group(1):
        return []
    return [p.strip().upper() for p in m.group(1).split(",") if p.strip()]


def _parse_walk_wait(tool_result: str | None) -> str | None:
    """The gateway's WALK_WAIT token, or None when the result carries none."""
    if not tool_result:
        return None
    m = _WALK_WAIT_RE.search(tool_result)
    return m.group(1) if m else None


def _walk_was_cancelled(tool_result: str | None) -> bool:
    """True when the operator cancelled this navigate while it was running.

    STOP-KILLS-TASK (C5, 2026-10-01).  Distinct from every timeout token:
    ``timeout_after_walk`` means the FSM never reported, while this means the
    operator ended the task on purpose.  The distinction is the whole point —
    run12 returned the former and the model, reasonably, tried again.
    """
    return _parse_walk_wait(tool_result) == "cancelled_by_operator"


_ENGINE_CANCELLED_RE = re.compile(r"engine_reply=CANCELLED:|^\s*CANCELLED[: ]")
_ENGINE_REPLY_RE = re.compile(r"engine_reply=([^\r\n]*)")

# the point the arm is sent to, as `pick_point=<x>,<y>,<z>` in the torso frame
# absent: the lift checks are skipped
# away and above the pick point = in the hand
_PICK_POINT_RE = re.compile(
    r"pick_point=\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)")

#: min 3-D distance from the pick point for an object in the hand
#: a miss is re-detected 0.027-0.041 m away, an in-hand object 0.23-0.29 m
#: 0.10 sits between the two
PICK_INHAND_MIN_DISPLACEMENT_M = 0.10
#: min rise above the pick point
#: in-hand objects rise +0.157 to +0.278 m, missed ones +0.006 to +0.020 m
#: a knocked-over object never rises, and a flat one stays 'visible'
#: 0.08 separates the two
PICK_INHAND_MIN_RISE_M = 0.08


def _parse_pick_point(*texts):
    """(x, y, z) from the first ``pick_point=`` annotation, else None."""
    for text in texts:
        if not text:
            continue
        m = _PICK_POINT_RE.search(text)
        if m:
            try:
                pt = tuple(float(v) for v in m.groups())
            except ValueError:
                continue
            if all(math.isfinite(v) for v in pt):
                return pt
    return None


# a label match cannot tell the picked object from a second object of the same
# label, so every same-label detection must be provably another object
# the pick context carries a preaction_dets= pre-pick snapshot to compare
# against
# any doubt keeps the conservative 'target still visible' answer
_PREACTION_DETS_RE = re.compile(r"preaction_dets=([^\n]*)")
#: closer than this to the pick point = the target is still there
PICK_BYSTANDER_MIN_FROM_PICK_M = 0.08
#: the pre-pick target is the detection nearest the pick point
#: within this distance
PICK_BYSTANDER_TARGET_GATE_M = 0.05
#: this close to a pre-pick non-target = that object
PICK_BYSTANDER_SAME_PLACE_M = 0.06
#: or a hue this far off the target's pre-pick hue (circular degrees):
#: a red cup is not the cyan cup
PICK_BYSTANDER_MIN_HUE_DEG = 60.0


def _parse_preaction_dets(*texts):
    """[(x, y, hue|None)] from the first ``preaction_dets=`` annotation.

    A malformed entry voids the WHOLE list (returns []): half a pre-pick
    scene could hide the target, and [] keeps the old rule.
    """
    for text in texts:
        if not text:
            continue
        m = _PREACTION_DETS_RE.search(text)
        if not m:
            continue
        out = []
        for part in m.group(1).split(";"):
            part = part.strip()
            if not part:
                continue
            f = [p.strip() for p in part.split(",")]
            if len(f) != 3:
                return []
            try:
                x, y = float(f[0]), float(f[1])
                hue = None if f[2] == "-" else float(f[2])
            except ValueError:
                return []
            if not (math.isfinite(x) and math.isfinite(y)) or (
                    hue is not None and not math.isfinite(hue)):
                return []
            out.append((x, y, hue))
        return out
    return []


def _hue_gap(a, b):
    """Circular distance in degrees between two hues, None if either is unknown."""
    if a is None or b is None:
        return None
    d = abs(float(a) % 360.0 - float(b) % 360.0)
    return min(d, 360.0 - d)


def _only_bystanders(obs, label, pick_point, pre):
    """True iff EVERY same-label detection in ``obs`` is provably another object.

    ``pre`` = the pre-pick same-label detections [(x, y, hue|None)].  False
    (= the old 'target still visible') when the pick point is unknown, no
    pre-pick detection lies within PICK_BYSTANDER_TARGET_GATE_M of it, no
    same-label detection is seen now, or any one of them is either within
    PICK_BYSTANDER_MIN_FROM_PICK_M of the pick point or matches neither
    bystander test (stood where a pre-pick NON-target stood / hue far from
    the target's).
    """
    if pick_point is None or label is None or not pre:
        return False
    px, py = pick_point[0], pick_point[1]
    dist = [math.hypot(x - px, y - py) for x, y, _h in pre]
    ti = min(range(len(pre)), key=dist.__getitem__)
    if dist[ti] > PICK_BYSTANDER_TARGET_GATE_M:
        return False
    target_hue = pre[ti][2]
    others = [p for i, p in enumerate(pre) if i != ti]
    match = [d for d in (getattr(obs, "detections", None) or [])
             if (d.label or "").lower() == label.lower()]
    if not match:
        return False
    for d in match:
        try:
            x, y = float(d.x), float(d.y)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        if math.hypot(x - px, y - py) < PICK_BYSTANDER_MIN_FROM_PICK_M:
            return False
        hue = getattr(d, "hue", None)
        try:
            gap = _hue_gap(target_hue, hue)
        except (TypeError, ValueError):
            gap = None
        if gap is not None and math.isfinite(gap) and gap > PICK_BYSTANDER_MIN_HUE_DEG:
            continue
        if any(math.hypot(x - ox, y - oy) <= PICK_BYSTANDER_SAME_PLACE_M
               for ox, oy, _h in others):
            continue
        return False
    return True


def _target_lifted_with_hand(obs, label, ref, pick_point):
    """(True, worst) iff EVERY detection matching the target sits at least
    PICK_INHAND_MIN_DISPLACEMENT_M from the pick point AND at least
    PICK_INHAND_MIN_RISE_M above it; worst = (min distance, min rise).
    (False, None) otherwise — any single matching detection near the pick
    point, or not risen, keeps the conservative 'still visible' answer."""
    if pick_point is None:
        return False, None
    px, py, pz = pick_point
    dets = getattr(obs, "detections", None) or []
    if ref is not None:
        match = [d for d in dets if d.ref == ref]
    elif label is not None:
        match = [d for d in dets
                 if (d.label or "").lower() == label.lower()]
    else:
        return False, None
    if not match:
        return False, None
    dists, rises = [], []
    for d in match:
        try:
            dx, dy, dz = float(d.x) - px, float(d.y) - py, float(d.z) - pz
        except (TypeError, ValueError):
            return False, None
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if not (math.isfinite(dist) and dist >= PICK_INHAND_MIN_DISPLACEMENT_M
                and dz >= PICK_INHAND_MIN_RISE_M):
            return False, None
        dists.append(dist)
        rises.append(dz)
    return True, (min(dists), min(rises))


#: an object near the pick point but risen, with the fingers held open, is
#: the held object still in view
PICK_HELD_IN_VIEW_MAX_XY_M = 0.25
PICK_HELD_IN_VIEW_MIN_RISE_M = 0.03
#: non-risen and this close means it is still on the table
#: farther out is another object (a second apple 0.14 m off, the held one
#: 0.17 m off and risen 0.10 m)
PICK_HELD_IN_VIEW_TABLE_XY_M = 0.06


def _target_held_in_view(obs, label, ref, pick_point):
    """(True, (xy distance, rise) of the held detection) iff at least one
    detection matching the target lies within PICK_HELD_IN_VIEW_MAX_XY_M
    (horizontal) of the pick point AND at least PICK_HELD_IN_VIEW_MIN_RISE_M
    above it (the object in the raised hand, in view), and NO non-risen
    matching detection sits within PICK_HELD_IN_VIEW_TABLE_XY_M of the pick
    point (the target still on the table).  Only meaningful when the fingers
    are measured held by an object (the caller checks).  (False, None)
    otherwise."""
    if pick_point is None:
        return False, None
    px, py, pz = pick_point
    dets = getattr(obs, "detections", None) or []
    if ref is not None:
        match = [d for d in dets if d.ref == ref]
    elif label is not None:
        match = [d for d in dets
                 if (d.label or "").lower() == label.lower()]
    else:
        return False, None
    held = None
    for d in match:
        try:
            dx, dy, dz = float(d.x) - px, float(d.y) - py, float(d.z) - pz
        except (TypeError, ValueError):
            return False, None
        dxy = math.sqrt(dx * dx + dy * dy)
        if not math.isfinite(dxy):
            return False, None
        if dz >= PICK_HELD_IN_VIEW_MIN_RISE_M:
            if dxy <= PICK_HELD_IN_VIEW_MAX_XY_M and (
                    held is None or dxy < held[0]):
                held = (dxy, dz)
        elif dxy <= PICK_HELD_IN_VIEW_TABLE_XY_M:
            return False, None      # the target is still on the table
    if held is None:
        # nothing risen and nothing same-label on the pick point while the
        # fingers are held: the object went into the hand and out of view
        # (a bystander 0.15 m off is not near the point); rise 0.0 = not seen
        # in the hand
        near = min(math.sqrt((float(d.x) - px) ** 2 + (float(d.y) - py) ** 2)
                   for d in match)
        return True, (near, 0.0)
    return True, held


def _task_was_cancelled(tool_result: str | None) -> bool:
    """True when the engine reply says the operator cancelled this pick/place.

    BRAIN B1 (2026-10-03, run19 forensics): the gateway wraps a cancelled
    engine reply as ``DONE pick executed; ...; engine_reply=CANCELLED:<where>``
    and a bare ``CANCELLED:<where>`` first line classifies as 'unknown'.  A
    cancelled task is NEVER a completed one.  Either shape is accepted.
    """
    if not tool_result:
        return False
    return _ENGINE_CANCELLED_RE.search(tool_result) is not None


def _parse_engine_reply(tool_result: str | None) -> str | None:
    """The value after ``engine_reply=`` or None when the reply carries none."""
    if not tool_result:
        return None
    m = _ENGINE_REPLY_RE.search(tool_result)
    return m.group(1).strip() if m else None


_TURN_CENTRED_RE = re.compile(r"\bTURN_RESULT=centred\b")


_BARE_DONE_RE = re.compile(r"DONE(?![\w:=])")


def _engine_reply_is_bare_done(engine_reply: str | None) -> bool:
    """True iff the engine's own line is the bare v5.5 ``DONE`` (optionally
    followed by gateway/lane suffix text), not ``DONE:<x>`` / ``DONEx``."""
    if not engine_reply:
        return False
    return _BARE_DONE_RE.match(engine_reply.strip()) is not None


def _parse_lane_state(tool_result: str | None) -> str | None:
    """'up' | 'down' | 'unknown' from the post-pick lane poll."""
    if not tool_result:
        return None
    m = _LANE_STATE_RE.search(tool_result)
    if not m:
        return None
    return m.group(1).lower()


def _parse_release_open(tool_result: str | None):
    """('right'|'left', 'True'|'False') from the engine's release-open check.

    None when the token is absent — UNMEASURED, which is NOT a pass: the
    caller keeps its original gate.  See the parser block above and
    `verify_place` for why the engine, not the gateway, supplies this one.
    """
    if not tool_result:
        return None
    m = _RELEASE_OPEN_RE.search(tool_result)
    if not m:
        return None
    return (m.group(1), m.group(2))


def _parse_grip_state_ts(tool_result: str | None) -> float | None:
    """The gateway's post-pick 'state [side]' poll timestamp."""
    return _parse_float_field(_GRIP_STATE_TS_RE, tool_result)


# point-cloud hold token, appended last as `; pc_hold=<TOKEN>`
# strict upper-case with underscore, so 'xpc_hold=MOVED' and 'pc_hold=MOVED_x'
# do not match
_PC_HOLD_RE = re.compile(r"(?:^|[;\s])pc_hold=([A-Z_]+)(?=$|[;\s])")
_PC_HOLD_TOKENS = ("STILL_ON_TABLE", "MOVED", "ABSENT_UNPROVEN", "UNKNOWN")


def _parse_pc_hold(tool_result: str | None) -> str | None:
    """H-BR-1: the point-cloud hold token of a pick reply, or None.

    None when the reply carries no well-formed ``pc_hold=`` token or is not
    text (such a reply is judged byte-for-byte as before).  Otherwise one of
    STILL_ON_TABLE | MOVED | ABSENT_UNPROVEN | UNKNOWN; any other upper-case
    value is read as UNKNOWN.  The LAST occurrence wins (the lane appends the
    token last).  Never raises.

    The token is a cloud HINT, never grasp evidence: STILL_ON_TABLE / MOVED
    are decisive NEGATIVE evidence; ABSENT_UNPROVEN and UNKNOWN prove NOTHING
    (a bottle in the hand and a bottle the detector missed read the same), so
    no value of this token may make a verdict more positive.
    """
    if not isinstance(tool_result, str) or not tool_result:
        return None
    found = _PC_HOLD_RE.findall(tool_result)
    if not found:
        return None
    token = found[-1]
    return token if token in _PC_HOLD_TOKENS else "UNKNOWN"


def _parse_nav_label(tool_result: str | None) -> str | None:
    """The requested target label from NAV_LABEL (None when absent)."""
    if not tool_result:
        return None
    m = _NAV_LABEL_RE.search(tool_result)
    if not m or not m.group(1).strip():
        return None
    return m.group(1).strip()


def _parse_prev_is_open(tool_result: str | None) -> tuple[str, str] | None:
    """(side, 'True'|'False'|'None') from prev_is_open=<s>:<v>.

    The PRE-DROP fresh 'state [side]' poll's measured-open judgment:
    'False' = the hand was in a NON-open shape before the drop (the open
    transition evidence), 'True' = already open, 'None' = unmeasurable
    (no trusted open reference yet / offline / stale).
    """
    if not tool_result:
        return None
    m = _PREV_IS_OPEN_RE.search(tool_result)
    if not m:
        return None
    return m.group(1), m.group(2)


def _parse_pick_verified(tool_result: str | None) -> tuple[str, str] | None:
    """(side, 'True'|'False') from prev_pick_verified=<s>:<v>.

    The gateway's per-side bookkeeping: True = the hold in that hand was
    created by an engine-verified pick (lane up, same side, this session),
    recorded at pick time and cleared on the next drop / open / new pick.
    """
    if not tool_result:
        return None
    m = _PICK_VERIFIED_RE.search(tool_result)
    if not m:
        return None
    return m.group(1), m.group(2)


def _parse_sides(tool_result: str | None) -> str | None:
    """'right' | 'left' | 'both' from the sides= annotation (None = absent)."""
    if not tool_result:
        return None
    m = _SIDES_RE.search(tool_result)
    if not m:
        return None
    return m.group(1).lower()


def _parse_place_into(tool_result: str | None) -> tuple[str | None, str | None, str | None]:
    """(into_ref, into_label, placed_object) from the place reply annotations.

    The brain place tool (agent lane) resolves an ``into:<container>``
    argument against the freshest detections + scene graph and annotates a
    real drop reply with ``into=<ref>; into_label=<label>;
    placed_object=<label>`` — the verifier's ONLY channel to the
    containment semantics (the gateway rejects place args, so a gateway-lane
    drop never carries these fields).  (None, None, None) when the place
    carried no into: argument — the release-only round-5 path.

    VALUE FORMAT (agreed with the writer in brain/agent/tools.py BY
    CONSTRUCTION): a value is written BARE when it contains only
    ``[A-Za-z0-9_.-]`` (single-token labels — every legacy annotation line
    parses to the same fields as before) and DOUBLE-QUOTED otherwise, so a
    multi-word label ('wine glass', 'trash can') parses intact INCLUDING
    its internal spaces and compares equal to a detection label of the
    same string.  The regexes accept the optional quote pair; both forms
    are exact-anchored on the value side — a bare value stops at the first
    char outside the safe set (never past ';' or '=' into the next
    key=value) and a quoted value terminates at its closing quote — so a
    bad line degrades to a truncated value or None, never a silent overlap
    of the keys appended after it.
    """
    if not tool_result:
        return None, None, None
    m = _INTO_REF_RE.search(tool_result)
    ref = m.group(1) if m and m.group(1) is not None else (m.group(2) if m else None)
    m = _INTO_LABEL_RE.search(tool_result)
    label = (m.group(1) if m and m.group(1) is not None
             else (m.group(2) if m else None))
    m = _PLACED_OBJECT_RE.search(tool_result)
    obj = (m.group(1) if m and m.group(1) is not None
           else (m.group(2) if m else None))
    return ref, label, obj


def _parse_into_ref_position(tool_result: str | None) -> tuple[float, float, float] | None:
    """(x, y, z) of the container ref's last-known position (fix nit ii).

    The tool layer resolves the into: argument to a scene-graph ref and —
    when that ref's last-known position is fresh (CONTAINER_REF_FRESH_S in
    tools.py, mirror of DETECTIONS_ALIVE_S) — annotates ``into_x/y/z``.
    These are MEASURED positions (the ref's last detection), used ONLY to
    disambiguate which same-label detection is the container
    ('ref-unique'); the containment offsets are always computed against the
    CURRENT detection, never the ref position.  None when the annotations
    are absent/unparseable — label matching then applies (and same-label
    ambiguity stays in_progress).
    """
    if not tool_result:
        return None
    m = _INTO_X_RE.search(tool_result)
    x = m.group(1) if m else None
    m = _INTO_Y_RE.search(tool_result)
    y = m.group(1) if m else None
    m = _INTO_Z_RE.search(tool_result)
    z = m.group(1) if m else None
    if x is None or y is None or z is None:
        return None
    try:
        return float(x), float(y), float(z)
    except ValueError:
        return None


def _parse_into_iid(tool_result: str | None) -> int | None:
    """B8j: the container's instance id (into_iid=<n>), else None."""
    m = _INTO_IID_RE.search(tool_result or "")
    if not m:
        return None
    v = int(m.group(1))
    return v if v >= 1 else None


def _parse_into_fp(tool_result: str | None) -> tuple | None:
    """B8j: (x_c, near_y, far_y, half_x, rim_z, bot_z) of a DEEP container
    (into_fp=...), else None (then the CONTAINER_FIT_M box applies)."""
    m = _INTO_FP_RE.search(tool_result or "")
    if not m:
        return None
    try:
        v = tuple(float(t) for t in m.group(1).split(","))
    except ValueError:
        return None
    if len(v) != 6 or not all(math.isfinite(t) for t in v):
        return None
    if not (v[2] > v[1] and v[3] > 0.0 and v[4] > v[5]):
        return None
    return v


def _parse_dropped_at(tool_result: str | None) -> float | None:
    """The drop-epoch annotation (dropped_at=...), stamped by the tool
    layer at drop EXECUTION (fix round 2, D4).  None when absent.

    Used ONLY as an identity token: the stationarity record is scoped to
    the task execution whose epoch this is — a different epoch (a NEW
    drop of the same container/object) resets the record.  Never used for
    age math (the base is the same time.time() the verifier and the
    gateway-stamped detections_ts use, but equality is all that matters).
    """
    return _parse_float_field(_DROPPED_AT_RE, tool_result)


def _parse_predrop(tool_result: str | None) -> tuple[str, list]:
    """(state, positions) from the pre-drop snapshot annotations.

    state == 'known': positions is the PRE-DROP same-label in-region set
    the tool layer read from the observation it held BEFORE executing the
    drop (fix round 2, D5) — the substitution blocker's anchor.  state ==
    'unknown': no usable pre-drop observation existed (stale/missing/
    ambiguous) — success then additionally requires displacement
    evidence.  Positions parse as [(x, y, z), ...] from
    ``predrop_obj=x,y,z;x,y,z...``; malformed entries are dropped (a
    malformed snapshot degrades to the empty set within its state, never
    a guessed position).
    """
    state = "unknown"
    if tool_result:
        m = _PREDROP_STATE_RE.search(tool_result)
        if m:
            state = m.group(1)
    positions: list = []
    if tool_result and state == "known":
        m = _PREDROP_OBJ_RE.search(tool_result)
        if m and m.group(1).strip():
            for chunk in m.group(1).split(";"):
                parts = chunk.split(",")
                if len(parts) != 3:
                    continue
                try:
                    positions.append((float(parts[0]), float(parts[1]),
                                      float(parts[2])))
                except ValueError:
                    continue
    return state, positions


def containment_predrop_snapshot(obs, into_label, placed_object) -> dict:
    """The PRE-DROP containment snapshot (fix round 2, requirement 3).

    The verifier's containment half only ever receives POST-tool
    observations (run_post / gateway _verify deliver the observation
    taken after the motion), but the substitution blocker must anchor on
    what was inside the container BEFORE the drop executed.  This helper
    reads that from the PRE-TOOL observation the executor holds — the
    observation the tool layer had before executing the drop — and
    returns a serializable snapshot the place tool annotates onto the
    reply (predrop_state / predrop_obj).

    Returns {"state": "known", "preexisting": [(x, y, z), ...]} — the
    same-label-as-placed_object detections inside CONTAINER_FIT_M of the
    container detection IN THAT SAME PRE-DROP OBSERVATION (the region
    math lives here, the verifier's single owner) — or
    {"state": "unknown", "preexisting": []} when the observation is
    unusable (None / not fresh / never observed) or the container is
    absent or ambiguous in it (the pre-drop scene then cannot bound the
    region honestly).
    """
    if obs is None:
        return {"state": "unknown", "preexisting": []}
    ts = getattr(obs, "detections_ts", 0.0) or 0.0
    if not getattr(obs, "fresh", False) or ts <= 0.0:
        return {"state": "unknown", "preexisting": []}
    dets = list(getattr(obs, "detections", None) or [])
    into = (into_label or "").lower()
    containers = [d for d in dets
                  if str(getattr(d, "label", "") or "").lower() == into]
    if len(containers) != 1:
        # 0 = absent from the pre-drop scene, >1 = cannot tell which: unknown
        # either way
        return {"state": "unknown", "preexisting": []}
    container = containers[0]
    obj = (placed_object or "").lower()
    positions = []
    for d in dets:
        if d is container:
            continue
        if str(getattr(d, "label", "") or "").lower() != obj:
            continue
        dx = abs(d.x - container.x)
        dy = abs(d.y - container.y)
        dz = abs(d.z - container.z)
        if dx <= CONTAINER_FIT_M and dy <= CONTAINER_FIT_M \
                and dz <= CONTAINER_FIT_M:
            positions.append((d.x, d.y, d.z))
    return {"state": "known", "preexisting": positions}


def _classify_reply(tool_result: str | None) -> tuple[str, str | None]:
    """Classify a tool reply line -> (kind, detail).

    kinds: 'error' (detail = message), 'contact' | 'full' | 'no_data'
    (close_adaptive verdicts), 'done' (plain DONE, detail = optional suffix,
    e.g. a state-JSON payload), 'query' (Q:), 'state' (STATE:), 'empty' (no
    reply), 'unknown'.
    """
    if tool_result is None:
        return ("empty", None)
    text = tool_result.strip()
    if not text:
        return ("empty", None)
    if "ERROR:" in text:  # any error anywhere is definitive
        idx = text.index("ERROR:")
        return ("error", text[idx + len("ERROR:"):].strip())
    if text.startswith("(drop reply empty)"):
        # the drop ran but the reply text is empty: an explicit marker, no fake
        # DONE
        # with no world signal nothing can be certified
        return ("empty_reply", None)
    first = text.splitlines()[0].strip()
    if first.startswith("DONE:"):
        detail = first[len("DONE:"):].strip()
        if detail == "contact":
            return ("contact", None)
        if detail == "full":
            return ("full", None)
        if detail == "no_data":
            return ("no_data", None)
        return ("done", detail)
    if first.startswith("DONE"):
        return ("done", None)
    if first.startswith("Q:"):
        return ("query", first[2:].strip())
    if first.startswith("STATE:"):
        return ("state", first[6:].strip())
    return ("unknown", first)


def _guard_reply(kind: str, detail: str | None) -> Verdict | None:
    """Shared prologue: reply types that decide the verdict before any obs check.

    Returns a final Verdict (error -> failure, no-world-signal replies ->
    in_progress) or None to continue with observation-based checks.
    """
    if kind == "error":
        msg = detail or "tool error"
        return Verdict.failure(f"tool reported error: {msg}", msg)
    if kind in ("query", "state", "unknown"):
        return Verdict.in_progress(
            f"re-observe; tool reply type {kind!r} carries no world post-condition signal")
    if kind == "empty":
        return Verdict.in_progress("re-observe; no tool reply recorded")
    if kind == "empty_reply":
        return Verdict.in_progress(
            "re-observe; the drop reply text was empty — no world "
            "post-condition signal (the release cannot be confirmed)")
    return None


# ── little helpers for reading observations ───────────────────────────────────


def _grip(obs: WorldObservation | None):
    """GripState or None (missing observation/grip)."""
    if obs is None:
        return None
    return getattr(obs, "grip", None)


def classify_finger_lag(grip) -> str | None:
    """BATCH 5 B1: 'held' | 'empty' | None from the MEASURED-vs-COMMANDED
    finger lag of a RIGHT-hand grip read (shape gates only; the caller owns
    freshness / ordering).

    Index 0 (thumb rotation) is EXCLUDED: on the seven run21 post-pick rows
    (4 held, 3 empty) its signed lag is -0.049..+0.041 rad in BOTH groups
    (MEASURED), so it carries no hold signal.  n = joints 1..6 with
    |commanded - joints| >= FINGER_HOLD_LAG_RAD; n >= FINGER_HOLD_MIN_JOINTS
    -> 'held', n == 0 -> 'empty', otherwise None (no claim).  Any missing /
    odd field -> None.
    """
    if grip is None or getattr(grip, "available", False) is not True:
        return None
    if getattr(grip, "side", None) != "right":
        return None
    if getattr(grip, "is_open", None) is True:
        return None
    joints = list(getattr(grip, "joints", None) or [])
    commanded = list(getattr(grip, "commanded", None) or [])
    if len(joints) != 7 or len(commanded) != 7:
        return None
    n = 0
    for i in range(1, 7):
        try:
            a, c = float(joints[i]), float(commanded[i])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(a) and math.isfinite(c)):
            return None
        if abs(c - a) >= FINGER_HOLD_LAG_RAD:
            n += 1
    if n >= FINGER_HOLD_MIN_JOINTS:
        return "held"
    if n == 0:
        return "empty"
    return None


def _finger_hold(obs: WorldObservation | None, tool_result: str | None,
                 now: float) -> str | None:
    """BATCH 5 B1: the post-lift finger-hold reading for verify_pick.

    None (the verdict path stays byte-identical) unless the grip is a fresh,
    available RIGHT-hand read taken AFTER the motion ended (stale_ts >= the
    reply's grip_state_ts), not measured-open, with 7 joints + 7 commanded.
    """
    grip = _grip(obs)
    if grip is None or not _grip_fresh(grip, now):
        return None
    motion_ts = _parse_grip_state_ts(tool_result)
    stale_ts = getattr(grip, "stale_ts", None)
    if motion_ts is None or stale_ts is None or stale_ts < motion_ts:
        return None
    return classify_finger_lag(grip)


#: the honest 'you missed' text
#: it never says grasp_failed, because a failure verdict would block the re-pick
PICK_MISSED_EVIDENCE = (
    "pick MISSED: after the lift the fingers reached their close command (no "
    "joint held off it by >= %.2f rad, MEASURED from the grip state) - nothing "
    "is between them; the hand is EMPTY, closed on nothing (hand_state=empty). "
    "Tell the operator the pick missed; never say you have it"
    % FINGER_HOLD_LAG_RAD)
#: for a place whose release passed but whose containment is unconfirmed: the
#: object is out of the hand, only the landing is unmeasured
PLACE_RELEASED_NOTE = (
    " || BUT THE OBJECT WAS RELEASED (%s): it is no longer in the hand - do "
    "not say it is still held, and do not pick or place it again unless the "
    "operator asks; ONLY whether it landed inside the container is not "
    "confirmed by measurement. LOOK at the camera image attached to this "
    "turn (or call scene_brief): if you can see where it landed, tell the "
    "operator what you see and say it is YOUR VISUAL CHECK from the camera, "
    "not a measured confirmation; if you cannot see it, say exactly that")
PLACE_TASK_DONE_SUFFIX = (
    " — this place step is complete; if it was the last step of the "
    "operator's request, the request is DONE: report it and stop (do not "
    "pick the object again)")
#: state what the fingers support: caveats read as doubt, and the lack of a
#: touch sensor is not doubt
HELD_SAY = (
    ". IT IS PICKED: tell the operator the object is in the hand - after the "
    "lift the fingers are MEASURED held open by it (there is no touch sensor, "
    "and that is not doubt); do not open the hand unless the operator asks")
#: tacked on to the evidence, never prepended
HELD_AFTER_LIFT_SUFFIX = (
    " ; post-lift finger check: fingers still held off the close command by "
    "an object (MEASURED), pick_outcome=held_after_lift")
HELD_AFTER_LIFT_SUFFIX = HELD_SAY + HELD_AFTER_LIFT_SUFFIX   # statement first


#: a quiet event-driven node with nothing in view is not a dead node
#: only these three tokens count, adjacent and in this order:
#: det_wait=still_silent@<s>; det_publishers=<n>; det_seen_since_dispatch=<n>
_DET_SILENT_RE = re.compile(
    r"det_wait=still_silent@[0-9]+(?:\.[0-9]+)?s;\s*det_publishers=(-?\d+);\s*"
    r"det_seen_since_dispatch=(\d+)(?!\d)")
_DET_FRESH_RE = re.compile(r"det_wait=fresh@")
#: appended to a pick judged off that silence
SILENT_EMPTY_NOTE = (
    " (detections stayed silent after the lift: for an event-driven node "
    "silence means nothing else is in view, so the picked object is no "
    "longer on the table - from the gateway's post-lift detections wait)")


def _silence_is_empty_scene(tool_result, obs=None) -> bool:
    """B8h: True only when the pick reply says the detections node DID publish
    after the dispatch (``det_seen_since_dispatch=1``), has at least one
    publisher (``det_publishers>=1``) and stayed silent for the whole
    post-motion wait (``det_wait=still_silent``).  Anything else -- publishers
    0 or unknown (-1), nothing ever seen, a ``det_wait=fresh`` token, no token
    at all, a non-string reply -- is False, so the old dead-node verdict
    stands.  With ``obs`` given, a packet stamped at or after the reply's
    ``grip_state_ts`` (the motion's end) is a frame the node published AFTER
    the lift: the silence is over and that frame is judged the normal way.
    The caller additionally requires hold == "held"."""
    if not isinstance(tool_result, str) or _DET_FRESH_RE.search(tool_result):
        return False
    hits = _DET_SILENT_RE.findall(tool_result)
    if len(hits) != 1:
        return False
    if obs is not None:
        motion_ts = _parse_grip_state_ts(tool_result)
        det_ts = getattr(obs, "detections_ts", 0.0) or 0.0
        if motion_ts is None or det_ts >= motion_ts:
            return False
    pubs, seen = int(hits[0][0]), int(hits[0][1])
    return pubs >= 1 and seen == 1


def _detections_alive(obs: WorldObservation | None, now: float) -> bool:
    """The detections CHANNEL is alive: the node ever spoke AND its last
    packet is within DETECTIONS_ALIVE_S (world-truth round 5, fix 3).

    Distinct from packet freshness: a quiet-but-running node stays alive
    while its observation ages out of freshness (silence == empty scene
    when the node is alive).  'Never spoke' or silent beyond the window ==
    DEAD node — nothing is confirmable.
    """
    if obs is None:
        return False
    ts = getattr(obs, "detections_ts", 0.0) or 0.0
    return ts > 0.0 and (now - ts) <= DETECTIONS_ALIVE_S


def _detections_dead_cause(obs: WorldObservation | None, now: float) -> str:
    """WHY the detections channel is dead, as a MEASURED phrase.

    ``_detections_alive`` collapses three DIFFERENT situations into one
    ``False``, and every verdict built on it used to name only the third:

      (a) ``obs is None`` — the observe READ returned nothing at all, so no
          channel was ever sampled.  This is a transport/lane fault and the
          one most likely to be blamed on the wrong component: no amount of
          staring at the detections node explains it.
      (b) ``detections_ts == 0.0`` — the node NEVER spoke since gateway
          start.
      (c) a positive-but-old ``detections_ts`` — the node spoke and then
          went quiet for a MEASURABLE number of seconds.

    Only (c) has an age that is a fact, so only (c) prints a number.
    WHY THIS EXISTS (2026-10-01, MEASURED on Stuart): a descended pick whose
    engine reply was ``DONE pick executed; lane_state=up`` — the grasp half
    ENGINE-VERIFIED — was reported as ``re-observe; detections node silent
    (never spoke, or silent beyond the liveness window)``.  That sentence is
    indistinguishable from the sibling ``target still visible`` branch, which
    is a MEASURED NEGATIVE about the grasp, so a dead sensor and a bottle
    still on the table read the same to the operator.  Naming the wrong
    cause, or naming no cause, sends the operator to the wrong machine.
    """
    if obs is None:
        return ("no observation at all — the observe read returned nothing, "
                "so the channel was never sampled")
    ts = getattr(obs, "detections_ts", 0.0) or 0.0
    if ts <= 0.0:
        return "the node never spoke (detections_ts == 0.0)"
    return ("no /object_detections_body packet for %.1f s (liveness window "
            "DETECTIONS_ALIVE_S = %.1f s)" % (now - ts, DETECTIONS_ALIVE_S))


def _state_available(detail: str | None) -> bool | None:
    """Read the 'available' flag out of a grip 'state [side]' JSON reply
    (DONE:<json>). None when the payload is not a state reply.
    """
    if not detail:
        return None
    try:
        payload = json.loads(detail)
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    avail = [v.get("available") for v in payload.values()
             if isinstance(v, dict) and "available" in v]
    if not avail:
        return None
    return bool(all(avail))


def _ts_fresh(ts: float | None, now: float) -> bool:
    """A state-read timestamp is fresh when it exists and is at most
    GRIP_STATE_MAX_AGE_S old (vs the pinned `now`)."""
    return ts is not None and ts > 0.0 and (now - ts) <= GRIP_STATE_MAX_AGE_S


def _verdict_fresh(ts: float | None, now: float) -> bool:
    """A folded close_adaptive verdict (its own contact_ts) is fresh when it
    exists and is at most CLOSE_CONTACT_WINDOW_S old (world-truth round 4,
    fix 2: a daemon'd grip's stale folded contact must not confirm)."""
    return ts is not None and ts > 0.0 \
        and (now - ts) <= CLOSE_CONTACT_WINDOW_S


def _grip_fresh(grip, now: float) -> bool:
    """The grip's state read is fresh (a never-read grip is not)."""
    if grip is None:
        return False
    return _ts_fresh(getattr(grip, "stale_ts", None), now)


def _resolve_target(contract) -> tuple[str | None, str | None]:
    """Target hint from contract.context -> (label, ref); (None, None) if absent.

    Conventions (context is free-form; hints separated by ';' or ','):
      target=<ref>          scene-graph ref, e.g. 'target=bottle_3'
      target_label=<label>  detection label, e.g. 'target_label=bottle'
      label=<label>         alias of target_label
    'target=' wins over label hints when both are present. Without a hint the
    verifier cannot name the picked object, so 'object gone' is only
    confirmable against an empty scene.
    """
    context = getattr(contract, "context", "") or ""
    m = re.search(r"target\s*=\s*([A-Za-z0-9_.\-]+)", context)
    if m:
        return None, m.group(1)
    m = re.search(r"target_label\s*=\s*([A-Za-z0-9_.\-]+)", context)
    if m:
        return m.group(1), None
    m = re.search(r"(?<![A-Za-z0-9_])label\s*=\s*([A-Za-z0-9_.\-]+)", context)
    if m:
        return m.group(1), None
    return None, None


#: an absent label outside the active vocabulary proves nothing
VOCAB_INACTIVE_EVIDENCE = (
    "re-observe; target label is not in the active vocabulary")


def _label_vocab_inactive(obs: WorldObservation | None,
                          label: str | None) -> bool:
    """True when ``label`` names an EXTRA class the detector is not running.

    D13: a hint label outside ``known_labels.TRACKED_CLASSES`` can only ever
    be published by the YOLOE backend with that label in its extras.  When
    the observation's ``vocab`` DICT does not show that (backend not
    'yoloe', or the label not in ``extras``), the label's absence from the
    packet is NOT evidence that the object is gone — the detector simply
    cannot see it — so the caller must treat it as unconfirmable.
    A base label (or a non-str / absent label) is never affected: today's
    logic runs unchanged.

    WHAT COUNTS AS A BASE LABEL (round-2 fix, reviewer finding P10-H1).  The
    raw hint text is NOT the only form a base object arrives in.  The gateway
    hint mirror (`gateway_node._pick_hint_label`) forwards the model's label
    verbatim — no sighting gate, no stem — and `make_pick` ADMITS a scene ref
    ('bottle_1', brain/agent/tools.py resolve_ref -> ref label) and a target
    alias ('me' -> 'person').  So a label is base when its text, its
    '_<n>'-stripped stem (mirrors brain/hooks.py), that stem with '_' read
    as a space (SceneGraph sanitizes 'cell phone' into the ref key
    'cell_phone_1'), or its TARGET_ALIASES value is a tracked class.
    With NO vocabulary block (`vocab` not a dict: the shipped OFF form, field
    None) the rule is OFF for EVERY label (returns False): every switch
    absent keeps today's verdict byte-for-byte, including the regex-truncated
    first word of a multi-word hint ('potted' from 'potted plant').
    """
    if not isinstance(label, str):
        return False
    text = label.strip().lower()
    stem = re.sub(r"_\d+$", "", text)
    cands = (text, stem)
    # a narrowed detector cannot see a label outside its set, so that label's
    # absence is no evidence
    _v = getattr(obs, "vocab", None)
    _s = _v.get("strict") if isinstance(_v, dict) else None
    if isinstance(_s, (list, tuple)) and _s:
        allowed = set(str(x).strip().lower() for x in _s)
        names = set(cands) | {stem.replace("_", " ")}
        names |= set(_TARGET_ALIASES.get(c) for c in list(names)
                     if _TARGET_ALIASES.get(c))
        if not (names & allowed):
            return True
    for cand in cands + (stem.replace("_", " "),):
        if (cand in _TRACKED_CLASSES
                or _TARGET_ALIASES.get(cand) in _TRACKED_CLASSES):
            return False
    vocab = getattr(obs, "vocab", None)
    if not isinstance(vocab, dict):
        # no vocab block: the rule is off and the normal label logic applies
        # the hint regex chops at the first space ('potted plant' -> 'potted'),
        # so keying on the full label would fail real picks
        return False
    if vocab.get("backend") == "yoloe":
        extras = vocab.get("extras")
        if isinstance(extras, (list, tuple)):
            for extra in extras:
                if (isinstance(extra, str)
                        and extra.strip().lower() in cands):
                    return False
    return True


def _target_present(obs: WorldObservation | None, label: str | None,
                    ref: str | None, silence_means_empty: bool = True,
                    ) -> bool | None:
    """Is the target still visible in the latest detections packet?

    Callers gate on channel-liveness FIRST (_detections_alive): this
    function reads the latest packet only.

    True  - a matching detection is present
    False - confirmed absent (empty scene, or no detection matches the hint)
    None  - unconfirmable: scene non-empty but no target hint, a ref hint
            with no ref-bearing detections to compare against, or an EMPTY
            scene under silence_means_empty=False (silence is then never
            empty-scene evidence), or (D13) a label hint naming an extra
            class the detector is not currently running
    """
    if ref is None and _label_vocab_inactive(obs, label):
        return None  # inactive extra's absence proves nothing
    dets = getattr(obs, "detections", None) or []
    if not dets:
        # an empty packet counts only with the switch on
        return False if silence_means_empty else None
    if ref is not None:
        if not any(d.ref is not None for d in dets):
            return None  # no refs in the scene, so absence is unconfirmable
        return any(d.ref == ref for d in dets)
    if label is not None:
        return any((d.label or "").lower() == label.lower() for d in dets)
    return None


# ── the verifier itself ───────────────────────────────────────────────────────

_HANDLER_METHOD = {
    "pick": "verify_pick",
    "place": "verify_place",
    "navigate": "verify_navigate",
    "open_hand": "verify_open_hand",
    "close_hand": "verify_close_hand",
    "pose": "verify_pose",
    "turn_around": "verify_turn_around",
}


# grasp->brain token for a blocked close with no tactile proof
# the gateway wraps it into engine_reply=
_GRASP_CAVEAT_RE = re.compile(r"\bgrasp_evidence=blocked_no_tactile\b")


#: engine wire token `; zfloor_limited=<s>`, where s is the smallest applied
#: scale on the reply line of the clamped move
_ZFLOOR_LIMITED_RE = re.compile(r"zfloor_limited=(\d+(?:\.\d+)?)")
_ZFLOOR_LIMITED_TOKEN_RE = re.compile(r";\s*zfloor_limited=\d+(?:\.\d+)?")
#: the note goes on the evidence, never on the status
ZFLOOR_NOTE = (" — NOTE: the arm stopped at the table floor after %d%% of this "
               "move; it is clear of the table but NOT fully at the %s")


def _zfloor_limited_note(target):
    """B8g G3: wraps verify_pose ('requested pose') and verify_place ('drop
    point').  The token is removed before the verdict is computed, so the status
    is the one the reply earns without it; when a token with s < 0.98 is
    present, the note is appended to the evidence of the verdict returned."""

    def deco(method):
        def wrapper(self, contract, obs, tool_result, *args, **kwargs):
            text = tool_result if isinstance(tool_result, str) else ""
            scales = [float(s) for s in _ZFLOOR_LIMITED_RE.findall(text)]
            if not scales:
                return method(self, contract, obs, tool_result, *args, **kwargs)
            verdict = method(self, contract, obs,
                             _ZFLOOR_LIMITED_TOKEN_RE.sub("", text), *args, **kwargs)
            if min(scales) >= 0.98:
                return verdict
            return Verdict(status=verdict.status,
                           evidence=(verdict.evidence or "") + ZFLOOR_NOTE % (
                               round(100 * min(scales)), target),
                           failure_reason=verdict.failure_reason)
        wrapper.__name__ = method.__name__
        wrapper.__doc__ = method.__doc__
        return wrapper
    return deco


class SkillVerifier:
    """Deterministic post-condition judge for pick/place/navigate/open_hand/close_hand.

    verify(contract, obs, tool_result[, now]) dispatches on the contract name
    to the per-skill method. Post-conditions per brain/tool_contract.py
    CONTRACTS:
      pick     - 'object gone from source AND grip contact' — PICK GRASP LAW
                 (world-truth round 4): the grasp half is confirmed ONLY by
                 the gateway's post-pick lane poll (lane_state=up + fresh
                 grip_state_ts <=> the engine's DONE was engine-verified);
                 lane down/unknown -> in_progress 're-observe; state [side]'
                 (blind mode — no evidence available); a serialized close
                 verdict NEVER certifies a pick (tabletop-only; a floor pick
                 is a capability-block, refused before any motion)
      place    - 'grip open AND object out of hand' — the RELEASE-ONLY
                 chain (no in-bin claim, no region check) is the round-5
                 HELD-THEN-OPEN: drop DONE + post state fresh +
                 MEASURED-OPEN (is_open True) AND (pre-drop is_open False
                 [transition] OR the hold was created by an engine-verified
                 pick).  Verifier v2 (ADDITIVE): when the reply carries the
                 agent-lane into: annotation (into_ref/into_label/
                 placed_object) a CONTAINMENT half runs after the release
                 chain — the placed object's projected position must lie
                 inside the container's containment region (CONTAINER_FIT_M
                 around the container's projected detection, torso frame);
                 success ONLY on demonstrated containment across TWO
                 consecutive post-drop in-region observations of a NEW
                 arrival on a POST-DROP FRESH packet (never one
                 instantaneous detection, never a pre-existing same-label
                 object); the stationarity record is SCOPED TO THE TASK
                 EXECUTION — keyed on the tool layer's drop epoch
                 (dropped_at) so a new drop of the same container/object
                 resets the record (fix round 2, D4) — its pre-existing
                 set is anchored on the tool layer's PRE-DROP observation
                 (predrop_state/predrop_obj; D5), never the first
                 post-drop frame, and the two agreeing observations must
                 be distinct frames within RE_OBSERVE_WINDOW_S (first_ts
                 is read); any ambiguity, staleness or unavailable
                 projection -> in_progress 're-observe' (never a guessed
                 in-bin)
      navigate - walk_status == REACHED ONLY within the dispatch epoch
                 (NAVIGATE EPOCH GATE: NAV_EPOCH + LATEST_WALK_TS + a
                 non-REACHED phase in PHASES_SEEN defeat stale/hold REACHED);
                 NEVER odom (/unitree/slam_mapping/odom does not exist);
                 'stop' claims no goal post-condition (no verdict, fix E)
      open_hand - grip open by MEASURED state: fresh read + is_open True
                 (CAUSAL measured-open reference — forms only from reads
                 after a gateway-dispatched open command) + NO literal
                 contact verdict on file; open_both checks the LEFT hand
                 independently too (fix 5)
      close_hand- grip closed (literal close_adaptive DONE:contact folded
                 with its verdict ts within CLOSE_CONTACT_WINDOW_S on a
                 FRESH read, or the full-stroke DONE:full — never success on
                 an unobserved/offline left hand, fix 4; a stale folded
                 contact never confirms, round-4 fix 2)

    Conservative rule: any missing/unfresh data, offline grip, indistinguishable
    target, or no-world-signal reply (Q:/STATE:) -> in_progress with evidence
    starting 're-observe'. Never false success.

    Instance-held state: the navigate epoch gate remembers the dispatch epoch
    of the last success (refuse re-success on the SAME epoch). Call
    :meth:`reset` to clear it. One instance must be shared across calls
    (the gateway keeps a single SkillVerifier).
    """

    def __init__(self, silence_means_empty: bool = True) -> None:
        self._nav_last_success_epoch = 0.0
        #: place-into record: mirror of the navigate epoch guard, keyed by
        #: (into_ref, into_label, placed_object) and scoped to the drop epoch,
        #: so one task never pairs with another
        self._containment_record: dict = {}
        #: silence switch: True means an empty detection set is confirmable
        self.silence_means_empty = silence_means_empty

    def reset(self) -> None:
        """Clear instance-held verification state."""
        self._nav_last_success_epoch = 0.0
        self._containment_record = {}

    def last_confirmed_epoch(self) -> float:
        """The navigate dispatch epoch this instance confirmed (0.0 = none).

        The gateway reads this from its ONE persistent SkillVerifier so the
        navigate reuse gate can force a fresh epoch after a confirmed arrival
        (fix spec A-e).
        """
        return self._nav_last_success_epoch

    def verify(self, contract, obs, tool_result: str,
               now: float | None = None) -> Verdict:
        """Verify one tool invocation. `contract` is a ToolContract (or its name).

        ``now`` (optional) pins the clock for freshness checks — tests and
        replay use it; callers without it get time.time().  hooks.py calls
        this with the classic 3 arguments.
        """
        name = contract.name if isinstance(contract, ToolContract) else str(contract)
        method = getattr(self, _HANDLER_METHOD.get(name, ""), None)
        if method is None:
            return Verdict.in_progress(
                f"re-observe; no verifier registered for tool {name!r}")
        now = time.time() if now is None else now
        return method(contract, obs, tool_result, now=now)

    # -- pick: object gone AND grip contact? ---------------------------------

    def verify_pick(self, contract, obs, tool_result: str,
                    now: float | None = None) -> Verdict:
        """Post-condition: object gone from source AND grip contact.

        PICK GRASP LAW (world-truth round 4): during a REAL pick the engine
        closes the grip over ITS OWN socket — the gateway never sees the
        close_adaptive verdict, so a serialized gateway-close verdict can
        NEVER certify a pick.  An engine pick DONE is COMMAND evidence only:
        in BLIND MODE (palm DDS lane down) the engine skips its grasp
        verification and returns DONE anyway; when the lane is UP the engine
        VERIFIES its own grasp (quadrant/palm-data check) and fails with
        ERROR:grasp_failed on an air-grab.  The gateway's immediate post-pick
        ``state [side]`` poll CANNOT see the engine's grip-socket traffic,
        but the poll's per-side available flag IS the lane-health probe:
        ``lane_state=up`` (available True on a poll no older than
        GRIP_STATE_MAX_AGE_S) <=> this engine DONE was engine-verified.
        The grasp half is confirmed IFF ``lane_state == up`` with a fresh
        ``grip_state_ts``; lane_state down/unknown -> in_progress
        're-observe; state [side]' (blind mode — no evidence available).
        The literal folded-close-verdict contact path is close_hand evidence
        ONLY and is NOT pick evidence (removed).

        Tool-level failures are definitive (engine ERROR -> failure,
        close_adaptive 'DONE:full' = closed without contact).  The target
        (contract.context 'target='/'target_label=' hint, else only an empty
        scene) must be absent from the latest detections packet —
        DETECTIONS-ALIVE GATE (world-truth round 5, fix 3): target-gone is
        confirmable when the channel is ALIVE (ever spoke, within
        DETECTIONS_ALIVE_S) and the latest packet lacks the target ref (or
        is empty — silence == empty scene); a dead node (never spoke, or
        silent beyond the liveness window) -> in_progress 're-observe;
        detections node silent'.  A fresh NON-EMPTY packet is NEVER required
        for target-gone.
        """
        now = time.time() if now is None else now
        if _task_was_cancelled(tool_result):
            # the cancel phase decides: 'nothing picked up' only before the
            # close, otherwise the hand state stays unknown
            _where = _self_state.parse_cancel_where(tool_result)
            if _self_state.cancel_hand_state(_where) == "empty":
                return Verdict.failure(
                    "the operator cancelled this pick while it was running - "
                    "the arm was stopped and NOTHING was picked up (hand_state="
                    "empty); this was an intentional stop, not a failed grasp. "
                    "Acknowledge it. The stop ended ONLY this pick: do not "
                    "retry it on your own, but treat the operator's next request "
                    "as a new instruction and carry it out (navigate or pick as "
                    "needed, without asking permission)",
                    "cancelled_by_operator")
            return Verdict.failure(
                "the operator cancelled this pick while it was running "
                "(phase %s) - the arm was stopped; the hand state is UNKNOWN "
                "(hand_state=unknown): it may or may not hold the object - "
                "check, or call open_hand. This was an intentional stop, not "
                "a failed grasp. "
                "Acknowledge it. The stop ended ONLY this pick: do not "
                "retry it on your own, but treat the operator's next request "
                "as a new instruction and carry it out (navigate or pick as "
                "needed, without asking permission)" % (_where or "?"),
                "cancelled_by_operator")
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        if kind == "full":
            return Verdict.failure(
                "pick closed to full stroke without contact",
                "closed to full without contact")
        if kind == "no_data":
            return Verdict.in_progress(
                "re-observe; close_adaptive returned no tactile data")
        lane = _parse_lane_state(tool_result)
        grip_ts = _parse_grip_state_ts(tool_result)
        if lane == "up" and _ts_fresh(grip_ts, now):
            grasp = "engine-verified (lane up)"
            # cloud hold token, lane-up only: STILL_ON_TABLE / MOVED are
            # negative evidence, ABSENT_UNPROVEN / UNKNOWN prove nothing and
            # only add words
            # it never promotes a verdict; without a token nothing changes
            _pc = _parse_pc_hold(tool_result)
            if _pc in ("STILL_ON_TABLE", "MOVED"):
                return Verdict.failure(
                    "point-cloud hold check: the target %s" %
                    ("never left its footprint" if _pc == "STILL_ON_TABLE"
                     else "was knocked"),
                    "pc_hold=%s" % _pc)
            if _pc in ("ABSENT_UNPROVEN", "UNKNOWN"):
                grasp = "engine-verified (lane up); point cloud: no evidence either way"
            _eng = _parse_engine_reply(tool_result)
            if _eng is not None and not _eng.startswith("DONE"):
                return Verdict.in_progress(
                    "re-observe; state [side] - the pick reply's "
                    "engine_reply=%r is not a DONE, so lane_state=up (a "
                    "DDS-writer-exists flag) is the ONLY grasp evidence; "
                    "cannot confirm the pick grasp" % _eng)
        else:
            # blind: no grasp evidence from the lane, so 'state [side]' is the
            # real path; a later poll may find the lane up, or a grasp failure
            if lane is None:
                return Verdict.in_progress(
                    "re-observe; state [side] — the pick reply carries no "
                    "lane_state annotation; grasp evidence: none; cannot "
                    "confirm the pick grasp")
            if not _ts_fresh(grip_ts, now):
                return Verdict.in_progress(
                    "re-observe; state [side] — the post-pick lane poll is "
                    "stale (grip_state_ts older than GRIP_STATE_MAX_AGE_S); "
                    "grasp evidence: none; cannot confirm the pick grasp")
            return Verdict.in_progress(
                "re-observe; state [side] — grasp evidence: none "
                "(lane_state=%s — the palm lane was not up during the pick, "
                "so the engine DONE carries no verified grasp); cannot "
                "confirm the pick grasp" % lane)
        # lane up with a bare DONE and the target gone also matches an empty
        # close, so the finger lag separates the two
        hold = _finger_hold(obs, tool_result, now)
        if hold == "empty":
            return Verdict.failure(PICK_MISSED_EVIDENCE,
                                   "pick_missed_empty_hand")
        # silence after a lift with the fingers held means an empty scene, not
        # a dead node; a stale pre-lift packet never feeds the object-gone half
        _silent_empty = (hold == "held"
                         and _silence_is_empty_scene(tool_result, obs))
        label, ref = _resolve_target(contract)
        if not _detections_alive(obs, now) and not _silent_empty:
            if hold == "held":
                # this never upgrades a status (a limp hand can freeze off
                # command)
                return Verdict.in_progress(
                    "re-observe; MAY BE HOLDING - after the lift the fingers "
                    "are still held off their close command by an object "
                    "(MEASURED from the grip state), but the detections node "
                    "is silent (%s), so the object-gone half could NOT BE "
                    "MEASURED. Grasp half: %s. This silence is never evidence "
                    "against the grasp: do not re-pick (a re-pick re-opens "
                    "the hand and would drop what it may be holding); place "
                    "is allowed if the operator asked for it."
                    % (_detections_dead_cause(obs, now), grasp))
            # unmeasurable, not a measured negative: the grasp half is established
            # and only the object-gone half is missing
            # the rule still forbids success here, because a DONE on an air-close
            # can read as success, and a false success is the worse error
            # a dead channel is not grasp evidence either
            # re-observe cannot help while the loop only feeds newer packets, so
            # get the channel back and never re-pick
            return Verdict.in_progress(
                "re-observe; detections node silent (%s) — the object-gone "
                "half of the pick post-condition could NOT BE MEASURED, so "
                "this pick is UNCONFIRMED, not failed. Grasp half: %s. "
                "This is a PERCEPTION-LANE fault, and never evidence against "
                "the grasp (a target still visible would have said so): "
                "re-observing cannot resolve it until the detections node "
                "publishes again, and a re-pick is NOT warranted on this "
                "evidence alone — the hand may already be holding the object."
                % (_detections_dead_cause(obs, now), grasp))
        present = _target_present(obs, label, ref,
                                  silence_means_empty=self.silence_means_empty)
        lifted_note = ""
        if _silent_empty:
            present = False
            lifted_note = SILENT_EMPTY_NOTE
        # every same-label detection is provably another object
        bystander_note = ""
        if present is True and ref is None and _only_bystanders(
                obs, label,
                _parse_pick_point(getattr(contract, "context", "") or "",
                                  tool_result),
                _parse_preaction_dets(getattr(contract, "context", "") or "")):
            present = False
            bystander_note = (
                " (the %s still seen on the table is a DIFFERENT %s that "
                "was already there or has another colour; the one you "
                "picked is the one in the hand)" % (label, label))
        if present is True:
            # a label match cannot tell the table from a parked hand
            # with pick_point, far and above means lifted
            # no annotation keeps the conservative answer
            lifted, worst = _target_lifted_with_hand(
                obs, label, ref,
                _parse_pick_point(getattr(contract, "context", "") or "",
                                  tool_result))
            # the fingers are held and a matching detection sits near the pick
            # point but risen: that is the held object, in view
            if not lifted and hold == "held":
                lifted, worst = _target_held_in_view(
                    obs, label, ref,
                    _parse_pick_point(getattr(contract, "context", "") or "",
                                      tool_result))
            if not lifted:
                if hold == "held":
                    return Verdict.in_progress(
                        "re-observe; target still visible in the latest "
                        "detections — but the fingers are still held off the "
                        "close command by an object (MEASURED): you MAY be "
                        "holding something; a same-label object is still "
                        "detected near the pick point")
                return Verdict.in_progress(
                    "re-observe; target still visible in the latest detections")
            if worst[1] <= 0.0:
                lifted_note = (
                    " (the pick point is empty - the nearest same-label "
                    "detection is %.2f m away, another object - and the "
                    "fingers are MEASURED held open by an object: the target "
                    "is in the hand, out of the camera's view)" % worst[0])
            else:
                lifted_note = (
                    " (the same-label detection now sits %.2f m from and %.2f m "
                    "above the pick point: it moved up with the hand, MEASURED "
                    "from detections)" % worst)
            present = False
        if present is None:
            if ref is None and _label_vocab_inactive(obs, label):
                vocab = getattr(obs, "vocab", None)
                # only advise a tool the detector vocabulary actually supports
                if not isinstance(vocab, dict):
                    return Verdict.in_progress(
                        "%s (hint label %r; vocabulary backend None, extras "
                        "None) — its absence from the detections cannot be "
                        "read as the object gone; the label is not a "
                        "detectable class on this robot; re-observe" % (
                            VOCAB_INACTIVE_EVIDENCE, label))
                return Verdict.in_progress(
                    "%s (hint label %r; vocabulary backend %r, extras %r) — "
                    "its absence from the detections cannot be read as the "
                    "object gone; re-observing alone cannot settle this — "
                    "the label must be in the detector's vocabulary first "
                    "(look_for), then re-observe" % (
                        VOCAB_INACTIVE_EVIDENCE, label,
                        vocab.get("backend"), vocab.get("extras")))
            return Verdict.in_progress(
                "re-observe; cannot confirm object gone (no target ref/label "
                "in contract context and scene non-empty%s)"
                % (", or silence is not empty-scene evidence "
                   "(silence_means_empty=False)" if not self.silence_means_empty
                   else ""))
        # an absent target reads as good news, so the wording says so; no check
        # is relaxed
        # a bare DONE is a blocked close, more than 0.06 rad short of the close
        # preset, and the explicit token counts the same way
        # no engine_reply uses the plain wording
        lifted_note += bystander_note  # stays empty unless bystanders are ruled out
        if (_GRASP_CAVEAT_RE.search(tool_result or "")
                or _engine_reply_is_bare_done(_parse_engine_reply(tool_result))):
            return Verdict.success(
                "grasp evidence: %s; fingers closed on an object and were "
                "blocked short of the close preset (grasp_evidence="
                "blocked_no_tactile: NO tactile confirmation); the object has left the table%s and MAY be in "
                "the hand (INFERRED: lane up + target gone; no in-hand sensor "
                "exists) — pick complete%s" % (
                    grasp, lifted_note,
                    HELD_AFTER_LIFT_SUFFIX if hold == "held" else ""))
        return Verdict.success(
            "grasp evidence: %s; the object has left the table%s and MAY be "
            "in the hand (INFERRED: lane up + target gone; no in-hand sensor "
            "exists) — pick complete%s" % (
                grasp, lifted_note,
                HELD_AFTER_LIFT_SUFFIX if hold == "held" else ""))

    # -- place: grip open AND object out of hand? ----------------------------

    @_zfloor_limited_note("drop point")
    def verify_place(self, contract, obs, tool_result: str,
                     now: float | None = None) -> Verdict:
        """Post-condition: grip open AND object out of hand (release-only).

        HELD-THEN-OPEN (world-truth round 4): post-drop contact == False is
        GUARANTEED — the drop itself clears both sides' close verdicts, so
        contact==False alone would certify ANY drop by command-inference,
        not observation.  Success requires ALL of:
          (a) drop DONE (any engine ERROR is a definitive failure);
          (b) the post state read FRESH (within GRIP_STATE_MAX_AGE_S),
              available, and the hand MEASURED-OPEN (is_open True — joints
              at the trusted measured-open reference);
          (c) the hold is established by EITHER:
              - a TRANSITION: the PRE-DROP fresh state poll measured the
                hand in a NON-open shape (``prev_is_open=<side>:False``) —
                the drop moved it from closed to open; OR
              - a VERIFIED HOLD: ``prev_pick_verified=<side>:True`` — the
                gateway's per-side bookkeeping records that the hold was
                created by an engine-verified pick (lane up, same side,
                this session; recorded at pick time, cleared on the next
                drop / open / new pick).
        NO success on measured-open alone when the hold itself is unverified
        AND no transition was observed — the drop's open is then
        command-inferred, not observed.  A missing/None pre-drop judgment, a
        stale/offline read, or no trusted open reference yet (bootstrap) is
        in_progress 're-observe'.  Verifier v2 (ADDITIVE): when the reply
        carries the agent-lane into: annotation (into_ref/into_label/
        placed_object — see _parse_place_into) a CONTAINMENT half follows
        the release chain: the placed object's projected position must lie
        inside the container's containment region (CONTAINER_FIT_M half-
        extent around the container's projected detection, torso frame).
        The containment half enforces the fix-round honesty gates: a
        POST-DROP FRESH packet (POST_DROP_FRESHNESS_S), a stateful
        two-observation stationarity record SCOPED TO THE TASK EXECUTION
        (the drop epoch dropped_at re-anchors it on every new drop), a
        RE_OBSERVE_WINDOW_S bound between the two agreeing frames
        (first_ts is read), and the pre-existing-object exclusion anchored
        on the tool layer's PRE-DROP observation (an unchanged in-region
        same-label object is never success — see
        _verify_place_containment).  Without into: this stays
        RELEASE-ONLY — NEVER claims in-bin, no region check runs.
        """
        now = time.time() if now is None else now
        if _task_was_cancelled(tool_result):
            return Verdict.failure(
                "the operator cancelled this place while it was running - the "
                "arm was stopped and the release did NOT complete; the object "
                "may still be in the hand. This was an intentional stop, not "
                "a failed release. Acknowledge it. The stop ended ONLY this "
                "place: do not retry it on your own, but treat the "
                "operator's next request as a new instruction and carry "
                "it out",
                "cancelled_by_operator")
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        prev_is_open = _parse_prev_is_open(tool_result)
        pick_verified = _parse_pick_verified(tool_result)
        if prev_is_open is None and pick_verified is None:
            return Verdict.in_progress(
                "re-observe; no pre-drop evidence recorded for the drop "
                "(no prev_is_open / prev_pick_verified annotations; cannot "
                "confirm the hand held before it)")
        side = (prev_is_open or pick_verified)[0]
        grip = _grip(obs)
        if grip is None:
            return Verdict.in_progress("re-observe; grip state missing")
        if str(grip.side).lower() != side.lower():
            return Verdict.in_progress(
                "re-observe; observed grip side %r does not match the "
                "pre-drop evidence side %r" % (grip.side, side))
        if not _ts_fresh(getattr(grip, "stale_ts", None), now):
            return Verdict.in_progress(
                "re-observe; grip state read stale (older than "
                "GRIP_STATE_MAX_AGE_S) — cannot confirm the release")
        if not grip.available:
            return Verdict.in_progress(
                "re-observe; grip offline (state [side] available=False)")
        release_measured_open = False
        if grip.is_open is not True:
            # the lift parks the holding hand closed before DONE, because open
            # fingers sweeping down would hit the table, so is_open can never
            # hold here
            # the release is measured at drop time instead, and that
            # release_open=<side>:<bool> counts
            rel = _parse_release_open(tool_result)
            if not (rel is not None and rel[0].lower() == side.lower()
                    and rel[1] == "True"):
                return Verdict.in_progress(
                    "re-observe; post-drop hand not measured-open (is_open=%s — "
                    "no trusted measured-open reference yet or joints away from "
                    "it); cannot confirm the release" % grip.is_open)
            release_measured_open = True
        open_evidence = (
            "the engine measured the hand OPEN AT THE RELEASE (step 6, finger "
            "joints vs the OPEN preset in measured motor-q); the LIFT closeout "
            "parked it closed afterwards (PLACE_PARK_HAND_CLOSED)"
            if release_measured_open else "grip measured-open after drop")
        transition = prev_is_open is not None \
            and prev_is_open[0].lower() == side.lower() \
            and prev_is_open[1] == "False"
        verified_hold = pick_verified is not None \
            and pick_verified[0].lower() == side.lower() \
            and pick_verified[1] == "True"
        if not (transition or verified_hold):
            return Verdict.in_progress(
                "re-observe; the hold is unverified (pre-drop is_open=%s, "
                "prev_pick_verified=%s:%s) and no open transition was "
                "observed (pre-drop hand not measured in a non-open shape) — "
                "cannot confirm the drop released anything"
                % (prev_is_open[1] if prev_is_open is not None else "None",
                   pick_verified[0] if pick_verified is not None else side,
                   pick_verified[1] if pick_verified is not None else "False"))
        hold_evidence = ("pre-drop measured non-open (is_open False)"
                         if transition else
                         "hold from an engine-verified pick (lane up)")
        into_ref, into_label, placed_object = _parse_place_into(tool_result)
        if into_ref is not None:
            _cv = self._verify_place_containment(
                obs, now, into_ref, into_label, placed_object, hold_evidence,
                tool_result, open_evidence)
            # a release read as in_progress makes the model say the object is
            # still in the hand, so append the release note
            if _cv.status == "in_progress":
                _cv = Verdict.in_progress(
                    (_cv.evidence or "") + PLACE_RELEASED_NOTE % (open_evidence,))
            return _cv
        return Verdict.success(
            "drop DONE, %s; hold evidence: %s — "
            "object released (release-only, no in-bin claim)%s"
            % (open_evidence, hold_evidence, PLACE_TASK_DONE_SUFFIX))

    def _verify_place_containment(
        self, obs, now: float, into_ref: str, into_label: str | None,
        placed_object: str | None, hold_evidence: str,
        tool_result: str | None = None,
        open_evidence: str = "grip measured-open after drop") -> Verdict:
        """Verifier v2 containment half (place into:<container_ref>).

        Runs ONLY after the release chain above passed and the reply
        carried the into: annotation.  Containment is demonstrated when the
        placed object's PROJECTED position (same depth-projector data the
        observation already carries — the observation has no pixel box and
        no refs on detections, so identity is by LABEL plus the temporal
        evidence this pass enforces, never fabricated) lies within
        CONTAINER_FIT_M (torso-frame half-extent) of the container's
        projected detection.  Fix-round honesty ladder (D1-D3 + nit ii):
          (a) missing into_label/placed_object annotations -> in_progress
              (the reply named the container but the tool layer did not
              name the object / container label — nothing to check);
          (b) detections channel not alive (DETECTIONS_ALIVE_S) ->
              in_progress (silence == empty scene; no projection available);
          (c) POST-DROP FRESHNESS (fix D2): the packet must be at most
              POST_DROP_FRESHNESS_S old — a packet up to 10 s old can
              predate the drop's completion and must not certify it;
          (d) container by label (lowered match): 0 candidates ->
              in_progress 'not in the current scene'; when the tool layer
              annotated the resolved scene-graph ref's fresh position
              (into_x/into_y/into_z) the same-label detection nearest it
              within PREEXISTING_ASSOC_M is used ('ref-unique' in the
              evidence — the scene graph identified ONE container); >1
              candidates without a ref-unique match -> in_progress 'cannot
              tell which received the drop';
          (e) object (by placed_object label, container identity excluded)
              0 candidates -> in_progress 'not observed after the drop'
              (occluded or out of frame — containment not demonstrated);
          (f) object >1 candidate -> in_progress ambiguity;
          (g) STATIONARITY + PRE-EXISTING (fixes D2/D3, fix round 2
              D4/D5): ONE observation never certifies.  A stateful
              per-(container, object) record (mirror of the navigate epoch
              guard) is SCOPED TO THE TASK EXECUTION: the tool layer
              stamps the drop epoch (dropped_at) onto every into: place
              reply, and the record re-anchors whenever the epoch changes
              — a NEW drop of the same key resets the record, so the
              previous task's frames can never pair with this task's first
              observation (the containment state leak).  The record's
              PRE-EXISTING set is built from a REAL PRE-DROP observation
              (the tool layer's predrop_state/predrop_obj annotations,
              read from the observation it held before executing the
              drop) — NEVER from the first post-drop frame; when no
              pre-drop observation was usable the set is UNKNOWN and
              success additionally requires displacement evidence (the
              object observed OUTSIDE the region at least once in this
              task's sequence, then in-region, then agree).  Success
              additionally requires this pass AND the previous pass to be
              in-region (two CONSECUTIVE agreeing in-region observations
              of THIS task) with the in-region candidate a NEW ARRIVAL —
              outside PREEXISTING_ASSOC_M of every pre-drop in-region
              position — and the two agreeing observations to be two
              DISTINCT frames within RE_OBSERVE_WINDOW_S of each other
              (first_ts is READ; a re-delivered packet or a pair
              straddling the window re-anchors instead of certifying).  A
              same-label object already in the container can never certify
              a drop: Detection.ref is always None, so an unchanged
              in-region object is consistent with pre-existence and reads
              in_progress-and-diagnose;
          (h) (g) satisfied with |offset| <= CONTAINER_FIT_M on all axes ->
              success with the measured offsets, the stationarity state
              actually held, and the settling caveat in the evidence;
          (i) any axis beyond the half-extent -> in_progress (the drop may
              still be settling or may have missed the container;
              're-observe' with the measured offsets).
        No branch below ever reads as success unless containment is
        demonstrated across two observations; no geometry is ever
        fabricated.
        """
        if into_label is None or placed_object is None:
            return Verdict.in_progress(
                "re-observe; the place reply named the container (%r) but "
                "carried no into_label/placed_object annotations — cannot "
                "check containment yet" % into_ref)
        if not _detections_alive(obs, now):
            # three distinct causes collapse here: no observation at all, the
            # node never spoke, or it went silent past the window
            return Verdict.in_progress(
                "re-observe; detections node silent (%s) — no projection "
                "available to check containment"
                % _detections_dead_cause(obs, now))
        det_ts = getattr(obs, "detections_ts", 0.0) or 0.0
        if det_ts <= 0.0 or (now - det_ts) > POST_DROP_FRESHNESS_S:
            return Verdict.in_progress(
                "re-observe; the detections packet is not post-drop fresh "
                "(%.1f s old; POST_DROP_FRESHNESS_S=%.1f s) — a packet that "
                "old can predate the drop's completion; cannot check "
                "containment" % (now - det_ts, POST_DROP_FRESHNESS_S))
        containers = [d for d in obs.detections
                      if str(d.label).lower() == into_label.lower()]
        # the instance id survives a relabel
        _iid = _parse_into_iid(tool_result)
        _fp = _parse_into_fp(tool_result)
        if _iid is not None:
            _by_id = [d for d in obs.detections
                      if getattr(d, "instance_id", None) == _iid]
            if _by_id:
                containers = _by_id
        if not containers:
            return Verdict.in_progress(
                "re-observe; container %r (%r) is not in the current scene "
                "— containment is not demonstrated" % (into_ref, into_label))
        # the scene graph resolved into: to one ref, so take the nearest
        # same-label detection; the offsets still come from the current one
        container = None
        ref_note = ""
        ref_pos = _parse_into_ref_position(tool_result)
        if ref_pos is not None:
            best, best_d2 = None, None
            for d in containers:
                d2 = ((d.x - ref_pos[0]) ** 2
                      + (d.y - ref_pos[1]) ** 2
                      + (d.z - ref_pos[2]) ** 2)
                if d2 <= PREEXISTING_ASSOC_M ** 2 \
                        and (best_d2 is None or d2 < best_d2):
                    best, best_d2 = d, d2
            if best is not None:
                container = best
                ref_note = " (container resolved ref-unique via scene graph)"
        if container is None:
            if len(containers) > 1:
                return Verdict.in_progress(
                    "re-observe; %d container candidates labelled %r — cannot "
                    "tell which received the drop" % (len(containers), into_label))
            container = containers[0]
        objects = [d for d in obs.detections
                   if d is not container
                   and str(d.label).lower() == placed_object.lower()]
        if not objects:
            if _fp is not None:   # a deep container hides its inside
                return Verdict.in_progress(
                    "re-observe; the placed object (%r) is not observed after "
                    "the drop (occluded or out of frame) — containment is not "
                    "demonstrated (the inside of a deep container is mostly "
                    "out of the camera's view)" % placed_object)
            return Verdict.in_progress(
                "re-observe; the placed object (%r) is not observed after "
                "the drop (occluded or out of frame) — containment is not "
                "demonstrated" % placed_object)
        if len(objects) > 1:
            return Verdict.in_progress(
                "re-observe; %d candidates labelled %r after the drop — "
                "cannot tell which was placed" % (len(objects), placed_object))
        obj = objects[0]
        dx = abs(obj.x - container.x)
        dy = abs(obj.y - container.y)
        dz = abs(obj.z - container.z)
        in_region = dx <= CONTAINER_FIT_M and dy <= CONTAINER_FIT_M \
            and dz <= CONTAINER_FIT_M
        if _fp is not None:   # the footprint the place aimed into
            _xc, _ny, _fy, _hx, _rz, _bz = _fp
            in_region = (abs(obj.x - _xc) <= _hx - DEEP_FIT_MARGIN_M
                         and _ny + DEEP_FIT_MARGIN_M <= obj.y
                         <= _fy - DEEP_FIT_MARGIN_M
                         and _bz - DEEP_BOT_BELOW_M <= obj.z
                         <= _rz + DEEP_RIM_ABOVE_M)
            ref_note += (" (deep container footprint x %.3f+-%.3f, y "
                         "%.3f..%.3f, z %.3f..%.3f)"
                         % (_xc, _hx - DEEP_FIT_MARGIN_M,
                            _ny + DEEP_FIT_MARGIN_M, _fy - DEEP_FIT_MARGIN_M,
                            _bz - DEEP_BOT_BELOW_M, _rz + DEEP_RIM_ABOVE_M))
        # stationarity and pre-existing record: mirror of the navigate epoch
        # guard, one per (container, object) key and scoped to the task
        # execution
        epoch = _parse_dropped_at(tool_result)
        if epoch is None:
            return Verdict.in_progress(
                "re-observe; the place reply carries no dropped_at drop-"
                "epoch annotation — the stationarity record cannot be "
                "scoped to this task execution (a record without its epoch "
                "cannot be safely re-anchored); cannot check containment")
        key = (into_ref, into_label, placed_object)
        rec = self._containment_record.get(key)
        if rec is None or rec["epoch"] != epoch:
            # a new task re-anchors on its epoch and its pre-drop snapshot,
            # never on this post-drop frame
            pre_state, pre_positions = _parse_predrop(tool_result)
            self._containment_record[key] = rec = {
                "epoch": epoch,
                "preexisting_state": pre_state,   # 'known' | 'unknown'
                "preexisting": pre_positions,     # pre-drop in-region positions
                "prev_in_region": False,          # no in-region sighting yet
                "first_ts": None,                 # first in-region det_ts
                "has_outside": False,             # outside sighting = displacement
            }
        if not in_region:
            # outside the region: record the displacement evidence; the
            # in-region chain breaks anyway
            rec["has_outside"] = True
            rec["prev_in_region"] = False
            return Verdict.in_progress(
                "re-observe; the placed object (%r) projects OUTSIDE "
                "container %r region (±%.2f m around (%.3f, %.3f, %.3f); "
                "measured offsets dx=%.3f dy=%.3f dz=%.3f m)%s — the drop "
                "may still be settling or may have missed the container; "
                "cannot confirm containment"
                % (placed_object, into_label, CONTAINER_FIT_M,
                   container.x, container.y, container.z, dx, dy, dz,
                   ref_note))
        if rec["prev_in_region"] is not True:
            # first in-region frame: note it and stay in_progress, since one
            # frame proves nothing
            rec["first_ts"] = det_ts
            rec["prev_in_region"] = True
            return Verdict.in_progress(
                "re-observe; first post-drop observation of this drop "
                "(dropped_at=%.6f) recorded: %r detected INSIDE container "
                "%r region (offsets dx=%.3f dy=%.3f dz=%.3f m, within "
                "CONTAINER_FIT_M=%.2f m)%s — one observation never "
                "certifies: at this point the object is indistinguishable "
                "from one that was already in the container (no refs on "
                "detections); re-observe on the next turn — a SECOND "
                "consecutive in-region observation within "
                "RE_OBSERVE_WINDOW_S=%.1f s showing the object as a NEW "
                "arrival (a position change or an object that was not "
                "there before) is required before success"
                % (epoch, placed_object, into_label, dx, dy, dz,
                   CONTAINER_FIT_M, ref_note, RE_OBSERVE_WINDOW_S))
        # the pair needs a distinct newer frame within the window, else
        # re-anchor
        first_ts = rec["first_ts"]
        if first_ts is None or det_ts <= first_ts \
                or (det_ts - first_ts) > RE_OBSERVE_WINDOW_S:
            rec["first_ts"] = det_ts  # re-anchor the pair here instead
            return Verdict.in_progress(
                "re-observe; %r detected INSIDE container %r region "
                "(offsets dx=%.3f dy=%.3f dz=%.3f m, within "
                "CONTAINER_FIT_M=%.2f m)%s — the previous in-region "
                "observation does not pair with this one (not a distinct "
                "newer frame within RE_OBSERVE_WINDOW_S=%.1f s of the "
                "first); the stationarity pair re-anchors on this frame; "
                "re-observe on the next turn"
                % (placed_object, into_label, dx, dy, dz, CONTAINER_FIT_M,
                   ref_note, RE_OBSERVE_WINDOW_S))
        # the object must be a new arrival against the pre-drop set, else an
        # unchanged object would certify
        preexisting = rec["preexisting"]
        if rec["preexisting_state"] == "known" and preexisting:
            min_d2 = min((obj.x - px) ** 2 + (obj.y - py) ** 2
                         + (obj.z - pz) ** 2
                         for (px, py, pz) in preexisting)
            if min_d2 <= PREEXISTING_ASSOC_M ** 2:
                return Verdict.in_progress(
                    "re-observe; the in-region object matches a same-label "
                    "object that was ALREADY inside container %r region in "
                    "the PRE-DROP observation (within PREEXISTING_ASSOC_M="
                    "%.2f m of a pre-existing position)%s — the only "
                    "in-region evidence is consistent with an object that "
                    "was already there, so the drop cannot be certified "
                    "(no refs on detections; a re-place or a position "
                    "change is required before success)"
                    % (into_label, PREEXISTING_ASSOC_M, ref_note))
            moved = (" the in-region object's position is OUTSIDE "
                     "PREEXISTING_ASSOC_M of every pre-drop in-region "
                     "position (displacement %.3f m) — a new arrival "
                     "relative to the pre-drop contents"
                     % math.sqrt(min_d2))
        elif rec["preexisting_state"] == "known":
            moved = (" the PRE-DROP observation showed NO same-label "
                     "object inside the container region — a new arrival")
        elif rec["has_outside"]:
            moved = (" no pre-drop observation was available, but the "
                     "object was observed OUTSIDE the container region at "
                     "least once during this task before settling "
                     "in-region — displacement evidence names it a new "
                     "arrival")
        else:
            return Verdict.in_progress(
                "re-observe; no PRE-DROP observation was available to name "
                "the container's pre-existing contents (predrop_state=unknown) "
                "and the placed object (%r) was never observed OUTSIDE "
                "container %r region during this task — the in-region "
                "evidence is consistent with an object that was already "
                "there, so the drop cannot be certified; re-observe (a "
                "sighting outside the region, then two agreeing in-region "
                "observations, is required before success)"
                % (placed_object, into_label))
        return Verdict.success(
            "drop DONE, %s; hold evidence: "
            "%s — containment: %r projected INSIDE container %r "
            "region on TWO CONSECUTIVE fresh post-drop observations of "
            "this drop (dropped_at=%.6f; offsets dx=%.3f dy=%.3f dz=%.3f "
            "m, within CONTAINER_FIT_M=%.2f m);%s%s — the object was "
            "observed in-region across two frames within "
            "RE_OBSERVE_WINDOW_S, so a still-settling or in-flight object "
            "cannot read success (the drop is confirmed as landed in the "
            "container region, not merely passed through it)"
            % (open_evidence, hold_evidence, placed_object, into_label, epoch,
               dx, dy, dz, CONTAINER_FIT_M, ref_note, moved))

    # -- navigate: did it actually get there? --------------------------------

    def verify_navigate(self, contract, obs, tool_result: str,
                        now: float | None = None) -> Verdict:
        """Post-condition: robot at target — walk_status == REACHED ONLY,
        confirmed within the DISPATCH EPOCH (the NAVIGATE EPOCH GATE).

        NEVER odom distance: /unitree/slam_mapping/odom does NOT exist on
        Stuart. The gateway stamps ``nav_epoch`` on each /walk_go publish and
        annotates the tool result with NAV_EPOCH, NAV_LABEL (the requested
        target), LATEST_WALK_TS (latest walk_status message time) and
        PHASES_SEEN (phases observed with ts >= epoch). Success requires ALL
        of:
          (a) nav_epoch parsed from the tool result;
          (b) latest walk_status ts >= nav_epoch (the FSM reacted to THIS
              dispatch — not a pre-dispatch snapshot);
          (c) walk_status phase == REACHED;
          (d) at least one NON-REACHED phase (SEARCHING/APPROACHING) observed
              in this epoch — the transition guard that defeats a stale or
              hold REACHED from a previous arrival (the FSM re-publishes
              REACHED at 1 Hz for REACHED_HOLD_S = 5 s after arrival);
          (e) this epoch was not already confirmed (no re-success on one
              epoch);
          (f) the walk_status payload label matches NAV_LABEL (fix 1): a
              REACHED for a DIFFERENT label is a stale arrival of an earlier
              target — mismatch -> in_progress 're-observe'.  Case-insensitive
              (the FSM lowercases /walk_target labels).
        Any miss -> in_progress with evidence 're-observe'. SEARCHING/
        APPROACHING = still moving; IDLE is ambiguous (arrived-and-finished
        vs never-started) -> in_progress, re-observe.

        THE ONE TERMINAL NON-SUCCESS (C5, 2026-10-01): when the operator
        cancelled the walk while it was running, this returns
        ``failure(reason='cancelled_by_operator')`` — checked BEFORE every
        epoch gate above, because a cancel clears the dispatch epoch.  It is
        deliberately NOT in_progress: re-observe is an instruction to try
        again, and the operator asked for the opposite.
        """
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        # terminal, checked before the epoch gates: a cancel clears the epoch,
        # so every epoch-gated outcome would read in_progress
        # failure with a reason is the honest status; the budget only acts on
        # success and failure, so a fourth status would sit inert
        if _walk_was_cancelled(tool_result):
            return Verdict.failure(
                "the operator cancelled this navigate while it was running — "
                "the walk FSM was stopped and the target was NOT reached; "
                "this was an intentional stop, not a failure to arrive. "
                "Acknowledge it; it ended ONLY this navigate. Do not repeat "
                "it on your own, but walk again (no permission needed) once the "
                "operator gives a new instruction",
                "cancelled_by_operator")
        nav_epoch = _parse_nav_epoch(tool_result)
        if nav_epoch is None:
            return Verdict.in_progress(
                "re-observe; navigate tool result carries no NAV_EPOCH — "
                "cannot gate the verdict on the dispatch epoch")
        # already at the target: REACHED standby, a fresh packet with no
        # transition, so the live phase and label decide
        if "WALK_WAIT=already_reached" in (tool_result or ""):
            walk = getattr(obs, "walk", None) if obs is not None else None
            phase = ((getattr(walk, "phase", None) or "") if walk else "").upper()
            nav_label = _parse_nav_label(tool_result)
            walk_label = (getattr(walk, "label", None) or "") if walk else ""
            if phase == "REACHED" and (nav_label is None or
                                       walk_label.strip().lower()
                                       == nav_label.strip().lower()):
                self._nav_last_success_epoch = nav_epoch
                return Verdict.success(
                    "already at the target: walk_status == REACHED standby on "
                    "%r (no new walk was needed) — do NOT navigate again; "
                    "go on with the next step (place / pick)" % (nav_label,))
            return Verdict.in_progress(
                "re-observe; gateway reported already_reached but the live "
                "walk_status is %r on %r" % (phase, walk_label))
        if nav_epoch > 0.0 and nav_epoch == self._nav_last_success_epoch:
            return Verdict.in_progress(
                "re-observe; dispatch epoch %.6f already confirmed — "
                "refusing re-success on the same epoch" % nav_epoch)
        latest_ts = _parse_latest_walk_ts(tool_result)
        if latest_ts is None or latest_ts < nav_epoch:
            return Verdict.in_progress(
                "re-observe; no walk_status observed since dispatch epoch "
                "%.6f (latest walk ts %.6f)"
                % (nav_epoch, latest_ts if latest_ts is not None else 0.0))
        phases_seen = _parse_phases_seen(tool_result)
        if not phases_seen:
            return Verdict.in_progress(
                "re-observe; no walk phases observed in dispatch epoch %.6f"
                % nav_epoch)
        walk = getattr(obs, "walk", None) if obs is not None else None
        if walk is None:
            return Verdict.in_progress("re-observe; no walk_status")
        phase = (walk.phase or "").upper()
        if phase != "REACHED":
            if phase in ("SEARCHING", "APPROACHING"):
                return Verdict.in_progress(
                    "re-observe; walk FSM still %s (epoch %.6f)"
                    % (phase, nav_epoch))
            if phase == "IDLE":
                return Verdict.in_progress(
                    "re-observe; walk FSM IDLE (REACHED not seen — ambiguous, "
                    "epoch %.6f)" % nav_epoch)
            return Verdict.in_progress(
                "re-observe; unknown walk phase %r (epoch %.6f)"
                % (phase, nav_epoch))
        # REACHED, which means a SEARCHING/APPROACHING phase came first this epoch
        if not any(p in ("SEARCHING", "APPROACHING") for p in phases_seen):
            return Verdict.in_progress(
                "re-observe; only REACHED observed in dispatch epoch %.6f — "
                "stale/hold REACHED from a previous arrival (no transition "
                "evidence; phases seen: %s)"
                % (nav_epoch, ",".join(phases_seen)))
        # the label must match the requested target; an absent NAV_LABEL skips
        # the check
        nav_label = _parse_nav_label(tool_result)
        if nav_label is not None:
            walk_label = getattr(walk, "label", None)
            if (walk_label or "").strip().lower() != \
                    (nav_label or "").strip().lower():
                return Verdict.in_progress(
                    "re-observe; walk_status payload label %r does not match "
                    "the requested target %r (epoch %.6f)"
                    % (walk_label, nav_label, nav_epoch))
        self._nav_last_success_epoch = nav_epoch
        return Verdict.success(
            "walk_status == REACHED within dispatch epoch %.6f (transition "
            "observed: %s)" % (nav_epoch, ",".join(phases_seen)))

    # -- open_hand: is the hand measured open? -------------------------------

    def verify_open_hand(self, contract, obs, tool_result: str,
                         now: float | None = None) -> Verdict:
        """Post-condition: grip open — the MEASURED hand state, never
        command-inference.

        MEASURED OPEN (world-truth round 3 fix 3 + round 4/5): the open
        itself clears the close verdicts, so post contact==False is
        GUARANTEED and proves nothing.  Success requires the post state read
        FRESH (within GRIP_STATE_MAX_AGE_S), the hand MEASURED-OPEN (joints
        at the CAUSAL measured-open reference — per side it forms ONLY from
        reads taken after a gateway-dispatched open command for that side,
        via a double-read agreement; is_open True) and NO literal contact
        verdict on file (grip.contact True = a close_adaptive DONE:contact
        still folded — the hand was holding at that close).  A still-folded
        literal contact is a failure; a stale read, an offline grip, or NO
        reference yet (no open command dispatched for the side, or the
        double-read hasn't completed — the verifying read is the FIRST of
        the two, so one re-observe confirms) is in_progress 're-observe'.
        A measured-open read on an open-commanded hand is a REAL
        confirmation path.  The pre-open prev_contact annotations are
        INFORMATIONAL ONLY.

        BOTH HANDS (fix 5): an open_both (sides=both) verifies the LEFT hand
        independently (left_available True + left state read fresh +
        left_contact not True + left measured-open); an unobserved, offline
        or unmeasurable left state is in_progress, never success — world
        truth runs on the left side as well.
        """
        now = time.time() if now is None else now
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        if kind in ("contact", "full"):
            return Verdict.failure(
                f"hand not open: close_adaptive reported {kind}", f"close_adaptive {kind}")
        sides = _parse_sides(tool_result) or "right"
        grip = _grip(obs)
        if grip is None:
            available = _state_available(detail)
            if available is False:
                return Verdict.in_progress(
                    "re-observe; grip offline (state reply available=False)")
            return Verdict.in_progress(
                "re-observe; grip state missing (state reply carries no contact)")
        if not _ts_fresh(getattr(grip, "stale_ts", None), now):
            return Verdict.in_progress(
                "re-observe; grip state read stale (older than "
                "GRIP_STATE_MAX_AGE_S) — cannot confirm the open")
        if not grip.available:
            return Verdict.in_progress(
                "re-observe; grip offline (state [side] available=False)")
        if grip.contact is True:
            return Verdict.failure(
                "hand still holding (literal close_adaptive contact verdict "
                "on file after open)", "still holding after open")
        if grip.is_open is not True:
            return Verdict.in_progress(
                "re-observe; hand not measured-open (is_open=%s — no "
                "trusted measured-open reference yet or joints away from "
                "it); cannot confirm the open" % grip.is_open)
        if sides == "both":
            left_available = getattr(grip, "left_available", None)
            left_stale_ts = getattr(grip, "left_stale_ts", None)
            left_is_open = getattr(grip, "left_is_open", None)
            left_contact = getattr(grip, "left_contact", None)
            if left_available is not True:
                return Verdict.in_progress(
                    "re-observe; left grip offline (left_available=%s) — "
                    "open_both cannot be confirmed" % left_available)
            if not _ts_fresh(left_stale_ts, now):
                return Verdict.in_progress(
                    "re-observe; left grip state read stale — open_both "
                    "cannot be confirmed")
            if left_contact is True:
                return Verdict.failure(
                    "left hand still holding after open_both (literal "
                    "close_adaptive contact verdict on file for the left "
                    "hand)", "left still holding after open")
            if left_is_open is not True:
                return Verdict.in_progress(
                    "re-observe; left hand not measured-open (left_is_open=%s) "
                    "— open_both cannot be confirmed" % left_is_open)
            return Verdict.success(
                "grip measured-open, no literal contact verdict on either hand")
        return Verdict.success(
            "grip measured-open, no literal contact verdict on file")

    # -- close_hand: did it really close on something? -----------------------

    def verify_close_hand(self, contract, obs, tool_result: str,
                          now: float | None = None) -> Verdict:
        """Post-condition: grip closed — literal contact OR full-stroke stop.

        close_adaptive replies are decisive: DONE:contact = the real grasp
        signal; DONE:full = full-stroke stop (post-condition met); DONE:no_data
        = no tactile evidence -> in_progress.  Plain DONE relies on grip
        state: the FOLDED contact (the gateway's OWN serialized
        close_adaptive verdict for this close).

        CLOSE FRESHNESS (world-truth round 4, fix 2): ALL success paths
        require a FRESH grip state read (GRIP_STATE_MAX_AGE_S), and the
        folded-contact path additionally requires the folded verdict's own
        timestamp (GripState.contact_ts) within CLOSE_CONTACT_WINDOW_S — a
        daemon'd grip's stale folded contact must NOT confirm.

        BOTH HANDS (world-truth round 5): the gateway's close_both dispatch
        is now REAL per-side close_adaptive (RIGHT then LEFT — per-side
        tactile evidence), and the tool result lists the per-side verdicts
        (``right_close=contact|full|no_data; left_close=...``).  Success
        requires BOTH sides' verdicts contact|full on FRESH, available
        reads — the both-hands check runs BEFORE any right-only success.  A
        no_data / missing / offline / stale side -> in_progress
        (re-observe), never success.  Replies without per-side fields fall
        back to the folded-contact path (both hands' literal contact, all
        fresh).
        """
        now = time.time() if now is None else now
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        if kind == "contact":
            if self._close_fresh(obs, now):
                return Verdict.success(
                    "close_adaptive DONE:contact on a fresh grip read — grip "
                    "closed on object")
            return Verdict.in_progress(
                "re-observe; close_adaptive DONE:contact but the grip state "
                "read is missing/stale/offline — cannot confirm on a dead "
                "lane")
        if kind == "full":
            if self._close_fresh(obs, now):
                return Verdict.success(
                    "close_adaptive DONE:full on a fresh grip read — closed "
                    "to full-stroke stop")
            return Verdict.in_progress(
                "re-observe; close_adaptive DONE:full but the grip state "
                "read is missing/stale/offline — cannot confirm on a dead "
                "lane")
        if kind == "no_data":
            return Verdict.in_progress(
                "re-observe; close_adaptive returned no tactile data")
        if _parse_sides(tool_result) == "both":
            return self._verify_close_both(obs, now, tool_result)
        grip = _grip(obs)
        if grip is None:
            available = _state_available(detail)
            if available is False:
                return Verdict.in_progress(
                    "re-observe; grip offline (state reply available=False)")
            return Verdict.in_progress(
                "re-observe; grip state missing (state reply carries no contact)")
        if not grip.available:
            return Verdict.in_progress(
                "re-observe; grip offline (state [side] available=False)")
        # a fresh read and a folded verdict timestamp inside the window
        if grip.contact and _verdict_fresh(getattr(grip, "contact_ts", None),
                                           now) \
                and _ts_fresh(getattr(grip, "stale_ts", None), now):
            return Verdict.success(
                "grip state literal contact True on a fresh read — grip "
                "closed on object")
        return Verdict.in_progress(
            "re-observe; grip not confirmed closed (no fresh literal contact "
            "evidence: folded verdict missing/stale or the state read is "
            "not fresh)")

    # -- pose: did the posture get commanded? --------------------------------

    @_zfloor_limited_note("requested pose")
    def verify_pose(self, contract, obs, tool_result: str,
                    now: float | None = None) -> Verdict:
        """Post-condition: the named posture was COMMANDED and the engine
        reported the move complete.

        READ THIS BEFORE WEAKENING OR STRENGTHENING THE SUCCESS PATH.

        There is no posture sensor in this system.  WorldObservation carries
        grip, walk, detections and a scene graph, and NOTHING that measures
        where the arm joints are.  So a pose cannot be MEASURED the way 'grip
        open' (a fresh joint/hand-state read) or 'walk REACHED' (the walk
        FSM) can.  The only evidence that exists is the pick engine's own
        completion report, which reaches us as 'DONE:<v1>[+<v2>]'.

        That is a WEAKER class of evidence, and this verifier says so in the
        evidence string instead of dressing it up as a measurement.  It
        accepts the engine's DONE because the engine is the arm's AUTHORITY —
        it drives the arm_sdk writer and blocks on the motion — which is the
        same footing on which verify_navigate accepts the walk FSM's REACHED
        as an independent subsystem report.  It is NOT the same footing as
        verify_open_hand, which reads the world back.

        What it refuses:
          - any reply that is not a DONE: error/query/state/unknown/empty all
            fall through the shared prologue;
          - a DONE with NO verb list (the mock form) — it names no posture, so
            it certifies nothing;
          - a DONE whose verb list is not ONE WHOLE POSE.  This is the check
            that matters: 'tuck' is TWO sequential engine moves (carry, then
            carry_left) because the engine's carry pose is per-arm.  A reply
            naming only 'carry' means the sequence died with the right arm
            parked and the left still where it was — a real, half-finished
            posture that must never read as done.  See
            tests/test_pose_tool.py scenario 4 for the matching backend pin.
        """
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        reported = tuple(v for v in (detail or "").split("+") if v.strip())
        if not reported:
            return Verdict.in_progress(
                "re-observe; pose reply carries no engine verb list — it names "
                "no posture, so it certifies none")
        # lazy import, because a module-scope import closes a cycle
        # the backend's table is the single source of truth: a drifted copy
        # would certify the wrong pose
        try:
            from gateway.brain_client import POSE_ENGINE_VERBS
        except Exception as exc:  # pragma: no cover - import environment only
            return Verdict.in_progress(
                "re-observe; cannot reach the pose verb table (%s) — refusing "
                "to certify a posture against a vocabulary I cannot check"
                % exc.__class__.__name__)
        named = [name for name, verbs in POSE_ENGINE_VERBS.items()
                 if tuple(verbs) == reported]
        if not named:
            return Verdict.in_progress(
                "re-observe; engine reported %r, which is not a whole pose "
                "(known sequences: %s) — a partial sequence is a half-finished "
                "posture, not a reached one"
                % ("+".join(reported),
                   ", ".join("+".join(v) for v in
                             sorted(POSE_ENGINE_VERBS.values()))))
        return Verdict.success(
            "engine reported DONE:%s == the %r posture, in full (engine-"
            "reported, NOT sensor-measured — no posture sensor exists)"
            % ("+".join(reported), named[0]))

    def verify_turn_around(self, contract, obs, tool_result: str,
                           now: float | None = None) -> Verdict:
        """Post-condition: the sweep ran and the person was CENTRED.

        BATCH-2 UPDATE (TURN interface): the gateway verb now blocks, sweeps
        for the label, stops, then rotates in place (omega only) until the
        detection bearing is ~0, and replies 'DONE turn_around: ... ;
        TURN_RESULT=centred' ONLY for a verified centring (every other
        outcome leads 'ERROR:turn_around').  Success therefore requires the
        DONE prefix AND the TURN_RESULT=centred token.  The historical notes
        below describe the pre-batch-2 fire-and-forget sweep.

        READ THIS BEFORE STRENGTHENING THE SUCCESS PATH.

        turn_around is NOT navigate, and it must never borrow navigate's
        verdict.  verify_navigate requires walk_status == REACHED inside the
        dispatch epoch; a turn-around sweep NEVER produces REACHED, by
        design — the FSM's _phase_hold stops the sweep on first sight and
        deliberately never approaches (skills/locomotion/walk_node.py
        _phase_hold, the DETECTED hold).  Gating on REACHED would answer
        in_progress forever and burn the model's budget on a task that had
        already succeeded.

        There is also NO epoch to gate on, unlike navigate.  The gateway's
        reply carries no NAV_EPOCH / PHASES_SEEN annotation (gateway_node's
        _do_turn_around returns the DONE line alone), and obs.walk carries
        only the LATEST phase — so a 'SEARCHING' seen here cannot be
        distinguished from a SEARCHING that predates this dispatch.
        Requiring a fresh phase WITHOUT an epoch is exactly the stale-arrival
        false success that navigate's (b)/(d) gates exist to defeat, and
        false-success is the worse error, so this verifier does not attempt
        it.  Adding an epoch to the turn_around reply is the prerequisite for
        ever tightening this.

        What it therefore certifies, and nothing more: the gateway accepted
        the verb and published the label on /walk_target.  Whether anyone is
        in view is NOT verified here and is never claimed — the FSM reports
        that later on /walk_status as DETECTED, which no tool result carries.

        What it refuses:
          - any non-DONE reply (the shared prologue: ERROR -> failure);
          - a bare 'DONE' that does not name the sweep — it certifies that
            SOME command succeeded, not that a sweep was published.
        """
        kind, detail = _classify_reply(tool_result)
        verdict = _guard_reply(kind, detail)
        if verdict is not None:
            return verdict
        first = ""
        if tool_result is not None and tool_result.strip():
            first = tool_result.strip().splitlines()[0].strip()
        # lazy import again, same cycle; the constant is the single source of
        # truth
        try:
            from gateway.brain_client import TURN_AROUND_DONE_PREFIX
        except Exception as exc:  # pragma: no cover - import environment only
            return Verdict.in_progress(
                "re-observe; cannot reach the turn_around reply vocabulary "
                "(%s) — refusing to certify a sweep against a shape I cannot "
                "check" % exc.__class__.__name__)
        # centring needs the DONE prefix and the TURN_RESULT=centred token; this
        # check runs first, and the prefix gate below stays the guard for a reply
        # that names no sweep
        if (first.startswith(TURN_AROUND_DONE_PREFIX)
                and _TURN_CENTRED_RE.search(first) is None):
            return Verdict.in_progress(
                "re-observe; turn_around reply does not carry "
                "TURN_RESULT=centred (%r) — the person was not verified "
                "centred in the frame" % first[:160])
        if not first.startswith(TURN_AROUND_DONE_PREFIX):
            return Verdict.in_progress(
                "re-observe; turn_around reply %r does not name the sweep "
                "(expected one starting %r) — a reply that does not name the "
                "sweep certifies no sweep was published"
                % (first[:120], TURN_AROUND_DONE_PREFIX))
        return Verdict.success(
            "gateway reported %r — the robot turned in place and the gateway "
            "rotated it until the target sat at the centre of the camera "
            "frame, then stopped (no walking; the bearing is gateway-measured "
            "from /object_detections_body; gateway-reported, NOT "
            "sensor-measured by any body-pose sensor; /walk_status was not "
            "consulted)"
            % first)

    def _close_fresh(self, obs, now: float) -> bool:
        """The grip state read behind a close is FRESH and available.

        All close_hand success paths require a fresh grip state read
        (GRIP_STATE_MAX_AGE_S) — a dead/daemon'd grip lane can never
        confirm (world-truth round 4, fix 2).
        """
        grip = _grip(obs)
        if grip is None:
            return False
        if not grip.available:
            return False
        return _ts_fresh(getattr(grip, "stale_ts", None), now)

