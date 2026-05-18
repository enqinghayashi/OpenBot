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
