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

