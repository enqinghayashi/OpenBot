"""assemble.py — pure-stdlib assembly of raw skill payloads into WorldObservation.

Turns the raw payload dicts the skill lane produces (detections rows on
``/object_detections_body``, the grip server's ``state`` reply line, the
walk node's ``/walk_status`` JSON) into the shared ``brain.observation``
dataclasses. No IO, no third-party imports, laptop-testable.

Freshness model (event-driven perception):
* every detections packet is timestamped on arrival; a packet (even an empty
  list) marks the scene as freshly observed;
* every ``/walk_status`` packet is timestamped on arrival too, and the walk
  frame is DROPPED (``walk=None``) once it is older than
  ``walk_max_stale_s`` — a latched phase must never be reported as current;
* silence == empty scene (WHEN ``silence_means_empty`` is True — the shipped
  profile, see below): ``detections`` stays at the last known rows (or
  ``[]``), but ``fresh`` flips to False once ``max_stale_s`` has passed
  without any packet;
* a scene never observed at all is ``fresh=False`` with ``detections=[]``.

SILENCE SWITCH (world-truth round 5): ``silence_means_empty`` (constructor
param, default True — the shipped profile MUST have True, from
``embodiment.perception.silence_means_empty``) is the switch behind the
silence semantics:
  True  — a live channel's empty last packet IS an empty scene (target-gone
          is confirmable), and a channel that is silent forever (never
          spoke, or beyond the liveness window) is a DEAD node, not an
          empty scene — :meth:`Assembler.detections_alive` is exactly that
          liveness signal (generous window, distinct from packet
          freshness);
  False — silence/emptiness is NOT empty-scene evidence: consumers (the
          verifier's target-gone path, wired from the same config) must
          treat a quiet scene as unconfirmable, never as a confirmed empty
          scene.

DETECTIONS FRAME (world-truth round 5, ``embodiment.perception.frame``):
detections arrive in the frame named by the config ('torso' on Stuart):
x = lateral RIGHT, y = forward, z = up (metres).  The same right-handed
frame governs the writer's ``cmd_set <vx> <vy> <omega>`` sign convention:
positive omega is the left-hand rule about +z (counter-clockwise seen from
above, yaw toward +x); a negative target x requires a positive omega, etc.
The gateway serves this mapping from the config value (see
gateway/gateway_node.py ``resolve_embodiment_config``); this module
documents it.

Grip semantics: the grip server does NOT auto-verify. ``contact`` is ONLY
the literal ``close_adaptive`` verdict (DONE:contact): the caller records the
verdict with ``Assembler.note_close_result`` and the next ``GripState`` folds
it in for ``contact_window_s``, exposing the verdict's own timestamp as
``contact_ts`` so stale folds can be aged by their own window.  CONTACT LAW
(world-truth round 4): contact is NEVER derived from joints — the deployed
grip_server ``_state_payload`` emits, per side, the last COMMANDED q as
top-level per-joint keys (``last_q``), the MEASURED motor q under the
``actual`` mapping (OMITTED — never faked from ``last_q`` — when the dex3
lane has no state; skills/arm/grip_server.py:1036-1048) + the per-side
``available`` flag (the dex3 hand-state/DDS lane-health probe), and there is
NO literal hand-state contact field in the state reply.  A hand closed at boot can match
a wrong first-read reference, so a ``not is_open`` derivation would claim
contact for an EMPTY closed hand — that derivation is REMOVED.

MEASURED SPACE (round 7 — the confirmed HIGH): ``GripState.joints`` is the
``actual`` vector (joint-id order 0..6; EMPTY when the reply carries no
usable ``actual``) and ``GripState.commanded`` the top-level ``last_q``
vector.  ``is_open``, ``_agree``, ``_update_open_ref`` and the observation
fold all operate on ``joints`` — measured space, exactly what their
docstrings below say.  Judging ``last_q`` instead 'would pass
unconditionally' (the grip server's own words): a latched motor (mode 0)
stays closed while ``last_q`` advances to the open preset, so the commanded
vector would verify a release that never happened.  With ``actual`` absent
the hand is UNMEASURABLE: no reference can form from it, and ``is_open`` is
None (no reference) or False — never True.

World-truth round 4 (fixes 1-2):

GRIP STATE AGING: every successful ``state`` reply stamps a per-side
``last_state_ts``. A GripState whose read is older than ``GRIP_STATE_MAX_AGE_S``
(5.0 s), or whose read errored (``Assembler.note_grip_read_error`` or an
``ERROR:``/malformed state line), is reported with ``available=False`` and
``stale_ts=last_state_ts`` — a daemon'd grip keeps its last good
``available=True`` forever no more. The error is superseded by the next
successful read (the ts comparison ``error_ts > last_read_ts`` decides).

World-truth round 5 (fix 1) — OPEN REFERENCE IS CAUSAL (the round-4
boot-window path is DELETED):

Per side, ``open_joints_ref(side)`` is the measured shape the hand takes
when COMMANDED open — the reference forms ONLY from state reads taken AFTER
a gateway-dispatched open command for that side (``note_open_commanded``:
open / open_left / open_both, or a successful drop which opens both hands).
TWO CONSECUTIVE agreeing reads (within ``OPEN_JOINT_TOLERANCE``), both
taken in that post-open context, define the reference.  The round-4 BOOT
WINDOW is gone: boot reads carry NO semantic guarantee (the robot can boot
mid-grasp or power down mid-task with the hands closed — two agreeing boot
reads on a closed shape would make the closed shape the 'open' reference
and a closed hand would verify 'open' — the round-5 finding).  Without an
open command, NO read — however agreeing — ever forms a reference.
``note_hand_command`` (a pick/close for that side) ends the side's
open-trust: reads after it no longer qualify until the next open command.
The reference is UPDATED only from a NEW post-open qualifying agreement
that is within tolerance of the current reference (drift: servo settling /
thermal drift) or when the reference is None (formation); a far or
untrusted read never moves it.

``is_open(side, joints)`` = reference exists AND max joint delta <
``OPEN_JOINT_TOLERANCE``; None (unknown) while no reference exists — open
and place verification answer in_progress 're-observe'.  FIRST OPEN OF A
SESSION: the verifying ``state`` read after the open command is the FIRST of
the two required reads; the second agreeing read arrives on the next
re-observe (one re-observe confirms — the natural workflow
``open -> verify -> (re-observe) -> pick -> drop -> place`` makes the path
real).

``OPEN_JOINT_TOLERANCE`` is 0.35 rad, a measured-space tolerance, NOT a
preset guess — the open/closed presets differ by up to ~2.4 rad on joints
3-6, so 0.35 rad cleanly separates them while tolerating servo noise.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from typing import Dict, List, Optional

from brain.observation import (
    Detection,
    GripState,
    WalkStatus,
    WorldObservation,
    _opt_body_side,
    _opt_color_name,
    _opt_hue, _opt_instance_id,
)

#: stale after this.
MAX_STALE_S = 5.0
#: walk frame dropped past this, not aged.
#: NOT MAX_STALE_S: worst live gap is
#: 5.0 + 1.047 + 0.3 ~= 6.35 s
#: (skills/locomotion/walk_node.py), so 5.0 s
#: would report a live search as dead.
#: APPROACHING heartbeats 5.0 s, REACHED 1 Hz.
#: 15.0 = >2x the worst gap; see
#: docs/STOP-KILLS-TASK-DESIGN.md section 3 C2.
WALK_MAX_STALE_S = 15.0
#: liveness window for detections_alive:
#: ever spoke = alive, even when quiet.
DETECTIONS_ALIVE_S = 10.0
#: how long a close verdict counts as contact.
CLOSE_CONTACT_WINDOW_S = 15.0
#: max age of a grip state read.
#: older -> available=False, stale_ts = last good.
#: polled every verify/observe, so only a dead
#: grip ages out.
GRIP_STATE_MAX_AGE_S = 5.0
#: measured-open tolerance (rad). NOT a preset
#: guess: presets differ by ~2.4 rad on the
#: fingers, so 0.35 separates them and absorbs
#: servo drift.
OPEN_JOINT_TOLERANCE = 0.35
#: a reference must be an open shape. Dex3
#: closed preset: joints 5/6 at ~1.4-1.7 rad,
#: open at ~0.0-0.2; above this bound = closed.
#: MEASURED: place parks the hand CLOSED after
#: release, side already open-commanded, and two
#: agreeing closed reads (1.54/1.71) became the
#: "open" reference - every later closed read
#: said is_open=True, the open hand False.
OPEN_REF_MAX_FINGER_RAD = 1.0
#: non-phase /walk_status events. DETECTED
#: fires once per hold entry -> SEARCHING.
WALK_STATE_MAP = {"DETECTED": "SEARCHING"}


def parse_detections(rows: List[dict], now: float) -> List[Detection]:
    """Normalize raw detection rows into ``Detection`` objects.

    Rows are the torso-frame payload of ``/object_detections_body``:
    ``{label, conf, x, y, z, orientation, ...}`` with x = lateral right,
    y = forward, z = height above the torso link (all metres).
    Malformed rows (missing label or non-numeric x/y/z) are dropped —
    perception emits well-formed rows, so a malformed one must not poison
    the verifier with NaNs.

    ORIENTATION (fix 2026-09-07): the depth projector emits the grasp
    approach ``orientation`` ('upright' | 'horizontal') for every row.  It
    is CARRIED onto the Detection (previously dropped, so a lying bottle
    defaulted to the upright approach upstream).  Absent/unknown values
    default to 'upright' (the safe/legacy approach) — never a guess.
    Optional object extents ``width_m`` / ``height_m`` are carried through
    when present; a non-numeric extent is dropped to None (informational,
    not a position).

    Optional perception colour + body-side semantics are carried through
    too: ``hue`` (degrees 0-360), ``color_name`` (colour band) and
    ``body_side`` ('left' | 'right') are read off each row so the model's
    scene prose can distinguish e.g. the red cup on the left from the green
    cup on the right.  ``frame_side`` (display-only) is DROPPED.  A
    malformed value parses to None (informational, not a position).

    NaN/Inf rows are DROPPED (world-truth round 2, fix 8): a NaN z would
    silently bypass the tabletop floor refusal (NaN < -0.35 is False) and a
    non-finite position is not a world state.  Dropping cannot fabricate —
    the row simply does not exist in the scene.  A non-finite ``conf``
    coerces to 0.0 (confidence, not a position — no phantom-object risk).
    """
    out: List[Detection] = []
    for row in rows:
        try:
            label = str(row["label"])
            x = float(row["x"])
            y = float(row["y"])
            z = float(row["z"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            continue  # non-finite row: can't invent a position
        try:
            conf = float(row.get("conf", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        if not math.isfinite(conf):
            conf = 0.0
        # absent/unknown -> safe upright approach.
        raw_ori = row.get("orientation")
        if isinstance(raw_ori, str) and raw_ori.strip().lower() in (
            "upright", "horizontal"
        ):
            orientation = raw_ori.strip().lower()
        else:
            orientation = "upright"
        width_m = height_m = None
        for extent_key in ("width_m", "height_m"):
            val = row.get(extent_key)
            if val is None or isinstance(val, bool):
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                if extent_key == "width_m":
                    width_m = val
                else:
                    height_m = val
        # colour + body side; frame_side is display-only.
        # malformed -> None: colour is not a position.
        hue = _opt_hue(row.get("hue"))
        color_name = _opt_color_name(row.get("color_name"))
        body_side = _opt_body_side(row.get("body_side"))
        # optional track id (yolo + OPENBOT_INSTANCE_IDS=1).
        # int >= 1 or None - never guessed.
        instance_id = _opt_instance_id(row.get("instance_id"))
        # ref is from the scene graph, never perception
        out.append(Detection(label=label, x=x, y=y, z=z, conf=conf,
                             ref=None, ts=now, orientation=orientation,
                             width_m=width_m, height_m=height_m,
                             hue=hue, color_name=color_name,
                             body_side=body_side, instance_id=instance_id))
    return out


def _joint_vector(mapping: dict) -> List[float]:
    """Top-level per-joint keys -> commanded vector (joint-id order).

    Non-digit keys (``available``, ``actual``, ``mode``, ``motorstate``)
    never masquerade as joints; an unparseable value is skipped.
    """
    out: List[float] = []
    for key in sorted((k for k in mapping.keys()
                       if isinstance(k, str) and k.isdigit()), key=int):
        try:
            out.append(float(mapping[key]))
        except (TypeError, ValueError):
            continue
    return out


def _measured_vector(actual) -> List[float]:
    """The reply's ``actual`` mapping -> MEASURED vector, or ``[]``.

    ``[]`` (unmeasurable) whenever ``actual`` is absent, not a mapping, has
    no digit-keyed entry, or carries ANY non-numeric / bool / non-finite
    value — a partially-readable measurement is no measurement (a NaN
    passes every abs-delta test in is_open, the exact false-pass the grip
    server's own NaN guard exists to prevent).  Never faked from the
    commanded values.
    """
    if not isinstance(actual, dict):
        return []
    keys = [k for k in actual.keys() if isinstance(k, str) and k.isdigit()]
    if not keys:
        return []
    out: List[float] = []
    for key in sorted(keys, key=int):
        value = actual[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []
        q = float(value)
        if not math.isfinite(q):
            return []
        out.append(q)
    return out


def parse_grip_line(line: str) -> Dict[str, GripState]:
    """Parse one grip-server ``state`` reply line -> per-side ``GripState``.

    Reply grammar (skills/arm/grip_server.py ``_state_payload``):
    ``DONE:<json>`` where the json is, for one or both sides,
    ``{side: {"<jid>": last_q, ..., "available": bool,
              "actual": {"<jid>": q, ...},      # OPTIONAL: measured motor q
              "mode": {...}, "motorstate": {...}}}``   # OPTIONAL (B0.2)
    The TOP-LEVEL digit keys are the last COMMANDED q (``controller.last_q``)
    -> ``GripState.commanded``.  ``actual`` is the MEASURED motor q, present
    whenever the Dex3 DDS lane has state and OMITTED otherwise — never faked
    from ``last_q`` (grip_server.py:1036-1048: comparing against ``last_q``
    'would pass unconditionally') -> ``GripState.joints``, ordered by joint
    id; ``[]`` when ``actual`` is absent or malformed (see
    :func:`_measured_vector`).  ``mode`` / ``motorstate`` are not read here.
    ``ERROR:`` / malformed lines yield ``{}`` (grip unknown — never a guess).
    ``contact`` is left False here; the caller folds the last
    ``close_adaptive`` verdict in via :meth:`Assembler.note_close_result`.
    CONTACT LAW (world-truth round 4): contact is NEVER derived from the
    joints, and no hand-state field is read from the reply; there is no
    literal hand-state contact in a state reply.
    """
    if not line.startswith("DONE:"):
        return {}
    try:
        payload = json.loads(line[len("DONE:"):])
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    states: Dict[str, GripState] = {}
    for side, data in payload.items():
        if side not in ("right", "left") or not isinstance(data, dict):
            continue
        states[side] = GripState(
            side=side,
            joints=_measured_vector(data.get("actual")),
            contact=False,
            available=bool(data.get("available", False)),
            commanded=_joint_vector(data),
        )
    return states


def parse_walk_status(payload: dict, ts: float = 0.0) -> Optional[WalkStatus]:
    """Normalize a ``/walk_status`` payload dict -> ``WalkStatus``.

    The walk node publishes ``{"state": <phase>, "label": ..., "detail": ...}``
    JSON (the payload carries NO timestamp — the caller stamps ``ts`` on
    arrival; the gateway does exactly that). DETECTED is a vocabulary event
    inside SEARCHING (see ``WALK_STATE_MAP``); any other state string passes
    through as-is.  The payload ``label`` is captured so the navigate
    verifier can reject a REACHED whose label does not match the requested
    target (fix 1).
    """
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    if not isinstance(state, str) or not state:
        return None
    label = payload.get("label")
    if label is not None and not isinstance(label, str):
        label = None
    return WalkStatus(phase=WALK_STATE_MAP.get(state, state),
                      label=label, ts=ts)


class Assembler:
    """Stateful, pure-stdlib assembler of skill payloads -> WorldObservation.

    The gateway node keeps one instance and feeds it every packet it sees;
    ``observation()`` is called on every poll (even silent ones) so the
    ``fresh`` flag always reflects the current gap since the last packet.
    """

    def __init__(self, max_stale_s: float = MAX_STALE_S,
                 contact_window_s: float = CLOSE_CONTACT_WINDOW_S,
                 silence_means_empty: bool = True,
                 walk_max_stale_s: float = WALK_MAX_STALE_S):
        self.max_stale_s = max_stale_s
        self.walk_max_stale_s = walk_max_stale_s
        self.contact_window_s = contact_window_s
        #: silence switch. True (shipped): a live channel's
        #: empty last packet IS an empty scene; silent
        #: forever = DEAD node. False: silence is never
        #: empty-scene evidence. the verifier's
        #: target-gone path wires the same value.
        self.silence_means_empty = silence_means_empty
        self._detections: List[Detection] = []
        self._last_detections_ts = 0.0  # 0.0 = never observed
        self._grip_states: Dict[str, GripState] = {}
        self._grip_state_ts: Dict[str, float] = {}  # side -> last read's ts
        self._grip_error_ts = 0.0  # read failed (0 = never)
        self._open_refs: Dict[str, List[float]] = {}  # side -> measured ref
        self._open_candidates: Dict[str, List[float]] = {}  # side -> first read
        #: what qualifies for the CAUSAL OPEN REFERENCE.
        #: open-trust starts at a dispatched open
        #: (note_open_commanded), ends at a pick/close.
        #: NO boot window: boot reads guarantee nothing -
        #: a mid-grasp boot would make the closed shape
        #: the 'open' ref. double-read agreement only.
        self._open_trusted: Dict[str, bool] = {"right": False, "left": False}
        self._close_verdicts: Dict[str, tuple] = {}  # side -> (verdict + when)
        self._active_side = "right"  # pick defaults to the right arm
        self._walk: Optional[WalkStatus] = None
        self._last_walk_ts = 0.0  # 0.0 = never seen (see observation())

    # ── taking packets in ──────────────────────────────────────────────────

    def update(self, detections: Optional[List[dict]] = None,
               grip_line: Optional[str] = None,
               walk_payload: Optional[dict] = None,
               now: Optional[float] = None) -> WorldObservation:
        """Ingest any subset of raw payloads, return the current observation.

        ``detections`` rows are timestamped on arrival (a packet, even an
        empty list, is a fresh observation of the scene). ``grip_line`` is the
        grip server's ``state`` reply. ``walk_payload`` is the parsed
        ``/walk_status`` dict, timestamped on arrival and served only while
        younger than ``walk_max_stale_s``.
        """
        now = time.time() if now is None else now
        if detections is not None:
            self._detections = parse_detections(detections, now)
            self._last_detections_ts = now
        if grip_line:
            # kept raw (contact=False); contact is derived on
            # read from verdict + window, so expiry is right.
            states = parse_grip_line(grip_line)
            if not states:
                # ERROR:/malformed = READ FAILURE: last good
                # state must not confirm forever. next good read
                # supersedes it (grip_state compares ts).
                self._grip_error_ts = now
            else:
                for side, st in states.items():
                    self._grip_states[side] = st
                    self._grip_state_ts[side] = now
                    self._update_open_ref(side, st.joints)
        if walk_payload is not None:
            walk = parse_walk_status(walk_payload, ts=now)
            if walk is not None:
                self._walk = walk
                self._last_walk_ts = now
        return self.observation(now=now)

    def note_close_result(self, side: str, verdict: str,
                          now: Optional[float] = None) -> None:
        """Record a ``close_adaptive`` reply (DONE:contact|full|no_data).

        This is the only source of truth for ``GripState.contact`` — the grip
        server does not auto-verify, so contact exists only right after a
        commanded close. Also marks ``side`` as the active one for
        ``WorldObservation.grip``.
        """
        if side not in ("right", "left"):
            return
        self._close_verdicts[side] = (verdict, time.time() if now is None else now)
        self._active_side = side

