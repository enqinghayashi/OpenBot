"""gateway_node.py — rclpy Foxy node bridging brain tools to the skill sockets.

Purpose: the Stuart-deploy-only ROS node that (1) subscribes
``/object_detections_body`` and ``/walk_status``, (2) assembles a
``WorldObservation`` (see gateway/assemble.py), (3) exposes tool verbs to the
brain agent over a JSON-lines socket (single-reader; thread-per-connection),
(4) bridges to the merged-writer / grip_server sockets and the pick engine
(stdin protocol), and (5) runs the SkillVerifier post-hook after motion verbs,
publishing the verdict on ``/skill_verdict``.

Platform laws enforced here (verified on Stuart):
* Locomotion ONLY via the writer ``cmd_set`` verb or the walk FSM topics
  (``/walk_target`` + ``/walk_go``). NEVER sport RPC 3102 while the writer
  owns the robot — this node never calls sport RPC at all.
* Navigate verifies on ``/walk_status == REACHED`` ONLY (no odom exists).
* Pick is TABLETOP-ONLY (workspace torso-frame z >= ~-0.35 m: z=0 is the
  torso link, ground objects read ~-0.75): a floor pick request is
  refused honestly with an ERROR — there is no ground-pick skill.
* Detections are event-driven: silence == empty scene; the observation's
  ``fresh`` flag derives from the gap since the last packet.
* GRASP EVIDENCE LAW (world-truth round 3, restated round 4): during a REAL
  pick the engine closes the grip over ITS OWN socket — the gateway never
  sees the close_adaptive verdict, so a serialized gateway-close verdict can
  never certify a real pick.  When the palm DDS lane is UP the engine
  VERIFIES its own grasp (quadrant/palm-data check) and fails with
  ERROR:grasp_failed on an air-grab; in BLIND MODE (lane down) it skips
  verification and returns DONE anyway.  The gateway CANNOT see the engine's
  grip-socket traffic; its IMMEDIATE post-pick ``state [side]`` poll's
  per-side available flag IS the lane-health probe the gateway CAN see:
  ``_do_pick`` records ``lane_state=up|down|unknown`` + ``grip_state_ts``
  into the tool result (lane up <=> this engine DONE was engine-verified).
  The serialized close-verdict path remains ONLY as evidence for the
  close_hand tool itself (and its informational prev_contact annotations on
  open_hand).  CONTACT LAW: contact is ONLY a literal close_adaptive
  DONE:contact — never derived from joints.  The grip state read AGES
  (GRIP_STATE_MAX_AGE_S): a daemon'd grip keeps its last good
  'available=True' forever no more, and open verification uses the MEASURED
  hand state (joints vs the CAUSAL measured-open reference — world-truth
  round 5: per side the reference forms ONLY from a double-read agreement
  taken AFTER a gateway-dispatched open command for that side; boot reads
  never qualify; never a single-first-read bootstrap), never
  command-inference.  close_both (sides=both) dispatches close_adaptive
  RIGHT then close_adaptive LEFT and lists the per-side verdicts — the
  plain 'close_both' verb survives ONLY as the grip passthrough, not the
  brain close_hand tool.  The detections channel gates target-gone on
  LIVENESS (DETECTIONS_ALIVE_S), not packet freshness; silence == empty
  scene per the silence_means_empty switch.
* navigate 'stop' claims NO goal post-condition: it is always routed to the
  coordinator (pick_walk_coordinator.py), which resets its own epoch/flow
  state and forwards the FSM's real stop trigger (/walk_target 'stop') — the
  gateway publishes /navigate_target only and gets NO verdict.

IMPORTANT — module top level imports ONLY stdlib. rclpy and the brain modules
are imported lazily inside methods, so this module imports cleanly on a laptop
with neither ROS nor brain/ present (self-test requirement).

Tools socket protocol (json lines, one request -> one reply line):
    IN:  {"verb": "<verb>", "args": [<json values>]}
    OUT: {"ok": true|false, "reply": "<text>", "verdict": "<VERDICT:...|null>"}
Verbs:
    pick <x> <y> <z> [nx ny nz] [orientation] [height_m]   -> pick engine
    pick_left ...                                          -> pick engine (left arm)
    place                                                  -> engine 'place' (release-only)
    navigate <label> | stop                                -> /navigate_target (two-step: the LABEL
                                                             then auto-'confirm' — coordinator does
                                                             the search-then-go handshake)
    turn_around [label]                                    -> BLOCKS (<= 115 s sweep + 12 s centre):
                                                             label (default 'person') goes
                                                             DIRECTLY on /walk_target (never
                                                             /walk_go); on sight the walk is
                                                             stopped and the person is CENTRED
                                                             in the frame by writer cmd_set
                                                             omega-only pulses (no walking).
                                                             Reply ends TURN_RESULT=centred
                                                             (only that one starts 'DONE') |
                                                             seen_not_centred | not_seen |
                                                             cancelled
    open_hand [side]                                       -> grip socket open
    close_hand [side] [max_fraction steps dt stop_threshold min_steps]
                                                           -> grip socket close_adaptive
    observe                                                -> current WorldObservation JSON
    image                                                  -> latest camera frame JSON
                                                             ({jpeg_b64, w, h, ts, fresh};
                                                             fresh:false + reason when
                                                             stale/no_camera/encode_error)
    status                                                 -> writer + grip + walk snapshot
    query_user <question>                                  -> print the question, wait for ONE
                                                             human answer line on stdin; reply
                                                             DONE:<answer> | ERROR:no_user_reply:timeout
    notify_user <message>                                  -> print a notice (non-blocking);
                                                             reply DONE:notified
    cmd_set <vx> <vy> <omega> [dur] | off                  -> writer cmd_set
    writer <verb> [args...]                                -> passthrough to writer socket
    grip <verb> [args...]                                  -> passthrough to grip socket
    engine <verb> [args...]                                -> passthrough to the
                                                              engine (its unix
                                                              socket when one is
                                                              live, else a spawn)
    ping                                                   -> PONG
The human-confirm gate is the brain agent's pre-hook (brain/hooks.py); this
node executes what it is told and reports honest results.

ROS topic (not a tools verb): /place_target <label> is the ONE-SHOT
"place the held object ON <label>" lane.  It resolves <label> against the
fresh /object_detections_body scene, computes a wrist target just above that
object and dispatches the coordinate place above -> the shipped
'place x y z' engine lane, so the held-side law, the unheld refusal and the
post-place verifier are inherited rather than re-derived.  It answers with a
phase table (resolve / target / execute / verify / VERDICT) on BOTH channels:
the gateway log, and one JSON object per phase on /place_status whose LAST
object also carries the table.  There is no confirm step.  Every unresolvable
target REFUSES with no engine command — the lane never degrades to a bare
current-pose release.  The pure half lives in gateway/place_target.py.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import select
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ── defaults (most can be overridden via ROS params) ────────────────────────
WRITER_SOCK_DEFAULT = "/tmp/merged_writer.sock"
GRIP_SOCK_DEFAULT = "/tmp/grip_server.sock"
TOOLS_SOCK_DEFAULT = "/tmp/openbot_gateway.sock"
#: the engine's own listener. connect first and spawn only when
#: nothing is listening: a second engine would drive both arms to
#: neutral and drop the arms before a place starts.
ENGINE_SOCK_DEFAULT = "/tmp/pick_engine.sock"
#: newest /object_detections_body packet, all rows, written
#: atomically so the engine can re-detect between attempts.
#: OPENBOT_BODY_ROWS=0 turns it off.
BODY_ROWS_FILE = "/tmp/openbot_body_rows.json"


def write_body_rows(rows, path=BODY_ROWS_FILE, now=None, environ=None):
    """Write {"t": now, "rows": rows} atomically; never raises.  True = written."""
    env = os.environ if environ is None else environ
    if str(env.get("OPENBOT_BODY_ROWS", "1")).strip() == "0":
        return False
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"t": time.time() if now is None else float(now), "rows": rows},
                      fh, separators=(",", ":"))
        os.replace(tmp, path)
        return True
    except Exception:
        return False


#: a transport failure after the command left: pipe readline dies
#: or times out, socket sendall fails, socket readline dies. a
#: failed send must not read as an engine error, so classify by the
#: prefix. 'pick engine not running' is pre-send and stays a plain
#: ConnectionError.
ENGINE_TRANSPORT_ERROR_PREFIX = "ERROR:transport:"
# ── measuring Dex3 closure (the daemon-hold tier) ───────────────────────────
#: the presets mirror the grip server's open/closed tables, which are
#: the source of truth. the gateway needs them to score a hand the
#: engine closed, because the grip server has no 'what fraction' verb.
#: if the presets move, keep these in step: a stale table feeds a
#: wrong fraction into the held-side call.
#: left mirrors right for j3-j6 only; j0-j2 differ in endpoints and
#: travel, so a both-closed compare lands on different rulers there.
#: a side that reads nothing yields None, so only a hand that
#: measures can be scored.
DEX3_JOINT_IDS = (0, 1, 2, 3, 4, 5, 6)
DEX3_OPEN = {
    "right": {0: 0.48, 1: 0.30, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
    "left": {0: -0.48, 1: -0.30, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
}
DEX3_CLOSE = {
    "right": {0: -0.02676787, 1: -1.01299822, 2: -1.68802154,
              3: 1.57079632, 4: 1.74532925, 5: 1.57079632, 6: 1.74532925},
    "left": {0: 0.00, 1: 0.45, 2: 1.00, 3: -1.57079632, 4: -1.74532925,
             5: -1.57079632, 6: -1.74532925},
}
#: sides whose absence from a measurement is excused: a missing
#: reading cannot name a holder, though a present one still can.
#: empty this tuple to require every hand to measure, which leaves
#: the tier inert while the left hand never measures.
#: load-bearing: a left pick opens both hands, and in the
#: mode-absent regime the right latch triad (j1/j3/j5 at close)
#: still reads 3/7 = 0.43 >= CLOSED_EFF_MIN, naming 'right' while
#: the left holds.
DEX3_OUT_OF_SERVICE_SIDES = ("left",)
#: mean closure at which a hand counts as meaningfully closed. the
#: failure mode is asymmetric by design: a false closed releases an
#: empty hand, one wasted motion, while a false not-closed refuses a
#: place while holding. so stay permissive; the daemon's own `held`
#: gate is what keeps it honest.
#: no joint-space rule separates a hand parked closed on air from
#: one holding (a parked empty pose still scores 0.3745), so this
#: is a not-open check, not a hold detector.
CLOSED_EFF_MIN = 0.25
#: budget (s) per `state <side>` probe: a wedged grip server then
#: delays a place by a known amount, not the 15 s verb default. two
#: probes run per place.
CLOSURE_PROBE_TIMEOUT_S = 2.0
#: a joint is scored only with a usable span between presets. j0
#: travels 0.507 rad and every other right joint at least 1.0; below
#: this the division amplifies noise into a fake fraction.
MIN_SPAN_RAD = 0.05
#: joints that must survive to score a hand at all. below this the
#: hand is unmeasurable (None) rather than scored on the remainder:
#: two limp joints can sum to ~1.0 and license a place into an empty
#: hand. a hard close latches three joints (j1/j3/j5), leaving four,
#: so do not raise this above 4.
#: with mode reported, mode 0 (limp) is excluded and the floor counts
#: live joints; without mode nothing is excluded and the floor counts
#: scored joints, so a live hold reads 0.668 instead of 0.786. do not
#: raise CLOSED_EFF_MIN to chase the filtered reading.
MIN_SCORED_JOINTS = 4
PICK_MIN_Z_M = -0.35        # tabletop-only pick bound, in the torso frame
#: z is height above the torso link. ground objects read ~-0.75;
#: tabletop detections land in [0.00, +0.07].
#: the engine applies the same bound as its sanity gate: a height_m
#: <= 0 or above the max falls back to the plain path. the
#: named-args shaper drops a bad height_m, so junk tolerance never
#: shifts the positional grammar.
PICK_HEIGHT_M_MAX = 0.5
#: torso-frame match radius (m) for resolving an 'auto' or missing
#: pick orientation to the freshest nearby detection's. the brain
#: binds orientation one layer up and passes it concretely, so this
#: only covers a direct caller that leaves it out and a lying object
#: still gets an upright approach. nothing in range, or a stale or
#: empty scene, gives the safe 'upright' default.
PICK_ORIENTATION_MATCH_M = 0.3
#: radius park: a row's radius history counts only if the row sits
#: within this distance (m) of the commanded point, and only for
#: samples this recent (s).
RADIUS_PARK_MATCH_M = 0.05
RADIUS_PARK_WINDOW_S = 4.0
# the projector's surface normal per object (nx, ny, nz, torso
# frame, fit at the mask centre) rides on the engine pick line:
# `pick x y z nx ny nz <orientation> [height_m] [x_off]`.
# looked up like radius_park: row within RADIUS_PARK_MATCH_M, its
# instance id, that instance's newest normal, no older than
# PICK_NORMAL_MAX_AGE_S and within PICK_NORMAL_NORM_TOL of unit
# length. the engine logs it but grips with fixed wrist targets, so
# normal-driven wrists stay a gantry decision.
# OPENBOT_PICK_NORMALS=0 (gateway env) turns it off.
PICK_NORMAL_MAX_AGE_S = 2.0
PICK_NORMAL_NORM_TOL = 0.1
#: a 'pca' fit's object_radius is a point spread, and 'sphere'
#: never publishes a radius (constant 0.035 default). for those,
#: take half the row's lateral width (p95-p5 of torso x).
#: OPENBOT_PICK_RADIUS_WIDTH=0 puts the fit radius back.
RADIUS_PARK_WIDTH_TYPES = ('pca', 'sphere')
#: default park margin (m) and the clamp for
#: OPENBOT_PICK_RADIUS_MARGIN_M. the engine caps the park at
#: 0.115 m, which must stay under the 0.13 m approach offset.
#: env 0.015 gives a tighter park margin.
RADIUS_PARK_MARGIN_DEFAULT_M = 0.050
RADIUS_PARK_MARGIN_MIN_M = 0.005
RADIUS_PARK_MARGIN_MAX_M = 0.060
#: a DONE place line carries `release_open=<side>:True` when the
#: engine measured the hand open at release, and the lift closeout
#: then parks that hand closed. that shape is the park-closed
#: marker: the next reading is closed, so the place must not mark
#: the side open-commanded, or the closed shape becomes the open
#: reference and flips every later is_open.
PLACE_PARK_CLOSED_RE = re.compile(r"release_open=(right|left):True")
#: how long (s) a DONE place keeps its name in the unheld refusal.
PLACE_RELEASE_NOTE_S = 300.0
#: default floor on arm-dispatch duration (s). the writer clamps at
#: 1.5 s, so the effective floor is max(embodiment min_arm_dur_s,
#: 1.5), resolved at runtime.
MIN_ARM_DUR_S_DEFAULT = 1.5
# ── descent-lane defaults for the low workspace ─────────────────────────────
#: nominal standing reference (m) the writer reports as
#: ``height_stand``. simulated, and only a gateway default: the live
#: value comes from ``embodiment.envelope.height_stand_m`` and the
#: lane is off.
HEIGHT_STAND_M_DEFAULT = 0.74
#: floor of the writer's height clamp (m). the descent budget is
#: HEIGHT_STAND_M_DEFAULT - HEIGHT_MIN_M_DEFAULT, and the writer
#: ships that band shut (min == max == standing), so descent is
#: mechanically impossible until the band is widened. matches the
#: EnvelopeConfig default.
HEIGHT_MIN_M_DEFAULT = 0.55
#: same-height hold tolerance (m of body height): reuse a descent
#: only if |h_need - h_cur| <= this. same as EnvelopeConfig.
HOLD_TOL_M_DEFAULT = 0.05
#: status-poll budget (s) for a confirmed descent or retreat. same
#: as EnvelopeConfig and as the lane's own
#: DESCENT_CONFIRM_TIMEOUT_S.
DESCENT_CONFIRM_TIMEOUT_S_DEFAULT = 3.0
#: default frame for detections (embodiment.perception.frame:
#: 'torso'; x lateral right, y forward, z up, in metres). the same
#: right-handed frame fixes the cmd_set omega sign.
DETECTIONS_FRAME_DEFAULT = "torso"
MAX_STALE_S_DEFAULT = 5.0
# ── turn_around centring ────────────────────────────────────────────────────
#: sweep budget (s), from a full 360 deg sweep at 77 s: 12 turns x
#: (DETECT_WINDOW_S 5.0 + TURN_DURATION_S 1.047 + TURN_SETTLE_S 0.3).
TURN_SWEEP_BUDGET_S = 90.0
#: longest wait (s) for the walk FSM to go IDLE after stop_walk(),
#: before the first centre pulse.
TURN_STOP_SETTLE_S = 2.0
#: centring budget (s): 180 deg at OMEGA_MAX 0.40 rad/s is 7.9 s,
#: plus settling.
TURN_CENTRE_BUDGET_S = 12.0
#: whole-call budget for the sweep wait (s). it must stay below
#: brain_client.GATEWAY_CALL_TIMEOUT_S 150.0, so the ladder is
#: 115 < 150 < Tool 180. centring can add TURN_CENTRE_BUDGET_S, and
#: a re-sweep is refused unless that much is left, so the worst case
#: is 115 + 12 = 127 s, still under 150.
TURN_TOTAL_BUDGET_S = 115.0
#: bearing that counts as in tolerance (rad), about 5 deg.
TURN_TOL_RAD = 0.09
#: yaw gain for the centring loop.
TURN_K_THETA = 1.5
#: yaw-rate clamp (rad/s).
TURN_OMEGA_MAX = 0.40
#: +omega turns left while body-frame +x points right, so the sign
#: is -1.
TURN_OMEGA_SIGN = -1.0
#: floor for yaw gait speed (rad/s).
TURN_OMEGA_MIN = 0.15
#: one centre pulse (s). the writer's cmd_set auto-expires through
#: cmd_auto_deadline, so a dead gateway stops the robot within
#: 0.35 s.
TURN_PULSE_S = 0.35
#: loop tick (s).
TURN_TICK_S = 0.1
#: a detection older than this (s) is not a bearing.
TURN_ROW_MAX_AGE_S = 0.8
#: target gone this long (s) counts as lost.
TURN_LOST_S = 2.0
#: in-tolerance reads, all distinct, needed to call it centred.
TURN_CONFIRM_READS = 2
#: re-sweeps allowed after losing the target mid-centring.
TURN_MAX_RESWEEPS = 2
#: 'person' plausibility: confidence floor for treating a
#: detection as a person. label 'person' only; stray detections read
#: 0.36-0.54 confidence and ~0.11 m height, real people 0.68-0.90
#: and ~0.94-0.97 m.
TURN_PERSON_MIN_CONF = 0.50
#: minimum height (m) for a person detection.
TURN_PERSON_MIN_HEIGHT_M = 0.30
#: confidence floor for non-person labels.
TURN_OTHER_MIN_CONF = 0.25
#: association gate (rad): once locked, a row more than this off the
#: locked bearing is a different object and is ignored, leaving nothing
#: inside the gate for TURN_LOST_S. about 2.3 pulses of travel at
#: OMEGA_MAX.
TURN_ASSOC_GATE_RAD = 0.35


def _turn_wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def _turn_bearing(x: float, y: float) -> float:
    """Bearing (rad) of a torso-frame point: x = RIGHT, y = forward.

    The SAME law as walk_node.py:1394/1423 ``yaw_err = _wrap_pi(
    math.atan2(x_b, y_b))``.  Positive = target is to the RIGHT of centre.
    """
    return _turn_wrap_pi(math.atan2(x, y))


def _turn_omega(yaw_err: float) -> float:
    """Yaw rate (rad/s) that turns the body toward the bearing.

    walk_node.py:1424 law, ``OMEGA_SIGN * K_THETA * max(-OMEGA_MAX,
    min(OMEGA_MAX, yaw_err))``, with two additions made here: the final
    magnitude is clamped to TURN_OMEGA_MAX (walk_node's clamp is on the
    ERROR, so its output can reach 0.60; the TURN spec caps the rate at 0.40),
    and a non-zero command is raised to TURN_OMEGA_MIN (gait floor).  Inside
    the TURN_TOL_RAD dead-band the answer is exactly 0.0.
    """
    if abs(yaw_err) <= TURN_TOL_RAD:
        return 0.0
    omega = TURN_OMEGA_SIGN * TURN_K_THETA * max(
        -TURN_OMEGA_MAX, min(TURN_OMEGA_MAX, yaw_err))
    mag = min(abs(omega), TURN_OMEGA_MAX)
    if mag < TURN_OMEGA_MIN:
        mag = TURN_OMEGA_MIN
    return math.copysign(mag, omega)
# ── hold evidence from the pick daemon (the /pick_state fold) ───────────────
#: the pick daemon's state topic (JSON at 5 Hz: state / label / held /
#: daemon_alive / side). the gateway refuses an unheld place, and its
#: other two hold tiers are made by the gateway itself, so a pick on the
#: daemon lane is invisible to both and a place on a payload it holds
#: refuses with 'nothing verified held'. the daemon already knows, so
#: read it here.
#: scope: consulted only inside the unheld-refusal branch. as a peer
#: branch above the refusal it would pre-empt the ambiguous-hint `else`
#: and release the daemon's hand where that path releases active_side.
#: the payload's `side` is required: it is the hand the daemon actually
#: closed. no side means the tier is inert and the refusal stands. never
#: fall back to a declared constant, because the daemon picks either
#: hand and a constant would open the empty right hand while the payload
#: stays gripped.
PICK_STATE_TOPIC_DEFAULT = "/pick_state"
#: older than this (s) is not evidence. the daemon publishes at 5 Hz,
#: so 2.0 s rides out ~10 dropped messages and stays far too short for
#: a dead daemon's last 'held: true' to authorise a place.
PICK_STATE_MAX_AGE_S = 2.0
#: marks a new pick cycle: _execute_pick's first act, before it
#: touches `held`. it ends the previous hold epoch in _on_pick_state,
#: which releases the consumed latch.
PICK_STATE_BUSY = "busy"
#: how long _await_pick_daemon waits for a pick motion before giving
#: up. the engine answers DONE about 3 s after the command while the
#: arm still moves; a grasp verdict lands near t+26 s and the carry tuck
#: near t+33 s, so 120 s covers about 3.5 cycles. a wait bound, not a
#: gate: past it nothing is refused, the wait stops and reports
#: `timeout`. do not drop below ~60 s.
PICK_DAEMON_WAIT_S = 120.0
#: how long _await_pick_daemon gives a cycle to begin (the daemon's
#: `busy` frame). a live daemon writes it well under a second after the
#: engine answered DONE, so 5.0 s is 25 publish periods at 5 Hz and a
#: lane with no daemon pays 5 s once instead of the whole motion
#: budget.
PICK_DAEMON_START_S = 5.0
#: poll period (s) while waiting. publishing is 5 Hz, so 0.1 s cannot
#: miss a frame; it only bounds how late the edge is noticed.
PICK_DAEMON_POLL_S = 0.1
#: how long `cancel` waits for the engine socket. the engine answers
#: from its connection thread, so the reply is immediate by
#: construction and this bounds only a wedged listener or an overloaded
#: host. short on purpose: a blocking cancel is useless to an operator
#: watching the arm keep moving.
CANCEL_REPLY_TIMEOUT_S = 5.0


def _defang_error_token(text: str) -> str:
    """Break the ``ERROR:`` literal so an embedded echo cannot poison a reply.

    ``_shape_reply`` (see its docstring) sets ``ok:false`` when ``ERROR:``
    appears ANYWHERE in the reply body, and the verifier's ``_classify_reply``
    reads it as a failure.  The cancel reply ECHOES the engine's own words for
    the wire trace — and the engine's error vocabulary is exactly ``ERROR:``.
    So an engine that answers a cancel with, say, ``ERROR:not busy`` would turn
    a cancel that WORKED into a failure envelope: the brain would report the
    stop as failed and the operator would be told the opposite of the truth.

    This is the same bug class the ``_shape_reply`` docstring already names for
    ``close_both`` (``right_close=ERROR:no_dds`` inside a DONE lead).

    The substitution is deliberately LOUD rather than silent — an underscore is
    visible in the wire trace, so a reader can still see what the engine said
    while the token no longer parses as an error.  Only the ECHO is defanged;
    a reply that genuinely IS an error still leads with its own ``ERROR:`` and
    is shaped as one.
    """
    return text.replace("ERROR:", "ERROR_")


#: Reason on every verdict minted for a cancelled pick or place. same
#: token the brain verifier uses for a cancelled navigate.
CANCEL_REASON = "cancelled_by_operator"

_CANCEL_PHASE_RE = re.compile(r"[A-Za-z0-9_.\-]+")


def _cancel_phase(reply: str) -> str:
    """The ``<where>`` of an engine ``CANCELLED:<where>`` reply, sanitised.

    Text after the first ``CANCELLED:``, cut at the first whitespace, kept only
    while it is ``[A-Za-z0-9_.-]`` and at most 40 characters; anything else
    (empty, or a first character outside the charset) is ``"unknown"``.  The
    phase is interpolated into a ``|``-delimited verdict, so a ``|`` or a
    newline must never survive this.
    """
    text = str(reply or "")
    idx = text.find("CANCELLED:")
    if idx < 0:
        return "unknown"
    tail = text[idx + len("CANCELLED:"):]
    parts = tail.split(None, 1)
    token = parts[0] if parts else ""
    m = _CANCEL_PHASE_RE.match(token)
    if m is None:
        return "unknown"
    return m.group(0)[:40]


def _cancelled_verdict(tool: str, where: str) -> str:
    """The VERDICT for a cancelled ``tool`` — the verifier's own grammar.

    Byte-equal to ``brain.verifier.Verdict.failure(evidence,
    'cancelled_by_operator').to_string()`` (pinned in
    tests/lane_cancel_and_verdict_pins.py).  ``status`` failure: a cancel is
    NEVER a completed pick or place, and the gateway does not add a 4th status.
    """
    return ("VERDICT:failure|Evidence:%s cancelled by the operator at phase %s "
            "- the arm was frozen mid-motion; this is NOT a completed %s"
            "|Reason:%s" % (tool, where, tool, CANCEL_REASON))


def _pre_descent_step_flag(envelope: Any) -> bool:
    """``embodiment.envelope.pre_descent_step`` — armed ONLY by a real ``True``
    (the loader's strict-bool discipline; a stub or an omitted key is False)."""
    return getattr(envelope, "pre_descent_step", False) is True


#: every shape the engine's command loop puts on the wire, from its
#: own stdout writes: READY at boot, DONE / DONE:<names> /
#: DONE; release_open=... for pick and place, ERROR:... (including the
#: ERROR:transport: label) and CANCELLED:<where>.
_ENGINE_REPLY_PREFIXES = ("DONE", "ERROR:", "CANCELLED:", "READY")


def _is_engine_reply(text: str) -> bool:
    """True iff ``text`` is shaped like something the ENGINE actually wrote.

    WHY THIS EXISTS (run10, 2026-10-01 — the operator's "the successful pickup
    was never confirmed").  Ipopt's licence banner is written through the
    Python ``sys.stdout`` OBJECT, and the daemon's entrypoint routes every
    Python-level write on ``sys.stdout`` to the CLIENT while a command is in
    flight.  The banner therefore landed in the reply stream ahead of the real
    reply and drained ONE LINE PER COMMAND: six successive commands consumed
    six banner lines (``gateway.log:26,36,45,85,96,103``).

    Both callers were fooled, and neither could tell, because both decided by
    a NEGATIVE test over a string they had not checked the shape of:

      * ``_do_pick``  — ``if not reply.startswith("ERROR")`` treats *any*
        non-error string, including ``""`` and ``"****"``, as an executed
        pick.  Three engine REFUSALS (OUT OF REACH) were reported to the brain
        as ``DONE pick executed``.
      * ``_do_place`` — ``if reply.startswith("ERROR")`` is its ENTIRE control
        flow, so a banner line takes the SUCCESS branch: the daemon hold record
        is not restored and the hand is recorded as released.

    This predicate makes "is this the engine's answer at all" a question that
    must be answered YES before either caller interprets it.  An unreadable
    reply is not a verdict — it is an UNKNOWN, and both callers now say so
    instead of guessing in the optimistic direction.

    The root cause is fixed where it belongs (``_SuppressStdout`` in
    ``pick_engine_v3.py`` now rebinds ``sys.stdout`` as well as fd 1) — this is
    the defence in depth, and it holds for a banner written by ANY library, or
    by a resident engine whose stream is already poisoned.
    """
    return (text or "").strip().startswith(_ENGINE_REPLY_PREFIXES)


# ── camera lane (optional 'image' verb, only when vision.enabled) ───────────
#: mirrors the vision config. the lane is off unless the config turns
#: it on; the getattr defaults keep mock configs with no ``vision``
#: attribute valid.
VISION_TOPIC_DEFAULT = "/camera/color/image_raw"
VISION_MAX_AGE_S_DEFAULT = 2.0
VISION_JPEG_QUALITY_DEFAULT = 80
#: minimum gap (s) between JPEG encodes on the worker; 30 fps would
#: otherwise cost 30 encodes/s. the worker encodes the newest raw frame
#: at most once per interval: a throttled cache, frames skipped, never
#: queued.
VISION_MIN_INTERVAL_S_DEFAULT = 1.0
#: max age (s) of an unconfirmed navigate epoch before a
#: re-invocation forces a fresh dispatch. a 90 s in-flight epoch is
#: almost certainly wedged, since the FSM's own timeout sits well below
#: this.
NAV_EPOCH_MAX_AGE_S = 90.0
#: a /walk_status REACHED newer than this (s) means the FSM stands at
#: that label, so navigate answers already_reached.
NAV_ALREADY_REACHED_FRESH_S = 3.0
#: total budget (s) for _await_walk_reached, the navigate completion
#: wait. a touch under the coordinator's WALK_REACHED_TIMEOUT_S, so a
#: wedged walk yields the gateway's own honest token rather than the
#: coordinator's walk timeout read as a navigate failure. a wait bound,
#: not a gate: past it nothing is refused, it reports timeout_after_walk.
#: ladder: NAV_WAIT_S 115 < GATEWAY_CALL_TIMEOUT_S 150 < Tool 180.
NAV_WAIT_S = 115.0
#: how long (s) _await_walk_reached waits for any in-epoch
#: /walk_status before deciding the lane is not answering this
#: dispatch. the coordinator relays the target, so the first frame
#: lands well under a second and the topic then publishes at 1 Hz;
#: 5.0 s is five periods.
NAV_START_S = 5.0
#: poll period (s) for the navigate wait. /walk_status is 1 Hz and
#: REACHED is held for 5 s, so 0.1 s cannot miss the window and stays an
#: order of magnitude inside it.
NAV_POLL_S = 0.1
#: budget (s) the query_user verb gets for one human answer line on
#: stdin (select polling, 0.2 s grace per iteration). timeout or EOF
#: gives the exact reply 'ERROR:no_user_reply:timeout'.
QUERY_USER_REPLY_TIMEOUT_S = 20.0
#: boot profile for the gateway process alone, when neither --profile
#: nor OPENBOT_PROFILE is given: 'offline', one of the four the loader
#: allows (laptop|cloud|voice|offline; any other raises). the node
#: cold-starts on the offline config, whose gateway section matches the
#: voice config's real transport on 127.0.0.1:3900.
DEFAULT_BOOT_PROFILE = "offline"

# skills.engine_cmd is dual-form on its first token: a verb token
# (the loader default 'pick x y z nx ny nz orientation height_m') means
# the engine is off. anything else is a live cold-start spawn, since
# connect-first normally attaches to the resident engine; the shipped
# config uses the env-fix spawn whose first token is 'env'. a verb form
# yields '' and an honest error.
_ENGINE_VERB_TOKENS = frozenset({
    "pick", "pick_left", "place", "open", "close", "open_left",
    "close_left",
    "open_both", "close_both", "ready", "carry", "carry_left", "neutral",
    "home", "drop", "release", "acquire", "hold", "list", "quit",
})


def _engine_spawn_cmd(engine_cmd: str) -> str:
    """The engine spawn command from the config value, or "".

    First-token discriminator: a verb-token value (the loader-default verb
    form — engine-off documentation) yields ""; config.yaml ships the REAL
    #106 env-fix spawn (first token 'env', exec python3
    .../tabletop_pick_place_engine.py) which passes through and is used as
    the FALLBACK when no resident engine is listening; no profile overrides
    the key.
    """
    cmd = (engine_cmd or "").strip()
    if not cmd:
        return ""
    if cmd.split()[0].lower() in _ENGINE_VERB_TOKENS:
        return ""
    return cmd


def resolve_skill_endpoints(cfg: Any) -> Tuple[str, str, str, int]:
    """Read the skill endpoints + tools TCP port from a loaded config.

    Consumes ``skills.writer_sock`` / ``skills.grip_sock`` /
    ``skills.engine_cmd`` (config.yaml ``skills:`` segment) and
    ``gateway.port`` (``gateway:`` segment).  Stdlib only — importable
    without rclpy, so laptop tests construct the wiring with a stub
    transport.  engine_cmd is DUAL-FORM: the loader-default verb form
    resolves to engine off; the REAL #106 spawn (config.yaml) passes through.
    Returns ``(writer_sock, grip_sock, engine_cmd, tools_port)``.
    """
    skills = getattr(cfg, "skills", None)
    writer_sock = (
        getattr(skills, "writer_sock", WRITER_SOCK_DEFAULT)
        if skills is not None
        else WRITER_SOCK_DEFAULT
    )
    grip_sock = (
        getattr(skills, "grip_sock", GRIP_SOCK_DEFAULT)
        if skills is not None
        else GRIP_SOCK_DEFAULT
    )
    engine_cmd = _engine_spawn_cmd(
        getattr(skills, "engine_cmd", "") if skills is not None else ""
    )
    gateway = getattr(cfg, "gateway", None)
    try:
        tools_port = int(getattr(gateway, "port", 0) or 0)
    except (TypeError, ValueError):
        tools_port = 0
    return writer_sock, grip_sock, engine_cmd, tools_port


def resolve_engine_sock(cfg: Any) -> str:
    """The pick/place engine's listener path from a loaded config (2026-09-13).

    Reads ``skills.engine_sock`` (config.yaml ``skills:`` segment), defaulting
    to :data:`ENGINE_SOCK_DEFAULT`.  Kept OUT of :func:`resolve_skill_endpoints`
    so that resolver's 4-tuple contract is unchanged.

    This value must MATCH the engine's own ``SOCK_PATH``
    (skills/arm/tabletop_pick_place_engine.py) or connect-first silently
    degrades to spawn-always — i.e. back to the arm-drop defect.  Stdlib only.
    """
    skills = getattr(cfg, "skills", None)
    path = getattr(skills, "engine_sock", "") if skills is not None else ""
    return path or ENGINE_SOCK_DEFAULT


def resolve_tools_bind_host(cfg: Any) -> str:
    """The tools-server TCP BIND host from a loaded config (2026-09-02).

    NEW server-side knob: ``gateway.bind_host`` (default ``0.0.0.0`` — the
    historical hardcoded value).  NOTE this is NOT ``gateway.host``: that
    key is client-DIAL-only (brain_client from_config); the TCP listener
    hardcoded 0.0.0.0 and ignored it.  A loopback bind closes the
    unauthenticated motion socket to the LAN (any peer on the robot's
    network can currently drive motion verbs) — but it also breaks any
    LAN dial, e.g. the laptop-brain profile config.cloud.yaml:146 — flip
    that profile to an SSH tunnel in the same change.  Stdlib only.
    """
    gateway = getattr(cfg, "gateway", None)
    host = getattr(gateway, "bind_host", "") if gateway is not None else ""
    return (host or "0.0.0.0")


def _descent_config_float(envelope: Any, attr: str, default: float) -> float:
    """One low-workspace law input off a (possibly stub) envelope, in metres.

    ``getattr``-defaulted for the same reason every read in
    :func:`resolve_embodiment_config` is: a stub config without an
    ``embodiment`` attribute (or an envelope that predates the lane) must still
    construct the node.  A malformed value falls back to the default rather than
    raising — the lane's own ``usable``/``effective_pick_floor`` guards fail
    closed on a bad band, so nothing here has to refuse.
    """
    try:
        value = getattr(envelope, attr, default)
    except Exception:  # noqa: BLE001 - an unreadable attr is a missing value
        return float(default)
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value: Any) -> Optional[float]:
    """One law input that may be legitimately ABSENT, as a float or ``None``.

    The operation budget is the only lane input with no gateway-side default:
    the number lives in exactly one place, ``descent_lane.LANE_OPERATION_BUDGET_S``
    (REUSE LAW — a restated 30.0 here would be a second authority for it), and
    ``None`` is the signal that says "keep the lane's own default".  So a
    malformed value degrades to ``None``/lane-default rather than raising, and a
    ``bool`` — an ``int`` in Python — is not a budget either: mirroring the
    lane's own ``_finite_num``, a ``True`` that reached a writer-hold comparison
    would be a wiring bug, and here it would arm the guard on a number nobody
    wrote.  The lane itself is the FINAL authority: a non-finite or
    non-positive value handed over is rejected by ``LaneLaw.__init__`` and
    replaced by its default, so nothing this side does can shrink the guard to
    nothing.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _descent_operation_budget(envelope: Any) -> Optional[float]:
    """The DECLARED lane operation budget off a (possibly stub) envelope.

    Reads ``embodiment.envelope.lane_operation_budget_s`` and returns
    ``None`` when the key is absent or unusable — which is every stub config and
    every config whose loader does not yet carry the key, because the loader
    half and this half land independently: an attribute access here would take
    the whole node (every pick and place, not just the lane) down on a tree
    where only the gateway half has landed.  ``None`` means the lane keeps
    ``descent_lane.LANE_OPERATION_BUDGET_S``.
    """
    try:
        value = getattr(envelope, "lane_operation_budget_s", None)
    except Exception:  # noqa: BLE001 - an unreadable attr is a missing value
        return None
    return _optional_float(value)


def _lane_law(lane_law_cls: Any, tabletop_min_z: float, height_stand_m: float,
              height_min_m: float, hold_tol_m: float,
              descend_confirm_timeout_s: float,
              operation_budget_s: Optional[float] = None) -> Any:
    """Build the lane's ``LaneLaw``, threading the confirm-timeout knob (A4).

    The lane's own slots for these two knobs are ``LaneLaw.timeout_s`` and
    ``LaneLaw.operation_budget_s`` — the lane chooses those spellings; the
    gateway populates them from the resolved envelope's
    ``descend_confirm_timeout_s`` (repair A4 threads the value, task B owns the
    slot) and ``lane_operation_budget_s`` (this round).  Each is handed over
    only when the class DECLARES the parameter: the two halves of each repair
    live in different files and land independently, and a tree where only the
    gateway half has landed must still CONSTRUCT — a construction that raised
    would take every pick/place, not just the lane, down with it.  When a slot
    is absent the lane keeps its own module default
    (``DESCENT_CONFIRM_TIMEOUT_S`` / ``LANE_OPERATION_BUDGET_S``), which is the
    value it used before the knob existed, so nothing is silently different: the
    knob is either threaded to the lane or left at the lane's own default.

    ``operation_budget_s=None`` is a VALUE, not an omission: it says "the lane
    keeps its own default", so it is passed through rather than dropped, and
    ``LaneLaw.__init__`` restores ``LANE_OPERATION_BUDGET_S`` itself.  Dropping
    it would be a different thing — it would rely on the lane's default merely
    happening to be right instead of saying which is meant.
    """
    try:
        params = inspect.signature(lane_law_cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - non-callable stubs
        params = {}
    kwargs: Dict[str, Any] = {}
    if "timeout_s" in params:
        kwargs["timeout_s"] = float(descend_confirm_timeout_s)
    if "operation_budget_s" in params:
        kwargs["operation_budget_s"] = operation_budget_s
    if not kwargs:
        return lane_law_cls(tabletop_min_z, height_stand_m, height_min_m,
                            hold_tol_m)
    return lane_law_cls(tabletop_min_z, height_stand_m, height_min_m, hold_tol_m,
                        **kwargs)


def resolve_embodiment_config(cfg: Any, with_descent: bool = False
                              ) -> Tuple[Any, ...]:
    """Runtime embodiment values from a loaded config (world-truth round 5).

    Reads ``embodiment.envelope.pick_min_height_m`` (m),
    ``embodiment.envelope.min_arm_dur_s`` (s),
    ``embodiment.perception.silence_means_empty`` (bool) and
    ``embodiment.perception.frame`` (str) via getattr-defaults so stub
    configs without an ``embodiment`` attribute (laptop tests) still
    construct.  Returns ``(pick_min_z_m, silence_means_empty, frame,
    min_arm_dur_s)``.  The default values mirror the config dataclasses in
    brain/config_loader.py (-0.35 / True / 'torso' / 1.5).

    ``with_descent=True`` ALSO computes the low-workspace descent lane's own
    inputs (plan §8 W4, plus the confirm-timeout knob added by repair A4 and the
    operation budget added by this round) and appends them as a 5th element —
    ``(low_workspace_lane, height_stand_m, height_min_m, hold_tol_m,
    descend_confirm_timeout_s, lane_operation_budget_s)``, plain values, so the
    caller needs no lane object and this function needs no import.  The
    operation budget is the ONE input read WITHOUT a gateway-side default: it is
    ``None`` whenever the envelope omits or malforms the key, and ``None`` means
    "the lane keeps its own module default"
    (``descent_lane.LANE_OPERATION_BUDGET_S``), so no second authority for that
    number exists here.  The
    DEFAULT arity is deliberately unchanged: the 4-tuple is what the shipped
    callers (gateway_from_config's non-lane branch, _RosGateway, and the
    laptop config/seam tests) unpack, so every pre-lane call site stays
    byte-identical.  ``low_workspace_lane`` is armed only by a real bool
    ``True`` — the strict discipline the config loader itself applies
    (brain/config_loader.py:802-805) — so a string "false" or a numeric 0 can
    never arm a motion lane.
    """
    emb = getattr(cfg, "embodiment", None)
    envelope = getattr(emb, "envelope", None) if emb is not None else None
    perception = getattr(emb, "perception", None) if emb is not None else None
    try:
        pick_min_z_m = float(
            getattr(envelope, "pick_min_height_m", PICK_MIN_Z_M) or PICK_MIN_Z_M)
    except (TypeError, ValueError):
        pick_min_z_m = PICK_MIN_Z_M
    try:
        min_arm_dur_s = float(
            getattr(envelope, "min_arm_dur_s", MIN_ARM_DUR_S_DEFAULT)
            or MIN_ARM_DUR_S_DEFAULT)
    except (TypeError, ValueError):
        min_arm_dur_s = MIN_ARM_DUR_S_DEFAULT
    silence_means_empty = bool(
        getattr(perception, "silence_means_empty", True)
        if perception is not None else True)
    frame = (
        getattr(perception, "frame", DETECTIONS_FRAME_DEFAULT)
        if perception is not None else DETECTIONS_FRAME_DEFAULT)
    if not with_descent:
        return pick_min_z_m, silence_means_empty, frame, min_arm_dur_s
    descent = (
        bool(getattr(envelope, "low_workspace_lane", False) is True),
        _descent_config_float(envelope, "height_stand_m",
                              HEIGHT_STAND_M_DEFAULT),
        _descent_config_float(envelope, "height_min_m", HEIGHT_MIN_M_DEFAULT),
        _descent_config_float(envelope, "hold_tol_m", HOLD_TOL_M_DEFAULT),
        _descent_config_float(envelope, "descend_confirm_timeout_s",
                              DESCENT_CONFIRM_TIMEOUT_S_DEFAULT),
        # no gateway-side default here, on purpose: None means the lane's own
        # LANE_OPERATION_BUDGET_S.
        _descent_operation_budget(envelope),
    )
    return pick_min_z_m, silence_means_empty, frame, min_arm_dur_s, descent


def resolve_pc_switches(cfg: Any) -> Tuple[bool, bool, bool]:
    """``(pc_shadow, pc_hold_check, pc_place_rim)`` off a (possibly stub) config.

    The batch-3b point-cloud switches of ``embodiment.envelope``.  Each is True
    ONLY for a real bool ``True`` — the loader's strict-bool discipline
    (brain/config_loader.py ``_as_bool``), applied the way
    ``_pre_descent_step_flag`` applies it — so a stub config, an omitted key, a
    string "true" or a numeric 1 is False.  A function of its own, NOT a
    further element of ``resolve_embodiment_config``'s descent tuple: that
    tuple's arity is pinned and unpacked by shipped callers, and a caller that
    never asks for the switches keeps today's behaviour byte for byte.
    """
    emb = getattr(cfg, "embodiment", None)
    envelope = getattr(emb, "envelope", None) if emb is not None else None
    flags = []
    for name in ("pc_shadow", "pc_hold_check", "pc_place_rim"):
        try:
            flags.append(getattr(envelope, name, False) is True)
        except Exception:  # noqa: BLE001 - an unreadable attr is a False switch
            flags.append(False)
    return flags[0], flags[1], flags[2]


def _pc_switch_kwargs(switches: Tuple[bool, bool, bool]) -> Dict[str, bool]:
    """The ``GatewayNode`` keywords for the point-cloud switches.

    ``{}`` when none is ON, so a gateway built with every switch off is built by
    EXACTLY the call it always was (no new keyword reaches a stub or a subclass
    that does not declare them).
    """
    if not any(switches):
        return {}
    return dict(zip(("pc_shadow", "pc_hold_check", "pc_place_rim"), switches))


def resolve_vision_config(cfg: Any) -> Tuple[bool, str, float, int, float]:
    """Vision-lane values from a loaded config (camera OFF by default).

    Consumes ``vision.enabled`` / ``vision.topic`` / ``vision.max_age_s`` /
    ``vision.jpeg_quality`` / ``vision.min_interval_s`` — the v4
    ``vision:`` segment (the VisionConfig dataclass in
    brain/config_loader.py).  ``enabled`` defaults to False, mirroring
    VisionConfig's default (the shipped config now also ships
    ``vision.enabled: false``): the camera lane is OFF by default unless
    the config sets ``vision.enabled: true``.  getattr-defaults keep mock
    configs WITHOUT a ``vision`` attribute valid: the lane stays OFF there
    too (gateway_smoke's stub configs never subscribe a camera).  Returns
    ``(enabled, topic, max_age_s, jpeg_quality, min_interval_s)``.  The
    defaults mirror VisionConfig
    (False / '/camera/color/image_raw' / 2.0 / 80 / 1.0).
    """
    vision = getattr(cfg, "vision", None)
    enabled = bool(getattr(vision, "enabled", False)
                   if vision is not None else False)
    topic = (
        getattr(vision, "topic", VISION_TOPIC_DEFAULT)
        if vision is not None else VISION_TOPIC_DEFAULT)
    try:
        max_age_s = float(getattr(vision, "max_age_s", VISION_MAX_AGE_S_DEFAULT)
                          or VISION_MAX_AGE_S_DEFAULT)
    except (TypeError, ValueError):
        max_age_s = VISION_MAX_AGE_S_DEFAULT
    try:
        jpeg_quality = int(
            getattr(vision, "jpeg_quality", VISION_JPEG_QUALITY_DEFAULT)
            or VISION_JPEG_QUALITY_DEFAULT)
    except (TypeError, ValueError):
        jpeg_quality = VISION_JPEG_QUALITY_DEFAULT
    try:
        min_interval_s = float(
            getattr(vision, "min_interval_s", VISION_MIN_INTERVAL_S_DEFAULT)
            or VISION_MIN_INTERVAL_S_DEFAULT)
    except (TypeError, ValueError):
        min_interval_s = VISION_MIN_INTERVAL_S_DEFAULT
    return enabled, topic, max_age_s, jpeg_quality, min_interval_s


def gateway_from_config(cfg: Any, *, ros: bool = False) -> "GatewayNode":
    """Build the gateway from a loaded config — the CLI wiring.

    Consumes ``skills.writer_sock`` / ``skills.grip_sock`` /
    ``skills.engine_cmd`` and ``gateway.port`` via
    :func:`resolve_skill_endpoints`, plus the runtime embodiment values
    (pick_min_z_m / silence_means_empty / frame / min_arm_dur_s) and the
    low-workspace descent lane's switch + law inputs (plan §8) via
    :func:`resolve_embodiment_config`.  ``ros=False`` returns the
    rclpy-free core (:class:`GatewayNode`) so laptop tests can construct
    the wiring with a stub transport; ``ros=True`` is the rclpy-only
    branch (Stuart CLI) that builds the :class:`_RosGateway` adapter with
    the config values as its ROS-parameter defaults.
    """
    writer_sock, grip_sock, engine_cmd, _tools_port = resolve_skill_endpoints(cfg)
    pick_min_z_m, silence_means_empty, frame, min_arm_dur_s, descent = \
        resolve_embodiment_config(cfg, with_descent=True)
    # low-workspace descent lane: the switch plus the law inputs the node
    # hands to the descent law. these are the config's own values; the
    # master switch is deliberately not a ROS parameter, so arming a motion
    # lane always shows up in the config file. lane_operation_budget_s stays
    # None when the key is absent, and None means the lane's own
    # LANE_OPERATION_BUDGET_S rather than a gateway default.
    (low_workspace_lane, height_stand_m, height_min_m, hold_tol_m,
     descend_confirm_timeout_s, lane_operation_budget_s) = descent
    if ros:
        import rclpy  # noqa: F401 — lazy; the rclpy-only branch

        return _RosGateway(config=cfg)
    vision_enabled, vision_topic, vision_max_age_s, vision_jpeg_quality, \
        vision_min_interval_s = resolve_vision_config(cfg)
    # point-cloud switches: {} when all are off, so the call below is
    # exactly the plain one.
    pc_kw = _pc_switch_kwargs(resolve_pc_switches(cfg))
    return GatewayNode(
        writer_sock=writer_sock, grip_sock=grip_sock, engine_cmd=engine_cmd,
        engine_sock=resolve_engine_sock(cfg),
        pick_min_z_m=pick_min_z_m, silence_means_empty=silence_means_empty,
        detections_frame=frame, min_arm_dur_s=min_arm_dur_s,
        vision_enabled=vision_enabled, vision_topic=vision_topic,
        vision_max_age_s=vision_max_age_s,
        vision_jpeg_quality=vision_jpeg_quality,
        vision_min_interval_s=vision_min_interval_s,
        low_workspace_lane=low_workspace_lane,
        height_stand_m=height_stand_m,
        height_min_m=height_min_m,
        hold_tol_m=hold_tol_m,
        descend_confirm_timeout_s=descend_confirm_timeout_s,
        operation_budget_s=lane_operation_budget_s,
        pre_descent_step=_pre_descent_step_flag(
            getattr(getattr(cfg, "embodiment", None), "envelope", None)),
        **pc_kw
    )


def _send_unix_verb(sock_path: str, line: str, timeout_s: float = 15.0,
                    skip_ready_banner: bool = False) -> str:
    """One line in -> one reply line over a unix stream socket (stdlib).

    The skill daemons (merged writer, grip server) speak a strict
    line-protocol: send a verb line, read exactly one reply line. A fresh
    connection per request is safe (both servers are multi-client).
    Raises OSError / ConnectionError on any failure — never guesses a reply.

    ``skip_ready_banner`` is REQUIRED by the pick/place engine and must
    stay opt-in.  The engine sends ``READY_BANNER`` (b"READY\\n") on EVERY
    connection BEFORE it answers anything
    (skills/arm/tabletop_pick_place_engine.py:76,287 — "Serve ONE client:
    READY banner, then one line in -> one line out"), so a plain one-line
    read returns the BANNER, not the verdict.

    MEASURED 2026-10-01, found by code reading and confirmed by the
    engine's own test client, which also consumes the banner first.  The
    cancel itself still WORKED — the verb line is sent either way and the
    engine sets the flag — but ``cancel_engine`` read back "READY", so
    ``_do_cancel`` reported ``engine=unavailable`` for a cancel that had
    in fact been accepted.  A false negative on the one verb whose result
    the operator most needs to trust.

    THIS IS THE IPOPT BANNER AGAIN, one layer down: a daemon that writes a
    greeting onto a stream another process reads one-line-at-a-time
    desynchronises that reader.  Every future caller of a bannered socket
    needs this flag; a caller of a bannerless one must NOT pass it, or it
    would block for a second line that never comes.
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout_s)
        sock.connect(sock_path)
        sock.sendall((line + "\n").encode("utf-8"))
        fh = sock.makefile("r", encoding="utf-8")
        reply = fh.readline()
        if skip_ready_banner and reply.rstrip("\n") == "READY":
            reply = fh.readline()
    finally:
        sock.close()
    if not reply:
        raise ConnectionError(
            "empty reply from %s for %r (daemon closed the connection)"
            % (sock_path, line))
    return reply.rstrip("\n")


#: greppable tag on every orphan line the bridge throws away: a
#: complete stdout line that would otherwise come back as the next
#: command's answer. a literal on purpose, since a tag built from other
#: constants cannot be grepped for.
ENGINE_ORPHAN_TAG = "[engine-orphan]"

#: cap on orphan lines discarded per send. a runaway stdout must not
#: be silently swallowed: past this bound the drain stops and the full
#: count sitting in the buffer is reported, so the runaway is visible. the
#: rest stay buffered for the next send.
ENGINE_ORPHAN_DRAIN_MAX = 8


class _LineReader:
    """Non-blocking line reader for a binary stream (select + own buffer).

    Reads the underlying fd directly so no buffered reader can swallow
    bytes between the READY handshake and the first command reply.
    """

    def __init__(self, stream):
        self._stream = stream
        self._buf = b""

    def readline(self, timeout_s: float) -> Optional[str]:
        deadline = time.time() + timeout_s
        while b"\n" not in self._buf:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([self._stream], [], [], remaining)
            if not ready:
                return None  # read deadline passed
            chunk = os.read(self._stream.fileno(), 4096)
            if not chunk:
                return None  # EOF: the engine process exited
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", "replace")

    def drain_complete_lines(self, limit: int):
        """Remove and return up to ``limit`` COMPLETE lines from the buffer.

        NEVER reads the stream and NEVER blocks: this is a pure look at bytes
        a previous ``readline`` already pulled in.  Returns
        ``(drained, remaining)`` where ``remaining`` counts the complete lines
        still buffered AFTER the drain, so a runaway stream is surfaced
        instead of silently swallowed.

        WHY THIS EXISTS (run18, 2026-10-03).  ``readline`` returns ONE line
        and keeps the rest — correct for a well-behaved engine,
        catastrophic when the engine writes TWO lines for ONE command
        (``pick_engine_v3.py`` ``ERROR:hand not verified open`` then
        ``ERROR:grasp_failed``).  The leftover was returned as the NEXT
        command's reply: ``run18/gateway.log:389`` shows
        ``engine ['home'] -> reply=ERROR:grasp_failed`` while
        ``run18/pick.log:448-450`` proves home really moved
        (``max_delta=0.940 rad``).  Engine and gateway ran one command out of
        phase, permanently.  Same defect class as the run10 Ipopt banner
        (``_is_engine_reply``, :440-471): the engine's stdout is a REPLY
        stream, and a line in it that is not the answer to the command in
        flight must never be read as one.
        """
        drained = []
        while len(drained) < limit and b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            drained.append(line.decode("utf-8", "replace"))
        remaining = self._buf.count(b"\n")
        return drained, remaining


class EngineTransportError(ConnectionError):
    """A POST-SEND engine transport failure — the command may have executed.

    Raised ONLY after the command bytes reached the transport (or the transport
    died in the act of sending them): the pipe/socket readline death-or-timeout
    at :762 / :781, and the socket sendall failure at :773.  The engine's
    execution state is UNKNOWN in all three, which is exactly why the place
    path must not treat them like an engine-REPORTED error: see the RESTORE
    GATE in _do_place.  A subclass of ConnectionError so existing callers that
    catch ConnectionError keep working; engine_verb catches it FIRST and
    re-labels the reply with ENGINE_TRANSPORT_ERROR_PREFIX.
    """


class EngineBridge:
    """Connect-first, spawn-fallback bridge to the pick/place engine.

    TWO TRANSPORTS, in preference order (2026-09-13):

    1. SOCKET — the engine (skills/arm/tabletop_pick_place_engine.py) listens
       on ``ENGINE_SOCK_DEFAULT`` and every engine verb goes down that
       connection. This is the structural fix for the double-engine defect:
       the pick lane's resident engine is REUSED, so a place continues from
       whatever pose the pick left behind. A connect never boots an engine,
       so nothing drives the arms to NEUTRAL behind the operator's back.
    2. PIPE — only when nothing is listening. The engine is spawned exactly as
       before (stdin/stdout, READY handshake, pgrep double-spawn guard). This
       is a genuine cold start: there is no resident state to preserve, so the
       boot move to NEUTRAL is correct here.

    Either way the engine prints READY then answers exactly one line
    (DONE / ERROR:<msg>) per command; diagnostics go to stderr. That contract
    is the ENGINE's to keep, and the engine-side ``_reply_sent`` guard is what
    keeps it (skills/arm/pick_engine_v3.py): every failure and refusal branch
    writes its one line under a command-scoped flag, so no path can fall
    through and write a second.

    The gateway's half is a PARTIAL catcher, NOT an enforcement — do not read
    it as a guarantee. ``_drain_orphans_unlocked`` discards (and logs as
    ``[engine-orphan]``) any complete line ALREADY BUFFERED when the next
    command is written. That closes the case where the stray line had reached
    this bridge's own buffer before the previous ``readline`` returned. It
    does NOT close the case where the stray line is still in the KERNEL at
    that moment: the buffer is empty, nothing is drained, and the next command
    still receives the stale line as its answer — silently, with no
    ``[engine-orphan]`` line. Only the engine-side guard covers that window.

    One further residual, on the pipe transport only: after a read timeout the
    pipe path keeps ``self._reader`` (with its buffer and the kernel intact),
    so a very late reply can still be paired with the following command —
    whereas the socket path drops its reader and so cannot. The asymmetry is
    deliberate today and is the reason the engine-side single-reply invariant
    is the load-bearing half of the fix.

    The engine is NON-owning: it commands the writer over the writer socket,
    so it cannot trip the motor-owner rule.
    """

    READY_TIMEOUT_S = 90.0
    REPLY_TIMEOUT_S = 60.0
    PICK_REPLY_TIMEOUT_S = 300.0  # pick = IK plus a multi-point slew, so slow
    #: a live engine answers a connect straight away; a hanging connect
    #: means a wedged listener, not a slow boot, since boot delays show
    #: up on the READY read instead.
    CONNECT_TIMEOUT_S = 2.0

    def __init__(self, cmd: str, sock_path: str = ENGINE_SOCK_DEFAULT):
        self._cmd = cmd
        self._sock_path = sock_path
        self._proc = None
        self._reader = None
        self._sock = None           # live socket to an already running engine
        self._sock_reader = None
        self._lock = threading.Lock()
        self._stderr_tail: List[str] = []
        #: the daemon thread draining the spawned engine's stderr. keep the
        #: handle so a failure path can join it before reading the tail;
        #: without the join the tail is read microseconds after the thread
        #: starts and always comes back empty.
        self._stderr_thread: Optional[threading.Thread] = None

    def _stderr_report(self) -> str:
        """Exit status + stderr tail of the spawned engine, collected SYNCHRONOUSLY.

        WHY THIS EXISTS (2026-09-21, measured on Stuart).  ``_drain_stderr``
        runs on a daemon thread and the handshake below raises the moment
        ``readline`` returns None — which is the moment stdout hits EOF, i.e.
        the moment a DYING engine exits.  The thread has then had
        microseconds, so ``self._stderr_tail`` is still empty and the operator
        is told ``pick engine did not print READY (stderr tail: )``.  The one
        diagnostic that could name the cause was guaranteed not to have run
        yet: the failure that prompted this fix reported an empty tail for a
        spawn that had in fact died before READY, and a full multi-agent
        forensic workflow could not recover the reason from it.

        The drainer ends at EOF, so joining it returns as soon as the process
        is gone.  The exit status is included because a rc the caller never
        saw was the other half of the blind spot: a spawn that died with a
        traceback and rc=1 and a spawn that never started at all (rc=127) read
        identically before this.
        """
        proc, thread = self._proc, self._stderr_thread
        rc = proc.poll() if proc is not None else None
        # join only when the process is already gone: stdout and stderr have
        # hit EOF by then and the join costs microseconds. on a reply timeout
        # against a live process there is no EOF, so joining would burn the
        # whole budget for nothing. report the tail as it stands.
        if rc is not None and thread is not None:
            try:
                thread.join(timeout=2.0)
            except RuntimeError:
                pass  # the thread never started
        tail = " | ".join(self._stderr_tail[-3:])
        return "rc=%s stderr tail: %s" % (rc, tail or "<empty>")

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def alive(self) -> bool:
        if self._sock is not None:
            return True
        proc = self._proc
        return proc is not None and proc.poll() is None

    def via_socket(self) -> bool:
        """True when the bridge is attached to a resident engine.

        Diagnostics only — it lets an operator confirm the place is reusing the
        pick's engine rather than having spawned a second one.
        """
        return self._sock is not None

    def start(self) -> None:
        """Attach to a resident engine, else spawn one. Raises on failure.

        CONNECT-FIRST is the point of this method. Spawning a fresh engine
        makes it boot to NEUTRAL, which is exactly the arm-drop the operator
        sees between a pick and a place; attaching to the engine that did the
        pick preserves the carry/tuck pose.
        """
        with self._lock:
            if self.alive():
                return
            if self._try_connect_unlocked():
                return
            self._spawn_unlocked()

    def _try_connect_unlocked(self) -> bool:
        """Attach to a resident engine's socket. False when none is listening.

        Never raises: "nothing is listening" is the ordinary cold-start case
        and must fall through to the spawn path.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(self.CONNECT_TIMEOUT_S)
            sock.connect(self._sock_path)
            # the engine sends READY once it reaches its command loop, so this
            # read doubles as the readiness wait: a connect during boot blocks here
            # instead of racing the init move. a pipe spawn waits on the same
            # handshake.
            reader = _LineReader(sock.makefile("rb"))
            line = reader.readline(self.READY_TIMEOUT_S)
            if line is None or line.strip() != "READY":
                sock.close()
                return False
            self._sock = sock
            self._sock_reader = reader
            return True
        except OSError:
            try:
                sock.close()
            except OSError:
                pass
            return False

    def _drop_socket_unlocked(self) -> None:
        """Forget a dead/failed socket so the next call can re-attach or spawn.

        Also used on a reply TIMEOUT: the engine may still be executing, and a
        late reply must never be mistaken for the next command's answer.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._sock_reader = None

    def _spawn_unlocked(self) -> None:
        try:
            probe = subprocess.run(
                ["pgrep", "-f", "[t]abletop_pick_place_engine.py"
                             "|[p]ick_engine_v[23].py"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if probe.returncode == 0:
                raise RuntimeError(
                    "pick engine already alive — refusing double spawn "
                    "(no socket at %s, so it cannot be reused)"
                    % self._sock_path)
        except FileNotFoundError:
            pass  # no pgrep available; the launcher guard covers this

        args = self._cmd.split()
        cwd = None
        if len(args) >= 2 and os.path.isfile(args[-1]):
            cwd = os.path.dirname(os.path.abspath(args[-1]))
        self._proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=cwd)
        self._reader = _LineReader(self._proc.stdout)
        self._stderr_tail = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        line = self._reader.readline(self.READY_TIMEOUT_S)
        if line is None:
            # _stderr_report() joins the drainer first; reading the attribute
            # directly here is always empty because this line runs the instant a
            # dying engine's stdout hits EOF.
            report = self._stderr_report()
            self._kill()
            raise RuntimeError(
                "pick engine did not print READY (%s)" % report)
        if line.strip() != "READY":
            report = self._stderr_report()
            self._kill()
            raise RuntimeError(
                "unexpected engine handshake line %r (%s)"
                % (line.strip(), report))

    def _drain_stderr(self) -> None:
        try:
            while True:
                chunk = os.read(self._proc.stderr.fileno(), 4096)
                if not chunk:
                    break
                self._stderr_tail.append(chunk.decode("utf-8", "replace"))
                if len(self._stderr_tail) > 50:
                    del self._stderr_tail[:25]
        except OSError:
            pass

    def _kill(self) -> None:
        try:
            if self._proc is not None:
                self._proc.kill()
        except OSError:
            pass
        self._proc = None
        self._reader = None

    # ── protocol (send/receive) ────────────────────────────────────────────────

    def _drain_orphans_unlocked(self, pending_line: str) -> int:
        """Discard complete lines already buffered from a PREVIOUS command.

        Called immediately BEFORE a command is written (BOTH transports), so a
        second line the engine wrote for the command that just finished can
        never be returned as this command's reply.  Each orphan is logged to
        stderr with the pending verb; drained lines are DISCARDED, never
        returned.  At most ``ENGINE_ORPHAN_DRAIN_MAX`` lines go per send — the
        rest stay buffered, and the FULL count present is reported, so a
        runaway stream is loud rather than slowly swallowed.

        Caller holds ``self._lock`` (same contract as the send paths).
        """
        reader = self._sock_reader if self._sock is not None else self._reader
        if reader is None:
            return 0
        # draining is an optional capability of a reader, not part of its
        # contract; a minimal double may implement only `readline`. it must not
        # raise, since engine_verb wraps the send in a blanket except and an
        # AttributeError here would become a generic engine error and destroy
        # the post-send EngineTransportError classification. a reader with no
        # buffer has nothing to drain, so skipping it is exact.
        drain = getattr(reader, "drain_complete_lines", None)
        if drain is None:
            return 0
        verb = (pending_line or "").split(" ", 1)[0]
        drained, remaining = drain(ENGINE_ORPHAN_DRAIN_MAX)
        for text in drained:
            sys.stderr.write(
                "%s pending=%s orphan=%r\n"
                % (ENGINE_ORPHAN_TAG, verb, text))
        if remaining:
            sys.stderr.write(
                "%s pending=%s OVERFLOW: %d complete line(s) present, "
                "drained %d, %d still buffered — the engine is not writing "
                "one reply per command\n"
                % (ENGINE_ORPHAN_TAG, verb, len(drained) + remaining,
                   len(drained), remaining))
        if drained or remaining:
            sys.stderr.flush()
        return len(drained)

    def send(self, line: str, timeout_s: Optional[float] = None) -> str:
        """Send one engine command, return its one-line reply. Raises on death."""
        if timeout_s is None:
            if line.startswith(("pick ", "pick_left ", "place ",
                                "place_left ")):
                timeout_s = self.PICK_REPLY_TIMEOUT_S
            else:
                timeout_s = self.REPLY_TIMEOUT_S
        with self._lock:
            if not self.alive():
                raise ConnectionError("pick engine not running")
            if self._sock is not None:
                return self._send_socket_unlocked(line, timeout_s)
            self._drain_orphans_unlocked(line)
            self._proc.stdin.write((line + "\n").encode("utf-8"))
            self._proc.stdin.flush()
            reply = self._reader.readline(timeout_s)
            if reply is None:
                # post-send: the command already went down the pipe, so the engine may
                # well have run it. raise EngineTransportError, never a plain
                # ConnectionError, so the place path does not restore a hold for a
                # command whose execution state is unknown.
                raise EngineTransportError(
                    "pick engine timed out / died on %r (%s)"
                    % (line, self._stderr_report()))
            return reply.strip()

    def _send_socket_unlocked(self, line: str, timeout_s: float) -> str:
        """One command down the resident-engine socket. Caller holds the lock."""
        self._drain_orphans_unlocked(line)
        try:
            self._sock.sendall((line + "\n").encode("utf-8"))
        except OSError as exc:
            self._drop_socket_unlocked()
            # post-send, and the one a substring test misses: sendall can die
            # after a partial write, so the engine may have received a truncated
            # command and be executing it. its text never says 'timed out', so
            # classify by type here and by prefix at engine_verb.
            raise EngineTransportError(
                "pick engine socket send failed on %r: %s" % (line, exc))
        reply = self._sock_reader.readline(timeout_s)
        if reply is None:
            # either the engine died or it outran the deadline. drop the
            # connection either way: a reply arriving after the wait ended must
            # never be read as the next command's answer.
            self._drop_socket_unlocked()
            # post-send
            raise EngineTransportError(
                "pick engine timed out / dropped the connection on %r "
                "(socket %s)" % (line, self._sock_path))
        return reply.strip()

    def quit(self) -> None:
        """Stop the engine — ONLY if this bridge spawned it.

        A resident engine is shared: the pick lane (pick_node) is usually its
        parent, and quitting it out from under that parent would be a
        cross-lane kill. Attached-to-someone-else's-engine therefore just
        drops our connection and leaves the process running.
        """
        with self._lock:
            if self._sock is not None:
                self._drop_socket_unlocked()
                return
            if self.alive():
                try:
                    self._proc.stdin.write(b"quit\n")
                    self._proc.stdin.flush()
                except OSError:
                    pass
                try:
                    self._proc.wait(timeout=10.0)
                except (subprocess.TimeoutExpired, OSError):
                    self._kill()


# ══════════════════════════════════════════════════════════════════════════
# arm safety, stale target, detections liveness.
#
# `hold_rc off` does not release the arm: it is only the RC-stick
# interlock, and past that the writer keeps PD-tracking its last `arm`
# target whatever the flag says. a stalled arm heats up until it is
# moved, and only slewing it to READY ends the stall. so every release
# below is `hold_rc off` plus a park, and replies name both (`hold_rc=`
# for the interlock, `arm=`/`arm_after=` for the arm). no token claims
# hold_rc off moved the arm.
#
# the stop release, post-task sweep, stale-target gate and re-observe
# all hang off the _b5_on master switch. it is false in the ROS-free
# core, and _RosGateway sets it from the `arm_safety` ROS param
# (default true). the idle watchdog has its own `arm_watchdog` param,
# also default true. neither is a config key.
# ══════════════════════════════════════════════════════════════════════════

#: the engine's own safe arm poses, copied numerically and checked against
#: the engine source by a test, so an engine edit turns that check red by
#: design. each entry is (LEFT7, RIGHT7) in the writer's q order
#: (st['q'][15:29], the same order READY_POSE uses).
SAFE_ARM_POSES: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "READY": ((0.40, 0.30, 0.50, -0.80, 0.00, 0.00, -0.01),
              (0.40, -0.30, -0.50, -0.80, 0.00, 0.00, 0.01)),
    "CARRY_TUCK": ((0.50, 0.05, 0.00, 0.00, -1.50, 0.00, -1.00),
                   (0.50, -0.05, 0.00, 0.00, 1.50, 0.00, 1.00)),
    "NEUTRAL": ((0.25522, 0.27104, -0.08654, 0.83254, -0.00101, 0.01032,
                 -0.00063),
                (0.26688, -0.28410, 0.09247, 0.80183, -0.00034, 0.00915,
                 0.00730)),
    "HOME": ((0.00, 0.30, 0.00, 1.50, -1.5708, 0.00, 0.00),
             (0.00, -0.30, 0.00, 1.50, 1.5708, 0.00, 0.00)),
}
#: per-joint (L-inf) tolerance the safe-pose test uses, derived from a
#: left-arm settle error of 0.264 rad plus margin. a stalled right arm
#: sat 0.719 rad from its nearest pose half and once 0.987 rad, while
#: the four poses are at least 1.499 rad apart.
SAFE_POSE_TOL_RAD = 0.35
#: if the per-arm z-floor clamp leaves an arm on the straight
#: joint-space segment joining these poses (within SAFE_POSE_TOL_RAD,
#: L-inf), it is safe for the post-task sweep only: along that segment
#: the clamp stops an arm at the table plane.
SAFE_SEGMENT_POSES = ("READY", "CARRY_TUCK")
#: how long (s) to wait for the engine's `park` reply, which is
#: one-shot with no banner.
PARK_REPLY_TIMEOUT_S = 20.0
#: reply bound (s) for the writer-direct fallback. the writer answers
#: `arm` when the slew starts, so this only has to cover a wedged
#: socket.
PARK_WRITER_TIMEOUT_S = 5.0
#: once a park route answers DONE, how long (s) `q` may take to read a
#: safe pose before the park counts as failed. the fallback READY slew
#: is 5.000 s plus 2 s of margin.
PARK_SETTLE_S = 7.0
PARK_SETTLE_POLL_S = 0.25
#: how long (s) a synchronous park may hold its caller.
PARK_JOIN_S = 12.0
#: the writer-direct fallback uses the verbatim 30-token READY line
#: the writer accepts: `arm <14 q> <dur> <14 tau>`. the hands are
#: untouched.
WRITER_READY_ARM_LINE = (
    "arm 0.400000 0.300000 0.500000 -0.800000 0.000000 0.000000 -0.010000 "
    "0.400000 -0.300000 -0.500000 -0.800000 0.000000 0.000000 0.010000 "
    "5.000 -0.997557 2.236980 1.265035 -2.648575 -0.032041 -0.913687 "
    "0.351945 -0.997557 -2.236980 -1.265035 -2.648575 0.032041 -0.913687 "
    "-0.351945")
#: how long (s) a post-cancel pick waits for a detection packet newer
#: than the cancel and park and matching the commanded target.
FRESH_DET_WAIT_S = 3.0
#: per-axis tolerance (m) between the commanded point and a fresh
#: detection. a bottle's static jitter over repeated observes stays
#: within 0.012 m; a frozen re-pick sat off by dz=+0.103 m.
STALE_TARGET_TOL_M = 0.04
#: re-observe budget (s) before verifying a pick on a silent channel.
PICK_DET_REOBSERVE_S = 4.0
#: poll period (s) for the stale-target gate and the re-observe.
B5_POLL_S = 0.1
#: read bound (s) for the writer's `q`. the stale-target gate reads q
#: once outside its PARK_JOIN_S park phase, so the timeout ladder has to
#: count it.
ARM_Q_READ_TIMEOUT_S = 1.0
#: period (s) of the idle watchdog.
WATCHDOG_PERIOD_S = 1.0
WATCHDOG_IDLE_GRACE_S = 8.0
WATCHDOG_TMAX_C = 90
WATCHDOG_HOT_HOLD_S = 5.0
WATCHDOG_RC_HOLD_S = 8.0
WATCHDOG_COOLDOWN_S = 60.0
WATCHDOG_MAX_ATTEMPTS = 3
WATCHDOG_LOG_MAX_AGE_S = 3.0
WATCHDOG_LOG_TAIL_BYTES = 8192
#: ceiling (s) on one watchdog park: reply, settle and margin.
WATCHDOG_PARK_JOIN_S = PARK_REPLY_TIMEOUT_S + PARK_SETTLE_S + 5.0
#: default path of the writer log the watchdog tails.
WRITER_LOG_PATH_DEFAULT = "/tmp/pick_walk_logs/writer.log"
#: the writer prints a status line once a second, e.g.
#: [t=1990.0] balance cmd=(+0.00,+0.00,+0.00) arm=RAW mode=6 Tmax=125C
#: ... hold=1 auto=0 ..., plus a `walk` variant and a padded clock in the
#: snapshot log. the writer's `status` reply carries no Tmax, so this log
#: line is the only temperature source that needs no writer change.
WRITER_STATUS_LINE_RE = re.compile(
    r"\[t=\s*([\d.]+)\].*?arm=(\S+) mode=(\d+) Tmax=(\d+)C.*?hold=(\d)")
#: verbs the arm-task in-flight counter counts.
ARM_TASK_VERBS = frozenset((
    "pick", "pick_left", "place", "engine", "writer", "open_hand",
    "close_hand", "cmd_set", "navigate", "turn_around",
    "return_home"))                 # the hand that carries the fetched object


def _arm_pose_deviation(q14: Any) -> Optional[float]:
    """Worst arm's L-infinity distance to its NEAREST safe-pose half (rad).

    None for malformed input (not 14 finite floats).
    """
    try:
        vals = [float(v) for v in q14]
    except (TypeError, ValueError):
        return None
    if len(vals) != 14 or not all(math.isfinite(v) for v in vals):
        return None
    worst = 0.0
    for half, arm in ((0, vals[:7]), (1, vals[7:])):
        nearest = min(max(abs(a - b) for a, b in zip(arm, poses[half]))
                      for poses in SAFE_ARM_POSES.values())
        worst = max(worst, nearest)
    return worst


def _seg_linf(arm: Any, a: Any, b: Any) -> float:
    """B8g G4: L-infinity distance (rad) from a 7-joint arm to the straight
    joint-space segment a -> b (t in [0, 1]).

    Exact, not sampled: max_j |arm_j - a_j - t*(b_j - a_j)| is convex and
    piecewise linear in t, so its minimum sits at t=0, t=1 or where two joints'
    lines cross: t = e_i/d_i, (e_i - e_j)/(d_i - d_j) or (e_i + e_j)/(d_i + d_j).
    """
    e = [x - y for x, y in zip(arm, a)]
    d = [y - x for x, y in zip(a, b)]
    ts = {0.0, 1.0}
    for i in range(len(e)):
        if d[i]:
            ts.add(e[i] / d[i])
        for j in range(i + 1, len(e)):
            if d[i] != d[j]:
                ts.add((e[i] - e[j]) / (d[i] - d[j]))
            if d[i] + d[j]:
                ts.add((e[i] + e[j]) / (d[i] + d[j]))
    return min(max(abs(ei - t * di) for ei, di in zip(e, d))
               for t in ts if 0.0 <= t <= 1.0)


def _arm_safe_pose(q14: Any, segment: bool = False) -> Optional[str]:
    """Name of the safe pose BOTH arms sit at, or None if either arm is out.

    Per arm: the 7-vector must lie within SAFE_POSE_TOL_RAD (per joint) of
    SOME pose's half for that arm.  ``segment=True`` (B8g G4, the post-task
    arm sweep ONLY) also accepts an arm within SAFE_POSE_TOL_RAD (L-inf) of
    the straight READY -> CARRY_TUCK segment for that arm (named
    ``READY~CARRY_TUCK``).  The default is the box-only classifier: a park
    verification, the cancel gate and the watchdog must not take a slew in
    transit along that segment for a settled pose.  Both halves at the same
    pose -> its name;
    different poses -> ``<LEFT>+<RIGHT>``.  Malformed input -> None (callers
    handle an unreadable ``q`` BEFORE classifying it, and never move on one).
    """
    try:
        vals = [float(v) for v in q14]
    except (TypeError, ValueError):
        return None
    if len(vals) != 14 or not all(math.isfinite(v) for v in vals):
        return None
    names = []
    for half, arm in ((0, vals[:7]), (1, vals[7:])):
        hit = None
        for name, poses in SAFE_ARM_POSES.items():
            ref = poses[half]
            if max(abs(a - b) for a, b in zip(arm, ref)) <= SAFE_POSE_TOL_RAD:
                hit = name
                break
        if hit is None and segment:
            # an arm the z-floor clamp left on the READY -> CARRY_TUCK joint
            # segment counts as safe for the post-task sweep, so it is not parked.
            seg_a, seg_b = (SAFE_ARM_POSES[n][half] for n in SAFE_SEGMENT_POSES)
            if _seg_linf(arm, seg_a, seg_b) <= SAFE_POSE_TOL_RAD:
                hit = "~".join(SAFE_SEGMENT_POSES)
        if hit is None:
            return None
        names.append(hit)
    return names[0] if names[0] == names[1] else "%s+%s" % (names[0], names[1])


def _b5_tok_text(text: Any, limit: int = 60) -> str:
    """Make arbitrary text safe inside a ``name=value(<text>)`` reply token.

    Defangs ``ERROR:`` (an embedded literal would flip the whole envelope to
    ok:false — see _defang_error_token) and replaces the reply's own
    delimiters (``;``, ``|``, parentheses) and newlines, so a token can never
    split the reply grammar the brain parses.
    """
    out = _defang_error_token(str(text))
    for ch in "\r\n":
        out = out.replace(ch, " ")
    for ch in ";|()":
        out = out.replace(ch, "_")
    return out.strip()[:limit]


def _insert_before_engine_reply(reply: str, tok: str) -> str:
    """Insert ``tok`` (``; name=value``) before ``; engine_reply=``, else append.

    brain/verifier.py ``_ENGINE_REPLY_RE = engine_reply=([^\\r\\n]*)`` is
    GREEDY to the end of the line, so every new token must precede it or it
    would be read as part of the engine's own words.
    """
    idx = reply.find("; engine_reply=")
    if idx < 0:
        return reply + tok
    return reply[:idx] + tok + reply[idx:]


#: the one post-send transport failure where the sweep goes
#: writer-direct: a failed socket send, whose text is
#: EngineBridge._send_socket_unlocked's own. the other post-send form,
#: timed out or dropped the connection, cannot tell a dead engine from
#: one still executing, so that sweep goes to the engine's queued `park`
#: first: a live engine answers after its command, and a dead one
#: refuses the connect so the park falls back to the writer.
ENGINE_SEND_FAILED_PREFIX = (ENGINE_TRANSPORT_ERROR_PREFIX
                             + "pick engine socket send failed")

#: label match: the hint is the brain's raw pick `label`, and the
#: taught workflow hands the model the scene-graph ref, so the brief
#: prints 'bottle_1: bottle x=...' and the model calls
#: pick(label='bottle_1'). perception rows carry the class label with
#: ref None, so a raw equality test would refuse every ref-form re-pick.
#: read the hint at class level instead: lowercased, a trailing _<n>
#: suffix stripped, '_' read as a space (the scene graph keys
#: 'cell phone' as 'cell_phone_1'), plus the brain's alias map
#: ('me' -> 'person').
_B5_REF_SUFFIX_RE = re.compile(r"_\d+$")


def _b5_norm_label(text: Any) -> str:
    """Lowercase, '_' read as a space, whitespace collapsed."""
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def _b5_resolve_alias(text: str) -> str:
    """brain.known_labels.resolve_alias, or the identity without it."""
    try:
        from brain.known_labels import resolve_alias  # lazy import keeps the module stdlib-only
        out = resolve_alias(text)
    except Exception:  # noqa: BLE001 — no alias map is no reason to refuse
        return text
    return out if isinstance(out, str) and out else text


#: strip leading colour words off a scene name ('green_cup_2' -> 'cup').
_B7_COLOUR_WORDS = frozenset((
    "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink",
    "white", "black", "grey", "gray", "teal", "turquoise", "aqua", "navy",
    "maroon", "crimson", "scarlet", "violet", "lilac", "magenta", "gold", "lime"))


def _b5_hint_class_keys(hint: Any) -> frozenset:
    """The class keys a pick hint names; empty == no hint (label inert).

    B7: a brain scene name ('green_cup_2') also yields its class ('cup'):
    perception rows never carry the colour prefix (MEASURED 2026-10-06: a
    fresh cup 2 mm from the target was refused as stale_target)."""
    text = str(hint or "").strip().lower()
    if not text:
        return frozenset()
    keys = set()
    forms = [text, _B5_REF_SUFFIX_RE.sub("", text)]
    for form in list(forms):
        words = form.replace("_", " ").split()
        if len(words) > 1 and words[0] in _B7_COLOUR_WORDS:
            forms.append(" ".join(words[1:]))
    for form in forms:
        keys.add(_b5_norm_label(form))
        keys.add(_b5_norm_label(_b5_resolve_alias(form)))
        keys.add(_b5_norm_label(_b5_resolve_alias(form.replace("_", " "))))
    keys.discard("")
    return frozenset(keys)


def _b5_row_matches_hint(det: Any, hint_keys: frozenset) -> bool:
    """Does a detection row name the hinted class?  No hint -> True."""
    if not hint_keys:
        return True
    for attr in ("label", "class_name"):
        val = getattr(det, attr, None)
        if isinstance(val, str) and _b5_norm_label(val) in hint_keys:
            return True
    return False


def _parse_writer_status_lines(text: str) -> List[Tuple[float, str, int, int, int]]:
    """Every writer status line in ``text`` as (t, arm, mode, Tmax, hold)."""
    out = []
    for line in text.splitlines():
        m = WRITER_STATUS_LINE_RE.search(line)
        if m is None:
            continue
        try:
            out.append((float(m.group(1)), m.group(2), int(m.group(3)),
                        int(m.group(4)), int(m.group(5))))
        except ValueError:
            continue
    return out


def _trailing_span(samples: List[Tuple[float, str, int, int, int]],
                   pred) -> float:
    """Writer-clock span of the trailing run of samples satisfying ``pred``.

    0.0 when the newest sample fails ``pred``.  The span is newest.t minus the
    t of the EARLIEST sample of the unbroken trailing run, so "hot for 5 s"
    needs hot samples covering 5 s of the writer's own clock (at 1 Hz: six
    consecutive hot samples) — four hot seconds after a cool one is NOT 5 s.
    """
    if not samples or not pred(samples[-1]):
        return 0.0
    first = samples[-1][0]
    for s in reversed(samples):
        if not pred(s):
            break
        first = s[0]
    return samples[-1][0] - first


class GatewayNode:
    """Standalone gateway: the rclpy node is a thin adapter over this class.

    Splitting the rclpy-free core out keeps every ROS import inside
    ``_RosGateway`` (see below) so laptop tests exercise this class directly.
    """

    def __init__(self, writer_sock: str = WRITER_SOCK_DEFAULT,
                 grip_sock: str = GRIP_SOCK_DEFAULT,
                 engine_cmd: str = "",
                 engine_sock: str = ENGINE_SOCK_DEFAULT,
                 max_stale_s: float = MAX_STALE_S_DEFAULT,
                 pick_min_z_m: float = PICK_MIN_Z_M,
                 min_arm_dur_s: float = MIN_ARM_DUR_S_DEFAULT,
                 silence_means_empty: bool = True,
                 detections_frame: str = DETECTIONS_FRAME_DEFAULT,
                 vision_enabled: bool = False,
                 vision_topic: str = VISION_TOPIC_DEFAULT,
                 vision_max_age_s: float = VISION_MAX_AGE_S_DEFAULT,
                 vision_jpeg_quality: int = VISION_JPEG_QUALITY_DEFAULT,
                 vision_min_interval_s: float = VISION_MIN_INTERVAL_S_DEFAULT,
                 low_workspace_lane: bool = False,
                 height_stand_m: float = HEIGHT_STAND_M_DEFAULT,
                 height_min_m: float = HEIGHT_MIN_M_DEFAULT,
                 hold_tol_m: float = HOLD_TOL_M_DEFAULT,
                 descend_confirm_timeout_s: float = (
                     DESCENT_CONFIRM_TIMEOUT_S_DEFAULT),
                 operation_budget_s: Optional[float] = None,
                 pre_descent_step: bool = False,
                 pc_shadow: bool = False,
                 pc_hold_check: bool = False,
                 pc_place_rim: bool = False):
        from gateway.assemble import (Assembler, CLOSE_CONTACT_WINDOW_S,  # lazy import
                                      GRIP_STATE_MAX_AGE_S)

        self._writer_sock = writer_sock
        self._grip_sock = grip_sock
        self._pick_min_z_m = pick_min_z_m
        #: runtime embodiment: arm-dispatch duration floor (config key
        #: envelope.min_arm_dur_s; the writer clamps at 1.5 s and the larger
        #: wins), the silence-means-empty switch
        #: (perception.silence_means_empty, which feeds the assembler's silence
        #: semantics), and the detections frame name (perception.frame, served
        #: by status and documenting the cmd_set omega sign).
        self._min_arm_dur_s = float(min_arm_dur_s)
        self._silence_means_empty = bool(silence_means_empty)
        self._detections_frame = detections_frame or DETECTIONS_FRAME_DEFAULT
        self._assembler = Assembler(max_stale_s=max_stale_s,
                                    silence_means_empty=self._silence_means_empty)
        self._obs_lock = threading.Lock()
        self._engine_sock = engine_sock or ENGINE_SOCK_DEFAULT
        self._engine = (
            EngineBridge(engine_cmd, self._engine_sock) if engine_cmd else None
        )
        # /place_target lane. the pure half is stdlib-only, rclpy-free and
        # unit-testable. one non-blocking lock turns a concurrent /place_target
        # into an honest refusal rather than a queue: a second place queuing
        # would release into a scene the first changed.

        from gateway import place_target as place_target_law  # lazy, needs package ctx
        self._place_target_law = place_target_law
        self._place_target_lock = threading.Lock()
        # ── low-workspace descent lane ─────────────────────────────────────────
        # the lane is a lazy import in try/except, like the sonic lane's
        # degeneracy guard: a missing or broken module leaves self._low_lane
        # None, so the dispatch guard never fires and every low pick or place
        # degrades to an honest refusal, never a crash or a silent success.
        # only a real bool true arms it.
        try:
            from gateway import descent_lane as descent_lane_law  # lazy import
        except Exception as _exc:  # noqa: BLE001 — degrade, never raise
            descent_lane_law = None
            print("[descent] descent_lane import FAILED (%r) — low-workspace "
                  "lane FORCED OFF" % (_exc,), flush=True)
        # one non-blocking lock for the whole lane skill: a second lane skill
        # queued behind the first would descend from, or retreat out of, a body
        # height the first still uses, so it is refused (DESCENT_BUSY_MSG),
        # never queued.
        self._descent_lock = threading.Lock()
        self._low_lane = None
        self._low_law = None
        # the confirm-timeout knob: the loader parses it into
        # EnvelopeConfig.descend_confirm_timeout_s and it is read here, so the
        # operator-visible key reaches the lane's status poll instead of lying
        # there. threaded into the LaneLaw below.
        self._descent_confirm_timeout_s = float(descend_confirm_timeout_s)
        # the operation budget: the declared per-operation budget the lane's
        # writer-hold guard refuses below, parsed into
        # EnvelopeConfig.lane_operation_budget_s and read here. held as
        # float-or-None and defaulting to None, not a number: None means the lane
        # keeps its own LANE_OPERATION_BUDGET_S, so this node restates no budget.
        # a config, a stub or a caller that never sets it all mean the same; a
        # malformed value degrades to None.
        self._descent_operation_budget_s = _optional_float(operation_budget_s)
        # re-entrancy: the /place_target lane sends its inner `place` through
        # dispatch, and the descent guard also sits in dispatch, so this flag
        # marks that call nested and the guard declines it. restored in the same
        # call's `finally`, never left set: a leaked true would quietly disable
        # the lane for the rest of the process.
        self._descent_nested = False
        self._descent_ctx = (descent_lane_law.DescentContext()
                             if descent_lane_law is not None else None)
        # the optional pre-descent stance step (envelope.pre_descent_step).
        # arms only on a real true and an armed lane, since there is nothing to
        # precede otherwise.
        self._pre_descent_step = (bool(pre_descent_step is True)
                                  and low_workspace_lane is True)
        if descent_lane_law is not None and low_workspace_lane is True:
            self._low_lane = descent_lane_law
            # the law inputs, threaded in; this node restates none of them.
            self._low_law = _lane_law(
                descent_lane_law.LaneLaw, float(pick_min_z_m),
                float(height_stand_m), float(height_min_m),
                float(hold_tol_m), self._descent_confirm_timeout_s,
                self._descent_operation_budget_s)
        # ── point-cloud consumer client ────────────────────────────────────────────
        # absent by default. unless one of embodiment.envelope.pc_shadow /
        # pc_hold_check / pc_place_rim is a real true, self._pc is None and
        # nothing here runs: no import, no publisher, no subscription, and every
        # call site below passes no extra keyword.
        # flipping a switch on builds the client through the same lazy-import and
        # degrade idiom, so a missing or broken client leaves the hooks forced
        # off. any one switch builds it; pc_hold_check makes the lane's hold
        # check decisive (otherwise it only logs); pc_place_rim lets a good rim
        # fit replace the place height (otherwise it is logged only). the ROS
        # publisher and subscription belong to _RosGateway.
        self._pc = None
        self._pc_place_rim = False
        #: the newest measured body pose from the odometry process
        #: (/openbot/body_pose): the client's pose source.
        self._body_pose = None
        self._body_pose_lock = threading.Lock()
        if pc_shadow is True or pc_hold_check is True or pc_place_rim is True:
            try:
                from gateway import pointcloud_client as pc_client_mod  # lazy import
                # the writer verb and every hook resolve per call, never
                # captured at construction, so a stubbed `writer_verb` or
                # `_publish_pc_arm` sees exactly what the client issues.
                self._pc = pc_client_mod.PcClient(
                    publish_arm=lambda text: self._publish_pc_arm(text),
                    clock_wall=time.time, clock_mono=time.monotonic,
                    writer_verb=lambda line: self.writer_verb(line),
                    decisive=bool(pc_hold_check is True),
                    log=lambda message: self._pc_log(message),
                    pose_source=lambda: self._latest_body_pose())
                self._pc_place_rim = bool(pc_place_rim is True)
            except Exception as _exc:  # noqa: BLE001 — degrade, never raise
                self._pc = None
                self._pc_place_rim = False
                print("[pc] pc_client import FAILED (%r) - point-cloud hooks "
                      "FORCED OFF" % (_exc,), file=sys.stderr, flush=True)
        # one persistent verifier instance: the navigate epoch gate has to
        # remember the last confirmed epoch across calls.
        self._verifier = None
        # a per-request target hint for the pick verifier.
        #
        # why: the pick tool's static contract prose carries no target_label=,
        # so the gateway's own verify_pick cannot tell what is being picked
        # and takes the unconfirmable branch, which reads as a
        # false 'I could not confirm the grasp'. the brain fixes this for its
        # own verifier by prepending target_label=, but that hook never runs on
        # this verdict, since _do_pick mints its own through _verify.
        #
        # threading: a thread-local, not an instance attribute, because the
        # server handles connections concurrently and a shared field would leak
        # one connection's target into another's verdict. correct because the
        # pick path is synchronous end to end: handle() -> dispatch() ->
        # _low_lane_pick() -> pick_sequence() -> _do_pick() -> _verify() all run
        # on that connection's thread. if the lane ever hands a pick to a
        # worker, this slot has to be passed explicitly.
        self._req_target_hint = threading.local()
        # /skill_verdict deferral: a lane run sets `defer` on its own thread so
        # the engine verdict _verify mints during the run is not published; the
        # lane then publishes the verdict it returns, after downgrade, once.
        # thread-local for the same reason as the hint above: another
        # connection's non-lane verdict must still publish while a lane run
        # defers.
        self._verdict_tls = threading.local()
        # close_adaptive verdicts, tracked gateway-side (side -> (verdict, ts)).
        # this map is evidence only for the close_hand tool, plus the
        # informational prev_contact annotations on open_hand. a real engine pick
        # closes over its own socket, invisible here, so no serialized verdict
        # can certify a pick; pick grasp evidence is the fresh post-pick
        # 'state [side]' probe.
        self._close_verdicts: Dict[str, Tuple[str, float]] = {}
        self._close_verdict_window_s = CLOSE_CONTACT_WINDOW_S
        # per-side record of the last engine-verified pick's hold (side -> pick
        # ts). recorded when a pick's lane poll comes back up, because the
        # engine's DONE is engine-verified and the hold is real; cleared at the
        # start of any new pick on that side, by an open command for that side,
        # and by a successful drop (both hands open). the verifier's place path
        # reads it as 'prev_pick_verified'.
        self._last_verified_grasp: Dict[str, float] = {}
        #: side -> time.time() of the last DONE place (refusal note only).
        self._last_place_release: Dict[str, float] = {}
        # daemon-lane hold evidence: the newest /pick_state payload as a dict
        # {held, alive, ts, side, state}, or None before the first one. read
        # only through _daemon_held_side(), which applies the freshness gate: a
        # stale record must never authorise a place (PICK_STATE_MAX_AGE_S).
        self._pick_state_hold: Optional[Dict[str, object]] = None
        #: flips true once any /pick_state frame has been folded in. the motion
        #: wait (_await_pick_daemon) must not burn its budget on a lane with no
        #: daemon, and 'never heard from it' is the only honest way to know: a
        #: currently-empty record looks like one between frames.
        self._pick_state_seen = False
        # consumed-once law. the daemon's `held` is its own flag and sticky: set
        # true on a successful daemon-lane pick and cleared only by the daemon's
        # own open or error, and the gateway never tells it about a place, since
        # a release goes to the engine or grip socket rather than
        # /arm_command. after a release it keeps republishing 'held: true' at
        # 5 Hz, undoing a cleared record within one interval (<= 200 ms). this
        # flag makes the clear stick: while set, a daemon 'held: true' is not
        # evidence. released only at a hold epoch end, either the daemon's own
        # held:false or the `busy` state starting its next cycle, because
        # _execute_pick does not touch `held` and two successful picks have no
        # held:false between them; keying off `held` alone would re-refuse every
        # later place.
        self._pick_state_consumed = False
        self._pick_state_topic = PICK_STATE_TOPIC_DEFAULT
        self._pick_state_max_age_s = PICK_STATE_MAX_AGE_S
        # motion-completion wait budgets (_await_pick_daemon), re-read from ROS
        # params during lane setup, like _pick_state_max_age_s.
        self._pick_daemon_wait_s = PICK_DAEMON_WAIT_S
        self._pick_daemon_start_s = PICK_DAEMON_START_S
        # navigate motion-completion wait budgets (_await_walk_reached). instance
        # attributes, like the pick pair, so a pin can shorten the wait without
        # editing source and a ROS param can retune it at runtime.
        self._nav_wait_s = NAV_WAIT_S
        self._nav_start_s = NAV_START_S
        # threshold for the measured held-side tier (_measured_held_side). an
        # instance attribute, not a bare constant, so a test can move it, and an
        # out-of-range value is a boot-time error rather than a quietly
        # different robot.
        self._closed_eff_min = CLOSED_EFF_MIN
        if not (0.0 < self._closed_eff_min < 1.0):
            raise ValueError(
                "closed_eff_min must be in (0, 1), got %r"
                % (self._closed_eff_min,))
        # navigate epoch gate: every /walk_go dispatch makes an epoch
        # (epoch_ts, target_label); phases pile up as (state, label, ts) for
        # ts >= epoch_ts, and the label comes from the walk_status payload, so a
        # REACHED for a different label can never verify. the epoch the verifier
        # confirmed sits inside the one persistent SkillVerifier
        # (_nav_confirmed_epoch).
        self._nav_epoch: Tuple[float, Optional[str]] = (0.0, None)
        self._phases_seen: List[Tuple[str, Optional[str], float]] = []
        self._last_walk_ts = 0.0
        # stop-kills-task: timestamp of the last cancel, stamped in _do_cancel and
        # polled by _await_walk_reached. without it the motion wait cannot tell
        # 'the operator ended this' from 'the walk is still going' and burns its
        # full NAV_WAIT_S. read in-epoch (>= the dispatch's epoch_ts), so a cancel
        # can never abort a navigate that went out after it. 0.0 means no cancel
        # stamped, deliberately not a valid epoch.
        self._cancel_ts = 0.0
        # ── arm safety / stale target / liveness state ────────────────────────────
        # master switch (see the module-level arm safety block): off in the core,
        # armed by _RosGateway from the `arm_safety` ROS param.
        self._b5_on = False
        #: when a cancel or abort path last found or left an arm outside
        #: SAFE_ARM_POSES, and when a park last ended verified at a safe pose. the
        #: watchdog's `disturbed` conjunct and the stale-target gate read them.
        self._arm_disturbed_ts = 0.0
        self._arm_parked_ts = 0.0
        #: when a post-cancel pick last passed the stale-target gate.
        self._arm_gate_cleared_ts = 0.0
        self._arm_safety: Dict[str, Any] = {}
        #: only one park at a time (non-blocking acquire in _park_arm); each park
        #: runs on its own thread, so `_park_thread` always names it.
        self._park_lock = threading.Lock()
        self._park_start_lock = threading.Lock()
        self._park_thread: Optional[threading.Thread] = None
        # unconditional arm-task in-flight counter (in dispatch), separate from
        # the open-vocab `_motion_enter` counter, which only counts when
        # `_open_vocab` is true.
        self._inflight_lock = threading.Lock()
        self._arm_task_inflight = 0
        self._arm_idle_since = time.time()
        self._wd_enabled = False
        self._wd_thread: Optional[threading.Thread] = None
        self._wd_stop = threading.Event()
        self._wd_writer_log_path = WRITER_LOG_PATH_DEFAULT
        self._wd_attempts = 0
        self._wd_last_attempt_ts = 0.0
        self._wd_gave_up = False
        #: the last disturbance the watchdog saw resolved (arm read safe):
        #: otherwise a stale _arm_disturbed_ts would park a later deliberate pose.
        self._wd_disturbed_seen_safe_ts = 0.0
        self._wd_log_state = "missing"
        self._wd_last_sample: Optional[Dict[str, Any]] = None
        # test seams (production uses the real clock and sleep), same as _turn_clock.
        self._b5_clock = time.time
        self._b5_sleep = time.sleep
        # turn_around centring state plus test seams (production uses the real clock
        # and sleep)
        self._turn_active = False
        self._turn_events = []          # raw un-folded (state, label, ts)
        self._turn_pulsed = False
        self._turn_clock = time.time
        self._turn_sleep = time.sleep
        # the container lock (identity survives label changes) and creep
        self._lock_setup()
        # the fetch start point (from odometry) and navigate-hold widening
        self._b8l_setup()
        # ── camera lane (optional, and off unless vision.enabled) ────────
        # the DDS callback does nothing but copy the raw frame buffer into the
        # lock-guarded slot below; JPEG encoding runs on its own daemon thread,
        # started lazily on the first frame, never on the DDS thread. cv2 and numpy
        # are imported lazily inside the encoder, so this core imports cleanly with
        # no OpenCV around.
        # throttled cache (vision.min_interval_s): the worker encodes at most one
        # frame per interval, always the newest; in-between frames are skipped,
        # never queued. one frame per decision turn stays the caller's job; this
        # worker only caps encode cost.
        self._vision_enabled = bool(vision_enabled)
        self._vision_topic = vision_topic or VISION_TOPIC_DEFAULT
        self._vision_max_age_s = float(vision_max_age_s or VISION_MAX_AGE_S_DEFAULT)
        self._vision_jpeg_quality = int(vision_jpeg_quality or VISION_JPEG_QUALITY_DEFAULT)
        self._vision_min_interval_s = float(
            vision_min_interval_s or VISION_MIN_INTERVAL_S_DEFAULT)
        self._cam_lock = threading.Lock()
        self._latest_raw: Optional[bytes] = None  # raw frame, never encoded here
        self._latest_raw_ts: Optional[float] = None
        self._latest_raw_w = 0    # width from sensor_msgs/Image (0 = unknown)
        self._latest_raw_h = 0    # height from sensor_msgs/Image (0 = unknown)
        self._latest_raw_enc = ""  # encoding from sensor_msgs/Image ('' = unknown)
        self._jpeg_b64: Optional[str] = None  # the encoded slot, worker-owned
        self._jpeg_w = 0
        self._jpeg_h = 0
        self._jpeg_ts: Optional[float] = None
        self._encode_error = False
        self._encode_error_reason: Optional[str] = None  # worker-owned diagnostic
        self._cam_stop = threading.Event()
        self._cam_thread: Optional[threading.Thread] = None

        # ── radius-based grasp park ──────────────────────────────────────────────
        # park the hand farther out using the object's median radius, which dodges
        # overshoots on small objects. the switch is OPENBOT_PICK_RADIUS_PARK and
        # the margin comes from OPENBOT_PICK_RADIUS_MARGIN_M.
        # the margin default is RADIUS_PARK_MARGIN_DEFAULT_M = 0.050 and the clamp
        # is [0.005, 0.060], wide enough for a 5 cm request; a narrower clamp would
        # silently cut it. OPENBOT_PICK_RADIUS_MARGIN_M=0.015 restores a tighter
        # park. the requested value is kept beside the clamped one so the pick
        # reply can name both when they differ.
        self._radius_park_on = os.environ.get('OPENBOT_PICK_RADIUS_PARK', '').strip().lower() == '1'
        self._radius_park_margin_req_m = RADIUS_PARK_MARGIN_DEFAULT_M
        try:
            _margin = float(os.environ.get('OPENBOT_PICK_RADIUS_MARGIN_M',
                                           repr(RADIUS_PARK_MARGIN_DEFAULT_M)).strip())
            if not math.isfinite(_margin):
                raise ValueError('non-finite margin')
            self._radius_park_margin_req_m = _margin
            self._radius_park_margin_m = max(
                RADIUS_PARK_MARGIN_MIN_M, min(RADIUS_PARK_MARGIN_MAX_M, _margin))
        except (TypeError, ValueError):
            self._radius_park_margin_req_m = RADIUS_PARK_MARGIN_DEFAULT_M
            self._radius_park_margin_m = RADIUS_PARK_MARGIN_DEFAULT_M
        # per-object radius history: instance_id -> [(ts, radius, quality), ...]
        # only quality >= 0.6 and 0.010 <= radius <= 0.080; samples older than 4.0 s
        # (relative to newest) and anything past 15 samples are trimmed.
        self._obj_radius_history: Dict[Any, List[Tuple[float, float, float]]] = {}
        #: instance id -> (wall time, nx, ny, nz) for its newest valid normal.
        self._obj_normal: Dict[Any, Tuple[float, float, float, float]] = {}
        #: width radius used for 'pca'/'sphere' fits (on by default).
        self._radius_park_width_on = (
            os.environ.get('OPENBOT_PICK_RADIUS_WIDTH', '1').strip() != '0')

    # ── ROS-facing hooks (all implemented by _RosGateway) ────────────────────

    def on_detections_packet(self, rows: List[dict]) -> None:
        """Ingest one /object_detections_body packet (event-driven)."""
        with self._obs_lock:
            self._assembler.update(detections=rows)
            # radius history for the radius_park feature. only samples with an
            # instance_id, quality >= 0.6 and radius in [0.010, 0.080]; at most 15 per
            # object, anything older than 4.0 s relative to the newest trimmed.
            now = time.time()
            self._note_normals(rows, now)        #: the pick-line normals
            for row in rows:
                iid = row.get('instance_id')
                if iid is None:
                    continue
                try:
                    r = float(row.get('object_radius', 0.0) or 0.0)
                    q = float(row.get('quality', 0.0) or 0.0)
                    # half the lateral width, for 'pca'/'sphere' fits
                    if (getattr(self, '_radius_park_width_on', True)
                            and str(row.get('primitive_type') or '')
                            in RADIUS_PARK_WIDTH_TYPES):
                        _w = float(row.get('width_m') or 0.0)
                        if math.isfinite(_w) and _w > 0.0:
                            r = 0.5 * _w
                    if q >= 0.6 and 0.010 <= r <= 0.080:
                        if iid not in self._obj_radius_history:
                            self._obj_radius_history[iid] = []
                        self._obj_radius_history[iid].append((now, r, q))
                except (TypeError, ValueError):
                    pass
            # trim: at most 15 samples, nothing older than 4.0 s
            for iid in list(self._obj_radius_history.keys()):
                samples = self._obj_radius_history[iid]
                if samples:
                    newest_ts = max(s[0] for s in samples)
                    # time filter first, then keep the 15 newest
                    samples[:] = [s for s in samples
                                  if s[0] >= newest_ts - 4.0]
                    if len(samples) > 15:
                        samples[:] = samples[-15:]
                    if not samples:
                        del self._obj_radius_history[iid]
        if getattr(self, "_container_lock", None) is not None:
            self._lock_ingest(rows)      # track the container
        self._b8l_rows = (time.time(), list(rows))   # row snapshot for the requester

    def on_walk_status_packet(self, payload: dict) -> None:
        """Ingest one /walk_status payload; feed the navigate epoch gate.

        Timestamps each message on arrival (the payload has no ts field).
        Phases observed at ts >= the current nav_epoch accumulate in
        ``phases_seen`` as (state, label, ts) tuples — the verifier's
        transition guard AND its label-vs-target check need them (fix 1).
        """
        now = time.time()
        phase = None
        phase_raw = None
        label = None
        if isinstance(payload, dict):
            state = payload.get("state")
            if isinstance(state, str) and state.strip():
                phase = state.strip().upper()
                phase_raw = phase  # un-folded: turn_around keeps DETECTED distinct
                if phase == "DETECTED":  # vocab event that happens inside SEARCHING
                    phase = "SEARCHING"
            lbl = payload.get("label")
            if isinstance(lbl, str) and lbl.strip():
                label = lbl.strip().lower()
        with self._obs_lock:
            self._assembler.update(walk_payload=payload, now=now)
            self._last_walk_ts = now
            if phase is not None:          #: the live FSM state
                self._last_walk_phase = phase
                self._last_walk_label = label
            if getattr(self, "_turn_active", False) and phase_raw:
                self._turn_events.append((phase_raw, label, now))
            if getattr(self, "_pose_active", False) and phase_raw:   # pose is active
                self._pose_events.append((phase_raw, label, payload, now))
            epoch_ts, _ = self._nav_epoch
            if phase is not None and epoch_ts > 0.0 and now >= epoch_ts:
                self._phases_seen.append((phase, label, now))
        if getattr(self, "_container_lock", None) is not None:
            self._lock_on_walk(phase_raw, label, payload)   # container lock
        if getattr(self, "_nav_strict", None) is not None:
            self._strict_on_walk(phase_raw, label, payload)  # strict

    # ── camera lane (vision.enabled only, fed by _RosGateway._on_camera) ────

    def on_camera_frame(self, raw: bytes, width: int = 0, height: int = 0,
                        encoding: str = "") -> None:
        """Ingest one raw camera frame (bytes) — NEVER encodes here.

        The DDS callback path copies the frame buffer to bytes and hands
        it off; this stamps the arrival time and stores the LATEST frame
        under the lock (CAMERA PLAN LAW: encode happens on the worker
        thread, never on the DDS thread).  The JPEG worker starts lazily
        on the first frame.

        The optional ``width``/``height``/``encoding`` (the
        sensor_msgs/Image metadata) are copied alongside the buffer so
        the worker can RESHAPE the uncompressed payload instead of
        imdecoding it.  The defaults preserve the historical
        dimensionless callers (encoded JPEG/PNG bytes) byte-for-byte —
        they keep the imdecode path.
        """
        with self._cam_lock:
            self._latest_raw = raw
            self._latest_raw_w = int(width or 0)
            self._latest_raw_h = int(height or 0)
            self._latest_raw_enc = str((encoding or "").strip())
            self._latest_raw_ts = time.time()
        self._ensure_camera_worker()

    def _ensure_camera_worker(self) -> None:
        """Start the JPEG-encode daemon lazily on the first frame."""
        with self._cam_lock:
            if self._cam_thread is not None or self._cam_stop.is_set():
                return
            thread = threading.Thread(target=self._camera_encode_loop,
                                      name="camera-jpeg", daemon=True)
            self._cam_thread = thread
        thread.start()  # start the thread outside the lock

    def _camera_encode_loop(self) -> None:
        """Daemon: throttled JPEG encode of the LATEST raw frame.

        THROTTLED-CACHE DESIGN: encodes at most one frame per
        ``vision.min_interval_s`` (default 1.0), NOT one per camera frame —
        the raw slot keeps updating on every callback, so the frame that
        encodes when the throttle opens is always the NEWEST one available
        (intermediate frames are skipped, never queued).  Client-side
        'one frame per decision turn' semantics stay at the caller.  cv2
        is imported lazily INSIDE this thread — the laptop/BOX core may
        lack OpenCV; a missing/broken cv2 or a failed encode marks
        ``_encode_error`` so the 'image' verb answers fresh:false honestly
        (a frame is attempted once; the error latches until the next frame
        succeeds).  The encoded slot lives under the same lock as the raw
        slot.  Runs until ``_cam_stop`` is set.
        """
        last_encode_ts = -1.0  # wall-clock time of the last encode, for the throttle
        last_encoded_raw_ts = -1.0  # raw-frame ts already encoded
        announced_reason = None  # last reason written to stderr, for dedupe
        while not self._cam_stop.is_set():
            with self._cam_lock:
                raw = self._latest_raw
                raw_ts = self._latest_raw_ts
                raw_w = self._latest_raw_w
                raw_h = self._latest_raw_h
                raw_enc = self._latest_raw_enc
            if raw is None or raw_ts is None or raw_ts <= last_encoded_raw_ts:
                self._cam_stop.wait(timeout=0.1)
                continue
            if time.time() - last_encode_ts < self._vision_min_interval_s:
                # inside the throttle interval: skip without consuming the raw slot, so
                # the newest frame still encodes once the interval opens (throttled cache,
                # not a queue).
                self._cam_stop.wait(timeout=0.1)
                continue
            encoded = self._encode_jpeg(raw, raw_w, raw_h, raw_enc)
            announce = None
            with self._cam_lock:
                last_encode_ts = time.time()
                last_encoded_raw_ts = raw_ts
                if encoded is None:
                    self._encode_error = True
                    # surface the reason once per distinct failure mode. the camera runs at
                    # ~12-30 Hz, so a per-frame line would be spam, but a reason nobody sees is
                    # worse: an encode failure that logs nothing is invisible.
                    reason = self._encode_error_reason or "unknown"
                    if reason != announced_reason:
                        announced_reason = reason
                        announce = reason
                else:
                    self._jpeg_b64, self._jpeg_w, self._jpeg_h = encoded
                    self._jpeg_ts = raw_ts
                    if self._encode_error:
                        announce = "recovered"
                    self._encode_error = False
                    self._encode_error_reason = None
                    announced_reason = None
            if announce is not None:
                # outside the lock: no I/O while holding it
                sys.stderr.write(
                    "openbot_gateway: camera encode %s (frame %sx%s %r)\n"
                    % (announce, raw_w, raw_h, raw_enc))
                sys.stderr.flush()
            if self._cam_stop.is_set():
                return
            self._cam_stop.wait(timeout=0.1)

    def _encode_jpeg(self, raw: bytes, width: int = 0, height: int = 0,
                     encoding: str = "") -> Optional[Tuple[str, int, int]]:
        """raw frame bytes -> (base64 jpeg str, w, h); None on failure.

        Two decode paths:
          * DIMENSIONED (width & height & encoding all set): the buffer is
            an UNCOMPRESSED sensor_msgs/Image payload — RESHAPE it via
            ``_reshape_raw_frame`` (bgr8 direct / rgb8 BGR-swapped / mono8
            single channel) instead of imdecoding.  A size mismatch or an
            unsupported encoding is an honest None (never a guessed
            reshape).
          * DIMENSIONLESS (the defaults): the historical callers feed
            ENCODED bytes (JPEG/PNG with a header); keep the imdecode
            route byte-for-byte.
        Lazy cv2/numpy imports (optional platform deps); any failure —
        missing cv2, undecodable frame, encoder error — returns None and
        never raises (the worker turns None into ``_encode_error``).
        Every failure also records a short reason in
        ``_encode_error_reason`` (worker-thread-owned diagnostic), so a
        fresh:false answer can say WHICH link failed.
        """
        try:
            import base64
            import cv2  # lazy: OpenCV is an optional platform dep
        except Exception:
            self._encode_error_reason = "missing_cv2"
            return None
        try:
            import numpy as np  # lazy: numpy is an optional platform dep
            if width and height and encoding:
                img = self._reshape_raw_frame(raw, width, height, encoding)
                if img is None:
                    return None
            else:
                img = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    self._encode_error_reason = "imdecode_failed"
                    return None
            ok_flag, buf = cv2.imencode(
                ".jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, int(self._vision_jpeg_quality)])
            if not ok_flag:
                self._encode_error_reason = "imencode_failed"
                return None
            return (base64.b64encode(buf.tobytes()).decode("ascii"),
                    int(img.shape[1]), int(img.shape[0]))
        except Exception:
            self._encode_error_reason = "encode_exception"
            return None

    def _reshape_raw_frame(self, raw: bytes, width: int, height: int,
                           encoding: str):
        """Uncompressed sensor_msgs/Image payload -> BGR ndarray; None honest.

        Only the three encodings the camera actually emits are accepted —
        bgr8 (encode directly), rgb8 (swap to BGR first; cv2.imencode
        expects BGR), mono8 (single channel).  Anything else, or a buffer
        whose length does not equal width*height*channels, returns None
        (HONESTY LAW: never a guessed reshape).  Lazy numpy import; never
        raises (the caller turns any exception into an honest None).
        """
        try:
            import numpy as np  # lazy: numpy is an optional platform dep
        except Exception:
            self._encode_error_reason = "missing_numpy"
            return None
        try:
            enc = (encoding or "").strip().lower()
            if enc == "mono8":
                channels = 1
            elif enc in ("bgr8", "rgb8"):
                channels = 3
            else:
                self._encode_error_reason = "unsupported_encoding:%s" % enc
                return None
            if len(raw) != width * height * channels:
                self._encode_error_reason = "size_mismatch"
                return None
            img = np.frombuffer(raw, dtype=np.uint8).reshape(
                (int(height), int(width), channels))
            if enc == "rgb8":
                img = img[:, :, ::-1]  # RGB -> BGR, cv2.imencode wants BGR
            return img
        except Exception:
            self._encode_error_reason = "reshape_failed"
            return None

    def close_camera(self) -> None:
        """Stop the JPEG worker and drop the frame slots (shutdown hygiene).

        The worker is a daemon (the process could exit without this), but a
        clean stop mirrors the node-destroy pattern: signal, join briefly,
        and clear the slots so nothing serves frames after close.  Safe to
        call when the lane was never enabled.
        """
        thread = None
        with self._cam_lock:
            self._cam_stop.set()
            thread = self._cam_thread
            self._latest_raw = None
            self._latest_raw_ts = None
            self._latest_raw_w = 0
            self._latest_raw_h = 0
            self._latest_raw_enc = ""
            self._jpeg_b64 = None
            self._jpeg_w = 0
            self._jpeg_h = 0
            self._jpeg_ts = None
        if thread is not None:
            thread.join(timeout=1.0)

    def _emit_verdict(self, verdict_str: str) -> None:
        """Publish ``verdict_str`` on /skill_verdict unless a lane run on THIS
        thread is deferring it (R4-DL-01).

        The lane path publishes the verdict it RETURNS — after the post-
        dispatch body check may have downgraded the engine's — from
        ``_low_lane_sequence``, once.  Everything else publishes here, once,
        exactly as ``_verify`` always did.
        """
        tls = getattr(self, "_verdict_tls", None)
        if tls is not None and getattr(tls, "defer", False):
            return                      # a lane run already owns the publish
        self.publish_verdict(verdict_str)

    def publish_verdict(self, verdict_str: str) -> None:
        """Hook: publish a /skill_verdict String (no-op outside ROS).

        The laptop-testable core is quiet; ``_RosGateway`` overrides this to
        publish on the topic. A verifier crash must never be masked by this
        hook — it must not raise into the verifier path.
        """
        return

    def navigate(self, label: str) -> None:
        """Hook: publish label + auto-'confirm' on /navigate_target (no-op).

        The coordinator (pick_walk_coordinator.py) runs a TWO-STEP handshake:
        a bare label only stages PENDING, and the search-then-go handshake
        fires only on a separate 'confirm' message. ``_RosGateway`` overrides
        this with the real topic publishes (label then 'confirm'); tests and
        the laptop core replace it with a recorder. The default is a no-op so
        the dispatch path is testable without ROS.
        """
        return

    def set_walk_instance(self, target: str) -> None:
        """Hook (B8): publish the navigate target on /walk_instance (no-op here).

        ``target`` is '<label>#<iid>@<hue>' to make the walk FSM follow ONE
        object of a class, or the bare label to clear that.  ``_RosGateway``
        overrides this; the coordinator and /navigate_target only ever see
        the bare label.
        """
        return

    def stop_walk(self) -> None:
        """Hook: route 'stop' through the coordinator (no-op outside ROS).

        The gateway publishes 'stop' to ``/navigate_target``; the coordinator
        (pick_walk_coordinator.py) resets its own epoch/flow state AND relays
        the walk FSM's REAL stop trigger — a ``/walk_target`` message whose
        data is 'stop' (or 'abort'), which walk_node.py queues as a
        ('stop', None) command for the FSM loop (skills/locomotion/walk_node.py
        _on_walk_target). ``/walk_go`` is NOT a stop channel: it queues an
        approach trigger and must never carry a stop label. ``_RosGateway``
        overrides this.
        """
        return

    def start_sweep(self, label: str) -> None:
        """Hook: publish a bare label DIRECTLY on /walk_target (no-op here).

        The turn-around sweep's one acting line (see ``_do_turn_around``).
        Deliberately NOT ``navigate()``: that path goes through
        /navigate_target, where pick_walk_coordinator.py's auto-approach
        (/walk_go on the 3rd consecutive detection beat) would walk the robot
        TO the person — the one thing the operator's ask excludes.  A direct
        /walk_target publish reaches walk_node._on_walk_target itself, which
        queues ('target', label) and starts the FSM when it is idle: the FSM
        sweeps and its own ``_sweep_turn`` interrupts on first sight.

        After the sweep sights the person the gateway ALSO calls ``stop_walk()``
        (batch 2) and then centres them itself; see ``_do_turn_around``.

        The laptop-testable core is quiet; ``_RosGateway`` overrides this to
        publish on the topic.  Tests replace it with a recorder.
        """
        return

    # ── skill bridges (arm / walk / grip) ───────────────────────────────────

    def writer_verb(self, line: str, timeout_s: float = 15.0) -> str:
        return _send_unix_verb(self._writer_sock, line, timeout_s)

    def _clamp_arm_dur(self, args: List) -> str:
        """Apply the runtime arm-duration floor to a writer passthrough line.

        Merged-writer grammar: ``arm <14 floats> [dur]`` — the writer's own
        clamp is dur >= 1.5 s; the EFFECTIVE floor is
        ``max(min_arm_dur_s, 1.5)`` (runtime embodiment, world-truth
        round 5).  Only ``arm`` lines with a dur token are touched; any
        other line passes through verbatim.  Returns the line to send.
        """
        if not args or str(args[0]).lower() != "arm" or len(args) < 16:
            return " ".join(str(a) for a in args)
        try:
            dur = float(args[15])
        except (TypeError, ValueError):
            dur = 0.0
        floor = max(float(self._min_arm_dur_s), 1.5)
        if dur < floor:
            dur = floor
        # %.3f either way (clamped or untouched)
        return " ".join(str(a) for a in args[:15]) + (" %.3f" % dur)

    def grip_verb(self, line: str, timeout_s: float = 15.0) -> str:
        return _send_unix_verb(self._grip_sock, line, timeout_s)

    def engine_verb(self, line: str, timeout_s: Optional[float] = None) -> str:
        if self._engine is None:
            return "ERROR:pick engine not configured (engine_cmd empty)"
        try:
            if not self._engine.alive():
                self._engine.start()
            return self._engine.send(line, timeout_s)
        except EngineTransportError as exc:
            # post-send failure: the engine may already have run the command, so the
            # reply says which kind. the prefix is the whole classification the restore
            # gate may use: a substring cannot tell a failed send from an engine
            # 'ERROR:OUT OF REACH', and the failed-send text never says 'timed out'.
            # caught before the generic clause, and the reply keeps the bare 'ERROR:'
            # shape so the brain and _classify_reply still read it.
            return ENGINE_TRANSPORT_ERROR_PREFIX + str(exc)
        except Exception as exc:
            return "ERROR:%s" % exc

    # ── observation (what the brain sees) ───────────────────────────────────

    def _refresh_grip_state(self) -> None:
        """Poll the grip server 'state' and feed it into the assembler.

        The grip server does NOT auto-verify and never pushes state — the
        observation's GripState (joints + available, contact folded from the
        last close verdict) exists only because something queries 'state'.
        A failed query is recorded as a READ ERROR (note_grip_read_error) so
        the last good state AGES OUT (available=False) instead of confirming
        forever (fix 2); an ERROR:/malformed state reply line is equally a
        read failure inside update().  The next successful read supersedes
        the error.
        """
        try:
            line = self.grip_verb("state")
            with self._obs_lock:
                self._assembler.update(grip_line=line)
        except Exception:
            with self._obs_lock:
                self._assembler.note_grip_read_error()

    def observe_dict(self) -> dict:
        """WorldObservation as a JSON-able dict (for the agent's 'observe')."""
        from dataclasses import asdict  # stdlib import

        self._refresh_grip_state()
        with self._obs_lock:
            obs = self._assembler.observation()
        out = {
            "detections": [_observe_row(asdict(d)) for d in obs.detections],
            "grip": asdict(obs.grip) if obs.grip is not None else None,
            "walk": asdict(obs.walk) if obs.walk is not None else None,
            "detections_ts": obs.detections_ts,
            "fresh": obs.fresh,
            "detections_alive": self._assembler.detections_alive(),
        }
        # the top-level ``vocab`` block appears only once a /yolo/classes_state is
        # cached (`open_vocab` param true and at least one state received).
        # otherwise it is absent and the reply is byte-identical to the plain one.
        vocab = _observe_vocab_block(self.yolo_vocab_state())
        if vocab is not None:
            out["vocab"] = vocab
        return out

    def _do_image(self, args: Optional[List] = None) -> str:
        """'image' verb: the latest camera frame as JSON text.

        FRESHNESS LAW: a frame ships under the green light (fresh:true)
        ONLY when a JPEG for it exists AND it is no older than
        vision.max_age_s.  Stale / missing / encode-error frames answer
        fresh:false with an honest ``reason`` — NEVER a stale frame with
        the green light.  With vision disabled there is no camera at all
        (reason 'no_camera').  ``ts`` is the frame time of the newest
        artifact we hold (the JPEG's frame on success; the raw frame's
        time when only a raw frame exists); 0 when no frame ever arrived.
        """
        now = time.time()
        with self._cam_lock:
            if not self._vision_enabled or self._latest_raw_ts is None:
                return json.dumps({"jpeg_b64": "", "w": 0, "h": 0, "ts": 0,
                                   "fresh": False, "reason": "no_camera"})
            ts = self._jpeg_ts if self._jpeg_ts is not None else self._latest_raw_ts
            if self._encode_error or self._jpeg_b64 is None:
                # the encode failed, or the first encode has not landed yet: no jpeg to
                # ship for any fresh frame.
                return json.dumps({"jpeg_b64": "", "w": 0, "h": 0, "ts": ts,
                                   "fresh": False, "reason": "encode_error"})
            if now - ts > self._vision_max_age_s:
                return json.dumps({"jpeg_b64": "", "w": 0, "h": 0, "ts": ts,
                                   "fresh": False, "reason": "stale"})
            # snapshot the artifact under the lock, because the worker can swap the
            # live slots the instant it drops; the fresh:true payload is built from
            # this copy, never from live fields read past the freshness gate.
            jpeg_b64, jpeg_w, jpeg_h = self._jpeg_b64, self._jpeg_w, self._jpeg_h
        if jpeg_b64 is None:  # defensive: never json.dumps a None b64
            return json.dumps({"jpeg_b64": "", "w": 0, "h": 0, "ts": ts,
                               "fresh": False, "reason": "encode_error"})
        return json.dumps({"jpeg_b64": jpeg_b64, "w": jpeg_w, "h": jpeg_h,
                           "ts": ts, "fresh": True})

    # ── tool dispatch (the verbs the agent calls) ────────────────────────────

    def dispatch(self, verb: str, args: Optional[List] = None) -> Tuple[str, Optional[str]]:
        """Tool verb -> (reply text, verdict string | None).

        Motion verbs (pick/place/navigate/open_hand/close_hand) run the
        SkillVerifier post-hook and return the verdict for the caller to
        publish on /skill_verdict. Unknown verbs / bad args -> ERROR reply.

        OPEN-VOCABULARY LANE (batch 4, P3).  Two additions, both inert unless
        the ROS param ``open_vocab`` is exactly True:

          * ``yolo_classes`` is answered FIRST — ahead of the descent lane's
            next-command rule.  It is a non-motion verb; reaching that rule
            with a hold outstanding would RISE the body for a vocabulary
            change.  With the param False it only returns the honest
            ``disabled`` refusal, and no other verb reaches this arm, so every
            existing verb's path is unchanged.
          * the actuator verbs in ``_YOLO_BUSY_VERBS`` are counted in flight
            (``_motion_enter``/``_motion_exit``) so ``yolo_classes`` can refuse
            while a pick/place/navigate is running on ANOTHER tools-socket
            thread.  With the param False the count is never touched and the
            call goes straight to ``_dispatch_verbs`` — today's body, verbatim.

        BATCH 5 (G3): every verb in ARM_TASK_VERBS is ALSO counted by the
        unconditional ``_arm_task_inflight`` counter, which the idle-arm
        watchdog reads.  It is independent of ``_open_vocab``, changes no
        reply, and is skipped on a stand-in built without ``__init__``.
        """
        if verb in ARM_TASK_VERBS and getattr(self, "_inflight_lock", None) \
                is not None:
            self._arm_task_enter()
            try:
                return self._dispatch_counted(verb, args)
            finally:
                self._arm_task_exit()
        return self._dispatch_counted(verb, args)

    def _arm_task_enter(self) -> None:
        with self._inflight_lock:
            self._arm_task_inflight += 1

    def _arm_task_exit(self) -> None:
        with self._inflight_lock:
            self._arm_task_inflight = max(0, self._arm_task_inflight - 1)
            if self._arm_task_inflight == 0:
                self._arm_idle_since = time.time()

    def _dispatch_counted(self, verb: str, args: Optional[List] = None
                          ) -> Tuple[str, Optional[str]]:
        """The pre-batch-5 ``dispatch`` body, unchanged."""
        if verb == "yolo_classes":
            return self._do_yolo_classes(args or []), None
        if verb == "yolo_strict":       # inside a strict detection hold
            return self._do_yolo_strict(args or []), None
        if (verb in self._YOLO_BUSY_VERBS
                and getattr(self, "_open_vocab", False) is True):
            token = self._motion_enter(verb)
            try:
                return self._dispatch_verbs(verb, args)
            finally:
                self._motion_exit(token)
        return self._dispatch_verbs(verb, args)

    def _dispatch_verbs(self, verb: str, args: Optional[List] = None
                        ) -> Tuple[str, Optional[str]]:
        """Today's verb table (the pre-batch-4 ``dispatch`` body, unchanged)."""
        args = args or []
        # ── low-workspace descent lane ───────────────────────────────────────────
        # one choke point between args and the pick arm: the brain socket, the ROS
        # adapter and every direct caller flow through dispatch, and _do_pick /
        # _do_place are not modified (the lane calls them directly, so no
        # recursion). _low_lane is None when the lane is off, the config switch is
        # off, or descent_lane failed to import, and then every arm below falls
        # through to the plain path, byte for byte.
        #
        # re-entrancy: the /place_target lane sends its inner `place` back through
        # this method, so _descent_nested is true for that one call and every arm
        # below declines, keeping the one-shot lane out of the descent lane.
        #
        # the post-cancel stale-target gate comes before the lane choke point, so
        # it compares the brain's raw (x, y, z) torso frame before any lane
        # z_engine shift; a refusal never has to descend the body.
        if (verb in ("pick", "pick_left")
                and getattr(self, "_b5_on", False) is True):
            gated = self._post_cancel_target_gate(verb, args)
            if gated is not None:
                return gated
        if (verb in ("pick", "pick_left") and not self._descent_nested
                and self._low_lane):
            lane_pick = self._low_lane_pick(verb, args)
            if lane_pick is not None:
                return lane_pick
        if verb == "place" and not self._descent_nested and self._low_lane:
            lane_place = self._low_lane_place(args)
            if lane_place is not None:
                return lane_place
        if self._low_lane is not None and not self._descent_nested:
            lane_held = self._low_lane_hold_rule(verb, args)
            if lane_held is not None:
                return lane_held
        if verb == "pick" or verb == "pick_left":
            return self._do_pick(verb, args)
        if verb == "place":
            return self._do_place(args)
        if verb == "navigate":
            return self._do_navigate_strict(args)   # inside a strict detection hold
        # turn-around sweep: additive, since the arm only fires for this verb
        # and every other dispatch path stays byte-identical as long as
        # nobody invokes it.
        if verb == "turn_around":
            self._lock_clear("turn_around")      # the body is turning
            return self._do_turn_around(args)
        # mid-task cancel: one verb covering all three lanes, pick/place (engine),
        # sweep and search (walk FSM). different from `navigate stop`, which stops
        # only the walk lane and remains the agent's precise "stop walking".
        if verb in ("cancel", "stop"):
            # get the full vocabulary back first: a strict failure is logged and never
            # holds the cancel up
            try:
                self._strict_close_all("cancel")
            except Exception as exc:  # noqa: BLE001 -- the cancel always runs
                self._strict_log("close_all (cancel) failed: %r; cancelling "
                                 "anyway" % (exc,))
            self._lock_clear("cancel")           # this one never raises
            self._home_stop("cancel")            # abandon before the pick, pause after it
            return self._do_cancel(args)
        if verb == "open_hand":
            return self._do_open_hand(args)
        if verb == "close_hand":
            return self._do_close_hand(args)
        if verb == "observe":
            return json.dumps(self.observe_dict()), None
        if verb == "status":
            return self._do_status(args), None
        if verb == "image":
            return self._do_image(args), None
        if verb == "cmd_set":
            self._lock_clear("cmd_set")          # an outside leg command arrived
            return self._do_cmd_set(args), None
        if verb == "container":                 # the container the node is locked on
            return self._do_container(args), None
        if verb == "creep":                     # the last centimetres
            return self._do_creep(args), None
        if verb == "pc_rim":                    # the point-cloud bowl rim
            return self._do_pc_rim(args), None
        if verb == "home":                      # head for the fetch start point
            return self._do_home(args), None
        if verb == "return_home":               # walk it back home
            self._lock_clear("return_home")      # the body walks off
            return self._do_return_home(args), None
        if verb == "approach_person":           # centre up then creep in
            self._lock_clear("approach_person")  # the body moves in
            return self._do_approach_person(args), None
        if verb == "step_back":                 # runs before a navigate sweep
            self._lock_clear("step_back")        # the body moves back
            return self._do_step_back(args), None
        if verb == "aim_turn":                  # object sits right of centre
            self._lock_clear("aim_turn")         # the body turns to aim
            return self._do_aim_turn(args), None
        if verb == "writer":
            if not args:
                return "ERROR:writer needs a verb line", None
            return self.writer_verb(self._clamp_arm_dur(args)), None
        if verb == "grip":
            if not args:
                return "ERROR:grip needs a verb line", None
            return self.grip_verb(" ".join(str(a) for a in args)), None
        if verb == "engine":
            if not args:
                return "ERROR:engine needs a verb line", None
            return self.engine_verb(" ".join(str(a) for a in args)), None
        # query_user and notify_user are human I/O verbs. both non-motion: the
        # verdict is always None (never from the SkillVerifier post-hook, since they
        # make no motion claim), so their envelopes come through
        # _shape_reply(reply, None) and nowhere else.
        #
        # exact reply grammar (the scenario-layer worker mirrors these strings
        # verbatim, so do not reword them):
        #   query_user  -> DONE:<answer>            (one line of human answer)
        #              or ERROR:no_user_reply:timeout (nothing within
        #                 QUERY_USER_REPLY_TIMEOUT_S, or stdin hit EOF)
        #   notify_user -> DONE:notified (a plain ack)
        # when an operator answer contains the literal 'ERROR:' (e.g.
        # 'ERROR:try again'), _shape_reply calls it ok:false. that is the
        # failure-budget direction and safety-conservative: the envelope can never
        # claim ok:true for a reply the verifier judges a failure. the scenario
        # mirror shares the same _shape_reply.
        if verb == "query_user":
            return self._do_query_user(args), None
        if verb == "notify_user":
            return self._do_notify_user(args), None
        if verb == "ping":
            return "PONG", None
        return "ERROR:unknown verb %r" % verb, None

    # ── low-workspace descent lane glue ──────────────────────────────────────
    # the thin gateway wiring around the descent lane. the pure half (ladder,
    # floor, confirm/refuse/warn text, phase log, pick/place sequences) is
    # stdlib-only and rclpy-free; this side supplies the sockets (writer_verb),
    # the single status probe, the shipped _do_pick/_do_place (called directly,
    # never re-dispatched) and the lease. every method below no-ops while
    # _low_lane is None.

    def _low_lane_args_z(self, args: List) -> Optional[float]:
        """The torso-frame ``z`` token of a pick/place argument list.

        ``None`` when there is no third token or it is not numeric — the same
        values ``_do_pick``/``_do_place`` themselves refuse, so the lane must
        not claim them: an absent z falls through to today's own error.
        """
        if len(args) < 3:
            return None
        try:
            return float(args[2])
        except (TypeError, ValueError):
            return None

    def _low_lane_pick(self, verb: str, args: List
                       ) -> Optional[Tuple[str, Optional[str]]]:
        """The lane's answer to a pick, or None to fall through to _do_pick."""
        return self._low_lane_claim(verb, args, self._low_lane_args_z(args),
                                    pick=True)

    def _low_lane_place(self, args: List
                        ) -> Optional[Tuple[str, Optional[str]]]:
        """The lane's answer to a coordinate place, or None to fall through."""
        return self._low_lane_claim("place", args, self._low_lane_args_z(args),
                                    pick=False)

    def _low_lane_claim(self, verb: str, args: List, z: Optional[float], *,
                        pick: bool) -> Optional[Tuple[str, Optional[str]]]:
        """Does the descent lane OWN this pick/place?  ``None`` = today's path.

        Three decisions, and the first two both leave the engine and the writer
        completely untouched:

          * a DEGENERATE band (``usable`` False — the writer's band is SHUT by
            default, D5, or the envelope is misconfigured) can rescue nothing,
            so the lane stays inert and today's honest refusal answers;
          * a target BELOW the rescue floor is genuinely unreachable, and the
            lane OWNS that refusal so the operator is told where the floor is
            and how much band a wider writer clamp would buy (plan §4 A3);
          * a target in the DESCEND band is the lane's — the sequence decides
            HOLD vs re-descend from the shared context (plan §6 D2).

        Anything else — at or above ``COMFORT_Z``, or no usable ``z`` — falls
        through: with nothing held that is today's path byte for byte, and with
        a body height held the next-command rule
        (:meth:`_low_lane_hold_rule`) owns the retreat.
        """
        from brain import descent_law  # lazy: only reachable with the lane armed
        law = self._low_law
        if not self._low_lane.usable(law):
            return None
        kind = descent_law.classify_pick_z(z, law.tabletop_min_z,
                                           law.max_descent_m)
        if kind == "DEEP_FLOOR":
            return self._low_lane.deep_floor_refusal(z, law), None
        if kind != "DESCEND":
            return None
        return self._low_lane_sequence(verb, args, z, pick=pick)

    def _low_lane_sequence(self, verb: str, args: List, z: float, *,
                           pick: bool) -> Tuple[str, Optional[str]]:
        """Run one lane skill under the lease and render its phase table.

        CONCURRENCY IS REFUSED, NEVER QUEUED (plan §6): the lock is taken
        non-blocking for the WHOLE skill, exactly as ``place_target`` does, so a
        second lane skill cannot descend from — or retreat out of — a body
        height the first one is still using.  The writer callables are resolved
        per call (never captured at construction), so a stubbed ``writer_verb``
        sees exactly the writes the lane issues.
        """
        if not self._descent_lock.acquire(blocking=False):
            return self._low_lane.DESCENT_BUSY_MSG, None
        try:
            # this run owns the /skill_verdict publish, so _verify (called inside
            # _do_pick/_do_place) defers on this thread and the verdict the lane returns
            # gets published once below.
            self._verdict_tls.defer = True
            # stamp the run so a cancel arriving while this lock is held
            # (_low_lane_cancel flags it) is seen by this run only.
            ctx = self._descent_ctx
            ctx.begin_run(time.monotonic())
            law = self._place_target_law
            log = self._low_lane.DescentPhaseLog(verb)
            log.start(time.time())
            if pick:
                log.side = "left" if verb == "pick_left" else "right"
                # point-cloud hooks: {} whenever self._pc is None (all switches off), so
                # the call below is exactly the plain one.
                pc_kw = self._pc_pick_kwargs()
                reply, verdict = self._low_lane.pick_sequence(
                    ctx, log, verb=verb, z=z, args=args,
                    law=self._low_law, writer_verb=self.writer_verb,
                    status_verb=self._low_lane_status, do_pick=self._do_pick,
                    pre_descent_step=self._pre_descent_step, **pc_kw)
            else:
                reply, verdict = self._low_lane.place_sequence(
                    ctx, log, z=z, args=args, law=self._low_law,
                    writer_verb=self.writer_verb,
                    status_verb=self._low_lane_status, do_place=self._do_place,
                    pre_descent_step=self._pre_descent_step)
                # which side got released is stated only in _do_place's own reply
                # annotation, so read it rather than re-deriving the held-side law here.
                log.side = law.side_from_reply(reply)
            # post-run cancel sweep: a cancel landing after the sequence's last check
            # but before this lock releases must never leave a hold behind.
            if (ctx.state == self._low_lane.HOLDING
                    and ctx.cancelled_this_run()):
                c_ok, _c_ref, _c_detail = self._low_lane._retreat_step(  # noqa: SLF001
                    ctx, log, writer_verb=self.writer_verb,
                    status_verb=self._low_lane_status, stamp=time.time,
                    clock=time.monotonic, sleep=time.sleep)
                if c_ok:
                    suffix = self._low_lane.HONESTY_SUFFIX
                    body = reply
                    if body.endswith(" — " + suffix):
                        body = body[:-len(" — " + suffix)]
                    reply = self._low_lane.honest(
                        "%s; operator cancel: the hold was released and the "
                        "body rose" % body)
            status, detail = law.verdict_row(verdict)
            self._low_lane_finish(log, status, detail)
            # the only publish for this lane run: exactly the verdict returned
            # (post-downgrade, honesty-suffixed). a refusal returns None and then
            # publishes nothing, including an unconfirmed post-engine retreat (fail
            # closed). a direct publish_verdict, because the deferral flag is still set
            # here.
            if verdict is not None:
                self.publish_verdict(verdict)
            return reply, verdict
        finally:
            self._verdict_tls.defer = False
            self._descent_lock.release()

    def _pc_pick_kwargs(self) -> Dict[str, Any]:
        """The optional point-cloud keywords for ``pick_sequence`` (batch 3b).

        ``{}`` when ``self._pc`` is None — every switch off — so the lane call
        carries no new keyword.  Otherwise ``{"pc": <client>, "pc_label":
        <label>}``, by keyword, each only if ``pick_sequence`` DECLARES it
        (``pc_client.supported_kwargs``: ``descent_lane.py`` and this file are
        copied to the robot one at a time, and a switch turned on against an
        older lane must leave the pick as it was, with one ``[pc]`` line
        saying so, never a TypeError mid-pick).

        ``pc_label`` is ``getattr(self._req_target_hint, "label", None)``: the
        label of the object THIS request named.  ``handle()`` clears the
        thread-local on every request and, for a named-args pick, sets it from
        the request's own ``label`` argument (``_pick_hint_label``); this runs
        on that SAME connection thread (handle -> dispatch -> _low_lane_pick ->
        _low_lane_sequence -> here is synchronous end to end).  It is None for
        a legacy positional request, for a label outside ``[A-Za-z0-9_.-]``
        (``_pick_hint_label`` refuses a space), and for a caller that is not a
        connection thread; the consumer reads None as "no label" and answers
        UNKNOWN.  The value is the brain's RAW label (often a scene-graph ref
        such as ``bottle_1``); ``PcClient.request`` turns it into the class
        noun the consumer matches rows by.  NEVER raises: any failure here is
        "no point-cloud keywords for this call".
        """
        try:
            pc = getattr(self, "_pc", None)
            if pc is None:
                return {}
            from gateway import pointcloud_client as pc_client_mod  # lazy, then cached
            wanted = {"pc": pc,
                      "pc_label": getattr(self._req_target_hint, "label", None)}
            given = pc_client_mod.supported_kwargs(
                self._low_lane.pick_sequence, wanted)
            if len(given) != len(wanted):
                self._pc_skew_once("pick_sequence", sorted(set(wanted) - set(given)))
            return given
        except Exception as exc:  # noqa: BLE001 — the pick path must not break
            self._pc_say("pick hooks off for this call: %s: %s"
                         % (type(exc).__name__, exc))
            return {}

    def _pc_place_kwargs(self, law: Any) -> Dict[str, Any]:
        """The optional ``pc_rim`` keyword for ``place_target_from_row``.

        Same rule as :meth:`_pc_pick_kwargs`: passed only if the function
        DECLARES it, so a ``place_target.py`` older than this file still
        places (without the rim hook, with one ``[pc]`` line).  Never raises.
        """
        try:
            from gateway import pointcloud_client as pc_client_mod  # lazy, then cached
            wanted = {"pc_rim": self._pc_rim_for}
            given = pc_client_mod.supported_kwargs(law.place_target_from_row, wanted)
            if len(given) != len(wanted):
                self._pc_skew_once("place_target_from_row", sorted(wanted))
            return given
        except Exception as exc:  # noqa: BLE001 — the place path must not break
            self._pc_say("place hooks off for this call: %s: %s"
                         % (type(exc).__name__, exc))
            return {}

    def _pc_skew_once(self, callee: str, missing: List[str]) -> None:
        """Say ONCE per callee that the point-cloud keywords were not passed."""
        seen = self.__dict__.setdefault("_pc_skew_seen", set())
        if callee in seen:
            return
        seen.add(callee)
        self._pc_say("%s declares no %s keyword(s): that module is older than "
                     "gateway_node.py - this hook is OFF until it is copied "
                     "over" % (callee, "/".join(missing)))

    def _low_lane_cancel(self) -> str:
        """Route an operator ``cancel``/``stop`` to the descent lane (R4 L2).

        Returns the ``; lane=<tok>`` segment for the cancel reply, or ``""``
        when the lane has nothing outstanding (no token, no motion — the cancel
        reply is then byte-identical to the lane-less one).

          * a lane skill is RUNNING (the lease is held) -> the cancel is
            FLAGGED on the context and this returns at once
            (``lane=cancel_flagged``); the running sequence acts on it at its
            last gate before the engine, after the engine (no hold), or in the
            post-run sweep.
          * a HOLD / descent is outstanding and nobody is running -> the body
            is risen now with the lane's one ``_retreat_step``
            (``lane=rose`` / ``lane=retreat_unconfirmed``; an unconfirmed rise
            has already emitted its loud warn-and-hold line).

        No token contains ``ERROR:``: the cancel itself worked, so its envelope
        stays ok:true.
        """
        ctx = self._descent_ctx
        if ctx is None:
            return ""
        if not self._descent_lock.acquire(blocking=False):
            ctx.request_cancel(time.monotonic())
            return "; lane=cancel_flagged"
        try:
            if not ctx.in_flight():
                return ""
            log = self._low_lane.DescentPhaseLog("cancel")
            log.start(time.time())
            ok, _ref, detail = self._low_lane._retreat_step(  # noqa: SLF001
                ctx, log, writer_verb=self.writer_verb,
                status_verb=self._low_lane_status, stamp=time.time,
                clock=time.monotonic, sleep=time.sleep)
            self._low_lane_finish(log, "ok" if ok else "refuse", detail)
            return "; lane=rose" if ok else "; lane=retreat_unconfirmed"
        finally:
            self._descent_lock.release()

    def _low_lane_status(self) -> str:
        """One ``status`` poll of the writer socket — the lane's ONLY probe.

        Resolved through ``self.writer_verb`` at call time (not captured at
        construction) so a test stub sees the poll.  There is no body-height
        sensor anywhere in this system: this line's height fragment is a
        REFERENCE the policy is being told to track, never a measurement.
        """
        return self.writer_verb("status")

    def _low_lane_finish(self, log: Any, status: str, detail: str) -> None:
        """Close the phase table and PRINT it — the operator's only view.

        The sequences close the table themselves on a refusal; a run that got
        as far as the engine returns with the VERDICT row still open, so it is
        closed here.  Rendering and printing reuse ``place_target``'s one table
        renderer and the gateway's own table hook (a second renderer would be a
        divergent copy of the repo's measured-elapsed format).
        """
        if not log.rows or log.rows[-1].get("phase") != "VERDICT":
            log.finish(status, detail, time.time())
        self.log_place_target(self._place_target_law.render_table(log))

    def _low_lane_retreat(self, verb: str, blocking: bool) -> bool:
        """Rise the body before a non-lane command runs (plan §6).

        ``descent_lane._retreat_step`` is the SAME step the sequences use, so
        the confirm/refuse/warn text and the ``height off`` + ``status`` pair
        exist exactly once in the repo.  Returns True when the body is
        confirmed standing; on failure it returns False for a BLOCKING command
        (the caller must refuse and dispatch nothing) and True for a best-effort
        one, which proceeds after the loud warn the step already printed.
        """
        log = self._low_lane.DescentPhaseLog(verb)
        log.start(time.time())
        ok, _ref, detail = self._low_lane._retreat_step(  # noqa: SLF001
            self._descent_ctx, log, writer_verb=self.writer_verb,
            status_verb=self._low_lane_status, stamp=time.time,
            clock=time.monotonic, sleep=time.sleep)
        self._low_lane_finish(log, "ok" if ok else "refuse", detail)
        return bool(ok) or not blocking

    def _low_lane_hold_rule(self, verb: str, args: List
                            ) -> Optional[Tuple[str, Optional[str]]]:
        """The operator's next-command rule, applied at command arrival (§6).

        The PREVIOUS skill cannot see the future, so the hold is decided here,
        at the next command: a lane skill needing the same height is already
        claimed by the pick/place arms and HOLDs there; everything else either
        rises first or forgets the hold.  Returns a REFUSAL only when a command
        must not run at a body height that may still be low; otherwise it
        performs the rise (if any) and returns None so today's path runs.

        A command arriving with no descent outstanding — which is every command
        on the lane-off path, and the common case on the lane-on path — does
        nothing here at all.
        """
        ctx = self._descent_ctx
        if ctx is None or not ctx.in_flight():
            return None
        action, reason = self._low_lane.next_command_action(
            ctx, verb, self._low_lane_args_z(args), self._low_law,
            time.monotonic(), args=args)
        if action in (self._low_lane.NEXT_NONE, self._low_lane.NEXT_HOLD):
            # NEXT_HOLD arrives here only for the release-only place clause in
            # descent_lane.next_command_action (a place or place_left with an empty
            # argument list inside a live hold). no lane arm above claims it, since its
            # target height is absent, and the body is already held at the confirmed
            # height, so there is nothing to rise for: the drop aims at no target
            # height. return None the way NEXT_NONE does and let the release run at the
            # held height. falling through to the retreat below would rise the body for
            # exactly the case the lane exists to hold down.
            #
            # NEXT_REDESCEND cannot reach this point: a lane skill whose target is in
            # the descend band is already claimed by the arms above, and one whose
            # target is not just answers RETREAT_* here.
            return None
        if self._low_lane_retreat(verb,
                                  action == self._low_lane.NEXT_RETREAT_BLOCKING):
            return None
        detail = self._low_lane.honest(
            "%s — body height may still be low; the %s was NOT dispatched (%s)"
            % (self._low_lane.RETREAT_UNCONFIRMED, verb, reason))
        return ("ERROR:low-workspace %s" % detail), None

    def _low_lane_retreat_first_if_held(self) -> Optional[str]:
        """Rise the body before a lane whose ENTRY bypasses ``dispatch`` (§6).

        ``/place_target``'s ROS callback calls :meth:`place_target` directly, so
        the next-command rule in ``dispatch`` never sees the request: with a
        descent held that lane would place at RAW coordinates, with no frame
        shift, and leave the body low (refuter MEDIUM-1).  This applies the same
        operator rule at the one-shot lane's own entry.

        Returns ``None`` when there is nothing to rise from — no lane, no
        context, or a context that is ``IDLE``, which is every call on the
        lane-off path and the common case on the lane-on path — and after a
        CONFIRMED rise.  Any other outcome returns the refusal REASON and the
        caller must dispatch nothing (fail-closed, plan §7).

        The step is the SAME ``_retreat_step`` the sequences and the
        next-command rule use, so the ``height off`` + ``status`` pair and the
        ``RETREAT UNCONFIRMED`` warn-and-hold text exist exactly once in the
        repo.  ``descent_lane.retreat_first_if_held`` is the pure half of this
        seam (repair task B); when it is present it drives the step, and when it
        has not landed this method drives the identical step itself.
        """
        ctx = self._descent_ctx
        if self._low_lane is None or ctx is None or not ctx.in_flight():
            return None
        log = self._low_lane.DescentPhaseLog("place_target")
        log.start(time.time())
        step = getattr(self._low_lane, "retreat_first_if_held", None)
        if step is None:
            ok, _ref, detail = self._low_lane._retreat_step(  # noqa: SLF001
                ctx, log, writer_verb=self.writer_verb,
                status_verb=self._low_lane_status, stamp=time.time,
                clock=time.monotonic, sleep=time.sleep)
            reason = None if ok else detail
        else:
            # the lane half takes its clocks by the names the sequences use (wall =
            # wall stamp, monotonic = hold clock), so no clock gets crossed.
            reason = step(ctx, log, writer_verb=self.writer_verb,
                          status_verb=self._low_lane_status, wall=time.time,
                          monotonic=time.monotonic, sleep=time.sleep)
        self._low_lane_finish(
            log, "refuse" if reason else "ok",
            reason or "body standing — the /place_target lane may run")
        return self._low_lane.honest(reason) if reason else None

    #: the characters verify_pick's hint regex accepts
    #: (target_label\s*=\s*([A-Za-z0-9_.\-]+)). anything else is silently
    #: truncated by that regex, and a truncated token followed by `=` could
    #: slip a second field into the contract text. so a label outside the
    #: charset is refused and no hint is sent: no hint gives the honest
    #: in_progress reply, the safe failure direction, while a mangled hint is
    #: not.
    _HINT_LABEL_CHARS = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")

    @classmethod
    def _pick_hint_label(cls, verb: str, args: dict) -> Optional[str]:
        """The object LABEL to hint `verify_pick` with, or None.

        Mirrors the brain's `_runtime_target_hint` (brain/hooks.py:587-616) —
        same pick-only restriction, because `verify_pick` is the one verifier
        that resolves its target out of the contract text, and handing the
        token to a place/navigate contract would be untested meaning.

        LABEL ONLY, deliberately, where the brain falls back to `ref`.  The
        hint is emitted as `target_label=<v>`, and `_target_present` compares
        it against `d.label`; a `ref` value placed in that slot matches no
        label and would read as "object gone" — a FALSE SUCCESS, the one
        direction the house error bias forbids.  `make_pick.prepare` always
        sets `label` (brain/agent/tools.py:618-620 refuses without it), so
        nothing is lost by declining to guess.
        """
        if verb not in ("pick", "pick_left"):
            return None
        raw = str(args.get("label") or "").strip()
        if not raw or not set(raw) <= cls._HINT_LABEL_CHARS:
            return None
        return raw

    def _shape_named_args(self, verb: str, args: dict) -> List:
        """Named-args request -> the legacy positional args list.

        The brain_client protocol (gateway/brain_client.py) sends
        ``{"tool": <verb>, "args": {<named json values>}}``; dispatch()
        consumes the positional form.  Keys follow the tools grammar
        documented at the top of this module; absent keys are omitted
        (dispatch enforces what is required).  Unknown verbs -> [] so
        dispatch answers the honest unknown-verb ERROR itself.

        PICK ORIENTATION (round 2026-09-07): the pick positional grammar ends
        ``... [orientation] [height_m]``, and the orientation token is ALWAYS
        emitted with a concrete value ('upright'|'horizontal') — never
        stripped.  An explicit concrete orientation passes through; 'auto' or
        an absent orientation resolves via ``_resolve_pick_orientation`` to
        the freshest nearby detection's orientation (defaulting to 'upright'
        — the safe/legacy behaviour).  This keeps a lying (horizontal)
        object from silently being picked with the upright approach.

        Defensive pick shaping (round-4 finding 3): the pick positional
        grammar is ``x y z [nx ny nz] [orientation] [height_m]`` — a
        PARTIAL normal vector (some of nx/ny/nz given) must never produce
        a truncated, misaligned list, so missing nx/ny/nz components are
        filled with 0.0 whenever any is present.  An out-of-range or
        negative ``height_m`` (the engine's own plausibility-gated
        tolerance: not in (0, PICK_HEIGHT_M_MAX]) is refused SILENTLY —
        never emitted, never raised — mirroring the engine's own gate
        exactly (pick_engine_v2.py / pick_engine_v3.py: <= 0 or > max ->
        legacy path).
        """
        if not isinstance(args, dict):
            raise ValueError("named-args request requires a JSON object")
        if verb in ("pick", "pick_left"):
            ordered: List[Any] = []
            for key in ("x", "y", "z"):
                if args.get(key) is not None:
                    ordered.append(args[key])
            normals = [args.get(k) for k in ("nx", "ny", "nz")]
            if any(v is not None for v in normals):
                # partial normal: fill the missing components with 0.0 so the three-slot
                # [nx ny nz] grammar position can never shift (the engine wants exactly 3).
                ordered.extend(0.0 if v is None else v for v in normals)
            # the orientation token is always concrete here, never 'auto' and never
            # stripped: 'auto' or absent resolves against the freshest nearby
            # detection, so a horizontal object is picked horizontal, else 'upright'.
            ordered.append(self._resolve_pick_orientation(args, ordered))
            height_m = args.get("height_m")
            if height_m is not None:
                try:
                    hf = float(height_m)
                except (TypeError, ValueError):
                    hf = float("nan")
                # same plausibility gate as the engine: refuse silently, never emit a junk
                # tolerance, and never let a non-finite value reach the engine.
                if math.isfinite(hf) and 0.0 < hf <= PICK_HEIGHT_M_MAX:
                    ordered.append(height_m)
                else:
                    # never silent in the log: a dropped height would otherwise vanish
                    # with no trace and leave the place lift at 0.
                    sys.stderr.write(
                        "openbot_gateway: pick height_m=%r DROPPED (outside "
                        "(0, %.1f]); the engine picks with no height\n"
                        % (height_m, PICK_HEIGHT_M_MAX))
                    sys.stderr.flush()
            return ordered
        if verb == "place":
            # a place can name an explicit target x y z (the engine's coordinate
            # place) or be release-only: [] comes back only when no coordinate is
            # present. z is taken exactly as given; type and finite guarding live in
            # _do_place, so a malformed coordinate is refused there, never stripped
            # down to a release drop.
            ordered = []
            for key in ("x", "y", "z"):
                if args.get(key) is not None:
                    ordered.append(args[key])
            # an optional 4th slot: object height for the engine's place-time extra
            # lift. only meaningful with coordinates: 'place' alone is release-only and
            # a lone token is a parse error. same gate as the pick slot, refusing
            # silently: an out-of-range height becomes 'unknown' (no extra lift), never
            # a block.
            if len(ordered) == 3:
                h_place = args.get("height_m")
                if h_place is not None:
                    try:
                        hpf = float(h_place)
                    except (TypeError, ValueError):
                        hpf = float("nan")
                    if math.isfinite(hpf) and 0.0 < hpf <= PICK_HEIGHT_M_MAX:
                        ordered.append(h_place)
            return ordered
        if verb == "navigate":
            return [str(args.get("target") or "")]
        if verb == "turn_around":
            # an optional label; the operator's ask is the bare sweep, so absent
            # means 'person' (COCO index 0, in the perception TRACKED_CLASSES
            # allowlist, since the walk FSM matches labels as exact strings).
            return [str(args.get("target") or "person")]
        if verb == "open_hand":
            return [str(args.get("side") or "right")]
        if verb == "close_hand":
            line = [str(args.get("side") or "right")]
            params = args.get("params")
            if isinstance(params, list):
                line.extend(str(p) for p in params)
            return line
        if verb == "cmd_set":
            off = args.get("off")
            if off is not None:
                return [str(off)]
            line = [str(args.get(k)) for k in ("vx", "vy", "omega")]
            if args.get("dur") is not None:
                line.append(str(args["dur"]))
            return line
        if verb in ("writer", "grip", "engine"):
            line = args.get("line")
            return [str(line)] if line is not None else []
        # query_user / notify_user named -> positional: brain_client sends
        # {'question': ...} / {'message': ...}, and dispatch consumes [arg0].
        if verb == "query_user":
            return [str(args.get("question") or "")]
        if verb == "notify_user":
            return [str(args.get("message") or "")]
        if verb in ("observe", "status", "ping"):
            return []
        if verb == "yolo_classes":
            # the vocabulary request goes as one JSON token, validated by
            # _do_yolo_classes rather than here, so the shaper stays thin. keys pass
            # straight through: an absent `op` is not defaulted, so a malformed request
            # is refused, not guessed at.
            req: Dict[str, Any] = {"op": args.get("op"),
                                   "classes": args.get("classes", [])}
            if args.get("min_conf") is not None:
                req["min_conf"] = args.get("min_conf")
            return [json.dumps(req)]
        if verb == "container":
            return []                     # a query, so no arguments
        if verb == "home":
            # one JSON token here, validated by _do_home
            hreq = dict((k, args.get(k)) for k in ("op", "why")
                        if args.get(k) is not None)
            return [json.dumps(hreq)]
        if verb == "return_home":
            return []                     # takes no arguments
        if verb == "approach_person":
            # one optional JSON token, validated by _do_approach_person
            areq = dict((k, args.get(k)) for k in ("x", "y")
                        if args.get(k) is not None)
            return [json.dumps(areq)] if areq else []
        if verb == "aim_turn":
            # one JSON token, validated by _do_aim_turn
            return [json.dumps(dict((k, args.get(k)) for k in
                                    ("label", "x", "y", "shift_m")
                                    if args.get(k) is not None))]
        if verb == "step_back":
            # one optional JSON token, validated by _do_step_back
            sreq = dict((k, args.get(k)) for k in ("m",)
                        if args.get(k) is not None)
            return [json.dumps(sreq)] if sreq else []
        if verb == "creep":
            # one JSON token, validated by _creep_parse
            creq = dict((k, args.get(k)) for k in ("label", "iid", "x", "y",
                                                "z", "w", "dd", "h", "dry")
                        if args.get(k) is not None)
            return [json.dumps(creq)]
        if verb == "pc_rim":
            # one JSON token, validated by _do_pc_rim
            preq = dict((k, args.get(k)) for k in ("label", "x", "y", "ts")
                        if args.get(k) is not None)
            return [json.dumps(preq)]
        if verb == "yolo_strict":
            # one JSON token, validated by _do_yolo_strict
            sreq = dict((k, args.get(k)) for k in ("op", "labels", "owner", "hold",
                                                   "wait_s")
                        if args.get(k) is not None)
            return [json.dumps(sreq)]
        return []

    def _resolve_pick_orientation(self, args: dict, ordered: List) -> str:
        """Concrete 'upright'|'horizontal' orientation token for a shaped pick.

        Gateway-side fallback (the brain lane binds the detection's
        orientation in tools.prepare and forwards it concretely through
        GatewayCallerBackend.pick).  An explicit upright/horizontal value
        passes through untouched; 'auto' or an absent/unknown value resolves
        to the freshest detection whose stored point sits within
        PICK_ORIENTATION_MATCH_M of the commanded x/y/z — so a lying bottle
        commanded with 'auto' is picked horizontal, never silently upright.
        No fresh observation, no nearby detection, or non-numeric
        coordinates keep the safe 'upright' default (the engine's own
        default — never a guess at a horizontal approach).
        """
        raw = args.get("orientation")
        if isinstance(raw, str) and raw.strip().lower() in ("upright", "horizontal"):
            return raw.strip().lower()
        # 'auto', absent, or unknown resolves against the scene, if that scene is
        # fresh. a stale or empty detection list never picks an approach.
        if len(ordered) < 3:
            return "upright"
        try:
            px, py, pz = float(ordered[0]), float(ordered[1]), float(ordered[2])
        except (TypeError, ValueError):
            return "upright"
        if not all(map(math.isfinite, (px, py, pz))):
            return "upright"
        with self._obs_lock:
            obs = self._assembler.observation()
        if not obs.fresh or not obs.detections:
            return "upright"
        best = "upright"
        best_d2: Optional[float] = None
        for det in obs.detections:
            d2 = (det.x - px) ** 2 + (det.y - py) ** 2 + (det.z - pz) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                ori = getattr(det, "orientation", "upright")
                best = ori if ori in ("upright", "horizontal") else "upright"
        if best_d2 is not None and best_d2 <= PICK_ORIENTATION_MATCH_M ** 2:
            return best
        return "upright"

    # ── per-verb implementations (one block each) ────────────────────────────

    @staticmethod
    def _parse_pick_args(args: List) -> Tuple[Optional[str], Optional[str]]:
        """Parse pick args to extract orientation and height_m.

        Returns (orientation, height_m) where both may be None if not found.
        """
        orientation = None
        height_m = None
        for i, arg in enumerate(args):
            arg_str = str(arg).strip().lower()
            if arg_str in ("upright", "horizontal"):
                orientation = arg_str
                # height_m is expected to be the next non-orientation arg after orientation
                if i + 1 < len(args):
                    try:
                        height_m = str(args[i + 1])
                        float(height_m)  # make sure it is actually a number
                        break
                    except (ValueError, TypeError):
                        pass
        return orientation or "upright", height_m

    def _note_normals(self, rows: List[dict], now: float) -> None:
        """B8k: keep each instance's newest VALID normal (finite, |n| within
        PICK_NORMAL_NORM_TOL of 1).  Caller holds ``_obs_lock``."""
        store = getattr(self, "_obj_normal", None)
        if store is None:
            return
        for row in rows:
            iid = row.get("instance_id")
            if iid is None:
                continue
            try:
                n = [float(row.get(k)) for k in ("nx", "ny", "nz")]
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in n):
                continue
            norm = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            if abs(norm - 1.0) > PICK_NORMAL_NORM_TOL:
                continue
            store[iid] = (now, n[0] / norm, n[1] / norm, n[2] / norm)
        for iid in [k for k, v in store.items()
                    if now - v[0] > RADIUS_PARK_WINDOW_S]:
            del store[iid]

    def _with_pick_normals(self, args: List, x: float, y: float,
                           z: float) -> List:
        """B8k: ``args`` with the target's normal inserted after x y z, or
        ``args`` unchanged (switch off, a caller-supplied normal, no matching
        row / instance id, a stale or missing normal).  Never raises."""
        try:
            if os.environ.get("OPENBOT_PICK_NORMALS", "1").strip() == "0":
                return args
            if len(args) >= 6 and all(
                    isinstance(a, (int, float)) and not isinstance(a, bool)
                    for a in args[3:6]):
                return args                      # the caller's own normal, untouched
            store = getattr(self, "_obj_normal", None)
            if not store:
                return args
            with self._obs_lock:
                obs = self._assembler.observation()
            best, best_d2 = None, None
            for det in (obs.detections or []):
                d2 = (det.x - x) ** 2 + (det.y - y) ** 2 + (det.z - z) ** 2
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = det, d2
            if best is None or best_d2 > RADIUS_PARK_MATCH_M ** 2:
                return args
            iid = getattr(best, "instance_id", None)
            with self._obs_lock:
                rec = store.get(iid) if iid is not None else None
            if rec is None or time.time() - rec[0] > PICK_NORMAL_MAX_AGE_S:
                return args
            return (list(args[:3]) + [round(rec[1], 4), round(rec[2], 4),
                                      round(rec[3], 4)] + list(args[3:]))
        except Exception:  # noqa: BLE001 — a normal is optional, never a refusal
            return args

    def _calc_pick_x_off(self, x: float, y: float, z: float,
                         orientation: str) -> Tuple[Optional[str], str]:
        """Calculate x_off for radius_park, or return fallback reason.

        Returns (x_off_str, radius_park_token):
        - x_off_str: the trailing x_off token (rounded to 4 decimals) to append
          after height_m, or None if conditions are not met
        - radius_park_token: text to append to the reply ('fallback(...)',
          or the full 'r_mean=...' token if x_off was calculated); empty string
          if switch OFF or orientation not upright (no token added for
          byte-identity when OFF or for horizontal/left picks)
        """
        if not self._radius_park_on:
            return None, ""
        if orientation != "upright":
            return None, ""  # no token when horizontal or not upright
        # caller reads side; this assumes the right hand
        # find the matching detection, same fresh-match logic
        with self._obs_lock:
            obs = self._assembler.observation()
        if not obs.fresh or not obs.detections:
            return None, "; radius_park=fallback(no_detection)"
        best_det = None
        best_d2: Optional[float] = None
        for det in obs.detections:
            d2 = (det.x - x) ** 2 + (det.y - y) ** 2 + (det.z - z) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_det = det
        # the brain sends the target row's own coordinates, so the row that
        # measured it sits within a few cm; a looser gate (0.30 m) could grab a
        # neighbour's radius.
        if best_det is None or best_d2 is None or \
                best_d2 > RADIUS_PARK_MATCH_M ** 2:
            return None, "; radius_park=fallback(not_found)"
        iid = getattr(best_det, 'instance_id', None)
        if iid is None:
            return None, "; radius_park=fallback(no_instance_id)"
        with self._obs_lock:
            history = self._obj_radius_history.get(iid, [])
        if len(history) < 3:
            return None, "; radius_park=fallback(few)"
        # only samples from the last RADIUS_PARK_WINDOW_S count, since an object
        # last measured ages ago is not good evidence for this pick.
        now = time.time()
        samples = [r for (t, r, q) in history
                   if q >= 0.6 and now - t <= RADIUS_PARK_WINDOW_S]
        if len(samples) < 3:
            return None, "; radius_park=fallback(few)"
        r_min = min(samples)
        r_max = max(samples)
        if r_max - r_min > 0.020:
            return None, "; radius_park=fallback(spread)"
        # use the mean of the gated samples for radius estimation.
        r_mean = sum(samples) / len(samples)
        r_eff = r_mean + self._radius_park_margin_m
        x_off = -(r_eff + 0.065)
        x_off_rounded = round(x_off, 4)
        n = len(samples)
        spread = r_max - r_min
        # if the clamp cut the requested margin, the token names the request
        # (margin_req=) ahead of the margin used; unclamped, the token stays
        # byte-identical to the plain one.
        _req = getattr(self, '_radius_park_margin_req_m',
                       self._radius_park_margin_m)
        _req_tok = ("margin_req=%.3f," % _req
                    if abs(_req - self._radius_park_margin_m) > 5e-7 else "")
        token = ("; radius_park=r_mean=%.3f,n=%d,spread=%.3f,%smargin=%.3f,"
                 "x_off=%.4f" %
                 (r_mean, n, spread, _req_tok, self._radius_park_margin_m,
                  x_off_rounded))
        return "%.4f" % x_off_rounded, token

    def _place_release_note(self) -> str:
        """B8i (10-09 wire 46): '' or the newest DONE place release within
        PLACE_RELEASE_NOTE_S, appended to the unheld refusal (the pinned
        prefix 'ERROR:nothing verified held - cannot place' is unchanged)."""
        rel = dict(getattr(self, '_last_place_release', {}) or {})
        now = time.time()
        best = None
        for s, ts in rel.items():
            age = now - ts
            if 0.0 <= age <= PLACE_RELEASE_NOTE_S and (
                    best is None or age < best[1]):
                best = (s, age)
        if best is None:
            return ""
        return ("; the %s hand already released its object %ds ago (the "
                "previous place completed) - do not place again; tell the "
                "operator it was released" % (best[0], int(round(best[1]))))

    def _do_pick(self, verb: str, args: List) -> Tuple[str, Optional[str]]:
        side = "left" if verb == "pick_left" else "right"
        if len(args) < 3:
            return ("ERROR:%s needs x y z [nx ny nz] [orientation] [height_m]"
                    % verb), None
        try:
            x, y, z = float(args[0]), float(args[1]), float(args[2])
        except (TypeError, ValueError):
            return "ERROR:pick x y z must be numbers", None
        # NaN and Inf must never reach the floor refusal: NaN < -0.35 is false,
        # so a NaN z would bypass the tabletop block and a phantom pick would run.
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return "ERROR:pick x y z must be finite numbers", None
        # capability block: pick is tabletop-only, so a floor pick is refused
        # honestly. the bound is a torso-frame height (z=0 at the torso link, the
        # frame perception and the engine use).
        if z < self._pick_min_z_m:
            return ("ERROR:pick capability is TABLETOP-ONLY (z >= %.2f m "
                    "torso frame, got z=%.3f) — floor/ground pick refused, "
                    "no ground-pick skill exists"
                    % (self._pick_min_z_m, z)), None
        # a pick touches the hand, so it ends the side's open trust for the causal
        # open reference: the hand may be closed, and reads do not qualify
        # again until the next open command the gateway dispatches. a new pick also
        # kills any prior verified-hold record, which belongs to this pick.
        with self._obs_lock:
            self._assembler.note_hand_command(side)
            self._assembler.set_active_side(side)  # obs.grip = the hand the pick used
            self._last_verified_grasp.pop(side, None)
            getattr(self, '_last_place_release', {}).pop(side, None)  # drop the place record
        args = self._with_pick_normals(args, x, y, z)      # splice in the normals
        tail = " ".join(str(a) for a in args[3:])
        cmd = verb + " " + " ".join(str(a) for a in args[:3]) + \
            (" " + tail if tail else "")
        # work out radius_park x_off when it applies (upright right hand only)
        x_off_token_str = None
        radius_park_token = ""
        if side == "right":
            # parse the args to pull out orientation and height_m
            orientation, height_m = self._parse_pick_args(args)
            x_off_token, radius_park_token = self._calc_pick_x_off(
                x, y, z, orientation)
            # a radius_park token with no height_m cannot send x_off, so swap in the
            # fallback token; the reply then reflects reality.
            if radius_park_token and height_m is None:
                radius_park_token = "; radius_park=fallback(no_height_m)"
                x_off_token = None
            # append x_off only when: switch on, x_off calculated, height_m present
            if x_off_token is not None and height_m is not None:
                cmd += " " + x_off_token
                x_off_token_str = x_off_token
        # inert unless `_b5_on`. dispatch_ts is stamped right before the engine
        # call, where the detections re-observe window starts.
        b5 = getattr(self, "_b5_on", False) is True
        dispatch_ts = self._b5_clock() if b5 else 0.0
        reply = self.engine_verb(cmd)
        if not _is_engine_reply(reply):
            # not a verdict: the engine wrote something that is not a reply (the
            # Ipopt banner) or the stream is desynced. saying 'the pick didn't start'
            # would guess pessimistically, while silence would be the optimistic lie
            # this guard exists to stop. report UNKNOWN and let the operator re-observe.
            return ("ERROR:pick engine reply was not a command reply (%r) — "
                    "the pick's outcome is UNKNOWN; re-observe before retrying"
                    % reply.strip()[:120]), None
        if reply.strip().startswith("CANCELLED:"):
            # a cancelled pick is never a completed one (`CANCELLED:arm_move`). it
            # would slip through the negative test below and come back 'DONE pick
            # executed', and with the lane up _last_verified_grasp got stamped for a
            # hand the engine had frozen. so: no motion wait, no lane probe, no hold
            # record (the pop above cleared any earlier one). the reply leads with
            # ERROR so _shape_reply renders ok:false, and the echoed engine text is
            # defanged so it cannot add a second ERROR: token.
            where = _cancel_phase(reply)
            verdict = _cancelled_verdict("pick", where)
            self._emit_verdict(verdict)
            cancelled_reply = (
                "ERROR:cancelled_by_operator pick cancelled at phase %s "
                "by the operator; the arm was frozen in place; NOT a "
                "completed pick; engine_reply=%s"
                % (where, _defang_error_token(reply.strip())[:120]))
            if b5:
                # the engine froze the arm right where the cancel found it, and frozen in
                # contact that is a thermal stall. park before replying. the token precedes
                # `engine_reply=`; the verdict above is unchanged.
                cancelled_reply = _insert_before_engine_reply(
                    cancelled_reply,
                    self._post_task_arm_sweep("pick", "cancelled"))
            return cancelled_reply, verdict
        # _verify judges the pick, so give it the reply without the arm_after=
        # token; the verdict and its Reason stay what they were. None means use the
        # reply itself.
        verify_reply: Optional[str] = None
        if b5 and reply.startswith("ERROR") \
                and not reply.startswith("ERROR:pick engine not configured"):
            # the engine ran and did not end DONE. a lift-pin abort answered
            # ERROR:grasp_failed with no hold_rc off and no READY move, so the writer
            # PD-held the blocked arm hot. the sweep reads q and parks any arm left
            # outside SAFE_ARM_POSES. a failed send parks through the writer only; a
            # reply timeout or dropped connection first tries the engine's queued
            # `park` (ENGINE_SEND_FAILED_PREFIX). the pre-engine refusals above and
            # 'engine not configured' both mean no engine ever ran.
            verify_reply = reply
            reply = _insert_before_engine_reply(
                reply, self._post_task_arm_sweep(
                    "pick", "error",
                    prefer_writer=reply.startswith(
                        ENGINE_SEND_FAILED_PREFIX)))
        if not reply.startswith("ERROR"):
            # motion-completion wait: engine_verb hands back DONE about 3 s in, while
            # the daemon keeps moving the arm for ~30 s more. returning then hands the
            # brain's verifier a pre-motion scene where the target is still on the
            # table, so verify_pick can only answer in_progress and the model never
            # reaches `place`.
            motion = self._await_pick_daemon()
            # pick grasp law: on a real pick the engine closes over its own socket, so
            # the gateway never sees the close_adaptive verdict. with the palm DDS lane
            # up the engine verifies its own grasp and would fail ERROR:grasp_failed on
            # an air-grab; in blind mode it skips that and returns DONE anyway. so the
            # only lane probe the gateway can see is a fresh 'state [side]' poll of the
            # available flag: poll right away and stamp it.
            lane_state, state_ts = self._poll_lane_state(side)
            if lane_state == "up":
                # engine DONE on an up lane means an engine-verified grasp, so record the
                # verified hold for the place verifier.
                with self._obs_lock:
                    self._last_verified_grasp[side] = time.time()
            # when the detections channel goes silent, re-observe (bounded) before the
            # verdict, carrying honest liveness evidence. the alive path sends ''.
            det_toks = self._pick_det_reobserve(dispatch_ts) if b5 else ""
            if det_toks and lane_state == "up":
                # the lane poll above ran before the <= 4 s detections wait, so the stamp
                # the verifier re-checks against GRIP_STATE_MAX_AGE_S (5.0 s) is ~4 s old,
                # leaving only 0.9 s of slack. poll once more and stamp that read. an 'up'
                # reading never gets downgraded, and a lane that is not up never gets
                # upgraded after the wait.
                lane2, ts2 = self._poll_lane_state(side)
                if lane2 == "up":
                    state_ts = ts2
            reply = ("DONE pick executed; lane_state=%s; grip_state_ts=%.6f; "
                     "motion=%s%s; engine_reply=%s"
                     % (lane_state, state_ts, motion, det_toks, reply))
        # give _verify the point the arm is sent to (on the descent lane that is
        # z_engine, the torso frame at the descended body height where post-pick
        # detections are taken), so verify_pick's in-hand displacement check
        # (_parse_pick_point) can tell the bottle in hand from the one still on the
        # table. lane armed only, a thread-local slot never in the reply text and
        # never in _verify's signature, cleared in `finally` so it cannot taint a
        # later verdict on this thread.
        slot = GatewayNode._PICK_POINT_TLS
        slot.point = ((x, y, z) if getattr(self, "_low_lane", None) is not None
                      else None)
        try:
            # append the radius_park token when it applies
            if radius_park_token and not reply.startswith("ERROR"):
                # only for DONE replies
                reply += radius_park_token
            elif radius_park_token and reply.startswith("ERROR"):
                # ERROR replies still get the token before engine_reply
                # (minimal or empty for errors)
                reply += radius_park_token
            if verify_reply is not None:
                # the verdict judges the pick here, not the park token
                return reply, self._verify("pick", verify_reply)
            return reply, self._verify("pick", reply)
        finally:
            slot.point = None

    #: a per-thread pick point for `_verify`. written only by `_do_pick`
    #: around its own `_verify` call, and None everywhere else.
    _PICK_POINT_TLS = threading.local()

    def _await_pick_daemon(self) -> str:
        """Block until the pick daemon's MOTION ENDS.  Bounded; never raises.

        WHY THIS EXISTS (MEASURED 2026-09-27, `--log-level DEBUG` wire trace
        against `pick.log`, same run)::

            21:36:53,613  model calls pick
            21:36:56,647  function_call_output SENT -- the verdict is taken HERE
            21:37:21,029  engine STILL closing the hand
            21:37:22,341  engine grasp verdict (hit_obstacle -> light_object)
            21:37:29,389  T5 step 5: right arm -> CARRY_TUCK

        The verdict precedes the engine's OWN grasp verdict by 25.7 s.  At
        t+3 s the bottle is untouched on the table, so `_target_present`
        returns True and `verify_pick` takes its `present is True` branch
        ("re-observe; target still visible") => `in_progress`, every time.
        The brain's own re-observe budget cannot bridge the gap either: it is
        `_MAX_RE_OBSERVES` 2 x `_RE_OBSERVE_SETTLE_S` 1.4 = 2.8 s, an order of
        magnitude short of a ~30 s motion.

        NOT the reply timeout: `PICK_REPLY_TIMEOUT_S` is ALREADY 300 s
        (gateway_node.py:861).  The gateway waits 300 s for the engine to
        ANSWER -- the engine simply answers early and runs the motion in its
        own daemon.  So the missing wait is for the MOTION, not the reply.

        THE PREDICATE IS THE BUSY EDGE, NOT "not busy".  Between two
        consecutive SUCCESSFUL picks the daemon emits
        busy(held=True) -> holding(held=True) and NEVER a held:false (see
        _on_pick_state).  A bare "state != busy" read would therefore be
        satisfied by the PREVIOUS cycle's carry-over frame and return
        instantly -- the stale-evidence trap this fix exists to close, one
        level down.  So we require an observed ``busy`` frame FIRST (cycle
        START) and only then a later non-``busy`` frame (cycle END).

        DEGRADES SAFELY.  A daemon that never publishes leaves the record None
        or never ``busy``; the wait then runs to its deadline and returns
        ``no_busy_signal`` -- the pre-fix behaviour.  It can therefore never
        HANG a pick, and never MANUFACTURES a completion the daemon did not
        report.  A ``timeout`` token means the verdict may still be stale; it
        is reported, not hidden.

        The returned token goes into the tool result, so it must NEVER contain
        the substring ``ERROR:`` -- `_shape_reply` flips the envelope to
        ok:false on that substring anywhere in the reply, and the verifier's
        _classify_reply reads an embedded ``ERROR:`` as a failure.
        """
        with self._obs_lock:
            lane_live = self._pick_state_seen
        if not lane_live:
            # never a single /pick_state frame since start: there is no daemon lane on
            # this robot, so there is nothing to wait for. check that before the clock
            # starts, so a lane without a daemon pays nothing for this, not even the
            # start window.
            return "no_daemon_lane"
        t0 = time.time()
        start_deadline = t0 + self._pick_daemon_start_s
        deadline = t0 + self._pick_daemon_wait_s
        saw_busy = False
        last_state = None
        while True:
            now = time.time()
            with self._obs_lock:
                rec = self._pick_state_hold
                rec = dict(rec) if rec is not None else None
            if rec is not None:
                last_state = rec.get("state")
                # freshness and daemon_alive are the same conjuncts the hold predicate uses
                # (_daemon_hold_record_unlocked): a stale frame or one from a dead daemon is
                # evidence of nothing, whichever way it points.
                if (now - rec["ts"]) <= self._pick_state_max_age_s \
                        and rec.get("alive"):
                    if last_state == PICK_STATE_BUSY:
                        saw_busy = True
                    elif saw_busy:
                        return "cycle_ended@%.1fs" % (now - t0)
            if not saw_busy and now >= start_deadline:
                # the topic is live but no cycle began, so this pick is not daemon-driven
                # (or the daemon is wedged). return early instead of burning the motion
                # budget on an event that is not coming.
                return "no_busy_signal (last_state=%r)" % (last_state,)
            if now >= deadline:
                return "timeout_after_busy@%.1fs" % (now - t0)
            time.sleep(PICK_DAEMON_POLL_S)

    def _await_walk_reached(self, label: str, epoch_ts: float) -> str:
        """Block until the walk FSM ARRIVES at ``label`` (or gives up).  Bounded.

        WHY THIS EXISTS (run8, 2026-09-28).  ``_do_navigate`` published
        /navigate_target and returned IMMEDIATELY — unlike ``_do_pick``, which
        waits for its daemon.  The reply snapshot was taken microseconds after
        the publish, so ``PHASES_SEEN`` was always EMPTY and
        ``verifier.verify_navigate`` could only answer ``in_progress``
        (:1533-1536).  The walk itself was fine — every physical walk in run8
        landed SEARCHING -> DETECTED -> APPROACHING -> REACHED — but the brain
        was told nothing, so it re-called navigate again and again: SIX tool
        calls for THREE arrivals (voice_run.log:39,168,478,495,548,561).
        "The walk did not report back reach status" is exactly this.

        NOT the reply timeout: the navigate Tool bound is already 180 s
        (tools.py) and ``GATEWAY_CALL_TIMEOUT_S`` is 150 s.  The gateway does
        not wait for the FSM to ANSWER — the FSM answers every dispatch — it
        never waited for the FSM to ARRIVE.  So the missing wait is for the
        MOTION, not for the reply.

        THE PREDICATE MIRRORS THE VERIFIER, DELIBERATELY.  A ``reached@`` token
        is a claim that ``verify_navigate`` will agree with, so the conjuncts
        are copied from it rather than invented: an IN-EPOCH REACHED whose
        payload label matches the request (fix 1 / check (f)) AND at least one
        IN-EPOCH SEARCHING/APPROACHING (the transition guard / check (d)).
        A hold-REACHED carried over from a PREVIOUS arrival satisfies neither
        and therefore cannot end this wait — the stale-evidence trap, one
        level down.

        THE LOAD-BEARING TIMING CONSTRAINT.  ``verify_navigate`` reads the
        LIVE ``obs.walk.phase`` (:1537-1541), and the FSM holds REACHED for
        only REACHED_HOLD_S = 5 s.  So this must RETURN ON SIGHT: poll fast,
        return the instant the predicate holds, and never sleep after it does.
        The REACHED is read from the in-epoch phase RECORD; a sleep after the
        predicate would let the hold expire and hand the verifier an IDLE.

        DEGRADES SAFELY, LIKE THE PICK WAIT.  No walk lane, no in-epoch frame,
        or an FSM that stops without arriving all return a token and stop
        waiting — none of which is a refusal.  Every token is reported, none
        is hidden, and (like ``_await_pick_daemon``) NO token may contain the
        substring ``ERROR:``: ``_shape_reply`` flips the envelope to ok:false
        on it anywhere in the reply, and the verifier's ``_classify_reply``
        reads an embedded ``ERROR:`` as a failure.
        """
        t0 = time.time()
        start_deadline = t0 + self._nav_start_s
        deadline = t0 + self._nav_wait_s
        want = (label or "").strip().lower()
        while True:
            now = time.time()
            with self._obs_lock:
                cancel_ts = self._cancel_ts
                phases = list(self._phases_seen)
            # the operator stopped the motion. checked first, ahead of every arrival
            # because a stop is not a timeout and must never be reported as one:
            # reporting it as a timeout made the model issue navigate again, restarting
            # the search while the brain's tool call stayed pending and every utterance
            # got dropped.
            # in-epoch scope: only a cancel at or after this dispatch's epoch can end
            # this wait, so an earlier task's cancel can never abort a later navigate.
            # the > 0.0 guard stops the never-cancelled default (0.0) from matching an
            # epoch_ts that is still 0.0.
            if cancel_ts > 0.0 and cancel_ts >= epoch_ts:
                return "cancelled_by_operator@%.1fs" % (now - t0)
            # in-epoch scope. _phases_seen is cleared at dispatch and the callback only
            # appends ts >= epoch_ts, so this filter is belt-and-braces, but it keeps a
            # stale REACHED from ending this wait.
            epoch_phases = [p for p in phases if p[2] >= epoch_ts]
            seen = set(p for p, _l, _t in epoch_phases)

            def _label_hit(phase):
                return any(
                    p == phase and (l or "").strip().lower() == want
                    for p, l, _t in epoch_phases
                )

            motion = bool({"SEARCHING", "APPROACHING"} & seen)
            if _label_hit("REACHED") and motion:
                return "reached@%.1fs" % (now - t0)
            if _label_hit("NOT_FOUND"):
                return "not_found@%.1fs" % (now - t0)
            if motion and "IDLE" in seen:
                # the FSM moved and then stopped without arriving: it is done with this
                # target and no arrival is coming. consistent with the in-flight rule, where
                # an in-epoch IDLE means 'not working on this'; without it a dead walk would
                # stall the brain for the whole budget.
                return "walk_idle_after_motion@%.1fs" % (now - t0)
            if not epoch_phases and now >= start_deadline:
                # not one in-epoch frame: this dispatch has no walk lane behind it, or the
                # FSM never heard it. pay the start window, not the motion budget, the same
                # early exit shape as the pick wait's no_busy_signal.
                return "no_walk_signal"
            if now >= deadline:
                return "timeout_after_walk@%.1fs" % (now - t0)
            time.sleep(NAV_POLL_S)

    def _poll_lane_state(self, side: str) -> Tuple[str, float]:
        """Fresh ``state <side>`` poll right after a pick motion completes.

        Returns ``(lane_state, ts)`` with lane_state one of
        'up' | 'down' | 'unknown':
          up      — the reply carried ``side`` with available True on a read
                     no older than GRIP_STATE_MAX_AGE_S: the palm DDS lane
                     was UP, so THIS engine DONE was ENGINE-VERIFIED
                     (platform law — available True <=> lane up <=> the
                     engine verified its own grasp);
          down    — the read failed/errored, the reply was malformed or
                     omitted ``side``, or available False: BLIND MODE — the
                     engine DONE carries no grasp evidence;
          unknown — the reply carried ``side`` with joints but NO available
                     flag: availability cannot be determined (the deployed
                     grip_server always emits available; defensive).
        The poll is immediate, so the read is fresh by construction; the
        verifier re-checks ``grip_state_ts`` against its freshness window.
        """
        ts = time.time()
        try:
            line = self.grip_verb("state %s" % side)
            with self._obs_lock:
                # stamp the poll's own ts so the freshness check stays exact: read_ts must
                # equal ts, and this reply carried the side.
                self._assembler.update(grip_line=line, now=ts)
                read_ts = self._assembler.grip_state_read_ts(side)
        except Exception:
            return "down", ts
        if read_ts is None or read_ts != ts:
            # the reply did not carry this side, so the poll got no fresh lane probe
            # for it at all.
            return "down", ts
        try:
            payload = json.loads(line[len("DONE:"):]) if line.startswith("DONE:") else None
            data = payload.get(side) if isinstance(payload, dict) else None
        except (ValueError, TypeError):
            data = None
        if not isinstance(data, dict):
            return "down", ts
        if "available" not in data:
            return "unknown", ts
        return ("up" if data["available"] else "down"), ts

    def _holding_state(self, side: str) -> str:
        """'True' | 'False' — was `side` holding just now?

        The pre-command evidence chain for open (fix 7, round 2): 'True' when
        a live unconsumed close_adaptive 'contact' verdict is on file within
        the window; 'False' otherwise.  Round 3 (fix 3): the PLACE pre-drop
        holding evidence is no longer verdict-based at all — it comes from a
        fresh 'state [side]' poll (see _do_place); this verdict fold survives
        only as the open_hand 'prev_contact' annotation.  Call BEFORE the
        command executes (the command clears the verdicts).
        """
        verdict, v_ts = self._close_verdicts.get(side, (None, 0.0))
        if verdict == "contact" and \
                (time.time() - v_ts) <= self._close_verdict_window_s:
            return "True"
        return "False"

    def _on_pick_state(self, msg) -> None:
        """Fold the pick_node daemon's /pick_state into the hold evidence.

        Never raises: an exception escaping a subscription callback would take
        the lane silently deaf, which is precisely the failure this fold
        exists to remove.  A payload that does not parse CLEARS the record
        instead of retaining the last good one — an unreadable daemon is not
        evidence of a hold.
        """
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                raise ValueError("pick_state payload is not an object")
            held = data.get("held") is True
            alive = data.get("daemon_alive") is True
            side = data.get("side")
            state = str(data.get("state") or "")
            parsed = True
        except Exception:
            held, alive, side, state, parsed = False, False, None, "", False
        with self._obs_lock:
            # latched before the parse outcome is consulted: the flag means 'this topic
            # is alive on this robot', and one unreadable frame does not disprove it.
            # _await_pick_daemon reads it to decide whether a wait is worth it.
            self._pick_state_seen = True
            self._pick_state_hold = {
                "held": held, "alive": alive, "ts": time.time(),
                "side": side if side in ("right", "left") else None,
                "state": state,
            }
            if parsed and (not held or state == PICK_STATE_BUSY):
                # a hold epoch has ended, so any claim already consumed is not the current
                # one: a later held:true is a new hold and can be trusted again. two
                # endings:
                #   * held parsed false, the daemon's own open or error;
                #   * state == 'busy', _execute_pick's first act, a new cycle starting. it
                #     does not touch `held`, so two successful picks emit busy(held=True)
                #     then holding(held=True) and never a held:false. keying the release
                #     off `held` alone would fire the tier once per daemon lifetime and
                #     re-refuse every later place.
                # `parsed` gates both: the except path synthesises held=False, and taking
                # that for an epoch end would let one garbage message re-arm a hold already
                # released, giving a spurious second release.
                self._pick_state_consumed = False

    def _daemon_hold_record_unlocked(self) -> Optional[Dict[str, object]]:
        """The evidence predicate.  THE CALLER HOLDS ``_obs_lock``.

        The whole predicate lives here so the TAKE in _do_place can re-run it
        and then null the slot INSIDE ONE LOCK HOLD: read-then-take as two
        separate critical sections is a race a concurrent caller (or the
        daemon's own 5 Hz publish) walks straight through.

        Evidence requires ALL FIVE of: a /pick_state received within
        _pick_state_max_age_s, ``held`` true, ``daemon_alive`` true, the state
        NOT ``busy``, and the hold NOT already consumed by a release this
        gateway performed.  Any one missing returns None, so the tiers built on
        this can only ever ADD a hold the daemon is asserting RIGHT NOW: they
        never keep a dead daemon's claim alive, and — because `held` is sticky
        on the daemon's side and the gateway never reports its releases back —
        never re-arm a hold this gateway has already placed or opened.  That
        last conjunct is the whole reason `_pick_state_consumed` exists;
        without it the clear sites below are undone by the daemon's next 5 Hz
        publish within 200 ms and a second place ships a release to the engine
        on an already-released hand.

        The ``busy`` conjunct is the 2026-09-21 round-6 fix.  ``busy`` is
        _execute_pick's FIRST act and it does NOT touch ``held``, so the daemon
        emits busy(held=True) -> holding(held=True) between two successful
        picks.  A record in that state describes a PICK IN PROGRESS: the hand
        is being driven onto a NEW payload, and a place riding it would release
        a hold that is being re-made.  It still ENDS the previous epoch in
        _on_pick_state (consumed -> False) — that is unchanged, and it is what
        makes the NEXT hold placeable — but a busy record is itself never
        evidence.

        Split out of _daemon_held_side so the MEASURED tier can share the gate
        without also inheriting the declaration: the shipped daemon publishes
        no ``side``, so a gate welded to ``rec["side"]`` would be permanently
        None and the measurement below would never run.
        """
        rec = self._pick_state_hold
        consumed = self._pick_state_consumed
        if rec is None or consumed:
            return None
        if not (rec["held"] and rec["alive"]):
            return None
        if rec["state"] == PICK_STATE_BUSY:
            return None
        if (time.time() - rec["ts"]) > self._pick_state_max_age_s:
            return None
        return rec

    def _daemon_hold_record(self) -> Optional[Dict[str, object]]:
        """The evidence predicate under the lock — see _unlocked for the law."""
        with self._obs_lock:
            return self._daemon_hold_record_unlocked()

    def _daemon_held_side(self) -> Optional[str]:
        """The side the daemon DECLARED, or None when it declared none.

        Last resort only (see _measured_held_side, which outranks it).  The
        shipped daemon's payload has no ``side`` key at all, so on this robot
        this returns None at every instant and the measured tier is what
        actually resolves a daemon-lane place.  It is kept because a daemon
        that DOES name a hand is better evidence than one the gateway infers.
        """
        rec = self._daemon_hold_record()
        if rec is None:
            return None
        return rec["side"]

    def _measured_closure(self, side: str) -> Optional[float]:
        """Mean Dex3 closure fraction of ``side``, or None if unscoreable.

        ``eff = (q - OPEN) / (CLOSE - OPEN)`` per joint, averaged over the
        joints that carry a usable span AND are in servo mode.  This is the
        SAME fraction the engine's bounded-lead close and grip_server's
        ``clamped_close_target`` speak in — the presets are duplicated at the
        top of this file because grip_server exposes no scoring verb (see the
        HAZARD note there).

        Returns None — never 0.0 — whenever the hand cannot honestly be
        scored: no reply, a non-DONE reply, a malformed envelope, the side
        reported ``available`` False (the lane saying its ``actual`` is stale),
        no ``actual`` vector (the LEFT hand on this robot returns none — it has
        no DDS reader), too few live joints, or no joint with a usable span.
        The distinction matters: 0.0 means "measured, and it is OPEN", which
        is a fact that licences a refusal; None means "not measured", which
        must fall through to the next tier instead.
        """
        try:
            line = self.grip_verb("state %s" % side, CLOSURE_PROBE_TIMEOUT_S)
            if not isinstance(line, str) or not line.startswith("DONE:"):
                return None
            payload = json.loads(line[len("DONE:"):])
            if not isinstance(payload, dict):
                return None
            st = payload.get(side)
            if not isinstance(st, dict):
                return None
            actual = st.get("actual")
            if not isinstance(actual, dict):
                return None
            # the lane's health flag for this side. available=False means the grip
            # server is saying the vector below is stale (its last good read, not a live
            # measurement), so scoring it reads a number the hand is not making. a grip server
            # answering available False can still have its last 'actual' score 0.668,
            # enough to name a holder. the guard sits outside the `mode` branch
            # deliberately: a robot emitting no `mode` would skip a check in there.
            if not st.get("available", True):
                return None
            live = None
            modes = st.get("mode")
            if isinstance(modes, dict):
                live = set()
                for k, v in modes.items():
                    try:
                        if int(v) == 1:
                            live.add(str(k))
                    except (TypeError, ValueError):
                        continue
            o, c = DEX3_OPEN[side], DEX3_CLOSE[side]
            vals = []
            for jid in DEX3_JOINT_IDS:
                key = str(jid)
                q = actual.get(key)
                if q is None:
                    continue
                if live is not None and key not in live:
                    # limp (mode == 0): a disabled motor, so this q is wherever gravity parked
                    # the joint, not where the grasp put it. j3 on a live cup hold reads -0.04
                    # and would pull the mean down.
                    continue
                span = c[jid] - o[jid]
                if abs(span) < MIN_SPAN_RAD:
                    continue
                vals.append((float(q) - o[jid]) / span)
            # one floor, applied to the joints actually scored and applied
            # unconditionally, deliberately not inside the mode branch: a grip server
            # that emits `actual` but no `mode` would skip a floor in there, and a hand
            # scored off two joints could read ~1.0 and license a place into an empty
            # hand.
            # it cannot bind in that regime: with `mode` absent nothing is excluded by
            # the mode gate and every right-hand span exceeds MIN_SPAN_RAD, so len(vals)
            # is always 7 against a floor of 4. it binds only with `mode` reported,
            # where the latch triad leaves just 4. kept because it is load-bearing there
            # and is the only guard against a short `actual` vector.
            if len(vals) < MIN_SCORED_JOINTS:
                return None
            return sum(vals) / len(vals)
        except Exception:
            # a probe must never bring the place path down: an unreachable or wedged
            # grip server is "unmeasured", and the caller handles that.
            return None

    def _measured_held_side(self) -> Optional[str]:
        """Which hand is HOLDING, measured — or None when that cannot be said.

        THE REAL LAW, in order:

        1. The daemon's evidence gate must hold (a FRESH /pick_state with
           `held` AND `daemon_alive` AND not consumed AND not `busy`), or this
           tier is inert.  A measurement is not a hold.
        2. Both hands are scored over the grip socket.  A hand that cannot be
           measured is UNSEEN.  If ANY UNSEEN hand is IN SERVICE — i.e. not in
           DEX3_OUT_OF_SERVICE_SIDES — return None: an in-service hand that
           cannot be measured could be the holder, and this tier does not
           guess.  This is the fix for the round-6 defect, where dropping an
           unmeasurable hand out of `eff` and answering from the remainder made
           "one hand closed, the other UNMEASURABLE" indistinguishable from
           "one closed, the other measured NOT-CLOSED", and — on this robot,
           whose left Dex3 never measures — reduced the tier to a de-facto
           declared constant 'right' whenever the right read closed.
        3. If every unseen hand is out of service (its absence is EXCUSED —
           see DEX3_OUT_OF_SERVICE_SIDES; its PRESENCE is still evidence and it
           is scored like any hand), decide from the measured hands:
           * BOTH meaningfully closed -> the hand that closed LESS is the one
             holding something.  A hand wrapped around an object is stopped by
             it and reads a smaller fraction than an empty hand that closed all
             the way onto its own stop;
           * exactly ONE closed -> that hand;
           * NEITHER closed, or NOTHING measurable -> None, i.e. no evidence,
             which leaves the caller's refusal standing.

        Note the asymmetry with _daemon_held_side: a declared side is taken on
        trust, whereas this is a measurement the gateway makes itself.  The
        measurement outranks the declaration because a declaration reflects
        the DAEMON's belief about its own pick, while this reflects the hand's
        actual pose at the instant of the place.
        """
        if self._daemon_hold_record() is None:
            return None
        eff = {}
        for side in ("right", "left"):
            e = self._measured_closure(side)
            if e is not None:
                eff[side] = e
        # an unseen in-service hand forbids an answer. read the tuple by name (a
        # runtime lookup, not a captured default), so an operator can empty it and
        # restore the sound rule that every hand must measure.
        if any(s not in DEX3_OUT_OF_SERVICE_SIDES
               for s in ("right", "left") if s not in eff):
            return None
        closed = {s: e for s, e in eff.items() if e >= self._closed_eff_min}
        if not closed:
            return None
        if len(closed) == 1:
            return next(iter(closed))
        return min(closed, key=closed.get)

    def _clear_close_verdicts(self, sides: Tuple[str, ...]) -> None:
        """Drop close_adaptive verdicts (gateway tracking + assembler fold).

        Called when a commanded open/close invalidates prior grasp evidence
        for the affected sides, so GripState.contact must not linger.
        """
        with self._obs_lock:
            for s in sides:
                if s in ("right", "left"):
                    self._close_verdicts.pop(s, None)
                    self._assembler.clear_close_result(s)

    def _do_place(self, args: List) -> Tuple[str, Optional[str]]:
        # a place can carry an explicit target x y z (the engine's coordinate
        # 'place x y z') or be release-only (no coords, bare 'place'). z is taken as
        # given, and an optional 4th token carries the object height for the
        # place-time extra lift. there is no orientation token for place.
        # type and finite handling mirror _do_pick, so a malformed coordinate is
        # refused, never stripped (a stripped coordinate must not collapse into a
        # release drop at the wrong place). tokens beyond x y z are dropped.
        coords: Optional[List[float]] = None
        place_height: Optional[float] = None
        if args:
            if len(args) < 3:
                return ("ERROR:place needs x y z when a coordinate is given, "
                        "else release-only (no args)"), None
            try:
                x, y, z = (float(args[0]), float(args[1]), float(args[2]))
            except (TypeError, ValueError):
                return "ERROR:place x y z must be numbers", None
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                return "ERROR:place x y z must be finite numbers", None
            coords = [x, y, z]
            # the optional 4th token: object height for the engine's place-time extra
            # lift. unlike a malformed coordinate, which is refused because a stripped
            # coordinate becomes a release drop at the wrong place, a malformed height
            # is not a misplacement: it becomes 'unknown', is dropped, and the place
            # goes ahead with no extra lift. the engine applies the same bound and does
            # the same warn-and-degrade.
            if len(args) >= 4:
                try:
                    hp = float(args[3])
                except (TypeError, ValueError):
                    hp = float("nan")
                if math.isfinite(hp) and 0.0 < hp <= PICK_HEIGHT_M_MAX:
                    place_height = hp
        # held-side determination: which hand will the place release? decided from
        # actual hold evidence, never the active_side hint alone, because
        # active_side is only the last hand the gateway observed and the place
        # schema has no arm argument. order:
        #   (1) exactly one side has an engine-verified grasp on record
        #       (_last_verified_grasp) -> that side;
        #   (2) else, if exactly one side measures holding at this instant
        #       (_holding_state right XOR left) -> that side. measurement outranks
        #       the declaration below, because the daemon's payload carries no
        #       side, so its tier can only assert a declared hand and letting a
        #       declaration override a measured hold releases the wrong hand;
        #   (3) else the daemon lane's measured hold (_measured_held_side: gated on
        #       a fresh /pick_state that has held and daemon_alive and not consumed
        #       and not `busy`; silent when an in-service hand could not be
        #       measured; else the less-closed hand). the daemon's pick never
        #       touches this object's own pick path, so without this tier (1) stays
        #       empty, both hands read false and an unheld refusal fires on a
        #       payload the robot holds. this is the tier that fires here;
        #   (4) else the side the daemon declared (_daemon_held_side), only for a
        #       daemon that names a hand. both daemon tiers sit inside the
        #       refusal's own condition, so they can only change the outcome in the
        #       case that would otherwise be refused;
        #   (5) else the active_side hint, unverified, for back-compat when no hold
        #       evidence can be read;
        #   (6) else right.
        # an unheld place is refused here, before any engine command: no
        # verified-holding side and neither hand measuring holding means the place,
        # especially a release-only bare 'place', would command motion on empty
        # hands for nothing.
        with self._obs_lock:
            verified_sides = [s for s in ("right", "left")
                              if s in self._last_verified_grasp]
            active_hint = self._assembler.active_side
        right_holding = self._holding_state("right")
        left_holding = self._holding_state("left")
        # try measured first, then declared. a daemon that publishes no `side`
        # leaves _daemon_held_side() None at every instant, so without the
        # measurement the tier stays inert and refuses on a payload the robot is
        # plainly holding. the measurement asks the grip socket what the hands are,
        # which works for a pick the engine ran and the gateway's own verdict tiers never see.
        # both tiers are evaluated exactly once into named locals and the log
        # reports both: measured= for what is measured (or None) and declared= for
        # what the daemon claimed (or None). a single daemon_side collapsed the two,
        # so the acceptance line could not tell a measured resolution from a
        # declared one. the choice keeps measured outranking declared.
        measured_side = self._measured_held_side()
        declared_side = self._daemon_held_side()
        daemon_side = measured_side if measured_side is not None else declared_side
        # whether this caller's `side` came from the daemon record. not the same as
        # daemon_side is not None: daemon_side resolves for every caller, so one
        # that took its side from tier (1), the XOR tier or the ambiguous hint also
        # has a non-None daemon_side. keying the take or the gate on daemon_side
        # would refuse such a caller once the record is gone and consume a record
        # it never used, stranding a genuine hold for the other hand.
        rode_daemon = False
        if len(verified_sides) == 1:
            side = verified_sides[0]
        elif right_holding != left_holding:
            # the XOR: exactly one side measured holding, so that side
            side = "right" if right_holding == "True" else "left"
        elif not verified_sides and right_holding == "False" \
                and left_holding == "False":
            # the unheld refusal, and the only place the daemon tier is consulted.
            # reading it here, nested inside the refusal's own condition rather than as
            # a peer branch above it, bounds the tier to the refused state by
            # construction: the tier-less gateway returns from this branch, so the
            # daemon can change the outcome but never a side that gateway would have
            # chosen. as a peer branch it also pre-empted the ambiguous-hint `else`
            # below, so on a both-hands-holding state it released the daemon's hand
            # where that path releases active_side.
            if daemon_side is None:
                # log it: a refusal with no trace is undiagnosable from the log alone.
                # stderr, not the ROS logger, because this is the ROS-free core. the
                # format is relied on: verified=, measured= and declared= are the three
                # separate evidence reads, so a reader can tell a measured resolution
                # from a declared one.
                sys.stderr.write(
                    "openbot_gateway: place REFUSED nothing-verified-held: "
                    "verified=%s measured=%s declared=%s right=%s left=%s\n"
                    % (verified_sides, measured_side, declared_side,
                       right_holding, left_holding))
                sys.stderr.flush()
                _rel = self._place_release_note()  # the release note
                if _rel:
                    return ("ERROR:nothing verified held - cannot place"
                            + _rel), None
                return "ERROR:nothing verified held - cannot place", None
            side = daemon_side
            rode_daemon = True
        else:
            # ambiguous (both verified, or both measuring holding): degrade to the
            # unverified active_side hint, since no evidence tells the hands apart and
            # no arm claim goes beyond the observed side.
            side = str(active_hint or "right")
            if side not in ("right", "left"):
                side = "right"
        # one line per place so a refusal is diagnosable from the log alone: which
        # tier fired and what every tier read. stderr, not the ROS logger, because
        # this is the ROS-free core. the format is relied on: measured= and
        # declared= are the two tier results (side or None), so it shows whether
        # the released hand is measured or declared, a distinction a single
        # collapsed `daemon=` field could not express.
        sys.stderr.write(
            "openbot_gateway: place held-side: verified=%s measured=%s "
            "declared=%s right=%s left=%s -> %s\n"
            % (verified_sides, measured_side, declared_side, right_holding,
               left_holding, side))
        sys.stderr.flush()
        # held-then-open law: verify_place succeeds only on the place DONE and the
        # hand measured open after the release, and either the hand measured closed
        # before the release (the release produced the open) or the hold is
        # engine-verified by this gateway's own pick. so record both pre-release
        # facts before the engine command executes: the engine place opens only the
        # holding hand and clears that side's close verdict, so a post-release
        # observation alone proves nothing.
        prev_is_open = "None"
        try:
            line = self.grip_verb("state %s" % side)
            with self._obs_lock:
                self._assembler.update(grip_line=line)
                st = self._assembler.grip_state(side)
            if st is not None:
                prev_is_open = ("True" if st.is_open is True else
                                ("False" if st.is_open is False else "None"))
        except Exception:
            prev_is_open = "None"  # unconfirmable, so stay conservative
        with self._obs_lock:
            pick_verified = self._last_verified_grasp.get(side) is not None
        place_verb = "place_left" if side == "left" else "place"
        engine_line = place_verb if coords is None else \
            place_verb + " " + " ".join(str(c) for c in coords)
        # the height token rides last and only with coordinates (release-only is a
        # bare verb). the engine raises the holding arm by
        # min(2.5*H, PLACE_EXTRA_LIFT_MAX_M) before its own ascend, object in hand,
        # release point unchanged.
        if place_height is not None and coords is not None:
            engine_line += " %.6f" % place_height
        # take the daemon hold before the blocking engine call, and hand it back if
        # the engine refuses. _do_place runs on a thread per client connection and
        # is not serialised (only place_target is, via _place_target_lock), so a
        # bare `place` from one client and a `place_target` from another can both
        # get in.
        #
        # the take alone does not stop the second caller shipping a second release:
        # it happens after the tier read, so both callers resolve `daemon_side` from
        # the same live record, and `daemon_taken is None` is only tested on the
        # engine-error path. both can then reach engine_verb, and the
        # single-threaded engine runs the second place on an already-empty hand and
        # returns DONE.
        #
        # so the gate is explicit below: the take re-runs the evidence predicate
        # under the same lock that nulls the slot, and a caller whose record no
        # longer qualifies lost the race and must refuse.
        #
        # the take cannot key off `_pick_state_hold is not None`, because
        # _on_pick_state writes a record on every 5 Hz frame and never writes None,
        # so the slot is non-None even in the states that mean 'do not release
        # yet': the daemon retracted the hold (held:false), or a new pick cycle
        # began (state 'busy', `held` still true). both mean 'do not release',
        # so refusing is right either way, and the predicate treats `busy` as not
        # evidence too, caught in the same lock hold. a caller whose side came from
        # a verified grasp does not depend on the record and stays unaffected.
        #
        # the take is undone on an engine error so a failed place can be retried.
        # that holds for an engine-reported error, raised before or during its own
        # move, but not for the post-send transport failures: the place may have
        # run to completion with the payload gone (300 s reply timeout; the coords
        # branch opens the hand before its post-release steps), and restoring the
        # hold there would re-license a release whose object is already out of
        # hand.
        #
        # both the take and the gate key on `rode_daemon`. doing it before the
        # branch chain is wrong in both directions: a verified-grasp caller got
        # refused, and the unconditional take ate a daemon hold that a `place_left`
        # resolved from tier (1) never used, stranding the right hand's hold for the
        # rest of the epoch.
        daemon_taken = None
        if rode_daemon:
            # one lock hold: re-run the predicate, then null the slot. two critical
            # sections would open a window a concurrent caller or a 5 Hz publish can walk
            # through.
            with self._obs_lock:
                rec = self._daemon_hold_record_unlocked()
                if rec is not None:
                    daemon_taken = rec
                    self._pick_state_hold = None
                    self._pick_state_consumed = True
            if daemon_taken is None:
                sys.stderr.write(
                    "openbot_gateway: place REFUSED hold-retracted-or-taken: "
                    "side=%s verified=%s right=%s left=%s\n"
                    % (side, verified_sides, right_holding, left_holding))
                sys.stderr.flush()
                return "ERROR:nothing verified held - cannot place", None
        reply = self.engine_verb(engine_line)
        cancelled = False
        where = "unknown"
        # `not_a_reply` flags the relabelled UNKNOWN below (the engine may still be
        # executing, so the arm is not swept out from under it; the idle watchdog
        # covers a stall once it ends). `verify_reply` is the reply `_verify`
        # actually judges (None means `reply`).
        not_a_reply = False
        verify_reply: Optional[str] = None
        if not _is_engine_reply(reply):
            not_a_reply = True
            # an unreadable reply is unknown, not a success. this reply drives the
            # entire control flow of this method, and its `else` branch marks the hand
            # released (clears the close verdicts, marks the side open-commanded); a
            # banner line took that branch and dumped a release into the world model
            # that possibly never happened.
            #
            # relabel with ENGINE_TRANSPORT_ERROR_PREFIX rather than adding a case: that
            # is the prefix the restore gate below already reads, and it means exactly
            # what is true here, that it is unknown whether the engine ran it. so the
            # hold is not falsely restored and the release is not falsely recorded.
            reply = ENGINE_TRANSPORT_ERROR_PREFIX + (
                "place engine reply was not a command reply (%r) — outcome "
                "UNKNOWN" % reply.strip()[:120])
        if reply.strip().startswith("CANCELLED:"):
            # a cancelled place leaves the hand state unknown. the engine's coords
            # branch opens the hand ahead of its post-release steps, so the release may
            # well have happened: the verified-hold record is dropped, the daemon hold
            # is never restored (_unconfirmed), and the release bookkeeping `else` is
            # skipped because the reply leads with ERROR. a false refusal on the
            # next place is the deliberate direction.
            cancelled = True
            where = _cancel_phase(reply)
            with self._obs_lock:
                self._last_verified_grasp.pop(side, None)
            reply = ("ERROR:cancelled_by_operator place cancelled at phase %s "
                     "by the operator; hand/payload state UNKNOWN - re-observe "
                     "before retrying; engine_reply=%s"
                     % (where, _defang_error_token(reply.strip())[:120]))
        if reply.startswith("ERROR"):
            # restore gate. an engine-reported error means the place did not complete,
            # so the hold is still real and the caller can retry. a post-send transport
            # failure means none of that is known: the engine can have finished the
            # whole place and died, or still be running it, so those are not restored.
            #
            # classification is by prefix. it cannot be the substring 'timed out' over
            # the whole reply: there are three post-send sites (pipe, socket send,
            # socket read) and the send failure's text carries no 'timed out', so a
            # failed send, whose execution state is equally unknown, would be
            # classified as an engine error and restored, re-licensing a second
            # release. engine_verb labels all three with ENGINE_TRANSPORT_ERROR_PREFIX
            # and this gate reads that prefix only.
            _unconfirmed = reply.startswith(ENGINE_TRANSPORT_ERROR_PREFIX)
            # a cancelled place is just as unknown as a post-send transport failure
            # (the release may already be out), so the taken daemon hold is not restored
            # either.
            _unconfirmed = _unconfirmed or cancelled
            with self._obs_lock:
                if daemon_taken is not None and not _unconfirmed:
                    if self._pick_state_hold is None:
                        # restore only into an empty slot: a /pick_state that lands while the
                        # engine runs describes a newer world and must not be clobbered.
                        self._pick_state_hold = daemon_taken
                    # nothing got released, so release the latch too; otherwise the retry
                    # refuses on a hold that is still genuinely held.
                    self._pick_state_consumed = False
            # mirrors _do_pick: the engine ran and the place did not end DONE (an
            # engine error, a cancelled place or a post-send transport failure; a failed
            # send parks via the writer only, a reply timeout first tries the engine's
            # queued `park`). every pre-engine refusal returned before engine_verb. the
            # verdict is judged without the arm_after= token.
            if (getattr(self, "_b5_on", False) is True and not not_a_reply
                    and not reply.startswith(
                        "ERROR:pick engine not configured")):
                verify_reply = reply
                reply = _insert_before_engine_reply(
                    reply, self._post_task_arm_sweep(
                        "place", "cancelled" if cancelled else "error",
                        prefer_writer=reply.startswith(
                            ENGINE_SEND_FAILED_PREFIX)))
        else:
            # a successful place goes hold_rc on -> LOW_CARRY -> open -> payload off ->
            # NEUTRAL -> hold_rc off. the engine opens only the holding hand, so the
            # place counts as a gateway-dispatched open for `side` only: only its close
            # verdict is cleared and only it is marked open-commanded, so only its
            # post-place reads qualify as the trusted open reference. the verified-hold
            # record is cleared, and the consumed latch carries the clear past the next
            # 5 Hz sticky 'held: true'.
            self._clear_close_verdicts((side,))
            # skip the open-commanded mark when the reply carries the park-closed
            # marker `release_open=<side>:True` (the lift closeout parked this hand
            # closed after a measured-open release): the reads that follow are of a
            # closed hand and must never become the trusted open reference. an absent or
            # false token (park skipped, release unmeasured) keeps the mark.
            _pc = PLACE_PARK_CLOSED_RE.search(reply)
            _parked_closed = _pc is not None and _pc.group(1) == side
            with self._obs_lock:
                if not _parked_closed:
                    self._assembler.note_open_commanded(side)
                self._last_verified_grasp.clear()
                self.__dict__.setdefault('_last_place_release', {})[side] = \
                    time.time()  # stamp the time
            reply = ("%s; prev_is_open=%s:%s; prev_pick_verified=%s:%s"
                     % (reply, side, prev_is_open, side,
                        "True" if pick_verified else "False"))
        if cancelled:
            v = _cancelled_verdict("place", where)
            self._emit_verdict(v)
            return reply, v
        if verify_reply is not None:
            # here the verdict judges the place, not the park token
            return reply, self._verify("place", verify_reply)
        return reply, self._verify("place", reply)

    # ── /place_target lane (one-shot: place onto the named target) ───────────

    def place_target(self, label: str) -> Dict[str, object]:
        """One-shot: release the held object ON the named target detection.

        The answer IS the rendered phase table — every exit (refusal or
        success) ends with the VERDICT row, is logged, and is published on
        /place_status, so a refused command is exactly as diagnosable as a
        good one.

        NEVER degrades to a bare current-pose release.  A target that cannot
        be resolved (empty, absent, stale, ambiguous, or positionless) refuses
        with NO engine command at all: a bare release would drop the payload
        wherever the arm happens to be holding it, which is the one outcome
        the operator did not ask for.

        The place itself is the shipped coordinate lane —
        ``dispatch("place", [x, y, z])`` — so the held-side law, the unheld
        refusal, the engine line and the post-place verifier are all inherited
        rather than re-derived.  (That call goes through ``dispatch`` because
        the shipped observation seam patches it — see
        tests/test_place_target_lane.py:445 — and the descent lane's own guard
        sits there too; the call is marked NESTED for its duration so the guard
        declines it and this one-shot lane stays out of the descent lane.)

        The lane RETREATS FIRST: with a descent held, a release here would
        happen at RAW coordinates, with no frame shift, and leave the body low
        (plan §6, refuter MEDIUM-1).  A rise that cannot be confirmed refuses
        the whole request.
        """
        law = self._place_target_law
        log = law.PhaseLog(label)
        log.start(time.time())
        text = str(label or "").strip()
        if not text:
            return self._place_target_finish(
                log, "refuse", "ERROR:place_target empty label")
        if not self._place_target_lock.acquire(blocking=False):
            # concurrent, never queued: a second place would release into a scene the
            # first place already changed.
            return self._place_target_finish(log, "refuse", law.PLACE_BUSY_MSG)
        try:
            # retreat-first: the ROS callback invokes place_target directly, skipping
            # dispatch, so the lane's next-command rule never sees this request. with a
            # descent held, this lane would place at raw coordinates with no frame
            # shift, leaving the body low. a reason means the rise is not confirmed:
            # refuse the request and dispatch nothing.
            held_detail = self._low_lane_retreat_first_if_held()
            if held_detail is not None:
                self._place_target_phase(log, "retreat", "refuse", held_detail)
                return self._place_target_finish(log, "refuse", held_detail)
            return self._place_target_run(log, text)
        except Exception as exc:  # noqa: BLE001 - a table beats a dead thread
            # a missing dependency still has to answer with a table. a deployment
            # can keep its instance labels outside the packaged tree, and its brain
            # can predate _drop_point, so any layout resolving nothing raises here.
            # an exception escaping the worker would leave the operator with no
            # output on either channel, undiagnosable from outside. every other
            # failure in this lane ends in a row; so must this one.
            detail = ("ERROR:place_target unavailable (%s): %s"
                      % (type(exc).__name__, exc))
            return self._place_target_finish(log, "error", detail)
        finally:
            self._place_target_lock.release()

    def _place_target_run(self, log, text: str) -> Dict[str, object]:
        law = self._place_target_law
        # point-cloud consumer: only with a switch on (`_pc` set). pre-subscribe for
        # 20 s so its DDS match is up when the rim request goes out. fire-and-
        # forget; the client never raises, and this is guarded again so no stub can
        # take the place lane down.
        pc = getattr(self, "_pc", None)
        if pc is not None:
            try:
                pc.warm(20.0)
            except Exception:  # noqa: BLE001
                pass
        with self._obs_lock:
            dets = list(self._assembler.observation().detections)
        rows = law.resolver_rows(dets)
        now = time.time()
        # check stale against absent before the resolver: both reach it as 'none',
        # and a frozen scene would then be reported 'not detected', sending the
        # operator to the camera instead of the detection lane.
        stale = law.stale_only(text, rows, now)
        result = law.resolve(text, rows, now)
        if result.get("status") != "single":
            status, detail = law.classify_refusal(text, result, stale)
            self._place_target_phase(log, "resolve", status, detail, now)
            return self._place_target_finish(log, status, detail)
        win = result["det"]
        self._place_target_phase(
            log, "resolve", "ok",
            "%s  conf=%s  scan=%d" % (win.get("label"),
                                      law.fmt_num(win.get("conf")), len(rows)),
            now)
        if pc is None:
            t_status, t_detail, coords = law.place_target_from_row(win)
        else:
            # `pc_rim` is the optional kwarg of place_target_from_row, passed only when
            # the client is built and only if the function declares it, so the off path
            # is exactly the plain call.
            t_status, t_detail, coords = law.place_target_from_row(
                win, **self._pc_place_kwargs(law))
        self._place_target_phase(log, "target", t_status, t_detail)
        if coords is None:
            return self._place_target_finish(log, t_status, t_detail)
        try:
            # through dispatch, the shipped seam: a test patches gw.dispatch to watch a
            # second place queue up behind this one, which a direct _do_place call
            # hides. the descent lane's guard sits in dispatch too, so the call is
            # marked nested for exactly its duration and the guard declines it, the
            # same exclusion a direct call gave without going round the seam. the flag
            # is restored in `finally`, so a raise cannot disable the lane.
            prev_nested = self._descent_nested
            self._descent_nested = True
            try:
                reply, verdict = self.dispatch("place", list(coords))
            finally:
                self._descent_nested = prev_nested
        except Exception as exc:  # a table must come out of this, always
            detail = "ERROR:place_target dispatch raised: %s" % exc
            self._place_target_phase(log, "execute", "error", detail)
            return self._place_target_finish(log, "error", detail)
        # which side got released is stated only in _do_place's own reply
        # annotation, so read it rather than re-deriving the held-side law here.
        log.side = law.side_from_reply(reply)
        e_status, e_detail = law.classify_dispatch(reply, verdict)
        line = "%s %s" % ("place_left" if log.side == "left" else "place",
                          " ".join(law.fmt_num(c, "%.3f") for c in coords))
        self._place_target_phase(
            log, "execute", e_status,
            line if e_status == "ok" else "%s -> %s" % (line, e_detail))
        v_status, v_detail = law.verdict_row(verdict)
        if e_status == "refuse":
            # the engine never ran here, because the gateway declined on evidence, so
            # there is nothing to verify. a 'verify' row would read as a verification
            # failure and bury the real reason behind 'no verdict'; the verdict row
            # headlines the refusal instead.
            return self._place_target_finish(log, e_status, e_detail, verdict)
        self._place_target_phase(log, "verify", v_status, v_detail)
        return self._place_target_finish(log, v_status, v_detail, verdict)

    def _place_target_phase(self, log, phase: str, status: str, detail: str,
                            now: Optional[float] = None) -> None:
        """Record + publish ONE phase row (elapsed measured, never invented)."""
        law = self._place_target_law
        if now is None:
            now = time.time()
        log.add(phase, status, detail, now)
        row = log.rows[-1]
        self.publish_place_status(law.status_message(
            log, phase, status, detail, row["elapsed_s"]))

    def _place_target_finish(self, log, status: str, detail: str,
                             verdict: Optional[str] = None) -> Dict[str, object]:
        """Close a run: VERDICT row, render, log, publish the final status."""
        law = self._place_target_law
        total = log.finish(status, detail, time.time())
        table = law.render_table(log)
        self.log_place_target(table)
        self.publish_place_status(law.status_message(
            log, "VERDICT", status, detail, total, verdict=verdict,
            table=table))
        return {"status": status, "detail": detail, "verdict": verdict,
                "table": table, "rows": list(log.rows), "total_s": total}

    def publish_place_status(self, message: str) -> None:
        """Hook: publish one /place_status payload (no-op outside ROS)."""
        return

    def log_place_target(self, table: str) -> None:
        """Hook: record the rendered /place_target table.

        The laptop-testable core prints — its stdout IS the log; ``_RosGateway``
        overrides this with the node logger so the table lands in the gateway
        log next to everything else.
        """
        print(table, flush=True)

    # ── point-cloud consumer hooks ───────────────────────────────────────────
    # thin shims over the pc client. nothing below is reachable with
    # `self._pc` None, which is every gateway with all three switches off.

    def _publish_pc_arm(self, text: str) -> None:
        """Hook: publish one /pc_arm String (no-op outside ROS)."""
        return

    def on_body_pose(self, text: Any) -> None:
        """B8k: one /openbot/body_pose message (JSON object) from the odometry
        process.  Anything that is not a JSON object is ignored; the client
        judges the pose (measured, complete, fresh) when it is used."""
        try:
            pose = json.loads(text) if isinstance(text, str) else None
        except ValueError:
            return
        if isinstance(pose, dict):
            with self._body_pose_lock:
                self._body_pose = pose

    def _latest_body_pose(self) -> Optional[Dict[str, Any]]:
        """B8k: a copy of the latest body pose, or None."""
        lock = getattr(self, "_body_pose_lock", None)
        if lock is None:
            return None
        with lock:
            pose = self._body_pose
        return dict(pose) if isinstance(pose, dict) else None

    def _pc_log(self, message: str) -> None:
        """One ``[pc] ...`` line on stderr (the gateway log).  Never raises."""
        try:
            sys.stderr.write("[pc] %s\n" % (message,))
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 — a log line must never break a lane
            pass

    def _pc_say(self, message: str) -> None:
        """``_pc_log`` for the shims' except paths: even a replaced / broken log
        hook cannot turn a degraded hook into a raise."""
        try:
            self._pc_log(message)
        except Exception:  # noqa: BLE001
            pass

    def _pc_rim_for(self, row: Any) -> Optional[Dict[str, Any]]:
        """``place_target_from_row``'s ``pc_rim`` hook (H-PT-1).

        The rim drop ``{"drop": {...}, "delta_z", "band", "band_formula"}`` for
        a resolved place-target row, or ``None`` — the caller then keeps the
        formula it always used.  The policy (the request, the one log line,
        and the gates: OK + RIM_FIT + the SAME measured pose + a fresh row,
        and ``pc_place_rim`` ON, else shadow = log only) is
        ``pc_client.rim_for``; this is only the wiring.  NEVER raises: a
        failure here must not take the place lane down.
        """
        try:
            pc = getattr(self, "_pc", None)
            if pc is None:
                return None
            from gateway import pointcloud_client as pc_client_mod  # lazy, then cached
            return pc_client_mod.rim_for(
                pc, row, getattr(self, "_pc_place_rim", False) is True,
                time.time, log=self._pc_log)
        except Exception as exc:  # noqa: BLE001 — wrap, log, return None
            self._pc_say("place_rim hook failed: %s: %s"
                         % (type(exc).__name__, exc))
            return None

    def _nav_confirmed_epoch(self) -> float:
        """The dispatch epoch the persistent SkillVerifier confirmed (0.0).

        The gateway holds ONE SkillVerifier instance across calls; the
        confirmed-epoch memory lives in it (fix spec A-e).  None before the
        first verify.
        """
        if self._verifier is None:
            return 0.0
        return self._verifier.last_confirmed_epoch()

    def cancel_engine(self, timeout_s: float = CANCEL_REPLY_TIMEOUT_S) -> str:
        """Tell the pick/place engine to cancel, WITHOUT ever spawning one.

        WHY THIS DOES NOT GO THROUGH ``engine_verb`` (2026-10-01).  That path
        is connect-first → SPAWN-FALLBACK (``EngineBridge`` :959, connect at
        ``_try_connect_unlocked`` :1070): calling it
        with nothing listening boots a cold engine, and a cold engine's first
        act is to drive BOTH ARMS TO NEUTRAL — the very defect the resident
        engine was built to remove.  So "cancel the pick" would have started a
        pick.  Here a direct one-shot connect is used instead: the engine's
        socket server is thread-per-connection and answers cancel out-of-band
        (``tabletop_pick_place_engine.CANCEL_VERBS``), so this returns in
        milliseconds while the engine is still blocked inside the pick.

        Returns the engine's own reply line, or a token describing why there
        is nothing to cancel.  NEVER raises — a cancel must not be the thing
        that fails the way a pick did.
        """
        try:
            # skip_ready_banner: the engine greets every new connection with READY
            # before it answers, so without it this would return "READY" for a cancel
            # that has been accepted.
            return _send_unix_verb(self._engine_sock, "cancel", timeout_s,
                                   skip_ready_banner=True)
        except FileNotFoundError:
            return "NOENGINE:no engine socket"
        except (ConnectionError, OSError, socket.timeout) as exc:
            return "NOENGINE:%s" % exc

    def _do_cancel(self, args: List) -> Tuple[str, Optional[str]]:
        """Cancel whatever is running: the pick/place engine, and the sweep.

        THE OPERATOR ASK (verbatim 2026-09-30): "i want new feature such that i
        can ask the robot to cancel search, task midway, such as stop picking
        up while its picking, or stop placing while its placing, stop sweeping
        while its sweeping."

        THREE LANES, TWO MECHANISMS, and this verb is the single entry point:

          * PICK / PLACE — the engine.  A flag is set over its socket
            (out-of-band, answered by the connection thread) and the engine's
            motion choke points raise; the unwind freezes the arm IN PLACE and
            releases the writer's RC interlock (``pick_engine_v3``
            ``except engine_cancel.Cancelled``).  This is the lane that had no
            stop at all before today.
          * SWEEP / WALK — the FSM.  Already had a stop; reused verbatim
            through ``stop_walk`` (which routes via the coordinator so the
            coordinator's own epoch/flow state is cleared too, rather than
            being left latched).
          * SEARCH — the same walk FSM, reached by the same ``stop_walk``.

        WHY THE ENGINE FREEZE IS *NOT* DUPLICATED HERE.  The gateway holds the
        writer socket and could send ``freeze`` itself a few ms earlier.  It
        deliberately does not: the engine's unwind is the only actor that
        knows whether a task was actually in flight, and a second actor
        freezing the writer on its own initiative would stop motion that
        nobody asked to stop — the engine boot move and the descent lane both
        drive that socket.  20 ms of latency is the price of having exactly one
        actor with authority over an in-flight arm move.

        NO VERDICT: a cancel claims no post-condition.  Nothing here asserts
        the arm reached a goal — ``_shape_reply(reply, None)``.
        """
        # stamp the cancel before stopping the walk. _await_walk_reached polls this
        # stamp, so it has to exist no later than the stopped FSM it explains;
        # otherwise a wait could see the stopped walk without the cancel that caused
        # it and fall through to 'walk_idle_after_motion', which reads as a normal
        # finish instead of an operator stop.
        with self._obs_lock:
            self._cancel_ts = time.time()
            # clear the epoch too. without it in_flight only needs a SEARCHING with no
            # IDLE, so a later navigate to the same label could be reused as still in
            # flight and answer 'reused_in_flight' for a walk that has already ended, the
            # same stale-liveness class as the walk frame latch problem.
            self._nav_epoch = (0.0, None)
            self._phases_seen = []
        engine_reply = self.cancel_engine()
        if engine_reply.startswith("CANCEL:accepted"):
            engine_tok = "engine=cancelling"
        elif engine_reply.startswith("CANCEL:refused"):
            # the engine is alive and idle, and its own words say why
            engine_tok = "engine=idle(%s)" % _defang_error_token(
                engine_reply.split(" ", 1)[-1])
        else:
            engine_tok = "engine=unavailable"
        try:
            self.stop_walk()
            walk_tok = "walk=stopped"
        except Exception as exc:  # noqa: BLE001 — report, never raise
            walk_tok = "walk=stop_failed(%s)" % _defang_error_token(str(exc))
        # turn_around drives the writer's cmd_set directly rather than through the
        # FSM, so the FSM stop above does not zero it: send an explicit
        # 'cmd_set off'. gated so a turn not in flight sends nothing and the reply
        # stays byte-identical. no token either way.
        if getattr(self, "_turn_active", False):
            try:
                self.writer_verb("cmd_set off")
            except Exception:  # noqa: BLE001 — best effort; pulse auto-expires
                pass
        # the descent lane. a cancel must be able to stop a lane pick before its
        # engine dispatch and release a hold. reached via getattr, so a stand-in
        # without lane attributes, and a lane-off node whose _low_lane is None, adds
        # no token and keeps the reply identical.
        lane_tok = ""
        lane = getattr(self, "_low_lane", None)
        if lane is not None:
            lane_tok = self._low_lane_cancel()
        # the engine reply is echoed for the wire trace, but its NOENGINE: prefix
        # must never leak into a reply the brain reads as a failure, since the
        # cancel did its job on the other lane. echoes of the engine's text are
        # defanged for the same reason: a cancel that succeeded must never be
        # shaped ok:false just because the engine phrased its answer with the
        # ERROR: token.
        #
        # release the RC interlock and park any orphaned arm, even with the engine
        # idle: a stop can answer 'engine=idle(no task running)' while the writer
        # PD-holds a blocked arm at Tmax 102-131 C. reached via getattr and only
        # with the master switch on, which keeps a stand-in and a core node on the
        # plain byte stream. the tokens go after lane_tok and before
        # `; engine_reply=`, which stays last since the verifier's match is
        # greedy.
        arm_tok = ""
        if getattr(self, "_b5_on", False) is True:
            rel = getattr(self, "_cancel_arm_release", None)
            if rel is not None:
                try:
                    arm_tok = rel(engine_reply)
                except Exception as exc:  # noqa: BLE001 — a stop never raises
                    arm_tok = "; arm=unknown(%s)" % _b5_tok_text(exc, 40)
        return ("DONE cancel: %s; %s%s%s; engine_reply=%s"
                % (engine_tok, walk_tok, lane_tok, arm_tok,
                   _defang_error_token(engine_reply)),
                None)

    # ── arm safety, stale target, liveness ───────────────────────────────────

    def _read_arm_q(self) -> Optional[List[float]]:
        """Writer ``q`` -> 14 floats [LEFT7, RIGHT7], or None.  Never raises.

        The reply grammar is the writer's own (``Q:`` + 14 floats,
        main_merged_writer.py:544 / writer_arm_client.py:187-190).
        """
        try:
            reply = self.writer_verb("q", timeout_s=ARM_Q_READ_TIMEOUT_S)
        except Exception:  # noqa: BLE001 — unreadable == unknown, never motion
            return None
        reply = str(reply or "")
        if not reply.startswith("Q:"):
            return None
        parts = reply[2:].split()
        if len(parts) != 14:
            return None
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return None
        if not all(math.isfinite(v) for v in vals):
            return None
        return vals

    def _release_rc(self) -> str:
        """Writer ``hold_rc off`` -> ``released`` | ``failed(<x>)``.

        The RC-STICK INTERLOCK ONLY (G0): this never moves, stops or relaxes
        the arm, and the token it feeds is ``hold_rc=``, never ``arm=``.
        """
        try:
            r = str(self.writer_verb("hold_rc off", timeout_s=1.0) or "")
        except Exception as exc:  # noqa: BLE001 — report, never raise
            return "failed(%s)" % _b5_tok_text(exc)
        if r.startswith("DONE"):
            return "released"
        return "failed(%s)" % _b5_tok_text(r)

    def _await_safe_pose(self, budget_s: float
                         ) -> Tuple[Optional[str], Optional[List[float]]]:
        """Poll ``q`` until both arms read a safe pose, or ``budget_s`` ends.

        Returns (pose name | None, last q | None).  A route's DONE is never
        taken as the arrival: the writer answers ``arm`` when the slew STARTS
        (writer_arm_client.py "DONE = slew STARTED"), and only a measured
        ``q`` at a safe pose makes a park real.
        """
        deadline = time.time() + budget_s
        while True:
            q = self._read_arm_q()
            pose = _arm_safe_pose(q) if q is not None else None
            if pose is not None:
                return pose, q
            remaining = deadline - time.time()
            if remaining <= 0.0:
                return None, q
            time.sleep(min(PARK_SETTLE_POLL_S, remaining))

    def _park_arm(self, reason: str = "", rc: str = "first",
                  prefer_writer: bool = False) -> Dict[str, Any]:
        """Move the arm(s) to a safe pose and VERIFY it.  Blocking; serialised
        (a second concurrent call returns ``park_in_progress`` and sends
        nothing); never raises.  Call it through ``_start_park_thread``.

        Route ``engine``: the engine's queued ``park`` verb over a DIRECT
        one-shot socket (banner skipped) — NEVER ``engine_verb``, which
        connects-or-SPAWNS, and a cold engine's first act is a boot move (see
        cancel_engine).  The engine queues it behind whatever it is still
        executing, so its answer also marks the end of the engine's own
        post-reply exit.  An engine ``ERROR:park ...`` is its own verdict and
        is NOT overridden.

        Route ``writer_ready``: when the engine is unreachable (nothing
        listening, or the connection closed unanswered), answers
        ``ERROR:unknown command`` (pick_engine_v3.py:7699 grammar) or
        ``prefer_writer`` is set.  An engine that is THERE but does not answer
        within PARK_REPLY_TIMEOUT_S (still executing, booting or wedged) is
        NEVER bypassed: the park fails unmoved rather than letting a second
        actor command the arm.  ``q`` is RE-READ first (the engine may have
        parked the arm itself meanwhile, and a redundant slew could push a
        relaxed, blocked arm again); an arm already safe is not moved, an
        unreadable ``q`` never moves; otherwise the verbatim 30-token READY
        line goes straight to the writer (hands untouched; NO retract along
        the approach and NO z-floor clamp — see the batch-5 risks).

        Either route's DONE is followed by a bounded ``q`` settle check
        (PARK_SETTLE_S): ``ok`` only when the arm READS a safe pose, and only
        then is ``_arm_parked_ts`` stamped.

        ``rc``: ``"first"`` sends ``hold_rc off`` before the route (stop-free
        callers: watchdog, G4 pre-pick); ``"last"`` after it (the G2 sweep —
        never release the interlock under the engine's own unfinished exit,
        which keeps it on until its arm motion ends); ``"skip"`` when the
        caller already sent it (G1).

        Returns the record also stored in ``_arm_safety['last_park']``:
        ``{reason, route, result, ok, pose, reply, why, rc, ts}`` with
        ``result`` parked | already_safe | park_failed.
        """
        if not self._park_lock.acquire(blocking=False):
            return {"reason": reason, "route": None, "ok": False,
                    "result": "park_in_progress", "pose": None,
                    "reply": "park_in_progress", "why": "", "rc": None,
                    "ts": time.time()}
        try:
            rc_tok = self._release_rc() if rc == "first" else None
            route = "writer_ready" if prefer_writer else "engine"
            why = "prefer_writer" if prefer_writer else ""
            reply = ""
            pose = None
            result = "park_failed"
            if not prefer_writer:
                try:
                    reply = _send_unix_verb(self._engine_sock, "park",
                                            PARK_REPLY_TIMEOUT_S,
                                            skip_ready_banner=True)
                except socket.timeout:
                    # the engine is there but did not answer in time: it queues `park` behind
                    # whatever it is still executing, or it is booting or wedged. a writer slew
                    # at that point would mean two actors commanding the arm, so nothing moves;
                    # watchdog retries on its cooldown and hands a persistent stall to the
                    # operator. caught first, because on py3.8 socket.timeout is an OSError.
                    reply = ("engine busy - no park reply in %.0fs; not "
                             "moved by a second actor" % PARK_REPLY_TIMEOUT_S)
                    why = "engine busy"
                except (OSError, ConnectionError) as exc:
                    # nothing listening (no socket file, or refused), or the engine closed the
                    # connection without answering, which it only does while shutting down. no
                    # engine can be commanding the arm, so the writer may.
                    why = "engine unreachable: %s" % (exc,)
                    route = "writer_ready"
                else:
                    if reply.startswith("ERROR:unknown command"):
                        why = reply
                        route = "writer_ready"
            if route == "engine":
                if reply.startswith("DONE"):
                    pose, _q = self._await_safe_pose(PARK_SETTLE_S)
                    if pose is None:
                        reply = "%s; not at a safe pose %.1fs later" % (
                            reply, PARK_SETTLE_S)
                    elif "already_safe" in reply:
                        result = "already_safe"
                    else:
                        result = "parked"
            else:
                q = self._read_arm_q()
                pose = _arm_safe_pose(q) if q is not None else None
                if q is None:
                    reply = "q_read_failed - no blind move"
                elif pose is not None:
                    result = "already_safe"
                    reply = "already_safe(%s)" % pose
                else:
                    try:
                        reply = str(self.writer_verb(
                            WRITER_READY_ARM_LINE,
                            timeout_s=PARK_WRITER_TIMEOUT_S) or "")
                    except Exception as exc:  # noqa: BLE001
                        reply = "writer failed: %s" % (exc,)
                    if reply.startswith("DONE"):
                        pose, _q = self._await_safe_pose(PARK_SETTLE_S)
                        if pose is not None:
                            result = "parked"
                        else:
                            reply = "%s; not at a safe pose %.1fs later" % (
                                reply, PARK_SETTLE_S)
            ok = result in ("parked", "already_safe")
            if ok:
                self._arm_parked_ts = time.time()
            if rc == "last":
                rc_tok = self._release_rc()
            rec = {"reason": reason, "route": route, "result": result,
                   "ok": ok, "pose": pose, "reply": _b5_tok_text(reply, 160),
                   "why": _b5_tok_text(why, 120), "rc": rc_tok,
                   "ts": time.time()}
            self._arm_safety["last_park"] = dict(rec)
            sys.stderr.write(
                "[arm-safety] park reason=%s route=%s result=%s pose=%s "
                "hold_rc=%s reply=%s\n"
                % (reason, route, result, pose, rc_tok or "unchanged",
                   rec["reply"]))
            sys.stderr.flush()
            return rec
        finally:
            self._park_lock.release()

    def _start_park_thread(self, reason: str, rc: str,
                           prefer_writer: bool) -> threading.Thread:
        """Run ONE ``_park_arm`` on a daemon thread; its record lands in the
        thread's ``b5_result`` dict.  ``_park_thread`` always names the
        latest park, so every caller can see (and join) it."""
        holder: Dict[str, Any] = {}

        def run() -> None:
            try:
                holder.update(self._park_arm(reason=reason, rc=rc,
                                             prefer_writer=prefer_writer))
            except Exception as exc:  # noqa: BLE001 — never die silently
                holder.update({"reason": reason, "route": None, "ok": False,
                               "result": "park_failed", "pose": None,
                               "reply": "park raised %r" % (exc,), "why": "",
                               "rc": None, "ts": time.time()})
                sys.stderr.write("[arm-safety] park raised: %r\n" % (exc,))
                sys.stderr.flush()

        t = threading.Thread(target=run, name="arm-park", daemon=True)
        t.b5_result = holder  # type: ignore[attr-defined]
        with self._park_start_lock:
            self._park_thread = t
            t.start()
        return t

    def _park_busy(self) -> bool:
        t = self._park_thread
        return (t is not None and t.is_alive()) or self._park_lock.locked()

    def _park_sync(self, reason: str, prefer_writer: bool = False,
                   rc: str = "first", join_s: float = PARK_JOIN_S
                   ) -> Dict[str, Any]:
        """Park and wait at most ``join_s`` in total.  A park already in
        flight is joined first; when it ended with the arm verified safe its
        record is returned and no second park is sent."""
        deadline = time.time() + join_s
        prev = self._park_thread
        if prev is not None and prev.is_alive():
            prev.join(max(0.0, deadline - time.time()))
            if prev.is_alive():
                return {"reason": reason, "route": None, "ok": False,
                        "result": "park_failed", "pose": None,
                        "reply": "an earlier park is still running after "
                                 "%.1fs" % join_s, "ts": time.time()}
            res = dict(getattr(prev, "b5_result", None) or {})
            if res.get("ok"):
                return res
        t = self._start_park_thread(reason, rc, prefer_writer)
        t.join(max(0.0, deadline - time.time()))
        if t.is_alive():
            return {"reason": reason,
                    "route": "writer_ready" if prefer_writer else "engine",
                    "ok": False, "result": "park_failed", "pose": None,
                    "reply": "park still running after %.1fs" % join_s,
                    "ts": time.time()}
        return dict(getattr(t, "b5_result", None) or {
            "reason": reason, "route": None, "ok": False,
            "result": "park_failed", "pose": None,
            "reply": "no park record", "ts": time.time()})

    def _cancel_arm_release(self, engine_reply: str) -> str:
        """G1: the stop's ``; hold_rc=...; arm=...`` tokens.  Never blocks long.

        (a) always ``hold_rc off`` (the stick interlock, G0);
        (b) ``CANCEL:accepted`` -> the engine owns the in-flight unwind:
            ``arm=engine_unwinding``, the disturbed stamp is set and nothing is
            parked here (the pick's own G2 sweep parks the frozen arm when the
            engine's CANCELLED reply lands; the G3 watchdog is the backstop);
        (c) otherwise (refused / NOENGINE / anything else) read ``q``: both
            arms safe -> ``arm=already_safe(<P>)``; unreadable ->
            ``arm=unknown(q_read_failed)`` and NO motion; out -> ONE daemon
            park thread and ``arm=park_started(<route>)`` at once, so the stop
            reply stays inside CANCEL_REPLY_TIMEOUT_S and the brain's
            CANCEL_TIMEOUT_S; a park already running -> ``arm=park_in_progress``.
        """
        out = "; hold_rc=%s" % self._release_rc()
        if engine_reply.startswith("CANCEL:accepted"):
            self._arm_disturbed_ts = time.time()
            return out + "; arm=engine_unwinding"
        q = self._read_arm_q()
        if q is None:
            return out + "; arm=unknown(q_read_failed)"
        pose = _arm_safe_pose(q)
        if pose is not None:
            return out + "; arm=already_safe(%s)" % pose
        self._arm_disturbed_ts = time.time()
        if self._park_busy():
            return out + "; arm=park_in_progress"
        # writer-direct only when no engine socket exists at all (the
        # FileNotFoundError form from cancel_engine). any other NOENGINE (timeout,
        # refused, reset) does not prove the engine idle: the park tries the
        # engine's queued `park` first and falls back to the writer by itself when
        # nothing is listening.
        prefer_writer = engine_reply.startswith("NOENGINE:no engine socket")
        self._start_park_thread("stop", "skip", prefer_writer)
        return out + "; arm=park_started(%s)" % (
            "writer_ready" if prefer_writer else "engine")

    def _post_task_arm_sweep(self, task: str, outcome: str,
                             prefer_writer: bool = False) -> str:
        """G2: ``; arm_after=...`` after a pick/place that did not end DONE.

        Reads q; an arm outside SAFE_ARM_POSES is parked SYNCHRONOUSLY,
        bounded by PARK_JOIN_S, with the RC interlock released LAST.  Never
        raises; no token contains ``ERROR:``.
        """
        q = self._read_arm_q()
        if q is None:
            return "; arm_after=unknown(q_read_failed)"
        pose = _arm_safe_pose(q, segment=True)
        if pose is not None:
            return "; arm_after=already_safe(%s)" % pose
        self._arm_disturbed_ts = time.time()
        rec = self._park_sync("%s_%s" % (task, outcome),
                              prefer_writer=prefer_writer, rc="last")
        if rec.get("ok"):
            if rec.get("result") == "already_safe":
                return "; arm_after=already_safe(%s)" % (
                    rec.get("pose") or "unknown")
            return "; arm_after=parked(%s)" % rec.get("route")
        return "; arm_after=park_failed(%s)" % _b5_tok_text(
            rec.get("reply") or rec.get("result"))

    # -- the post-cancel stale-target gate --------------------------------

    def _post_cancel_target_gate(self, verb: str, args: List
                                 ) -> Optional[Tuple[str, None]]:
        """Refuse a post-cancel/abort pick whose target no FRESH detection
        confirms, or None to fall through to today's path.

        Active only after a cancel or an arm disturbance that no passing pick
        has cleared since.  (1) Join a park in flight, then park an arm still
        outside SAFE_ARM_POSES (the whole park phase bounded by PARK_JOIN_S).
        (2) Within FRESH_DET_WAIT_S, look for a detection packet newer than
        max(cancel, park) — and newer than the gate itself when the arm could
        not be verified safe.  (3) Pass on the first such packet with a row
        within STALE_TARGET_TOL_M per axis of args[0:3] (and of the hinted
        label's CLASS, when one exists: 'bottle_1' matches a 'bottle' row);
        the pass clears the gate.

        MEASURED F5 (run21 snapshot gateway.log): after a cancel at
        hand_close the engine froze the arm AT the bottle; every later
        detection read z~0.108 against the original 0.005, and the re-pick
        :316 [0.106, 0.403, 0.108] was taken from such a packet — 'fresh'
        (ts 781 > cancel 710), so a cancel-timestamp gate alone would have
        passed it.  Park first, then a fresh match, is what refuses it.
        """
        with self._obs_lock:
            cancel_ts = self._cancel_ts
        disturbed = self._arm_disturbed_ts
        cleared = self._arm_gate_cleared_ts
        if not ((disturbed > 0.0 and disturbed >= cleared)
                or cancel_ts > cleared):
            return None
        try:
            px, py, pz = float(args[0]), float(args[1]), float(args[2])
        except (TypeError, ValueError, IndexError):
            return None  # _do_pick refuses malformed coords itself
        if not all(math.isfinite(v) for v in (px, py, pz)):
            return None
        # (1) the park phase: one deadline for the join and the park
        park_deadline = time.time() + PARK_JOIN_S
        prev = self._park_thread
        if prev is not None and prev.is_alive():
            prev.join(max(0.0, park_deadline - time.time()))
        q = self._read_arm_q()
        verified = q is not None and _arm_safe_pose(q) is not None
        if q is not None and not verified:
            self._arm_disturbed_ts = time.time()
            remaining = park_deadline - time.time()
            if remaining > 0.0:
                rec = self._park_sync("pre_pick_after_cancel",
                                      join_s=remaining)
                verified = bool(rec.get("ok"))
        since = max(cancel_ts, self._arm_parked_ts)
        if not verified:
            # if the arm is unparked or unreadable, at least demand a packet taken
            # after this gate gave up waiting for it.
            since = max(since, time.time())
        # (2) and (3): the fresh-match wait. the label is compared at class level
        # (_b5_hint_class_keys), because a ref-form hint like 'bottle_1' names the
        # class 'bottle' the perception rows actually carry.
        hint = getattr(getattr(self, "_req_target_hint", None), "label", None)
        hint_keys = _b5_hint_class_keys(hint)

        def _label_ok(d) -> bool:
            return _b5_row_matches_hint(d, hint_keys)

        clock, sleep = self._b5_clock, self._b5_sleep
        deadline = clock() + FRESH_DET_WAIT_S
        newest = None
        while True:
            with self._obs_lock:
                ts = self._assembler.last_detections_ts
                rows = (list(self._assembler.observation().detections)
                        if ts > since else [])
            if ts > since:
                newest = (ts, rows)
                for d in rows:
                    if (max(abs(d.x - px), abs(d.y - py), abs(d.z - pz))
                            <= STALE_TARGET_TOL_M and _label_ok(d)):
                        self._arm_gate_cleared_ts = time.time()
                        return None
            now = clock()
            if now >= deadline:
                break
            sleep(min(B5_POLL_S, deadline - now))
        if newest is None:
            newest_txt = "none since the park"
        elif not newest[1]:
            newest_txt = "empty packet ts=%.3f" % newest[0]
        else:
            best = min(newest[1], key=lambda d: (
                not _label_ok(d),
                (d.x - px) ** 2 + (d.y - py) ** 2 + (d.z - pz) ** 2))
            newest_txt = "%s@(%.3f,%.3f,%.3f) ts=%.3f" % (
                _b5_tok_text(best.label, 32), best.x, best.y, best.z,
                newest[0])
        return (("ERROR:stale_target pick target (%.3f,%.3f,%.3f) is not "
                 "confirmed by any detection newer than the post-cancel arm "
                 "park (park_ts=%.3f); newest: %s; re-observe and call pick "
                 "with fresh coordinates" % (px, py, pz, since, newest_txt)),
                None)

    # -- bounded re-observe before a pick verification -----------------------

    def _detections_publisher_count(self) -> int:
        """Publishers on /object_detections_body; -1 == unknown (no ROS)."""
        return -1

    def _pick_det_reobserve(self, dispatch_ts: float) -> str:
        """G5: ``""`` when the channel is alive; else wait <= 4 s for a packet
        newer than the motion end and return the evidence tokens.

        The detection topic is event-driven (silence == empty scene), so a
        quiet channel after a pick is NOT a dead node by itself (MEASURED F4:
        yolo.log kept printing ``Candidate:`` lines through the 34.7 s gap).
        The gateway does NOT change the verdict here; it hands the verifier
        the freshest observation it can get and states what it saw.  No token
        contains ``ERROR:`` (see _await_pick_daemon's docstring).
        """
        clock, sleep = self._b5_clock, self._b5_sleep
        motion_end = clock()
        with self._obs_lock:
            alive = self._assembler.detections_alive(now=motion_end)
        if alive:
            return ""
        deadline = motion_end + PICK_DET_REOBSERVE_S
        while True:
            with self._obs_lock:
                ts = self._assembler.last_detections_ts
            now = clock()
            if ts > motion_end:
                wait_tok = "fresh@%.1fs" % (now - motion_end)
                break
            if now >= deadline:
                wait_tok = "still_silent@%.1fs" % (now - motion_end)
                break
            sleep(min(B5_POLL_S, deadline - now))
        try:
            pubs = int(self._detections_publisher_count())
        except Exception:  # noqa: BLE001
            pubs = -1
        seen = 1 if (ts > 0.0 and ts >= dispatch_ts) else 0
        return ("; det_wait=%s; det_publishers=%d; det_seen_since_dispatch=%d"
                % (wait_tok, pubs, seen))

    # -- the idle-held arm watchdog (no writer changes) ----------------------

    def _wd_read_writer_log(self, now: float) -> Tuple[str, Optional[Tuple],
                                                       float, float]:
        """(state, newest sample, hot span s, hold span s) from writer.log.

        ``state`` is ok | missing | stale (mtime older than
        WATCHDOG_LOG_MAX_AGE_S) | unparsed (fresh, but no status line in the
        tail).  Spans are 0.0 unless ``ok``.
        """
        path = self._wd_writer_log_path
        try:
            mtime = os.path.getmtime(path)
            size = os.path.getsize(path)
            with open(path, "rb") as fh:
                if size > WATCHDOG_LOG_TAIL_BYTES:
                    fh.seek(size - WATCHDOG_LOG_TAIL_BYTES)
                data = fh.read(WATCHDOG_LOG_TAIL_BYTES)
        except OSError:
            return "missing", None, 0.0, 0.0
        text = data.decode("utf-8", "replace")
        if size > WATCHDOG_LOG_TAIL_BYTES:
            text = text.split("\n", 1)[-1]  # drop the half-written first line
        samples = _parse_writer_status_lines(text)
        newest = samples[-1] if samples else None
        if now - mtime > WATCHDOG_LOG_MAX_AGE_S:
            return "stale", newest, 0.0, 0.0
        if newest is None:
            return "unparsed", None, 0.0, 0.0
        hot = _trailing_span(samples, lambda s: s[3] >= WATCHDOG_TMAX_C)
        hold = _trailing_span(samples, lambda s: s[4] == 1)
        return "ok", newest, hot, hold

    def _arm_watchdog_tick(self, now: float) -> Optional[str]:
        """One watchdog decision.  Returns the park reason, or None.

        TRIGGER (all): no arm task in flight AND no park running AND no arm
        activity the counter cannot see (_arm_busy_elsewhere) AND idle >=
        WATCHDOG_IDLE_GRACE_S AND at least one of [Tmax >= 90 C over the last
        5 s of writer samples] / [hold=1 for >= 8 s] / [a cancel or abort
        left the arm out: _arm_disturbed_ts newer than the last verified-safe
        park AND than the last time this watchdog saw that disturbance
        resolved] AND an arm outside SAFE_ARM_POSES (``q`` is read only once
        a conjunct holds; an unreadable ``q`` never moves the arm).  The idle
        gate is load-bearing: Tmax >= 100 C also occurs DURING legitimate
        engine slews (run20 up to 128 C, run17 up to 122 C), so the watchdog
        must never look at a busy arm.  Rate: WATCHDOG_COOLDOWN_S between
        attempts, WATCHDOG_MAX_ATTEMPTS, then ONE GIVE UP line.
        """
        state, newest, hot_span, hold_span = self._wd_read_writer_log(now)
        self._wd_log_state = state
        if newest is not None:
            self._wd_last_sample = {"t": newest[0], "arm": newest[1],
                                    "mode": newest[2], "Tmax": newest[3],
                                    "hold": newest[4]}
        with self._inflight_lock:
            inflight = self._arm_task_inflight
            idle_since = self._arm_idle_since
        if inflight > 0 or self._park_busy() or self._arm_busy_elsewhere(now):
            return None
        if now - idle_since < WATCHDOG_IDLE_GRACE_S:
            return None
        why = []
        if hot_span >= WATCHDOG_HOT_HOLD_S:
            why.append("hot")
        if hold_span >= WATCHDOG_RC_HOLD_S:
            why.append("hold")
        disturbed = self._arm_disturbed_ts
        if disturbed > max(self._arm_parked_ts,
                           self._wd_disturbed_seen_safe_ts):
            why.append("disturbed")
        if not why:
            self._wd_attempts = 0
            self._wd_gave_up = False
            return None
        q = self._read_arm_q()
        if q is None:
            return None
        if _arm_safe_pose(q) is not None:
            # looks safe (e.g. a bottle held in CARRY_TUCK): never disturbed, and a
            # disturbance seen resolved does not get re-armed until a newer one.
            if "disturbed" in why:
                self._wd_disturbed_seen_safe_ts = disturbed
            self._wd_attempts = 0
            self._wd_gave_up = False
            return None
        if self._wd_attempts >= WATCHDOG_MAX_ATTEMPTS:
            if not self._wd_gave_up:
                self._wd_gave_up = True
                sys.stderr.write(
                    "[arm-watchdog] GIVE UP after %d parks — OPERATOR: E-stop "
                    "/ kill the stack\n" % self._wd_attempts)
                sys.stderr.flush()
            return None
        if (self._wd_last_attempt_ts > 0.0
                and now - self._wd_last_attempt_ts < WATCHDOG_COOLDOWN_S):
            return None
        self._wd_attempts += 1
        self._wd_last_attempt_ts = now
        reason = "watchdog:" + "+".join(why)
        dev = _arm_pose_deviation(q)
        s = self._wd_last_sample or {}
        sys.stderr.write(
            "[arm-watchdog] PARK reason=%s Tmax=%sC hold=%s arm=%s "
            "q_dev=%.2frad attempt=%d/%d\n"
            % (reason, s.get("Tmax", "?"), s.get("hold", "?"),
               s.get("arm", "?"), dev if dev is not None else float("nan"),
               self._wd_attempts, WATCHDOG_MAX_ATTEMPTS))
        sys.stderr.flush()
        t = self._start_park_thread(reason, "first", False)
        t.join(WATCHDOG_PARK_JOIN_S)
        return reason

    def _arm_busy_elsewhere(self, now: float) -> bool:
        """Arm activity the dispatch counter cannot see: a FRESH /pick_state
        ``busy`` frame (the pick_node daemon keeps moving the arm after the
        engine's DONE, and can be driven without the gateway), a descent-lane
        body height outstanding (never swing an arm next to a table while the
        body is lowered), the /place_target lane (enters via its ROS
        callback), or a turn-around.  Unknown -> busy (the watchdog never
        moves an arm it cannot prove idle)."""
        with self._obs_lock:
            rec = self._pick_state_hold
            rec = dict(rec) if rec is not None else None
        if (rec is not None and rec.get("state") == PICK_STATE_BUSY
                and now - float(rec.get("ts", 0.0))
                <= self._pick_state_max_age_s):
            return True
        ctx = getattr(self, "_descent_ctx", None)
        if ctx is not None:
            try:
                if ctx.in_flight():
                    return True
            except Exception:  # noqa: BLE001
                return True
        pt_lock = getattr(self, "_place_target_lock", None)
        if pt_lock is not None and pt_lock.locked():
            return True
        return bool(getattr(self, "_turn_active", False))

    def start_arm_watchdog(self) -> None:
        """Start the G3 daemon thread (ROS wiring only; never the core ctor)."""
        if self._wd_thread is not None and self._wd_thread.is_alive():
            return
        self._wd_enabled = True
        self._wd_stop.clear()

        def _loop() -> None:
            while not self._wd_stop.wait(WATCHDOG_PERIOD_S):
                try:
                    self._arm_watchdog_tick(time.time())
                except Exception as exc:  # noqa: BLE001 — never die silently
                    sys.stderr.write("[arm-watchdog] tick error: %r\n" % (exc,))
                    sys.stderr.flush()

        self._wd_thread = threading.Thread(target=_loop, name="arm-watchdog",
                                           daemon=True)
        self._wd_thread.start()

    def stop_arm_watchdog(self) -> None:
        """Stop the G3 thread (node teardown)."""
        self._wd_stop.set()
        self._wd_enabled = False

    def _arm_safety_status(self) -> Dict[str, Any]:
        return {"watchdog": "on" if self._wd_enabled else "off",
                "writer_log": self._wd_log_state,
                "last_sample": self._wd_last_sample,
                "last_park": self._arm_safety.get("last_park"),
                "attempts": self._wd_attempts}

    # ── yolo_classes: the open-vocabulary request ─────────────────────────────
    #
    # the gateway publishes one /yolo/classes message per call (QoS
    # TRANSIENT_LOCAL, depth 1, RELIABLE, created eagerly in
    # _RosGateway.__init__ and only with the ROS param `open_vocab` exactly
    # true), then waits for the yolo node's /yolo/classes_state echo carrying
    # the same (epoch, seq). any outcome other than a matching applied state
    # becomes an ERROR:yolo_classes reply (ok:false), so a change the node
    # never confirmed is never reported as applied.

    #: the topics (std_msgs/String, JSON object).
    YOLO_CLASSES_TOPIC = "/yolo/classes"
    YOLO_CLASSES_STATE_TOPIC = "/yolo/classes_state"
    #: bounded wait (s) for the yolo node's subscription to match.
    _YOLO_CLASSES_SUB_WAIT_S = 1.0
    #: bounded wait (s) for the matching state echo to land. a bank swap takes
    #: milliseconds, and 3.0 s stays far inside the brain's socket timeout, so
    #: the brain always gets this verb's own honest answer.
    _YOLO_CLASSES_ACK_S = 3.0
    _YOLO_CLASSES_POLL_S = 0.05
    #: the backend's capacity limit on extra classes (20 base + up to 236 extras).
    _YOLO_CLASSES_MAX = 236
    _YOLO_MIN_CONF_RANGE = (0.20, 0.90)
    _YOLO_OPS = ("set", "reset", "query")
    _YOLO_LABEL_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
    _YOLO_PROMPT_RE = re.compile(r"[a-z][a-z0-9 \-]{0,47}")
    #: the actuator verbs a vocabulary swap must never overlap (counted in
    #: flight by dispatch, and only while `open_vocab` is true). wider than the
    #: verifier's motion set on purpose, since it covers every verb that
    #: commands legs, arms or hands. the conservative direction is a refusal.
    _YOLO_BUSY_VERBS = frozenset((
        "pick", "pick_left", "place", "navigate", "turn_around",
        "open_hand", "close_hand", "cmd_set", "writer", "grip", "engine",
        "return_home"))                 # these all move the body
    #: walk FSM phases where the body is still searching for or walking to a
    #: target. navigate's dispatch count can drop to 0 while the FSM still
    #: walks, and a swap that dropped the walk's extra-class target would
    #: blind it mid-approach. DETECTED is here even though the assembler
    #: folds it into SEARCHING anyway.
    _YOLO_WALK_BUSY_PHASES = frozenset(("SEARCHING", "DETECTED", "APPROACHING"))
    #: a busy walk phase stays busy until a terminal frame (REACHED / NOT_FOUND
    #: / IDLE) replaces it. this is only a staleness cap, for the case where the
    #: walk node dies without publishing IDLE, never a freshness window:
    #: APPROACHING re-sends every APPROACH_HEARTBEAT_S = 5.0 s, SEARCHING once
    #: per DETECT_WINDOW_S = 5.0 s window plus a ~1.05 s turn, while DETECTED is
    #: published once and then held silently until /walk_go, a retarget or an
    #: escape, unbounded while the target stays visible. 120.0 s > NAV_WAIT_S
    #: 115.0, so the cap can never release inside navigate's own wait.
    _YOLO_WALK_BUSY_STALE_CAP_S = 120.0

    def _yolo_vocab_setup(self, publisher: Any) -> None:
        """Arm the verb: called by _RosGateway.__init__ ONLY when the param
        ``open_vocab`` is exactly True, after the eager publisher exists.

        ``epoch`` is this gateway process's boot time in ns (§3.4): constant
        for the process, so a restarted gateway is always a NEW epoch at the
        node and its seq 1 is applied, never mistaken for a duplicate.
        """
        from collections import deque  # stdlib import
        self._classes_pub = publisher
        self._vocab_epoch = time.time_ns()
        self._vocab_seq = 0
        self._vocab_lock = threading.Lock()
        self._vocab_cond = threading.Condition(self._vocab_lock)
        self._vocab_state = None            # last /yolo/classes_state seen (dict)
        self._vocab_recent = deque(maxlen=16)
        self._yolo_call_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._motion_inflight: Dict[str, int] = {}

    def yolo_vocab_state(self) -> Optional[Dict[str, Any]]:
        """A copy of the last cached /yolo/classes_state, or None.

        None whenever the lane is off (param False) or no state has arrived.
        Read under the lock; the copy is the caller's (P5b's observe ``vocab``
        block reads through here).
        """
        lock = getattr(self, "_vocab_lock", None)
        if lock is None:
            return None
        with lock:
            state = self._vocab_state
            return dict(state) if isinstance(state, dict) else None

    def _yolo_vocab_store_state(self, state: Any) -> bool:
        """State-callback core: cache a parsed state dict and wake the waiter.

        Returns False (and caches nothing) for a non-dict payload.
        """
        if not isinstance(state, dict):
            return False
        cond = getattr(self, "_vocab_cond", None)
        if cond is None:
            return False
        with cond:
            self._vocab_state = dict(state)
            self._vocab_recent.append(dict(state))
            cond.notify_all()
        return True

    def _motion_enter(self, verb: str) -> str:
        with self._motion_lock:
            self._motion_inflight[verb] = self._motion_inflight.get(verb, 0) + 1
        return verb

    def _motion_exit(self, token: str) -> None:
        with self._motion_lock:
            n = self._motion_inflight.get(token, 0) - 1
            if n > 0:
                self._motion_inflight[token] = n
            else:
                self._motion_inflight.pop(token, None)

    def _yolo_busy_verb(self) -> Optional[str]:
        """Name of an actuator activity in flight, or None.

        The dispatch-counted verbs first, then the in-flight signals the
        cancel path (_do_cancel / _low_lane_cancel) reads: the descent lane's
        context, the turn-around sweep, and the /place_target lane's lock
        (that lane enters via its ROS callback, not through a tools socket).
        """
        lock = getattr(self, "_motion_lock", None)
        if lock is not None:
            with lock:
                for verb in sorted(self._motion_inflight):
                    if self._motion_inflight[verb] > 0:
                        return verb
        ctx = getattr(self, "_descent_ctx", None)
        if ctx is not None:
            try:
                if ctx.in_flight():
                    return "descent_lane"
            except Exception:  # noqa: BLE001 — unknown == busy (fail closed)
                return "descent_lane"
        if getattr(self, "_turn_active", False):
            return "turn_around"
        pt_lock = getattr(self, "_place_target_lock", None)
        if pt_lock is not None and pt_lock.locked():
            return "place_target"
        if self._yolo_walk_active():
            return "walk"
        return None

    def _yolo_walk_active(self) -> bool:
        """True while the last PARSED /walk_status phase is SEARCHING /
        DETECTED / APPROACHING and no terminal frame has replaced it, bounded
        by ``_YOLO_WALK_BUSY_STALE_CAP_S``.

        Read-only on the assembler's last parsed walk frame AND the
        assembler's own ``_last_walk_ts`` (assemble.py: moved only when
        parse_walk_status accepts the packet), so a malformed packet can
        never keep a latched phase alive.
        """
        obs_lock = getattr(self, "_obs_lock", None)
        assembler = getattr(self, "_assembler", None)
        if obs_lock is None or assembler is None:
            return False
        with obs_lock:
            walk = getattr(assembler, "_walk", None)
            ts = getattr(assembler, "_last_walk_ts", 0.0)
        phase = getattr(walk, "phase", None)
        if not isinstance(phase, str) or not ts:
            return False
        if phase.strip().upper() not in self._YOLO_WALK_BUSY_PHASES:
            return False
        return (time.time() - ts) <= self._YOLO_WALK_BUSY_STALE_CAP_S

    @classmethod
    def _yolo_classes_validate(cls, args: List
                               ) -> Tuple[Optional[Dict[str, Any]], str]:
        """Parse + validate a request (§3.4 rules).  ``(req, "")`` or
        ``(None, why)`` — a bad request never reaches ROS."""
        if len(args) != 1:
            return None, "expected one JSON object argument, got %d" % len(args)
        raw = args[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, "not JSON"
        if not isinstance(raw, dict):
            return None, "request must be a JSON object"
        unknown = sorted(set(raw) - {"op", "classes", "min_conf"})
        if unknown:
            return None, "unknown key %s" % unknown[0]
        op = raw.get("op")
        if op not in cls._YOLO_OPS:
            return None, "op must be one of set|reset|query, got %r" % (op,)
        classes = raw.get("classes")
        if classes is None:
            classes = []
        if not isinstance(classes, list):
            return None, "classes must be a list"
        if op != "set" and classes:
            return None, "op %s takes no classes" % op
        if len(classes) > cls._YOLO_CLASSES_MAX:
            return None, "too many classes (%d > %d)" % (
                len(classes), cls._YOLO_CLASSES_MAX)
        try:
            from brain.known_labels import TRACKED_CLASSES  # lazy import
        except Exception as exc:  # noqa: BLE001 — fail closed
            return None, "base vocabulary unavailable (%s)" % exc
        base = set(TRACKED_CLASSES) | {
            b.replace(" ", "_") for b in TRACKED_CLASSES}
        out: List[Dict[str, str]] = []
        seen = set()
        for entry in classes:
            if not isinstance(entry, dict):
                return None, "each class must be an object {label, prompt}"
            label, prompt = entry.get("label"), entry.get("prompt")
            if not isinstance(label, str) or not cls._YOLO_LABEL_RE.fullmatch(
                    label):
                return None, "bad label %r" % (label,)
            if label in base:
                return None, "label %r is a base class (always on)" % label
            if not isinstance(prompt, str) or not cls._YOLO_PROMPT_RE.fullmatch(
                    prompt):
                return None, "bad prompt %r for label %s" % (prompt, label)
            if label in seen:
                continue                     # duplicates by label: first wins
            seen.add(label)
            out.append({"label": label, "prompt": prompt})
        req: Dict[str, Any] = {"op": op, "classes": out}
        if "min_conf" in raw and raw["min_conf"] is not None:
            mc = raw["min_conf"]
            lo, hi = cls._YOLO_MIN_CONF_RANGE
            if (isinstance(mc, bool) or not isinstance(mc, (int, float))
                    or not math.isfinite(mc) or not lo <= mc <= hi):
                return None, "min_conf must be a number in [%.2f, %.2f], got %r" % (
                    lo, hi, mc)
            req["min_conf"] = float(mc)
        return req, ""

    def publish_yolo_classes(self, text: str) -> None:
        """Hook: publish one /yolo/classes String.  The rclpy-free core has
        no publisher; _RosGateway overrides this."""
        raise RuntimeError("no /yolo/classes publisher outside ROS")

    def _yolo_wait_subscriber(self) -> bool:
        """True once the yolo node's subscription has matched (<= 1.0 s).

        A publisher without ``get_subscription_count`` (an rclpy older than
        the API) skips the wait: the ack wait below is still the authority, so
        a missing node then answers ``no_ack`` — still an honest ERROR.
        """
        count = getattr(self._classes_pub, "get_subscription_count", None)
        if count is None:
            return True
        deadline = time.monotonic() + self._YOLO_CLASSES_SUB_WAIT_S
        while True:
            try:
                if count() > 0:
                    return True
            except Exception:  # noqa: BLE001 — treat as not matched
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._YOLO_CLASSES_POLL_S)

    @staticmethod
    def _yolo_is_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _yolo_wait_ack(self, seq: int, timeout_s: Optional[float] = None
                       ) -> Optional[Dict[str, Any]]:
        """The state echo with this process's (epoch, seq), or None after
        ``_YOLO_CLASSES_ACK_S``.  A state for any other (epoch, seq) is
        ignored."""
        deadline = time.monotonic() + (self._YOLO_CLASSES_ACK_S if timeout_s is None
                                       else timeout_s)
        with self._vocab_cond:
            while True:
                for st in reversed(self._vocab_recent):
                    if (self._yolo_is_int(st.get("epoch"))
                            and self._yolo_is_int(st.get("seq"))
                            and st.get("epoch") == self._vocab_epoch
                            and st.get("seq") == seq):
                        return dict(st)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._vocab_cond.wait(remaining)

    @staticmethod
    def _yolo_state_text(state: Dict[str, Any]) -> str:
        """``active=20+3 extras=a,b rejected=x:not_in_bank`` (defanged)."""
        base = state.get("base")
        extras = state.get("extras")
        extras = [str(e) for e in extras] if isinstance(extras, list) else []
        rejected = state.get("rejected")
        rej = ""
        if isinstance(rejected, dict):
            rej = ",".join("%s:%s" % (k, v) for k, v in rejected.items())
        text = "active=%s+%d extras=%s rejected=%s" % (
            len(base) if isinstance(base, list) else "?", len(extras),
            ",".join(extras), rej)
        return _defang_error_token(text.replace("\n", " "))[:400]

    def _do_yolo_classes(self, args: List) -> str:
        """``yolo_classes`` verb (§3.5).  Returns the reply text; no verdict."""
        if getattr(self, "_open_vocab", False) is not True:
            return ("ERROR:yolo_classes disabled (gateway param open_vocab is "
                    "false)")
        if not self._yolo_call_lock.acquire(blocking=False):
            # one request at a time: the /yolo/classes publisher is transient-local with
            # depth 1, so a second publish before the node takes the first could replace
            # it in the history (the node then never sees seq N, a false no_ack), and two
            # overlapping swaps would race on the node's single active vocabulary.
            return "ERROR:yolo_classes busy (yolo_classes in flight)"
        try:
            busy = self._yolo_busy_verb()
            if busy:
                return "ERROR:yolo_classes busy (%s in flight)" % busy
            req, why = self._yolo_classes_validate(list(args))
            if req is None:
                return "ERROR:yolo_classes bad_request:%s" % _defang_error_token(
                    why.replace("\n", " "))
            if not self._yolo_wait_subscriber():
                return ("ERROR:yolo_classes no_subscriber (yolo node not "
                        "running with OPENBOT_YOLO_CLASSES=1)")
            # re-checked after the (<= 1 s) subscriber wait: a motion that started in
            # the meantime still blocks the publish.
            busy = self._yolo_busy_verb()
            if busy:
                return "ERROR:yolo_classes busy (%s in flight)" % busy
            with self._vocab_cond:
                self._vocab_seq += 1
                seq = self._vocab_seq
                # a state cached for this epoch with seq >= the one about to be sent cannot
                # be an echo of it, since nothing with that seq is published, so drop it; it
                # can never ack this request.
                self._vocab_recent = type(self._vocab_recent)(
                    (st for st in self._vocab_recent
                     if not (st.get("epoch") == self._vocab_epoch
                             and self._yolo_is_int(st.get("seq"))
                             and st.get("seq") >= seq)),
                    maxlen=self._vocab_recent.maxlen)
            msg: Dict[str, Any] = {"v": 1, "epoch": self._vocab_epoch,
                                   "seq": seq, "op": req["op"],
                                   "classes": req["classes"]}
            if "min_conf" in req:
                msg["min_conf"] = req["min_conf"]
            msg["origin"] = "brain"
            try:
                self.publish_yolo_classes(json.dumps(msg))
            except Exception as exc:  # noqa: BLE001 — report, never raise
                return "ERROR:yolo_classes publish_failed:%s" % (
                    _defang_error_token(str(exc).replace("\n", " ")))
            state = self._yolo_wait_ack(seq)
            if state is None:
                return ("ERROR:yolo_classes no_ack seq=%d within %.1fs"
                        % (seq, self._YOLO_CLASSES_ACK_S))
            return self._yolo_reply(req["op"], seq, state)
        finally:
            self._yolo_call_lock.release()

    def _yolo_reply(self, op: str, seq: int, state: Dict[str, Any]) -> str:
        """Map a matching state echo to the verb's reply (§3.5 step 5)."""
        error = state.get("error")
        if op == "query":
            if error:
                return "ERROR:yolo_classes state_error:%s" % (
                    _defang_error_token(str(error).replace("\n", " "))[:200])
            return "DONE:state seq=%d backend=%s %s" % (
                seq, _defang_error_token(str(state.get("backend"))),
                self._yolo_state_text(state))
        backend = state.get("backend")
        if backend == "yolo":
            return ("ERROR:yolo_classes backend_fixed (node runs the "
                    "fixed-vocabulary engine)")
        if backend != "yoloe":
            return "ERROR:yolo_classes bad_state:backend=%s" % (
                _defang_error_token(str(backend))[:60])
        if error:
            return "ERROR:yolo_classes swap_failed:%s" % (
                _defang_error_token(str(error).replace("\n", " "))[:200])
        if state.get("dup") is True:
            return "DONE:applied (dup) seq=%d %s" % (
                seq, self._yolo_state_text(state))
        return "DONE:applied seq=%d %s" % (seq, self._yolo_state_text(state))

    # ── strict detection: holds on /yolo/strict ──────────────────────────────
    # one hold narrows the yolo node to [targets] + KEEP_LABELS ('person') for one
    # phase: navigate (opened around _do_navigate, ends with the walk) or look_for
    # / place / pick target resolution (the brain, via yolo_strict). the node
    # keeps the operator's full vocabulary and restores it on 'release', or when
    # its lease (LEASE_S, renewed every RENEW_S by start_strict_renew) runs out.
    # never motion, never a verdict.
    YOLO_STRICT_TOPIC = "/yolo/strict"

    def _strict_setup(self, publisher: Any) -> None:
        """Arm strict detection: _RosGateway.__init__ ONLY, with open_vocab
        True and OPENBOT_STRICT_VOCAB not '0'.  Absent = every path inert."""
        from gateway import strict_vocab as strict_mod  # lazy, needs package ctx
        self._strict_mod = strict_mod
        self._strict_pub = publisher
        self._strict_lock = threading.Lock()
        self._strict_sent = False           # true when the last publish narrowed
        self._strict_stop = threading.Event()
        self._strict_holds = strict_mod.StrictHolds()

    def publish_yolo_strict(self, text: str) -> None:
        """Hook: publish one /yolo/strict String (_RosGateway overrides)."""
        raise RuntimeError("no /yolo/strict publisher outside ROS")

    @staticmethod
    def _strict_log(text: str) -> None:
        try:
            sys.stderr.write("[strict] %s\n" % text)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- a log line never breaks a verb
            pass

    def _strict_wide_labels(self) -> set:
        """Labels a hold may name: the 20 base classes + the node's extras
        (from the last cached /yolo/classes_state)."""
        wide = set()
        try:
            from brain.known_labels import TRACKED_CLASSES  # lazy import
            wide.update(TRACKED_CLASSES)
        except Exception:  # noqa: BLE001 -- fail closed: no base labels
            pass
        state = self.yolo_vocab_state()
        if isinstance(state, dict):
            for key in ("base", "extras"):
                vals = state.get(key)
                if isinstance(vals, list):
                    wide.update(v for v in vals if isinstance(v, str))
        return wide

    def _strict_sync(self) -> Optional[int]:
        """Publish what the open holds need NOW: 'narrow' with their union
        (this is also the renewal) or ONE 'release' after the last hold
        closed.  Returns the published seq, None when nothing was sent."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None:
            return None
        with self._strict_lock:
            labels = holds.labels()
            if not labels and not self._strict_sent:
                return None
            with self._vocab_cond:
                self._vocab_seq += 1
                seq = self._vocab_seq
            msg: Dict[str, Any] = {"v": 1, "epoch": self._vocab_epoch,
                                   "seq": seq,
                                   "op": "narrow" if labels else "release",
                                   "origin": "gateway"}
            if labels:
                msg["labels"] = list(labels)
                msg["lease_s"] = self._strict_mod.LEASE_S
            try:
                self.publish_yolo_strict(json.dumps(msg))
            except Exception as exc:  # noqa: BLE001 -- the lease still bounds it
                self._strict_log("publish failed: %s" % exc)
                return None
            self._strict_sent = bool(labels)
            return seq

    @staticmethod
    def _strict_stamp(state: Any) -> Optional[float]:
        stamp = state.get("stamp") if isinstance(state, dict) else None
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            return float(stamp)
        return None

    def _strict_open(self, owner: str, targets: List[str],
                     max_s: float) -> Tuple[Optional[int], float, str]:
        """Open a hold on the targets that are in the wide vocabulary and
        wait (<= ACK_S) for the node's echo.  (hold id, applied_ts, '') or
        (None, 0.0, why); a hold the node did not confirm is closed again."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None:
            return None, 0.0, "disabled"
        wide = self._strict_wide_labels()
        keep = [t for t in targets if t in wide]
        if not keep:
            return None, 0.0, "not_in_vocabulary (%s)" % ",".join(targets)
        hid = holds.open(owner, keep, max_s)
        seq = self._strict_sync()
        state = (self._yolo_wait_ack(seq, timeout_s=self._strict_mod.ACK_S)
                 if seq is not None else None)
        active = state.get("strict") if isinstance(state, dict) else None
        if (state is None or state.get("error")
                or not isinstance(active, list)
                or not all(t in active for t in keep)):
            holds.close(hid)
            self._strict_sync()
            why = ("no_ack (seq=%s within %.1fs)" % (seq, self._strict_mod.ACK_S)
                   if state is None else "refused (%s)" % _defang_error_token(
                       str(state.get("error") or "not applied"))[:120])
            self._strict_log("%s %s: no hold, %s" % (owner, ",".join(keep), why))
            return None, 0.0, why
        ts = self._strict_stamp(state)
        ts = time.time() if ts is None else ts
        self._strict_log("hold %d OPEN %s labels=%s applied_ts=%.3f"
                         % (hid, owner, ",".join(active), ts))
        return hid, ts, ""

    def _strict_wait_packet(self, after_ts: float, max_s: float) -> bool:
        """True once a detections packet ARRIVED at or after ``after_ts``
        (gateway clock), polling every POLL_S for at most ``max_s``.  Reads
        the assembler stamp only (no grip refresh, unlike observe)."""
        deadline = time.monotonic() + max(0.0, max_s)
        while True:
            with self._obs_lock:
                ts = self._assembler.last_detections_ts
            if ts >= after_ts:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._strict_mod.POLL_S)

    def _strict_close(self, hid: Optional[int], wait: bool,
                      why: str) -> Optional[float]:
        """Close one hold and publish the new union (or the release).  With
        ``wait`` returns the node's stamp of that change (None: no echo)."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None or hid is None or not holds.close(hid):
            return None
        seq = self._strict_sync()
        self._strict_log("hold %d CLOSED (%s)" % (hid, why))
        if not wait or seq is None:
            return None
        return self._strict_stamp(
            self._yolo_wait_ack(seq, timeout_s=self._strict_mod.ACK_S))

    def _strict_close_all(self, why: str) -> int:
        """cancel / stop / navigate 'stop' / shutdown: close every hold."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None:
            return 0
        n = holds.close_all()
        if n:
            self._strict_log("%d hold(s) CLOSED (%s)" % (n, why))
        self._strict_sync()
        return n

    def _strict_tick(self) -> None:
        """Every RENEW_S (start_strict_renew; tests call it directly): close
        capped holds and done navigate holds whose walk ended, then renew
        (or release).  Never waits for an echo."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None:
            return
        try:
            walk_active = self._yolo_walk_active()
        except Exception:  # noqa: BLE001 -- unknown == still walking (capped)
            walk_active = True
        for hid, owner, why in holds.expire(walk_active):
            self._strict_log("hold %d CLOSED %s (%s)" % (hid, owner, why))
        self._strict_sync()

    def start_strict_renew(self) -> None:
        """The renewal thread (ROS wiring only, like start_arm_watchdog)."""
        if getattr(self, "_strict_holds", None) is None:
            return

        def _loop() -> None:
            while not self._strict_stop.wait(self._strict_mod.RENEW_S):
                try:
                    self._strict_tick()
                except Exception as exc:  # noqa: BLE001 -- never die silently
                    self._strict_log("tick error: %r" % (exc,))

        threading.Thread(target=_loop, name="strict-renew",
                         daemon=True).start()

    def _strict_nav_open(self, args: List) -> Optional[int]:
        """navigate: open a hold on the bare target label ('cup#165@188' ->
        'cup'); 'stop' closes every hold instead; malformed -> None."""
        if getattr(self, "_strict_holds", None) is None:
            return None
        if len(args) != 1 or not isinstance(args[0], str) or not args[0].strip():
            return None
        label = args[0].strip()
        if label.lower() == "stop":
            try:
                self._strict_close_all("navigate_stop")
            except Exception as exc:  # noqa: BLE001 -- the stop always runs
                self._strict_log("close_all (navigate_stop) failed: %r; "
                                 "stopping anyway" % (exc,))
            return None
        if "#" in label or "@" in label:
            parts = self._nav_split(label)
            if parts is None:
                return None
            label = parts[0]
        hid, _ts, _why = self._strict_open("navigate", [label.strip().lower()],
                                           self._strict_mod.NAV_HOLD_MAX_S)
        return hid

    def _strict_nav_done(self, hid: Optional[int]) -> None:
        """navigate returned: close now if the walk is over, else the tick
        closes it when the walk ends (or at NAV_HOLD_MAX_S)."""
        holds = getattr(self, "_strict_holds", None)
        if holds is None or hid is None:
            return
        holds.mark_done(hid)
        if not self._yolo_walk_active():
            self._strict_close(hid, wait=False, why="navigate returned")

    def _do_navigate_strict(self, args: List) -> Tuple[str, Optional[str]]:
        """B8i: _do_navigate (unchanged) inside a navigate strict hold.
        B8j: a navigate (also 'stop') ends the container lock first."""
        self._lock_clear("navigate")
        hid = self._strict_nav_open(args)
        ns = self._nav_strict_begin(args, hid)      # begin the nav
        try:
            return self._do_navigate(args)
        finally:
            self._strict_nav_done(self._nav_strict_end(ns, hid))

    def _do_yolo_strict(self, args: List) -> str:
        """``yolo_strict`` verb (B8i, the brain's look_for / place / pick).
        begin {labels, owner[, wait_s]} -> 'DONE:strict hold=<n> owner=<o>
        applied_ts=<t> fresh=yes|no labels=<csv>' once a packet arrived
        SETTLE_S after the swap (<= SETTLE_MAX_S); end {hold[, wait_s]} ->
        'DONE:strict released hold=<n> released_ts=<t> fresh=yes|no' once one
        arrived REFILL_S after the restore (<= REFILL_MAX_S).  wait_s (0-5)
        lowers both bounds.  Refusals: 'ERROR:yolo_strict <why>'."""
        if getattr(self, "_strict_holds", None) is None:
            return ("ERROR:yolo_strict disabled (needs open_vocab:=true and "
                    "OPENBOT_STRICT_VOCAB not 0)")
        if len(args) != 1:
            return "ERROR:yolo_strict bad_request:expected one JSON object"
        raw = args[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return "ERROR:yolo_strict bad_request:not JSON"
        if not isinstance(raw, dict):
            return "ERROR:yolo_strict bad_request:not a JSON object"
        m = self._strict_mod
        wait = raw.get("wait_s")
        if wait is not None and (isinstance(wait, bool)
                                 or not isinstance(wait, (int, float))
                                 or not 0.0 <= wait <= 5.0):
            return "ERROR:yolo_strict bad_request:wait_s must be 0-5"
        op = raw.get("op")
        if op == "end":
            hid = raw.get("hold")
            if isinstance(hid, bool) or not isinstance(hid, int) or hid < 1:
                return "ERROR:yolo_strict bad_request:hold must be an int >= 1"
            ts = self._strict_close(hid, wait=True, why="end")
            ts = time.time() if ts is None else ts
            fresh = self._strict_wait_packet(ts + m.REFILL_S, m.REFILL_MAX_S
                                             if wait is None else
                                             min(float(wait), m.REFILL_MAX_S))
            return "DONE:strict released hold=%d released_ts=%.3f fresh=%s" % (
                hid, ts, "yes" if fresh else "no")
        if op != "begin":
            return "ERROR:yolo_strict bad_request:op must be begin|end"
        owner = raw.get("owner")
        if owner not in self._strict_mod.BRAIN_OWNERS:
            return "ERROR:yolo_strict bad_request:owner must be look_for|place|pick"
        labels = raw.get("labels")
        if (not isinstance(labels, list) or not labels or len(labels) > 6
                or not all(isinstance(x, str) and x.strip() for x in labels)):
            return "ERROR:yolo_strict bad_request:labels must be 1-6 strings"
        if owner == "place":
            # the narrowing dropped the locked bin's track (its label had turned into
            # 'backpack') and silenced the detector for 3.1 s.
            skip = self._lock_strict_skip([x.strip().lower() for x in labels])
            if skip:
                return "ERROR:yolo_strict skipped (%s)" % skip
        hid, ts, why = self._strict_open(
            owner, [x.strip().lower() for x in labels], m.BRAIN_HOLD_MAX_S)
        if hid is None:
            return "ERROR:yolo_strict %s" % why
        fresh = self._strict_wait_packet(ts + m.SETTLE_S, m.SETTLE_MAX_S
                                         if wait is None else
                                         min(float(wait), m.SETTLE_MAX_S))
        return "DONE:strict hold=%d owner=%s applied_ts=%.3f fresh=%s labels=%s" % (
            hid, owner, ts, "yes" if fresh else "no",
            ",".join(self._strict_holds.labels()))

    # ── the container lock and creep (the pure half) ─────────────────────────
    # the navigated container is kept by identity (instance id across label
    # changes, otherwise geometry), a locked container's place never narrows the
    # detector (once it dropped the bin's track and the place refused
    # NOT_DETECTED), and creep covers the last few centimetres in short, bounded
    # cmd_set pulses. switches (gateway terminal):
    # OPENBOT_CONTAINER_LOCK=0 -> no lock; OPENBOT_CREEP=0 -> creep refused;
    # OPENBOT_CREEP_EDGE_M sets the target near edge (0.25).
    LOCK_ROWS_MAX_AGE_S = 2.0       # REACHED arms on a packet no older than this
    LOCK_PENDING_S = 3.0            # ... otherwise on the next packet inside this

    def _lock_setup(self) -> None:
        """__init__ (B8j): the lock, its row cache and its clock seam.  An
        import failure leaves every B8j path inert (B8i behaviour)."""
        self._container_lock = None
        self._lock_m = None
        self._lock_rows: List[dict] = []
        self._lock_rows_ts = 0.0
        self._lock_pending = None
        self._lock_armed_epoch = None
        self._lock_clock = time.time
        try:
            from gateway import strict_vocab as lock_mod  # lazy, needs package ctx
            self._lock_m = lock_mod
            self._container_lock = lock_mod.ContainerLock()
        except Exception as exc:  # noqa: BLE001 -- degrade, never raise
            self._container_lock = None
            self._lock_m = None
            print("[lock] container lock import FAILED (%r) - B8j lock/creep "
                  "OFF" % (exc,), flush=True)

    def _lock_on(self) -> bool:
        mod = getattr(self, "_lock_m", None)
        return (mod is not None
                and getattr(self, "_container_lock", None) is not None
                and mod.env_on(mod.LOCK_ENV))

    @staticmethod
    def _lock_log(text: str) -> None:
        try:
            sys.stderr.write("[lock] %s\n" % text)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- a log line never breaks a verb
            pass

    @staticmethod
    def _creep_log(text: str) -> None:
        try:
            sys.stderr.write("[creep] %s\n" % text)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- a log line never breaks a verb
            pass

    def _lock_now(self) -> float:
        return float(getattr(self, "_lock_clock", time.time)())

    def _lock_clear(self, why: str) -> None:
        """Any robot motion other than the creep ends the lock."""
        lock = getattr(self, "_container_lock", None)
        if lock is None:
            return
        self._lock_pending = None
        try:
            ev = lock.clear(why)
        except Exception:  # noqa: BLE001 -- never blocks the verb
            ev = None
        if ev:
            self._lock_log(ev)

    def _lock_ingest(self, rows: List[dict]) -> None:
        """on_detections_packet: cache the RAW rows (they carry depth_m) and
        follow the lock; arm a pending REACHED."""
        lock = getattr(self, "_container_lock", None)
        if lock is None:
            return
        now = self._lock_now()
        raw = [dict(r) for r in rows if isinstance(r, dict)]
        with self._obs_lock:
            self._lock_rows = raw
            self._lock_rows_ts = now
        pend = self._lock_pending
        if pend is not None:
            self._lock_pending = None
            label, ev, t_req = pend
            if self._lock_on() and now - t_req <= self.LOCK_PENDING_S:
                msg = lock.arm(label, raw, now, ev)
                self._lock_log(msg if msg else
                               "NOT ARMED %s (no row matches the walk's stop)"
                               % label)
                return
        try:
            events = lock.update(raw, now)
        except Exception as exc:  # noqa: BLE001 -- the packet path never raises
            events = ["update failed: %r" % (exc,)]
        for ev in events:
            self._lock_log(ev)

    def _lock_on_walk(self, phase_raw: Optional[str], label: Optional[str],
                      payload: Any) -> None:
        """on_walk_status_packet: a walk that moves again ends the lock; the
        REACHED of THIS navigate's container label arms it (once)."""
        lock = getattr(self, "_container_lock", None)
        mod = getattr(self, "_lock_m", None)
        if lock is None or mod is None or not phase_raw:
            return
        if phase_raw in ("SEARCHING", "DETECTED", "APPROACHING"):
            if lock.armed():
                self._lock_clear("walk_" + phase_raw.lower())
            return
        if phase_raw != "REACHED" or not self._lock_on():
            return
        if label not in mod.CONTAINER_CLASSES:
            return
        now = self._lock_now()
        with self._obs_lock:
            epoch_ts, epoch_label = self._nav_epoch
            rows = list(self._lock_rows)
            rows_ts = self._lock_rows_ts
        if epoch_ts <= 0.0 or epoch_label != label:
            return
        if self._lock_armed_epoch == epoch_ts:
            return                      # REACHED gets re-published at 1 Hz
        self._lock_armed_epoch = epoch_ts
        detail = payload.get("detail") if isinstance(payload, dict) else None
        ev = detail.get("stop_law") if isinstance(detail, dict) else None
        if rows and now - rows_ts <= self.LOCK_ROWS_MAX_AGE_S:
            msg = lock.arm(label, rows, now, ev)
            if msg:
                self._lock_log(msg)
                return
        self._lock_pending = (label, ev, now)
        self._lock_log("ARM PENDING %s (no matching fresh row yet)" % label)

    def _lock_strict_skip(self, labels: List[str]) -> Optional[str]:
        """yolo_strict begin owner=place: the reason the hold is SKIPPED (the
        locked container is one of its labels), else None."""
        lock = getattr(self, "_container_lock", None)
        if lock is None or not self._lock_on():
            return None
        snap = lock.snapshot(self._lock_now())
        if snap is None or snap["label"] not in labels:
            return None
        why = ("container lock %s#%s det=%s row_age=%.1fs: the navigated "
               "container is held by identity, no narrowing"
               % (snap["label"], snap["iid"], snap["det_label"],
                  snap["row_age_s"]))
        self._lock_log("place strict hold SKIPPED (%s)" % why)
        return why

    # ── navigate widening + the fetch start point ─────────────────────────────
    # a navigate hold kept on {bin, person} stays narrowed for the whole walk, so
    # the bin's carrier labels (backpack 0.67, suitcase 0.64 on the same track)
    # are gone and a narrowed window can see no bin at all. narrowing helps find
    # the label; from APPROACHING on, the walk follows the object by identity, so
    # the hold is closed (the tracker keeps the id) and re-opened if the walk
    # falls back to the sweep. OPENBOT_NAV_WIDEN=0 keeps narrowing throughout.
    NAV_WIDEN_ENV = "OPENBOT_NAV_WIDEN"
    RETURN_WAIT_S = 130.0           # a pose walk cannot exceed 120 s
    RETURN_START_S = 5.0            # no RETURNING this fast means the walk never started
    HOME_ROWS_MAX_AGE_S = 1.0       # the requester comes out of a packet this fresh

    def _b8l_setup(self) -> None:
        """__init__ (B8l): home anchor, odometry slot, pose-walk wait state,
        navigate widening state.  An import failure leaves home OFF."""
        self._home = None
        self._home_m = None
        self._odom_pose = None
        self._odom_lock = threading.Lock()
        self._pose_active = False
        self._pose_events: List[tuple] = []
        self._pose_lock = threading.Lock()
        self._b8l_rows = (0.0, [])
        self._nav_strict = None
        self._nav_strict_lock = threading.Lock()
        try:
            from gateway import home_anchor as home_mod  # lazy, needs package ctx
            self._home_m = home_mod
            self._home = home_mod.HomeAnchor()
        except Exception as exc:  # noqa: BLE001 -- degrade, never raise
            self._home = None
            self._home_m = None
            print("[home] home anchor import FAILED (%r) - return_home OFF"
                  % (exc,), flush=True)

    @staticmethod
    def _home_log(text: str) -> None:
        try:
            sys.stderr.write("[home] %s\n" % text)
            sys.stderr.flush()
        except Exception:  # noqa: BLE001 -- a log line never breaks a verb
            pass

    def on_odom(self, x: float, y: float, yaw: float) -> None:
        """One /unitree/slam_mapping/odom pose (x, y, yaw; REP-103)."""
        try:
            pose = (float(x), float(y), float(yaw))
        except (TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in pose):
            return
        lock = getattr(self, "_odom_lock", None)
        if lock is None:
            return
        with lock:
            self._odom_pose = (pose, time.time())

    def _odom_now(self, max_age_s: float) -> Optional[tuple]:
        """(x, y, yaw) when the last odometry pose is at most ``max_age_s``
        old, else None."""
        lock = getattr(self, "_odom_lock", None)
        if lock is None:
            return None
        with lock:
            slot = self._odom_pose
        if slot is None or time.time() - slot[1] > max_age_s:
            return None
        return slot[0]

    def _nav_bare_label(self, args: List) -> Optional[str]:
        if len(args) != 1 or not isinstance(args[0], str):
            return None
        label = args[0].strip()
        if not label or label.lower() == "stop":
            return None
        if "#" in label or "@" in label:
            parts = self._nav_split(label)
            if parts is None:
                return None
            label = parts[0]
        return label.strip().lower()

    def _nav_strict_begin(self, args: List, hid: Optional[int]):
        """F2: track this navigate's hold so the walk status can widen /
        re-narrow it.  None (nothing tracked) without a hold or switched off."""
        lk = getattr(self, "_nav_strict_lock", None)
        if hid is None or lk is None:
            return None
        if os.environ.get(self.NAV_WIDEN_ENV, "").strip() == "0":
            return None
        label = self._nav_bare_label(args)
        if label is None:
            return None
        ns = {"label": label, "hid": hid, "widened": 0, "narrowed": 0}
        with lk:
            self._nav_strict = ns
        return ns

    def _nav_strict_end(self, ns, hid: Optional[int]) -> Optional[int]:
        """F2: the navigate returned -> the hold it holds NOW (None when it
        is widened), which _strict_nav_done then hands to the walk's end."""
        if ns is None:
            return hid
        with self._nav_strict_lock:
            if self._nav_strict is ns:
                self._nav_strict = None
            cur = ns["hid"]
        if ns["widened"]:
            self._strict_log("navigate %s returned (widened %d, re-narrowed %d)"
                             % (ns["label"], ns["widened"], ns["narrowed"]))
        return cur

    def _strict_on_walk(self, phase_raw: Optional[str], label: Optional[str],
                        payload: Any) -> None:
        """F2 (walk-status callback, never blocks): APPROACHING for the
        navigate's label closes its hold; any SEARCHING after that (the
        approach ended: a re-sweep, a retarget, the sweep's own frames)
        re-opens it.  The walk's reacquire speaks APPROACHING, so it stays
        wide while it looks for the object it lost."""
        lk = getattr(self, "_nav_strict_lock", None)
        holds = getattr(self, "_strict_holds", None)
        if lk is None or holds is None or not phase_raw:
            return
        with lk:
            ns = self._nav_strict
            if ns is None or label != ns["label"]:
                return
            if phase_raw == "APPROACHING" and ns["hid"] is not None:
                act, hid = "widen", ns["hid"]
                ns["hid"] = None
                ns["widened"] += 1
            elif phase_raw == "SEARCHING" and ns["hid"] is None:
                act, hid = "narrow", None
            else:
                return
        try:
            if act == "widen":
                self._strict_close(hid, wait=False,
                                   why="walk APPROACHING: the walk follows %s "
                                   "by identity, full vocabulary back" % label)
                return
            new = holds.open("navigate", [label], self._strict_mod.NAV_HOLD_MAX_S)
            with lk:
                keep = self._nav_strict is ns
                if keep:
                    ns["hid"] = new
                    ns["narrowed"] += 1
            if not keep:
                holds.close(new)            # the navigate has already returned
                return
            self._strict_sync()
            self._strict_log("hold %d RE-OPENED navigate labels=%s (walk re-sweep)"
                             % (new, ",".join(holds.labels())))
        except Exception as exc:  # noqa: BLE001 -- the status path never raises
            self._strict_log("navigate widen/narrow failed: %r" % (exc,))

    def _home_rows(self) -> List[dict]:
        ts, rows = getattr(self, "_b8l_rows", (0.0, []))
        if time.time() - ts > self.HOME_ROWS_MAX_AGE_S:
            return []
        return [r for r in rows if isinstance(r, dict)]

    def _do_home(self, args: List) -> str:
        """``home`` verb (B8l, no motion): one JSON object {op, why}.
        op status (default) -> 'DONE:home {...}' | 'DONE:home none (<why>)';
        mark / auto -> 'DONE:home marked {...}' | 'DONE:home kept {...}' |
        'DONE:home none (<why>)'; clear -> 'DONE:home cleared' | 'DONE:home
        none (no start point)'.  'ERROR:home bad_request:<why>' otherwise."""
        home = getattr(self, "_home", None)
        if home is None:
            return "DONE:home none (unavailable)"
        req: Any = {}
        if args:
            if len(args) != 1:
                return "ERROR:home bad_request:expected one JSON object"
            req = args[0]
            if isinstance(req, str):
                try:
                    req = json.loads(req) if req.strip() else {}
                except ValueError:
                    return "ERROR:home bad_request:not JSON"
            if not isinstance(req, dict):
                return "ERROR:home bad_request:not a JSON object"
        op = str(req.get("op") or "status").strip().lower()
        now = time.time()
        pose = self._odom_now(self._home_m.ODOM_FRESH_S)
        if op == "status":
            snap = home.snapshot(pose, now)
            if snap is None:
                return "DONE:home none (no start point recorded)"
            snap["odom_fresh"] = pose is not None
            return "DONE:home " + json.dumps(snap, sort_keys=True)
        if op in ("mark", "auto"):
            if pose is None:
                self._home_log("%s refused: odometry stale" % op)
                return ("DONE:home none (odometry stale: the start point was "
                        "NOT recorded)")
            rows = self._home_rows()
            if op == "mark":
                ev, marked = home.mark(pose, rows, now, src="mark"), True
            else:
                marked, ev = home.auto(pose, rows, now)
            if ev:
                self._home_log(ev)
            snap = home.snapshot(pose, now) or {}
            return "DONE:home %s %s" % ("marked" if marked else "kept",
                                        json.dumps(snap, sort_keys=True))
        if op in ("candidate", "promote"):     # candidate/promote ops
            if pose is None:
                self._home_log("%s refused: odometry stale" % op)
                return ("DONE:home none (odometry stale: the start point was "
                        "NOT recorded)")
            rows = self._home_rows()
            if op == "candidate":
                ev = home.candidate(pose, rows, now)
                if ev:
                    self._home_log(ev)
                return "DONE:home candidate"
            ev = home.promote(pose, rows, now)
            if ev:
                self._home_log(ev)
            snap = home.snapshot(pose, now) or {}
            return "DONE:home marked " + json.dumps(snap, sort_keys=True)
        if op == "picked":
            ev = home.set_held(now)
            if ev:
                self._home_log(ev)
                return "DONE:home held"
            return "DONE:home none (no start point)"
        if op == "clear":
            ev = home.clear(str(req.get("why") or "brain")[:40])
            if ev:
                self._home_log(ev)
                return "DONE:home cleared"
            return "DONE:home none (no start point)"
        return ("ERROR:home bad_request:op must be "
                "status|mark|auto|candidate|promote|picked|clear")

    def _home_stop(self, why: str) -> None:
        """B8l (operator decision 2026-10-17): a stop / cancel ends a fetch that
        has not picked yet, and only pauses one carrying the object (never
        raises)."""
        home = getattr(self, "_home", None)
        if home is None:
            return
        try:
            ev = home.stop(why, time.time())
        except Exception:  # noqa: BLE001 -- never blocks the verb
            ev = None
        if ev:
            self._home_log(ev)

    #: return_home: after the coordinator reset, wait this long before the pose
    #: goal, so its relayed /walk_target 'stop' lands first.
    RETURN_COORD_SETTLE_S = 0.5

    def publish_walk_pose(self, text: str) -> None:
        """Hook: publish one /walk_pose String (_RosGateway overrides)."""
        raise RuntimeError("no /walk_pose publisher outside ROS")

    def _do_return_home(self, args: List) -> str:
        """``return_home`` verb (B8l F7, motion, BLOCKS <= RETURN_WAIT_S):
        walk_node walks to the recorded start point on odometry and faces the
        requester (or the start heading).  'DONE return_home: ...' on
        HOME_REACHED; 'ERROR:return_home <why>' otherwise."""
        home = getattr(self, "_home", None)
        if home is None:
            return "ERROR:return_home unavailable (no home anchor module)"
        if args:
            return "ERROR:return_home takes no arguments"
        now = time.time()
        pose = self._odom_now(self._home_m.ODOM_FRESH_S)
        if pose is None:
            return ("ERROR:return_home odometry is stale (no fresh "
                    "/unitree/slam_mapping/odom): the start point cannot be "
                    "found")
        goal = home.goal(now)
        if goal is None:
            return ("ERROR:return_home no start point recorded (the first "
                    "navigate of a fetch request - 'get me ...', 'bring me ...' - "
                    "with an empty hand records it; it expires after 15 min)")
        snap = home.snapshot(pose, now) or {}
        # the walk's obstacle stop needs the detector, and an empty view and a dead
        # node look the same, so refuse to start when it has been silent for the
        # assembler's whole liveness window.
        with self._obs_lock:
            alive = self._assembler.detections_alive()
        if not alive:
            return ("ERROR:return_home the detector has been silent for more "
                    "than 10 s: the walk could not see a person in its way. "
                    "Nothing moved.")
        if not self._pose_lock.acquire(blocking=False):
            return "ERROR:return_home busy (a return_home is already running)"
        try:
            with self._obs_lock:
                self._pose_events = []
                self._pose_active = True
            # a coordinator watch left over from an earlier navigate could time out
            # mid-return and send its own 'stop', so reset it first; its relayed
            # /walk_target 'stop' then finds the walk idle.
            try:
                self.stop_walk()
            except Exception as exc:  # noqa: BLE001 -- the return still runs
                self._home_log("coordinator reset failed: %r" % (exc,))
            time.sleep(self.RETURN_COORD_SETTLE_S)
            t_sent = time.time()
            try:
                self.publish_walk_pose(json.dumps(goal, sort_keys=True))
            except Exception as exc:  # noqa: BLE001
                return "ERROR:return_home publish failed: %s" % exc
            self._home_log("RETURN sent goal=%s (%.2f m away)"
                           % (json.dumps(goal, sort_keys=True),
                              snap.get("robot_to_home_m", -1.0)))
            started = False
            seen = 0
            while True:
                time.sleep(0.1)
                with self._obs_lock:
                    events = list(self._pose_events)
                for phase, _lab, payload, ts in events[seen:]:
                    if ts < t_sent:
                        continue
                    det = payload.get("detail") if isinstance(payload, dict) else None
                    det = det if isinstance(det, dict) else {}
                    if phase == "RETURNING":
                        started = True
                    elif phase == "HOME_REACHED":
                        faced = det.get("faced") is True
                        face = ("the person who asked" if goal.get("face_x") is not None
                                else "the start heading")
                        if det.get("requester_ahead_m") is not None:
                            return ("DONE return_home: stopped in front of the "
                                    "person who asked, %.2f m short of the start "
                                    "point; %s them; HOME_ERR_M=%.3f; "
                                    "REQUESTER_SEEN=yes; REQUESTER_AHEAD=yes; "
                                    "PERSON_RADIUS_M=%.1f"
                                    % (float(det.get("requester_ahead_m")),
                                       "facing" if faced else "could not fully turn to",
                                       float(det.get("err_m") or 0.0),
                                       snap.get("radius_m", 0.0)))
                        return ("DONE return_home: back at the start point "
                                "(%.2f m from it by odometry); %s %s; "
                                "HOME_ERR_M=%.3f; REQUESTER_SEEN=%s; "
                                "PERSON_RADIUS_M=%.1f"
                                % (float(det.get("err_m") or 0.0),
                                   "facing" if faced else "could not fully turn to",
                                   face, float(det.get("err_m") or 0.0),
                                   "yes" if goal.get("face_x") is not None else "no",
                                   snap.get("radius_m", 0.0)))
                    elif phase == "HOME_FAILED":
                        why = str(det.get("reason") or "unknown")
                        if why == "stop":
                            return ("ERROR:return_home cancelled: the walk was "
                                    "stopped before the start point")
                        return "ERROR:return_home failed: %s" % why
                seen = len(events)
                el = time.time() - t_sent
                if not started and el > self.RETURN_START_S:
                    self._return_stop()
                    return ("ERROR:return_home the walk node did not start "
                            "(no RETURNING within %.0f s: is the B8l walk_node "
                            "running, OPENBOT_WALK_POSE not 0?)"
                            % self.RETURN_START_S)
                if el > self.RETURN_WAIT_S:
                    self._return_stop()
                    return "ERROR:return_home timeout after %.0f s" % el
        finally:
            with self._obs_lock:
                self._pose_active = False
            self._pose_lock.release()

    def _return_stop(self) -> None:
        try:
            self.publish_walk_pose("stop")
        except Exception:  # noqa: BLE001 -- best effort
            pass

    def _do_approach_person(self, args: List) -> str:
        """``approach_person`` verb (B8l-hf5, operator 10-17 "Centre + creep
        in"): turn in place until the person is centred (turn_around law),
        then close in with short forward pulses (the creep pulse) until the
        person is within the arm's reach (home_anchor.HANDOVER_ARRIVE_M),
        never planning a pulse that could end nearer than HANDOVER_MIN_M.

        Request: one optional JSON token {x, y} = the person row the brain
        means (body frame); the first lock is the plausible person nearest it.
        Every pulse is preceded by a cancel / budget / lost / obstacle check;
        the lane is the turn_around lane (_turn_active: _do_cancel sends
        'cmd_set off' while it is held).  Replies (ONE line):
        'DONE approach_person: arrived ...; RANGE_M=; BEARING_DEG=; X=; Y=;
        TRAVEL_M=; PULSES=; TURNS=; APPROACH_RESULT=arrived' or
        'ERROR:approach_person <why>; APPROACH_RESULT=<lost|not_seen|
        cancelled|busy|timeout|writer_refused|blocked|max_travel|too_far|
        disabled|bad_args>; RANGE_M=; TRAVEL_M=; PULSES='."""
        mod = getattr(self, "_home_m", None)
        tail0 = "RANGE_M=-1.000; TRAVEL_M=0.000; PULSES=0"
        if mod is None or not hasattr(mod, "handover_step"):
            return ("ERROR:approach_person unavailable (no home anchor "
                    "module); APPROACH_RESULT=disabled; " + tail0)
        if not mod.handover_on(os.environ):
            return ("ERROR:approach_person disabled (%s=0); "
                    "APPROACH_RESULT=disabled; %s" % (mod.HANDOVER_ENV, tail0))
        seed = None
        if args:
            try:
                req = json.loads(args[0]) if isinstance(args[0], str) else args[0]
                if len(args) != 1 or not isinstance(req, dict):
                    raise ValueError("one JSON object")
                if req.get("x") is not None or req.get("y") is not None:
                    sx, sy = float(req["x"]), float(req["y"])
                    if not (math.isfinite(sx) and math.isfinite(sy)
                            and abs(sx) <= 3.0 and 0.0 < sy <= 3.0):
                        raise ValueError("x / y")
                    seed = (sx, sy)
            except Exception:  # noqa: BLE001 -- a malformed request is a refusal
                return ("ERROR:approach_person bad_args; "
                        "APPROACH_RESULT=bad_args; " + tail0)
        if self._yolo_walk_active():
            return ("ERROR:approach_person busy (the walk FSM is moving); "
                    "APPROACH_RESULT=busy; " + tail0)
        with self._obs_lock:
            busy = bool(getattr(self, "_turn_active", False))
            if not busy:
                self._turn_active = True
                self._turn_events = []
        if busy:
            return ("ERROR:approach_person busy (a turn_around or creep is in "
                    "flight; this request did NOT move the robot); "
                    "APPROACH_RESULT=busy; " + tail0)
        clock, sleep = self._turn_clock, self._turn_sleep
        cancel0 = self._cancel_ts
        self._turn_pulsed = False
        pulses = turns = 0
        travel = 0.0
        result, detail = None, ""
        last = None                  # (x, y) of the person being tracked
        rng, err = -1.0, 0.0
        centred = False
        unseen = 0                   # forward pulses since that last row
        fwd_from = None              # range at which the last forward pulse left
        max_drop = 0.0               # biggest range drop seen per pulse
        try:
            if not self._creep_walk_settled():
                result, detail = "busy", "the walk FSM did not reach IDLE"
            t0 = clock()
            last_seen = t0
            next_read = t0               # a row has to arrive after this point
            first = True
            while result is None:
                now = clock()
                if self._cancel_ts > cancel0:
                    result = "cancelled"
                    break
                if now - t0 >= mod.HANDOVER_BUDGET_S:
                    result = "timeout"
                    break
                rows, pts = self._turn_rows()
                row = None
                if pts > next_read and (now - pts) <= TURN_ROW_MAX_AGE_S:
                    row = mod.handover_track(rows, last, seed if first else None)
                if row is None:
                    if now - last_seen >= mod.HANDOVER_LOST_S:
                        result = "not_seen" if first else "lost"
                        # get this close to a person and they drop out of the frame
                        est = (mod.handover_close_lost(rng, centred, unseen)
                               if not first else None)
                        if est is not None:
                            result = "close_lost"
                            detail = ("last seen %.2f m away, %d pulse(s) ago"
                                      % (rng, unseen))
                            last = (last[0], last[1]
                                    - mod.HANDOVER_P_NOM_M * unseen)
                            rng = est
                        break
                    sleep(TURN_TICK_S)
                    continue
                if first:
                    x0, y0 = float(row["x"]), float(row["y"])
                    if math.hypot(x0, y0) > mod.HANDOVER_START_MAX_M:
                        result = "too_far"
                        detail = ("the person is %.2f m away (> %.2f m): "
                                  "navigate first" % (math.hypot(x0, y0),
                                                      mod.HANDOVER_START_MAX_M))
                        rng = math.hypot(x0, y0)
                        break
                first = False
                last_seen = now
                unseen = 0
                last = (float(row["x"]), float(row["y"]))
                rng = math.hypot(last[0], last[1])
                err = _turn_bearing(last[0], last[1])
                if fwd_from is not None:
                    # measure what the pulse actually closed (or the person). walking too close
                    # is a safety failure.
                    max_drop = max(max_drop, fwd_from - rng)
                    fwd_from = None
                if abs(err) <= mod.HANDOVER_TOL_RAD:
                    centred = True
                step = mod.handover_step(rng, err, centred, pulses,
                                         mod.handover_p_hi(max_drop))
                if step == "arrived":
                    result = "arrived"
                    break
                if step == "max_travel":
                    result = "max_travel"
                    detail = "%d pulses sent, the person is still %.2f m away" % (
                        pulses, rng)
                    break
                if step == "turn":
                    vx, omega, pulse = 0.0, _turn_omega(err), TURN_PULSE_S
                    dur = TURN_PULSE_S
                else:
                    blocked = mod.handover_blocked(rows, row)
                    if blocked:
                        result, detail = "blocked", blocked
                        break
                    vx, omega = mod.HANDOVER_VX, mod.handover_corr_omega(err)
                    pulse = mod.HANDOVER_PULSE_S
                    dur = mod.HANDOVER_PULSE_S + mod.HANDOVER_SETTLE_S
                line = "cmd_set %.3f 0.000 %.3f %.3f" % (vx, omega, pulse)
                self._turn_pulsed = True
                try:
                    wreply = str(self.writer_verb(line))
                except Exception as exc:  # noqa: BLE001 -- report, never raise
                    wreply = "ERROR:%s" % exc
                if not wreply.startswith("DONE"):
                    result = "writer_refused"
                    detail = " ".join(_defang_error_token(wreply).split())[:60]
                    break
                if step == "turn":
                    turns += 1
                    fwd_from = None
                else:
                    pulses += 1
                    unseen += 1
                    fwd_from = rng
                    travel += mod.HANDOVER_P_NOM_M
                self._home_log("APPROACH %s %s range=%.2f bearing=%+.1f deg "
                               "omega=%+.3f pulses=%d turns=%d max_drop=%.3f"
                               % (step, "centred" if centred else "centring",
                                  rng, math.degrees(err), omega, pulses, turns,
                                  max_drop))
                t_end = clock() + dur
                next_read = t_end if step != "turn" else clock() + TURN_PULSE_S
                last_seen = max(last_seen, t_end)
                while clock() < t_end:
                    if self._cancel_ts > cancel0:
                        break
                    sleep(TURN_TICK_S)
        finally:
            if self._turn_pulsed:
                try:
                    self.writer_verb("cmd_set off")
                except Exception:  # noqa: BLE001 -- the pulse auto-expires
                    pass
                sleep(mod.HANDOVER_FINAL_SETTLE_S)
            with self._obs_lock:
                self._turn_active = False
            self._turn_pulsed = False
        if result in mod.HANDOVER_RESULTS_OK and self._cancel_ts > cancel0:
            result, detail = "cancelled", "stop/cancel during the final settle"
        self._home_log("APPROACH END result=%s range=%.2f bearing=%+.1f deg "
                       "pulses=%d turns=%d %s" % (result, rng, math.degrees(err),
                                                  pulses, turns, detail))
        if result not in mod.HANDOVER_RESULTS_OK:
            return ("ERROR:approach_person %s%s; APPROACH_RESULT=%s; "
                    "RANGE_M=%.3f; TRAVEL_M=%.3f; PULSES=%d"
                    % (result, (" (%s)" % detail) if detail else "", result,
                       rng, travel, pulses))
        how = ("arrived" if result == "arrived" else
               "lost sight of the person at close range (%s)" % detail)
        return ("DONE approach_person: %s after %d pulse(s) and %d turn "
                "pulse(s); RANGE_M=%.3f; BEARING_DEG=%+.1f; X=%.3f; Y=%.3f; "
                "TRAVEL_M=%.3f; PULSES=%d; TURNS=%d; APPROACH_RESULT=%s"
                % (how, pulses, turns, rng, math.degrees(err), last[0], last[1],
                   travel, pulses, turns, result))

    def _do_aim_turn(self, args: List) -> str:
        """``aim_turn`` verb (B8l-hf7, Stuart 10-17 23:06: a pick refused
        'self_collision' because the bottle sat a few cm left of the right
        arm's limit): turn IN PLACE, closed loop on the object's own
        detection (the turn_around law), until it sits ``shift_m`` further
        right than it was at the first lock; never a step.  Request: one
        JSON token {label, x, y, shift_m} (x/y = the row the brain means).
        Lane: the turn_around lane.  Replies (ONE line): 'DONE aim_turn: ...;
        BEARING_DEG=; TARGET_DEG=; X=; Y=; Z=; TURNS=; AIM_RESULT=aimed' or
        'ERROR:aim_turn <why>; AIM_RESULT=<not_seen|lost|no_small_turn|
        max_turns|timeout|cancelled|busy|writer_refused|disabled|bad_args>;
        TURNS='."""
        mod = getattr(self, "_home_m", None)
        if mod is None or not hasattr(mod, "aim_track"):
            return ("ERROR:aim_turn unavailable (no home anchor module); "
                    "AIM_RESULT=disabled; TURNS=0")
        if not mod.aim_on(os.environ):
            return ("ERROR:aim_turn disabled (%s=0); AIM_RESULT=disabled; "
                    "TURNS=0" % mod.AIM_ENV)
        try:
            req = json.loads(args[0]) if isinstance(args[0], str) else args[0]
            if len(args) != 1 or not isinstance(req, dict):
                raise ValueError("one JSON object")
            label = str(req.get("label") or "").strip().lower()
            sx, sy = float(req["x"]), float(req["y"])
            shift = float(req["shift_m"])
            if not (label and math.isfinite(sx) and math.isfinite(sy)
                    and abs(sx) <= 2.0 and 0.0 < sy <= 2.0
                    and math.isfinite(shift) and 0.0 < shift <= 1.0):
                raise ValueError("fields")
        except Exception:  # noqa: BLE001 -- a malformed request is a refusal
            return "ERROR:aim_turn bad_args; AIM_RESULT=bad_args; TURNS=0"
        if self._yolo_walk_active():
            return ("ERROR:aim_turn busy (the walk FSM is moving); "
                    "AIM_RESULT=busy; TURNS=0")
        with self._obs_lock:
            busy = bool(getattr(self, "_turn_active", False))
            if not busy:
                self._turn_active = True
                self._turn_events = []
        if busy:
            return ("ERROR:aim_turn busy (a turn_around or creep is in flight; "
                    "this request did NOT move the robot); AIM_RESULT=busy; "
                    "TURNS=0")
        clock, sleep = self._turn_clock, self._turn_sleep
        cancel0 = self._cancel_ts
        self._turn_pulsed = False
        turns = 0
        result, detail = None, ""
        last, row = None, None
        target, bearing = None, 0.0
        try:
            if not self._creep_walk_settled():
                result, detail = "busy", "the walk FSM did not reach IDLE"
            t0 = clock()
            last_seen = t0
            next_read = t0
            while result is None:
                now = clock()
                if self._cancel_ts > cancel0:
                    result = "cancelled"
                    break
                if now - t0 >= mod.AIM_BUDGET_S:
                    result = "timeout"
                    break
                rows, pts = self._turn_rows()
                got = None
                if pts > next_read and (now - pts) <= TURN_ROW_MAX_AGE_S:
                    got = mod.aim_track(rows, label, last,
                                        (sx, sy) if last is None else None)
                if got is None:
                    if now - last_seen >= mod.AIM_LOST_S:
                        result = "not_seen" if last is None else "lost"
                        break
                    sleep(TURN_TICK_S)
                    continue
                row = got
                last_seen = now
                last = (float(row["x"]), float(row["y"]))
                bearing = _turn_bearing(last[0], last[1])
                if target is None:
                    target = mod.aim_target_bearing(last[0], last[1], shift)
                    if target is None:
                        result = "no_small_turn"
                        detail = ("%s at x=%.3f y=%.3f cannot move %.2f m right "
                                  "by a small turn" % (label, last[0], last[1],
                                                       shift))
                        break
                err = bearing - target
                if mod.aim_done(err):
                    result = "aimed"
                    break
                if turns >= mod.AIM_MAX_PULSES:
                    result = "max_turns"
                    break
                omega = mod.aim_omega(err)
                line = "cmd_set %.3f %.3f %.3f %.3f" % (0.0, 0.0, omega, TURN_PULSE_S)
                self._turn_pulsed = True
                try:
                    wreply = str(self.writer_verb(line))
                except Exception as exc:  # noqa: BLE001 -- report, never raise
                    wreply = "ERROR:%s" % exc
                if not wreply.startswith("DONE"):
                    result = "writer_refused"
                    detail = " ".join(_defang_error_token(wreply).split())[:60]
                    break
                turns += 1
                self._home_log("AIM %s bearing=%+.1f deg target=%+.1f deg "
                               "omega=%+.3f turns=%d"
                               % (label, math.degrees(bearing),
                                  math.degrees(target), omega, turns))
                t_end = clock() + TURN_PULSE_S + 0.3
                next_read = t_end
                last_seen = max(last_seen, t_end)
                while clock() < t_end:
                    if self._cancel_ts > cancel0:
                        break
                    sleep(TURN_TICK_S)
        finally:
            if self._turn_pulsed:
                try:
                    self.writer_verb("cmd_set off")
                except Exception:  # noqa: BLE001 -- the pulse auto-expires
                    pass
                sleep(mod.AIM_FINAL_SETTLE_S)
            with self._obs_lock:
                self._turn_active = False
            self._turn_pulsed = False
        if result == "aimed" and self._cancel_ts > cancel0:
            result, detail = "cancelled", "stop/cancel during the final settle"
        self._home_log("AIM END result=%s bearing=%+.1f deg target=%s turns=%d %s"
                       % (result, math.degrees(bearing),
                          "%+.1f deg" % math.degrees(target)
                          if target is not None else "-", turns, detail))
        if result != "aimed":
            return ("ERROR:aim_turn %s%s; AIM_RESULT=%s; TURNS=%d"
                    % (result, (" (%s)" % detail) if detail else "", result,
                       turns))
        z = row.get("z") if isinstance(row, dict) else None
        try:
            z = float(z)
        except (TypeError, ValueError):
            z = float("nan")
        return ("DONE aim_turn: %s now at %+.1f deg (+ = right) after %d turn "
                "pulse(s); BEARING_DEG=%+.1f; TARGET_DEG=%+.1f; X=%.3f; Y=%.3f; "
                "Z=%.3f; TURNS=%d; AIM_RESULT=aimed"
                % (label, math.degrees(bearing), turns, math.degrees(bearing),
                   math.degrees(target), last[0], last[1], z, turns))

    def _do_step_back(self, args: List) -> str:
        """``step_back`` verb (B8l-hf6, operator 10-17: "step back 5 steps
        slowly before actually started sweeping"): walk BACKWARD at the gait
        floor.  The brain calls it before a navigate whose target is not in
        its current view.  Request: one optional JSON token {m} (metres,
        0 < m <= 1; the gateway switch home_anchor.STEPBACK_ENV wins and the
        distance is capped at STEPBACK_MAX_M).  Timed (the gateway runs no
        odometry loop): one auto-expiring writer command re-armed every
        STEPBACK_REARM_S for distance / STEPBACK_VX seconds, a cancel check
        every tick, then 'cmd_set off' and a settle.  The lane is the
        turn_around lane (_turn_active: _do_cancel sends 'cmd_set off' while
        it is held).  Replies (ONE line): 'DONE step_back: ...; STEPBACK_M=;
        STEPBACK_RESULT=done|off' or 'ERROR:step_back <why>;
        STEPBACK_RESULT=<cancelled|busy|writer_refused|bad_args|disabled>;
        STEPBACK_M=<nominal metres moved>'."""
        mod = getattr(self, "_home_m", None)
        tail0 = "STEPBACK_M=0.000"
        if mod is None or not hasattr(mod, "stepback_distance"):
            return ("ERROR:step_back unavailable (no home anchor module); "
                    "STEPBACK_RESULT=disabled; " + tail0)
        req_m = None
        if args:
            try:
                req = json.loads(args[0]) if isinstance(args[0], str) else args[0]
                if len(args) != 1 or not isinstance(req, dict):
                    raise ValueError("one JSON object")
                if req.get("m") is not None:
                    req_m = float(req["m"])
                    if not (math.isfinite(req_m) and 0.0 < req_m <= 1.0):
                        raise ValueError("m")
            except Exception:  # noqa: BLE001 -- a malformed request is a refusal
                return ("ERROR:step_back bad_args; STEPBACK_RESULT=bad_args; "
                        + tail0)
        want = mod.stepback_distance(os.environ.get(mod.STEPBACK_ENV), req_m)
        if want <= 0.0:
            return ("DONE step_back: off (%s=0), nothing moved; %s; "
                    "STEPBACK_RESULT=off" % (mod.STEPBACK_ENV, tail0))
        if self._yolo_walk_active():
            return ("ERROR:step_back busy (the walk FSM is moving); "
                    "STEPBACK_RESULT=busy; " + tail0)
        with self._obs_lock:
            busy = bool(getattr(self, "_turn_active", False))
            if not busy:
                self._turn_active = True
                self._turn_events = []
        if busy:
            return ("ERROR:step_back busy (a turn_around or creep is in "
                    "flight; this request did NOT move the robot); "
                    "STEPBACK_RESULT=busy; " + tail0)
        clock, sleep = self._turn_clock, self._turn_sleep
        cancel0 = self._cancel_ts
        self._turn_pulsed = False
        t_run = want / mod.STEPBACK_VX
        result, detail = None, ""
        elapsed = 0.0
        try:
            if not self._creep_walk_settled():
                result, detail = "busy", "the walk FSM did not reach IDLE"
            t0 = clock()
            next_arm = t0
            while result is None:
                now = clock()
                elapsed = min(now - t0, t_run) if self._turn_pulsed else 0.0
                if self._cancel_ts > cancel0:
                    result = "cancelled"
                    break
                if self._turn_pulsed and now - t0 >= t_run:
                    result = "done"
                    break
                if now >= next_arm:
                    left = t_run - (now - t0) if self._turn_pulsed else t_run
                    if not self._turn_pulsed:
                        t0 = now           # the clock starts when the motion does
                    dur = max(0.05, min(mod.STEPBACK_CMD_S, left))
                    line = "cmd_set %.3f 0.000 0.000 %.3f" % (-mod.STEPBACK_VX,
                                                              dur)
                    self._turn_pulsed = True
                    try:
                        wreply = str(self.writer_verb(line))
                    except Exception as exc:  # noqa: BLE001 -- report, never raise
                        wreply = "ERROR:%s" % exc
                    if not wreply.startswith("DONE"):
                        result = "writer_refused"
                        detail = " ".join(_defang_error_token(wreply).split())[:60]
                        break
                    next_arm = now + mod.STEPBACK_REARM_S
                sleep(TURN_TICK_S)
        finally:
            if self._turn_pulsed:
                try:
                    self.writer_verb("cmd_set off")
                except Exception:  # noqa: BLE001 -- the command auto-expires
                    pass
                sleep(mod.STEPBACK_SETTLE_S)
            with self._obs_lock:
                self._turn_active = False
            self._turn_pulsed = False
        if result == "done" and self._cancel_ts > cancel0:
            result, detail = "cancelled", "stop/cancel during the settle"
        moved = mod.STEPBACK_VX * elapsed
        self._home_log("STEPBACK END result=%s want=%.2f m moved~%.2f m %s"
                       % (result, want, moved, detail))
        if result != "done":
            return ("ERROR:step_back %s%s; STEPBACK_RESULT=%s; STEPBACK_M=%.3f"
                    % (result, (" (%s)" % detail) if detail else "", result,
                       moved))
        return ("DONE step_back: stepped back about %.2f m at %.2f m/s; "
                "STEPBACK_M=%.3f; STEPBACK_RESULT=done"
                % (moved, mod.STEPBACK_VX, moved))

    def _do_container(self, args: List) -> str:
        """``container`` verb (B8j, query, no motion): the locked container
        as one JSON line, 'DONE:container {...}', or 'DONE:container none
        (<why>)'."""
        lock = getattr(self, "_container_lock", None)
        if lock is None:
            return "DONE:container none (unavailable)"
        if not self._lock_on():
            return "DONE:container none (disabled: OPENBOT_CONTAINER_LOCK=0)"
        snap = lock.snapshot(self._lock_now())
        if snap is None:
            return ("DONE:container none (no lock: no container REACHED since "
                    "the last motion)")
        return "DONE:container " + json.dumps(snap, sort_keys=True)

    _CREEP_LABEL_RE = re.compile(r"^[a-z][a-z0-9_ ]{0,31}$")

    @classmethod
    def _creep_parse(cls, args: List) -> Tuple[Optional[Dict[str, Any]], str]:
        """One JSON object -> (spec, '') or (None, why)."""
        if len(args) != 1:
            return None, "expected one JSON object"
        raw = args[0]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, "not JSON"
        if not isinstance(raw, dict):
            return None, "not a JSON object"
        label = raw.get("label")
        if not isinstance(label, str) or not cls._CREEP_LABEL_RE.match(
                label.strip().lower()):
            return None, "label"
        iid = raw.get("iid")
        if iid is not None and (isinstance(iid, bool) or not isinstance(iid, int)
                                or iid < 1):
            return None, "iid"
        spec: Dict[str, Any] = {"label": label.strip().lower(), "iid": iid,
                                "dry": raw.get("dry") is True}
        bounds = {"x": (-2.0, 2.0), "y": (0.05, 2.0), "z": (-1.5, 1.5),
                  "w": (0.0, 1.5), "dd": (0.0, 1.5), "h": (0.0, 1.5)}
        for key, (lo, hi) in bounds.items():
            v = raw.get(key)
            if v is None and key in ("w", "dd", "h"):
                spec[key] = None
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None, key
            v = float(v)
            if not math.isfinite(v) or not lo <= v <= hi:
                return None, key
            spec[key] = v
        return spec, ""

    def _creep_reference(self, spec: Dict[str, Any], arm: bool
                         ) -> Dict[str, Any]:
        """The container the creep closes on: the LOCK when it is the same
        container (same id, or within 0.12 m), else the caller's record (a
        real creep arms the lock on it so the pulses follow it by identity)."""
        mod = self._lock_m
        lock = self._container_lock
        now = self._lock_now()
        snap = lock.snapshot(now) if self._lock_on() else None
        same = snap is not None and (
            (spec["iid"] is not None and snap["iid"] == spec["iid"])
            or math.hypot(snap["x"] - spec["x"], snap["y"] - spec["y"]) <= 0.12)
        if not same and arm and self._lock_on():
            msg = lock.arm_pose(spec["label"], spec["iid"], spec["x"],
                                spec["y"], spec["z"], spec["w"], spec["dd"],
                                spec["h"], now, src="creep_spec")
            if msg:
                self._lock_log(msg)
            snap = lock.snapshot(now)
            same = snap is not None
        if same:
            return {"label": snap["label"], "iid": snap["iid"],
                    "x": snap["x"], "y": snap["y"], "z": snap["z"],
                    "half": snap["half"], "rim": snap["rim_z"],
                    "bot": snap["bot_z"], "near": snap["near_m"], "src": "lock"}
        ext = [v for v in (spec["w"], spec["dd"]) if v is not None and v > 0.0]
        half = (max(ext) / 2.0) if ext else mod.LOCK_HALF_DEFAULT_M
        half = max(mod.LOCK_HALF_MIN_M, min(mod.LOCK_HALF_MAX_M, half))
        h = spec["h"] or 0.0
        return {"label": spec["label"], "iid": spec["iid"], "x": spec["x"],
                "y": spec["y"], "z": spec["z"], "half": half,
                "rim": spec["z"] + h / 2.0, "bot": spec["z"] - h / 2.0,
                "near": math.hypot(spec["x"], spec["y"]) - half, "src": "spec"}

    def _creep_rows(self) -> Tuple[List[dict], float]:
        with self._obs_lock:
            return list(self._lock_rows), float(self._lock_rows_ts)

    def _creep_blocked(self, rows: List[dict], iid: Optional[int],
                       c_near: float) -> Optional[str]:
        """A reason to stop: a plausible person within CREEP_PERSON_STOP_M, or
        another object low in the corridor nearer than the container."""
        mod = self._lock_m
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
                conf = float(r.get("conf") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            if iid is not None and r.get("instance_id") == iid:
                continue
            d = math.hypot(x, y)
            if d < mod.CREEP_SELF_ZONE_M:
                continue                     # the robot's own arm and hand rows
            lab = str(r.get("label") or "").strip().lower()
            if lab == "person":
                try:
                    h = float(r.get("height_m") or 0.0)
                except (TypeError, ValueError):
                    h = 0.0
                if (conf >= mod.CREEP_PERSON_MIN_CONF
                        and h >= mod.CREEP_PERSON_MIN_H_M
                        and d <= mod.CREEP_PERSON_STOP_M):
                    return "person at %.2f m" % d
                continue
            if (conf >= mod.CREEP_OBST_MIN_CONF and z <= mod.CREEP_OBST_Z_MAX
                    and abs(x) <= mod.CREEP_OBST_CORRIDOR_M
                    and d < c_near - 0.05):
                return "%s at %.2f m, nearer than the container" % (lab, d)
        return None

    def _creep_omega(self, x: float, y: float) -> float:
        mod = self._lock_m
        b = math.atan2(x, y)
        if abs(b) <= mod.CREEP_TOL_RAD:
            return 0.0
        w = mod.CREEP_OMEGA_SIGN * mod.CREEP_K_THETA * b
        return max(-mod.CREEP_OMEGA_MAX, min(mod.CREEP_OMEGA_MAX, w))

    def _creep_walk_settled(self) -> bool:
        """The walk FSM must not command the legs during the creep: a REACHED
        hold (<= 5 s after REACHED) is ended with stop_walk() and IDLE is
        awaited (turn_around's settle).  Lane already claimed."""
        with self._obs_lock:
            walk = getattr(self._assembler, "_walk", None)
            ts = getattr(self._assembler, "_last_walk_ts", 0.0)
            marker = len(self._turn_events)
        phase = str(getattr(walk, "phase", "") or "").upper()
        if phase != "REACHED" or time.time() - ts > 6.5:
            return True
        self.stop_walk()
        return self._turn_wait_idle(marker)

