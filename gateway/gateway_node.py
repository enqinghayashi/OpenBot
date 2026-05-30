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
