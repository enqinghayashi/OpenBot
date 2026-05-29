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


