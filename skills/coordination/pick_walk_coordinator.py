#!/usr/bin/env python3
"""
pick_walk_coordinator.py — T3 coordinator between the pick pipeline and
walk_node (Task 3).

Two-step navigation handshake on /navigate_target (std_msgs/String):
    1. a LABEL      -> PENDING (replaces any prior pending label)
    2. 'confirm'    -> forward the label to /walk_target (only if the pick
                       gate holds); 'cancel' -> drop the pending label
    'stop' / 'abort' -> forwarded to /walk_target immediately (any state)

Task 10 (post-confirm): the confirm enters the 3-consecutive-confirmation
flow instead of a bare REQUESTED.  If the label is NOT in vision at
confirm -> status 1 and the walk side sweeps.  If it IS freshly detected
-> the first beat counts immediately (statuses 3+4).  The gate completes
on the 3rd CONSECUTIVE fresh /object_detections_body beat (<= 1 s gaps
at the projector cadence; a missed beat — no label beat within 1 s, or a
detection message without the label — resets the streak to 0).  The 3rd
beat publishes the target-check table and then AUTOMATICALLY sends the
approach trigger on /walk_go (String label) — no operator confirm step,
fully autonomous per the user's directive.  Reviewer F1 (2026-08-18): NO
/walk_target replay as the trigger — the walknode session's walk_node
treats ANY /walk_target replay as a RETARGET (its check precedes /walk_go
consumption in the same hold tick), which would force a perpetual
re-sweep and the paired /walk_go would never be consumed.  /walk_target
remains only for: the confirm-time sweep start and the DETECTION LOST
re-sweep re-send.  If detections stop after the trigger (> 2 s), the
coordinator publishes DETECTION LOST and re-sends the label on
/walk_target so the walk side re-sweeps and the gate re-runs.  stop/abort
reset flow + streak + epoch (everything).

PICK GATE (checked at confirm time): walk_node receives the label ONLY
when the latest /pick_state says pick state in {'idle', 'holding'} AND
the message is fresh (<= 2 s).  daemon_alive is NOT checked — since the
T8 phase split, the pick daemon MUST be dead while walking (any alive
arm_sdk writer trips the FSM into 801), so a dead daemon is the normal,
REQUIRED walk-time condition, not a hazard.  The gate's only job is to
refuse a walk while a pick is in flight ('busy') or while pick state is
stale/unknown.  'idle' + dead daemon is the DEFAULT state (--no-daemon
launches with no pick at all) and passes; 'holding' + dead daemon (pick
+ `quit` before the walk) passes too.  If /pick_state has never
arrived, the coordinator falls back to parsing /pick_status text
("picked <label> OK" / "<cmd> OK" / "ERROR: ...") with the same
staleness bound, and publishes a WARNING that the pick state is UNKNOWN.
If neither topic has ever produced a message, the gate fails with ERROR.

/navigate_status (std_msgs/String) events (each at most once per
navigate unless noted):
    "PENDING <label>"                        (pre-confirm ack — label
                                             absent/stale)
    "PENDING <label> at x=+.. y=+.. z=.. (d=.. m)"
                                             (pre-confirm ack — fresh
                                             /object_detections_body match;
                                             d = planar hypot(x, y))
    "<label> requested not in vision, Proceed to Sweep Search"
                                             (confirm with no fresh
                                             detection — walk side sweeps)
    "SEARCHING <label>"                      (walk side sweeping;
                                             transition-deduped, re-fires on
                                             each re-entry)
    "DETECTED <label> ONCE, EXIT Sweep Motion Search"
                                             (first fresh beat of a run)
    "WAIT FOR THREE CONSECUTIVE CONFIRMATION"
    "Three Confirmation Received, outputting the target check table"
                                             (3rd consecutive beat)
    <target-check table, multi-line>         (label, torso x/y/z, planar
                                             d = hypot(x, y), conf, trigger
                                             note) — follows the status
                                             above on /navigate_status
    "APPROACHING d=.. m"                     (walk_node distance heartbeat,
                                             deduped to >= 0.02 m changes
                                             or state transitions, only
                                             while navigating)
    "REACHED <label>"                        (once — walk_node published
                                             REACHED)
    "DETECTION LOST — re-entering sweep"     (post-trigger: no fresh beat
                                             for > DETECTION_LOST_GAP_S —
                                             label re-sent on /walk_target)
    "NOT_FOUND <label>"                      (once — walk_node exhausted its
                                             sweep and went IDLE; the watch
                                             ENDS here, with no stop and no
                                             `ERROR: walk timeout`)
    "CANCELLED <label>"
    "STOPPED"                  (stop/abort forwarded)
    "WARNING: <reason>"        (e.g. pick state unknown — /pick_status
                                fallback used)
    "ERROR: <reason>"          (gate failure, walk timeout, etc.)

Topics:
    sub /navigate_target, /pick_state, /pick_status, /walk_status,
        /object_detections_body
    pub /walk_target, /walk_go, /navigate_status

Run (Stuart, tv env — same as pick_node):
    /home/unitree/miniconda3/envs/tv/bin/python3 pick_walk_coordinator.py
"""

import importlib
import json
import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── ref-shaped targets are resolved to plain labels ──────────────────────────
# brain may send 'bottle_3'
# lookups are VERBATIM keys
# here and in walk_node
# -> no match
# 77 s to NOT_FOUND
# 374/714 had it
# window 1790595203-1790595280
# 119 were plain 'bottle'
# gateway/place_target.py: same pair
# missing -> raise
_REF_TO_LABEL_MODULES = ("skills.perception.instance_labels",   # repo: skills/perception/instance_labels.py
                         "instance_labels")                    # robot: openbot/instance_labels.py


def _load_ref_to_label():
    """`ref_to_label` from whichever layout is present; raises if neither is."""
    last = None
    for name in _REF_TO_LABEL_MODULES:
        try:
            return importlib.import_module(name).ref_to_label
        except ImportError as exc:
            last = exc
    raise ImportError("instance_labels not importable (tried %s): %s"
                      % (", ".join(_REF_TO_LABEL_MODULES), last))


ref_to_label = _load_ref_to_label()

TOPIC_NAV_TARGET = '/navigate_target'
TOPIC_NAV_STATUS = '/navigate_status'
TOPIC_PICK_STATE = '/pick_state'
TOPIC_PICK_STATUS = '/pick_status'
TOPIC_WALK_TARGET = '/walk_target'
TOPIC_WALK_STATUS = '/walk_status'
TOPIC_OBJECT_DETECTIONS = '/object_detections_body'
WALK_GO_TOPIC = '/walk_go'   # approach trigger (String label)

PICK_STATE_STALE_S = 2.0
WALK_REACHED_TIMEOUT_S = 120.0
GATE_STATES = ('idle', 'holding')

# PENDING / echo knobs
# projector is EVENT-DRIVEN
# only publishes with detections
# stale entry = last ghost
# never a made-up guess
# 1.0 s guard
DETECTION_FRESH_S = 1.0    # fresh this long
DET_MIN_CONF = 0.20        # pick_node MIN_CONF
APPROACH_DEDUPE_M = 0.02   # min delta

# 3-beat gate
# after each sweep
# projector ~30 fps in view
# beats land inside 1 s
# gap > CONFIRM_BEAT_GAP_S
# msg without label
CONFIRM_BEAT_GAP_S = 1.0     # -> streak reset
CONFIRM_REQUIRED = 3         # max beat gap
# beats to finish
# post-trigger quiet
# LOST + resend
DETECTION_LOST_GAP_S = 2.0   # 2 missed beats at 1 s
# re-arm after a rollback
# /walk_go is ONE-SHOT
# counting stops
# FSM bails -> SEARCHING
# -> DETECTED hold
# only exit: fresh /walk_go
# nobody will send
# needs SILENCE: live object
# keeps beats coming
# froze twice, 'person' held 36 s
# until the 120 s timeout
# STALL_BUDGET = 2
# grace > GO_STALE_S 5.0 s
# + time to collect CONFIRM_REQUIRED beats
REARM_GRACE_S = 8.0


class PickWalkCoordinator(Node):
