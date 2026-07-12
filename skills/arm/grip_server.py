#!/usr/bin/env python3
"""
grip_server.py — persistent Dex3 grip server (Task 8, 2026-08-17).

Phase-split handoff fix: the pick daemon EXITS between the pick phase and
the walk phase (PHASE-SPLIT), but the dex3 hand service reacts to its
rt/dex3/*/cmd writer vanishing — hardware-verified: the hand OPENED and
dropped the object when the daemon quit.  A single HandCmd_ write does NOT
persist; the MCU needs a continuous writer.  THIS SERVER owns the
rt/dex3/{side}/cmd writers 100%: launcher-owned, survives the daemon quit,
re-asserts the last commanded q at 20 Hz per side.

Transport: UNIX socket /tmp/grip_server.sock, one line in -> one line out
(DONE... / ERROR:...), walk_loco_daemon.py:105-126 grammar.  Multi-client
by sequential accept (daemon pre-quit + operator CLI post-quit never
overlap).

NEVER opens the hands on boot: each side is seeded from the CURRENT motor
state (rt/dex3/{side}/state motor_state.q), falling back to
/tmp/grip_last_cmd.json, then zeros.  Boot is a hold of current q, never a
preset.

Constants (DEX3_JOINTS / KP / KD + presets) live HERE as the single source
of truth — the daemon and grip_client import them.

NO_DDS mode (SDK import failure, e.g. laptop): driving commands still
update state and return DONE (degraded but functional protocol); pressure
feedback is impossible, so close_adaptive returns ERROR:no_dds
(sensing_daemon.py NO_DDS precedent).

Protocol:
    open | close                     (right hand preset ramp)
    open_left | close_left
    open_both | close_both
    set <side> <7f> [steps] [dt]
    close_adaptive <side> [max_fraction steps dt stop_threshold min_steps]
                          [fingers_neutral]
                                   -> DONE:contact | DONE:full |
                                      DONE:no_data | ERROR:no_dds
    state [side]                   -> DONE:<json> (last commanded q, plus
                                      "available": bool per side — true
                                      when the DDS writer is live, false
                                      under NO_DDS, so degradation shows at
                                      a glance — observer DEV1; plus
                                      "actual", "mode" and "motorstate" per
                                      joint — observer B0.2, so a consumer
                                      can always tell a LIMP joint from a
                                      stalled one.  The three are OMITTED
                                      (never faked) whenever their source is
                                      unusable, and INDEPENDENTLY of each
                                      other: "mode"/"motorstate" need
                                      motor_limits, "actual" needs motor_q.
                                      A missing key means "no usable state" —
                                      a consumer must REFUSE, never assume
                                      values.  Each key is also withheld if
                                      its sample is non-finite: a NaN would
                                      otherwise read as a payload the engine
                                      can act on)
    hold [side]                    (no-op — 20 Hz re-assert always active)
    quit
"""

import json
import math
import os
import socket
import sys
import threading
import time

import numpy as np

# ── one source of truth: imported by the daemon and the client ───────
SOCK_PATH     = '/tmp/grip_server.sock'
LAST_CMD_FILE = '/tmp/grip_last_cmd.json'

DEX3_JOINTS = [0, 1, 2, 3, 4, 5, 6]
DEX3_KP = 2.0
DEX3_KD = 0.3
# kp/kd: one pair for all 7 motors, no per-joint override.
# At the joint limits 6 of 7 joints saturate the 1.4 N.m cap
# (2.0 * 1.5708 = 3.14 N.m), so raising kp adds no grip force.
# When the lead is capped at RISER_MAX_RAD the torque is kp * lead with no
# endpoint term: 0.10 N.m at 0.05 rad, so the kp value is a hardware choice.

# DDS motor order, not URDF order. Thumb is 0/1/2 on both hands; the
# hands differ only in index/middle order. Commands use sums and set only,
# so these names are labels, and tests pin them.
DEX3_JOINT_NAMES = {
    'right': {0: 'thumb_0',  1: 'thumb_1',  2: 'thumb_2',
              3: 'index_0',  4: 'index_1',  5: 'middle_0', 6: 'middle_1'},
    'left':  {0: 'thumb_0',  1: 'thumb_1',  2: 'thumb_2',
              3: 'middle_0', 4: 'middle_1', 5: 'index_0',  6: 'index_1'},
}

# Presets for both hands, in DDS motor order. OPEN is the wide-open pose and
# sets thumb travel: a joint equal in OPEN and CLOSE cannot move, so fix OPEN
# when the thumb will not close.
# CLOSE commands the URDF finger limit; the servo settles below it under load,
# while commanding the settled value would leave the hand short. The right
# thumb closes to the contact rest pose, and a shallower close reads
# as half-closed.
# OPEN is an exact sign mirror (L == -R per joint); the left URDF flips y
# origins and negates limits, and tests pin it. The left CLOSE thumb is not
# mirrored because the left hand has no full close.
# A close inside the URDF limits can still jam the thumb: a full close left
# 0.039-0.098 mm clear against a 1.0 mm floor. Full close ends in contact at
# 0.13-0.22 mm; point-to-point sweeps cannot tell touching from crossing, and
# every sweep only describes the base pose it started from.
# Re-measure after any change here.
R_HAND_OPEN  = {0: 0.48, 1: 0.30,  2: 0.0,         3: 0.0, 4: 0.0,
                5: 0.0,  6: 0.0}
R_HAND_CLOSE = {0:-0.02676787, 1:-1.01299822, 2:-1.68802154, 3: 1.57079632,
                4: 1.74532925, 5: 1.57079632, 6: 1.74532925}

# Right-hand grasps tuned per object geometry, between OPEN and CLOSE.
# thumb_0 0.0 matches CLOSE while the other joints stay partly open, so
# these stay inside the OPEN-to-CLOSE envelope. Retune only on hardware.
R_HAND_CYLINDRICAL = {0: 0.0, 1:-0.20, 2:-0.90, 3: 0.85, 4: 0.85, 5: 0.85, 6: 0.85}
R_HAND_PINCH = {0: 0.0, 1:-0.25, 2:-0.95, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.25}
R_HAND_WIDE = {0: 0.0, 1:-0.15, 2:-0.80, 3: 1.10, 4: 1.10, 5: 1.10, 6: 1.10}

# Left presets, in left motor order: 3/5 are middle_0/index_0 and 4/6 are
# middle_1/index_1. Finger values are the URDF lower limits, and OPEN is
# the exact sign mirror of the right hand.
L_HAND_OPEN  = {0:-0.48, 1:-0.30, 2: 0.0,         3: 0.0, 4: 0.0,
                5: 0.0,  6: 0.0}
L_HAND_CLOSE = {0: 0.00, 1: 0.45,  2: 1.00,       3:-1.57079632,
                4:-1.74532925, 5:-1.57079632, 6:-1.74532925}

OPEN_PRESETS  = {'right': R_HAND_OPEN,  'left': L_HAND_OPEN}
CLOSE_PRESETS = {'right': R_HAND_CLOSE, 'left': L_HAND_CLOSE}

# Ramp pacing: 40 steps at 0.02 s = 0.8 s for open/close and 50 steps at
# 0.02 s = 1.0 s for the adaptive close. Writes use the ramp dt, with the
# 20 Hz re-assert running between ramps.
STEP_DEFAULT = 40
DT_DEFAULT   = 0.02

# close_adaptive defaults: full travel, 50 steps at 0.02 s, stop at
# pressure 200000, and at least 10 steps before a contact stop is allowed.
ADAPTIVE_MAX_FRACTION = 1.0
ADAPTIVE_STEPS        = 50
ADAPTIVE_DT           = 0.02
ADAPTIVE_STOP         = 200000
ADAPTIVE_MIN_STEPS    = 10

# Thumb first: the thumb is settled before the fingers close onto it.
# A single cosine over every joint can sweep the thumb diagonally through
# the fingers' volume, which reads as a finger collision.
# THUMB_BASE_JID 0 and THUMB_BASE_CENTRE 0.0 are the same on both hands.
THUMB_BASE_JID = 0
THUMB_BASE_CENTRE = 0.0
STAGED_CLOSE_SKIP_TOL = 0.10   # rad; within this of centre the stage is skipped

#: THUMB_JIDS 0/1/2 and FINGER_JIDS 3-6 are motor ids, the same on both
#: hands. Only the index/middle labels differ by side (middle_0/1 are 5/6
#: on the right, 3/4 on the left). Code keyed on these ids is side-safe;
#: a literal "middle" id is not.
THUMB_JIDS  = (0, 1, 2)
FINGER_JIDS = (3, 4, 5, 6)

# Cap on the close fraction. 1.0 means uncapped: the ramp is a straight
# line in joint space and passes through the contact band, so a lower cap
# made every close a half-close.
# clamped_close_target stays the single source for the commanded and the
# scored pose. If they diverged, a blocked close would score against the
# uncapped preset and report a false capture.
# Clearance sweeps are point-to-point, always >= 0, and cannot tell
# touching from crossing; real mesh-local checks show 0.13-0.22 mm contact
# at full close, not penetration.
F_MAX_SAFE = 1.0


def clamped_close_target(side, max_fraction):
    """The {jid: q} the closure will ACTUALLY ramp to, after the F_MAX_SAFE cap.

    Single source of truth for the ramp (`close_adaptive`), the client's
    `last_q` bookkeeping, and the engine's post-close hold + closure-block
    score — so the pose commanded and the pose scored can never drift apart.
    (If they drift, a capped close in free air scores a large margin against the
    UNCAPPED preset and reports a false capture.)

    Returns (targets, effective_fraction).
    """
    eff = min(max(float(max_fraction), 0.0), F_MAX_SAFE)
    return ({jid: OPEN_PRESETS[side][jid]
             + eff * (CLOSE_PRESETS[side][jid] - OPEN_PRESETS[side][jid])
             for jid in DEX3_JOINTS}, eff)


WRITE_HZ = 20.0

# ── residual bound: placeholders, not values ───────────────────
# RISER_MAX_RAD and SETTLE_S must be measured on hardware. Do not guess
# them or halve a measured value: RISER_MAX_RAD is the only holding force
# for a blocked close, so a wrong value drops the object.
#
# SETTLE_S is the time after the last ramp write before the bound arms.
# An unloaded close settles within 0.06 rad on all 7 joints in roughly
# 0.2-1.0 s, depending on the ramp; start at 1.0 s and time a free-air
# close. Longer is safe, because last_q keeps driving the servo.
#
# RISER_MAX_RAD is how far a command may lead measured q, closing
# direction only. Measure with a 30 s press on a rigid object, one value
# per trial over {0.005, 0.01, 0.02, 0.03, 0.05, 0.08} rad, recording
# motorstate, temperature and tau_est. The pass value is the smallest
# lead that holds for 30 s with motorstate 0 throughout; ship that value.
# If none holds, redesign the press rather than widen the sweep, and
# power-cycle first: a latched hand reads motorstate != 0 from t=0.
#
# The placeholders keep the bound inert: SETTLE_S 1.0e9 never arms it,
# and RISER_MAX_RAD inf makes abs(lead) > inf false, so last_q is written
# unchanged. The bound never rewrites last_q; it recomputes per write, so
# a short clamped value cannot freeze the hand.
RISER_MAX_RAD = float('inf')   # placeholder: see above, do not tune.
SETTLE_S = 1.0e9               # placeholder: see above, do not tune.

# ── SDK import: any failure falls back to NO_DDS ────────────────
NO_DDS = False
try:
    from unitree_sdk2py.core.channel import (  # noqa: E402
        ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: E402
        HandCmd_, HandState_, MotorState_, PressSensorState_)
    from unitree_sdk2py.idl.default import (  # noqa: E402
        unitree_hg_msg_dds__HandCmd_)
except Exception:  # noqa: BLE001 - any SDK failure means we go NO_DDS
    NO_DDS = True
    ChannelPublisher = ChannelSubscriber = None
    HandCmd_ = HandState_ = MotorState_ = PressSensorState_ = None


def _log(msg):
    sys.stderr.write(f'[grip_server] {msg}\n')
    sys.stderr.flush()


def _build_cmd(q_dict):
    """Build a HandCmd_ for one hand from a {joint_id: q} dict."""
    msg = unitree_hg_msg_dds__HandCmd_()
    for jid in DEX3_JOINTS:
        msg.motor_cmd[jid].mode = (jid & 0x0F) | (0x01 << 4)
        msg.motor_cmd[jid].kp   = DEX3_KP
        msg.motor_cmd[jid].kd   = DEX3_KD
        msg.motor_cmd[jid].q    = float(q_dict.get(jid, 0.0))
    return msg


class PressureReader:
    """Per-side rt/dex3/{side}/state subscriber (non-blocking).

    finger_pressure() replicates real_grasp_detector.py:377-413 (packages
    3-8 = finger pads + fingertips; 0-2 = palm).  Returns 0 if no data.
    """

