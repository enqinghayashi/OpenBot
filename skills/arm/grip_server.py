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

    def __init__(self, side):
        self._side = side
        self._state = None
        self._lock = threading.Lock()
        if NO_DDS:
            return
        self._sub = ChannelSubscriber(f'rt/dex3/{side}/state', HandState_)
        self._sub.Init(self._on_state)

    def _on_state(self, msg):
        with self._lock:
            self._state = msg

    def read_state(self):
        with self._lock:
            return self._state

    def has_data(self):
        return self.read_state() is not None

    def finger_pressure(self):
        state = self.read_state()
        if state is None:
            return 0.0
        max_p = 0.0
        pkgs = state.press_sensor_state
        n_pkgs = len(pkgs)
        for pkg_idx in range(3, min(n_pkgs, 9)):
            for p in pkgs[pkg_idx].pressure:
                if p > max_p:
                    max_p = float(p)
        return max_p

    def motor_q(self):
        """Current motor q per joint (seed source), or None if no data."""
        state = self.read_state()
        if state is None or state.motor_state is None:
            return None
        try:
            return {jid: float(state.motor_state[jid].q) for jid in DEX3_JOINTS}
        except Exception:
            return None

    def motor_limits(self):
        """Per-joint (q, dq, mode, motorstate), or None if no DDS state.

        B0.1 — the observability half of the latched-motor defect.  `motor_q`
        alone cannot distinguish a STALLED joint (mode=1, servo on, ~kp*err of
        holding torque) from a LIMP one (mode=0 = DISABLED: ~0 N.m AND
        mechanically free), because both report the same q.  The DDS IDL
        carries the discriminator (`_MotorState_.py:25-33`) and this stack read
        only `.q`; this is that read.

        Same None semantics as motor_q(), and for the same reason: None (never
        a zero-filled dict) when there is no state or the field is missing, so
        every caller can honestly say "no DDS state" instead of acting on a
        faked zero.  dq and mode/motorstate are returned as the IDL types them
        (float32 / uint8 / uint32).
        """
        state = self.read_state()
        if state is None or state.motor_state is None:
            return None
        try:
            return {jid: (float(state.motor_state[jid].q),
                          float(state.motor_state[jid].dq),
                          int(state.motor_state[jid].mode),
                          int(state.motor_state[jid].motorstate))
                    for jid in DEX3_JOINTS}
        except Exception:
            return None


class HandController:
    """One hand: last commanded q + single 20 Hz writer (ramps + re-assert).

    The writer thread is THE only DDS writer for its side.  Ramps run under
    the controller lock (synchronous with the command handler, preserving
    the old daemon's exact step/dt pacing); between ramps the thread
    re-asserts last_q at 20 Hz so the dex3 MCU never sees the writer go
    quiet.
    """

    def __init__(self, side, seed_q=None, reader=None):
        self._side = side
        self._lock = threading.Lock()
        self._pub = None
        self._running = True
        self._seed_logged = False
        # Reader: the controller's own view of motor state (mode/dq/motorstate).
        # Optional; None or a reader without motor_limits means no state, and
        # the bound and fault checks are skipped.
        self._reader = reader
        # +inf leaves the bound unreachable until set() or close_adaptive
        # arms a deadline, so boot writes last_q unchanged. Boot holds current
        # q, never a preset; the seed may come from the persisted last command.
        self._bound_deadline = float('inf')
        # fault annunciator edge state
        self._fault_active = False
        self._last_fault_check = 0.0
        if NO_DDS:
            _log(f'NO_DDS: {side} controller has no DDS writer')
            self._thread = None  # stop() would AttributeError otherwise
            self.last_q = {jid: float(seed_q.get(jid, 0.0)) if seed_q else 0.0
                           for jid in DEX3_JOINTS}
            return
        self._pub = ChannelPublisher(f'rt/dex3/{side}/cmd', HandCmd_)
        self._pub.Init()
        if seed_q is not None:
            self.last_q = {jid: float(seed_q.get(jid, 0.0)) for jid in DEX3_JOINTS}
        else:
            self.last_q = {jid: 0.0 for jid in DEX3_JOINTS}
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    # ── writer loop: 20 Hz re-assert in between ramps ───────────────────────
    def _writer_loop(self):
        while self._running:
            try:
                with self._lock:
                    # Before the deadline this writes last_q (same lock, value
                    # and rate). After it, recompute the bound per write without
                    # touching last_q, so the preset returns as soon as the
                    # residual drops under the cap. No extra lock or thread.
                    if time.time() < self._bound_deadline:
                        q = self.last_q
                    else:
                        q = self._residual_bound()
                    self._write(q)
            except Exception as e:
                _log(f'{self._side} writer error: {e}')
            # Edge-triggered mode != 1 alarm: the 1 Hz gate runs here because
            # this loop is 20 Hz, and one line is emitted per change instead of
            # 20 per second.
            try:
                line = self._check_faults()
                if line:
                    _log(line)
            except Exception as e:
                _log(f'{self._side} fault-check error: {e}')
            time.sleep(1.0 / WRITE_HZ)

    def _write(self, q_dict):
        if self._pub is None:  # NO_DDS: no writer, just state
            return
        self._pub.Write(_build_cmd(q_dict))

    # ── residual bound + fault annunciator ───────────────────
    def _read_limits(self):
        """motor_limits() off our reader, or None.  NEVER raises.

        The reader is optional and so is motor_limits (older/fake readers have
        only motor_q), so this uses getattr and swallows everything: a missing
        or misbehaving reader must degrade to "no DDS state", which both
        callers treat as "do nothing".  That is load-bearing — an unguarded
        dereference here sits inside _writer_loop / _state_payload and would
        turn every `state` reply into ERROR and every release into an abort.
        """
        if self._reader is None:
            return None
        fn = getattr(self._reader, 'motor_limits', None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception:
            return None

    def _closing_dir(self, jid):
        """+1/-1: the sign q MOVES in when this joint closes; 0 if pinned.

        Derived from the presets, not hand-typed per joint, so it cannot
        disagree with the ramp.  A pinned joint (OPEN == CLOSE) has no closing
        direction and is never bounded.
        """
        delta = CLOSE_PRESETS[self._side][jid] - OPEN_PRESETS[self._side][jid]
        if delta > 0.0:
            return 1.0
        if delta < 0.0:
            return -1.0
        return 0.0

    def _residual_bound(self):
        """last_q with each joint's CLOSING lead over the measured q capped.

        Returns the value to WRITE.  self.last_q is NOT rewritten: a clamp
        that persisted would freeze the hand at the short value exactly as the
        rejected design did, whereas recomputing per write means the re-assert
        returns to the preset the instant the residual falls back under
        RISER_MAX_RAD.

        No DDS state (no reader, no motor_limits, None) -> self.last_q
        UNCHANGED, so an absent reader can never silently reshape a command.

        Inert by construction at the shipped placeholders: RISER_MAX_RAD is
        inf, so `abs(lead) > RISER_MAX_RAD` is false for every joint and this
        is the identity.  See the RISER_MAX_RAD/SETTLE_S block.
        """
        limits = self._read_limits()
        if limits is None:
            return self.last_q
        q = dict(self.last_q)
        for jid in DEX3_JOINTS:
            try:
                q_meas = float(limits[jid][0])
            except Exception:
                continue  # bad sample; skip
            # Non-finite q is unusable. NaN comparisons are all false, so the
            # bound checks below would pass NaN to the MCU; reject it here.
            if not math.isfinite(q_meas):
                continue
            lead = q[jid] - q_meas
            if abs(lead) <= RISER_MAX_RAD:
                continue  # identity case
            cdir = self._closing_dir(jid)
            # Bound only the closing excess; OPEN stays untouched, and open
            # verification uses a 0.2 rad tolerance.
            if cdir == 0.0 or lead * cdir <= 0.0:
                continue
            q[jid] = q_meas + cdir * RISER_MAX_RAD
        return q

    def _faulted_modes(self):
        """{jid: mode} for every joint whose mode != 1, or None for no state.

        mode != 1 means NOT in servo mode.  mode = 0 is DISABLED (LIMP): ~0
        N.m and mechanically free, so a grip LOOSENS and the object slips —
        which is why `available` (a writer-liveness flag) cannot stand in for
        this and why the payload now carries `mode` itself.
        """
        limits = self._read_limits()
        if limits is None:
            return None
        bad = {}
        for jid in DEX3_JOINTS:
            try:
                mode = int(limits[jid][2])
            except Exception:
                continue
            if mode != 1:
                bad[jid] = mode
        return bad

    def _check_faults(self, now=None):
        """EDGE-TRIGGERED annunciator line, or None.  At most 1 Hz.

        B0.3 — one line when any mode != 1 APPEARS and one line when it
        CLEARS; nothing in between.  Called from the 20 Hz writer loop, so the
        1 Hz gate lives here rather than in the caller.  No DDS state emits
        nothing AND leaves the edge state alone: "no state" is not evidence
        that a fault cleared.
        """
        if now is None:
            now = time.time()
        if now - self._last_fault_check < 1.0:
            return None
        self._last_fault_check = now
        bad = self._faulted_modes()
        if bad is None:
            return None
        if bad and not self._fault_active:
            self._fault_active = True
            return (f'{self._side} FAULT: motor(s) NOT in servo mode '
                    f'(mode != 1): {bad} — mode=0 is DISABLED/LIMP (~0 N.m, '
                    f'backdrivable), so a grip held by it LOOSENS')
        if not bad and self._fault_active:
            self._fault_active = False
            return f'{self._side} FAULT CLEARED: all motors mode == 1'
        return None

    # ── the commands that actually drive the hand ───────────────────────────
    def set(self, targets, steps=STEP_DEFAULT, dt=DT_DEFAULT):
        """Cosine positional ramp from last commanded q to targets."""
        steps = max(1, int(steps))
        dt = max(0.005, float(dt))
        with self._lock:
            start = dict(self.last_q)
            for i in range(1, steps + 1):
                r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
                q = {jid: float(start[jid] + r * (targets.get(jid, start[jid])
                                                  - start[jid]))
                     for jid in DEX3_JOINTS}
                self._write(q)
                time.sleep(dt)
            self.last_q = {jid: float(targets.get(jid, start[jid]))
                           for jid in DEX3_JOINTS}
            # Arm the settle window after the endpoint is recorded, so the
            # 20 Hz re-assert keeps driving onto the preset for SETTLE_S.
            # That is what makes an air close converge instead of latching.
            self._bound_deadline = time.time() + SETTLE_S

    def open(self):
        self.set(OPEN_PRESETS[self._side])

    def close(self):
        self.set(CLOSE_PRESETS[self._side])

    def set_scalar(self, scalar):
        scalar = float(np.clip(scalar, -1.0, 1.0))
        t = (scalar + 1.0) / 2.0  # [-1,1] -> [0,1]
        targets = {}
        for jid in DEX3_JOINTS:
            targets[jid] = (OPEN_PRESETS[self._side][jid]
                            + t * (CLOSE_PRESETS[self._side][jid]
                                   - OPEN_PRESETS[self._side][jid]))
        self.set(targets)

    def _stage_pre_close(self, fingers_neutral=False):
        """Settle the thumb BEFORE the fingers close onto it.

        Returns the list of stage names actually issued (empty = nothing to
        do).  `set` holds any joint ABSENT from its `targets` at the current
        value, so each stage moves only the joints it names.

        DEFAULT (`fingers_neutral=False`) -- move ONLY the thumb base to its
        centreline, holding every other joint at its LIVE measured q.  This
        needs no loaded/empty sensor at all, because nothing is ever commanded
        toward OPEN and so a held object cannot be released.  It is the shape
        approved on 2026-09-27, and it is right for a hand arriving from the
        OPEN preset, where thumb_0 still has a 0.48 rad diagonal to remove.

        OPT-IN (`fingers_neutral=True`) -- straighten the four FINGER motors to
        0.0 FIRST, then drive the whole thumb to its CLOSE values, then close.

        WHY THE OPT-IN SHAPE EXISTS, and it is measured.  At the operator's
        2026-09-30 close the right hand arrived from a grasp-lead pose whose
        thumb_0 travel to CLOSE was exactly +0.0000 -- so the DEFAULT shape
        targets the one joint that needed nothing, and its 0.10 rad skip
        tolerance skips it outright -- while thumb_1 travelled -0.5605 and
        thumb_2 -0.8043 against a curling middle_0 at +0.7695.  The sweep that
        caused the reported finger collision is in the thumb's DISTAL joints,
        which neither this staging nor the 2026-09-27 one ever touched.

        WHY IT IS OPT-IN AND NOT DEFAULT.  Straightening the fingers IS a
        release if the hand holds something, and this hardware admits no
        PRE-close discriminator: the pads read a delta of ~0 for 5 of the 13
        characterised objects (Airpods, torch, lunchbox, epipen, spray-short),
        and the closure-block margin reads a FALSE 'blocked' on an empty hand
        after any bounded-lead close, because that close stops short of the
        preset BY DESIGN.  Both candidate sensors fail on the operator's own
        empty hand, so the risk cannot be engineered away -- it is routed to
        the one caller that is an explicit human close command instead.  Every
        other caller omits the flag and gets the default shape unchanged.
        """
        issued = []
        if not fingers_neutral:
            try:
                cur = float(self.last_q.get(THUMB_BASE_JID, THUMB_BASE_CENTRE))
            except Exception:
                return issued
            if abs(cur - THUMB_BASE_CENTRE) <= STAGED_CLOSE_SKIP_TOL:
                return issued
            self.set({THUMB_BASE_JID: THUMB_BASE_CENTRE})
            issued.append('thumb_0->centre')
            return issued

        # fingers_neutral: clear the volume the thumb's distals sweep, first.
        try:
            held_f = [float(self.last_q.get(j, 0.0)) for j in FINGER_JIDS]
        except Exception:
            held_f = []
        if held_f and any(abs(v) > STAGED_CLOSE_SKIP_TOL for v in held_f):
            self.set({j: 0.0 for j in FINGER_JIDS})
            issued.append('fingers->0.0')

        # then the whole thumb to its close preset.
        close = CLOSE_PRESETS[self._side]
        try:
            held_t = [float(self.last_q.get(j, 0.0)) for j in THUMB_JIDS]
            behind = any(abs(v - float(close[j])) > STAGED_CLOSE_SKIP_TOL
                         for j, v in zip(THUMB_JIDS, held_t))
        except Exception:
            behind = True
        if behind:
            self.set({j: close[j] for j in THUMB_JIDS})
            issued.append('thumb->close')
        return issued

    def close_adaptive(self, reader, max_fraction=ADAPTIVE_MAX_FRACTION,
                       steps=ADAPTIVE_STEPS, dt=ADAPTIVE_DT,
                       stop_threshold=ADAPTIVE_STOP,
                       min_steps=ADAPTIVE_MIN_STEPS,
                       fingers_neutral=False):
        """Incrementally close until finger pressure exceeds threshold.

        Thresholds preserved from the old daemon (close_until_contact):
        steps=50, dt=0.02, stop_threshold=200000, min_steps=10.  Pressure
        feedback is read IN-SERVER.

        max_fraction is still passed through clamped_close_target, i.e. capped
        at F_MAX_SAFE — which is 1.0 as of 2026-09-19, so the cap is currently
        a NO-OP on the endpoint.  The cap machinery is deliberately kept (see
        the F_MAX_SAFE comment): it is the single source of truth that stops
        the pose COMMANDED here and the pose SCORED in _score_closure_blocked
        from drifting apart.  Callers that want the preset without the ramp
        still have `set(CLOSE_PRESETS[side])`.

        Returns:
            'contact' — stopped early, finger grip detected
            'full'    — reached the (F_MAX_SAFE-clamped) fraction without
                        threshold
            'no_data' — no pressure data throughout (closed to max)
        Raises:
            RuntimeError('no_dds') — server in NO_DDS mode (no feedback
            channel at all); the client surfaces this as ERROR:no_dds.
        """
        if NO_DDS:
            raise RuntimeError('no_dds')
        steps = max(1, int(steps))
        dt = max(0.005, float(dt))
        min_steps = max(1, int(min_steps))
        # F_MAX_SAFE cap: the ramp is a straight line in joint space, so an
        # uncapped fraction transits the contact band.
        targets, _eff = clamped_close_target(self._side, max_fraction)
        if _eff < float(max_fraction):
            _log(f'{self._side} close_adaptive: max_fraction '
                 f'{float(max_fraction):.3f} CLAMPED to {_eff:.3f} '
                 f'(F_MAX_SAFE={F_MAX_SAFE:.3f}) — the OPEN->CLOSE ramp '
                 f'transits a self-collision band above this')
        baseline = reader.finger_pressure()
        _log(f'{self._side} close_adaptive: baseline={baseline:.0f}, '
             f'stop_threshold={stop_threshold:.0f}, min_steps={min_steps}, '
             f'steps={steps}')
        # Staging runs outside the lock because _stage_pre_close calls set(),
        # which takes self._lock, and threading.Lock is not reentrant.
        staged = self._stage_pre_close(fingers_neutral)
        if staged:
            _log(f'{self._side} close_adaptive: staged '
                 f'{", ".join(staged)} before the close ramp')
        with self._lock:
            start = dict(self.last_q)
            for i in range(1, steps + 1):
                r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
                q = {jid: float(start[jid] + r * (targets.get(jid, start[jid])
                                                  - start[jid]))
                     for jid in DEX3_JOINTS}
                self._write(q)
                time.sleep(dt)
                p = reader.finger_pressure()
                if i % 10 == 0 or (i > min_steps and p > baseline + 10000):
                    _log(f'{self._side} close_adaptive step {i}/{steps}: '
                         f'finger_pressure={p:.0f} (baseline={baseline:.0f}, '
                         f'stop_at={stop_threshold:.0f})')
                if i > min_steps and p > stop_threshold:
                    _log(f'{self._side} close_adaptive: GRIP DETECTED at '
                         f'step {i}/{steps} (finger_pressure={p:.0f} > '
                         f'threshold={stop_threshold:.0f}, '
                         f'delta={p - baseline:.0f})')
                    self.last_q = {jid: float(q[jid]) for jid in DEX3_JOINTS}
                    # Both exits arm the settle window after writing last_q,
                    # including the contact return.
                    self._bound_deadline = time.time() + SETTLE_S
                    return 'contact'
            self.last_q = {jid: float(targets[jid]) for jid in DEX3_JOINTS}
            self._bound_deadline = time.time() + SETTLE_S
        if baseline == 0:
            _log(f'{self._side} close_adaptive: NO DDS DATA — '
                 f'closed to max without feedback')
            return 'no_data'
        _log(f'{self._side} close_adaptive: reached full close, no grip '
             f'detected (final_pressure={reader.finger_pressure():.0f}, '
             f'baseline={baseline:.0f})')
        return 'full'

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)


# ── persistence: the last commanded q is saved so a restart re-asserts it ──
def _load_last_cmd():
    try:
        with open(LAST_CMD_FILE, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        q = {}
        for side in ('right', 'left'):
            side_q = data.get(side)
            if isinstance(side_q, dict):
                q[side] = {int(k): float(v) for k, v in side_q.items()}
        return q or None
    except Exception:
        return None


def _persist(controllers):
    try:
        payload = {side: {str(jid): controllers[side].last_q[jid]
                          for jid in DEX3_JOINTS}
                   for side in ('right', 'left')}
        payload['ts'] = time.time()
        with open(LAST_CMD_FILE, 'w') as f:
            json.dump(payload, f)
    except Exception as e:
        _log(f'WARNING: persist failed: {e}')


# ── command dispatch (one line in -> one line back out) ─────────────────────
def _handle_line(line, controllers, readers):
    parts = line.split()
    if not parts:
        return 'ERROR:empty command'
    cmd = parts[0]
    try:
        if cmd in ('open', 'close', 'open_left', 'close_left',
                   'open_both', 'close_both'):
            sides = []
            if cmd in ('open_both', 'close_both'):
                sides = ['right', 'left']
            else:
                sides = ['left' if cmd.endswith('_left') else 'right']
            for side in sides:
                if cmd.startswith('open'):
                    controllers[side].open()
                else:
                    controllers[side].close()
            _persist(controllers)
            return 'DONE'

        elif cmd == 'set':
            if len(parts) < 3 or parts[1] not in ('right', 'left'):
                return 'ERROR:set <side> <7f> [steps] [dt]'
            vals = []
            for tok in parts[2:]:
                if len(vals) >= 7:
                    break
                try:
                    vals.append(float(tok))
                except ValueError:
                    return 'ERROR:bad q value'
            if len(vals) != 7:
                return 'ERROR:need exactly 7 joint values'
            targets = {jid: vals[jid] for jid in DEX3_JOINTS}
            steps = STEP_DEFAULT
            dt = DT_DEFAULT
            if len(parts) >= 10:
                try:
                    steps = int(parts[9])  # set side q7 steps dt
                except ValueError:
                    return 'ERROR:bad steps'
            if len(parts) >= 11:
                try:
                    dt = float(parts[10])
                except ValueError:
                    return 'ERROR:bad dt'
            controllers[parts[1]].set(targets, steps=steps, dt=dt)
            _persist(controllers)
            return 'DONE'

        elif cmd == 'close_adaptive':
            if len(parts) < 2 or parts[1] not in ('right', 'left'):
                return 'ERROR:close_adaptive <side> [max_fraction steps dt stop_threshold min_steps] [fingers_neutral]'
            side = parts[1]
            # fingers_neutral may appear in any trailing position, so it can be
            # added without disturbing the five positional overrides; absence
            # means false. Strip the token before parsing, or float() hits it
            # and returns 'ERROR:bad max_fraction'.
            fingers_neutral = 'fingers_neutral' in parts[2:]
            if fingers_neutral:
                parts = [p for p in parts if p != 'fingers_neutral']
            mf = ADAPTIVE_MAX_FRACTION
            steps = ADAPTIVE_STEPS
            dt = ADAPTIVE_DT
            stop = ADAPTIVE_STOP
            min_steps = ADAPTIVE_MIN_STEPS
            if len(parts) >= 3:
                try:
                    mf = float(parts[2])
                except ValueError:
                    return 'ERROR:bad max_fraction'
            if len(parts) >= 4:
                try:
                    steps = int(parts[3])
                except ValueError:
                    return 'ERROR:bad steps'
            if len(parts) >= 5:
                try:
                    dt = float(parts[4])
                except ValueError:
                    return 'ERROR:bad dt'
            if len(parts) >= 6:
                try:
                    stop = float(parts[5])
                except ValueError:
                    return 'ERROR:bad stop_threshold'
            if len(parts) >= 7:
                try:
                    min_steps = int(parts[6])
                except ValueError:
                    return 'ERROR:bad min_steps'
            try:
                verdict = controllers[side].close_adaptive(
                    readers[side], max_fraction=mf, steps=steps, dt=dt,
                    stop_threshold=stop, min_steps=min_steps,
                    fingers_neutral=fingers_neutral)
            except RuntimeError as e:
                # no_dds is the only error close_adaptive raises
                return f'ERROR:{e}'
            _persist(controllers)
            return f'DONE:{verdict}'

        elif cmd == 'state':
            side = parts[1] if len(parts) > 1 else None
            if side is not None and side not in ('right', 'left'):
                return 'ERROR:state [side]'
            if side is None:
                payload = {s: _state_payload(controllers[s], readers[s])
                           for s in ('right', 'left')}
            else:
                payload = {side: _state_payload(controllers[side], readers[side])}
            return 'DONE:' + json.dumps(payload)

        elif cmd == 'pressure':
            # Finger-pad pressure from the same reader the adaptive close
            # stops on. 'max' is read by call sites; 'ok' is the availability
            # flag the insert guard keys off. When ok is false, 'max' is a 0.0
            # stand-in, not a measurement.
            side = parts[1] if len(parts) > 1 else None
            if side is not None and side not in ('right', 'left'):
                return 'ERROR:pressure [side]'
            sides = ('right', 'left') if side is None else (side,)
            payload = {}
            for s in sides:
                has = readers[s].has_data()
                payload[s] = {
                    'ok': bool(has),
                    'max': float(readers[s].finger_pressure()) if has else 0.0,
                }
            return 'DONE:' + json.dumps(payload)

        elif cmd == 'hold':
            # no-op: 20 Hz re-assert holds last_q
            side = parts[1] if len(parts) > 1 else None
            if side is not None and side not in ('right', 'left'):
                return 'ERROR:hold [side]'
            return 'DONE'

        elif cmd == 'quit':
            return 'DONE'

        return f'ERROR:unknown command "{cmd}"'
    except Exception as e:  # one bad command isn't fatal
        return f'ERROR:{e}'
