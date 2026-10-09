#!/usr/bin/env python3
"""mj_height_sweep.py — laptop-only MuJoCo height-reference sweep (no robot).

Drives the REAL GR00T Balance policy through the EXACT obs stack the writer
uses (obs_builder.ObsBuilder + policy_runner.PolicyRunner) in g1_gear_wbc.xml,
ramping the policy HEIGHT REFERENCE f[3] (obs.height_cmd) DOWN, measuring the
settled response at each height, and rendering video so the operator can SEE
the squat.

Why mutating obs.height_cmd reaches the actor (CONFIRMED-TRACE):
  ObsBuilder.__init__ stores height_cmd once, but build_frame() dereferences
  self.height_cmd LIVE on every call (obs_builder.py:101, `f[3] = self.height_cmd`)
  and push() copies that freshly-built frame into the obs tail
  (obs_builder.py:111). So assigning `obs.height_cmd = <new>` before the next
  obs.push() reaches the actor on the VERY NEXT policy tick (~20 ms at 50 Hz).
  Caveat: only the newest of the 6 history frames carries the new value; the
  other 5 age out over 6 ticks (~0.12 s) — a smooth ramp hides that latency.

Modes:
  stepped = reset to --start-height, settle, command each rung for --hold-s
            (instant step), measure settled.
  ramped  = continuous --ramp-s descent (the rendered one).  JUDGES ONLY the
            endpoint rung's settled tail; intermediate rungs are crossed, not
            judged — the summary now says so explicitly.
  inband  = settle, then a SLEW-LIMITED (writer height_slew_rate) settled
            stepped sweep over the WRITER'S ACTUAL DEPLOYED BAND
            [height_cmd_min, height_cmd_max] read from the config, with each
            rung clamped to that band.  This is the transient the deployed
            writer really produces (the writer rate-limits every command and
            clamps it into the band); every rung is settled and judged.
  recover = the ASCENT question.  Per ladder rung: RESET at --start-height
            (standing), slew the reference DOWN to the rung at the writer slew
            rate, HOLD until settled, then slew it straight BACK UP to standing
            at the same rate and measure whether the policy stands back up.
            Per rung: RECOVERED (settled back within --hold-tol of standing),
            final settled z, time to ascend (s), overshoot, and whether it fell.
            Descending is gravity-assisted; this measures the concentric push
            out of the deep lean with the heels lifted.
  chained = ONE continuous run, NO re-settling between rungs: a single
            writer-rate slew DOWN through the whole ladder to the deepest rung,
            then straight back UP to standing.  The realistic single-command
            usage pattern (not a staircase of independent re-settles).  Per-rung
            crossings are reported on BOTH legs (momentary, not settled) plus
            the final standing state.

Fall detection is HEIGHT-AWARE: mj_policy_check.py:141's hardcoded
`qpos[2] < 0.4` scores a legal deep squat as a FALL and is NOT reused; here
fall <=> z < height_cmd - fall_drop, or |roll|/|pitch| > max_tilt, or
z < min_abs_z.  The gate is NEVER weakened to make a rung pass.

ASCENT-AWARE GATE (recover / chained only).  `z < height_cmd - fall_drop` is a
LOWER-BOUND test on the reference.  It is sound while height_cmd is
NON-INCREASING (descent / steady): lag then leaves z ABOVE the reference, and
losing > fall_drop below a settled command is a real collapse.  While the
reference RISES the roles invert — the reference leads the body, tracking lag
pulls z BELOW height_cmd, and the raw test fires on ordinary lag, not a fall
(a STEPPED rise, a fast --slew-rate, or a recovery from a perturbed pose all
false-trip).  So recover/chained gate on the RUNNING MINIMUM of height_cmd over
the segment, `gate_ref = min_{s<=t} height_cmd(s)`, with the SAME fall_drop:
    fall <=> z < gate_ref - fall_drop  OR  |roll| > max_tilt
             OR  |pitch| > max_tilt  OR  z < min_abs_z
PROOF the descent gate is unchanged: on any non-increasing segment
min_{s<=t} height_cmd(s) == height_cmd(t), so gate_ref == height_cmd and the
test is bit-identical to the plain descent gate (this is exactly why a running
MINIMUM is used and not a running PEAK — a peak-relative gate would false-trip
every legal deep squat on the way down, weakening the descent gate, which is
forbidden).  On the rising leg gate_ref stays frozen at the deepest commanded
height, i.e. the gate becomes "do not collapse below the deepest rung you were
actually told to hold" — the lower bound that stays meaningful while the body
pushes up.  The tilt and min_abs_z terms are ascent-safe and UNCHANGED.  In
every other mode ascent_aware is False, so those modes keep the literal
`z < height_cmd - fall_drop` gate unchanged.

Usage: python3 tools/mj_height_sweep.py --mode ramped  |  --mode stepped
       python3 tools/mj_height_sweep.py --mode inband
       python3 tools/mj_height_sweep.py --mode recover   (ascent / recovery)
       python3 tools/mj_height_sweep.py --mode chained   (ascent / recovery)
"""

import argparse
import json
import os
import sys
import time

# OSMesa isn't on this box; EGL works here
# Renderer dies with
# "'NoneType' object has no attribute 'glGetError'"
# (probe: 320x240 uint8 OK)
# setdefault, so an override wins
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import mujoco

from obs_builder import ObsBuilder
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")
XML_DEFAULT = os.path.join(PKG, "sim", "g1_gear_wbc.xml")

# XML joint names
# doc order == G1JointIndex 0..14
POLICY_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
]

# report indices (== G1JointIndex)
I_LKNEE, I_LANKP = 3, 4
I_RKNEE, I_RANKP = 9, 10

DEFAULT_LADDER = [0.74, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]


def band_rungs(lo, hi, stand, step=0.05):
    """Descending rungs spanning the writer's deployed band [lo, hi].

    Steps down from hi by `step`, always includes lo and the nominal standing
    height when it lies inside the band.  Every returned rung is inside
    [lo, hi] — the same interval obs_builder.set_height / the writer clamps to,
    so no rung here can exercise a height the deployed writer could not.
    """
    rungs = []
    h = hi
    while h > lo + 1e-9:
        rungs.append(round(h, 3))
        h -= step
    rungs.append(round(lo, 3))
    if lo - 1e-9 <= stand <= hi + 1e-9:
        rungs.append(round(stand, 3))
    return sorted({r for r in rungs}, reverse=True)


def _win_median(ts, zs, t_a, t_b):
    """Median z over [t_a, t_b]; NaN if the window is empty (a truncated trace)."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = (ts >= t_a) & (ts <= t_b)
    return float(np.median(zs[sel])) if bool(sel.any()) else float("nan")


def _win_max(ts, zs, t_a, t_b):
    """Peak z over [t_a, t_b]; NaN if the window is empty."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = (ts >= t_a) & (ts <= t_b)
    return float(zs[sel].max()) if bool(sel.any()) else float("nan")


def _first_at_or_above(ts, zs, thr, t_from=0.0):
    """First t >= t_from with z >= thr; NaN if never reached (truncated trace)."""
    ts = np.asarray(ts)
    zs = np.asarray(zs)
    if ts.size == 0:
        return float("nan")
    sel = np.flatnonzero((ts >= t_from) & (zs >= thr))
    return float(ts[int(sel[0])]) if sel.size else float("nan")


