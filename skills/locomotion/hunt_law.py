#!/usr/bin/env python3
"""
hunt_law.py — pure hunt budget + stop/collision law for walk_node (Phase 1).

Implements the measurable, falsifiable gates of docs/NAV-DESIGN.md §3 + §8
folds 1&2, extracted HERE so they are unit-testable without rclpy. This
module is stdlib-only and imports nothing from walk_node.py; walk_node.py
imports it and passes live values in (no shared mutable state).

Gates implemented (the rest live in the FSM and are box-verified):
  G5/G17b — stop gate + near-edge law: floor classes stop on the centroid
            distance; wide-table and tall/large classes stop on the
            near-edge (d_centroid - max(width_m, depth_m)/2) when the
            projector published an extent; without published extent the
            centroid law applies unchanged (regression-proof).
  G5b     — StallWatchdog: a 15 s window in which d must decrease >= 0.05 m,
            else the approach stalls (FSM stops + SEARCHING / give-up).
  G6/G15  — HuntBudget: turn/round counters. 12 completed 30-degree turns =
            1 round; pre-measurement (decision 2026-09-04 #1) give-up happens
            at round 1 — no relocation. Stall accrual (D1 graft): stalls
            >= 2 per epoch -> epoch-exhaust -> give-up.
  G17a    — obstructs(): the approach freezes on another fresh label only
            when it stands BETWEEN the robot and the target (bearing within
            OBSTRUCTION_SECTOR_RAD of the target bearing AND nearer than the
            target). User decision 2026-09-04: the yolo gate grows to full
            COCO capacity in the future — ANY-label-everywhere would freeze
            on the target's own context in furnished scenes; the corridor
            law stays correct at 80 classes (it was chosen for that future).

Per-type class sets are the design's (NAV-DESIGN §8 fold 1&2 G5 table) and
include 'table' even though the 16-class gate cannot produce it today (the
gate expands later, at which point the law is already right).

Values marked box-calibratable are placeholders with the same status as
walk_node's STOP_DISTANCE_M — every one is listed in the re-entry checklist.
"""

import math


# ── G6 / G15 - the hunt budget ───────────────────────────────────────────────
SWEEP_TURNS_PER_ROUND  = 12    # 12 x 30° = 360°
STALL_BUDGET_PER_ROUND = 2     # 2 stalls = give up

# ── G5/G17b - the stop-gate law ─────────────────────────────────────────────
EDGE_STOP_MARGIN_M    = 0.25   # near-edge margin
                               # wide tables only
                               # calibrate on the box
MAX_REACHABLE_HEIGHT_M = 1.30  # tall z guard -> z_above_reach (honest)
                               # box-calibratable

WIDE_TABLE_CLASSES = {'table', 'chair', 'keyboard', 'mouse', 'laptop'}
TALL_LARGE_CLASSES = {'potted plant', 'clock', 'vase'}
FLOOR_CLASSES      = {'bottle', 'cup', 'apple', 'sports ball', 'backpack',
                      'book'}

# ── G17a - the obstruction corridor ──────────────────────────────────────────
OBSTRUCTION_SECTOR_RAD = 0.70          # ±40° bearing
# margin: the law said strictly nearer
# zero margin
# 0.680 m, the chair at 0.691 m -
# an 11 mm ranking flip that froze
# the approach.  We want evidence,
# not a coin-flip ordering.  Safe:
# the robot stops at 0.50 m anyway
# and re-tests the corridor each tick,
# a blocker still freezes early.
# operator-set margin, with
# the walk_node hold-not-abandon change.
OBSTRUCTION_MARGIN_M = 0.30
# hold budget: a blocked corridor may
# hold the target instead of spending
# the epoch's give-up budget; bounded
# so a wall still ends the epoch as
# NOT_FOUND instead of spinning forever
# through freeze -> hold -> re-trigger.
COLLISION_HOLD_MAX_S = 45.0
STALE_S = 2.0                          # freshness upstream

# ── G18 - the target-position stability gate ─────────────────────────────────
# measured: the latched target jumped
# 0.3-0.5 m / 0.1 s
# ticks; walked to the cup, said REACHED.
# Nothing real moves that fast.  At
# VX_WALK 0.30 m/s
# (0.06 m/0.2 s) + OMEGA_MAX 0.40 rad/s
# (~0.05 m lateral at 0.6 m), so the
# largest legit change is < 0.08 m.
# 0.15 m bound
# separates motion from a teleport.
# honest refusal
INSTABILITY_JUMP_M   = 0.15   # jump too big
INSTABILITY_MIN_DT_S = 0.15   # skip repeat reads
                              # 5.6 Hz detector
INSTABILITY_STRIKES  = 3      # 3 strikes = refuse


def _wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ── walking to ONE named object ──────────────────────────────
# label[#iid][@hue]
# iid 1+, hue 0..359
# top conf per label
# left one of two cups unaimable; ids
# 139->168->182; hue
# re-finds it.  We adopt the new id.

#: hue tol, deg
#: same colour
INSTANCE_HUE_TOL_DEG = 25.0
#: no hue
#: nearest 0.30 m
#: from last row
INSTANCE_NEAR_M = 0.30


def parse_target(s):
    """'<label>[#<iid>][@<hue>]' -> (label, iid|None, hue|None).  Never raises.

    Anything that does not fit the grammar exactly (empty label, iid < 1,
    hue > 359, a second '#'/'@', signs, spaces inside a number) returns the
    WHOLE stripped string as a bare label, which then behaves as before B8.
    """
    s = '' if s is None else str(s).strip()
    head, sep, hue_s = s.partition('@')
    label, sep2, iid_s = head.partition('#')
    label = label.strip()
    ok = bool(label) and (sep or sep2)
    if sep and not (hue_s.isdigit() and hue_s.isascii() and int(hue_s) <= 359):
        ok = False
    if sep2 and not (iid_s.isdigit() and iid_s.isascii() and int(iid_s) >= 1):
        ok = False
    if not ok:
        return (s, None, None)
    return (label, int(iid_s) if sep2 else None, int(hue_s) if sep else None)


def hue_gap(a, b):
    """Circular distance in degrees between two hues; None if either is unusable."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    d = abs(a % 360.0 - b % 360.0)
    return min(d, 360.0 - d)


def select_instance(rows, iid, hue, last_xy):
    """(row|None, 'iid'|'hue'|'near'|'lost') among same-label rows of ONE message.

      1. the row whose instance_id == iid                              'iid'
      2. hue known: rows within INSTANCE_HUE_TOL_DEG of it - nearest to
         last_xy if known, else the highest conf                        'hue'
      3. hue UNKNOWN and last_xy known: nearest row within INSTANCE_NEAR_M  'near'
      4. otherwise (None, 'lost') - NEVER another object of the class.
    The caller adopts the chosen row's instance_id after a 'hue'/'near' match.
    """
    def _xy(r):
        try:
            x, y = float(r['x']), float(r['y'])
        except (KeyError, TypeError, ValueError):
            return None
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None

    def _dist(r):
        p = _xy(r)
        if p is None or last_xy is None:
            return float('inf')
        return math.hypot(p[0] - last_xy[0], p[1] - last_xy[1])

    rows = [r for r in (rows or []) if isinstance(r, dict) and _xy(r) is not None]
    if iid is not None:
        for r in rows:
            v = r.get('instance_id')
            if isinstance(v, int) and not isinstance(v, bool) and v == iid:
                return r, 'iid'
    if hue is not None:
        same = []
        for r in rows:
            g = hue_gap(hue, r.get('hue'))
            if g is not None and g <= INSTANCE_HUE_TOL_DEG:
                same.append(r)
        if same:
            if last_xy is not None:
                return min(same, key=_dist), 'hue'
            return max(same, key=lambda r: r.get('conf', 0) or 0), 'hue'
        return None, 'lost'
    if last_xy is not None and rows:
        best = min(rows, key=_dist)
        if _dist(best) <= INSTANCE_NEAR_M:
            return best, 'near'
    return None, 'lost'


class HuntBudget(object):
    """G6/G15 proof counters for one hunt epoch (per position; Phase 1 = 1).

    exhausted() == (round complete) or (stall budget spent). The proof dict
    is exactly what the FSM publishes in the NOT_FOUND status detail and
    what the Phase-2 verifier evidence gate checks against the constants.
    """

    def __init__(self, turns_per_round=SWEEP_TURNS_PER_ROUND,
                 stall_budget=STALL_BUDGET_PER_ROUND):
        self._turns_per_round = turns_per_round
        self._stall_budget = stall_budget
        self.reset()

    def reset(self):
        self._turns = 0
        self._windows = 0
        self._stalls = 0
        self._stall_reasons = {}

    @property
    def turns(self):
        return self._turns

    @property
    def windows(self):
        return self._windows

    @property
    def stalls(self):
        return self._stalls

    def add_window(self):
        """One stationary DETECT_WINDOW_S window elapsed."""
        self._windows += 1

    def add_turn(self):
        """One COMPLETED sweep turn (G6: interrupted turns do not count)."""
        self._turns += 1

    def add_stall(self, reason):
        """One no-progress/stall event; reason is a string label."""
        self._stalls += 1
        self._stall_reasons[reason] = self._stall_reasons.get(reason, 0) + 1
        return self._stalls

    @property
    def round_complete(self):
        return self._turns >= self._turns_per_round

    @property
    def exhausted(self):
        """G6 (round) or D1-graft (stall budget) -> epoch-exhaust."""
        return self.round_complete or self._stalls >= self._stall_budget

