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

    @property
    def reason(self):
        """Exhaustion classifier for the NOT_FOUND detail."""
        if self.round_complete:
            return 'round_complete'
        return 'stall_budget'

    def proof(self):
        """NOT_FOUND detail — the Phase-2 evidence-gate payload."""
        return {
            'reason': self.reason,
            'rounds': self._turns // self._turns_per_round,
            'turns': self._turns,
            'windows': self._windows,
            'stalls': self._stalls,
            'stall_reasons': dict(self._stall_reasons),
            'turns_per_round': self._turns_per_round,
            'stall_budget': self._stall_budget,
            'positions_attempted': 1,     # no measurement yet
            'hop_implemented': False,     # hop on hold
        }


class CorridorHold(object):
    """G17a corridor-hold clock (2026-09-21, operator decision).

    Tracks ONE unbroken streak of collision freezes.  `frozen(now)` returns
    ('hold', held_s) while the streak is inside the budget and
    ('give_up', held_s) once it exceeds it; `clear()` breaks the streak.

    Why a hold at all: a freeze used to spend the epoch's give-up budget
    immediately, so a person stepping through the corridor abandoned the
    object.  Holding keeps the target — the coordinator's re-arm re-triggers
    /walk_go on the next confirming beats, and the approach resumes the moment
    the corridor clears.

    Why a budget: without one, freeze -> hold -> re-trigger is unbounded, and
    a permanently blocked corridor (a wall, a chair wedged in front) would
    spin forever instead of ending the epoch honestly.

    Pure time-function (pass `now` in) so unit tests never sleep.
    """

    def __init__(self, hold_max_s=COLLISION_HOLD_MAX_S):
        self._max = hold_max_s
        self._since = 0.0

    def clear(self):
        self._since = 0.0

    def frozen(self, now):
        if not self._since:
            self._since = now
        held = now - self._since
        if held > self._max:
            self._since = 0.0
            return 'give_up', held
        return 'hold', held


class StallWatchdog(object):
    """G5b: within STALL_WINDOW_S the approach distance must decrease by at
    least STALL_MIN_PROGRESS_M; otherwise update() returns 'stall' exactly
    once and the window restarts.

    Pure time-function (pass `now` in) so unit tests never sleep.
    """

    def __init__(self, window_s=15.0, min_progress_m=0.05):
        self._window = window_s
        self._min_progress = min_progress_m
        self.reset()

    def reset(self):
        self._t0 = None
        self._d0 = 0.0

    def update(self, d, now):
        """Returns 'ok' (window open or progress made) or 'stall'."""
        if self._t0 is None:
            self._t0, self._d0 = now, d
            return 'ok'
        if now - self._t0 >= self._window:
            stalled = (self._d0 - d) < self._min_progress
            self._t0, self._d0 = now, d
            return 'stall' if stalled else 'ok'
        return 'ok'


class TargetStability(object):
    """G18: refuse a target whose body-frame position teleports.

    Counts CONSECUTIVE over-bound moves (INSTABILITY_JUMP_M) and returns
    'unstable' once INSTABILITY_STRIKES is reached; any in-bound move clears
    the streak, so a genuinely moving object the robot is tracking correctly
    never trips it.  Two reads closer than INSTABILITY_MIN_DT_S are treated as
    one measurement (the control loop outruns the detector), so a stale repeat
    cannot manufacture a zero-length "jump" or reset the streak for free.

    Pure time-function (pass `now` in) so unit tests never sleep — the
    StallWatchdog convention.
    """

    def __init__(self, jump_m=INSTABILITY_JUMP_M,
                 min_dt_s=INSTABILITY_MIN_DT_S, strikes=INSTABILITY_STRIKES):
        self._jump_m = jump_m
        self._min_dt = min_dt_s
        self._max_strikes = strikes
        self.reset()

    def reset(self):
        self._t = None
        self._x = 0.0
        self._y = 0.0
        self._strikes = 0
        self._last_jump = 0.0

    def update(self, x, y, now):
        """Returns 'ok' or 'unstable'.  `last_jump` holds the newest measured
        jump (0.0 until a second distinct measurement arrives)."""
        if self._t is None:
            self._t, self._x, self._y = now, x, y
            return 'ok'
        if now - self._t < self._min_dt:
            return 'ok'                     # repeat read
        self._last_jump = math.hypot(x - self._x, y - self._y)
        self._t, self._x, self._y = now, x, y
        if self._last_jump > self._jump_m:
            self._strikes += 1
        else:
            self._strikes = 0
        return 'unstable' if self._strikes >= self._max_strikes else 'ok'

    @property
    def strikes(self):
        return self._strikes

    @property
    def last_jump(self):
        return self._last_jump


def stop_law(label, x, y, z, width_m, depth_m, stop_distance_m):
    """G5 + G17b per-type stop zone. Returns (in_zone, evidence).

    evidence carries the law + numbers for the REACHED detail so the honest
    stop reason survives to downstream consumers (e.g. z_above_reach).
    floor or unknown classes -> centroid law (design: 'in front of' = d).
    """
    d = math.hypot(x, y)
    label = str(label or '')
    ev = {'law': 'centroid', 'd_m': round(d, 3), 'z_m': round(z, 3)}
    if label in WIDE_TABLE_CLASSES or label in TALL_LARGE_CLASSES:
        if width_m is None or depth_m is None or max(width_m, depth_m) <= 0.0:
            # no extent
            ev['law'] = 'centroid_no_extent'
            return d <= stop_distance_m, ev
        half = max(width_m, depth_m) / 2.0
        d_edge = max(0.0, d - half)
        ev['law'] = 'near_edge'
        ev['d_edge_m'] = round(d_edge, 3)
        if label in TALL_LARGE_CLASSES:
            ev['z_above_reach'] = z > MAX_REACHABLE_HEIGHT_M
        return d_edge <= EDGE_STOP_MARGIN_M, ev
    return d <= stop_distance_m, ev


def arrival_latch(entry, now, stop_distance_m, latch_window_s=5.0, law=None):
    """#214 (nav-close): arrival latch on the RETAINED measurement.

    The approach loop only commits REACHED while a FRESH detection (or a
    fresh-odom latch) is present, but the depth projector drops the label at
    close range — the robot can be physically inside the stop gate with no
    fresh sub-gate measurement ever arriving again, so the FSM rolled back to
    SEARCHING exactly as it arrived. This law re-evaluates the last
    measurement retained for the label (which survives staleness in
    walk_node's ``self._targets``): if it is still inside the stop zone and
    not older than ``latch_window_s``, the approach is told REACHED.

    Returns (in_zone, evidence). evidence carries the stop-law numbers plus
    the measurement age ('latch_age_s').  Honest negative control: an entry
    beyond the window — or one that was never in-zone — NEVER latches
    (returns False); the FSM keeps its normal lost-track rollback for those.

    B8j: ``law`` is the stop law to re-evaluate with (walk_node passes
    near_edge_law unless OPENBOT_STOP_LAW=legacy); None = the B8i stop_law.
    """
    if entry is None:
        return False, {}
    age = now - entry['t']
    if age > latch_window_s:
        return False, {'law': 'expired', 'latch_age_s': round(age, 1)}
    in_zone, ev = (law or stop_law)(
        entry.get('label'), entry.get('x', 0.0), entry.get('y', 0.0),
        entry.get('z', 0.0), entry.get('width_m'), entry.get('depth_m'),
        stop_distance_m)
    ev['latch_age_s'] = round(age, 1)
    return in_zone, ev
