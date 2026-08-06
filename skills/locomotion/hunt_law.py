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


def obstructs(tx, ty, ox, oy, sector_rad=OBSTRUCTION_SECTOR_RAD,
              margin_m=OBSTRUCTION_MARGIN_M):
    """G17a: is (ox, oy) BETWEEN the robot and the target at (tx, ty)?

    Both in the body frame (x lateral, y forward). Requires: bearing within
    ±sector_rad of the target bearing, AND nearer than the target by more
    than margin_m.

    The margin is the 2026-09-21 fix for a zero-margin law that froze an
    approach over an 11 mm ranking flip between two noisy depths (see
    OBSTRUCTION_MARGIN_M).  A target inside the margin is never obstructed —
    the robot is already inside the stop gate there, so the stop law owns it.
    """
    t_d = math.hypot(tx, ty)
    o_d = math.hypot(ox, oy)
    if o_d >= t_d - margin_m:
        return False
    t_bearing = math.atan2(tx, ty)
    o_bearing = math.atan2(ox, oy)
    return abs(_wrap_pi(o_bearing - t_bearing)) <= sector_rad


# ── near-edge stop law, every object ────────────────────────
# near-edge law: every
# object stops on its NEAR EDGE.
# pick 0.50 m
# a bottle stops where it does today
# (centre ~0.54 m); bigger ones stop
# earlier, not closer.  Pick gate =
# STOP_DISTANCE_M 0.55 with half-size 0.05.
# Non-pickables END at 0.45 m; the gate
# fires LARGE_LEAD_M early (measured travel).
# the bin row
# vanished at 0.742 m, came back as chair/
# handbag/backpack at x~0, z~-0.55, h~0.42;
# walked blind 2.0 s at vx 0.28, lost it.
# This law stops that bin at 0.706 (near
# edge 0.472); stop_law unchanged;
# OPENBOT_STOP_LAW=legacy falls back.

#: mirror list
#: of tools.py NOT_PICKABLE_CLASSES
#: ('mouse' is pickable there); the box
#: can't import brain/ - a drift test holds
#: it: the walk-law drift test.
NOT_PICKABLE_MIRROR = frozenset((
    "person", "chair", "couch", "bed", "dining table", "tv", "laptop",
    "keyboard", "toilet", "refrigerator", "oven", "sink",
    "microwave", "monitor", "desk", "table", "door", "window", "bench",
    "potted plant",
))
#: place targets
#: containers
#: luggage: brain allows, Dex3 can't lift;
#: SEED_TABLE bin, box, basket, bucket;
#: base suitcase/backpack/handbag = every
#: place-INTO container class (tools.py
#: CONTAINER_CLASSES == strict_vocab.py)
#: minus 'bowl'; tray, pot, pan, lunchbox,
#: toolbox (0 rows in 89 depth.logs) also
#: END at 0.45 m
#: drift: an integration test.
LARGE_EXTRA_CLASSES = frozenset(('bin', 'box', 'basket', 'bucket',
                                 'suitcase', 'backpack', 'handbag',
                                 'tray', 'pot', 'pan', 'lunchbox', 'toolbox'))
#: rest = 'pick'
NONPICKABLE_CLASSES = NOT_PICKABLE_MIRROR | LARGE_EXTRA_CLASSES

#: class prior
#: extents publish 2-4x their real size
#: (bottle 0.126 vs 0.035 m); 0.05 = the
#: gate; bigger classes stop earlier.
PICK_HALF_DEFAULT_M = 0.05
PICK_HALF_M = {
    'bowl': 0.10,   # bowl: w 0.19-0.23, log p50 0.233 m
    'book': 0.10,   # book: every log w p50 0.34 m (flat)
}
PICK_EDGE_M = 0.50  # = STOP_DISTANCE_M 0.55 - 0.05 (pinned)

#: large tier
#: ENDS at LARGE_EDGE_M 0.45; LARGE_LEAD_M
#: fires early: +0.018/0.028/0.037/0.044 m
LARGE_EDGE_M = 0.45
LARGE_LEAD_M = 0.05
# clamp
#: floor stops a truncated extent (rim out of
# frame), cap stops a leaked/merged one.
# chair 1.03 x 1.81 m REACHED at
# 1.122; a tub plus the box under it,
# dd 0.73 -> stop 0.77, near edge ~0.53.
# max(w,dd)/2 at 0.35-0.9 m: bin p10/p50/
# p90 .196/.230/.350, box .037/.231/.253,
# person .058/.183/.349, chair .089/.186/
# .431; operator bin 0.45 (half .225);
# >0.50 m reads (half-0.25) farther: 0.60
# box ~0.40, clear of the 0.29 m toe.
LARGE_HALF_MIN_M = 0.15
LARGE_HALF_MAX_M = 0.25
#: person
#: (floor, cap) when the default is wrong:
#: torso w 0.33 / dd 0.30 (half ~0.16),
#: moving p90 0.35 (12:44: w 0.49,
#: dd 0.71); body ~0.25 m deep -> (0.12,
#: 0.20), else handover stops 0.05 early.
LARGE_HALF_BOUNDS = {
    'person': (0.12, 0.20),
}


def stop_tier(label):
    """'large' for a non-pickable class (NONPICKABLE_CLASSES), else 'pick'."""
    return ('large' if str(label or '').strip().lower() in NONPICKABLE_CLASSES
            else 'pick')


def _published_half(width_m, depth_m):
    """max(width_m, depth_m) / 2 over the usable values, or None."""
    vals = []
    for v in (width_m, depth_m):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            vals.append(v)
    return max(vals) / 2.0 if vals else None


def near_edge_law(label, x, y, z, width_m, depth_m, stop_distance_m):
    """B8j stop law (same signature and return shape as stop_law).  Tier =
    stop_tier(label): the class decides, never the published geometry.

    pick tier : half = PICK_HALF_M.get(label, PICK_HALF_DEFAULT_M); in zone
                iff d <= stop_distance_m + (half - PICK_HALF_DEFAULT_M), i.e.
                the near edge d - half is within stop_distance_m -
                PICK_HALF_DEFAULT_M (0.50).  For the default half the test is
                literally the B8i `d <= stop_distance_m`.
    large tier: half = max(width_m, depth_m)/2 clamped to the class bounds
                (LARGE_HALF_BOUNDS, else [LARGE_HALF_MIN_M, LARGE_HALF_MAX_M]);
                no usable extent -> the cap, the cautious side; in zone iff
                max(0, d - half) <= LARGE_EDGE_M + LARGE_LEAD_M.
    A non-finite position is never in zone.  Evidence keeps the B8i keys
    first (law, d_m, z_m, d_edge_m[, z_above_reach]) and appends tier,
    half_m, half_src (class|pub|floor|cap|none), gate_m, x_m, y_m.
    """
    try:
        x, y, z = float(x), float(y), float(z)
    except (TypeError, ValueError):
        return False, {'law': 'invalid'}
    d = math.hypot(x, y)
    if not (math.isfinite(d) and math.isfinite(z)):
        return False, {'law': 'invalid'}
    label = str(label or '')
    key = label.strip().lower()
    ev = {'law': 'near_edge', 'd_m': round(d, 3), 'z_m': round(z, 3)}
    tier = stop_tier(key)
    if tier == 'large':
        lo, hi = LARGE_HALF_BOUNDS.get(key, (LARGE_HALF_MIN_M,
                                             LARGE_HALF_MAX_M))
        pub = _published_half(width_m, depth_m)
        if pub is None:
            half, src = hi, 'none'
        elif pub < lo:
            half, src = lo, 'floor'
        elif pub > hi:
            half, src = hi, 'cap'
        else:
            half, src = pub, 'pub'
        gate = LARGE_EDGE_M + LARGE_LEAD_M
        d_edge = max(0.0, d - half)
        in_zone = d_edge <= gate
    else:
        half, src = PICK_HALF_M.get(key, PICK_HALF_DEFAULT_M), 'class'
        gate = stop_distance_m - PICK_HALF_DEFAULT_M
        d_edge = max(0.0, d - half)
        in_zone = d <= stop_distance_m + (half - PICK_HALF_DEFAULT_M)
    ev['d_edge_m'] = round(d_edge, 3)
    if label in TALL_LARGE_CLASSES:
        ev['z_above_reach'] = z > MAX_REACHABLE_HEIGHT_M
    ev['tier'] = tier
    ev['half_m'] = round(half, 3)
    ev['half_src'] = src
    ev['gate_m'] = round(gate, 3)
    ev['x_m'] = round(x, 3)
    ev['y_m'] = round(y, 3)
    return in_zone, ev


# ── target continuity when the label flips ──────────────────
# iid outlives flips
# measured: bin -> backpack, iid 158, same
# xyz to 1 mm;
# 12:41:36 bowl -> person, iid 188,
# 0.04 m apart; the id continues the object.
#: radius 0.20 m
#: = SAME_OBJECT_M
#: walk_node's 'two labels on one object'
CONT_RADIUS_M = 0.20
#: dz 0.15 m; torso z~0 never
#: continues a floor bin z~-0.5
CONT_DZ_M = 0.15
#: window 1.0 s
CONT_WINDOW_S = 1.0


def _row_xyz(r):
    try:
        x, y, z = float(r['x']), float(r['y']), float(r['z'])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return x, y, z


def _row_iid(r):
    v = r.get('instance_id') if isinstance(r, dict) else None
    if isinstance(v, int) and not isinstance(v, bool) and v >= 1:
        return v
    return None


