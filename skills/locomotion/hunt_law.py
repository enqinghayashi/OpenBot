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
