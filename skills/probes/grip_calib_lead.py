#!/usr/bin/env python3
"""grip_calib_lead.py — BOUNDED-LEAD right-hand grip calibration.  Runs ON the robot.

WHY THE PREVIOUS TWO ATTEMPTS WERE BOTH WRONG, AND THIS ONE IS NOT.

Attempt 1 (`grip_calib.py`) called `client.close()` — every joint to the CLOSE
preset in one blind write.  That is the rail path.  Result: thumb_1, index_1,
middle_0 latched (q frozen between `closed` and `reopened`), 2026-09-27.

Attempt 2 (`grip_calib_stepped.py`, DELETED) stepped the closure in fractions
0.0 .. 0.6 with `set_scalar`.  It looked safer and is NOT: `set_scalar`
commands an ABSOLUTE pose from the presets, so on a finger the object has
stopped, every step advances the COMMAND another ~0.17 rad while q_meas does
not move.  The lead over the measured pose ACCUMULATES — 0.6 * 1.745 = 1.047
rad by the last step, well past the ~0.83 rad rail.  Same latch, slower.

The engine already states the rule (`pick_engine_v3.py:2377`): "because the
reference is a FRESH q_meas, never the previous command, the lead cannot
accumulate across presses".  This script is that rule.

WHAT THIS DOES.  A walk of single-write presses, each one commanding

    target[jid] = clamp( q_meas[jid] + eps * closing_dir , toward the preset )

read from a LIVE q_meas, keyed by INT jid, all seven joints present, written
with `steps=1` so the only write IS the endpoint and its lead is eps BY
CONSTRUCTION.  This is `_bounded_lead_target` reproduced from the engine.

It sweeps eps over 0.06 .. 0.25 rad — the design window is 0.06 < lead < 0.83,
and 0.25 is the shipped `HAND_CLOSE_LEAD_EPS`.  At each eps it walks to a fixed
point and records the pressure, producing the curve that a pressure threshold
has to be read off:

    lead(rad) -> pressure max

That curve is the calibration.  No constant is set here and none is touched.

SAFETY.  Socket client only — no DDS, no publisher, no arm, no walking.
Every write is lead-bounded against a fresh q_meas, so torque is bounded by
tau ~= bias + 16.6 * lead with no model in the loop.  Max commanded lead 0.25
rad = ~30 % of the measured rail.  The hand is left OPEN.

USAGE (on the robot)
    python3 /tmp/grip_calib_lead.py            # empty hand
    python3 /tmp/grip_calib_lead.py bottle     # seats the bottle first (ssh -t)
"""

import json
import math
import sys
import time

sys.path.insert(0, '/home/unitree/leon')
sys.path.insert(0, '/home/unitree/leon/openbot')   # engine's :153 insert; this one wins

from grip_client import GripClient                  # noqa: E402
from grip_server import (                           # noqa: E402
    DEX3_JOINTS, OPEN_PRESETS, CLOSE_PRESETS)

SIDE = 'right'
EPS_LADDER = [0.06, 0.10, 0.15, 0.20, 0.25]
MAX_PRESSES = 40               # per eps; usually converges sooner
MAX_CEILING_PRESSES = 3        # extra presses once the command sits AT the preset
DT = 0.02
PRESS_SETTLE_S = 0.25
TARGET_TOL = 2.0e-3            # commands this close == fixed point
CEIL_TOL = 1.0e-6              # command already IS the preset
RETURN_TOL = 0.05              # won't return this far == latched
LABEL = 'bottle' if len(sys.argv) > 1 and sys.argv[1] == 'bottle' else 'empty'


def close_dir(jid):
    """+1/-1: the sign q moves when `jid` closes.  Derived from the presets."""
    d = float(CLOSE_PRESETS[SIDE][jid]) - float(OPEN_PRESETS[SIDE][jid])
    return 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)


def q_meas(c):
    """Live per-joint q, INT-keyed.  {} if the state channel is unreadable."""
    try:
        payload = c.state(SIDE).get(SIDE, {})
    except Exception as e:                                    # noqa: BLE001
        print(f'  state read failed: {type(e).__name__}: {e}')
        return {}
    return {int(k): float(v) for k, v in (payload.get('actual') or {}).items()}


def bounded_target(q, eps):
    """`_bounded_lead_target` from pick_engine_v3.py:2361, in behaviour.

    None on any joint we cannot see or that is non-finite — NEVER invents a
    command (an inf q would clamp straight to the preset, which is the exact
    full-rail command this exists to remove).
    """
    tgt = {}
    for jid in DEX3_JOINTS:
        if jid not in q or not math.isfinite(q[jid]):
            return None
        cd = close_dir(jid)
        if cd == 0.0:
            tgt[jid] = q[jid]        # pinned: no direction, no travel
            continue
        want = q[jid] + cd * float(eps)
        ceil_j = float(CLOSE_PRESETS[SIDE][jid])
        tgt[jid] = float(min(want, ceil_j) if cd > 0.0 else max(want, ceil_j))
    return tgt


def within(a, b, tol=TARGET_TOL):
    return all(abs(a[j] - b[j]) <= tol for j in DEX3_JOINTS)


def show(q):
    return {j: round(q.get(j, float('nan')), 3) for j in sorted(q)}


def main():
    c = GripClient(SIDE)
    print(f'===== grip_calib_lead: {LABEL} =====')
    print('hand-only. Every write is lead-bounded against a LIVE q_meas.')
    print('The ARM will not move.\n')

    print('reopening to establish the free-air reference')
    c.open()
    time.sleep(2.5)
    q_open = q_meas(c)
    pr = c.pressure(SIDE).get(SIDE, {})
    base_p = float(pr.get('max') or 0.0)
    print(f'  free air: ok={pr.get("ok")} max={base_p}')
    print(f'  q_open  : {show(q_open)}\n')
    sys.stdout.flush()

    if LABEL == 'bottle':
        input('>>> seat the BOTTLE in the open hand, then press Enter ')

    print(f'{"eps":>6} {"presses":>8} {"how":>10} {"pressure":>12} {"delta":>10}  q_at_end')
    results = []
    for eps in EPS_LADDER:
        written = None
        presses = ceil_presses = 0
        how = 'cap'
        last = None
        while presses < MAX_PRESSES:
            q = q_meas(c)
            if not q:
                how = 'no_state'
                break
            tgt = bounded_target(q, eps)
            if tgt is None:
                how = 'incomplete'
                break
            at_ceil = all(abs(tgt[j] - float(CLOSE_PRESETS[SIDE][j])) <= CEIL_TOL
                          for j in DEX3_JOINTS)
            if not at_ceil and written is not None and within(tgt, written):
                how = 'stopped'
                break
            if at_ceil and ceil_presses >= MAX_CEILING_PRESSES:
                how = 'ceiling'
                break
            c.set(tgt, steps=1, dt=DT)
            written = tgt
            presses += 1
            if at_ceil:
                ceil_presses += 1
            time.sleep(PRESS_SETTLE_S)
            last = c.pressure(SIDE).get(SIDE, {})

        pm = float(last.get('max') or 0.0) if last else None
        qend = q_meas(c)
        results.append((eps, presses, how, pm, dict(qend)))
        d = (pm - base_p) if pm is not None else None
        print(f'{eps:>6.2f} {presses:>8} {how:>10} {pm if pm is None else round(pm,1):>12} '
              f'{d if d is None else round(d,1):>10}  {show(qend)}')
        sys.stdout.flush()

    print('\n--- CALIBRATION CURVE (lead -> pressure rise over free air) ---')
    print(f'  free-air baseline: {base_p}')
    for eps, presses, how, pm, _ in results:
        d = (pm - base_p) if pm is not None else None
        print(f'  lead {eps:.2f} rad -> delta {d if d is None else round(d, 1)}  ({how})')

    print('\nreopening — leaving the hand OPEN')
    c.open()
    time.sleep(2.5)
    q_re = q_meas(c)
    pr = c.pressure(SIDE).get(SIDE, {})
    stuck = [j for j in DEX3_JOINTS
             if j in q_re and j in q_open and abs(q_re[j] - q_open[j]) > RETURN_TOL]
    print(f'  q_reopen: {show(q_re)}')
    print(f'  pressure: ok={pr.get("ok")} max={pr.get("max")}')
    if stuck:
        print(f'  *** JOINTS THAT DID NOT RETURN TO OPEN: {stuck}  '
              f'-> LATCH. Power cycle required. ***')
    else:
        print('  all joints returned to the open preset — no latch.')

    c.close_socket()
    print('done — hand left OPEN')
    return 0
