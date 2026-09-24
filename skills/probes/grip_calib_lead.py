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


