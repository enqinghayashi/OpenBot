#!/usr/bin/env python3
"""
restore_arm_pose.py — one-shot posture restore after the phase-split quit
(Task 8, posture handover).

Reads /tmp/last_arm_q.json (written by the daemon's `quit` handler BEFORE
release_arms — {"left": [7], "right": [7], "ts": ...}) and smooth-moves
both arms back to the recorded carry/neutral posture.

Upper-only controller ONLY (arms 15-28 + gate 29; legs/waist NEVER written)
so the locomotion controller keeps full authority.  Runs in NON-801 BEFORE
R2+A — the writer exists only for the move and exits before any 801
engagement (trip-safe; exits before walking).  One-shot: moves, exits.

Guards (either failure → print REFUSED, exit 2 — no arm motion):
  - /tmp/last_arm_q.json exists, parses, and yields left[7] + right[7]
  - FSM mode == 0 (G1Loco GET 7002, same mechanism as fsm_check.py) —
    refuse to write while the FSM is in 801/damped/gait-blocked states;
    an unreadable FSM is also a refuse (cannot verify).

Usage (Stuart, operator):
    /home/unitree/miniconda3/envs/tv/bin/python3 restore_arm_pose.py [iface]
Default iface: eth0.  Launcher hook: launch_pick_walk.sh --restore-pose
(default OFF — manual step is the default, per user decision).
"""

import json
import sys
import time

import numpy as np

IFACE       = sys.argv[1] if len(sys.argv) > 1 else 'eth0'
HANDOFF_JSON = '/tmp/last_arm_q.json'

# copy the daemon's move_arm_smooth exactly: same speed/hz/cosine,
# so a restore feels like the pick flow
MOVE_SPEED = 0.3   # rad/s
MOVE_HZ    = 50    # controller rate

sys.path.insert(0, '/home/unitree/xr_teleoperate')
sys.path.insert(0, '/home/unitree/unitree_sdk2_python')
sys.path.insert(0, '/home/unitree/leon')
sys.path.insert(0, '/home/unitree/leon/openbot')


