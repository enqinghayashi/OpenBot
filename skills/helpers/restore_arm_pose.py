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


def refuse(msg):
    print(f'restore_arm_pose: REFUSED: {msg}')
    sys.exit(2)


def check_fsm():
    """Guard: restore runs ONLY when fsm_id != 801 AND fsm_mode == 0,
    else refuse (and refuse on any read failure — cannot verify).

    HARDWARE-CORRECTED 2026-08-17: the fsm_mode==0-only guard was
    INSUFFICIENT — the robot can be IN 801 (walking-capable FSM state)
    with mode 0, and an arm_sdk writer in 801 tripped the RC sticks
    (restore ran with fsm_id=801/mode=0 → sticks died, recovery ladder
    needed).  fsm_id is the actual NON-801 gate: check GET 7001 AND 7002.
    """
    from g1_loco import G1Loco
    try:
        loco = G1Loco()
        loco.connect()
        code_id, fsm_id = loco.get_fsm_id()
        code_mode, fsm_mode = loco.get_fsm_mode()
    except Exception as e:
        refuse(f'fsm unreadable ({e}) — cannot verify NON-801')
    if code_id != 0:
        refuse(f'fsm_id read error (code={code_id}) — cannot verify NON-801')
    if code_mode != 0:
        refuse(f'fsm_mode read error (code={code_mode}) — cannot verify '
               f'NON-801')
    if fsm_id == 801:
        refuse(f'fsm_id=801 — restore must run OUTSIDE the loco FSM 801 '
               f'(mode={fsm_mode} is irrelevant; writer+801 is the trip '
               f'condition). Verify NON-801 (e.g. fsm_check.py exit 0 '
               f'AND fsm_id != 801) before re-running')
    if fsm_mode != 0:
        refuse(f'fsm_mode={fsm_mode} — restore must run in NON-801 '
               f'(fsm_mode=0) BEFORE R2+A')
    print(f'restore_arm_pose: fsm_id={fsm_id} (not 801) + fsm_mode=0 OK '
          f'(NON-801, safe to write)')


