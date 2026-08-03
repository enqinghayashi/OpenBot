#!/usr/bin/env python3
"""
manual_arm_joints.py — manual arm-joint manipulation tool for Stuart.

Run in a clean shell with the tv conda env (NO ROS2 env), like the pick daemon:
    /home/unitree/miniconda3/envs/tv/bin/python3 manual_arm_joints.py eth0 [--upper-only]
        ... then interactive REPL, OR one of:
    --right <pitch roll yaw elbow wr wp wy>     move RIGHT arm to these 7 joints
    --left  <pitch roll yaw elbow wr wp wy>     move LEFT arm to these 7 joints
    --both  <14 values: left 7 then right 7>    move both arms
    --pose <name>                               load a pose from manual_poses.json
    --state                                     print current joints and exit

Dual-arm q layout (14-DoF, same as the pick daemon):
    q[0:7]  = LEFT  arm   [shoulder_pitch, shoulder_roll, shoulder_yaw,
                           elbow, wrist_roll, wrist_pitch, wrist_yaw]
    q[7:14] = RIGHT arm   (same order)

REPL commands:
    right <7 vals> | left <7 vals> | both <14 vals> | state
    ready | lift | carry_tuck | carry_tuck_right | carry_tuck_left |
    neutral | both | home | quit

Moves are smooth cosine interpolations (0.3 rad/s, min 1.5 s) — never instant.
Values beyond +/-3.0 rad trigger a warning but still execute.

TASK 7 GUARD (2026-08-16): refuses to start while the pick daemon stack is
alive (pick_node / pick_daemon processes).  Running this tool while the
upper-only daemon holds the arm channels puts a SECOND arm_sdk writer on the
bus — the 2026-08-16 incident flipped fsm_mode 0 -> 3 (gait blocked).  Stop
the stack first; the guard applies with AND without --upper-only.
"""
import os
import sys
import json
import time
import numpy as np

os.chdir('/home/unitree/xr_teleoperate/teleop/robot_control')
sys.path.insert(0, '/home/unitree/xr_teleoperate')
sys.path.insert(0, '/home/unitree/unitree_sdk2_python')
sys.path.insert(0, '/home/unitree/leon')

from teleop.robot_control.robot_arm import G1_29_ArmController
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

MOVE_SPEED = 0.3          # rad/s, same as the pick daemon
MOVE_HZ = 50
MAX_WARN_RAD = 3.0

JOINT_NAMES = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw',
               'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw']

# preset constants, identical to the pick daemon's
# per-arm: pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
# READY elbow -1.0 fits URDF limit [-1.0472, +2.0944];
# -1.2 overshoots by 0.153 rad; -1.0 stays inside the URDF limit
READY_LEFT = np.array([0.40, 0.30, 0.50, -1.00, 0.00, 0.00, -0.01])
READY_RIGHT = np.array([0.40, -0.30, -0.50, -1.00, 0.00, 0.00, 0.01])
LIFT_LEFT = np.array([0.00, 0.20, 0.30, -1.00, 0.00, -0.60, 0.20])
LIFT_RIGHT = np.array([0.00, -0.20, -0.30, -1.00, 0.00, -0.60, -0.20])
CARRY_TUCK_LEFT = np.array([0.50, 0.05, 0.00, 0.00, -1.50, 0.00, -1.00])
CARRY_TUCK_RIGHT = np.array([0.50, -0.05, 0.00, 0.00, 1.50, 0.00, 1.00])
# NEUTRAL: lowstate-sourced hanging rest.  SIGN RISK:
# yaw/elbow signs may be flipped
# vs the arm_sdk COMMAND convention
# HARDWARE-VERIFY on Stuart first
# flip only acting-up joints: yaw, then roll.  elbow is out:
# -1.0 is in-limits and user-approved
NEUTRAL_LEFT = np.array([0.25522, 0.27104, -0.08654, 0.83254,
                         -0.00101, 0.01032, -0.00063])
NEUTRAL_RIGHT = np.array([0.26688, -0.28410, 0.09247, 0.80183,
                          -0.00034, 0.00915, 0.00730])

# the LEFT arm
# twisted toward the chest on same-sign yaw pairs ->
# LEFT yaw flipped POSITIVE (+0.5 ready / +0.3 lift)
# right arm fine, unchanged

PRESETS = {
    'ready': (READY_LEFT, READY_RIGHT),
    'lift': (LIFT_LEFT, LIFT_RIGHT),
    'carry_tuck': (CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT),
    'carry_tuck_right': (None, CARRY_TUCK_RIGHT),
    'carry_tuck_left': (CARRY_TUCK_LEFT, None),
    'neutral': (NEUTRAL_LEFT, NEUTRAL_RIGHT),
    # 'both': both arms in carry_tuck walking posture
    # (carry_tuck_* single-arm = per-arm verification)
    'both': (CARRY_TUCK_LEFT, CARRY_TUCK_RIGHT),
}

POSES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'manual_poses.json')


def move_arm_smooth(arm, target_q):
    """Cosine-interpolated 14-DoF move (same math as the pick daemon)."""
    for i, v in enumerate(target_q):
        if abs(v) > MAX_WARN_RAD:
            print(f'WARNING: q[{i}]={v:+.3f} exceeds +/-{MAX_WARN_RAD} rad — '
                  'executing anyway')
    start_q = arm.get_current_dual_arm_q().copy()
    max_delta = float(np.max(np.abs(target_q - start_q)))
    duration = max(1.5, max_delta / MOVE_SPEED)
    steps = max(1, int(duration * MOVE_HZ))
    dt = duration / steps
    print(f'moving: max_delta={max_delta:.3f} rad, '
          f'duration={duration:.1f}s, steps={steps}')
    for i in range(1, steps + 1):
        r = 0.5 - 0.5 * np.cos(np.pi * i / steps)
        arm.ctrl_dual_arm(start_q + r * (target_q - start_q), np.zeros(14))
        time.sleep(dt)
    arm.ctrl_dual_arm(target_q, np.zeros(14))


def print_state(arm, label='CURRENT'):
    q = arm.get_current_dual_arm_q()
    print(f'{label} 14-DoF q:')
    for side, sl in (('LEFT ', slice(0, 7)), ('RIGHT', slice(7, 14))):
        parts = ', '.join(f'{n}={q[sl][i]:+.4f}'
                          for i, n in enumerate(JOINT_NAMES))
        print(f'  {side}: {parts}')
    return q


def target_from_parts(arm, left_vals=None, right_vals=None):
    q = arm.get_current_dual_arm_q().copy()
    if left_vals is not None:
        q[0:7] = np.array(left_vals, dtype=np.float64)
    if right_vals is not None:
        q[7:14] = np.array(right_vals, dtype=np.float64)
    return q


def do_home(arm):
    print('moving arms home (ctrl_dual_arm_go_home)...')
    arm.arm_velocity_limit = 2.0
    arm.ctrl_dual_arm_go_home()
    time.sleep(2.0)
    arm.arm_velocity_limit = 1.5


def _proc_comm(pid):
    """Process name from /proc — empty string if the pid is gone."""
    try:
        with open('/proc/%s/comm' % pid, 'r') as f:
            return f.read().strip()
    except Exception:
        return ''


def check_pick_stack_alive():
    """Task 7: is the pick daemon stack running?  (pre-controller guard)

    2026-08-16 incident: this tool + the upper-only pick daemon = TWO
    arm_sdk writers; the STOCK 35-joint controller also wrote legs/waist
    kp=300, the arms jittered, and the balance FSM dropped fsm_mode to 3
    (gait blocked).  Refusal is ABSOLUTE (with --upper-only too): the
    upper-only controller still writes the same arm channels the daemon
    owns.

    pgrep -f 'pick_node' / 'pick_daemon' covers every deployed name
    (pick_node.py, pick_daemon.py, pick_daemon_v2.py, ...); matches are
    only trusted when /proc/<pid>/comm is a python interpreter, so a
    wrapper shell whose cmdline merely CONTAINS the pattern string cannot
    false-trigger the refusal.

    Returns (alive: bool, matched: str or None).
    """
    import subprocess
    for pat in ('pick_node', 'pick_daemon'):
        try:
            out = subprocess.run(['pgrep', '-f', pat], capture_output=True,
                                 text=True, timeout=5)
            pids = [p for p in out.stdout.split() if p]
        except Exception:
            pids = []
        pids = [p for p in pids if _proc_comm(p).startswith('python')]
        if pids:
            return True, pat
    return False, None


def load_pose(name):
    if not os.path.exists(POSES_FILE):
        print(f'ERROR: {POSES_FILE} not found')
        return None
    with open(POSES_FILE, 'r') as f:
        poses = json.load(f)
    pose = poses.get(name)
    if pose is None:
        print(f'ERROR: pose "{name}" not in {POSES_FILE}. '
              f'Available: {sorted(poses.keys())}')
        return None
    return pose
