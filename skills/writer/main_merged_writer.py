#!/usr/bin/env python3
"""main_merged_writer.py — C3 merged whole-body writer for Stuart (G1 EDU).

ONE process owns rt/lowcmd for ALL 35 slots, from boot to shutdown. No phase
split: legs+waist (0-14) from the GR00T Balance/Walk ONNX policy at 50 Hz,
arms (15-28) from arm_guard pose logic at 250 Hz, gate joint 29 + trailing
30-34 held. Walk while holding the carry-tuck. No rclpy, no listener-mode
subscribers, no sport RPC. See docs/MERGED_WRITER.md.

Bring-up: flock -> MotionSwitcher CheckMode/ReleaseMode until name empty ->
first lowstate -> capture q35 -> base frame (mode 0x01, mode_pr=0,
mode_machine echoed) -> operator GATE prompt -> 120 ms coupling ramp -> loop.

Shutdown (SIGINT/SIGTERM/quit/watchdog): rc zero -> policy 1 s -> arm release
ramp (skipped with a warning if payload flag set) -> legs hold 1 s ->
ZeroCmd -> SHUTDOWN COMPLETE -> handoff ladder printed.

v2b socket (Stage 4): thread-per-connection UNIX bridge at cfg sock_path,
one reply line per request (DONE/ERROR/Q:/STATE:). New verbs: arm <14f>
[duration_s], freeze, q, armstate, hold_rc on|off. DONE means the slew was
STARTED (the client polls armstate for completion — the guarded insert
monitors q mid-slew, so a blocking DONE would serialize the monitor out).
rc_hold clamps the RC sticks (pick interlock) exactly like _shutting_down.

Runtime height lane (2026-09-10; REPO-SIDE ONLY, NOT deployed): the `height
<m> [duration_s]` verb rate-limits the obs height reference (obs_builder f[3])
so the policy tracks a squat; height 0.74 restores standing.  It is NOT
orthogonal to cmd the way this docstring used to claim.  The pick engine sets
rc_hold during arm motion, which forces rc_live = False in the precedence
block, which gates the height suspend off — so the live RC stick is
UNREACHABLE for the whole arm motion and CANNOT abort a squat while the engine
holds the body.  A stick push during such a squat is a PAUSE, not a cancel:
the writer stashes the target and the ORIGINAL deadline and the resume restores
BOTH.  It carries a monotonic deadline watchdog (recovery = standing, not zero)
and shutdown ascends the legs before the limp frame.  See the CORRECTED block
in _tick, marker "CORRECTED 2026-09-27 (D2)".
"""

import argparse
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- SONIC lane
# gain and joint tables come from sonic_core.
# no onnxruntime is used; an import failure forces the lane off.
_SONIC_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "sonic")
if os.path.isdir(_SONIC_DIR):
    sys.path.insert(0, _SONIC_DIR)
try:
    import sonic_core as SC
except Exception as _e:            # noqa: BLE001 — degrade, never raise
    SC = None
    print(f"[sonic] sonic_core import FAILED ({_e!r}) — SONIC lane FORCED OFF",
          flush=True)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from arm_guard import ArmGuard
from obs_builder import ObsBuilder
from policy_runner import PolicyRunner
from rc_target import RCTarget
from state_reader import StateReader

PosStopF = 2146000000.0    # pos-stop sentinel
VelStopF = 16000.0         # vel-stop sentinel
MODE_ENABLE = 1            # enable mode

# stance: ankle roll per hip roll, holding the sole tilt flat.
# FK: -0.907/-0.889 crouch, -0.831/-0.829 stand, -0.799/-0.799 squat.
# A 6 deg splay holds ~0.5 deg sole error. Mid-range default.
STANCE_ANKLE_RATIO = -0.85

# ---------------------------------------------------------------- step lane
# every step constant is derived from the URDF forward kinematics,
# not from body measurements.
STEP_SEP_LEVER_M = 0.4955        # hip->ankle, 8.65 mm/deg
                                 # W = 0.4955*(sin d1 - sin d7)
STEP_COM_CM_PER_DEG = 0.6856     # same-sign, cm/deg
STEP_KNEE_LEVER_M = 0.109        # N = |tau_knee|/0.109
STEP_LIFT_DZ_CM_PER_DEG = 0.2333  # K -> foot rise, cm/deg
STEP_LIFT_HIP_PITCH = -0.472     # d_pol[0]/d_pol[6] per K
STEP_LIFT_ANKLE_PITCH = -0.528   # d_pol[4]/d_pol[10] per K
STEP_ANKLE_STOP_DEG = 15.0       # 15 deg stop, URDF
STEP_ANKLE_BASE_DEG = 1.0        # +1.0, both legs
STEP_TWO_STEP_MAX_CM = 27.4      # ceiling 27.4 cm

# honesty suffix: the writer has no body-height sensor, so every [step]
# body-state line says the value is writer-reported, not sensor-measured.
# Appended in exactly one place.
STEP_HONESTY_SUFFIX = ("writer-reported, NOT sensor-measured — "
                       "no body-height sensor exists")

CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "config", "merge_config.yaml")


def kb_safe_sleep(sec):
    """Sleep that survives SIGINT. Inside shutdown a second ^C must never
    abort the crash-safe tail (the final ZeroCmd frame + lockfile removal +
    sport handoff must always complete). With the Python SIGINT handler
    installed a terminal INT normally just runs the handler and resumes the
    sleep, but a KeyboardInterrupt escaping from any path must not kill the
    tail either."""
    try:
        time.sleep(sec)
    except KeyboardInterrupt:
        print("[shutdown] KeyboardInterrupt swallowed — shutdown continues "
              "(ZeroCmd frame + lockfile removal + handoff complete)", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="C3 merged whole-body writer")
    p.add_argument("--stage", type=int, default=1, choices=[0, 1, 2, 3, 4],
                   help="0=hold-only ownership probe (no policy); >=1 policy active")
    p.add_argument("--onnx-dir", default=os.environ.get("MERGE_ONNX_DIR"),
                   help="override ONNX dir (default: <pkg>/policy)")
    p.add_argument("--no-rc", action="store_true", help="disable rc (cmd stays zero)")
    p.add_argument("--interface", default=None, help="DDS interface (default: eth0)")
    p.add_argument("--config", default=CFG_PATH, help="config yaml path")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()
