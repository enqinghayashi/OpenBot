#!/usr/bin/env python3
"""
grip_client.py — GripClient (Task 8) + operator CLI.

GripClient exposes the SAME API names as the daemon's old Dex3Hand so the
pick-flow call sites are unchanged in structure:
    set(targets, steps=40, dt=0.02)
    open() / close()
    set_scalar(scalar)
    close_until_contact(get_pressure_fn, max_fraction=1.0, steps=50,
                        dt=0.02, stop_threshold=200000, min_steps=10)
        -> 'contact' | 'full' | 'no_data'
    last_q  (dict {joint_id: q} — kept for the daemon's debug prints)

get_pressure_fn is ACCEPTED AND IGNORED: pressure feedback is read
IN-SERVER (the daemon's old in-process lambda no longer exists; the server
owns the rt/dex3/*/state subscriber).

ERROR replies (including ERROR:no_dds) RAISE GripError — a failed COMMAND
must never silently fall through to the blind path.  (Blind mode for grasp
VERIFICATION stays daemon-side via RealGraspDetector; that is not touched.)

Standalone CLI (operator — post-quit grip verification + destination drop):
    grip_client.py state [side]
    grip_client.py open [right|left|both]
    grip_client.py close [right|left|both]
    grip_client.py set <side> <7f> [steps] [dt]
    grip_client.py close_adaptive <side>

The `state` JSON reply carries "available": true|false per side (true when
the DDS writer is live, false under NO_DDS) so degradation is visible at a
glance (observer DEV1).
"""

import json
import socket
import sys
import time

import numpy as np

from grip_server import (  # constants live here
    SOCK_PATH,
    DEX3_JOINTS,
    OPEN_PRESETS,
    CLOSE_PRESETS,
    STEP_DEFAULT,
    DT_DEFAULT,
    ADAPTIVE_MAX_FRACTION,
    ADAPTIVE_STEPS,
    ADAPTIVE_DT,
    ADAPTIVE_STOP,
    ADAPTIVE_MIN_STEPS,
    F_MAX_SAFE,
    clamped_close_target,
)


class GripError(Exception):
    """Raised on any ERROR: reply from the grip server (e.g. ERROR:no_dds)."""


