"""rc_target.py — rc command from lowstate.wireless_remote.

Uses the SDK Joystick class verbatim (unitree_sdk2py/utils/joystick.py,
wireless_remote[2:4] buttons, [4:8] lx, [8:12] rx, [12:16] ry, [20:24] ly).
Mapping + scale live in config (rc_mapping/rc_scale_*); signs calibrated on
gantry Stage 2.

Safety: inactivity > rc_inactivity_timeout s => cmd forced to zero
(guaranteed stop — mirrors the RC-inactive rule from the working pipeline).
"""

import time

import numpy as np
from unitree_sdk2py.utils.joystick import Joystick

STICK_NAMES = ("lx", "ly", "rx", "ry")


