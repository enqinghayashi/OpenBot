"""state_reader.py — poll-only rt/lowstate reader for the C3 merged writer.

STRICT RULE (GIL lesson, pick-node-arm-recoil-gil): NO listener-mode
ChannelSubscriber callbacks in any rt/ process of ours. A DDS receive-thread
callback grabbing the GIL starves the writer loop (Stuart arm recoil bug,
2026-08-22). This module polls sub.Read(timeout) from the single writer
thread only.

Watchdog: lowstate gap > watchdog_gap_ms  => freeze warning (keep writing last
cmd); gap > watchdog_crash_ms => raise RuntimeError (main calls safe shutdown).
"""

import time

import numpy as np
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

NUM_MOTORS = 35  # LowState_/LowCmd_ length (hg IDL)


class StateReader:
    """Owns the rt/lowstate subscriber; poll() returns a state dict or None."""

