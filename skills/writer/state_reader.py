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

    def __init__(self, cfg):
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init()
        self._timeout = cfg["lowstate_timeout_ms"] / 1000.0  # SDK wants seconds
        self._gap_freeze_ms = float(cfg["watchdog_gap_ms"])
        self._gap_crash_ms = float(cfg["watchdog_crash_ms"])
        self._last_t = None
        self._frozen = False
        self.poll_count = 0
        self.timeout_count = 0
        # set first in shutdown(): no crash raise past here
        # (a raise killed it mid-shutdown: uncontrolled motors,
        # stale lockfile). poll() only warns
        self.suppress_crash = False

    def wait_first(self, timeout_s):
        """Block until the first lowstate arrives (bring-up). Raises on timeout."""
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            msg = self._sub.Read()  # blocking read ok: only consumer
            if msg is not None:
                self._last_t = time.perf_counter()
                return self._unpack(msg)
        raise RuntimeError(f"no lowstate within {timeout_s} s")

    def poll(self):
        """One non-blocking poll. Returns state dict, or None on poll timeout.

        Raises RuntimeError if the inter-message gap exceeds the crash threshold.
        """
        msg = self._sub.Read(self._timeout)
        self.poll_count += 1
        now = time.perf_counter()
        if msg is None:
            self.timeout_count += 1
            if self._last_t is not None:
                gap_ms = (now - self._last_t) * 1000.0
                if gap_ms > self._gap_crash_ms and not self.suppress_crash:
                    raise RuntimeError(
                        f"LOWSTATE GAP {gap_ms:.0f} ms > crash {self._gap_crash_ms:.0f} ms")
                if gap_ms > self._gap_freeze_ms and not self._frozen:
                    self._frozen = True
                    print(f"[watchdog] lowstate gap {gap_ms:.0f} ms > "
                          f"{self._gap_freeze_ms:.0f} ms (freeze, holding last cmd)")
            return None
        self._last_t = now
        self._frozen = False
        return self._unpack(msg)

    def try_poll(self):
        """poll() that never raises — for shutdown loops.

        A crash-threshold gap must not abort shutdown mid-flight (C2 incident
        2026-08-25: the unguarded poll() in shutdown() re-raised LOWSTATE GAP,
        so the process died before ZeroCmd / lockfile removal / SelectMode('ai'),
        leaving the motors uncontrolled and a stale lockfile). Logs and returns
        None instead; the caller decides the safety action.
        """
        try:
            return self.poll()
        except RuntimeError as e:
            print(f"[reader] try_poll swallowed: {e}")
            return None

