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


class GripClient:
    """Socket client for the persistent grip server (one hand)."""

    def __init__(self, side, connect_retries=10, retry_delay=0.5):
        if side not in ('right', 'left'):
            raise ValueError(f'GripClient side must be right|left, got {side!r}')
        self._side = side
        self._sock = None
        # copy of old Dex3Hand.msg.motor_cmd[id].q:
        # daemon debug prints read it after close.
        self.last_q = {jid: 0.0 for jid in DEX3_JOINTS}
        self._connect(connect_retries, retry_delay)

    def _connect(self, retries, retry_delay):
        for attempt in range(1, retries + 1):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(30.0)
                s.connect(SOCK_PATH)
                self._sock = s
                return
            except (FileNotFoundError, ConnectionRefusedError, OSError) as e:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
                if attempt == retries:
                    raise GripError(
                        f'cannot connect to grip server {SOCK_PATH} '
                        f'after {retries} tries: {e} (is grip_server.py '
                        f'running?)')
                time.sleep(retry_delay)

    def _cmd(self, line, timeout=30.0):
        if self._sock is None:
            raise GripError('grip server connection closed')
        self._sock.settimeout(timeout)
        self._sock.sendall((line + '\n').encode())
        raw = self._sock.recv(4096)
        if not raw:
            raise GripError(f'grip server closed the connection on "{line}"')
        resp = raw.decode().strip()
        if resp.startswith('ERROR:'):
            raise GripError(resp)
        return resp

    # ── commands that move the hand (API-compatible with old Dex3Hand) ─────
    def set(self, targets, steps=STEP_DEFAULT, dt=DT_DEFAULT):
        vals = ' '.join(f'{float(targets.get(jid, 0.0)):.6f}'
                        for jid in DEX3_JOINTS)
        resp = self._cmd(f'set {self._side} {vals} {int(steps)} {float(dt)}')
        self.last_q = {jid: float(targets.get(jid, self.last_q.get(jid, 0.0)))
                       for jid in DEX3_JOINTS}
        return resp

    def open(self):
        name = 'open' if self._side == 'right' else 'open_left'
        resp = self._cmd(name)
        self.last_q = dict(OPEN_PRESETS[self._side])
        return resp

    def close(self):
        name = 'close' if self._side == 'right' else 'close_left'
        resp = self._cmd(name)
        self.last_q = dict(CLOSE_PRESETS[self._side])
        return resp

