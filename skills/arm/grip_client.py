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

    def set_scalar(self, scalar):
        scalar = float(np.clip(scalar, -1.0, 1.0))
        t = (scalar + 1.0) / 2.0  # [-1,1] -> [0,1]
        targets = {}
        for jid in DEX3_JOINTS:
            targets[jid] = (OPEN_PRESETS[self._side][jid]
                            + t * (CLOSE_PRESETS[self._side][jid]
                                   - OPEN_PRESETS[self._side][jid]))
        return self.set(targets)

    def close_until_contact(self, get_pressure_fn=None, max_fraction=1.0,
                            steps=50, dt=0.02, stop_threshold=200000,
                            min_steps=10):
        """Adaptive close — pressure feedback read IN-SERVER.

        get_pressure_fn is accepted for API compatibility with the old
        Dex3Hand and IGNORED.  Defaults match the old daemon EXACTLY
        (max_fraction=1.0, steps=50, dt=0.02, stop_threshold=200000,
        min_steps=10) — the server also defaults to these, so callers may
        pass either explicit values or nothing.

        max_fraction is capped at F_MAX_SAFE IN-SERVER (1.0 = no-op; the
        0.600 cap was DISABLED 2026-09-19 — see grip_server.py's F_MAX_SAFE
        block); `last_q` below is set from the same clamped target so the
        client's idea of where the hand ended and the server's can never
        disagree.

        Returns 'contact' | 'full' | 'no_data'.
        Raises GripError on ERROR:no_dds (or any other ERROR reply).
        """
        # signature defaults mirror the
        # server's EXACT thresholds.
        # pass all; never guess.
        resp = self._cmd(
            f'close_adaptive {self._side} {float(max_fraction)} '
            f'{int(steps)} {float(dt)} {float(stop_threshold)} '
            f'{int(min_steps)}')
        if resp.startswith('DONE:'):
            verdict = resp[5:]
            if verdict in ('contact', 'full', 'no_data'):
                # endpoint = open + min(max_fraction,
                # F_MAX_SAFE)*(close-open). shared
                # helper, no drift from the ramp.
                self.last_q = clamped_close_target(self._side, max_fraction)[0]
                return verdict
        # ERROR:no_dds or any bad reply:
        # raise loudly, never a silent
        # blind path.
        raise GripError(resp)

    def state(self, side=None):
        cmd = 'state' if side is None else f'state {side}'
        resp = self._cmd(cmd)
        if not resp.startswith('DONE:'):
            raise GripError(resp)
        return json.loads(resp[5:])

    def pressure(self, side=None):
        """Measured finger-pad pressure, read SERVER-side (DDS truth).

        Added 2026-09-26.  This method DID NOT EXIST.  Every
        ``client.pressure(...)`` call in the engines (7 sites in
        pick_engine_v3.py alone) raised AttributeError on the missing
        attribute, was swallowed by ``except Exception``, and read a
        constant 0.0 — so the pressure-delta discriminators (the light-grip
        band, the insert-guard) could never fire, and that 0.0 was never
        evidence about the object either way.

        Payload is ``{side: {'ok': bool, 'max': float}}``: 'max' is the key
        every engine call site reads, 'ok' is the availability flag the
        insert-guard keys on.  An OLDER server replies 'ERROR:unknown
        command "pressure"', which ``_cmd`` raises and every call site
        swallows — so an un-upgraded server degrades to exactly the old 0.0
        and never to a wrong number.
        """
        cmd = 'pressure' if side is None else f'pressure {side}'
        resp = self._cmd(cmd)
        if not resp.startswith('DONE:'):
            raise GripError(resp)
        return json.loads(resp[5:])

    def hold(self, side=None):
        cmd = 'hold' if side is None else f'hold {side}'
        return self._cmd(cmd)

    def close_socket(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ── operator CLI (run it from a terminal) ────────────────────────────────────
def _cli_state(side):
    client = GripClient('right' if side in (None, 'both') else side)
    print(json.dumps(client.state(side if side != 'both' else None),
                     indent=2))
    client.close_socket()


def _cli_drive(action, side):
    if side is None:
        sides = ['right']
    elif side == 'both':
        sides = ['right', 'left']
    else:
        sides = [side]
    for s in sides:
        client = GripClient(s)
        if action == 'open':
            print(f'{s}: {client.open()}')
        elif action == 'close':
            print(f'{s}: {client.close()}')
        elif action == 'hold':
            print(f'{s}: {client.hold()}')
        client.close_socket()


def _cli_set(side, vals, steps, dt):
    client = GripClient(side)
    targets = {jid: vals[jid] for jid in DEX3_JOINTS}
    print(client.set(targets, steps=steps, dt=dt))
    client.close_socket()


def _cli_close_adaptive(side):
    client = GripClient(side)
    verdict = client.close_until_contact()
    print(f'{side}: DONE:{verdict}')
    client.close_socket()


