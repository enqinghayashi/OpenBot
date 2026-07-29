#!/usr/bin/env python3
"""tabletop_pick_place_engine — the tabletop pick/place engine entrypoint.

WHY THIS FILE EXISTS (2026-09-13, operator decision — structural fix)

The gateway used to SPAWN its own pick_engine_v2.py subprocess on the first
engine verb.  The pick lane separately runs its own engine.  That is two OS
processes running the same script, both commanding the same writer socket —
and the gateway-spawned one boots by driving both arms to NEUTRAL, which is
the observed "place drops the arms before it starts" defect.  A pgrep guard
made it worse: if an engine WAS already alive the spawn refused outright.

This module gives the engine a UNIX-socket listener the way
``skills/arm/grip_server.py`` already has one, so the gateway can CONNECT to
an engine that is already resident instead of spawning a second one.  A
connect never boots, so a place resumes from the carry/tuck pose the pick
left behind.

WHAT IS *NOT* HERE: the command grammar.  This file is transport + entrypoint
only.  ``pick_engine_v3.main()`` still owns the boot sequence (init move,
diagnostics, READY) and the 3383-line command loop; that body is untouched.
This file swaps the loop's two channels:

  input   ``sys.stdin``  -> a stream that also yields socket-delivered lines
  output  ``sys.stdout`` -> a router that sends each reply back to the client
                            that asked for it

Both are installed BEFORE ``pick_engine_v3.main()`` runs, so ``READY`` still
goes to the real stdout and the launcher / spawn handshake is unchanged.

ARGV is unchanged and must stay that way — the launcher passes it:
    tabletop_pick_place_engine.py [iface] [active_arm] [upper] [nomove]

TRANSPORT: unix stream socket, one line in -> one line out (the grip-server
house protocol), except that the FIRST thing a new connection receives is a
``READY`` banner.  That banner is the readiness handshake: the accept thread
starts only once the command loop is reached, i.e. once the boot sequence has
finished and READY has been printed.  A client that connects mid-boot simply
blocks on the banner instead of racing the init move.

SERIALIZATION: pick and place are minutes-long and drive one arm trajectory
at a time, so exactly ONE command is ever in flight.  Connection threads
enqueue and block on their request's completion Event; the single-threaded
command loop is the only consumer.  This is deliberately NOT grip_server's
thread-per-connection-executes model — there the commands are quick and
independent, here they are long and must not interleave.

STDIN: still served, so an operator at a terminal keeps working.  Its EOF is
deliberately NOT an exit — under the old entrypoint a closed stdin ended
main(), which would kill a resident engine the moment its pipe closed.  Only
an explicit ``quit`` verb (or SIGTERM) stops this engine.
"""

from __future__ import annotations

import os
import queue
import select
import socket
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import engine_cancel  # noqa: E402 — we need _HERE on sys.path first

#: gateway connects here
#: override: PICK_ENGINE_SOCK
#: gateway reads config.yaml ``skills.engine_sock``
SOCK_PATH = os.environ.get("PICK_ENGINE_SOCK", "/tmp/pick_engine.sock")

#: first thing a client gets
#: also the readiness handshake
READY_BANNER = b"READY\n"

#: accept backlog
#: gateway + maybe a CLI or diagnostic
#: so connect() never fails
LISTEN_BACKLOG = 8

#: verbs that CANCEL, not execute
#: checked before the queue
#:
#: why out-of-band: one loop, one command at a time
#: a pick takes minutes, so a queued cancel
#: waits behind the pick it should stop
#: (box-fsm-splice-wire-missing trap)
#: answer it in the connection thread;
#: only the flag crosses into the loop
#: thread-per-connection serves a second
#: client while the loop is stuck in a pick
#:
#: no collision: no cancel/stop/abort verb
#: (checked against all 25 dispatch verbs)
CANCEL_VERBS = frozenset({"cancel", "stop", "abort"})


def _is_cancel_line(line: str) -> bool:
    """True for a bare cancel verb, optionally with one argument ignored.

    ``cancel`` / ``stop`` / ``abort`` only.  Deliberately NOT a prefix match
    on the whole line: ``stop`` must not be able to swallow some future
    ``stop_here ...`` motion verb, and a stray argument must not turn a
    cancel into a silent no-op.
    """
    return line in CANCEL_VERBS


# ── who gets the reply ──────────────────────────────────────────────────────
# every reply: write + flush, 69 sites
# all in the loop body, line-terminated
# swap the stream, don't touch them
# router sends each to whoever asked
# _current_writer: dispatcher-only, no lock
# per-connection writers bring their own
_current_writer = None


class _ConnWriter:
    """Reply sink for ONE socket connection.

    File-like (``write`` / ``flush``) because that is what the command loop
    calls.  ``write`` never raises: a client that hung up mid-command must not
    take the engine down with it — the reply is simply dropped and the
    connection marked dead.
    """

    def __init__(self, conn: socket.socket):
        self._f = conn.makefile("wb")
        self._lock = threading.Lock()
        self._dead = False

    def write(self, data) -> int:
        if isinstance(data, str):
            data = data.encode("utf-8", "replace")
        with self._lock:
            if self._dead:
                return len(data)
            try:
                self._f.write(data)
            except OSError:
                self._dead = True
            return len(data)

    def flush(self) -> None:
        with self._lock:
            if self._dead:
                return
            try:
                self._f.flush()
            except OSError:
                self._dead = True

    def close(self) -> None:
        with self._lock:
            self._dead = True
            try:
                self._f.close()  # does NOT close the socket itself
            except OSError:
                pass


class _ReplyRouter:
    """``sys.stdout`` replacement that routes a reply to the asking client.

    Falls through to the real stdout whenever no socket command is in flight,
    which is what keeps READY, the startup diagnostics, and terminal-typed
    commands behaving exactly as before.
    """

    def __init__(self, real):
        self._real = real

    def _sink(self):
        w = _current_writer
        return self._real if w is None else w

    def write(self, data):
        return self._sink().write(data)

    def flush(self):
        return self._sink().flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        # whatever gets probed: isatty, fileno
        # we never touch fd 1, so _SuppressStdout's
        # os.dup2 redirect still works
        return getattr(self._real, name)


# ── the command stream ──────────────────────────────────────────────────────


class _Request:
    """One command line plus the writer its reply belongs on.

    ``writer is None`` means "came from stdin" -> reply goes to real stdout.
    ``done`` is set by the dispatcher once the command has fully finished
    (i.e. when the loop asks for the next one), which is what lets the owning
    thread release and report completion.
    """

    __slots__ = ("line", "writer", "done")

    def __init__(self, line: str, writer):
        self.line = line
        self.writer = writer
        self.done = threading.Event()
