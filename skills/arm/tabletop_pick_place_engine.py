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
