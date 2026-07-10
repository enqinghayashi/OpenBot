"""engine_cancel.py — the pick/place engine's mid-motion CANCEL flag.

WHY THIS FILE EXISTS (2026-10-01, operator ask)

    "i want new feature such that i can ask the robot to cancel search, task
     midway, such as stop picking up while its picking, or stop placing while
     its placing, stop sweeping while its sweeping."

WHY IT IS A SEPARATE MODULE AND NOT A GLOBAL IN THE ENGINE
----------------------------------------------------------
Three processes/threads have to agree on one bit, and two of them must never
import the third:

  * ``tabletop_pick_place_engine.py`` — the entrypoint's socket thread SETS it
    (a cancel arrives on the socket while the command loop is busy).  Its
    tests deliberately do NOT import ``pick_engine_v3``, which cannot even be
    imported off-box (module-level ``os.chdir('/home/unitree/...')``).
  * ``pick_engine_v3.py`` — the command loop POLLS it at its motion choke
    points.
  * ``writer_arm_client.py`` — the blocking slew wait polls it so an
    in-flight ``arm.slew()`` is interruptible.

A module-level singleton is the only shape that needs no plumbing through
ARGS (the engine's call graph is 7 000 lines deep and 3383 lines of it are one
inline command loop) and no import of the heavy engine by the light one.

WHY ``Cancelled`` DERIVES FROM ``BaseException``, NOT ``Exception``
-------------------------------------------------------------------
**This is load-bearing and was verified, not assumed.**  ``pick_engine_v3.py``
contains **54** ``except Exception`` handlers — best-effort cleanup around
sensor reads, logging, payload flags, encoder probing.  A cancel raised at a
motion choke point would be swallowed by the first such handler on the way
out and the arm would simply keep moving.  Deriving from ``BaseException``
makes every one of those handlers transparent to it, so it propagates
unmodified to the one place that can unwind it.

Checked at the same time: the engine and the entrypoint contain **zero** bare
``except:`` clauses, so nothing catches ``BaseException`` either.  If a bare
except is ever added to either file, this design breaks silently — hence the
pin in ``tests/test_engine_cancel.py``.

WHY THE FLAG IS GATED ON 'A TASK IS RUNNING'
---------------------------------------------
A sticky flag is a guard that refuses legitimate work: if a cancel that
arrived while the engine was idle stayed set, the NEXT pick would abort on its
first motion step.  That is the ``guard-proxy-fires-on-legitimate-input``
failure mode.  So :func:`request` REFUSES when nothing is in flight and says
so, rather than arming a trap.  The entrypoint's dispatcher owns the busy bit
(``_CommandStream._dispatch`` is the single place that knows a command has
been handed out and not yet come back).

WHAT THIS MODULE DOES **NOT** DO
--------------------------------
It does not stop the arm.  The physical stop is the writer's ``freeze`` verb
(``skills/writer/arm_guard.py:105``: halt the in-flight slew at its current
interpolated position, state HOLD_RAW), which the engine already calls on its
contact-stop and exception paths.  This module only carries the INTENT; the
engine's ``except engine_cancel.Cancelled`` arm performs the stop.  Keeping
intent and action separate is what lets the whole thing be tested without a
robot.
"""

from __future__ import annotations

import threading
import time

__all__ = [
    "Cancelled",
    "request",
    "pending",
    "where",
    "check",
    "clear",
    "mark_busy",
    "mark_idle",
    "busy",
    "snapshot",
    "reset_for_test",
]


