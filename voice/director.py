"""director: the OpenBot v4 voice lane state director (L2; plan §3
director + §5b lane fallback + the boundary-only-transitions risk).

Job: decide WHEN the voice conversation moves between the cloud Realtime
lane and the offline cascade lane — and nothing else.  The director is a
pure event-driven state machine over the conversation's BOUNDARIES: a
lane signal can only change the state when the conversation sits at a
conversation boundary, defined as NO turn in flight AND NO pending tool
batch (a query_user round counts as a turn — the session's gate is
deliberately open while the human's answer is awaited, so the director
tracks query rounds itself).  Mid-turn lane signals are HELD and applied
at the next boundary, in order.

The director owns no I/O beyond the injected probe; it never calls
OfflineChain.handle_utterance / run_pipeline directly (cascade.py) — the
app activates/collapses the offline lane on OFFLINE_CASCADE entry/exit
(the chain's expose/collapse signals live app-side; the director's state
and transition log are the app's cue).  The director never sends to the
backend either (no sends, no close) — it only reads.

--------------------------------------------------------------------------
THE STATE MACHINE (single source of truth + one lock)
--------------------------------------------------------------------------

    ONLINE  -> DEGRADING          transient signal (warning / one failed
                                  connect with retries remaining)
    ONLINE  -> OFFLINE_CASCADE    hard failure (auth error, retry
                                  exhaustion — realtime.run_forever's
                                  loud "error" exit, session failover
                                  reason 'error')
    ONLINE  -> REOPENING          60-min session-cap signal (the client's
                                  clean reset reconnect — the probe
                                  re-validates the lane before ONLINE)
    DEGRADING -> OFFLINE_CASCADE  hard failure OR the degrade deadline
                                  elapsed (the transient window is
                                  bounded; deadline configurable)
    DEGRADING -> ONLINE           the lane reports healthy again
                                  ("success" — the lane proved itself)
    DEGRADING -> REOPENING        60-min session-cap signal
    OFFLINE_CASCADE -> REOPENING  reopen backoff elapsed (configurable;
                                  also entered from ONLINE/DEGRADING on
                                  the 60-min cap signal)
    REOPENING  -> ONLINE          ONLY on a successful probe — a timer
                                  NEVER produces ONLINE by itself
    REOPENING  -> OFFLINE_CASCADE a failed probe (the backoff then gates
                                  the next attempt — no probe hammering)

Everything else is a same-state no-op: no transition, no announcement,
no reset.  Each transition is recorded in ``transitions`` as
(state, event, reason, ts) plus the from-state.

--------------------------------------------------------------------------
BOUNDARY DEFINITION + HOLD POLICY
--------------------------------------------------------------------------

Boundary = NOT turn_active AND the boundary_source.  ``turn_active`` is
maintained by the director from the turn-bookkeeping events the app
feeds it (gate_changed open:False, tool_call, query_waiting open a turn;
turn_complete and answer close it, and gate_changed open:True reopens
the boundary — the session's own reset/failover path emits it; query
rounds count as a turn, and their gate is deliberately open).
``boundary_source`` is injected (default: ``backend.gate_open`` when the
backend has it, else True — the session's half-duplex gate already means
"no response in flight AND no pending tool results", L1).

HOLD POLICY (documented): a lane signal (warning/connect_failed/hard/
success/session_expired/tick) that arrives while the boundary is CLOSED
is queued in order and applied at the next boundary — after the
turn_complete, the answer, or a gate_changed open:True (the session's
own reset/failover path emits it) that reopens the boundary, the queue
drains one at a time, each signal through the same boundary-gated step.
Ticks mid-turn are held like any signal: timers are evaluated lazily,
never on their own.  The held queue is capped (HELD_QUEUE_CAP): beyond
it the idempotent TICK entries are the drop candidates — a real signal
evicts the first held tick (signals never lose their slot to a tick),
and a tick to a full queue evicts an older tick (or is dropped itself).
Drain may fire several transitions back-to-back (each announcing once,
in order).

--------------------------------------------------------------------------
TIMERS (all lazy — evaluated on event processing, never background)
--------------------------------------------------------------------------

* DEGRADING: a deadline (``degrade_deadline_s``, default 30 s) starts on
  entry; once the clock passes it, the next event/tick transitions to
  OFFLINE_CASCADE.
* OFFLINE_CASCADE: a backoff (``reopen_backoff_s``, default 5 s) starts
  on entry; once elapsed, the next event/tick transitions to REOPENING.
* REOPENING: probes are spaced by ``probe_interval_s`` (default 3 s).
  On entering REOPENING the next step may probe immediately; a probe
  runs inside the step, and its result is the ONLY path into ONLINE.

The clock is injectable (``cfg['clock']``, default ``time.monotonic``) —
tests drive a fake clock; a tick with no event ever transitions
anything, so wall time alone can never move the machine, and no timer
ever yields ONLINE (only a successful probe does).

--------------------------------------------------------------------------
PROBE (owned by the director)
--------------------------------------------------------------------------

``probe()`` is an injected lightweight connectivity check (e.g. a short-
timeout cloud backend probe call) returning truthy when the lane is
reachable.  It runs under the transition lock — keep it FAST (the
contract: a probe is a lightweight check, not a health dump); a raising
probe counts as a failure (logged).  REOPENING -> ONLINE happens ONLY on
probe success; a failed probe returns to OFFLINE_CASCADE, whose backoff
governs the next attempt (no hammering).

--------------------------------------------------------------------------
RESET ON EVERY TRANSITION (the Backend contract adaptation)
--------------------------------------------------------------------------

Every committed transition calls the reset path EXACTLY once.  The
injected ``reset`` wins when given; the default adapts to the existing
session contract — there is NO public reset method on VoiceSession
today (session.py exposes only the private ``_reset_turn_state``, called
on its own error/session-expired/close paths), so the default:

  1. prefers a public ``reset`` / ``reset_session`` / ``reset_context``
     / ``session_reset`` / ``clear`` on the backend when one exists;
  2. else calls the nearest available equivalent, the session's
     ``_reset_turn_state`` (documented gap: private, but it IS the exact
     turn-local reset the session itself uses on failover);
  3. else logs a warning that no reset path exists and proceeds — the
     director never fabricates a reset.

What the reset clears (session.py ``_reset_turn_state``): in-flight
response flags, pending tool call ids, turn call ids, the query_user
intercept state, the playback lead buffer and queued-audio accounting.
The L3 dedup map (``_emitted_outputs``) is deliberately KEPT — old
call_ids cannot re-arrive.  Transitions only ever fire at a boundary
(no turn in flight), so the reset cannot kill a live turn.

--------------------------------------------------------------------------
ANNOUNCEMENTS
--------------------------------------------------------------------------

``speak(text)`` is injected (the app's speaker).  ONE announcement per
transition — the director's own short strings (ANNOUNCEMENTS), spoken
directly through ``speak``, NEVER through the brain being reset.  The
initial state announces nothing (the app speaks "online" itself at
startup if it wants).

--------------------------------------------------------------------------
EVENT VOCABULARY (the app feeds on_event(name, payload))
--------------------------------------------------------------------------

Lane signals (may transition, held mid-turn):
  warning        — transient signal          (payload {"detail": ...})
  connect_failed — one connect attempt failed (payload
                   {"retries_remaining": int})  — "warn" aliases warning
  hard           — auth error / retry exhaustion; "error" and session
                   failover reason 'error' alias it (payload detail)
  success        — the lane reported healthy again
  session_expired— 60-min cap signal; failover reason 'session_expired'
                   and wire "session.expired" alias it
  tick           — evaluate the lazy timers (app calls it on idle)

Turn bookkeeping (never transition; maintain the boundary):
  gate_changed {"open": false/true}, tool_call, query_waiting,
  answer, turn_complete — the session's own event names, passed through
  verbatim.  Everything else is ignored (logged at debug).

--------------------------------------------------------------------------
CONFIG (cfg: dict or BrainConfig-shaped object; missing keys fall back)
--------------------------------------------------------------------------

  lane              — 'realtime'|'transcript'|'offline' (BrainConfig.lane
                      semantics).  The offline lane starts at degrade:
                      lane 'offline' -> initial state OFFLINE_CASCADE;
                      anything else -> ONLINE.
  degrade_deadline_s / reopen_backoff_s / probe_interval_s — timers.
  clock             — callable () -> float seconds; default time.monotonic.

--------------------------------------------------------------------------
CONCURRENCY
--------------------------------------------------------------------------

Single source of truth ``_state`` plus one ``threading.Lock`` (also
public as ``lock``).  All transitions commit under the lock; the
injected reset() and speak() run AFTER the lock is released (they may
call back into the app).  Reads of ``state`` take the lock.  The probe
runs under the lock (lightweight contract, above).  The app's event
loop thread is the only caller in the realtime lane (session handlers
are sync); the lock also makes the director safe from the mic pump
thread.

Module-level import surface is stdlib-only (logging/threading/time/
typing) — importing this module succeeds dep-free on any laptop,
mirroring the voice/ package pattern.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger("voice.director")

# ------------------------------------------------------------- the four states

ONLINE = "ONLINE"
DEGRADING = "DEGRADING"
OFFLINE_CASCADE = "OFFLINE_CASCADE"
REOPENING = "REOPENING"
STATES = (ONLINE, DEGRADING, OFFLINE_CASCADE, REOPENING)

#: our own words,
#: spoken via speak - never
#: through the brain we reset.
ANNOUNCEMENTS: Dict[str, str] = {
    DEGRADING: "switching to degraded",
    OFFLINE_CASCADE: "switching to offline",
    REOPENING: "reconnecting",
    ONLINE: "restored",
}

# ------------------------------------------------------------- the defaults

#: DEGRADING window.
#: after this, collapse to OFFLINE.
#: cfg sets it.
DEGRADE_DEADLINE_S = 30.0
#: reopen backoff.
#: a failed probe restarts it.
#: cfg sets it.
REOPEN_BACKOFF_S = 5.0
#: probe spacing.
#: while REOPENING; cfg sets it.
PROBE_INTERVAL_S = 3.0
#: held-queue cap.
#: past it we drop TICKs first -
#: a real signal kicks out a tick;
#: only a tick-less queue loses its
#: oldest real signal.
HELD_QUEUE_CAP = 16

#: lane signals.
#: held mid-turn.
SIGNAL_EVENTS = frozenset((
    "warning", "connect_failed", "hard", "success", "session_expired",
    "tick",
))
#: boundary only.
TURN_EVENTS = frozenset((
    "gate_changed", "tool_call", "query_waiting", "answer", "turn_complete",
))


