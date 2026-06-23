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


class Director:
    """The voice lane state director (module docstring: states, boundary
    definition, hold policy, timers, probe, reset, announcements).

    API: ``Director(backend, probe, speak, reset=None, boundary_source=None,
    cfg=None)``; ``on_event(name, payload=None)`` returns the resulting
    state; ``state`` (locked read) and ``lock`` are public; the static
    ``transition()`` is the pure per-event transition table, reachable
    for tests.  Feed the app's session events + lane signals verbatim
    (see the module docstring's event vocabulary).
    """

    def __init__(
        self,
        backend: Any,
        probe: Callable[[], bool],
        speak: Callable[[str], Any],
        reset: Optional[Callable[[], Any]] = None,
        boundary_source: Optional[Callable[[], bool]] = None,
        cfg: Optional[Any] = None,
    ) -> None:
        if probe is None or not callable(probe):
            raise ValueError("Director requires a probe callable () -> bool")
        if speak is None or not callable(speak):
            raise ValueError("Director requires a speak callable (text) -> Any")
        self.backend = backend
        self._probe = probe
        self._speak = speak
        self._reset: Callable[[], Any] = (
            reset if reset is not None else self._default_reset
        )
        self._cfg = cfg if cfg is not None else {}
        self._clock = self._cfg_get("clock", time.monotonic)
        self._degrade_deadline_s = float(
            self._cfg_get("degrade_deadline_s", DEGRADE_DEADLINE_S)
        )
        self._reopen_backoff_s = float(self._cfg_get("reopen_backoff_s", REOPEN_BACKOFF_S))
        self._probe_interval_s = float(self._cfg_get("probe_interval_s", PROBE_INTERVAL_S))

        #: boundary source.
        #: default: the session's half-duplex
        #: gate (nothing in flight, no pending
        #: tools), else True - we trust the
        #: event flags then.
        if boundary_source is None:
            boundary_source = lambda: bool(getattr(self.backend, "gate_open", True))
        self._boundary_source = boundary_source

        #: one state.
        #: one lock.
        self._lock = threading.Lock()
        self.lock = self._lock
        #: offline -> OFFLINE_CASCADE;
        #: else ONLINE.
        lane = str(self._cfg_get("lane", "realtime") or "realtime")
        self._state: str = OFFLINE_CASCADE if lane == "offline" else ONLINE

        #: transition log.
        #: oldest first; from/state/event/
        #: reason/ts + detail
        self.transitions: List[Dict[str, Any]] = []
        self._held: List[Tuple[str, Dict[str, Any]]] = []
        self._pending_notify: List[Tuple[str, str]] = []  # (from, to) pairs

        # -- keeping track of the boundary --
        self._turn_active = False
        self._query_open = False

        # -- the lazy timers --
        self._degrade_deadline: Optional[float] = None
        self._offline_entered: Optional[float] = None
        self._reopen_entered: Optional[float] = None
        self._last_probe_at: Optional[float] = None

    # ------------------------------------------------------------ public bits

    @property
    def state(self) -> str:
        """The current state (locked read)."""
        with self._lock:
            return self._state

    @property
    def held(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Copy of the currently held mid-turn signals (diagnostics)."""
        with self._lock:
            return list(self._held)

    def on_event(self, name: str, payload: Optional[Dict[str, Any]] = None) -> str:
        """Feed one event; returns the resulting state.

        Lane signals apply only at a conversation boundary (held
        otherwise); turn-bookkeeping events update the boundary; the
        held queue drains whenever the boundary is open.  Injected
        reset()/speak() run once per committed transition, after the
        lock is released.
        """
        name = str(name)
        payload = dict(payload or {})
        with self._lock:
            event = self._normalize(name, payload)
            if event is not None:
                if event in TURN_EVENTS:
                    self._apply_turn_event_locked(event, payload)
                elif self._at_boundary_locked():
                    self._step_locked(event, payload)
                else:
                    self._hold_locked(event, payload)
                self._drain_locked()
        self._notify_outside_lock()
        return self.state

    # -------------------------------------------------- the transition table

    @staticmethod
    def transition(
        state: str,
        event: str,
        *,
        probe_ok: Optional[bool] = None,
        deadline_passed: bool = False,
        backoff_elapsed: bool = False,
        retries_remaining: Optional[int] = None,
    ) -> Optional[str]:
        """The PURE per-event transition table (no I/O, no mutation) —
        reachable for tests.

        Returns the new state or None (same-state no-op).  ``probe_ok``
        is the ONLY path into ONLINE; ``deadline_passed`` /
        ``backoff_elapsed`` are the timer facts for the lazy timers;
        ``retries_remaining`` disambiguates connect_failed.
        """
        if state not in STATES:
            return None
        if event == "probe":
            # probe or nothing.
            if state == REOPENING:
                return ONLINE if probe_ok else OFFLINE_CASCADE
            return None
        if event == "tick":
            if state == DEGRADING and deadline_passed:
                return OFFLINE_CASCADE
            if state == OFFLINE_CASCADE and backoff_elapsed:
                return REOPENING
            return None  # timer never ONLINE
        if event == "warning":
            return DEGRADING if state == ONLINE else None
        if event == "connect_failed":
            retries = 0 if retries_remaining is None else int(retries_remaining)
            if state == ONLINE:
                # retries left = transient.
                # none left = hard.
                return DEGRADING if retries > 0 else OFFLINE_CASCADE
            if state == DEGRADING:
                return OFFLINE_CASCADE if retries <= 0 else None
            return None
        if event == "hard":
            return OFFLINE_CASCADE if state in (ONLINE, DEGRADING) else None
        if event == "session_expired":
            # 60-min cap: client reset itself.
            # we probe the lane first.
            # only then ONLINE.
            return REOPENING if state in (ONLINE, DEGRADING) else None
        if event == "success":
            return ONLINE if state == DEGRADING else None
        return None

    # -------------------------------------------------------- the internals

    def _cfg_get(self, key: str, default: Any) -> Any:
        cfg = self._cfg
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _normalize(self, name: str, payload: Dict[str, Any]) -> Optional[str]:
        """Map the app's raw event names onto the canonical vocabulary."""
        if name in ("warn",):
            return "warning"
        if name == "failover":
            reason = str(payload.get("reason") or "error")
            # failover reason:
            # 'session_expired' = 60-min cap,
            # anything else = hard.
            return "session_expired" if reason == "session_expired" else "hard"
        if name in ("error",):
            return "hard"
        if name in ("session.expired",):
            return "session_expired"
        if name in SIGNAL_EVENTS or name in TURN_EVENTS:
            return name
        return None  # not ours.

    def _at_boundary_locked(self) -> bool:
        """Boundary = no turn in flight (and no pending tool batch / open
        query round) AND the injected boundary source."""
        if self._turn_active:
            return False
        source = self._boundary_source
        try:
            return bool(source())
        except Exception as exc:  # noqa: BLE001 - a broken source must not
            log.warning("director boundary_source raised: %s", exc)
            return True  # just fall back to the event-derived flags

    def _apply_turn_event_locked(self, event: str, payload: Dict[str, Any]) -> None:
        """Turn bookkeeping: maintain ``_turn_active`` (a query_user round
        counts as a turn — the gate is open while the answer is awaited)."""
        if event == "gate_changed":
            if payload.get("open") is False:
                self._turn_active = True
            else:
                # gate_changed open:True reopens
                # the boundary - held queue drains.
                # a query round still counts as a
                # turn: its gate is knowingly open.
                if not self._query_open:
                    self._turn_active = False
        elif event == "tool_call":
            self._turn_active = True
        elif event == "query_waiting":
            self._query_open = True
            self._turn_active = True
        elif event == "answer":
            # answer closes the round.
            # fresh turn on the next
            # gate_changed.
            self._query_open = False
            self._turn_active = False
        elif event == "turn_complete":
            self._query_open = False
            self._turn_active = False

    def _hold_locked(self, event: str, payload: Dict[str, Any]) -> None:
        """HOLD POLICY: a lane signal mid-turn is queued in order; beyond
        HELD_QUEUE_CAP the idempotent TICK entries are the drop
        candidates — a real signal evicts the FIRST held tick (in place,
        so real signals keep their order and never lose a slot to a
        tick), a tick to a full queue evicts an older tick (or is dropped
        itself when the queue holds only real signals), and only a queue
        with no ticks left evicts its OLDEST entry (the most recent real
        signals are the relevant ones)."""
        if len(self._held) >= HELD_QUEUE_CAP:
            if event == "tick":
                # swap in the newer tick - its
                # timer check runs during the drain.
                # no older tick? a real-signal queue
                # drops the incoming one instead.
                for i, (ev, _) in enumerate(self._held):
                    if ev == "tick":
                        dropped = self._held.pop(i)
                        log.warning(
                            "director held queue full: dropped tick entry "
                            "%r (idempotent — no real signal evicted)",
                            dropped[0],
                        )
                        self._held.append((event, dict(payload)))
                        return
                log.info(
                    "director held queue full of real signals: incoming "
                    "idempotent tick dropped (lazy timers evaluate during "
                    "the drain anyway)"
                )
                return
            # real signal: replace the first
            # held tick in place, order kept.
            for i, (ev, _) in enumerate(self._held):
                if ev == "tick":
                    self._held[i] = (event, dict(payload))
                    log.info(
                        "director held queue full: replaced idempotent "
                        "tick %r with %r (signals keep their order)",
                        ev, event,
                    )
                    return
            dropped = self._held.pop(0)
            log.warning(
                "director held queue full: dropped oldest held signal %r "
                "(no ticks left to evict — most recent signals win)",
                dropped[0],
            )
            self._held.append((event, dict(payload)))
            return
        self._held.append((event, dict(payload)))
        log.info(
            "director holds %r until the next conversation boundary "
            "(turn in flight / pending tool batch / query round)",
            event,
        )

    def _drain_locked(self) -> None:
        """Apply held signals, in order, whenever the boundary is open."""
        while self._held and self._at_boundary_locked():
            event, payload = self._held.pop(0)
            self._step_locked(event, payload)

    def _step_locked(self, event: str, payload: Dict[str, Any]) -> None:
        """One step at a boundary: lazy timers first (elapsed reality),
        then the event itself; at most one transition per step."""
        now = float(self._clock())
        nxt: Optional[str] = None
        reason = ""
        if (
            self._state == DEGRADING
            and self._degrade_deadline is not None
            and now >= self._degrade_deadline
        ):
            nxt = self.transition(self._state, "tick", deadline_passed=True)
            reason = "degrade deadline elapsed ({:.1f}s) — the transient window closed".format(
                self._degrade_deadline_s
            )
        elif (
            self._state == OFFLINE_CASCADE
            and self._offline_entered is not None
            and now >= self._offline_entered + self._reopen_backoff_s
        ):
            nxt = self.transition(self._state, "tick", backoff_elapsed=True)
            reason = "reopen backoff elapsed ({:.1f}s)".format(self._reopen_backoff_s)
        elif (
            self._state == REOPENING
            and self._last_probe_at is not None
            and now >= self._last_probe_at + self._probe_interval_s
        ):
            self._last_probe_at = now
            ok = self._run_probe_locked()
            nxt = self.transition(self._state, "probe", probe_ok=ok)
            reason = "probe {} — ONLINE only ever follows probe success".format(
                "succeeded" if ok else "failed"
            )
        if nxt is None:
            nxt, reason = self._event_step_locked(event, payload)
        if nxt is not None and nxt != self._state:
            self._commit_locked(nxt, event, reason or "lane signal", payload)

