"""session: the OpenBot v4 turn-management layer (L2, plan §3 + §4 + §5b).

Sits BETWEEN the lane wire client and the app/director.  The lane client
(voice/realtime.RealtimeClient today; the transcript/offline impls come
later) is a dumb pipe; the app is the director (tools execution, sinks).
This module is the LAW-ENFORCEMENT layer: the exact turn laws the plan
pins on the session, the frame-cadence seam, the query_user two-phase
intercept, and the per-turn cost ledger.

    RealtimeClient (wire, voice/realtime.py)   -- dumb pipe, locked wire
    BrainBackend    (this module)              -- ONE interface, both lanes
    VoiceSession    (this module)              -- the turn laws
    app / director  (later slice)              -- tool execution, sinks

Plan §5b wants one backend interface both lanes implement: utterance in,
events out (spoken/text), a frame per decision turn, tool results in,
failover signal out.  :class:`BrainBackend` is that contract;
:class:`RealtimeBackend` (the cloud lane) lives HERE as part of session;
the offline/transcript impls come later and implement the same surface.

--------------------------------------------------------------------------
THE LAWS (each with its implementation pointer)
--------------------------------------------------------------------------

L1. HALF-DUPLEX GATE (plan §3).  While a response is in flight AND the
    last intermediate response.done carried a tool call, the gate stays
    SHUT — no new response.create.  Implemented as: the gate is open only
    when no response is in flight AND no tool results are pending
    (``_pending_call_ids`` empty).  A done with no tool calls reopens it;
    a done with tool calls leaves it shut until the batch's outputs are
    delivered and the ONE continuation response completes.
    QUEUED-MIC-TURN POLICY (exactly one, documented): DROP.  A user
    transcription that lands while the gate is shut is logged and
    discarded (``dropped_utterance`` event) — never queued for replay:
    replaying stale audio after a tool turn corrupts the plan, and the
    human re-speaks or VAD re-captures on the next open turn.
    LANE NOTE (transcript): the transcript lane owns its server turns
    lane-side and fires NO transcription event for a normal committed
    server turn (the L9 query-answer cancel path excepted) — the lane's
    commit IS the admission, so this L1 branch is the realtime lane's
    path and no false ``dropped while gate shut`` is ever logged for an
    utterance about to be answered (2026-09-04 fix).

L2. BARGE-IN (plan §3).  On server ``speech_started`` while a response is
    in flight and BEFORE any tool results of that response: truncate the
    assistant item + response.cancel.  NEVER truncate after tool results
    are flowing (``_pending_call_ids`` non-empty) — the model's calls are
    committed; cutting would orphan their outputs.  Wire note: the locked
    slice allowlist in realtime.py does NOT contain
    ``conversation.item.truncate`` (``send_truncate`` is a documented dead
    end that raises AssertionError through ``_emit``), so the session
    attempts truncate and ALWAYS sends cancel; until the orchestrator
    allowlists truncate, the barge-in is cancel-only and logs a warning.

L3. function_call_output — EXACTLY ONE per call_id (plan §3).  Client-side
    dedup map ``_emitted_outputs``, NEVER cleared for the session's
    lifetime: every call_id the session processes is answered — NEVER
    zero (intercepted queries answer immediately; refused gate-shut calls
    answer with an explicit refusal envelope).  Call ids are per-client
    monotonic (the C1 fix: the transcript lane's client mints a fresh id
    for every parsed call), so the wire never re-delivers one; a
    duplicate landing in the map is a duplicate/stale DIRECTOR answer or
    an id collision — it is answered ONCE with a logged warning, never
    twice, never silently.
    The query_user two-phase is the documented exception (plan §4): the
    intercept emits the placeholder ``OK: awaiting…`` at once and the
    answer ``DONE:<answer>`` when the human replies — two deliberate
    emissions for one call_id, forced through the dedup map.

L4. ONE response.create per batch of tool outputs (plan §3), never one
    per output.  ``respond_tool_result`` may deliver outputs one by one
    (the wire accepts them at any time), but the continuation response is
    created exactly once, when the batch's last output lands.
    The query_user continuation is the documented exception shape (plan
    §4): a dedicated ask-turn response and then the plan-continuation
    response, both session-created and single.  An ANSWER-tool batch
    (ANSWER_TOOLS — the registry's notify_user, see voice/session.py
    below) is the second documented exception: its drain CLOSES
    the turn (the answer was the model's final words — never a
    continuation that would re-ask it), and MAX_DECISION_ROUNDS_PER_TURN
    caps decision continuations per utterance turn the same way.

L5. MUTE / QUIET (plan §3).  While the echoed audio is queued for
    playback (``_queued_s > 0``) OR the session mute flag is set, NO
    silence frames are uploaded: ``upload_mic_audio`` drops the chunk.
    The session NEVER fabricates silence — the exact bytes given are the
    only bytes that ever reach the lane; a drop is a drop, never a padded
    low-energy frame "to keep the buffer warm".

L6. PLAYBACK QUEUE CAP (plan §3): 60 s of queued audio per response plus
    a 0.6 s lead.  Deltas beyond the cap are dropped with a logged warning
    (``audio_dropped``).  The lead: the first 0.6 s of a response is held
    in ``_lead_buffer`` before any audio is released to the sink
    (``audio`` events) so playback never starts on a model false-start.

L7. 60-MIN SESSION CAP (plan §3).  The realtime client performs the clean
    context-reset reconnect and fires ``session.expired``; the session
    clears its turn-local state (the old conversation context is dead) and
    surfaces the expiry as a failover signal AND a ``session_expired``
    event — the app decides the lane switch.

L8. COST LEDGER (plan §3).  Every ``response.done`` usage lands in
    ``_cost_ledger``; ``last_cost()`` / ``cost_summary()`` serve the
    bake-off consumer.  Usage is never assumed constant: absent usage is
    reported as "Not reported", never guessed.

L9. QUERY_USER TWO-PHASE INTERCEPT (plan §4; cross-checked against
    brain/agent/tools.py — the registry builds make_user_tool
    ("query_user", blocking=True) and ("notify_user", blocking=False), so
    the query set is EXACTLY {"query_user"}; notify_user is deliberately
    not intercepted — it stays instant non-motion through the executor).
    Phase 1: the session answers the call_id with ``OK: awaiting…``
    IMMEDIATELY (L3), then speaks the question through ONE response.create
    whose instructions say "ask the human and stop" with tools disabled
    (``tools: []`` / ``tool_choice: "none"``).  Phase 2: the gate reopens
    and the NEXT transcription is the answer: ``DONE:<answer>`` output +
    the human's input_text attached + ONE response.create with tools
    enabled to continue the plan.

L10. FRAME CADENCE SEAM (plan §3 hard contract).  The session owns only
    the seam: a per-decision-turn frame hook (callable returning a fresh
    frame envelope or None) attached once at construction (settable later
    via :meth:`set_frame_hook` — L3 wires the gateway).  On each new
    model decision turn the hook is called once and ``input_image`` is
    attached ONLY when the payload is fresh and present; the session never
    fabricates a frame.  Where the seam runs is a LANE capability
    (round-3, S2, ``BrainBackend.created_precedes_model_call``):
    session-initiated turns run it before every ``response.create``; the
    TRANSCRIPT lane additionally runs it at a bare ``response.created``
    (there the worker fires the created immediately before its ONE chat
    call, so a frame staged at the created attaches to that chat), while
    the REALTIME lane never runs it at a server-VAD created (the server
    snapshotted the input before the event reached us — imageless by
    construction).  None/stale -> the turn goes out imageless and a
    per-turn instruction tells the model it is blind (session-initiated
    turns only; a server-VAD turn cannot be instructed after the fact —
    the created-time call sites drop the note and fire ``frame_absent``
    instead).

--------------------------------------------------------------------------
CONCURRENCY MODEL (documented per plan §3)
--------------------------------------------------------------------------
RealtimeClient's connect/read/reconnect loop runs on the app's asyncio
loop (the app schedules ``RealtimeBackend.run_forever()`` as a task);
inbound events dispatch inline on the loop to the session's sync
handlers, which only mutate session state and fire outward events (they
never block).  Sends from the loop thread are scheduled with
``asyncio.ensure_future``; sends from any other thread (the mic UDP pump)
bridge with ``asyncio.run_coroutine_threadsafe`` inside RealtimeBackend.
VoiceSession state is confined to the loop thread EXCEPT ``muted`` —
a plain bool read by the mic path, written from the loop thread — the
single documented cross-thread read.

Module-level import surface is stdlib-only (abc/asyncio/base64/inspect/
json/logging/time/typing) plus one sibling wire constant from
voice/realtime.py (CONVERSATION_ITEM_TRUNCATE, used to pre-validate the
truncate dead end), matching the established voice/ sibling pattern
(voice/__init__.py): third-party dependencies load lazily inside
methods, so importing this module succeeds dep-free on any laptop.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import inspect
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from voice.realtime import CONVERSATION_ITEM_TRUNCATE

log = logging.getLogger("voice.session")

# -------------------------------------------------------- the fixed numbers

#: wire audio: 24 kHz PCM16 mono
AUDIO_SAMPLE_RATE = 24000
#: hold the first 0.6 s of a response before the sink hears anything, so
#: playback never starts on a false start
PLAYBACK_LEAD_S = 0.6
#: cap on queued audio per response, in seconds
PLAYBACK_QUEUE_CAP_S = 60.0

#: tools intercepted at the session: only query_user blocks for a human
#: answer, so it is the only one intercepted; notify_user does not block
QUERY_TOOLS = frozenset({"query_user"})

#: the answer tool, notify_user: non-blocking, its message is the model's
#: closing notice spoken by the executor (no separate say tool exists).
#: a drained batch holding one closes the turn instead of re-asking.
ANSWER_TOOLS = frozenset({"notify_user"})

#: decision continuations allowed per utterance turn.  each drain creates
#: at most one, but answering every round with a non-answer tool call would
#: chain without bound; at the cap the turn closes.
MAX_DECISION_ROUNDS_PER_TURN = 8

#: phase 1 placeholder.  the "OK:" prefix reads as neutral in the
#: executor's budget classifier: the tool ran, no world-goal claim.
QUERY_OK_PLACEHOLDER = "OK: awaiting…"
#: phase 2 prefix for the human's answer
ANSWER_PREFIX = "DONE:"

#: per-turn note when the frame hook yields no fresh frame
NO_FRAME_INSTRUCTION = (
    "NO fresh camera frame is available for this turn. Do NOT describe the "
    "scene; proceed from the conversation alone."
)

#: ask-turn instruction: speak the question once, with tools off
ASK_INSTRUCTION_TEMPLATE = (
    "Ask the human operator the following question OUT LOUD, then STOP. "
    "Do NOT call any tools. Question: {question}"
)

#: refusal for a call that arrived with the gate shut (a server
#: auto-create raced a mid-batch turn): answered once, never executed and
#: never truncated
GATE_SHUT_REFUSAL_TEMPLATE = (
    "ERROR: {name} not executed — the session was mid-turn with pending "
    "tool results (half-duplex gate shut). Repeat the request when the "
    "turn completes."
)

#: events the session fires, payloads per handler
SESSION_EVENTS = frozenset((
    "tool_call", "tool_refused", "user_text", "dropped_utterance",
    "spoken", "text", "audio", "audio_dropped", "playback_started",
    "playback_finished", "speech_started", "barge_in", "answer",
    "query_waiting", "frame_attached", "frame_absent", "turn_complete",
    "gate_changed", "failover", "session_expired", "started", "closed",
))

#: wire events the session subscribes to; beta and GA spellings of the
#: transcript events are both listed so either wire name is routed
BACKEND_EVENTS = (
    "response.created",
    "response.output_item.done",
    "response.function_call_arguments.done",
    "response.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "conversation.item.input_audio_transcription.completed",
    "input_audio_buffer.speech_started",
    "error",
    "session.expired",
)

# -------------------------------------------- BrainBackend (the shared API)


class BrainBackend(abc.ABC):
    """One interface BOTH lanes implement (plan §5b): utterance in, events
    out (spoken/text), a frame per decision turn, tool results in, failover
    signal out.

    The VoiceSession drives EVERY law through this contract and never
    touches a lane client directly — the transcript/offline impls later
    implement the same surface.  Events flow lane -> session via
    :meth:`on_event` (same ``(name, payload)`` callback shape as the
    realtime client); everything else flows session -> lane.

    Failover: the lane signals its own fatal conditions (auth failure,
    retry exhaustion, the 60-min session-cap reconnect) as the wire events
    ``error`` / ``session.expired``; the session turns those into the
    ``failover`` signal.

    Concurrency: each backend brings its own threading model and documents
    it; the session calls backend methods from ANY thread (the mic pump)
    and expects them to be safe.  Send methods return True when the call
    was accepted for delivery, False when it was dropped (loudly logged —
    never silently).

    Lane capability (round-3, S2): ``created_precedes_model_call`` —
    True when, on THIS lane, a bare ``response.created`` (no input item
    yet attached) is ALWAYS followed by a session-owned model call whose
    chat payload is built AFTER the created was handled.  The transcript
    lane qualifies (the worker fires created, then chats); the realtime
    lane does NOT (created precedes nothing — the session never creates
    a response there, and a server-VAD created can be followed by a
    user item the session does not stage for).  The session uses the
    flag to run the L10 frame seam at a bare created on the qualifying
    lane only — see ``_handle_response_created``.
    """

    # lane capability: a bare created is followed by a session-owned model call
    created_precedes_model_call = False

    # --------------------------------------------------- start up / shut down

    @abc.abstractmethod
    def start(self) -> None:
        """Bring the lane up (wire handlers; capture the app loop)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Tear the lane down."""

    @abc.abstractmethod
    def on_event(self, name: str, callback: Callable[[str, dict], Any]) -> None:
        """Subscribe to lane events; callbacks are ``(name, payload)``."""

    # --------------------------------------------- utterance, frame, results

    def submit_mic_audio(self, chunk: bytes) -> bool:
        """PCM16/24k mono chunk -> the lane (realtime lane only).

        The transcript/offline lanes have no live mic: the default drops
        (returns False).  The session's mute law (L5) sits ABOVE this.
        """
        return False

    @abc.abstractmethod
    def send_user_text(self, text: str) -> bool:
        """Inject a text user turn (the query answer attachment, L9)."""

    @abc.abstractmethod
    def send_user_frame(self, jpeg_b64: str) -> bool:
        """Attach one JPEG (already base64) as a user image part (L10)."""

    @abc.abstractmethod
    def send_tool_result(self, call_id: str, output_text: str) -> bool:
        """Answer one function call (function_call_output item).  The model
        does nothing until a create_response follows — the session owns
        that second step (L4)."""

    # --------------------------------------------------- driving the response

    @abc.abstractmethod
    def create_response(
        self, *, tools: bool = True, instruction: Optional[str] = None
    ) -> bool:
        """Ask for ONE response.

        ``tools`` False disables the model's tools for this response
        (``tools: []`` / ``tool_choice: "none"`` override); ``instruction``
        is a per-response instruction override.  The realtime lane sends a
        plain ``response.create`` when both take their defaults and a
        response-level override otherwise (the locked slice's
        ``RealtimeClient.create_response`` supports neither — the override
        event is built through the client's own ``_emit`` choke point so
        the allowlist enforcement stays intact).
        """

    @abc.abstractmethod
    def cancel_response(self) -> bool:
        """Cancel the in-flight response (barge-in, L2)."""

    @abc.abstractmethod
    def truncate_response(self, item_id: str, audio_end_ms: int) -> bool:
        """Rewrite an assistant item to only the audio actually heard (L2).

        The realtime lane's ``conversation.item.truncate`` is a documented
        dead end while the locked slice allowlist excludes it
        (voice/realtime.py ``send_truncate``): the backend then returns
        False with a logged warning and the barge-in proceeds cancel-only.
        """


# ------------------------------------------ RealtimeBackend (the cloud lane)


class RealtimeBackend(BrainBackend):
    """The cloud lane's BrainBackend: wraps a
    voice.realtime.RealtimeClient.

    All sends bridge onto the client's asyncio loop: from the loop thread
    they are scheduled with ``asyncio.ensure_future``; from any other
    thread (the mic UDP pump) with ``asyncio.run_coroutine_threadsafe``.
    The app MUST schedule :meth:`run_forever` as a task on the loop (it
    drives the client's connect/read/reconnect loop) and call :meth:`start`
    from the loop thread before events must flow.  A send made with no
    captured loop is dropped with a logged error — loud, never silent.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # --------------------------------------------------- start up / shut down

    def start(self) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        log.debug("realtime backend started (loop=%s)", self._loop is not None)

    async def run_forever(self) -> None:
        """Drive the client's connect/read/reconnect loop (app task)."""
        return await self._client.run_forever()

    def close(self) -> bool:
        if self._client is not None:
            return self._bridge(self._client.close())
        return True

    def on_event(self, name: str, callback: Callable[[str, dict], Any]) -> None:
        self._client.on_event(name, callback)

    # ------------------------------------------------- bridging onto the app's loop

    def _bridge(self, coro: Any) -> bool:
        """Schedule ``coro`` on the app loop; True when accepted."""
        loop = self._loop
        if loop is None or loop.is_closed():
            log.error(
                "realtime backend send dropped: no live asyncio loop "
                "(call start() on the loop and schedule run_forever())"
            )
            return False
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            asyncio.ensure_future(coro)
        else:
            asyncio.run_coroutine_threadsafe(coro, loop)
        return True

    # -------------------------------------------------------- the send calls

    def submit_mic_audio(self, chunk: bytes) -> bool:
        return self._bridge(self._client.send_audio(bytes(chunk)))

    def send_user_text(self, text: str) -> bool:
        return self._bridge(self._client.send_input_text(str(text)))

    def send_user_frame(self, jpeg_b64: str) -> bool:
        return self._bridge(self._client.send_image(str(jpeg_b64)))

    def send_tool_result(self, call_id: str, output_text: str) -> bool:
        return self._bridge(
            self._client.send_function_call_output(str(call_id), str(output_text))
        )

    def create_response(
        self, *, tools: bool = True, instruction: Optional[str] = None
    ) -> bool:
        if tools and instruction is None:
            return self._bridge(self._client.create_response())
        # per-response override: the client's create_response() sends a
        # plain {"response": {}}, so build the event through its emit choke
        # point (allowlist enforced) and hand it to the raw send primitive
        override: Dict[str, Any] = {}
        if instruction is not None:
            override["instructions"] = str(instruction)
        if not tools:
            override["tools"] = []
            override["tool_choice"] = "none"
        event = self._client._emit("response.create", {"response": override})
        return self._bridge(self._client._send_raw(event))

    def cancel_response(self) -> bool:
        return self._bridge(self._client.cancel_response())

    def truncate_response(self, item_id: str, audio_end_ms: int) -> bool:
        """The realtime lane's ``conversation.item.truncate`` is a
        documented dead end while the locked slice allowlist excludes it
        (voice/realtime.py ``send_truncate``): the event type is
        pre-validated at the client's OWN ``_emit`` choke point HERE —
        ``send_truncate`` is an async coroutine whose body (and its
        ``_emit`` AssertionError) runs only when AWAITED, so a try/except
        around the call can never see the refusal (it used to escape as
        an unhandled task exception on the live loop while this method
        returned True).  The refusal therefore returns False here,
        synchronously, with a logged warning, and the barge-in proceeds
        cancel-only.  When the type IS allowlisted the send is scheduled
        inside a guarded task (see :meth:`_truncate_guard`): any wire
        refusal while the coroutine RUNS is caught inside the task and
        logged — never an unhandled task exception.
        """
        try:
            item = str(item_id)
            end_ms = max(0, int(audio_end_ms))
            # pre-validate the exact event through the client's emit choke
            # point: an allowlist refusal surfaces here, synchronously,
            # instead of inside the coroutine
            self._client._emit(
                CONVERSATION_ITEM_TRUNCATE,
                {"item_id": item, "content_index": 0, "audio_end_ms": end_ms},
            )
        except AssertionError as exc:
            # truncate is not in the client's event allowlist: the barge-in
            # stays cancel-only and the refusal is reported here
            log.warning(
                "conversation.item.truncate is a documented dead end on "
                "the realtime lane (%s); barge-in proceeds with "
                "response.cancel alone", exc
            )
            return False
        except Exception as exc:  # noqa: BLE001 - pre-validation is only half
            # any failure in pre-validation (disconnected client, bad
            # audio_end_ms) also goes cancel-only and never propagates
            log.warning(
                "conversation.item.truncate unavailable (%s); barge-in "
                "proceeds with response.cancel alone", exc
            )
            return False
        try:
            coro = self._client.send_truncate(item, end_ms)
        except Exception as exc:  # noqa: BLE001 - building it failed
            # a client without send_truncate falls back to cancel-only
            log.warning(
                "conversation.item.truncate unavailable (%s); barge-in "
                "proceeds with response.cancel alone", exc
            )
            return False
        return self._bridge(self._truncate_guard(coro))

    async def _truncate_guard(self, coro: Any) -> None:
        """Run the truncate send INSIDE the scheduled task: any wire
        refusal (the ``_emit`` AssertionError included — it fires when
        the coroutine body executes, not when it is built) is caught and
        logged HERE, so no unhandled task exception can escape.  The
        barge-in path (L2) always reaches response.cancel regardless."""
        try:
            await coro
        except AssertionError as exc:
            log.warning(
                "conversation.item.truncate refused by the wire (%s); "
                "barge-in proceeds with response.cancel alone", exc
            )
        except Exception as exc:  # noqa: BLE001 - awaiting can raise
            # awaiting can raise (a gone client); any failure goes
            # cancel-only and never propagates
            log.warning(
                "conversation.item.truncate unavailable (%s); barge-in "
                "proceeds with response.cancel alone", exc
            )

# ----------------------------------------------- VoiceSession (the turn laws)


class VoiceSession:
    """The turn-law enforcement layer (laws L1-L10, module docstring).

    Build it with a :class:`BrainBackend` and the optional frame hook,
    subscribe to session events, :meth:`start` it, then feed it:

    * mic chunks via :meth:`upload_mic_audio` (any thread);
    * tool results via :meth:`respond_tool_result` (the director, after
      executing a ``tool_call`` event);
    * nothing else — every lane event arrives through the backend.

    Outward events (subscribe via :meth:`on_event`; payloads documented at
    each ``_fire`` site): ``tool_call``, ``tool_refused``, ``user_text``,
    ``dropped_utterance``, ``spoken``, ``text``, ``audio``,
    ``audio_dropped``, ``playback_started``, ``playback_finished``,
    ``speech_started``, ``barge_in``, ``answer``, ``query_waiting``,
    ``frame_attached``, ``frame_absent``, ``turn_complete``,
    ``gate_changed``, ``failover``, ``session_expired``, ``started``,
    ``closed``.
    """

    def __init__(
        self,
        backend: BrainBackend,
        *,
        frame_hook: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
        query_tools: frozenset = QUERY_TOOLS,
        playback_lead_s: float = PLAYBACK_LEAD_S,
        playback_queue_cap_s: float = PLAYBACK_QUEUE_CAP_S,
    ) -> None:
        if backend is None:
            raise ValueError("VoiceSession requires a BrainBackend")
        self.backend = backend
        #: frame seam: callable returning a fresh frame envelope or None,
        #: called once per decision turn
        self._frame_hook = frame_hook
        self._query_tools = frozenset(query_tools)
        self._answer_tools = frozenset(ANSWER_TOOLS)
        self._playback_lead_s = float(playback_lead_s)
        self._playback_queue_cap_s = float(playback_queue_cap_s)

        self._handlers: Dict[str, List[Callable[[str, dict], Any]]] = {}
        self._closed = False
        self._started = False

        # -- the half-duplex gate --
        self._response_in_flight = False
        self._response_kind: Optional[str] = None  # "decision"|"ask"|"server"
        self._pending_call_ids: List[str] = []  # calls awaiting their tool output
        self._turn_call_ids = set()  # call_ids seen this turn, for dedup
        self._continuation_created = False  # one create per batch
        #: True while the current batch holds an ANSWER_TOOLS call; its
        #: drain closes the turn instead of creating a continuation
        self._batch_has_answer_call = False
        #: decision continuations since the turn seed, capped at
        #: MAX_DECISION_ROUNDS_PER_TURN; reset by a new seed, a server-kind
        #: created, or a drain that closes the turn
        self._decision_rounds_this_turn = 0
        #: True after a response.done that left a batch open: a later call
        #: belongs to a different response (a server auto-create raced the
        #: turn), so it is refused and answered once, never run or truncated
        self._tools_flowing_after_done = False
        self._frame_attached_this_turn = False

        # -- query_user, two-phase --
        self._query_call_id: Optional[str] = None
        self._ask_sent = False
        self._query_question = ""

        # -- the dedup map --
        self._emitted_outputs: Dict[str, str] = {}

        # -- mute and playback --
        self._muted = False
        self._queued_s = 0.0
        self._lead_buffer: List[tuple] = []
        self._lead_accum_s = 0.0
        #: a local stop dropped the rest of this response's audio; cleared
        #: at the next created or done
        self._drop_audio = False

        # -- the cost ledger --
        self._cost_ledger: List[Dict[str, Any]] = []

    # --------------------------------------------------- start up / shut down

    def start(self) -> None:
        """Bring the lane up and wire the backend events: one call starts
        the whole stack.  (The realtime app additionally schedules
        ``RealtimeBackend.run_forever()`` as a task on the loop — that is
        the lane's connect/read/reconnect loop; see the module docstring's
        concurrency model.)"""
        if self._started:
            return
        try:
            self.backend.start()
        except Exception:  # noqa: BLE001 - a lane hiccup can't kill the wiring
            log.exception("backend.start() raised")
        for name in BACKEND_EVENTS:
            self.backend.on_event(name, self._on_backend_event)
        self._started = True
        self._fire("started", {})

    def close(self) -> None:
        """Tear down: backend close, turn state cleared, ``closed`` fired."""
        if self._closed:
            return
        self._closed = True
        self._reset_turn_state()
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            log.exception("backend.close() raised")
        self._fire("closed", {})

    def on_event(self, name: str, callback: Callable[[str, dict], Any]) -> None:
        """Subscribe to session-level events (see the class docstring).

        Sync callables run inline; an awaitable return is scheduled as a
        task on the running loop (same contract as the realtime client).
        """
        self._handlers.setdefault(name, []).append(callback)

    def _fire(self, name: str, payload: Dict[str, Any]) -> None:
        if name not in SESSION_EVENTS:
            log.debug("session fires undocumented event %r", name)
        for callback in list(self._handlers.get(name, ())):
            try:
                result = callback(name, payload)
            except Exception:  # noqa: BLE001 - one bad listener can't kill
                log.exception("session callback for %r failed", name)
                continue
            if inspect.isawaitable(result):
                try:
                    asyncio.ensure_future(result)
                except RuntimeError:  # no running loop: the awaitable is dropped
                    log.debug("session callback %r returned awaitable outside a loop", name)

    def _on_backend_event(self, name: str, payload: Dict[str, Any]) -> None:
        """Route every subscribed backend event to its handler."""
        handler = getattr(self, "_handle_" + name.replace(".", "_"), None)
        if handler is None:
            log.debug("session ignores backend event %r", name)
            return
        try:
            handler(payload)
        except Exception:  # noqa: BLE001 - one bad handler can't kill
            log.exception("session handler for %r failed", name)

    # ------------------------------------------------------ gate, mic and mute

    @property
    def gate_open(self) -> bool:
        """The half-duplex gate (L1): open only when no response is in
        flight AND no tool results are pending."""
        return (not self._response_in_flight) and (not self._pending_call_ids)

    @property
    def muted(self) -> bool:
        """Mute law (L5): the explicit session mute flag OR echoed audio
        currently queued for playback."""
        return self._muted or self._queued_s > 0.0

    def set_muted(self, flag: bool) -> None:
        """Set the session mute flag (L5)."""
        self._muted = bool(flag)

    def upload_mic_audio(self, chunk: bytes) -> bool:
        """PCM16/24k mono chunk from the mic pump (any thread).

        Mute law (L5): while the echoed audio is queued for playback OR the
        session mute flag is set, the chunk is DROPPED — never uploaded,
        never padded.  The session NEVER fabricates silence: the exact
        bytes given are the only bytes that ever reach the lane, and a
        drop is a drop (no synthetic low-energy frame is uploaded "to keep
        the buffer warm").  Returns True when the chunk was handed to the
        lane.
        """
        if self._closed or not chunk:
            return False
        if self.muted:
            return False
        return self.backend.submit_mic_audio(chunk)

    def pending_tool_calls(self) -> List[str]:
        """call_ids whose outputs the director has not delivered yet."""
        return list(self._pending_call_ids)

    # --------------------------------------------------- the tool-result batch

    def respond_tool_result(self, call_id: str, output_text: str) -> bool:
        """The director's answer for one executed tool call (L3 + L4).

        Delivers the output item (dedup: exactly once per call_id) and, on
        the batch's LAST output, decides the batch's ONE follow-up: an
        ANSWER-tool batch (see ANSWER_TOOLS — notify_user) CLOSES the
        turn — the answer was the model's final words — and so does a
        batch whose drain would exceed MAX_DECISION_ROUNDS_PER_TURN; any
        other batch gets the single decision continuation response (never
        one per output).  Returns True when the output was accepted.  A
        call_id the session never recorded (a duplicate director answer,
        or a stale call from a reset session) is dropped with a warning.

        Concurrency note (C-L1, round 3): this method runs on the app's
        PER-CALL director threads (app.py spawns one daemon thread per
        tool execution), NOT on the lane's worker.  The batch bookkeeping
        below — ``_pending_call_ids`` removal, the drained-batch decision
        (``_continuation_created`` / ``_batch_has_answer_call``), the
        round counter and the one-create-per-batch response — is therefore
        deliberately UNLOCKED: each call_id has exactly one writer thread,
        the GIL serializes the list reads/removes, and the queueing to
        the transcript lane (FIFO) serializes the effects.  A full
        locking rewrite is deliberately NOT made here (out of round-3
        scope) — the drain's check-then-create races only when the app
        delivers two outputs for the SAME batch concurrently, which the
        director's one-thread-per-call + one-call-per-response contract
        prevents.
        """
        if self._closed:
            return False
        if call_id not in self._pending_call_ids:
            log.warning(
                "respond_tool_result for unknown/duplicate call_id %r "
                "(pending=%r) — dropped", call_id, self._pending_call_ids
            )
            return False
        self._send_function_call_output(call_id, output_text)
        self._pending_call_ids.remove(call_id)
        if not self._pending_call_ids:
            # batch drained: the post-done refusal stops, and the next
            # response's calls join their own fresh batch
            self._tools_flowing_after_done = False
        self._fire("gate_changed", {"open": self.gate_open})
        if not self._pending_call_ids and not self._continuation_created:
            self._continuation_created = True
            if self._batch_has_answer_call:
                # the drained batch held an answer (notify_user) call: its
                # final notice is out and the executor already spoke it, so
                # close the turn instead of a continuation that would re-ask
                self._batch_has_answer_call = False
                self._close_turn_after_drain("the drained batch was an "
                                             "answer (notify_user) call")
                return True
            if self._decision_rounds_this_turn >= MAX_DECISION_ROUNDS_PER_TURN:
                # safety cap: a model answering every round with a
                # non-answer tool call would chain forever; close the turn
                log.warning(
                    "decision round cap (%d) reached in one utterance "
                    "turn — closing the turn without a further "
                    "continuation",
                    MAX_DECISION_ROUNDS_PER_TURN,
                )
                self._close_turn_after_drain("decision round cap reached")
                return True
            self._decision_rounds_this_turn += 1
            # one create per batch, triggered by the last output
            self._create_response(tools=True, instruction=None, kind="decision")
        return True

    def _close_turn_after_drain(self, reason: str) -> None:
        """Close the utterance turn at a batch drain (the answer-final
        path): fire ``turn_complete`` + gate open exactly like a
        call-less response.done would have, and reset the decision-round
        count for the next turn.  ``reason`` names the close for the log.
        An ask still owed its turn (L9: a query_user intercepted in the
        SAME response) starts FIRST — the question must be asked before
        the turn closes; the ask's own done then fires ``turn_complete``
        and reopens the gate for the human's answer."""
        self._decision_rounds_this_turn = 0
        if self._query_call_id is not None and not self._ask_sent:
            log.info("turn closed at the batch drain (%s) — the query "
                     "ask still owed its turn; asking first", reason)
            self._start_ask_turn()
            return
        log.info("turn closed at the batch drain (%s)", reason)
        # usage and status of the batch-owning response, from the ledger:
        # the drain site never sees the done payload itself
        status = "completed"
        usage = None
        if self._cost_ledger:
            status = str(self._cost_ledger[-1].get("status") or status)
            usage = self._cost_ledger[-1].get("usage")
        self._fire("turn_complete", {"usage": usage, "status": status})
        self._fire("gate_changed", {"open": self.gate_open})

    def _send_function_call_output(
        self, call_id: str, output_text: str, *, force: bool = False
    ) -> bool:
        """L3: the single emission choke for function_call_output.

        A call_id already in the dedup map is answered exactly once: the
        repeat is logged and dropped.  ``force`` is the query_user
        two-phase exception (L9): the ``DONE:<answer>`` emission succeeds
        the ``OK: awaiting…`` placeholder for the same call_id.
        """
        if call_id in self._emitted_outputs and not force:
            # a new call cannot land here: the client mints a fresh id per
            # parsed call, so a repeat is a stale or duplicate tool answer
            # or an id collision, never a server re-emit
            log.warning(
                "duplicate function_call_output for call_id %r (already "
                "emitted for this session; a duplicate director answer or "
                "an id collision, not a server repeat) — answering once, "
                "never zero", call_id
            )
            return False
        self._emitted_outputs[call_id] = output_text
        return self.backend.send_tool_result(call_id, output_text)

    # --------------------------------------------------------- the frame seam

    def set_frame_hook(self, hook: Optional[Callable[[], Optional[Dict[str, Any]]]]) -> None:
        """Attach/replace the per-decision-turn frame hook (L10; L3 wires
        the gateway capture here — the seam stays pluggable)."""
        self._frame_hook = hook

    def _prepare_frame_for_turn(self, instruction: Optional[str]) -> Optional[str]:
        """L10: one frame-hook call per decision turn.

        Attaches ``input_image`` ONLY when the envelope is fresh and
        present (never fabricated); otherwise returns ``instruction`` with
        the NO_FRAME note appended so the model is told it is blind.

        Called from the places whose ordering lets the frame actually
        inform the answer (round-3, S2):

        * ``_create_response(kind="decision")`` — BEFORE ``response.create``
          goes out (every lane);
        * a bare ``response.created`` on a lane whose
          ``created_precedes_model_call`` is True (the transcript lane):
          there the worker fires the created immediately before its ONE
          chat call, so a frame staged at the created attaches to that
          chat (see _handle_response_created).

        It is NOT called for realtime server-VAD turns (that lane's
        capability is False): by the time ``response.created`` arrives the
        input is already snapshotted, so a send there would return True
        and fire "frame_attached" for a turn the model answers blind.
        For the same reason the NO_FRAME note is only meaningful before
        creation — a response already created cannot be instructed after
        the fact, so the note's return value is dropped at the
        created-time call sites (there is no instruction channel on those
        turns).
        """
        if self._frame_attached_this_turn:
            return instruction
        self._frame_attached_this_turn = True
        envelope = None
        if self._frame_hook is not None:
            try:
                envelope = self._frame_hook()
            except Exception:  # noqa: BLE001 - one hook hiccup must not kill
                log.exception("frame hook raised")
                envelope = None
        if (
            isinstance(envelope, dict)
            and envelope.get("fresh") is True
            and envelope.get("jpeg_b64")
        ):
            jpeg = str(envelope["jpeg_b64"])
            if self.backend.send_user_frame(jpeg):
                self._fire("frame_attached", {"bytes": len(jpeg)})
                return instruction
            # the lane refused the send (no live loop, dropped bridge):
            # claim nothing; the turn is imageless and frame_absent says so
            self._fire("frame_absent", {
                "reason": "frame send refused by the backend (not attached)",
            })
            note = NO_FRAME_INSTRUCTION
            return (instruction + " " + note) if instruction else note
        if envelope is None:
            reason = "no frame source (hook returned None)"
        elif isinstance(envelope, dict) and envelope.get("fresh") is False:
            reason = "stale frame (fresh=False)"
        else:
            reason = "invalid frame envelope"
        self._fire("frame_absent", {"reason": reason})
        note = NO_FRAME_INSTRUCTION
        return (instruction + " " + note) if instruction else note

    # ---------------------------------------------------- creating a response

    def _create_response(
        self,
        *,
        tools: bool,
        instruction: Optional[str],
        kind: str,
    ) -> bool:
        """The ONE place the session asks for a response (L1/L4/L9/L10).

        ``kind`` labels the turn for the done-accounting: "decision"
        (tools-enabled plan turns), "ask" (the query question, L9), and
        "server" (VAD auto-creates — never created here).  Decision turns
        run the frame seam first; ask turns never carry a frame.
        """
        if self._closed:
            return False
        if kind == "decision":
            instruction = self._prepare_frame_for_turn(instruction)
        self._response_in_flight = True
        self._response_kind = kind
        self._frame_attached_this_turn = True
        ok = self.backend.create_response(tools=tools, instruction=instruction)
        self._fire("gate_changed", {"open": False})
        if not ok:
            log.error("create_response was dropped by the backend (kind=%r)", kind)
            self._response_in_flight = False
            self._response_kind = None
        return ok

    # ----------------------------------------------------- the turn handlers

    def _handle_response_created(self, payload: Dict[str, Any]) -> None:
        """A response started.  Session-created responses already carry
        their kind; a bare response.created means a lane-started turn.
        What the frame seam does there is a LANE capability (round-3,
        S2, ``BrainBackend.created_precedes_model_call``):

        * Realtime lane (False): the server VAD started the response and
          snapshotted its input strictly before this event reached us —
          the turn is IMAGELESS by construction.  The seam therefore does
          NOT run here; it fires ``frame_absent`` with the ordering
          reason so the transcript records what actually happened.
        * Transcript lane (True): the worker fires this created
          immediately before its ONE chat call, so the created-to-chat
          window exists: the seam runs inline and a fresh frame attaches
          to the very response this created announces.  ``no_model_call``
          createds (the lane's empty-transcript retry — no chat follows)
          and createds fired while a query answer is being captured (L9
          phase 2) skip the seam: nothing will see a frame, and staging
          one would misattribute it to a later turn.

        While a query answer is being captured (L9 phase 2), a VAD
        auto-create is cancelled immediately on BOTH lanes: the human's
        speech is the answer, not a new turn.
        """
        self._drop_audio = False          # a new response may play
        if self._response_kind is None:
            self._response_kind = "server"
            self._response_in_flight = True
            # fresh utterance-turn seed: session-created responses arrive
            # with their kind set and skip this; the round budget resets
            self._decision_rounds_this_turn = 0
            if self.backend.created_precedes_model_call:
                # transcript lane: a bare created precedes one chat call, so
                # a staged frame informs it.  creates with no model call, or
                # made while an answer is owed, are skipped silently.
                if not (
                    payload.get("no_model_call")
                    or (self._query_call_id is not None and self._ask_sent)
                ):
                    self._prepare_frame_for_turn(None)
            else:
                # realtime lane: server VAD snapshotted the input before this
                # event, so the turn is imageless.  send nothing and leave
                # the seam armed so the continuation gets a frame.
                self._frame_attached_this_turn = False
                self._fire("frame_absent", {
                    "reason": "server-VAD turn: the server created the "
                              "response before any frame could be attached "
                              "— this turn is imageless (not a camera "
                              "failure)",
                })
        if (
            self._query_call_id is not None
            and self._ask_sent
            and self._response_kind == "server"
        ):
            log.info(
                "cancelling VAD auto-response while waiting for the query "
                "answer (the transcription is the answer, not a new turn)"
            )
            self.backend.cancel_response()
        self._fire("gate_changed", {"open": self.gate_open})

    def _handle_response_output_item_done(self, payload: Dict[str, Any]) -> None:
        """A response output item finished: function_call -> tool dispatch;
        message -> text event."""
        item = payload.get("item") or {}
        if not isinstance(item, dict):
            return
        itype = item.get("type")
        if itype == "function_call":
            self._on_function_call(
                name=str(item.get("name") or ""),
                call_id=str(item.get("call_id") or ""),
                arguments_text=str(item.get("arguments") or ""),
            )
        elif itype == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    self._fire("text", {"text": str(part.get("text") or "")})

    def _handle_response_function_call_arguments_done(self, payload: Dict[str, Any]) -> None:
        """Final arguments for a function call.  The call may already have
        been recorded via output_item.done (partial args there); the
        intercept refreshes its question text from the final arguments."""
        call_id = str(payload.get("call_id") or "")
        name = str(payload.get("name") or "")
        arguments_text = str(payload.get("arguments") or "")
        if call_id and call_id not in self._turn_call_ids:
            self._on_function_call(name, call_id, arguments_text)
        elif call_id == self._query_call_id:
            args = self._parse_arguments(arguments_text)
            question = str(args.get("question") or "")
            if question:
                self._query_question = question

    def _on_function_call(self, name: str, call_id: str, arguments_text: str) -> None:
        """A function call entered the turn (L2/L3/L9).

        Dedup: a server that repeats a call_id is ignored with a warning —
        its output is still answered exactly once.  Query-tool calls are
        intercepted here (L9 phase 1); normal calls join the batch freely
        while the CURRENT response is in flight (a normal multi-call
        response — L4 batches them all).  Only a call arriving AFTER the
        batch-owning response has done (``_tools_flowing_after_done``: a
        server auto-create raced the mid-batch turn) is refused — never
        executed, never truncated, answered exactly once.
        """
        if not call_id:
            log.error("function_call without a call_id — dropped")
            return
        if call_id in self._turn_call_ids:
            output_sent = call_id in self._emitted_outputs
            pending = call_id in self._pending_call_ids
            log.warning(
                "server repeated function_call for call_id %r (name %r) — "
                "the repeat is ignored; output_already_sent=%s pending=%s; "
                "its output is answered exactly once",
                call_id, name, output_sent, pending,
            )
            return
        self._turn_call_ids.add(call_id)
        args = self._parse_arguments(arguments_text)
        if name in self._query_tools:
            self._intercept_query(call_id, name, args)
            return
        if self._tools_flowing_after_done:
            # never truncate once tool results flow; a parallel batch would
            # break one-create-per-batch.  answer once, never zero.
            self._refuse_gate_shut_call(call_id, name)
            return
        self._pending_call_ids.append(call_id)
        if name in self._answer_tools:
            # an answer (notify_user) call joins the batch; at drain the
            # turn closes instead of a decision continuation, since the
            # answer is the model's final words
            self._batch_has_answer_call = True
        self._fire("tool_call", {
            "name": name, "call_id": call_id, "arguments": args,
            "intercepted": False,
        })

    @staticmethod
    def _parse_arguments(text: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(text) if text else {}
        except ValueError:
            return {"raw": text}
        return parsed if isinstance(parsed, dict) else {"raw": text}

    def _refuse_gate_shut_call(self, call_id: str, name: str) -> None:
        """Answer a gate-shut call exactly once with an explicit refusal
        (L3 never-zero; never executed, never truncated)."""
        text = GATE_SHUT_REFUSAL_TEMPLATE.format(name=name)
        self._send_function_call_output(call_id, text)
        self._fire("tool_refused", {"name": name, "call_id": call_id, "reason": "half-duplex gate shut"})

    def _intercept_query(self, call_id: str, name: str, args: Dict[str, Any]) -> None:
        """L9 phase 1: intercept a query_user call at the session.

        Emits the ``OK: awaiting…`` placeholder IMMEDIATELY (L3 — the
        call_id is never left at zero outputs), records the call for the
        ask-turn, and defers the question's speech to the current
        response's done.  A query_user from a response that started AFTER
        the batch-owning response has done (``_tools_flowing_after_done``:
        a server auto-create raced the mid-batch turn) is refused — a
        question cannot be interleaved mid-batch.  A query_user that is a
        SIBLING of the current response's calls (same response, same
        batch) is intercepted normally and simply waits its turn: the
        ask starts when the batch drains (see ``_handle_response_done``).
        """
        if self._tools_flowing_after_done:
            self._refuse_gate_shut_call(call_id, name)
            return
        if self._query_call_id is not None:
            # one open question at a time: a second query_user while the
            # first waits for its ask is refused, never stacked or dropped;
            # the refusal is its exactly-one output
            self._refuse_gate_shut_call(call_id, name)
            return
        self._query_call_id = call_id
        self._ask_sent = False
        self._query_question = str(args.get("question") or "")
        # placeholder emission: the first of the two outputs this call_id
        # is owed; the answer step sends the second
        self._emitted_outputs[call_id] = QUERY_OK_PLACEHOLDER
        self.backend.send_tool_result(call_id, QUERY_OK_PLACEHOLDER)
        self._fire("tool_call", {
            "name": name, "call_id": call_id, "arguments": args,
            "intercepted": True,
        })
        self._fire("gate_changed", {"open": self.gate_open})

    def _start_ask_turn(self) -> None:
        """L9 phase 1 (the speak step): ONE response.create whose
        instructions say "ask the human and stop", with tools disabled."""
        question = self._query_question or "(no question text)"
        instruction = ASK_INSTRUCTION_TEMPLATE.format(question=question)
        ok = self._create_response(tools=False, instruction=instruction, kind="ask")
        if not ok:
            # the question could not be spoken (a lane with no live loop);
            # the answer is captured when it arrives
            log.error(
                "ask-turn response.create failed — the question could not "
                "be spoken; still awaiting the answer"
            )
            self._ask_sent = True
            self._fire("query_waiting", {"call_id": self._query_call_id})

    def _complete_query(self, answer_text: str) -> None:
        """L9 phase 2: the next transcription is the answer.

        Delivers ``DONE:<answer>`` (the documented SECOND emission for
        this call_id — the placeholder closed the model's wait, this
        delivers the answer), attaches the human's input_text, and creates
        ONE response.create with tools enabled to continue the plan.
        """
        call_id = self._query_call_id
        self._query_call_id = None
        self._ask_sent = False
        self._query_question = ""
        self._send_function_call_output(call_id, ANSWER_PREFIX + answer_text, force=True)
        self.backend.send_user_text(answer_text)  # attach what the human said
        self._fire("answer", {"call_id": call_id, "text": answer_text})
        self._create_response(tools=True, instruction=None, kind="decision")

    def _handle_response_done(self, payload: Dict[str, Any]) -> None:
        """End-of-response accounting (L1/L4/L8/L9).

        Records the usage ledger entry, releases the trailing lead-buffered
        audio, clears any queue still booked from this response's stream
        (L5 mute-leak guard: a cancelled/errored response never reaches
        output_audio.done, whose handler owns the normal queue reset), and
        decides the next move: an ask-response done opens the answer
        capture; a done with the intercepted query's call and no pending
        batch starts the ask-turn; a done with a pending batch leaves the
        gate shut until the director's outputs land.
        """
        response = payload.get("response") or {}
        status = str(response.get("status") or "unknown")
        usage = response.get("usage")
        self._cost_ledger.append({
            "ts": time.monotonic(), "status": status, "usage": usage,
        })
        kind = self._response_kind or "server"
        self._response_in_flight = False
        self._response_kind = None
        self._turn_call_ids.clear()
        self._frame_attached_this_turn = False
        self._continuation_created = False
        self._flush_lead()
        self._drop_audio = False          # the dropped response has ended
        if self._queued_s > 0:
            # mute-leak guard: a cancelled or errored response never reaches
            # output_audio.done, whose handler owns the queue reset, so the
            # queue stays booked, ``muted`` sticks, and mic audio is dropped
            self._queued_s = 0.0
            self._fire("playback_finished", {})
        if kind == "ask":
            # the ask response finished: the question is out (a barge-in
            # may cut it; the wait survives); the next transcription is the answer
            self._ask_sent = True
            self._fire("query_waiting", {"call_id": self._query_call_id})
        elif (
            self._query_call_id is not None
            and not self._ask_sent
            and not self._pending_call_ids
        ):
            # the intercepted query's response settled with the batch
            # drained, so the ask waited for its own done
            self._start_ask_turn()
        if self._pending_call_ids:
            # the done left a batch open: until it drains, a call belongs to
            # a different response (a server auto-create raced the turn), so
            # refuse it.  calls during this response already joined the batch.
            self._tools_flowing_after_done = True
        if not self._pending_call_ids:
            self._fire("turn_complete", {"usage": usage, "status": status})
        self._fire("gate_changed", {"open": self.gate_open})

    def _handle_conversation_item_input_audio_transcription_completed(
        self, payload: Dict[str, Any]
    ) -> None:
        """A user utterance transcription (L1/L9).

        While a query answer is being captured it IS the answer (L9 phase
        2).  Otherwise: gate open -> forwarded as ``user_text`` (the server
        VAD owns the response start); gate shut -> DROPPED per the
        documented one-policy (L1) — logged, never queued for replay.
        Lane note: on the TRANSCRIPT lane this event is fired only for
        the L9 query-answer case — a normal committed server turn fires
        none (the lane's commit is the admission); this branch is the
        realtime lane's path.
        """
        transcript = (str(payload.get("transcript") or "")).strip()
        if not transcript:
            return
        if self._query_call_id is not None and self._ask_sent:
            self._complete_query(transcript)
            return
        if not self.gate_open:
            self._fire("dropped_utterance", {
                "text": transcript,
                "reason": "half-duplex gate shut (response in flight or "
                          "tool results pending) — DROP policy (L1)",
            })
            log.info("dropped utterance while gate shut: %r", transcript)
            return
        # log the user's words: without this a tool call has no visible
        # request behind it
        log.info("user said: %r", transcript)
        self._fire("user_text", {"text": transcript})

    def _handle_input_audio_buffer_speech_started(self, payload: Dict[str, Any]) -> None:
        """Barge-in signal (L2): truncate + cancel only before tool
        results are flowing; NEVER after."""
        item_id = str(payload.get("item_id") or "")
        audio_start_ms = int(payload.get("audio_start_ms") or 0)
        self._fire("speech_started", {
            "item_id": item_id, "audio_start_ms": audio_start_ms,
        })
        if not self._response_in_flight:
            return  # nothing to cut; VAD auto-creates
        if self._pending_call_ids:
            log.info(
                "barge-in speech ignored: tool results are flowing — NEVER "
                "truncate after tool results (L2)"
            )
            return
        self._barge_in(item_id, audio_start_ms)

    def stop_playback_now(self) -> Dict[str, Any]:
        """B8j-hf19: the local stop spotter heard 'stop' (any thread).

        The robot must go quiet NOW: the rest of the in-flight response's
        audio is dropped (``_drop_audio``, cleared at the next created/done),
        the response is CANCELLED when L2 allows a barge-in (no tool results
        flowing; after tool results it keeps running silently, never cut),
        and the queued-audio booking is cleared so the mute law (L5) stops
        dropping the operator's next words.  The caller cuts the sink.
        """
        cancelled = False
        if self._response_in_flight:
            self._drop_audio = True
            if not self._pending_call_ids:
                try:
                    cancelled = bool(self.backend.cancel_response())
                except Exception:  # noqa: BLE001 - stop is never allowed to raise
                    log.exception("stop_playback_now: cancel_response failed")
        cleared = self._queued_s > 0
        if cleared:
            self._queued_s = 0.0
            self._fire("playback_finished", {})
        log.info("local stop: audio dropped=%s cancelled=%s queue cleared=%s",
                 self._drop_audio, cancelled, cleared)
        return {"dropped": self._drop_audio, "cancelled": cancelled,
                "queue_cleared": cleared}

    def _barge_in(self, item_id: str, audio_end_ms: int) -> None:
        """Truncate the assistant item + cancel the response (L2).

        Truncate may be a documented dead end on the realtime lane (the
        backend returns False and logs); cancel always fires.
        """
        phase = "ask" if self._response_kind == "ask" else "decision"
        self.backend.truncate_response(item_id, audio_end_ms)
        self.backend.cancel_response()
        self._fire("barge_in", {
            "item_id": item_id, "audio_end_ms": audio_end_ms, "phase": phase,
        })

    # --------------------------------------------------------- audio playback

