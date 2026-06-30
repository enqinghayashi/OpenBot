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

