"""transcript: the REST cloud lane — mic audio -> whisper.cpp ->
chat-completions -> piper (L3, plan §5b + §3; voice/__init__.py
architecture, brain.lane 'transcript').

Position in the v4 speech protocol, next to the realtime WebSocket cloud
lane and the offline cascade:

    G1 mic frames -> EnergyGate -> session.upload_mic_audio (24 kHz)
        -> TranscriptBackend.submit_mic_audio (per-turn buffer)
        -> WhisperSubmit (composition — voice/cascade.py; 16 kHz WAV
           POST to the whisper.cpp /inference endpoint, 127.0.0.1:8082)
        -> CloudLLMClient.chat (composition — brain/agent/llm_cloud.py;
           REST /chat/completions, stream never, one frame per decision
           turn via the image_jpeg_b64 part)
        -> PiperTTS (composition — voice/cascade.py; 22050 Hz raw PCM)
        -> resampled to 24 kHz -> response.output_audio.delta events

The lane is a :class:`BrainBackend` (voice/session.py): the SAME
VoiceSession the realtime lane drives enforces laws L1-L10 above this
backend, and every event this backend fires is one of BACKEND_EVENTS with
the exact payload shape the session's handlers read (the session never
sees a new vocabulary).  The session NEVER calls :meth:`create_response`
for a fresh spoken utterance — exactly like the realtime lane, where the
server's VAD owns fresh turns — so this backend OWNS fresh-speech turn
starts and mirrors the server-VAD event sequence: a bare
``response.created`` (the session marks the turn "server" and — on THIS
lane, whose created precedes the chat call — runs the L10 frame seam
inline, step 3 below), then the transcription, then output items,
``response.done``.

--------------------------------------------------------------------------
THE TURN (server-kind: a fresh utterance owns its turn)
--------------------------------------------------------------------------

1. The mic pump delivers ONE energy-gated utterance per upload (app.py
   pump contract: ``energy.feed`` emits a completed utterance, hangover
   and all).  :meth:`submit_mic_audio` accumulates it into the per-turn
   buffer and kicks the turn worker.
2. The worker closes the buffer: 24 kHz -> 16 kHz (the exact 2:3 linear
   resample) -> WhisperSubmit.  A whisper TRANSPORT failure (request
   timeout / connection refused / HTTP error) means the local STT infra
   is dead: the lane fires ``error`` and the session surfaces the
   failover signal (director -> OFFLINE_CASCADE), where the cascade's
   whisper-decline law speaks for the human — never a silent skip.
   An EMPTY transcript (whisper heard nothing intelligible) is NOT a
   transport failure — the human simply was not heard: the lane stays
   up and speaks its own honest retry line (``EMPTY_TRANSCRIPT_RETRY``)
   as a normal turn, never silence, never a failover.
3. ``response.created`` (bare — the session sets kind "server"; the
   worker fires the created immediately before its ONE chat call, so
   the created-to-chat window exists on THIS lane that the realtime
   lane lacks, and the session runs the L10 frame seam inline (the
   transcript-lane capability branch,
   ``BrainBackend.created_precedes_model_call``): a fresh hook envelope
   at that moment is staged via :meth:`send_user_frame` and attaches to
   the very chat that follows; no hook / stale / invalid envelope ->
   ``frame_absent`` (the turn is imageless and never claims a frame for
   a response that was not instructed).  ``no_model_call`` createds
   (the empty-transcript retry — no chat follows) and createds fired
   while a query answer is owed (L9 phase 2 — cancelled right after)
   skip the seam: nothing will see a frame.
4. ``conversation.item.input_audio_transcription.completed`` — fired
   ONLY for the L9 query-answer case (the cancel-at-start branch: the
   session's ``_complete_query`` consumes it as the human's answer).  A
   NORMAL committed server turn fires NO transcription event: the lane
   already committed the utterance (step 5), and the only other session
   route would be the L1 drop — which would log a FALSE loss
   (``dropped utterance while gate shut``) for an utterance about to be
   answered (the lane fires created first, so the gate is always shut
   and the session would always take the drop branch).  The lane's
   commit IS the admission; ``user_text`` stays the realtime lane's
   forwarding path.
5. The transcript commits to the OpenAI-style history as the turn's user
   message and ONE chat call runs — one model call per turn, never
   more.  A frame attaches to that chat only when the seam staged one
   at the created (L10: one frame-hook call per decision turn);
   otherwise the turn is imageless and said so (``frame_absent``).
6. Tool calls -> ``response.output_item.done`` (the function_call shape:
   {item: {type: "function_call", name, call_id, arguments: json_str}})
   and the assistant ``tool_calls`` history message (the G1Agent
   ``_as_openai_call`` shape); prose -> the message item, the
   spoken-text events (``response.audio_transcript.delta`` / ``.done``)
   and local TTS audio (``response.output_audio.delta`` chunks of 0.25 s
   at 24 kHz — the session's L6 accounting rate — then
   ``response.output_audio.done``).  ``response.done`` closes the turn;
   the payload carries no usage (the cloud client reports none) so the
   cost ledger honestly reports "Not reported" (L8 — never guessed).

A session-initiated :meth:`create_response` (a tool continuation, an ask
turn, a query answer) closes the per-turn buffer the same way — the task
contract: "the turn is closed when the session starts a response
(create_response), then the buffer goes to WhisperSubmit".  The buffer
is consumed EXACTLY once: whichever of the transcription or a
TOOLS-ENABLED create_response closes the turn first, the other never
re-transcribes the same audio (the L9 answer, whose audio was already
transcribed, is therefore never doubled).  An ASK turn (tools disabled)
is the one documented exception: it has no mic input of its own, and any
buffered audio at that moment is the human's awaited answer (L9 phase 2)
— consuming it into the ask turn would strand the query, so the ask turn
never closes the buffer and the answer is transcribed by the server turn
that follows.

--------------------------------------------------------------------------
MID-BATCH UTTERANCE HOLD AND LANE RESET (round-3: C-H2 + C-H3)
--------------------------------------------------------------------------

A fresh utterance can land while a tool batch is still draining: the
assistant ``tool_calls`` message stands in the history with tool
messages still owed (the director is executing; the turn has not
closed).  Committing the utterance's user message THEN would break the
wire law (a user message inside the round -> provider 400) and speaking
over the batch would garble it — so the lane HOLDS the utterance
(:attr:`_held`; C-H2): nothing about it is committed or fired until
:meth:`_pump_held` delivers it — through the FULL standard server-turn
path (created -> transcription -> commit -> one chat call), exactly the
shape a live utterance would have — which happens only once the batch's
turn has closed (the tool results drained and their continuation or
answer turn ended), when the wire is legal again.  A held utterance is
never delivered into the L9 answer-capture window
(:attr:`_awaiting_query_answer` — set at an ASK create, i.e. a query
answer owed): the held audio predates the question, and delivering it
then would complete the query with the wrong text.

A lane failure while a batch is still pending (a whisper TRANSPORT
failure — the local STT is dead, the failover signal — or any failover
reset) must not leave the open round awaiting tool messages that can
never arrive: the lane closes its open unfinished round WHOLE
(:meth:`_close_open_round`, run before the error is surfaced; C-H3) —
the assistant ``tool_calls`` message is trimmed from the history, so
the next payload is wire-legal from a closed state and a LATE tool
result can never re-orphan the head: the session drops the stale answer
(its pending set was cleared at the reset) and the lane's orphan guard
refuses any tool message whose owning assistant is gone (never an
orphaned half of a pair).

--------------------------------------------------------------------------
BARGE-IN (L2 mirror)
--------------------------------------------------------------------------

A mic chunk that lands while a turn is active is the interruption: it is
kept in the buffer and flagged (with the CURRENT turn's epoch — a chunk
that lands between the last boundary check and the turn close belongs to
the NEXT utterance's buffer, never to an interruption of it); at the
next stage boundary the worker fires
``input_audio_buffer.speech_started`` and the session applies L2
(truncate + cancel before tool results; ignore after tool results).
:meth:`cancel_response` sets its flag synchronously (thread-safe), so the
abort is honored at the very next boundary even while the worker is
blocked in a stage call — barge-in latency is at most the current
stage's remainder (STT / one cloud call / one piper synthesis, each
bounded by its own timeout), a documented deviation from the realtime
lane's instant server-side cancel.  The aborted turn fires
``response.done`` {status: "cancelled"} (the session's done handler owns
the queue release) and a fresh server turn auto-starts from the buffered
audio (the server-VAD mirror).  A cancel that lands AT the
``response.created`` fire means the session is waiting for a query
answer (L9 phase 2): the transcription that follows IS the answer, the
aborted turn closes with a cancelled done, and the session's own answer
response drives — never a second turn on the same audio.
:meth:`truncate_response` is a documented dead end on this lane (the REST
wire has no item rewrite): it returns False with a logged warning and
the barge-in proceeds cancel-only, exactly like the realtime lane's
allowlist refusal.

--------------------------------------------------------------------------
CONCURRENCY MODEL
--------------------------------------------------------------------------

All event firing happens on ONE dedicated daemon worker thread (the
"turn pump", started by :meth:`start`); send methods only mutate state
under a lock and enqueue operations.  The session's handlers therefore
run INLINE on the turn worker thread — session state is never mutated
concurrently — mapping the realtime lane's loop-thread confinement onto
the turn worker thread.  A mic op queued mid-turn is processed at the
next turn boundary (bounded by the stage timeouts); the barge-in signal
is fired from the stage-boundary check, not from the queue drain.

--------------------------------------------------------------------------
TOOLS (documented deviation from the task's "flat" wording)
--------------------------------------------------------------------------

The task says "build the flat tool list via adapters flattening for the
request 'tools'"; the REST wire, however, is the NESTED Chat-Completions
shape: CloudLLMClient's skeleton mode validates ``spec["function"]
["name"]`` (valid_names) and tests/test_llm_cloud.py pins the nested
wire shape — flat specs silently degrade skeleton mode to native.  So the
constructor takes the NESTED registry specs (what ToolRegistry.get_specs
and llm_cloud.chat consume), and :meth:`flatten_specs` (voice/adapters.py
— the realtime session.update shape) is applied eagerly to validate them
and expose :attr:`tools_flat` for realtime-shape parity and diagnostics
(malformed specs raise ValueError at construction: a loud build, never a
silently broken tool list).  ``create_response(tools=False)`` sends
``tools=[]`` — the provider-callable equivalent of the realtime lane's
override ``tool_choice: "none"`` (the client never sends tool_choice);
with tools present, ``auto`` is the provider default.

--------------------------------------------------------------------------
COST AND LATENCY (bake-off measurement targets)
--------------------------------------------------------------------------

Voice is FREE on this lane: STT and TTS are local subprocesses/services
(whisper.cpp, piper) and the only billed resource is the cloud model's
token usage — zero voice cost by construction.  The latency stack to
measure in the bake-off (documented here as a target, not yet measured):
STT (one whisper /inference POST, ``whisper_submit_timeout_s``) + TTFT
(the cloud endpoint's first-token time inside ``CloudLLMClient.timeout_s``
— the client never streams, so TTFT == the full round trip) + TTS (one
piper synthesis, ``piper_tts_timeout_s``) + audio delivery (the session's
0.6 s lead buffer, L6).  Every stage is bounded by its own timeout — the
cascade L1 style (15 s whisper, 20 s piper, 120 s cloud): no unbounded
wait anywhere, no silent hang.

Module-level import surface is stdlib-only plus first-party modules that
are themselves stdlib-only at import (voice/cascade.py, voice/adapters.py,
voice/session.py, brain/agent/llm.py): third-party dependencies load
lazily inside methods, so importing this module — and ``import voice`` —
succeeds dep-free on any laptop (voice/__init__.py pattern).
"""

from __future__ import annotations

import array
import asyncio
import base64
import inspect
import json
import logging
import queue
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from brain.agent.llm import ChatTransportError
from voice.adapters import flatten_specs
from voice.cascade import (
    PIPER_RETRIES,
    PIPER_TTS_TIMEOUT_S,
    PiperError,
    PiperTTS,
    WHISPER_LAUNCH_TIMEOUT_S,
    WHISPER_SUBMIT_TIMEOUT_S,
    WhisperSubmit,
)
from voice.session import AUDIO_SAMPLE_RATE, BrainBackend

log = logging.getLogger("voice.transcript")

# ---------------------------------------------------------- module constants

#: mic chunk rate, 24 kHz PCM16 mono
MIC_RATE = AUDIO_SAMPLE_RATE
#: 16 kHz PCM16 mono for WhisperSubmit
STT_RATE = 16000
#: 0.25 s at 24 kHz = 12000 bytes per delta chunk
#: chunk and lead math both divide by AUDIO_SAMPLE_RATE
AUDIO_DELTA_S = 0.25
#: history cap of 14 messages
#: the same limit the agent lane enforces
MAX_KEPT_HISTORY_MESSAGES = 14
#: 60 s cap: 32 KB/s -> 1_920_000 bytes
#: a long tool run would ride one POST and the 15 s submit
#: timeout would drop it whole, so the oldest audio goes first
MIC_BUFFER_CAP_SECONDS = 60
MIC_BUFFER_CAP_BYTES = MIC_BUFFER_CAP_SECONDS * STT_RATE * 2
#: spoken when whisper hears nothing intelligible
#: a normal turn, never silence: failover is
#: reserved for transport failure
EMPTY_TRANSCRIPT_RETRY = "I did not catch that — could you say it again?"


# -------------------------------------------------------------------------------
# TranscriptBackend — the BrainBackend that drives the REST cloud lane
# -------------------------------------------------------------------------------


class TranscriptBackend(BrainBackend):
    """The REST cloud lane's :class:`BrainBackend` (module docstring:
    turn flow, barge-in, concurrency, tools shape, cost/latency).

    Wraps a chat client (the CloudLLMClient family) and composes the
    cascade's :class:`WhisperSubmit` and :class:`PiperTTS` boundaries by
    composition — never copied.  All boundaries are injectable (``submit``
    / ``tts``), so tests hand in fakes exactly like OfflineChain.
    """

    # response.created fires immediately before the single chat
    # call, leaving a window in which the frame seam can run
    created_precedes_model_call = True

    def __init__(
        self,
        client: Any,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        submit: Optional[Callable[[bytes], str]] = None,
        tts: Optional[Callable[[str], Tuple[bytes, int]]] = None,
        whisper_launch_timeout_s: float = WHISPER_LAUNCH_TIMEOUT_S,
        whisper_submit_timeout_s: float = WHISPER_SUBMIT_TIMEOUT_S,
        piper_tts_timeout_s: float = PIPER_TTS_TIMEOUT_S,
    ) -> None:
        """Validate the L1 stage timeouts, validate+flatten the tool
        specs, and wire the boundaries.  Pure: no I/O (WhisperSubmit /
        PiperTTS construct lazily and perform no I/O here)."""
        if client is None:
            raise ValueError("TranscriptBackend requires a chat client")
        bad_timeouts = [
            name
            for name, value in (
                ("whisper_launch_timeout_s", whisper_launch_timeout_s),
                ("whisper_submit_timeout_s", whisper_submit_timeout_s),
                ("piper_tts_timeout_s", piper_tts_timeout_s),
            )
            if not value or value <= 0
        ]
        if bad_timeouts:
            raise ValueError(
                "stage timeouts must be positive (no unbounded waits): "
                + ", ".join(bad_timeouts)
            )
        self._client = client
        #: nested chat-completions tool specs
        self._tools = [dict(spec) for spec in (tools or [])]
        #: flat realtime-shape specs
        #: flattening here makes a malformed spec raise at construction
        self.tools_flat = flatten_specs(self._tools)
        self._submit = (
            submit
            if submit is not None
            else WhisperSubmit(
                launch_timeout_s=whisper_launch_timeout_s,
                submit_timeout_s=whisper_submit_timeout_s,
            )
        )
        self._tts = (
            tts if tts is not None else PiperTTS(tts_timeout_s=piper_tts_timeout_s)
        )

        # -- chat history
        self._history: List[Dict[str, Any]] = []
        #: call_id -> tool name for the role:"tool" messages
        self._call_names: Dict[str, str] = {}

        # -- frames
        #: newest jpeg b64 frame; the next chat consumes it once,
        #: and none means the turn is imageless
        self._staged_frame: Optional[str] = None

        # -- held speech
        #: a user message mid-batch breaks the wire law (400),
        #: so speech is held and delivered after the batch closes
        self._held: List[Dict[str, Any]] = []
        #: set by an ask create (tools=False), cleared by the next
        #: tools-enabled one: held speech predates the question and
        #: must never land in its answer window
        self._awaiting_query_answer = False

        # -- mic buffer
        self._mic_buffer = bytearray()
        #: the turn's audio is taken exactly once; only the first
        #: closer transcribes it
        self._buffer_consumed = False

        # -- turn state
        self._turn_active = False
        self._cancel_requested = False
        self._barge_pending = False
        #: per-turn epoch and barge epoch: a late chunk
        #: belongs to the next utterance
        self._turn_epoch = 0
        self._barge_epoch = 0
        self._closed = False
        self._seq = 0  # item_id counter

        self._lock = threading.Lock()
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, List[Callable[[str, dict], Any]]] = {}

    # --------------------------------------------------------------- start & stop

    def start(self) -> None:
        """Bring the lane up: start the turn worker thread (daemon)."""
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._loop, name="transcript-turn", daemon=True
            )
            self._thread = thread
        thread.start()
        log.debug("transcript backend started (turn worker thread)")

    async def run_forever(self) -> None:
        """Async no-op — the app MUST schedule this task (the lane's
        ``run_forever`` law, app.py: the task's return is read as the
        lane's loud-exit signal).  The transcript lane's turn pump is
        the dedicated worker thread ALREADY running (``start()``), so
        this task only keeps the app's task alive and must NEVER
        duplicate the pump — it sleeps forever."""
        while True:
            await asyncio.sleep(3600)

    def close(self) -> None:
        """Tear the lane down: stop the worker and close the whisper
        server subprocess THIS instance launched (composition; never
        raises)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._q.put(("close",))
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        submit = getattr(self._submit, "close", None)
        if callable(submit):
            try:
                submit()
            except Exception as exc:  # noqa: BLE001 - cleanup must not blow up
                log.debug("transcript whisper submit close raised: %s", exc)

    def on_event(self, name: str, callback: Callable[[str, dict], Any]) -> None:
        """Subscribe to lane events.  Callbacks run INLINE on the turn
        worker thread (the session's handlers are sync — this IS the
        confinement model, module docstring); an awaitable return is
        scheduled as a task on the running loop, dropped outside one."""
        self._handlers.setdefault(name, []).append(callback)

    def health_probe(self) -> bool:
        """The lane-aware probe the app's director wiring expects: a
        SYNC, low-cost connectivity check that reaches the network,
        bounded to ~2 s by the delegated probe's own timeout (the
        parallel wiring adds ``CloudLLMClient.probe()``).  Delegates via
        getattr: a client without a probe reports False (the lane is not
        healthy until the probe exists); a raising probe counts as a
        failure — never raises (the director contract)."""
        probe = getattr(self._client, "probe", None)
        if not callable(probe):
            return False
        try:
            return bool(probe())
        except Exception as exc:  # noqa: BLE001
            log.warning("transcript lane: health probe raised: %s", exc)
            return False

    def _fire(self, name: str, payload: Dict[str, Any]) -> None:
        """Dispatch one BACKEND_EVENTS event to listeners (must run on
        the turn worker thread — the session's handlers mutate session
        state and must never run concurrently).  Best-effort teardown
        (D4c): after :meth:`close` has returned the worker emits
        nothing — close() joins only briefly, so a worker mid-chat when
        close() ran must not keep firing into a torn-down session."""
        if self._closed:
            return
        for callback in list(self._handlers.get(name, ())):
            try:
                result = callback(name, payload)
            except Exception:  # noqa: BLE001
                log.exception("transcript callback for %r failed", name)
                continue
            if inspect.isawaitable(result):
                try:
                    asyncio.ensure_future(result)
                except RuntimeError:  # no loop, drop it
                    log.debug(
                        "transcript callback %r returned awaitable outside a loop", name
                    )

    # --------------------------------------------------------- the send methods

    def submit_mic_audio(self, chunk: bytes) -> bool:
        """Accumulate one 24 kHz PCM16 mic chunk into the per-turn buffer
        and kick the turn worker.  The pump delivers one energy-gated
        utterance per upload; a chunk that lands while a turn is active
        is the barge-in signal (flagged; the worker fires
        ``input_audio_buffer.speech_started`` at its next stage
        boundary).  Returns True when accepted for delivery."""
        if self._closed or not chunk:
            return False
        with self._lock:
            self._mic_buffer += bytes(chunk)
            overflow = len(self._mic_buffer) - MIC_BUFFER_CAP_BYTES
            if overflow > 0:
                # cap the POST: a long tool run would ride one
                # POST and the 15 s timeout would drop it whole
                del self._mic_buffer[:overflow]
                log.warning(
                    "transcript lane: per-turn mic buffer capped at %d "
                    "bytes (%d s of speech) — dropped %d bytes of "
                    "overflow (oldest audio)",
                    MIC_BUFFER_CAP_BYTES, MIC_BUFFER_CAP_SECONDS, overflow,
                )
            self._buffer_consumed = False
            if self._turn_active:
                self._barge_pending = True
                self._barge_epoch = self._turn_epoch
        self._q.put(("mic",))
        return True

    def send_user_text(self, text: str) -> bool:
        """Inject a text user turn (the L9 answer attachment)."""
        if self._closed:
            return False
        self._q.put(("user_text", str(text)))
        return True

    def send_user_frame(self, jpeg_b64: str) -> bool:
        """L10: store the LATEST frame (fresh gateway-cache image, jpeg
        base64); the NEXT chat call attaches it — one frame per decision
        turn, no replay: the chat call consumes the store exactly once,
        and a turn without a stored frame attaches nothing (no fresh
        frame must be representable) while its text observation still
        travels."""
        if self._closed:
            return False
        with self._lock:
            self._staged_frame = str(jpeg_b64)
        return True

    def send_tool_result(self, call_id: str, output_text: str) -> bool:
        """Answer one function call (a role:"tool" history message, in
        order behind the assistant ``tool_calls`` message that owned it)."""
        if self._closed:
            return False
        self._q.put(("tool_result", str(call_id), str(output_text)))
        return True

    def create_response(
        self, *, tools: bool = True, instruction: Optional[str] = None
    ) -> bool:
        """Close the per-turn mic buffer (task contract: the turn is
        closed when the session starts a response) and run ONE chat call
        — the session's tool continuation / ask turn / query answer."""
        if self._closed:
            return False
        self._q.put(("create", bool(tools), instruction))
        return True

    def cancel_response(self) -> bool:
        """Cancel the in-flight turn (barge-in, L2).  The flag is set
        synchronously (thread-safe) so the worker honors the abort at
        its next stage boundary even while blocked in a stage call; the
        queued op only drains the accounting."""
        if self._closed:
            return False
        with self._lock:
            self._cancel_requested = True
        self._q.put(("cancel",))
        return True

    def truncate_response(self, item_id: str, audio_end_ms: int) -> bool:
        """The REST wire has no item rewrite: truncate is a documented
        dead end on this lane — returns False with a logged warning and
        the barge-in proceeds cancel-only (the RealtimeBackend contract)."""
        log.warning(
            "conversation.item.truncate is a documented dead end on the "
            "transcript lane (no item rewrite on the REST wire; item_id=%r, "
            "audio_end_ms=%d) — barge-in proceeds with response.cancel alone",
            item_id, max(0, int(audio_end_ms)),
        )
        return False

    # ------------------------------------------------------------- the turn worker

    def _loop(self) -> None:
        """The turn pump: process operations one at a time; every event
        fires from THIS thread."""
        while True:
            try:
                op = self._q.get()
            except Exception as exc:  # noqa: BLE001
                log.error("transcript worker queue failed: %s", exc)
                return
            kind = op[0]
            if kind == "close":
                return
            if self._closed:
                # closed: drain, emit nothing
                continue
            try:
                if kind == "mic":
                    self._on_mic()
                elif kind == "create":
                    self._on_create(bool(op[1]), op[2])
                elif kind == "cancel":
                    self._on_cancel_op()
                elif kind == "tool_result":
                    self._on_tool_result(str(op[1]), str(op[2]))
                elif kind == "user_text":
                    self._on_user_text(str(op[1]))
                else:
                    log.debug("transcript worker: unknown op %r", kind)
            except Exception:  # noqa: BLE001
                log.exception("transcript worker op %r failed", kind)
            try:
                # deliver held speech after every op
                self._pump_held()
            except Exception:  # noqa: BLE001
                log.exception("transcript worker held-utterance delivery failed")

    def _on_mic(self) -> None:
        """A mic kick.  Mid-turn kicks are the barge-in signal (the
        session applies L2); otherwise a fresh server-kind turn starts
        from the buffered utterance."""
        with self._lock:
            active = self._turn_active
        if active:
            self._seq += 1
            self._fire("input_audio_buffer.speech_started", {
                "item_id": "utterance_{}".format(self._seq),
                "audio_start_ms": 0,
            })
            return
        with self._lock:
            if not self._mic_buffer:
                log.debug(
                    "transcript lane: mic op with no buffered audio — dropped"
                )
                return
        self._run_server_turn()

    def _on_cancel_op(self) -> None:
        """Drain a cancel op.  The flag was already set synchronously by
        cancel_response; a stale cancel (no turn active) is cleared so it
        can never kill the NEXT turn."""
        with self._lock:
            if not self._turn_active:
                self._cancel_requested = False
                log.debug("transcript lane: stale cancel drained")

    def _on_tool_result(self, call_id: str, output_text: str) -> None:
        """Commit one function_call_output to the history — a role:"tool"
        message, unique per call_id (the provider law: each call answered
        EXACTLY once, and each tool message immediately follows the
        assistant ``tool_calls`` message that owned the call).  When a
        message for this call_id ALREADY exists (the L9 placeholder
        "OK: awaiting…", queued in the ask turn), it is REPLACED IN
        PLACE: the content is overwritten and the position kept, so the
        answer's tool message stays contiguous behind its assistant
        tool_calls message and the provider never sees a duplicate
        call_id — the wire violation that 400'd every query_user round.
        Only when no prior message exists for the call_id is a new
        message appended (a normal tool round)."""
        name = self._call_names.get(call_id, "")
        for message in self._history:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == call_id
            ):
                message["content"] = output_text
                if name:
                    message["name"] = name
                log.debug(
                    "transcript lane: tool message for call_id %r replaced "
                    "in place (the L9 answer closes the placeholder)",
                    call_id,
                )
                return
        message: Dict[str, Any] = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output_text,
        }
        if name:
            message["name"] = name
        # orphan guard: a tool message without its owning
        # tool_calls message would poison the wire (400)
        if not any(
            m.get("role") == "assistant"
            and any(str(c.get("id") or "") == call_id
                    for c in (m.get("tool_calls") or ()))
            for m in self._history
        ):
            log.warning(
                "transcript lane: refusing to append the tool message for "
                "call_id %r — no assistant tool_calls message in the "
                "history owns it (never an orphaned tool message)",
                call_id,
            )
            return
        self._append_history(message)

    def _on_user_text(self, text: str) -> None:
        """Commit an injected text user turn (the L9 answer attach)."""
        self._append_history({"role": "user", "content": text})

    # ------------------------------------- scanning open rounds + the held pump

    def _open_round_origin(self) -> Optional[int]:
        """The history index of the OPEN tool round's assistant
        ``tool_calls`` message, or None when every round is closed.
        Walks BACK from the tail over the role:"tool" messages; the
        first non-tool message is the round's owner (the wire law: a
        round's tool messages sit directly behind its assistant
        ``tool_calls`` message, so only the LAST round can be open and
        only its owner can sit behind the trailing tool messages).  The
        round is OPEN exactly when its call_ids are not all covered by
        the trailing tool messages — its results are still owed, i.e.
        the batch is draining."""
        owed = set()
        for i in range(len(self._history) - 1, -1, -1):
            message = self._history[i]
            role = str(message.get("role") or "")
            if role == "tool":
                call_id = message.get("tool_call_id")
                if call_id is not None:
                    owed.add(call_id)
                continue
            if role == "assistant" and message.get("tool_calls"):
                calls = {
                    str(call.get("id") or "")
                    for call in (message.get("tool_calls") or ())
                }
                if calls - owed:
                    return i
            return None
        return None

    def _close_open_round(self, reason: str) -> None:
        """Drop the history's OPEN tool round WHOLE (S4: a lane error —
        whisper failure / failover — while a batch was pending).  The
        round's assistant ``tool_calls`` message and the tool messages
        already committed behind it are deleted together, so the
        surviving history holds no round awaiting results that will
        never come and no half-round (never an assistant without its
        tool messages — and a LATE tool result for a dropped call is
        refused by the orphan guard, never appended).  ``reason`` names
        the reset for the log.  No-op when no round is open."""
        origin = self._open_round_origin()
        if origin is None:
            return
        log.warning(
            "transcript lane: dropping the open tool round at history "
            "index %d (%s)",
            origin, reason,
        )
        del self._history[origin:]

    def _pump_held(self) -> None:
        """Deliver held utterances (S3) after the batch that held them
        has fully closed.  Runs from the worker at the END of every op;
        a delivery replays the SAME full server-turn event shape a live
        utterance would have had (:meth:`_start_transcribed_server_turn`
        — or :meth:`_speak_retry` for a held empty-transcript retry),
        so the session answers the utterance exactly as if it had been
        heard when spoken — never dropped, never wire-illegal.

        The delivery is gated on ALL of (checked fresh per delivery):
          * not closed, no turn active (never mid-turn),
          * a held item exists,
          * the op FIFO is EMPTY — never preempt a drained batch's
            queued follow-up op: the drain's single decision/ask create
            was queued synchronously by the SAME director call that
            queued the batch's last tool result, so by the time the
            worker has committed that result and reached this pump the
            follow-up is queued (or the turn already closed with no
            follow-up) — delivering before it would answer the held
            utterance out of conversational order,
          * no OPEN tool round (a drained batch whose response returned
            further calls has re-opened the round — deliver only when
            it too has closed),
          * the session is NOT awaiting a query answer
            (``_awaiting_query_answer`` — set at an ask create): a held
            utterance predates the question, and delivering it into the
            L9 answer-capture window would complete the query with the
            wrong text.
        """
        while True:
            with self._lock:
                if self._closed or self._turn_active or not self._held:
                    return
                if not self._q.empty():
                    # FIFO: follow-up goes first
                    return
            if self._awaiting_query_answer:
                return
            if self._open_round_origin() is not None:
                return
            item = self._held.pop(0)
            if "retry" in item:
                log.info(
                    "transcript lane: delivering the held empty-transcript "
                    "retry (%d held item(s) remain)",
                    len(self._held),
                )
                self._speak_retry(str(item["retry"]))
            else:
                transcript = str(item.get("transcript") or "").strip()
                if not transcript:
                    continue
                log.info(
                    "transcript lane: delivering the held utterance %r "
                    "(%d held item(s) remain)",
                    transcript[:60], len(self._held),
                )
                self._start_transcribed_server_turn(transcript)

    # ----------------------------------------------------------- the turn pipeline

    def _run_server_turn(self) -> None:
        """A fresh utterance's turn (the server-VAD mirror): close the
        mic buffer -> whisper -> then either the S3 hold (a tool round
        was OPEN when the utterance arrived — committing it mid-batch
        would break the wire law, so it is held and delivered after the
        batch closes), the empty-transcript retry, or
        :meth:`_start_transcribed_server_turn` (the created fire — the
        session runs the L10 frame seam at a bare created via the
        lane-capability branch — the transcription, one chat call)."""
        pcm16 = self._close_buffer()
        if pcm16:
            try:
                transcript = str(self._submit(pcm16) or "").strip()
            except Exception as exc:  # noqa: BLE001
                # STT is dead: drop the open round first so a late
                # result cannot re-orphan the head, then surface
                # the failover signal
                self._close_open_round(
                    "whisper failure (STT dead) reset the lane"
                )
                log.error("transcript lane: whisper stage failed: %s", exc)
                self._fire("error", {
                    "message": "whisper stage failed: {}".format(exc),
                })
                return
        else:
            transcript = ""
        if self._open_round_origin() is not None:
            # mid-batch: a user message here would break the
            # wire law (400) and garble the round, so hold it
            if transcript:
                self._held.append({"transcript": transcript})
                reason = "a tool round was open (mid-batch)"
            else:
                self._held.append({"retry": EMPTY_TRANSCRIPT_RETRY})
                reason = "a tool round was open (mid-batch); nothing heard"
            log.info(
                "transcript lane: utterance held (%s) — %d held item(s) "
                "awaiting the batch close",
                reason, len(self._held),
            )
            return
        if not transcript:
            # empty transcript on a healthy transport is not a
            # failover: speak the retry line and stay up
            log.warning(
                "transcript lane: whisper heard nothing — speaking the "
                "retry line (no failover)"
            )
            self._speak_retry(EMPTY_TRANSCRIPT_RETRY)
            return
        self._start_transcribed_server_turn(transcript)

    def _start_transcribed_server_turn(self, transcript: str) -> None:
        """The post-whisper half of a fresh server turn, shared by a
        live utterance (:meth:`_run_server_turn`) and a HELD utterance
        (:meth:`_pump_held` — the delivery replays the SAME event shape
        a live turn would have, so the session cannot tell the audio
        arrived mid-batch): turn state up -> the bare
        ``response.created`` (the session's created handler runs the
        lane's L10 frame seam here — this lane's capability, the
        created-to-chat window) -> the L9-cancel check -> (only when
        the turn was cancelled at start: the transcription — the
        session's query-answer capture — before the user message) ->
        the user message -> ONE chat call.  A NORMAL turn fires NO
        transcription: the commit is the admission, and a transcription
        event would arrive gate-shut and log a FALSE loss (2026-09-04
        fix — the session's drop branch is the realtime lane's path)."""
        with self._lock:
            self._cancel_requested = False
            self._turn_active = True
            self._turn_epoch += 1
        self._fire("response.created", {})
        with self._lock:
            cancelled_at_start = self._cancel_requested
        if cancelled_at_start:
            # cancelled at the created fire: the session awaits a
            # query answer and this speech is it. Close the aborted
            # turn first; the transcription handler opens its own response
            self._turn_active = False
            self._fire("response.done", {"response": {"status": "cancelled"}})
            self._fire("conversation.item.input_audio_transcription.completed", {
                "transcript": transcript,
            })
            return
        # no event on the normal path: the commit
        # below is the admission
        self._append_history({"role": "user", "content": transcript})
        self._run_model_call(tools_enabled=True, instruction=None)

    def _on_create(
        self, tools_enabled: bool, instruction: Optional[str]
    ) -> None:
        """A session-initiated response (tool continuation / ask turn /
        query answer).  A TOOLS-ENABLED turn closes the per-turn mic
        buffer — the task contract: "the turn is closed when the session
        starts a response (create_response)" — then ONE chat call.  An
        ASK turn (tools disabled) NEVER closes the buffer: its turn has
        no mic input, and any buffered audio is the human's awaited
        answer (L9 phase 2) — consuming it into the ask turn would
        strand the query; the answer is transcribed by the server turn
        that follows (documented deviation).  A whisper hiccup on a
        tools-enabled turn must not strand the session's own response (a
        tool batch or an answer): the transcript is a bonus, so it warns
        and proceeds without it."""
        # an ask create opens the answer window; held speech
        # predates the question and must not land here
        with self._lock:
            self._awaiting_query_answer = not tools_enabled
        transcript = ""
        if tools_enabled:
            pcm16 = self._close_buffer()
            if pcm16:
                try:
                    transcript = str(self._submit(pcm16) or "").strip()
                except Exception as exc:  # noqa: BLE001 - see docstring
                    log.warning(
                        "transcript lane: whisper stage failed on a session "
                        "turn (%s) — proceeding without the mic transcript",
                        exc,
                    )
                    transcript = ""
        if transcript:
            self._append_history({"role": "user", "content": transcript})
        with self._lock:
            self._cancel_requested = False
            self._turn_active = True
            self._turn_epoch += 1
        self._fire("response.created", {})
        with self._lock:
            cancelled_at_start = self._cancel_requested
        if cancelled_at_start:
            # cancel landed mid-op or mid-whisper: close with no
            # model call, and the buffer goes to the next turn
            self._turn_active = False
            self._fire("response.done", {"response": {"status": "cancelled"}})
            self._restart_if_barge()
            return
        self._run_model_call(tools_enabled=tools_enabled, instruction=instruction)

    def _run_model_call(
        self, *, tools_enabled: bool, instruction: Optional[str]
    ) -> None:
        """ONE chat call for the turn (never more): commit the result to
        the history, fire the session's output events, speak the prose.
        A transport failure, a dropped call, or an empty result (never
        an empty response) fires ``error`` -> the session's failover
        signal.

        When prose AND tool calls arrive together, TWO consecutive
        assistant messages are committed: the prose message FIRST, then
        the tool_calls message — a legal shape (two assistant messages
        in a row), and the tool messages that follow answer the
        tool_calls message they immediately follow.  The tool_calls
        message therefore naturally precedes its tool messages; the
        prose message rides in the same turn before both."""
        messages = self._history_messages(instruction)
        with self._lock:
            frame, self._staged_frame = self._staged_frame, None
        # frames: decision turns only, never an ask; the
        # store is cleared either way so nothing replays
        image = frame if tools_enabled else None
        tools = self._tools if tools_enabled else []
        try:
            result = self._client.chat(
                messages, tools, image_jpeg_b64=image
            )
        except ChatTransportError as exc:
            log.error("transcript lane: cloud transport failed: %s", exc)
            self._fail_turn("cloud transport failed: {}".format(exc))
            return
        except Exception as exc:  # noqa: BLE001 - whatever it throws
            log.error("transcript lane: cloud chat failed: %s", exc)
            self._fail_turn("cloud chat failed: {}".format(exc))
            return
        # a barge-in may have landed during the call
        self._check_barge()
        if self._finish_turn():
            return
        if result.dropped:
            log.error("transcript lane: cloud chat dropped (single-flight)")
            self._fail_turn(
                "cloud chat dropped (a previous call was in flight)"
            )
            return
        text = (result.text or "").strip() if result.text else ""
        calls = result.tool_calls or []
        if not text and not calls:
            # an empty response (no prose, no call) is a failure
            # and takes the failover path
            log.error(
                "transcript lane: the model returned neither prose nor a "
                "tool call (never an empty response)"
            )
            self._fail_turn(
                "the model returned neither prose nor a tool call "
                "(never an empty response)"
            )
            return
        item_id = "transcript_item_{}".format(self._seq)
        self._seq += 1
        if text:
            self._append_history({"role": "assistant", "content": text})
            self._fire("response.output_item.done", {
                "item": {
                    "type": "message",
                    "content": [{"type": "text", "text": text}],
                },
            })
        if calls:
            assistant: Dict[str, Any] = {
                "role": "assistant", "content": None, "tool_calls": [],
            }
            for n, call in enumerate(calls):
                call_id = str(call.id or "call_{}".format(n))
                name = str(call.name or "")
                try:
                    arguments_json = json.dumps(call.arguments)
                except (TypeError, ValueError):
                    arguments_json = "{}"
                assistant["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments_json},
                })
                self._call_names[call_id] = name
                self._fire("response.output_item.done", {
                    "item": {
                        "type": "function_call",
                        "name": name,
                        "call_id": call_id,
                        "arguments": arguments_json,
                    },
                })
            if assistant["tool_calls"]:
                self._append_history(assistant)
        if text:
            # prose: log it, then speak it with local TTS
            log.info("transcript lane prose reply: %s", text)
            self._fire("response.audio_transcript.delta", {
                "delta": text, "item_id": item_id,
            })
            self._fire("response.audio_transcript.done", {
                "transcript": text, "item_id": item_id,
            })
            self._speak(text)
        if self._finish_turn():
            return
        self._turn_active = False
        self._fire("response.done", {"response": {"status": "completed"}})

    def _check_barge(self) -> None:
        """Fire the barge-in signal for audio that arrived mid-turn (L2),
        at a stage boundary.  The session's inline handler decides
        truncate + cancel (before tool results) or ignore (after tool
        results — the audio then simply joins the next turn's buffer).

        Per-turn epoch guard (D4): a chunk that landed between the LAST
        boundary check and the turn close set ``_barge_pending`` while
        the turn was still active — without the guard it would fire at
        the NEXT turn's first boundary and the session would cancel a
        fresh turn, silently dropping the new utterance.  The flag only
        fires when its epoch matches the CURRENT turn: a late chunk
        belongs to the next turn's buffer, not to an interruption."""
        with self._lock:
            pending = self._barge_pending
            self._barge_pending = False
            epoch = self._barge_epoch
        if pending and epoch == self._turn_epoch:
            self._seq += 1
            self._fire("input_audio_buffer.speech_started", {
                "item_id": "utterance_{}".format(self._seq),
                "audio_start_ms": 0,
            })

    def _finish_turn(self) -> bool:
        """Close the active turn's stage work at a boundary.  Returns
        True when a barge-in cancel aborted the turn (the caller must
        stop): ``response.done`` {status: "cancelled"} closes the
        session's accounting (its done handler owns the queue release)
        and a fresh server turn auto-starts from the buffered audio."""
        with self._lock:
            cancelled = self._cancel_requested
            self._cancel_requested = False
        if not cancelled:
            return False
        self._turn_active = False
        log.info("transcript lane: turn cancelled (barge-in)")
        self._fire("response.done", {"response": {"status": "cancelled"}})
        self._restart_if_barge()
        return True

    def _restart_if_barge(self) -> None:
        """After a barge-in abort, auto-start a fresh server turn from
        the buffered audio (the server-VAD mirror: the interruption IS
        the next utterance).  Nothing restarts when the buffer was
        already consumed — the L9 answer cancel, whose audio became the
        transcription, never doubles the turn."""
        with self._lock:
            if self._closed or self._turn_active:
                return
            if not self._mic_buffer or self._buffer_consumed:
                return
        self._run_server_turn()

    def _fail_turn(self, message: str) -> None:
        """A lane failure (whisper dead, cloud transport, dropped call,
        empty model result): surface the failover signal — the only
        backend failure event the session understands ("error" -> the
        director's hard failover -> OFFLINE_CASCADE).  No
        ``response.done``: the session's error handler resets the turn
        state itself.  S4: an open tool round (a batch whose results
        were still pending when the lane failed) is dropped whole FIRST
        — the reset must not leave a round awaiting tool messages that
        will never arrive, and a LATE tool result must never re-orphan
        the head (the orphan guard refuses it downstream)."""
        self._close_open_round("lane failure ({}) reset the lane".format(
            (message or "")[:60]
        ))
        with self._lock:
            self._cancel_requested = False
            self._barge_pending = False
        self._turn_active = False
        self._fire("error", {"message": message})

    def _close_buffer(self) -> bytes:
        """Close the per-turn mic buffer: take its audio (24 kHz), mark
        it consumed, resample to 16 kHz for WhisperSubmit.  Called at
        turn close — the transcription and a session-initiated
        create_response race on the same buffer and only the FIRST
        consumes it (the L9 answer is never transcribed twice)."""
        with self._lock:
            if self._buffer_consumed or not self._mic_buffer:
                self._mic_buffer = bytearray()
                self._buffer_consumed = True
                return b""
            pcm24 = bytes(self._mic_buffer)
            self._mic_buffer = bytearray()
            self._buffer_consumed = True
        return _resample_pcm16(pcm24, MIC_RATE, STT_RATE)

    def _history_messages(self, instruction: Optional[str]) -> List[Dict[str, Any]]:
        """A copy of the history with the per-response instruction
        prepended as a system message.  Per-response instructions do NOT
        persist (the realtime lane's response-level override semantics)."""
        out = [dict(m) for m in self._history]
        if instruction:
            out.insert(0, {"role": "system", "content": str(instruction)})
        return out

    # --------------------------------------------------- keeping history in bounds

    def _append_history(self, message: Dict[str, Any]) -> None:
        """Append one history message, then enforce the cap.  Every
        append goes through here so the window can never grow without
        bound (it is re-sent every turn; an uncapped window overflows
        the provider window -> 400 -> failover)."""
        self._history.append(message)
        self._trim_history()

    def _trim_history(self) -> None:
        """Enforce the history cap (MAX_KEPT_HISTORY_MESSAGES, the
        agent.py law): drop the oldest whole CLOSED units first — never
        a bare prefix that would slice a round in half.  Index 0 is the
        conversation anchor (the oldest retained message — on this lane
        the first utterance; never the head of a droppable round) and is
        never dropped.  The current OPEN round (an assistant
        ``tool_calls`` whose tool messages have not all arrived) is
        never dropped.  A tool round is dropped only as a WHOLE unit, in
        BOTH directions: an assistant ``tool_calls`` message is never
        retained without its tool messages, and a role:"tool" message is
        never retained without its owning assistant.  Warns on every
        drop."""
        if len(self._history) <= MAX_KEPT_HISTORY_MESSAGES:
            return
        need = len(self._history) - MAX_KEPT_HISTORY_MESSAGES  # over the cap
        # start at 1: index 0 is the anchor and is never dropped.
        # step over whole closed units; tool results arrive before
        # the next user message, so only the last round can be open
        drop_end = 1  # legal drop cutoff
        ends = []  # cut points
        owed = set()  # owed call_ids
        round_origin = 1  # round start
        for i in range(1, len(self._history)):
            message = self._history[i]
            role = str(message.get("role") or "")
            if role == "tool":
                if not owed:
                    break  # no owner, bail
                owed.discard(message.get("tool_call_id"))
                if not owed:
                    drop_end = i + 1  # round closed here
                    ends.append(drop_end)
            elif role == "assistant":
                if owed:
                    break  # a round is still open, bail
                calls = message.get("tool_calls") or ()
                if calls:
                    owed = {str(call.get("id") or "") for call in calls}
                    round_origin = i
                else:
                    drop_end = i + 1  # prose message, closed here
                    ends.append(drop_end)
            elif role == "user":
                if owed:
                    break  # user message while a call is owed
                drop_end = i + 1
                ends.append(drop_end)
            else:
                break  # unknown role: do not drop blind
        if owed:
            # open round never dropped
            drop_end = min(drop_end, round_origin)
        if drop_end <= 1:
            return  # nothing droppable
        # drop the fewest whole units that clear the overage;
        # undershoot is fine because the cap is a ceiling. A bare
        # prefix cut can slice a round in half and 400 the request
        drop = drop_end - 1  # fallback: drop everything droppable
        for end in ends:
            if end - 1 >= need:
                drop = end - 1
                break
        if drop <= 0:
            return
        log.warning(
            "transcript lane: history capped at %d messages — dropped %d "
            "oldest fully-closed round message(s)",
            MAX_KEPT_HISTORY_MESSAGES, drop,
        )
        del self._history[1:1 + drop]

    # ------------------------------------------------------------ speech (TTS)

