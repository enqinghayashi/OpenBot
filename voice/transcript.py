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

