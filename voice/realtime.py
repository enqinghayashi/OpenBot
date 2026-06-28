"""RealtimeClient: the OpenAI Realtime WebSocket wire layer (cloud lane).

The exact GA ``session.update`` field set is LOCKED in this file (OpenBot
v4 slice spec, reference-verified byte-for-byte against the OpenBot
``build_session_update`` in ``chat/cloud/events.py`` and OpenAI's live GA
docs - all three agree).  The locked field set, verbatim:

    {
      "type": "session.update",
      "session": {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
          "input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "gpt-realtime-whisper", "language": "en"},
            "turn_detection": {"type": "server_vad", "threshold": 0.75,
                               "prefix_padding_ms": 300,
                               "silence_duration_ms": 500,
                               "create_response": true}
          },
          "output": {"format": {"type": "audio/pcm", "rate": 24000},
                     "voice": "cedar", "speed": 0.9}
        },
        "tools": <flat list>,
        "tool_choice": "auto"
      }
    }

Field-by-field, as the reference ships it: ``output_modalities``;
``audio.input.format`` {``type``, ``rate``}; ``audio.input.transcription``
{``model``, ``language``}; ``audio.input.turn_detection`` {``type``,
``threshold``, ``prefix_padding_ms``, ``silence_duration_ms``,
``create_response``}; ``audio.output.format`` {``type``, ``rate``};
``audio.output.voice``; ``audio.output.speed``.  There is NO
``sample_rate`` key anywhere, no top-level ``modalities``, and no
top-level ``input_transcription`` / ``input_turn_detection`` -
transcription and turn detection nest under ``audio.input``.  ``tools`` /
``tool_choice`` are attached ONLY when tools were supplied (reference
parity - never an empty list).  ``instructions`` is an ADD-ON (the
OpenBot identity; absent by default — see ``_session_update``): a GA
session.update field, and the ONE deviation from the reference's field
set.

Wire facts taken from the reference:

* Endpoint ``wss://api.openai.com/v1/realtime?model=<model>`` with a
  bearer token.  GA removed the retired ``OpenAI-Beta: realtime=v1``
  header; this client sends none.
* Consequence quote (reference, verbatim): "A wrong field name does not
  close the socket; the server sends one ``error`` event and carries on,
  so the symptom is a session that connects and then never speaks."
  ``session.updated`` must therefore arrive within 10 s of connect or the
  update was rejected.
* Reconnect: ladder 1, 2, 4, 8, 8, 8, 8, 8 s (8 retries max, then a loud
  error); HTTP 401/402/403/404 are NEVER retried (credentials/model will
  not fix themselves); 429 gets a single retry honouring ``Retry-After``
  when present and parseable, otherwise it falls through to the generic
  ladder.
* Sessions are capped at 60 min; on expiry the client performs a clean
  normal reconnect whose conversation context is reset, sets a
  ``context_reset`` flag and fires the ``on_session_expired`` callback.

Events are dispatched as ``(name, payload dict)`` to callbacks registered
with :meth:`on_event`.  Audio ``delta`` payloads reach callbacks intact,
but the LOGGER never sees them: :func:`redact` strips blob fields and any
string over 64 chars (64 or fewer pass through untouched), and audio
events are logged as lengths only.

LAW: ``input_audio_buffer.commit`` is lifetime-banned on this lane -
server VAD (``audio.input.turn_detection.server_vad``) owns turn
segmentation and a manual commit would double-turn every utterance.  Every
client->server event passes the :meth:`RealtimeClient._emit` choke point,
which raises ``AssertionError`` on the commit type and on any type
outside ``_REALTIME_SEND_TYPES``.

``websockets`` is imported lazily inside :meth:`connect`, so importing
this module never requires it.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("voice.realtime")

# ----------------------------------------------------------- the wire spec

DEFAULT_BASE_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2.1-mini"
DEFAULT_VERSION = "2024-11-10"  # contract pin, info

#: 8 retries on a 1/2/4/8 s ladder
RETRY_LADDER_S = (1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0)
MAX_RETRIES = 8
#: 401-404: no retry
NO_RETRY_STATUS = frozenset((401, 402, 403, 404))
#: 10 s to ack
SESSION_UPDATED_TIMEOUT_S = 10.0
#: 60-min cap
SESSION_CAP_S = 3600.0
#: 1 MiB is tight
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# client -> server
SESSION_UPDATE = "session.update"
INPUT_AUDIO_APPEND = "input_audio_buffer.append"
INPUT_AUDIO_CLEAR = "input_audio_buffer.clear"
CONVERSATION_ITEM_CREATE = "conversation.item.create"
CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
RESPONSE_CREATE = "response.create"
RESPONSE_CANCEL = "response.cancel"

#: only these event types may leave
#: _emit asserts on anything else
#: commit is banned
_REALTIME_SEND_TYPES = frozenset((
    "input_audio_buffer.append",
    "input_audio_buffer.clear",
    "conversation.item.create",
    "response.create",
    "response.cancel",
    "session.update",
))

#: name -> aliases
#: raw name first
CANONICAL_ALIASES = {
    "response.created": ("response.started",),
    "response.output_item.done": ("output_item.done",),
    "response.output_audio.delta": ("response.audio.delta",),
    "response.audio.delta": ("response.audio.delta",),  # old beta dialect, still kept
}
AUDIO_DELTA_TYPES = frozenset(("response.output_audio.delta", "response.audio.delta"))

#: audio/base64 blob keys are redacted
_BLOB_KEYS = frozenset(("audio", "delta"))


class RealtimeError(RuntimeError):
    """A Realtime wire failure (retryable unless ``status`` says otherwise)."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class RealtimeAuthError(RealtimeError):
    """Bad key or no access to the model.  Retrying will not help."""


class RealtimeClient:
    """One authenticated, reconnectable WebSocket to the Realtime API.

    Build it (no I/O), register callbacks, then drive it with
    :meth:`run_forever` (connect-with-retry + handshake + read/dispatch
    loop with auto-reconnect and the 60-min session cap).  ``tools`` is
    the flat tool list produced by ``voice.adapters.flatten_specs``.

    The GA ``session.update`` field set is LOCKED (reference-verified
    byte-for-byte against OpenBot ``build_session_update``): audio is a
    single ``audio`` object whose ``input`` nests ``format`` {``type``,
    ``rate``}, ``transcription`` {``model``, ``language``} and
    ``turn_detection`` {``type``, ``threshold``, ``prefix_padding_ms``,
    ``silence_duration_ms``, ``create_response``}, and whose ``output``
    nests ``format`` {``type``, ``rate``}, ``voice`` and ``speed`` -
    ``output_modalities`` (never ``modalities``), no ``sample_rate`` key
    anywhere, no top-level ``input_transcription`` /
    ``input_turn_detection``; ``tools`` / ``tool_choice`` only when tools
    were supplied; and no ``OpenAI-Beta`` header.  Consequence quote
    (reference): "A wrong
    field name does not close the socket; the server sends one ``error``
    event and carries on, so the symptom is a session that connects and
    then never speaks."
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        api_key: str = "",
        version: str = DEFAULT_VERSION,
        tools: Optional[List[Dict[str, Any]]] = None,
        instructions: str = "",
        session_updated_timeout_s: float = SESSION_UPDATED_TIMEOUT_S,
        session_cap_s: float = SESSION_CAP_S,
        on_session_expired: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.version = version  # pin, info only
        self.tools = list(tools or [])
        #: identity text the model speaks as
        #: same channel as DEFAULT_INSTRUCTIONS
        #: unset -> field absent (tests pin it)
        self.instructions = str(instructions)
        self.session_updated_timeout_s = session_updated_timeout_s
        self.session_cap_s = session_cap_s
        self.on_session_expired = on_session_expired

        self._ws = None  # type: Optional[Any]
        self._handlers = {}  # type: Dict[str, List[Callable[[str, dict], Any]]]
        self._closing = False
        self._retried_429 = False
        self._session_started_at = 0.0
        # the 60-min cap resets the context
        self.context_reset = False

    # ------------------------------------------------ events and handlers

    def on_event(self, name: str, callback: Callable[[str, dict], Any]) -> None:
        """Register ``callback(name, payload_dict)`` for events named ``name``.

        Sync callables run inline; if a callback returns an awaitable it is
        scheduled as a task on the running loop.
        """
        self._handlers.setdefault(name, []).append(callback)

    def _fire(self, name: str, payload: Dict[str, Any]) -> None:
        """Dispatch an INBOUND event to listeners (raw name + aliases).

        Sync callables run inline; if a callback returns an awaitable it is
        scheduled as a task on the running loop.  (The SEND-side choke
        point is :meth:`_emit`; this method only fans inbound events out.)
        """
        for callback in list(self._handlers.get(name, ())):
            try:
                result = callback(name, payload)
            except Exception:  # noqa: BLE001 - keep the client up
                log.exception("realtime callback for %r failed", name)
                continue
            if inspect.isawaitable(result):
                try:
                    asyncio.ensure_future(result)
                except RuntimeError:  # no running loop
                    log.debug("realtime callback %r returned awaitable outside a loop", name)

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """THE send choke point: every client->server event passes here.

        Validates ``event_type`` against ``_REALTIME_SEND_TYPES`` and
        returns the wire event ``{"type": event_type, **payload}``.
        ``input_audio_buffer.commit`` is lifetime-banned - server VAD owns
        turn segmentation and a manual commit would double-turn every
        utterance - and any type outside the allowlist is equally
        rejected; both raise ``AssertionError``.  All ``send_*`` methods
        (and the session.update send) route through this method.
        """
        if event_type == "input_audio_buffer.commit":
            raise AssertionError("input_audio_buffer.commit is lifetime-banned — never committed")
        if event_type not in _REALTIME_SEND_TYPES:
            raise AssertionError(
                "unsendable event type {!r}: not in _REALTIME_SEND_TYPES".format(event_type)
            )
        event = {"type": event_type}
        event.update(payload)
        return event

    # ------------------------------------------------ lifecycle / reconnect

    @property
    def endpoint(self) -> str:
        return "{}?model={}".format(self.base_url, self.model)

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self) -> None:
        """One connect + session.update handshake.  Raises on failure.

        Sends ``session.update`` with the locked GA field set and waits up
        to ``session_updated_timeout_s`` for ``session.updated`` - a
        timeout is a loud failure (almost always a rejected field name).
        """
        if not self.api_key:
            raise RealtimeAuthError(
                "no api key; pass api_key= or set OPENAI_API_KEY", status=401
            )
        try:
            import websockets  # lazy: net dep
        except ImportError as exc:
            raise RealtimeError("websockets is not installed: {}".format(exc))
        try:
            self._ws = await _ws_connect(
                self.endpoint,
                self._headers(),
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
                max_size=MAX_MESSAGE_BYTES,
                open_timeout=self.session_updated_timeout_s + 10.0,
            )
        except websockets.InvalidStatusCode as exc:  # this is the websockets < 14 spelling
            self._raise_status(getattr(exc, "status_code", 0), exc)
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", 0)
            if status:
                self._raise_status(status, exc)
            raise RealtimeError("could not open the Realtime socket: {}".format(exc))
        self._session_started_at = time.monotonic()
        self._retried_429 = False
        self._fire("open", {})
        log.info("realtime socket open (%s)", self.endpoint)
        await self._send_raw(self._session_update())
        await self._await_handshake()

    def _raise_status(self, status: int, exc: Exception) -> None:
        """Map an HTTP handshake status to the right error class."""
        if status in NO_RETRY_STATUS:
            raise RealtimeAuthError(
                "OpenAI rejected the handshake (HTTP {}): {}".format(status, exc),
                status=status,
            )
        if status == 429:
            raise RealtimeError(
                "rate limited (HTTP 429): {}".format(exc),
                status=status,
                retry_after=_retry_after_seconds(getattr(exc, "headers", None)),
            )
        raise RealtimeError("handshake failed (HTTP {}): {}".format(status, exc), status=status)

    async def _await_handshake(self) -> None:
        """Wait for session.updated; any other event is dispatched and any
        error event means the update was rejected."""
        while True:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(), timeout=self.session_updated_timeout_s
                )
            except asyncio.TimeoutError:
                self._fire("timeout", {"seconds": self.session_updated_timeout_s})
                raise RealtimeError(
                    "the server never acknowledged session.update within {} s; "
                    "the field names may have been rejected (look for an "
                    "'error' event above)".format(self.session_updated_timeout_s)
                )
            except Exception as exc:  # noqa: BLE001 - counts as connect fail
                raise RealtimeError(
                    "realtime socket closed during the handshake: {}".format(
                        str(exc).strip() or exc.__class__.__name__
                    )
                )
            event = _decode(message)
            if event.get("type") == "session.updated":
                self._dispatch(event)
                log.info("session.updated: handshake complete")
                return
            if event.get("type") == "error":
                self._dispatch(event)
                raise RealtimeError(
                    "session.update rejected: {}".format(_error_text(event))
                )
            self._dispatch(event)

    async def run_forever(self) -> None:
        """Connect (with retry ladder), then read/dispatch until close.

        Auto-reconnects on drops (clean handshake each time) and performs
        the clean 60-min session-cap reconnect; returns only on deliberate
        :meth:`close`, an auth error, or after retry exhaustion (loud).
        """
        while not self._closing:
            try:
                await self.connect_with_retry()
            except RealtimeAuthError:
                await self._close_socket()
                raise
            except RealtimeError as exc:
                await self._close_socket()
                self._fire("error", {"message": str(exc)})
                log.error("realtime: %s", exc)
                return  # loud stop
            try:
                while not self._closing:
                    if self._session_started_at and (
                        time.monotonic() - self._session_started_at >= self.session_cap_s
                    ):
                        self._expire_session()
                        break
                    try:
                        message = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:  # noqa: BLE001 - a dead socket ends the read
                        log.warning("realtime read ended: %s", exc)
                        break
                    event = _decode(message)
                    self._dispatch(event)
            finally:
                was_open = self._ws is not None
                await self._close_socket()
                if was_open:
                    self._fire("close", {})

    async def connect_with_retry(self) -> None:
        """Connect with the ladder 1, 2, 4, 8, 8, 8, 8, 8 s (8 retries max).

        401/402/403/404 raise immediately (never retried); 429 gets a
        single retry honouring ``Retry-After`` when present and parseable,
        otherwise it falls through to the generic ladder; exhaustion
        raises a loud :class:`RealtimeError`.
        """
        for attempt in range(1, MAX_RETRIES + 2):
            try:
                await self.connect()
                return
            except RealtimeAuthError:
                raise  # creds and model won't fix themselves
            except RealtimeError as exc:
                if exc.status in NO_RETRY_STATUS:
                    raise
                if attempt > MAX_RETRIES:
                    # budget spent; checked before 429
                    # a "retrying once" log with no retry = lie
                    raise RealtimeError(
                        "could not reach the Realtime API after {} retries: {}".format(
                            MAX_RETRIES, exc
                        )
                    )
                if exc.status == 429 and not self._retried_429:
                    self._retried_429 = True
                    if exc.retry_after is not None:
                        wait = min(exc.retry_after, 60.0)
                        log.warning("realtime rate limited; retrying once in %.0fs", wait)
                        await asyncio.sleep(wait)
                        continue
                    # no usable Retry-After -> generic
                    # ladder below (the 429 retry is spent)
                delay = RETRY_LADDER_S[attempt - 1]
                log.warning(
                    "realtime connect attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt,
                    MAX_RETRIES + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    def _expire_session(self) -> None:
        """60-min session cap: flag the context reset and reconnect cleanly."""
        self.context_reset = True
        self._fire("session.expired", {"seconds": int(self.session_cap_s)})
        callback = self.on_session_expired
        if callback is not None:
            try:
                callback()
            except Exception:  # noqa: BLE001
                log.exception("on_session_expired callback failed")

    async def close(self) -> None:
        """Deliberate close: stop reconnecting and release the socket.

        ``close`` is emitted only when the socket was actually open - the
        ``was_open`` pattern from ``run_forever``'s finally - so a
        never-connected client stays silent.
        """
        self._closing = True
        was_open = self._ws is not None
        await self._close_socket()
        if was_open:
            self._fire("close", {})

    async def _close_socket(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            log.debug("closing the realtime socket raised: %s", exc)

    # ------------------------------------------------------ the senders

    def _headers(self) -> Dict[str, str]:
        """GA: no ``OpenAI-Beta`` header - the retired beta header is
        removed (the GA docs dropped it; sending it only risks the beta
        dialect)."""
        return {"Authorization": "Bearer {}".format(self.api_key)}

    def _session_update(self) -> Dict[str, Any]:
        """The LOCKED GA session.update payload (reference-verified
        byte-for-byte against OpenBot ``build_session_update``, minus
        the intentional deviation below).  ``audio.input.format`` is the
        object shape {"type": "audio/pcm", "rate": 24000} and
        ``transcription`` / ``turn_detection`` nest under ``audio.input``
        - there is no ``sample_rate`` key anywhere.  ``tools`` /
        ``tool_choice`` are attached ONLY when tools were supplied (never
        an empty list).  Routes through ``_emit`` like every send.
        ``instructions`` is present ONLY when configured (the identity
        the model speaks as — see ``RealtimeClient.__init__``): the
        pre-identity wire kept it absent and the locked-shape tests pin
        that default; OpenBot ships its identity text in
        ``config.voice.yaml realtime.instructions`` the same channel
        OpenBot ships its DEFAULT_INSTRUCTIONS on."""
        session = {
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-realtime-whisper", "language": "en"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.75,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "cedar",
                    "speed": 0.9,
                },
            },
        }
        if self.tools:
            # parity: tools only when supplied
            # never an empty list
            session["tools"] = self.tools
            session["tool_choice"] = "auto"
        if self.instructions:
            # identity: sent once, at connect time
            # per-turn wins
            # unset -> absent (the tests pin that)
            session["instructions"] = self.instructions
        return self._emit(SESSION_UPDATE, {"session": session})

    async def _send_raw(self, event: Dict[str, Any]) -> None:
        ws = self._ws
        if ws is None:
            raise RealtimeError("send on a closed realtime socket")
        payload = json.dumps(event)
        try:
            await ws.send(payload)
        except Exception as exc:  # noqa: BLE001
            raise RealtimeError("send failed: {}".format(exc))
        if log.isEnabledFor(logging.DEBUG):
            if event.get("type") == INPUT_AUDIO_APPEND:
                log.debug("-> input_audio_buffer.append (length only)")
            else:
                log.debug("-> %s %s", event.get("type"), redact(event))

    def _require_connected(self) -> None:
        if self._ws is None:
            raise RealtimeError("realtime socket is not connected; call connect() first")

    async def send_audio(self, bytes_pcm24k: bytes) -> None:
        """Append one 24 kHz mono PCM16 chunk.  No per-chunk ack exists."""
        if not bytes_pcm24k:
            return
        self._require_connected()
        await self._send_raw(
            self._emit(
                INPUT_AUDIO_APPEND,
                {"audio": base64.b64encode(bytes_pcm24k).decode("ascii")},
            )
        )

    async def send_image(self, jpeg_b64: str) -> None:
        """Attach one JPEG (already base64) as a user message part."""
        self._require_connected()
        await self._send_raw(
            self._emit(
                CONVERSATION_ITEM_CREATE,
                {
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/jpeg;base64,{}".format(jpeg_b64),
                            }
                        ],
                    }
                },
            )
        )

    async def send_input_text(self, text: str) -> None:
        """Inject a text user turn (tests / non-audio prompts)."""
        self._require_connected()
        await self._send_raw(
            self._emit(
                CONVERSATION_ITEM_CREATE,
                {
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    }
                },
            )
        )

    async def send_function_call_output(self, call_id: str, output_text: str) -> None:
        """Answer a function call; the model does nothing until a
        ``create_response`` follows - the caller owns that second step."""
        self._require_connected()
        await self._send_raw(
            self._emit(
                CONVERSATION_ITEM_CREATE,
                {
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": output_text,
                    }
                },
            )
        )

    async def send_truncate(self, item_id: str, audio_end_ms: int) -> None:
        """Rewrite an assistant item to only the audio actually heard.

        NOTE: ``conversation.item.truncate`` is NOT in
        ``_REALTIME_SEND_TYPES`` (the locked slice allowlist), so the
        ``_emit`` choke point raises ``AssertionError``.  The method is
        kept as a documented dead end until the orchestrator decides
        whether truncation belongs on this lane.
        """
        self._require_connected()
        await self._send_raw(
            self._emit(
                CONVERSATION_ITEM_TRUNCATE,
                {
                    "item_id": item_id,
                    "content_index": 0,
                    "audio_end_ms": max(0, int(audio_end_ms)),
                },
            )
        )

    async def send_clear(self) -> None:
        """Clear the server's input audio buffer."""
        self._require_connected()
        await self._send_raw(self._emit(INPUT_AUDIO_CLEAR, {}))

    async def create_response(self) -> None:
        """Ask for a response now (server VAD also creates them)."""
        self._require_connected()
        await self._send_raw(self._emit(RESPONSE_CREATE, {"response": {}}))

