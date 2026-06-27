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

