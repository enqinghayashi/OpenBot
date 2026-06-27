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


