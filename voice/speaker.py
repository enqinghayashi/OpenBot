"""Speakers: the G1's own speaker, a laptop sink, and a recording mock.

Wire facts (OpenBot protocol reference, devices/g1/speaker.py):
``AudioClient.PlayStream(app_name, stream_id, chunk)`` accepts PCM16 /
16 kHz / mono only, and Unitree's own helper sends 96000 bytes (3 s at
16 kHz) per call - nothing in the SDK enforces a cap, so that is the only
size known to be safe.  Two SDK properties shape this class:

* Every call is synchronous and blocks the calling thread, and the RPC
  client is not safe to drive from several threads at once - so ALL RPC
  calls sit under ONE ``threading.Lock``.
* Return types are inconsistent: ``PlayStream``/``GetVolume`` return
  ``(code, data)``, ``SetVolume``/``LedControl`` return a bare ``int``,
  and ``PlayStop`` throws its result away and returns a literal ``0``.
  Every call unpacks the right shape; ``PlayStop`` proves nothing, which
  is why ``stop()`` is documented as a no-op in effect.

Building the client (``AudioClient()`` + ``Init()``) blocks ~0.7 s (two
sleeps inside the SDK's channel setup), so the constructor does it exactly
once and the instance is reused - never rebuild per utterance (and never
more than once per process: rebuilding creates a second reader/writer pair
on the voice topics).  ``unitree_sdk2py`` is imported lazily inside the
constructor, so importing this module never requires it; laptop code uses
``SounddeviceSink``, tests use ``MockSink``.

THE PACED PLAYBACK QUEUE (the lag/artefact fix - OpenBot protocol
reference, audio/playback_queue.py).  Three wire facts drive it, and the
reference is the implementation that sounds right on its real G1:

1. ``play`` MUST be non-blocking.  Before this fix each incoming audio
   delta was played by a SYNCHRONOUS PlayStream RPC from the WebSocket
   receive thread (the session's ``audio`` event consumers run there).
   Every RPC is a DDS round trip that can hold that thread for seconds,
   so the reply streamed in bursts with gaps between them - the user's
   "laggy" report - and a long-holding RPC froze the whole lane (FR-8
   of the reference: "PlayStream() must never be called from inside the
   WebSocket receive path ... the SDK's RPC client is not safe to drive
   from several threads").  All speaker I/O now happens on ONE worker
   thread; the caller only enqueues bytes.
2. ONE stream id per speech burst.  The device treats a new
   ``stream_id`` as a fresh playback timeline; the old per-call id meant
   a restart per audio delta - device buffer priming per chunk plus
   audible pops between chunks.  A stream id is minted once per burst
   (first enqueue after an idle/stop) and kept until ``stop()``; the
   reference keeps one stream across the whole conversation and clears
   it only on cancel.
3. PACED at wall-clock rate: the worker keeps at most ``max_lead_s``
   (0.6 s) of audio ahead of real time and sleeps the excess lead away
   before each chunk.  Pushing a whole reply into the robot's buffer as
   fast as it arrives (Unitree's own example pushes 3 s of audio per
   1 s of wall clock) plays correctly but - the reference's argument -
   audio the robot has already swallowed cannot be cancelled quickly, so
   a small lead is the barge-in/error knob.  ``sink_latency_s`` (0.35 s)
   is the time between a PlayStream call and actual audibility: without
   accounting for it the pace clock runs ahead and the robot would hear
   itself (the reference measured this on their rig, their config
   comments it at 0.35).

The queue replaces the old "one call, cap into ``cap_bytes`` chunks"
loop.  ``cap_bytes`` stays as the upper bound; the actual chunk size is
``chunk_ms`` (200 ms) of audio, the reference's value.
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, List, Optional

log = logging.getLogger("voice.speaker")

G1_APP_NAME = "uwa_csse_voice"
#: 100. always.
#: 100 everywhere
#: not configurable
#: no probe or body-side change
#: survives a boot
#: same wire as the robot's TTS
#: SetVolume(100)
#: a pin, not a default
G1_VOLUME = 100
G1_SAMPLE_RATE = 16000
G1_CHANNELS = 1
#: the only size known safe
#: 3 s at 16 kHz
MAX_PLAYSTREAM_BYTES = 96000

#: 200 ms chunks, <=0.6 s ahead of real time
#: 0.35 s from PlayStream to audibility
#: 60 s cap
#: further behind, audio drops
#: (the session's own 60 s cap is the real gate)
PLAYBACK_CHUNK_MS = 200
PLAYBACK_LEAD_S = 0.6
PLAYBACK_SINK_LATENCY_S = 0.35
PLAYBACK_QUEUE_CAP_S = 60.0

#: poll interval
#: also worst-case stop()/close()
PLAYBACK_POLL_S = 0.1
_STOP = object()

#: from unitree_sdk2py/rpc/internal.py
RPC_ERRORS = {
    0: "OK",
    3001: "RPC_ERR_UNKNOWN",
    3102: "RPC_ERR_CLIENT_SEND (DDS write failed - is the voice service up?)",
    3103: "RPC_ERR_CLIENT_API_NOT_REG (AudioClient.Init() was not called)",
    3104: "RPC_ERR_CLIENT_API_TIMEOUT (no response from the voice service)",
    3105: "RPC_ERR_CLIENT_API_NOT_MATCH",
    3106: "RPC_ERR_CLIENT_API_DATA",
    3107: "RPC_ERR_CLIENT_LEASE_INVALID",
    3201: "RPC_ERR_SERVER_SEND",
    3202: "RPC_ERR_SERVER_INTERNAL",
    3203: "RPC_ERR_SERVER_API_NOT_IMPL",
    3204: "RPC_ERR_SERVER_API_PARAMETER",
}


def describe_code(code: int) -> str:
    return "{} ({})".format(code, RPC_ERRORS.get(code, "unknown / voice-service specific"))


class SpeakerError(RuntimeError):
    """A speaker call failed.  Never downgrade this to a warning."""
