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


class G1Speaker:
    """PCM16 / 16 kHz / mono playback on the G1's own speaker.

    Constructor performs the ~0.7 s SDK client build once; it requires
    ``unitree_sdk2py`` (robot only).  All RPC calls serialise under one
    lock.  ``play`` ENQUEUES and the paced playback worker drains it as
    ``chunk_ms`` chunks under one stream id (see the module docstring —
    before the queue, each chunk was one synchronous RPC per audio delta
    from the WebSocket thread, which is the reported lag/artefact);
    ``stop`` cancels queued audio then issues ``PlayStop`` — a documented
    no-op in effect, so silence after ``stop()`` is not promised.
    Construction also runs the volume boot gate: SetVolume(100) + a
    read-back that must equal 100 (``G1_VOLUME``), else the ctor raises
    and the process fails fast - the volume pin is architectural.
    """

    def __init__(
        self,
        app_name: str = G1_APP_NAME,
        sample_rate: int = G1_SAMPLE_RATE,
        cap_bytes: int = MAX_PLAYSTREAM_BYTES,
        timeout_s: float = 10.0,
        chunk_ms: int = PLAYBACK_CHUNK_MS,
        max_lead_s: float = PLAYBACK_LEAD_S,
        sink_latency_s: float = PLAYBACK_SINK_LATENCY_S,
        queue_cap_s: float = PLAYBACK_QUEUE_CAP_S,
        monotonic: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if sample_rate != G1_SAMPLE_RATE:
            raise ValueError(
                "G1Speaker accepts 16 kHz only (PlayStream contract), got {}".format(sample_rate)
            )
        self.app_name = app_name
        self.cap_bytes = int(cap_bytes)
        self.timeout_s = timeout_s
        # RPC + this state under ONE lock
        # (SDK client is not thread-safe), never
        # held across the pacing sleep
        self.chunk_bytes = min(
            int(cap_bytes), int(sample_rate * 2 * chunk_ms / 1000.0)
        )
        self.max_lead_s = float(max_lead_s)
        self.sink_latency_s = float(sink_latency_s)
        self.queue_cap_bytes = int(sample_rate * 2 * float(queue_cap_s))
        self._bytes_per_second = sample_rate * 2
        self._monotonic = monotonic
        self._sleep = sleep_fn
        self._lock = threading.Lock()
        self._queue = queue.Queue()  # type: queue.Queue
        self._queued_bytes = 0
        self._generation = 0
        self._stream_id = ""  # one stream per burst
        self._play_until = 0.0  # when the queued audio runs out
        self._stream_started = 0.0
        self._stream_bytes = 0
        self._speaking = False
        self._drained = threading.Event()
        self._drained.set()
        self._fatal = None  # type: Optional[Exception]
        self._worker = None  # type: Optional[threading.Thread]
        self._client = self._build_client()  # ~0.7 s, done once
        self._ensure_volume()  # boot gate: 100
        self._worker = threading.Thread(
            target=self._worker_main, name="g1-playback", daemon=True
        )
        self._worker.start()
        log.info(
            "playback queue: 16 kHz mono, %d ms chunks (%d B), bound %.1f s, "
            "lead %.1f s (one stream per burst)",
            chunk_ms, self.chunk_bytes, self.queue_cap_bytes / self._bytes_per_second,
            self.max_lead_s,
        )

    def _build_client(self) -> Any:
        # box SDK checkout:
        # /home/unitree/unitree_sdk2_python
        # bare 'voice' env never pip-installed it
        # (RUNBOOK §13.3)
        # no-op on a laptop
        # import stays lazy
        if os.path.isdir(
            "/home/unitree/unitree_sdk2_python"
        ) and "/home/unitree/unitree_sdk2_python" not in sys.path:
            sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
        try:
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        except ImportError as exc:  # pragma: no cover - robot-only
            raise SpeakerError(
                "unitree_sdk2py is not installed ({}); G1Speaker runs on "
                "the robot, use SounddeviceSink/MockSink on a laptop".format(exc)
            )
        # channel factory MUST init before any client
        # skip it: DDS participant None, then
        # "'NoneType' object has no attribute '_ref'"
        # in the cyclonedds Topic ctor
        # eth0 iface
        # idempotent
        # G1Mic: multicast
        ChannelFactoryInitialize(0, "eth0")
        client = AudioClient()
        client.SetTimeout(self.timeout_s)
        client.Init()
        return client

    @property
    def connected(self) -> bool:
        return self._client is not None

    # -------------------------------------------------------------- playback

    def play(self, pcm16: bytes) -> None:
        """Enqueue PCM16/16 kHz/mono for paced playback (never blocks).

        The caller (the session's audio-event thread) only enqueues; the
        ONE worker thread drains at wall-clock rate under a single stream
        id (module docstring: the lag/artefact fix).  Raises
        ``SpeakerError`` immediately on a previous wire failure (the
        fatal state, surfaced via :meth:`stop`-able path: the app's
        ``play_pcm`` catches it and sets its error flag).
        """
        if not pcm16:
            return
        if len(pcm16) % 2:
            raise SpeakerError(
                "PCM buffer of {} bytes is not whole 16-bit frames".format(len(pcm16))
            )
        seconds = len(pcm16) / float(self._bytes_per_second)
        with self._lock:
            if self._fatal is not None:
                raise self._fatal
            if self._queued_bytes + len(pcm16) > self.queue_cap_bytes:
                overflow = True
            else:
                overflow = False
                if not self._stream_id:
                    # new burst: fresh stream id + clock
                    # first chunk isn't audible until the
                    # buffer fills (sink_latency_s)
                    self._stream_id = "{:d}".format(int(time.time() * 1000))
                    self._stream_started = 0.0
                    self._stream_bytes = 0
                    self._play_until = 0.0
                self._queued_bytes += len(pcm16)
        if overflow:
            log.error(
                "speaker queue overflow (%.2f s buffered > %.2f s cap): dropping "
                "%.2f s of audio (the robot is falling behind the model)",
                self._queued_bytes / float(self._bytes_per_second),
                self.queue_cap_bytes / float(self._bytes_per_second),
                seconds,
            )
            return
        self._drained.clear()
        self._queue.put((self._generation, pcm16))

    def stop(self) -> None:
        """Cancel queued playback and end the stream (best-effort).

        Bumps the generation (the worker drops everything still queued
        — that is the part that actually works), resets the stream, then
        issues ``PlayStop`` — a documented NO-OP in effect: the SDK
        discards the RPC result and always returns 0, so silence after
        ``stop()`` is not promised.  Never raises.  The next ``play``
        starts a new burst on a fresh stream id.
        """
        with self._lock:
            self._generation += 1
            self._queued_bytes = 0
            self._stream_id = ""
            self._play_until = 0.0
            self._stream_started = 0.0
            self._stream_bytes = 0
            self._speaking = False
        self._drain_queue()
        self._drained.set()
        client = self._client
        if client is None:
            return
        try:
            with self._lock:
                client.PlayStop(self.app_name)
        except Exception as exc:  # noqa: BLE001
            log.debug("PlayStop raised: %s", exc)

    @property
    def is_speaking(self) -> bool:
        """True while audio is queued or still sounding from the speaker."""
        with self._lock:
            return self._speaking or self._monotonic() < self._play_until

    def wait_until_drained(self, timeout: Optional[float] = None) -> bool:
        """Block until the queued audio has finished sounding.  ``False``
        on timeout.  Test/teardown seam; the live app never calls it."""
        return self._drained.wait(timeout)

    # ------------------------------------------------------ playback worker

    def _worker_main(self) -> None:
        """One thread for ALL PlayStream calls (FR-8, module docstring)."""
        buffer = bytearray()
        buffer_generation = self._generation
        while True:
            try:
                item = self._queue.get(timeout=PLAYBACK_POLL_S)
            except queue.Empty:
                if self._generation != buffer_generation:
                    buffer.clear()  # got cancelled while we held audio
                    buffer_generation = self._generation
                if buffer:
                    # idle-flush: a reply's last <200 ms
                    # must not wait for the next enqueue,
                    # or every answer loses its last word
                    # same stream id: device concatenates
                    if self._generation == buffer_generation:
                        if not self._emit(bytes(buffer), buffer_generation):
                            return  # fair enough - the wire died
                    buffer = bytearray()
                # done-when-quiet decay runs on the
                # real clock (time.monotonic, same as the
                # injected test clock); pacing leads stay
                # on self._monotonic
                with self._lock:
                    empty = self._queued_bytes == 0
                    still = time.monotonic() < self._play_until
                if empty and not still:
                    self._speaking = False
                    self._drained.set()
                continue
            if item is _STOP:
                break
            generation, item = item
            if generation != self._generation:
                continue  # a stop() already superseded this
            if generation != buffer_generation:
                buffer = bytearray()
                buffer_generation = generation
            with self._lock:
                self._queued_bytes = max(0, self._queued_bytes - len(item))
            buffer += item
            while len(buffer) >= self.chunk_bytes:
                if self._generation != generation:
                    buffer.clear()
                    buffer_generation = self._generation
                    break
                chunk = bytes(buffer[: self.chunk_bytes])
                del buffer[: self.chunk_bytes]
                if not self._emit(chunk, generation):
                    return  # wire failed, worker ends

    def _emit(self, chunk: bytes, generation: int) -> bool:
        """Pace, then hand ONE chunk to the speaker.  ``False`` = stop."""
        now = self._monotonic()
        with self._lock:
            if generation != self._generation or self._fatal is not None:
                return True  # superseded or fatal: drop it
            if self._stream_started == 0.0:
                # first chunk isn't audible until the
                # buffer fills; clock starts one latency late
                self._stream_started = now
                self._play_until = now + self.sink_latency_s
            elif self._play_until < now:
                # starved after an idle gap: give it the
                # same buffer latency, or the pace clock
                # runs ahead and the echo guard opens the
                # mic early
                self._play_until = now + self.sink_latency_s
            lead = self._play_until - now
            stream_id = self._stream_id or "{:d}".format(int(time.time() * 1000))
            self._stream_id = stream_id
        if lead > self.max_lead_s:
            self._sleep(lead - self.max_lead_s)
            with self._lock:
                if generation != self._generation or self._fatal is not None:
                    return True  # stop() landed during the sleep
        self._set_speaking(True)
        self._drained.clear()
        client = self._require()
        try:
            with self._lock:
                # last-instant re-check: stop() can land
                # while this chunk waits; a cancelled burst
                # never plays
                if generation != self._generation or self._fatal is not None:
                    return True
                result = client.PlayStream(self.app_name, stream_id, chunk)
        except Exception as exc:  # noqa: BLE001 - a wire failure is fatal
            self._fatalize(
                "PlayStream raised {}: {}".format(type(exc).__name__, exc)
            )
            return False
        code = _unpack_play(result)
        if code != 0:
            self._fatalize(
                "PlayStream failed: {}".format(describe_code(code))
            )
            return False
        duration_s = len(chunk) / float(self._bytes_per_second)
        with self._lock:
            self._play_until += duration_s
            self._stream_bytes += len(chunk)
        return True

    def _fatalize(self, message: str) -> None:
        """The wire failed mid-playback.

        Records the failure (every later ``play`` raises it, so the app
        sets its error flag within one delta) and ends the worker's
        playing.  Never a warning: a dead speaker is a silent robot.
        """
        log.error("playback is unrecoverable: %s", message)
        with self._lock:
            if self._fatal is None:
                self._fatal = SpeakerError(message)
        self._speaking = False
        self._drained.set()

    def _set_speaking(self, value: bool) -> None:
        with self._lock:
            self._speaking = value

    def _drain_queue(self) -> int:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
                dropped += 1
            except queue.Empty:
                return dropped

    # ------------------------------------------------- volume / led (safe)

    def get_volume(self) -> int:
        """Current speaker volume, 0-100 (``(code, data)`` shape)."""
        client = self._require()
        with self._lock:
            code, data = client.GetVolume()
        if code != 0:
            raise SpeakerError("GetVolume failed: {}".format(describe_code(code)))
        if not isinstance(data, dict) or "volume" not in data:
            raise SpeakerError("GetVolume returned unexpected payload: {!r}".format(data))
        try:
            return int(data["volume"])
        except (TypeError, ValueError) as exc:
            # code-0 with a non-integer volume leaked
            # a raw TypeError; docstring promises only
            # SpeakerError
            raise SpeakerError(
                "GetVolume returned non-integer volume: {!r}".format(data["volume"])
            ) from exc

    def set_volume(self, volume: int) -> None:
        """Set speaker volume, 0-100 (bare-``int`` shape)."""
        if not 0 <= volume <= 100:
            raise ValueError("volume must be 0-100, got {}".format(volume))
        client = self._require()
        with self._lock:
            code = client.SetVolume(volume)  # bare int, so do NOT unpack it
        if code != 0:
            raise SpeakerError(
                "SetVolume({}) failed: {}".format(volume, describe_code(code))
            )

    def _ensure_volume(self) -> None:
        """Boot gate: pin the speaker to ``G1_VOLUME`` and prove it.

        Set then read back on EVERY G1Speaker boot (one construction
        site per process, so this covers the whole architecture).  A
        SetVolume RPC failure raises via ``set_volume``; a successful
        write that does NOT stick (body-side clamp, board state) is a
        hard ``SpeakerError`` — never a warning, because a quiet robot
        is a silently broken robot.  This gate exists because a B2
        probe set volume 20 for a sine test and the robot stayed quiet;
        after this change, any boot re-pins 100.
        """
        self.set_volume(G1_VOLUME)
        actual = self.get_volume()
        if actual != G1_VOLUME:
            raise SpeakerError(
                "volume gate failed: SetVolume({}) but read back {} "
                "- volume is pinned to {} (user directive)".format(
                    G1_VOLUME, actual, G1_VOLUME
                )
            )

    def set_led(self, red: int, green: int, blue: int) -> None:
        """Head LED, 0-255 each.  Failures are logged, never raised."""
        client = self._client
        if client is None:
            return
        try:
            with self._lock:
                code = client.LedControl(  # bare int, so do NOT unpack it
                    max(0, min(255, int(red))),
                    max(0, min(255, int(green))),
                    max(0, min(255, int(blue))),
                )
        except Exception as exc:  # noqa: BLE001 - a light isn't worth a
            log.debug("LedControl raised: %s", exc)  # whole conversation
            return
        if code != 0:
            log.debug("LedControl failed: %s", describe_code(code))

    # ------------------------------------------------------------ internals

    def _require(self) -> Any:
        client = self._client
        if client is None:
            raise SpeakerError("G1 audio client is not connected")
        return client

    def close(self) -> None:
        """Best-effort teardown; never raises.  Stops the queue first."""
        self.stop()  # kills queued audio, then PlayStop
        self._queue.put(_STOP)
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.join(5.0)
        client, self._client = self._client, None
        if client is None:
            return
        try:
            with self._lock:
                client.PlayStop(self.app_name)
        except Exception as exc:  # noqa: BLE001
            log.debug("PlayStop during close raised: %s", exc)


class SounddeviceSink:
    """Laptop speaker via sounddevice (lazy import; same 16 kHz sink format
    as the G1 on purpose, so half-duplex timing tuned on a laptop transfers
    to the robot unchanged - OpenBot protocol reference)."""

    def __init__(
        self,
        sample_rate: int = G1_SAMPLE_RATE,
        channels: int = G1_CHANNELS,
        dtype: str = "int16",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._stream = None  # type: Optional[Any]
        self._device = None  # type: Optional[Any]

    def _open(self) -> None:
        if self._stream is not None:
            return
        try:
            import sounddevice as sd  # lazy: laptop-only dep
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SpeakerError("sounddevice is not installed: {}".format(exc))
        try:
            self._stream = sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
            )
            self._stream.start()
            self._device = sd
        except Exception as exc:  # noqa: BLE001 - surface whatever the device throws
            self._stream = None
            raise SpeakerError("cannot open sounddevice output: {}".format(exc))

    def play(self, pcm16: bytes) -> None:
        if not pcm16:
            return
        self._open()
        stream = self._stream
        if stream is None:  # pragma: no cover - _open() raised already
            return
        stream.write(pcm16)

    def stop(self) -> None:
        """No-op on the laptop: playback is synchronous and already done."""

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                pass

    def __enter__(self) -> "SounddeviceSink":
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002 - protocol stub
        self.close()


class MockSink:
    """Records everything it is asked to play (for tests)."""

    def __init__(self) -> None:
        self.played = []  # type: List[bytes]
        self.stopped = 0

    def play(self, pcm16: bytes) -> None:
        if pcm16:
            self.played.append(pcm16)

    def stop(self) -> None:
        self.stopped += 1

    def close(self) -> None:
        pass

    @property
    def total_bytes(self) -> int:
        return sum(len(p) for p in self.played)


