"""EnergyGate: a thin pure-stdlib voice-activity gate in front of the brain.

Why this exists (measured box facts, OpenBot protocol reference
audio/speech_gate.py, B2 re-measure gate): with the robot awake, the
G1's own fan measures peaks 836-1122 while speech peaks 2700-14674, and
the cloud lane's server VAD kept treating fan noise as speech - each false
turn billed.  Level separates cleanly on this hardware, so the gate is a
plain adaptive energy detector: it tracks the room's own noise floor, so it
does not need retuning when the venue changes.

Behaviour (locked slice spec):

* ``feed(bytes)`` returns ``None`` while it is accumulating and returns the
  completed utterance as one ``bytes`` when the hangover closes it.
* Opens when a chunk's peak exceeds ``max(min_peak, floor_factor *
  noise_floor)``; the noise floor is the rolling ``floor_percentile``
  percentile of recent peaks (all chunks are learned - speech has gaps
  between words, so the low percentile stays a floor even mid-utterance).
* The ``prefix_ms`` of audio buffered BEFORE the first speech is prepended,
  so the first syllable is never clipped.
* ``hangover_ms`` of quiet after the last loud chunk closes the utterance.
* ``cooldown_s`` of dead time follows every emission (a repeat trigger
  guard); input during the cooldown is ignored.

Pure stdlib (``array`` for peaks, ``collections.deque`` for windows), so it
imports on any laptop.
"""

from __future__ import annotations

import array
import collections
import time
from typing import Deque, List, Optional

G1_SAMPLE_RATE = 16000


class EnergyGate:
    """Adaptive level VAD; :meth:`feed` emits completed utterances."""

    def __init__(
        self,
        sample_rate: int = G1_SAMPLE_RATE,
        min_peak: int = 2200,
        floor_factor: float = 3.0,
        floor_percentile: float = 30.0,
        floor_window_s: float = 5.0,
        prefix_ms: int = 400,
        hangover_ms: int = 900,
        cooldown_s: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        # 2200: fan peaks 836-1122, clangs 1501, speech 5315 (p95) /
        # 26400-32571 (max). clears the clang band, stays under speech.
        # the old 1200 let a 1501 clang open the gate; that fed the
        # empty-transcript whisper-decline loop.
        self.min_peak = min_peak
        self.floor_factor = floor_factor
        self.floor_percentile = floor_percentile
        self.hangover_ms = hangover_ms
        self.prefix_ms = prefix_ms
        self.cooldown_s = cooldown_s

        self._bytes_per_s = sample_rate * 2  # 16-bit mono
        # ~40 ms chunks, so this window is always floor_window_s of audio
        window = max(8, int(floor_window_s * 1000.0 / 40.0))
        self._peaks = collections.deque(maxlen=window)  # type: Deque[float]
        # run-up while shut, capped at prefix_ms
        self._prefix = collections.deque()  # type: Deque[bytes]
        self._prefix_bytes = 0
        self._speech = bytearray()
        self._open = False
        self._quiet_chunks = 0
        self._cooldown_until = 0.0

        self.opened = 0  # diagnostics only

    # ------------------------------------------------------- quick queries

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def noise_floor(self) -> float:
        if not self._peaks:
            return 0.0
        ordered = sorted(self._peaks)
        index = int(self.floor_percentile / 100.0 * (len(ordered) - 1))
        return float(ordered[index])

    @property
    def threshold(self) -> float:
        return max(float(self.min_peak), self.noise_floor * self.floor_factor)

    # ------------------------------------------------------ taking audio in

    def feed(self, pcm: bytes) -> Optional[bytes]:
        """One chunk of 16 kHz mono PCM16.

        Returns a completed utterance (prefix + speech + intra-utterance
        gaps) when the hangover closes it, else ``None``.
        """
        if not pcm:
            return None
        now = time.monotonic()
        if now < self._cooldown_until:
            return None  # still cooling down

        peak = _peak(pcm)
        self._peaks.append(peak)

        if self._open:
            if peak >= self.threshold:
                self._quiet_chunks = 0
                self._speech += pcm
            else:
                self._quiet_chunks += 1
                self._speech += pcm  # intra-utterance gap / hangover
                if self._quiet_chunks >= _chunks(self.hangover_ms, self._bytes_per_s, len(pcm)):
                    return self._close()
            return None

        # rolling run-up, so the first word never clips
        self._prefix.append(pcm)
        self._prefix_bytes += len(pcm)
        while self._prefix_bytes > _bytes(self.prefix_ms, self._bytes_per_s):
            dropped = self._prefix.popleft()
            self._prefix_bytes -= len(dropped)
        if peak < self.threshold:
            return None

        self._open = True
        self._quiet_chunks = 0
        self._speech = bytearray()
        self._speech += b"".join(self._prefix)
        self._prefix.clear()
        self._prefix_bytes = 0
        self._speech += pcm
        return None

    def _close(self) -> Optional[bytes]:
        self._open = False
        self._quiet_chunks = 0
        utterance = bytes(self._speech)
        self._speech = bytearray()
        self.opened += 1
        self._cooldown_until = time.monotonic() + self.cooldown_s
        return utterance if utterance else None

