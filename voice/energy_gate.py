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


