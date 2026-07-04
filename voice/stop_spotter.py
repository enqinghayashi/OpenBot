"""StopSpotter: an always-on, local 'stop' / 'cancel' detector (B8j-hf19).

Operator, 2026-10-16: *"the stop command is not instant at all, because while
robot is talking, it does not intake the stop command im saying, i want the
stop command to be able to stop it instantly, stop/cancel, instantly"*.

Why the old route was deaf and slow (three stacked filters, all upstream of
the only stop interception at ``app.py on_utterance``):

1. ``EchoGuard`` drops every mic frame quieter than the robot's own playback
   (bar = coupling x margin x playing peak, capped 12000 int16), so a normal
   voice under the robot's speech never leaves the pump.
2. The session's mute law (``session.upload_mic_audio``) drops every
   utterance while reply audio is queued (``_queued_s > 0``).
3. Even when heard, the stop waited for the energy gate's hang-over, the
   upload and the CLOUD transcript before ``_stop_intent`` saw any text.

This spotter sits BEFORE all three: the pump hands it every raw 16 kHz frame
(``submit``), a worker thread runs a FULL-vocabulary Vosk recognizer on the
wake word's model (see ``make_vosk_recognizer`` for why not a restricted
grammar) and acts locally on the FIRST partial result that holds a stop word:

* 'stop' / 'cancel' / 'abort' / 'halt' / 'freeze' (or a segment that is only
  'top', the model's reading of a clipped "Stop.") -> ``on_stop``: the caller
  cuts the speech and sends the same gateway stop as the spoken route;
* 'stop talking' / 'stop speaking' (final result) -> ``on_stop_talking``;
* 'don't stop' / 'never stop' never halt.

One action per recognizer segment; motion stops are spaced by ``cooldown_s``.

Echo rule (there is no acoustic echo cancellation): while the robot is
SAYING a stop-family word ('I've stopped', 'the search is cancelled') its own
voice reaches the mic, so hits are SUPPRESSED from the moment that text is
noted (``note_spoken``) until the robot has been quiet for ``tail_s``.  In that
window the old cloud route is the only route (exactly today's behaviour).

Modes (``OPENBOT_STOP_SPOTTER`` in the voice app's environment): ``1``/unset
= act, ``log`` = log every hit but act on none, ``0`` = off (not built).

Pure stdlib apart from the injected recognizer (``vosk`` is imported lazily by
``make_vosk_recognizer``), Python 3.8 syntax.
"""

from __future__ import annotations

import array
import json
import logging
import os
import queue
import threading
import time
from typing import Any, Callable, List, Optional

log = logging.getLogger("voice.stop_spotter")

#: halt words
#: brain/agent/fastpath.py HALT_WORDS minus the common ones:
#: 'stand', 'wait', 'quit', 'brake' would fire
#: on any one heard word, no sentence
STOP_WORDS = ("stop", "cancel", "abort", "halt", "freeze")
#: 'stop talking'/'stop speaking': speech only, no motion halt
SPEECH_ONLY_NEXT = ("talking", "speaking")
#: halt word after these never halts motion
NEGATIONS = ("don't", "never")
#: ONLY one of these = a misheard stop
#: the small model hears a clipped "Stop." as "top"
#: whole segment only: 'the top shelf' never counts
LONE_ALIASES = ("top",)
#: final segment of ONLY these, with 'top': a stop
#: ('top', 'top it', 'hey stewart top')
#: 'on top' never counts
COMMAND_WORDS = frozenset(("top", "stop", "it", "now", "please", "that", "hey",
                           "stewart", "stuart", "steward", "robot"))
#: substrings that flag our OWN speech
ECHO_STEMS = ("stop", "cancel", "abort", "halt", "freez")

DEFAULT_TAIL_S = 1.5
DEFAULT_COOLDOWN_S = 2.0
#: CPU saver: after HANGOVER_FRAMES quiet ones,
#: stop feeding (hang-over closes the segment)
#: G1 room tone ~1100 (echo_guard.py); one loud frame resumes
QUIET_PEAK = 1500
HANGOVER_FRAMES = 8        # 8 frames x 160 ms = 1.28 s
QUEUE_FRAMES = 50          # 50 x 160 ms = 8 s of audio before dropping


def spotter_mode(env: Any = None) -> str:
    """'act' | 'log' | 'off' from OPENBOT_STOP_SPOTTER (unset = 'act')."""
    raw = str((os.environ if env is None else env).get("OPENBOT_STOP_SPOTTER", "")).strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw == "log":
        return "log"
    return "act"


def make_vosk_recognizer(model: Any, rate: int = 16000) -> Any:
    """A FULL-vocabulary KaldiRecognizer over ``model``, with words.

    NOT a restricted grammar (unlike the wake word): MEASURED 2026-10-16 with
    the box's model (vosk-model-small-en-us-0.15) on piper speech, a
    stop-words-only grammar forced ordinary robot sentences onto the grammar
    ('the top shelf' -> 'stop', 'ready for the next' -> 'abort' / 'freeze';
    4 of 4 sentences gave a false hit, one a motion stop).  The full
    vocabulary decoded the same sentences as themselves (0 of 5 false hits)."""
    from vosk import KaldiRecognizer  # lazy: laptop tests inject a fake
    rec = KaldiRecognizer(model, rate)
    try:
        rec.SetWords(True)
    except Exception:  # noqa: BLE001 - an old vosk that has no per-word conf
        log.warning("stop spotter: SetWords unavailable - confidence treated as 1.0")
    return rec


class StopSpotter(object):
    """See the module docstring.  ``process`` is the synchronous core (tests
    drive it directly); ``start`` + ``submit`` run it on a worker thread."""

    def __init__(self,
                 recognizer_factory: Callable[[], Any],
                 on_cut_speech: Callable[[str], None],
                 on_stop: Callable[[str], None],
                 on_stop_talking: Callable[[str], None],
                 speaking: Callable[[], bool],
                 mode: str = "act",
                 tail_s: float = DEFAULT_TAIL_S,
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 clock: Callable[[], float] = time.monotonic,
                 record: Optional[Callable[[dict], None]] = None) -> None:
        self._factory = recognizer_factory
        self._rec = None  # first frame; model load stays off boot
        self.on_cut_speech = on_cut_speech
        self.on_stop = on_stop
        self.on_stop_talking = on_stop_talking
        self.speaking = speaking
        self.mode = mode
        self.tail_s = float(tail_s)
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self.record = record
        self._lock = threading.Lock()
        self._echo_armed = False
        self._last_speaking = float("-inf")
        self._acted_this_segment = False
        self._spoke_cut = False
        self._pending = None      # None | 'word' (stop at end) | 'alias'
        self._last_action = float("-inf")
        self._q = queue.Queue(maxsize=QUEUE_FRAMES)
        self._thread = None
        self._stop_evt = threading.Event()
        self.dropped_frames = 0
        self.skipped = 0           # quiet frames skipped (CPU saver)
        self._quiet = 0
        self.busy_s = 0.0          # time spent inside the recognizer
        self.frames = 0
        self.events = []  # type: List[dict]
        self._broken = False

    # -- keeping track of echoes (any thread) ------------------------------
    def note_spoken(self, text: str) -> None:
        """The robot is about to say / is saying ``text``."""
        low = (text or "").lower()
        if any(s in low for s in ECHO_STEMS):
            with self._lock:
                self._echo_armed = True
                self._last_speaking = self.clock()

