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

    def _echo_suppressed(self, now: float) -> bool:
        with self._lock:
            if self._echo_armed and now - self._last_speaking > self.tail_s:
                self._echo_armed = False
            return self._echo_armed

    # -- worker thread ------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="stop-spotter")
        self._thread.start()

    def close(self) -> None:
        self._stop_evt.set()

    def submit(self, frame: bytes) -> None:
        """Hand one raw 16 kHz PCM16 mic frame over (pump thread; never blocks)."""
        if self._broken or not frame:
            return
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self.dropped_frames += 1
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass

    def _run(self) -> None:
        last_report = self.clock()
        while not self._stop_evt.is_set():
            try:
                frame = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.process(frame)
            except Exception:  # noqa: BLE001 - the spotter must never take the voice app down with it
                log.exception("stop spotter failed; disabled (the cloud stop route still works)")
                self._broken = True
                return
            now = self.clock()
            if now - last_report >= 60.0:
                log.info("stop spotter: %d frames, recognizer busy %.1f%% of a core, "
                         "%d frames dropped", self.frames,
                         100.0 * self.busy_s / max(1e-6, now - last_report),
                         self.dropped_frames)
                self.busy_s = 0.0
                self.frames = 0
                last_report = now

    # -- the actual logic -----------------------------------------------------
    def process(self, frame: bytes) -> None:
        now = self.clock()
        try:
            if self.speaking():
                with self._lock:
                    self._last_speaking = now
        except Exception:  # noqa: BLE001 - an unknown speaker state means not speaking
            pass
        if _peak(frame) < QUIET_PEAK:
            self._quiet += 1
            if self._quiet > HANGOVER_FRAMES:
                self.skipped += 1
                return
        else:
            self._quiet = 0
        if self._rec is None:
            self._rec = self._factory()
        t0 = time.perf_counter()
        final = self._rec.AcceptWaveform(frame)
        if final:
            res = json.loads(self._rec.Result() or "{}")
        else:
            res = json.loads(self._rec.PartialResult() or "{}")
        self.busy_s += time.perf_counter() - t0
        self.frames += 1
        if final:
            self._on_final(res, now)
            self._acted_this_segment = False
            self._spoke_cut = False
            self._pending = None
        else:
            self._on_partial(str(res.get("partial") or ""), now)

    def _event(self, kind: str, text: str, acted: bool, why: str = "") -> None:
        ev = {"event": "stop_spotter", "kind": kind, "text": text,
              "acted": acted, "why": why}
        self.events.append(ev)
        del self.events[:-50]
        log.info("stop spotter: %s %r -> %s%s", kind, text,
                 "ACT" if acted else "no action", (" (%s)" % why) if why else "")
        if self.record is not None:
            try:
                self.record(ev)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _norm(tokens: List[str]) -> List[str]:
        """Drop '[unk]'; a 'top' right before talking/speaking is 'stop'
        (the model's reading of a clipped "Stop talking.", MEASURED)."""
        toks = [t for t in tokens if t and t != "[unk]"]
        return ["stop" if (t == "top" and i + 1 < len(toks)
                           and toks[i + 1] in SPEECH_ONLY_NEXT) else t
                for i, t in enumerate(toks)]

    @staticmethod
    def classify(tokens: List[str]) -> Optional[str]:
        """'stop' | 'stop_talking' | 'negated' | None for one FINAL segment."""
        toks = StopSpotter._norm(tokens)
        if toks and "top" in toks and set(toks) <= COMMAND_WORDS:
            return "stop"               # e.g. 'top', 'top it', 'hey stewart top'
        for i, tok in enumerate(toks):
            if tok not in STOP_WORDS:
                continue
            if i > 0 and toks[i - 1] in NEGATIONS:
                return "negated"
            if tok == "stop" and i + 1 < len(toks) and toks[i + 1] in SPEECH_ONLY_NEXT:
                return "stop_talking"
            return "stop"
        return None

    def _on_partial(self, text: str, now: float) -> None:
        """Earliest evidence.  A halt word other than 'stop' acts at once;
        'stop' acts at once when the word after it is already heard and is
        not talking/speaking; a trailing 'stop' (or a lone 'top') only cuts
        the speech now and leaves the motion decision to the final."""
        if self._acted_this_segment:
            return
        toks = self._norm(text.split())
        for i, tok in enumerate(toks):
            if tok not in STOP_WORDS:
                continue
            if i > 0 and toks[i - 1] in NEGATIONS:
                return
            if tok != "stop":
                self._act("stop", "partial", text, now)
                return
            if i + 1 < len(toks):
                if toks[i + 1] in SPEECH_ONLY_NEXT:
                    self._cut("partial", text, now)   # final -> treat it as stop talking
                    self._pending = "word"
                else:
                    self._act("stop", "partial", text, now)
                return
            self._cut("partial", text, now)
            self._pending = "word"
            return
        if toks == ["top"] and self._pending is None:
            self._cut("partial", text, now)
            self._pending = "alias"

