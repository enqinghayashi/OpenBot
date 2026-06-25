"""StewardKws: offline wake/sleep word spotting with a restricted grammar.

Vosk with a restricted grammar is proven on the G1 (OpenBot protocol
reference, wake/detector.py): a small model is reliable for one fixed
phrase only when the grammar is restricted to it - without that, a general
model considers the whole language and mishears far more.  The grammar
here is exactly the configured wake phrases — wake-only since the user
directive of 2026-09-10 removed the sleep family — so the recogniser can
only hear those words.

The robot's name has homophones (the recogniser may spell what it hears
"steward", "stewart" or "stuart"), so every phrase list carries the
family, and the default wake list follows that pattern.  Wake words are
checked on *partial* as well as final results - measured on the G1, the
partials are the stronger signal.

``vosk`` is imported lazily inside the model build, so importing this
module never requires it; laptop code uses ``MockKws`` (also used by
``--audio mock``).  Feed it 16 kHz mono PCM16 - that is what Vosk models
are trained at.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import List, Optional, Sequence

log = logging.getLogger("voice.kws")

#: vosk models are trained at 16 kHz
VOSK_RATE = 16000

# the robot is the OpenBot steward; the recogniser spells it a few
# ways (steward/stewart/stuart), so every list carries the family.
# constructor can override - the director has the final say.
DEFAULT_WAKE_PHRASES = (
    "hey steward", "hi steward", "hello steward",
    "hey stewart", "hey stuart",
)
# wake words only. the sleep family
# ('sleep', 'go to sleep', 'power down', 'stand down') is gone -
# no sleep phrase can be heard, the window just times out
# (voice.app LISTEN_WINDOW_S). sleep_phrases stays, empty, for config.
DEFAULT_SLEEP_PHRASES: tuple = ()


class KwsError(RuntimeError):
    """The wake-word engine could not be built or run."""


class StewardKws:
    """Restricted-grammar Vosk wake/sleep detector.

    ``feed(pcm16k)`` returns the matched phrase (wake or sleep) or None;
    ``feed_text(text)`` does the same on already-recognised text (tests,
    or the offline lane's transcript).  Accepts are spaced by
    ``cooldown_s``; after a hit the recogniser is reset so the same words
    cannot immediately re-trigger from a stale partial.
    """

    def __init__(
        self,
        model_dir: str,
        phrases: Sequence[str] = DEFAULT_WAKE_PHRASES,
        sleep_phrases: Sequence[str] = DEFAULT_SLEEP_PHRASES,
        cooldown_s: float = 3.0,
    ) -> None:
        self.model_dir = model_dir
        self.phrases = [p.strip().lower() for p in phrases if p.strip()]
        self.sleep_phrases = [p.strip().lower() for p in sleep_phrases if p.strip()]
        self.cooldown_s = cooldown_s
        self._model = None  # type: Optional[object]
        self._recognizer = None  # type: Optional[object]
        self._lock = threading.RLock()
        self._last_fired = float("-inf")  # -inf not 0: time base is boot-relative
        if not self.phrases:
            raise KwsError("at least one wake phrase is required")

    # ------------------------------------------- built lazily on first use

    def _ensure_engine(self) -> None:
        """Build the Vosk model + recognizer once, on first use."""
        with self._lock:
            if self._recognizer is not None:
                return
            if not os.path.isdir(self.model_dir):
                raise KwsError(
                    "wake-word model not found at {}: run the model fetch "
                    "script".format(self.model_dir)
                )
            try:
                from vosk import KaldiRecognizer, Model, SetLogLevel
            except ImportError as exc:
                raise KwsError("vosk is not installed: {}".format(exc))
            SetLogLevel(-1)  # else Kaldi chatters on stdout
            self._model = Model(self.model_dir)
            self._recognizer = KaldiRecognizer(
                self._model, VOSK_RATE, self._grammar()
            )
            log.info(
                "steward kws ready: %s | sleep: %s (model %s)",
                ", ".join(repr(p) for p in self.phrases),
                ", ".join(repr(p) for p in self.sleep_phrases) or "none",
                os.path.basename(self.model_dir),
            )

    def _grammar(self) -> str:
        words = set(self.phrases + self.sleep_phrases)
        return json.dumps(sorted(words))

    # ----------------------------------------------------- the public api

    def feed(self, pcm16k: bytes) -> Optional[str]:
        """One 16 kHz mono PCM16 chunk; returns the matched phrase or None."""
        if not pcm16k:
            return None
        try:
            self._ensure_engine()
        except KwsError:
            raise
        text = ""
        try:
            with self._lock:
                recognizer = self._recognizer
                if recognizer is None:  # pragma: no cover - _ensure_engine ran
                    return None
                if recognizer.AcceptWaveform(pcm16k):
                    text = json.loads(recognizer.Result()).get("text", "")
                else:
                    text = json.loads(recognizer.PartialResult()).get("partial", "")
        except Exception as exc:  # noqa: BLE001 - a recogniser fault must
            # not take down the capture path; the next chunk simply retries.
            log.warning("steward kws recogniser error: %s", exc)
            return None
        return self._match(text)

    def feed_text(self, text: str) -> Optional[str]:
        """Match already-recognised text (tests / offline transcript lane)."""
        return self._match(text or "")

    def _match(self, text: str) -> Optional[str]:
        text = (text or "").strip().lower()
        if not text:
            return None
        now = time.monotonic()
        if now - self._last_fired < self.cooldown_s:
            return None
        # sleep first - "go to sleep" contains the wake family
        for phrase in self.sleep_phrases:
            if phrase in text:
                self._fire(now)
                return phrase
        for phrase in self.phrases:
            if phrase in text:
                self._fire(now)
                return phrase
        return None

    def _fire(self, now: float) -> None:
        self._last_fired = now
        # drop recogniser state; the same words can't re-fire
        # straight away from a stale partial
        try:
            with self._lock:
                if self._recognizer is not None:
                    self._recognizer.Reset()
        except Exception:  # noqa: BLE001 - older vosk builds lack Reset()
            self._recognizer = None

    def reset(self) -> None:
        """Drop recogniser state (used on session boundaries)."""
        with self._lock:
            if self._recognizer is not None:
                try:
                    self._recognizer.Reset()
                except Exception:  # noqa: BLE001 - rebuilt next feed
                    self._recognizer = None


class MockKws:
    """A scripted wake-word detector for tests and ``--audio mock``.

    ``script`` is a list consumed one entry per ``feed``/``feed_text``
    call: a string is returned as the matched phrase, ``None`` returns
    None.  Useful for walking a robot through wake -> conversation ->
    sleep without any model.

    MIRROR SURFACE: the app's pump reads ``kws.phrases`` and
    ``kws.sleep_phrases`` in BOTH lanes to classify a matched phrase
    (wake vs sleep), so the mock exposes exactly the StewardKws
    attributes, normalised the same way (stripped, lower-cased,
    empties dropped) and defaulted the same way — the mock never
    invents a new API.  ``phrases``/``sleep_phrases`` mirror the
    constructor args (the app passes the same wake list it would give
    StewardKws); ``cooldown_s`` is accepted for signature parity (the
    scripted mock never enforces a cooldown itself).
    """

    def __init__(
        self,
        script: Optional[Sequence] = None,
        phrases: Sequence[str] = DEFAULT_WAKE_PHRASES,
        sleep_phrases: Sequence[str] = DEFAULT_SLEEP_PHRASES,
        cooldown_s: float = 3.0,
    ) -> None:
        self.script = list(script or [])  # type: List
        self.calls = 0
        self.phrases = [p.strip().lower() for p in phrases if p.strip()]
        self.sleep_phrases = [p.strip().lower() for p in sleep_phrases if p.strip()]
        self.cooldown_s = float(cooldown_s)

