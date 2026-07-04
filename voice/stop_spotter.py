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
