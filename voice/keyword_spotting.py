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


