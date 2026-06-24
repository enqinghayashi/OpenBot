"""EchoGuard: the level-bar defence against the robot hearing itself.

There is no acoustic echo cancellation anywhere in the voice path, and an
adaptive filter was measured (OpenBot protocol reference,
audio/echo_guard.py) at only ~10 dB of echo return loss at full volume -
below its own usable threshold.  So the defence is a level bar, honest
about the same physics: a microphone frame is admitted only while nothing
of ours is audibly playing, or when its peak beats what we are playing
times a coupling learned from the room times a margin.

This slice's guard is deliberately the simple, pure-stdlib core of that
idea (the reference's full double-talk ducking axis lives in a later
slice).  Constants come from the reference config: bootstrap coupling 0.5
(err toward suppressing while the room is unknown), live margin 1.6,
a ceiling of 12000 int16 (measured speech peaks 2700-14674, so this keeps
'uninterruptible' off the table), and ``ref_active_peak`` 2000 - below
that the speaker is not audibly running and the guard stands down
entirely (measured room tone peaks near 1100).

Pure stdlib (``array`` for the peak), so it imports on any laptop.
"""

from __future__ import annotations

import array
from typing import Optional

#: Below this the speaker is quiet enough that the mic can't mistake it for
#: speech, so the guard stands down.  G1 room tone peaks near 1100.
REF_ACTIVE_PEAK = 2000.0

#: Above this the mic hears the speaker louder than we sent - that's a
#: broken calibration, not a room.
MAX_COUPLING = 2.0


class EchoGuard:
    """Level bar: ``admit``/``maybe`` per microphone frame.

    ``playing_peak`` is the loudness of what the speaker is currently
    playing (0.0 / ``None`` = nothing playing).  The bar is
    ``coupling * margin * playing_peak``, capped at ``max_peak``; a frame
    whose own peak clears the bar is admitted.
    """

    def __init__(
        self,
        coupling: float = 0.5,
        margin: float = 1.6,
        max_peak: float = 12000.0,
        ref_active_peak: float = REF_ACTIVE_PEAK,
    ) -> None:
        if not 0.0 <= coupling <= MAX_COUPLING:
            raise ValueError("coupling must be 0-{}".format(MAX_COUPLING))
        self.coupling = coupling
        self.margin = margin
        self.max_peak = max_peak
        self.ref_active_peak = ref_active_peak

        self.admitted = 0
        self.suppressed = 0

