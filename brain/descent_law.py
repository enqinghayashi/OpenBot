"""brain/descent_law.py — the shared low-workspace condition law.

ONE LAW, TWO SKILLS.  Pick and place are highly correlated and inter-related
(the operator's words, 2026-09-27).  Neither skill owns the rule for "how far
down must the body go for this target" — both ask THIS module, and this module
holds no state.  That is what makes the two skills one coherent condition set
instead of two divergent branches.

WHAT PROBLEM IT SOLVES.  The pick lane is tabletop-only: a target whose
torso-frame ``z`` falls below the envelope's ``pick_min_height_m`` is a
capability-block and the robot refuses honestly without moving.  Ground objects
read far below that bound — but so does a legitimately LOW TABLE, and the body
can buy back part of that gap by lowering itself.  The realised descent law
(MuJoCo GR00T-Balance, **NOT hardware**) is bounded by the writer's hard clamp,
so the rescue budget is FINITE and the if/else cannot be a two-way branch on
one threshold.  It is one clamped linear ladder, not a table of cuts:

    d(z) = clamp(COMFORT_Z - z, 0.0, max_descent_m)        # body descent, m

``d(z) == 0`` for every ``z >= COMFORT_Z``, so the normal-tabletop branch is
EXACTLY where the function is zero and **no extra band-top constant is
invented**.  The band where the ladder SATURATES is the rescue band.  Both
bands ride the same branch; the split only decides which refusal text a
refusal replaces in the log.

The rescue floor is DERIVED, never written down:

    rescue_floor(tabletop_min_z, max_descent_m) = tabletop_min_z - max_descent_m

and ``max_descent_m`` is itself ``height_stand_m - height_min_m`` — the writer
clamp's own band, computed by the caller.  There is deliberately NO numeric
literal for the floor in this file: bring-up changes one config value and the
floor moves with it, with no code change.

ESTABLISHED, NOT ASSUMED.  The engine's pick handler already drives both arms
to READY as its own step 1 (``skills/arm/pick_engine_v3.py:3663``, ``:3680``,
``:3685``), and place already runs drop -> neutral internally
(``pick_engine_v3.py:6610-6666``).  So nothing here builds a redundant ready
step: this module only decides the BODY height the engine is handed.

REUSE LAW — one implementation of every fact.

  * ``TABLETOP_MIN_Z`` comes from the envelope (``Envelope.pick_min_height_m``,
    via :func:`effective_pick_floor`); it is never restated here.
  * The ``HEIGHT_*`` numbers (the standing reference, the band floor) are
    THREADED IN as arguments — ``max_descent_m``, ``stand_m``,
    ``confirmed_ref_m``, ``tabletop_min_z``.  This module owns no writer
    constant and hardcodes no band number, which is what makes it unit-testable
    with no gateway, no rclpy and no writer.
  * The writer's status grammar is parsed in exactly ONE place
    (:func:`parse_height_status`), and its three forms are taken from the real
    emitter at ``skills/writer/main_merged_writer.py:758-768``.

ERROR DISCIPLINE — a number-producing function never fabricates a number.

  * :func:`descent_for_z`, :func:`rescue_floor` and :func:`engine_z` raise
    ``ValueError`` on a missing, non-finite, or (for a descent budget)
    negative input.  Silently returning ``0.0`` for an unknown ``z`` would read
    as "no descent needed" — a wrong answer in the PERMISSIVE direction.
  * A decision function answers conservatively instead: :func:`classify_pick_z`
    returns ``"ABSENT"`` and :func:`same_height` returns ``False``, so a caller
    that forgets to guard cannot descend, and cannot HOLD, on garbage.
  * :func:`parse_height_status` returns ``None`` on anything it cannot parse,
    including a non-finite numeric field.  It never guesses, never defaults.

HONESTY BOUNDARY.  This module is arithmetic.  Everything downstream that
reports a descent is reporting the writer's REFERENCE — a command, not a
reading — because there is no body-height sensor anywhere in this system.
Nothing here may be read as a MEASUREMENT of where the body physically is.
The numbers behind the ladder are SIMULATED (MuJoCo GR00T-Balance, never
hardware-measured), and the torso-pitch coupling — the policy also pitches the
torso, measured 53.9 deg standing / 37.5 deg at the deepest squat — is the
explicitly UNVERIFIED term, so the 1:1 torso-frame delta remains inferred.

Stdlib only.  Imports nothing from ``gateway/`` and nothing from ``brain/``:
every platform number arrives as a function argument.
"""

from __future__ import annotations

import math
import re

__all__ = [
    "COMFORT_Z",
    "descent_for_z",
    "rescue_floor",
    "classify_pick_z",
    "engine_z",
    "same_height",
    "parse_height_status",
    "effective_pick_floor",
]

#: how high the arm likes to work in the torso frame. it sits inside the
#: tabletop range we actually saw live, so ``descent_for_z`` is zero for any
#: ``z`` at or above it - the normal branch is the ladder's zero, which is why
#: we don't need a second threshold constant to split it off from the rescue band.
COMFORT_Z = 0.0


# ─────────────────────────────────────────────────────────────── sanity checks
def _is_finite_number(value):
    """True only for a real, finite ``int``/``float``.

    ``bool`` is excluded on purpose: ``True`` is an ``int`` in Python, and a
    boolean that reached a height comparison would be a wiring bug, not a
    height.  Explicit ``math.isfinite`` — never a comparison that silently
    falls through on ``None`` or ``nan``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)
