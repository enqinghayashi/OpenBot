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


def _finite(value, name):
    """Coerce ``value`` to ``float`` or raise ``ValueError``.

    Used by the number-producing functions only (see the module docstring's
    error discipline): a height that did not arrive is an error, never a 0.0.
    """
    if not _is_finite_number(value):
        raise ValueError(
            "descent_law: %s must be a finite number, got %r" % (name, value))
    return float(value)


def _finite_descent(value, name):
    """:func:`_finite` plus the descent-budget invariant ``>= 0``.

    A negative budget is not a small budget: ``clamp(COMFORT_Z - z, 0.0, cap)``
    with ``cap < 0`` is an empty interval, so there is no honest number to
    return.
    """
    budget = _finite(value, name)
    if budget < 0.0:
        raise ValueError(
            "descent_law: %s must be >= 0, got %r" % (name, value))
    return budget


# ────────────────────────────────────────────────────────────── the law itself
def descent_for_z(z, max_descent_m):
    """Desired body descent for a target at torso-frame ``z``, in metres.

    ``d(z) = clamp(COMFORT_Z - z, 0.0, max_descent_m)``.  One clamped linear
    function: zero at and above :data:`COMFORT_Z` (the normal branch), then
    proportional, then saturated at the caller's budget — and the saturated
    band is the rescue band.  ``max_descent_m`` is the caller's: it is
    ``height_stand_m - height_min_m``, the writer clamp's own band.

    Raises ``ValueError`` for a non-finite ``z`` or a non-finite/negative
    ``max_descent_m`` (see the module docstring's error discipline).
    """
    zf = _finite(z, "z")
    budget = _finite_descent(max_descent_m, "max_descent_m")
    return min(max(COMFORT_Z - zf, 0.0), budget)


def rescue_floor(tabletop_min_z, max_descent_m):
    """The lowest torso-frame ``z`` the body can still bring into reach.

    Derived on EVERY call — ``tabletop_min_z - max_descent_m`` — so no number
    for it exists anywhere in this file, and widening the writer's band (or
    changing the envelope's floor) moves it without a code change.

    ``max_descent_m`` is the caller's budget (``height_stand_m -
    height_min_m``), never a constant here.  Raises ``ValueError`` for a
    non-finite input or a negative budget, so a caller can never receive a
    fabricated floor.
    """
    return (_finite(tabletop_min_z, "tabletop_min_z")
            - _finite_descent(max_descent_m, "max_descent_m"))


def classify_pick_z(z, tabletop_min_z, max_descent_m):
    """The branch a pick target falls in, as one of four EXACT strings.

        ``"ABSENT"``      ``z`` is ``None`` or not a finite number
        ``"NORMAL"``      ``z >= COMFORT_Z`` — today's path, byte-identical
        ``"DESCEND"``     ``rescue_floor(...) <= z < COMFORT_Z`` — the new lane
        ``"DEEP_FLOOR"``  ``z < rescue_floor(...)`` — today's honest refusal

    The floor is computed here from the two arguments, so the boundary is
    inclusive at the bottom: ``z`` exactly AT the floor is ``"DESCEND"``.
    ``"ABSENT"`` is returned WITHOUT touching the floor arguments — an absent
    detection is not a config probe.
    """
    if not _is_finite_number(z):
        return "ABSENT"
    if z >= COMFORT_Z:
        return "NORMAL"
    if z >= rescue_floor(tabletop_min_z, max_descent_m):
        return "DESCEND"
    return "DEEP_FLOOR"


def engine_z(z, stand_m, confirmed_ref_m):
    """The z the engine must be commanded, in the DESCENDED torso frame.

    ``z + (stand_m - confirmed_ref_m)``.  The torso frame rides with the torso,
    so a body that has descended by ``stand_m - confirmed_ref_m`` sees the same
    fixed object ``that much`` higher.  ``confirmed_ref_m`` is the CONFIRMED
    reference read back from the writer's status — never the requested command
    — which is what makes the precondition fail-closed rather than assumed.

    That 1:1 torso-frame delta is INFERRED from the frame definition, not
    measured; the torso-pitch coupling is the unverified term (see the module
    docstring).  With ``confirmed_ref_m == stand_m`` this is the identity, so
    the lane-off path cannot perturb a target.
    """
    zf = _finite(z, "z")
    stand = _finite(stand_m, "stand_m")
    ref = _finite(confirmed_ref_m, "confirmed_ref_m")
    return zf + (stand - ref)


def same_height(h_need, h_cur, tol_m):
    """The operator's hold rule: is the height we need the height we have?

    ``abs(h_need - h_cur) <= tol_m``.  A decision, not a measurement, so a
    non-finite input answers ``False`` — the conservative direction, which
    retreats first rather than HOLDING a body height nobody can vouch for.
    """
    if not (_is_finite_number(h_need) and _is_finite_number(h_cur)
            and _is_finite_number(tol_m)):
        return False
    return abs(h_need - h_cur) <= tol_m


# ────────────────────────────────────────── how the writer spells its height status
# the writer's real status output, copied line for line.
#
#   height=<ref>(stand <stand>)                inactive  -> active=False
#   height=<ref>-><target> active rem=<s>s     active    -> active=True
#   height=<ref> SUSPENDED-><t> rem=<s>s       suspended -> suspended=True
#
# the `(stand ...)` form is the no-op detector: it's exactly what a shut band
# produces, because `_apply_height` calls
# `_return_to_standing("height standing")` whenever the clamped value is within
# `height_stand_eps` of standing. so a shut band can't be mistaken for a
# descent. the fragment sits mid-line in the real reply - the writer's `status`
# verb returns "STATE: ... <cmd_auto> <height> <sonic>" - so the
# parser searches the line instead of anchoring to its start.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_RE_SUSPENDED = re.compile(
    r"height=(%s)\s+SUSPENDED->(%s)\s+rem=(%s)s" % (_NUM, _NUM, _NUM))
_RE_ACTIVE = re.compile(
    r"height=(%s)->(%s)\s+active\s+rem=(%s)s" % (_NUM, _NUM, _NUM))
_RE_STAND = re.compile(
    r"height=(%s)\s*\(stand\s+(%s)\)" % (_NUM, _NUM))


def parse_height_status(line):
    """Parse the writer's inline height fragment; ``None`` if unparseable.

    Returns a dict with these keys, or ``None``:

        ``ref``        last reference the policy was given (float)
        ``target``     tracked target (float), or ``None`` when nothing is
                       tracked
        ``stand``      nominal standing reference (float), or ``None``
        ``rem``        seconds left on the hold (float), or ``None``
        ``active``     True only for the ``active rem=`` form
        ``suspended``  True only for the ``SUSPENDED->`` form

    ``target`` is ``None`` — NOT a copy of ``stand`` — for the inactive form:
    the writer is tracking nothing, and copying the standing reference into
    ``target`` would let a ``|ref - target| <= tol`` test "confirm" a descent
    that never happened.  A caller that wants a confirmed descent must require
    ``active`` AND a matching pair.

    Fail-closed on everything else: a non-string line, a field that does not
    match the grammar, or a numeric field that is not finite, all return
    ``None``.  Never a guess, never a default.
    """
    if not isinstance(line, str):
        return None

    m = _RE_SUSPENDED.search(line)
    if m is not None:
        ref, target, rem = (float(m.group(1)), float(m.group(2)),
                            float(m.group(3)))
        if not all(math.isfinite(v) for v in (ref, target, rem)):
            return None
        return {"ref": ref, "target": target, "stand": None, "rem": rem,
                "active": False, "suspended": True}

    m = _RE_ACTIVE.search(line)
    if m is not None:
        ref, target, rem = (float(m.group(1)), float(m.group(2)),
                            float(m.group(3)))
        if not all(math.isfinite(v) for v in (ref, target, rem)):
            return None
        return {"ref": ref, "target": target, "stand": None, "rem": rem,
                "active": True, "suspended": False}

    m = _RE_STAND.search(line)
    if m is not None:
        ref, stand = float(m.group(1)), float(m.group(2))
        if not all(math.isfinite(v) for v in (ref, stand)):
            return None
        return {"ref": ref, "target": None, "stand": stand, "rem": None,
                "active": False, "suspended": False}

    return None


# ──────────────────────────────────────────────── the lane switch and the envelope
def effective_pick_floor(envelope):
    """The pick floor the brain's refusal nets should enforce, in metres.

    THE SWITCH IS READ, THE FLOOR IS NOT REWRITTEN.  With the lane switch off
    this returns ``envelope.pick_min_height_m`` UNCHANGED — the same object,
    read once, with no other attribute touched — so a config that omits the
    switch is indistinguishable from today's behaviour at the call site.

    With the lane ON the floor relaxes to
    ``rescue_floor(pick_min_height_m, height_stand_m - height_min_m)``: the
    tabletop floor is still the envelope's authority, and the writer's band is
    threaded in from the envelope's law inputs.

    Envelope fields read (the last three are added by the config work; this
    module assumes none of them):

        ``pick_min_height_m``   the tabletop floor — EXISTING field
        ``low_workspace_lane``  the switch; only a real ``bool`` ``True`` arms
                                the lane, mirroring the config loader's
                                strict-bool discipline — a string or a number
                                refuses rather than silently arming it
        ``height_stand_m``      nominal standing reference (m)
        ``height_min_m``        the writer clamp's band floor (m)

    FAIL-CLOSED.  If the switch is on but the band inputs are missing,
    non-numeric, non-finite, inverted, or zero-width, the lane cannot relax
    anything and this returns the UNRELAXED floor rather than raising — a
    misconfigured envelope must degrade to today's honest refusal, never to a
    crash and never to a silently widened workspace.  Reads are ``getattr`` so
    an envelope that predates the lane still constructs and still answers.
    """
    floor = getattr(envelope, "pick_min_height_m", None)
    if not _is_finite_number(floor):
        raise ValueError(
            "descent_law: envelope.pick_min_height_m must be a finite number, "
            "got %r" % (floor,))

    if getattr(envelope, "low_workspace_lane", False) is not True:
        return floor

    stand = getattr(envelope, "height_stand_m", None)
    hmin = getattr(envelope, "height_min_m", None)
    if not (_is_finite_number(stand) and _is_finite_number(hmin)):
        return floor
    if stand <= hmin:
        return floor
    return rescue_floor(floor, stand - hmin)
