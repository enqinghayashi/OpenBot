"""gateway/descent_lane.py — the low-workspace descent sequence machine.

A target whose torso-frame ``z`` is below the tabletop floor is today a
capability-block: the robot refuses honestly and does not move.  Some of that
band is not truly out of reach — the BODY can buy part of it back by lowering
itself (bounded by the writer's hard clamp), and this module is the piece of
the gateway that sequences that: descend -> pick/place -> verify ->
retreat-or-hold, as ONE coherent condition set shared by both skills.

WHY THIS LIVES IN THE GATEWAY AND NOT IN THE BRAIN OR THE ENGINE.  The gateway
owns all three sockets (writer / grip / engine), it is the last refusal before
the engine, and it can CONFIRM a descent by polling the writer's own status.
The brain has no writer socket; the engine's ``writer_arm_client`` has no
``height()`` method at all.  See the design plan §5 — the guard is one choke
point in ``GatewayNode.dispatch``, and ``_do_pick`` / ``_do_place`` are never
modified.

WHY THIS IS A MODULE AND NOT A NODE.  Same reason as ``gateway/place_target.py``
(READ THAT FILE FIRST — it is the template): the fail-closed decision, the
phase bookkeeping and the confirm/refuse text are pure functions of a handful
of numbers and a few injected callables.  Keeping them here, stdlib-only and
rclpy-free, is what makes the whole sequence exercisable in a unit test with
fakes and no gateway, no ROS and no robot.

INJECTED CALLABLES — THE WHOLE REASON THIS IS TESTABLE.  Nothing here reaches
for a global, a socket or a node.  ``pick_sequence`` / ``place_sequence`` take
the gateway's own callables as arguments:

    ``writer_verb(line) -> str``   one line to the writer socket, its reply
    ``status_verb() -> str``       one ``status`` poll (a ``writer_verb``
                                   bound to ``"status"``)
    ``do_pick(verb, args)``        ``GatewayNode._do_pick``  -> (reply, verdict)
    ``do_place(args)``             ``GatewayNode._do_place`` -> (reply, verdict)

plus ``sleep`` / ``monotonic`` so a test never waits 3 real seconds.

REUSE LAW — one implementation of every fact (plan §9 W2).

  * the condition law, the ladder, the floor, ``engine_z``, ``same_height``
    and the writer's status grammar -> ``brain.descent_law`` (W1, landed) —
    nothing about the law is restated here;
  * ``PhaseLog`` / ``render_table`` / ``status_message`` / ``DETAIL_WRAP`` ->
    ``gateway.place_target``.  This module subclasses ``PhaseLog`` as
    ``DescentPhaseLog`` and overrides ``header()`` ONLY: a second table
    renderer would be a divergent copy of the one measured-elapsed format the
    repo already has.  The gateway renders this lane's log with
    ``place_target.render_table`` and publishes with
    ``place_target.status_message``, exactly as it does for ``/place_target``;
  * the grasp check -> the FOLDED ``_do_pick`` pair.  There is no standalone
    "verify grasp"; ``_do_pick`` returns ``(reply, self._verify("pick", reply))``
    and stamps ``lane_state=`` / ``grip_state_ts=`` into the reply.  That folded
    pair IS the grasp check, and the lane reads it rather than re-deriving one.

FAIL-CLOSED (plan §7).  A descent counts as CONFIRMED only on BOTH

    (a) the writer answering a line starting ``DONE`` to ``height <cmd>``, and
    (b) a ``status`` poll inside the timeout whose height fragment shows a
        ``ref`` that has reached the COMMANDED height within ``tol``, has LEFT
        standing (by more than the confirm band itself), and is framed by the
        writer's OWN reported standing reference.

Anything else — an ERROR, a raised socket error, a timeout, an unparseable
fragment, a SUSPENDED hold, or the ``(stand ...)`` form a SHUT BAND produces —
REFUSES the whole motion and the engine is NEVER called.  A shut band cannot be
mistaken for a descent: the writer's ``_apply_height`` returns to standing
whenever the clamped value is within ``height_stand_eps`` of nominal, which is
exactly the ``(stand 0.74)`` form the parser reports with ``target`` None.

A CONFIRMED DESCENT IS CHECKED, NOT MERELY REMEMBERED — AND NEVER RENEWED.  The
writer's ``height <m>`` verb is a LIFETIME, not a liveness probe
(``main_merged_writer.py:716,734`` sets a monotonic deadline; ``:1660-1662``
re-stands on the clock alone, regardless of whether the driving process is
alive, and a ``status`` poll does not refresh it).  A descent confirmed once can
therefore expire while the engine is still working.  There is deliberately NO
background thread here re-issuing the command to hold the body down: the writer
already owns that mechanism, and it is a CONFIG value — ``height_hold_s``
(``skills/writer/config/merge_config.yaml:61``) is the lifetime of ONE
``height`` command, set long enough to cover a whole descent->pick->place, so
no renewal thread is needed at all.  What the lane adds instead is a
FAIL-CLOSED OPERATION-BUDGET GUARD in front of every descended dispatch: a
single FRESH synchronous ``status`` poll (see :func:`probe_active`) that must
still find the writer ACTIVE at the held height, plus
:func:`_lifetime_refusal`, which REFUSES the dispatch — loudly, naming
``height_hold_s`` — unless the writer's own reported remaining hold covers
:data:`LANE_OPERATION_BUDGET_S` ALONE.  (The descent's own slew margin is NOT
added to that bar; it moves the ``height_hold_s`` the refusal tells you to SET,
because the writer's clock has already spent it by the time ``rem`` is read.)
A short lifetime is refused and named, never renewed behind the operator's
back.  A hold is entered only when that poll succeeds, and its window is
``min(HOLD_WINDOW_S, the writer's own remaining hold)``.

THE OPERATION IS CHECKED AFTERWARDS TOO, ON EVERY PATH (D4).  A guard in front
of the dispatch says the body was low when the arm was COMMANDED; it says
nothing about the seconds that follow.  So BOTH sequences take one more writer
poll after the engine returns — on the grasp-succeeded and the grasp-FAILED
path alike — and compare the MEASURED elapsed of the engine call against
:data:`LANE_OPERATION_BUDGET_S`.  A writer that is no longer holding the
commanded reference, or a call that outran the budget it declared, is a LOUD
``body_restood`` row and a DOWNGRADED verdict: an operation the lane cannot
vouch for is never reported as success.  The same anomaly refuses the hold, so
an unvouched-for height is never handed on to a following place.

THE HOLD WINDOW IS DISCLOSED, AND EVERY HOLD REPLY SAYS WHAT IT REALLY IS
(D2/D5, corrected 2026-10-03, R4 HRC-2).  The operator's sticks are clamped only
while the arm engine holds ``rc_hold`` — i.e. DURING an arm motion — and the
writer forces ``rc_live = False`` only under that clamp
(``skills/writer/main_merged_writer.py:2719-2742``).  The engine RELEASES the
clamp when the pick completes (``skills/arm/pick_engine_v3.py``,
``_hold_rc_writer(arm, False, 'pick complete (step 5)')`` on both the right and
the left pick path), so
inside the hold window that follows a pick a stick push SUSPENDS the squat.
The old text here ("the sticks CANNOT take the body back" for the whole hold)
was false for exactly that window.  DERIVED from source; the box's behaviour is
UNVERIFIABLE-FROM-HERE.  The window is BOUNDED (the writer's own
``height_hold_s`` lifetime re-stands the body, and the lane gives it back on
retreat) and REPORTED: :func:`lockout_notice` is appended to every reply and
row that enters or continues a hold.

HONESTY BOUNDARY.  There is NO body-height sensor in this system, so every
descent/retreat claim made here is WRITER-REPORTED — a statement about the
reference the policy is being told to track, never a measurement of where the
body physically is.  ``HONESTY_SUFFIX`` is appended to every descent/retreat
verdict, reply and phase detail, mirroring ``brain/verifier.py:1755``'s
``(engine-reported, NOT sensor-measured — no posture sensor exists)`` for the
pose boundary.  The ladder's own numbers are SIMULATED (MuJoCo
GR00T-Balance), never hardware-measured, and the torso-pitch coupling is the
explicitly unverified term — see ``brain/descent_law``.
"""

from __future__ import annotations

import math
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from brain import descent_law
from gateway import place_target as _place_target

__all__ = [
    "DESCENT_BUSY_MSG",
    "HONESTY_SUFFIX",
    "RETREAT_UNCONFIRMED",
    "HEIGHT_CONFIRM_TOL_M",
    "DESCENT_CONFIRM_TIMEOUT_S",
    "DESCENT_POLL_INTERVAL_S",
    "HOLD_TOL_M",
    "HOLD_WINDOW_S",
    "STAND_EPS_M",
    "MIN_CONFIRMED_DESCENT_M",
    "NO_DESCENT_BAND_M",
    "descent_negligible",
    "LANE_OPERATION_BUDGET_S",
    "LANE_PC_MARGIN_S",
    "HEIGHT_SLEW_RATE_M_S",
    "BODY_RESTOOD",
    "BUDGET_OVERRUN",
    "IDLE", "DESCENDING", "HOLDING", "RETREATING",
    "NEXT_NONE", "NEXT_HOLD", "NEXT_REDESCEND", "NEXT_RETREAT_BLOCKING",
    "NEXT_RETREAT_BEST_EFFORT",
    "LaneLaw",
    "DescentContext",
    "DescentPhaseLog",
    "usable",
    "height_command",
    "deep_floor_refusal",
    "honest",
    "lockout_notice",
    "slew_margin_s",
    "probe_active",
    "poll_descent",
    "poll_retreat",
    "next_command_action",
    "retreat_first_if_held",
    "pick_sequence",
    "place_sequence",
    "hold_unusable_reason",
    "ALREADY_STANDING",
    "parse_step_status",
]

# ── the lane lease ───────────────────────────────────────────────────────────
#: Refusal when a lane skill arrives while a descent is in flight. Concurrency
#: is refused, never queued: a queued skill would descend from, or retreat out
#: of, a body height the running one still uses.
DESCENT_BUSY_MSG = "ERROR:low_workspace busy - a descent is already in flight"

#: Appended to every descent and retreat verdict, reply and phase detail: there
#: is no body-height sensor, so the writer's own reference is the only honest
#: description of the evidence.
HONESTY_SUFFIX = ("writer-reported, NOT sensor-measured — "
                  "no body-height sensor exists")

#: Loud warn-and-hold marker, defined once so a row, a loud line and a reply
#: carry one string and the source holds a single occurrence of it.
RETREAT_UNCONFIRMED = "RETREAT UNCONFIRMED"

#: Reference agreement for a confirmed descent (m): the writer's own height ref
#: must reach the commanded target within this band.
HEIGHT_CONFIRM_TOL_M = 0.01
#: How long the status poll waits for the reference to arrive (s).
DESCENT_CONFIRM_TIMEOUT_S = 3.0
#: Poll spacing inside that window (s), a bound on writer round-trips.
DESCENT_POLL_INTERVAL_S = 0.1
#: Operator hold tolerance (m): "the same height" means within this distance of
#: body height, so a matching place may reuse a hold.
HOLD_TOL_M = 0.05
#: Upper bound on the hold window (s). `height <m>` is a lifetime on the writer,
#: not something a status poll refreshes, so a hold lasts the smaller of this and
#: the writer's own reported remaining hold, and that honoured window is reported.
HOLD_WINDOW_S = 45.0
#: The writer's own standing epsilon (m). A clamped height within it makes the
#: writer return to standing and track nothing, which is the `(stand ...)` form a
#: shut band produces. Not the lane's gate: the confirm band is an order larger.
STAND_EPS_M = 1e-3
#: Smallest realized descent a confirmation accepts (m), derived from the confirm
#: band: a move the band cannot tell from zero is not a descent, and reading it as
#: one would be a false success.
MIN_CONFIRMED_DESCENT_M = HEIGHT_CONFIRM_TOL_M
#: A required descent at or below this (m) is no descent: the lane runs the skill
#: at standing instead of commanding a move the confirmation would then refuse.
#: Twice the confirmable minimum, to absorb drift between the two stand values.
NO_DESCENT_BAND_M = 2.0 * MIN_CONFIRMED_DESCENT_M
#: Declared budget for one engine call (s): a judgment, not a guarantee. Far under
#: the 300 s transport reply timeout, which bounds a socket rather than a squat;
#: an overrun is reported and downgrades the verdict, never prevented.
LANE_OPERATION_BUDGET_S = 30.0
#: The writer's own reference slew rate (m/s), the rate limit that keeps the height
#: reference from stepping. A fact about the writer, held once; slew_margin_s
#: divides the clamp band by it.
HEIGHT_SLEW_RATE_M_S = 0.15
#: Marker for a body that had left the commanded reference when the engine
#: returned: one string for the row, the loud line and the reply.
BODY_RESTOOD = "BODY RE-STOOD DURING THE OPERATION"
#: Marker for an engine call that outran the declared lane budget.
BUDGET_OVERRUN = "OPERATION OVERRAN THE DECLARED LANE BUDGET"
#: Marker for a rise this lane did not perform: the writer already reported the
#: body standing before the lane sent `height off`.
ALREADY_STANDING = "BODY WAS ALREADY STANDING"

# ── optional point-cloud hold check (embodiment.envelope.pc_hold_check) ──────
#: Negative evidence only: STILL_ON_TABLE and MOVED are decisive, while
#: ABSENT_UNPROVEN and UNKNOWN prove nothing (a target in the hand and one the
#: detector missed look alike), so the hook can only downgrade a verdict.
#:
#: Margin (s) between the end of the hold check and the end of the writer's own
#: height lifetime; the consumer keeps the same margin under its 60 s span cap.
LANE_PC_MARGIN_S = 6.0
#: `pc.warm` time-to-live (s): long enough for DDS discovery to finish before the
#: baseline is requested.
LANE_PC_WARM_TTL_S = 20.0
#: Baseline request: the consumer-side TTL (s) and the lane-side deadline (s), the
#: deadline also being the spare writer hold the baseline needs. Without that
#: spare the baseline is skipped, never squeezed in.
LANE_PC_BASELINE_TTL_S = 2.0
LANE_PC_BASELINE_DEADLINE_S = 2.5
#: The hold check is asked only when at least LANE_PC_CHECK_MIN_S of the writer's
#: hold remains beyond the margin, and never waits past LANE_PC_CHECK_MAX_S; its
#: deadline is that window plus LANE_PC_DEADLINE_SLACK_S.
LANE_PC_CHECK_MIN_S = 3.0
LANE_PC_CHECK_MAX_S = 5.0
LANE_PC_DEADLINE_SLACK_S = 0.3
#: The only values `; pc_hold=` may carry; anything else the consumer says
#: (BASELINE_OK, RIM_FIT, a missing verdict) reads as UNKNOWN.
PC_HOLD_TOKENS = ("STILL_ON_TABLE", "MOVED", "ABSENT_UNPROVEN", "UNKNOWN")
#: The two verdicts that may downgrade a pick, both decisive negative evidence.
PC_DECISIVE_VERDICTS = ("STILL_ON_TABLE", "MOVED")

# ── optional pre-descent stance step (embodiment.envelope.pre_descent_step) ──
#: Legs the step widens, in order. One leg: the writer latches DONE and refuses
#: another `step shift` until `step off`, and `step go r` needs a negative
#: installed shift. The loop is parametric, so a chaining writer needs only this.
PRE_STEP_LEGS = ("l",)
#: Shift magnitude sent with `step shift` (deg), chosen to widen the stance
#: within the writer's own shift limits.
PRE_STEP_SHIFT_DEG = 10.0
#: Per-leg widening requested with `step go` (cm); two legs at this width stay
#: under the writer's 27.4 cm two-step ceiling. The writer clamps it and reports
#: the width it achieved.
PRE_STEP_CM = 12.5
#: Bound on the `step status` poll after `step go` (s). Generous, because the
#: step's own motion time is not modeled.
PRE_STEP_GO_TIMEOUT_S = 20.0
#: Smallest installed shift accepted as a real shift (deg). A shift the writer
#: clamped to near zero is a shut band, not a widening.
PRE_STEP_MIN_SHIFT_DEG = 1.0
#: The writer's own shift slew rate (deg/s), the rate the settle bound is derived
#: from. A slower unit makes the settle poll time out and refuse, never go early.
PRE_STEP_SLEW_DEG_S = 8.0
#: Multiplier and additive margin on the derived slew time for the settle poll,
#: covering tick jitter and one poll interval: 10 deg at 8 deg/s gives 3.5 s.
PRE_STEP_SHIFT_SLACK = 2.0
PRE_STEP_SHIFT_MARGIN_S = 1.0
#: Phases in which the writer accepts `step go`. A bare `step shift` reports SHIFT
#: until the slew settles and IDLE after, and any other phase answers "step busy".
PRE_STEP_GO_READY_PHASES = ("IDLE", "DONE")


def pre_step_shift_timeout_s(shift_deg: Any,
                             slew_deg_s: float = PRE_STEP_SLEW_DEG_S) -> float:
    """Bound (s) on the post-``step shift`` settle poll, DERIVED from the
    commanded shift and the writer's slew rate: ``|shift| / rate * SLACK +
    MARGIN``.  10 deg at 8 deg/s -> 1.25 s -> 3.5 s."""
    return (abs(float(shift_deg)) / float(slew_deg_s) * PRE_STEP_SHIFT_SLACK
            + PRE_STEP_SHIFT_MARGIN_S)

# ── the state machine's vocabulary ──────────────────────────────────────────
#: `DescentContext.state` values. A context that is not IDLE means a body height
#: is outstanding, so the next lane skill must refuse.
IDLE = "IDLE"
DESCENDING = "DESCENDING"
HOLDING = "HOLDING"
RETREATING = "RETREATING"

# ── next-command actions (the operator rule, written as a predicate) ─────────
NEXT_NONE = "NONE"                              # lane not involved
NEXT_HOLD = "HOLD"                              # run at the held height
NEXT_REDESCEND = "RETREAT_THEN_DESCEND"         # rise, then descend again
NEXT_RETREAT_BLOCKING = "RETREAT_BLOCKING"      # rise and confirm before acting
NEXT_RETREAT_BEST_EFFORT = "RETREAT_BEST_EFFORT"  # rise, warn loudly on failure

#: Verbs the lane owns. A lane skill's own height need decides hold versus
#: retreat-and-re-descend; everything else sits outside the lane.
LANE_SKILL_VERBS = frozenset({"pick", "pick_left", "place", "place_left"})

#: The place-family verbs, the only ones a release-only reuse may hold for: a
#: place can carry no coordinates at all, so its absent target height does not
#: mean an unknown target.
PLACE_FAMILY_VERBS = frozenset({"place", "place_left"})

#: Commands that need the body standing first. Squat-walking is unmeasured, so a
#: walk blocks until standing is confirmed; `cmd_set` is the writer's own
#: locomotion verb, and leaving it out would let a walk run at a lowered body.
BODY_BLOCKING_VERBS = frozenset({"navigate", "cmd_set"})

#: Commands that only need the body risen best-effort with a loud warning. An
#: arm-only posture cannot threaten the reach, so a failed rise must not block it.
BEST_EFFORT_VERBS = frozenset({"pose", "open_hand", "close_hand",
                               "engine", "writer"})

# There is no takeover verb set: nothing dispatches a takeover, suspend or estop
# verb through the gateway, so a "forget the hold" branch could never fire. The
# lane reports the lockout it holds instead of guessing at a release.


# ────────────────────────────────────────────────────── sanity checks on values
def _finite_num(value: Any) -> bool:
    """True only for a real, finite ``int``/``float`` (``bool`` excluded).

    ``bool`` is an ``int`` in Python; a boolean that reached a height
    comparison would be a wiring bug, and treating it as a height is the
    permissive direction.  Mirrors ``brain/descent_law._is_finite_number``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def release_only_args(args: Any) -> bool:
    """True when a lane skill's argument list carries NO coordinates at all.

    A RELEASE-ONLY place (the brain's ``place []``, seen on the robot at
    ``run18`` 20:14:36) is the one lane skill that aims at NO target height: it
    drops whatever the hand holds, exactly where the body already is.  That is
    what makes it safe to run at a HELD height — there is no target height for
    the held frame to mis-place (see :func:`next_command_action`).

    ONLY A LITERALLY EMPTY LIST QUALIFIES, and a caller with no argument list
    (``None``, the default on :func:`next_command_action`) gets False: "no
    information" must never read as "no coordinates", or a caller that forgot
    to thread the args would silently start reusing holds.  Anything else —
    one token, two tokens, a token that only fails to parse as a float, a
    non-sequence — is NOT release-only and keeps the fail-closed retreat: the
    cheap mislabel (retreating when a reuse was possible) is a lost reuse,
    while the expensive one (reusing when the place had coordinates) drops from
    a height nobody vetted.
    """
    if args is None:
        return False
    try:
        return len(args) == 0
    except TypeError:
        return False


def honest(text: Any) -> str:
    """``text`` ending with :data:`HONESTY_SUFFIX`, appended exactly once.

    Never double-appends, so a detail that is passed through two layers still
    reads as one statement.  The result always ENDS with the suffix: trailing
    punctuation after it would make the boundary look like an afterthought.
    """
    body = str(text or "").rstrip()
    if not body:
        return HONESTY_SUFFIX
    if body.endswith(HONESTY_SUFFIX):
        return body
    return "%s — %s" % (body, HONESTY_SUFFIX)


def _loud(text: str) -> None:
    """One LOUD stderr line.  The operator must not have to read a table.

    stderr, not stdout: a warning about a body that may still be low has to be
    visible even when the caller is rendering the phase table.
    """
    sys.stderr.write("!! [descent] %s\n" % text)
    sys.stderr.flush()


def lockout_notice(lockout_s: Any) -> str:
    """The operator-facing HOLD WINDOW statement: how long the body is held
    low, and what the sticks can and cannot do inside that window.

    WHY THIS EXISTS.  A held descent must never be SILENT (D2), and it must
    never be described as something it is not (R4 HRC-2, corrected
    2026-10-03).  The previous text said the operator's sticks "will NOT take
    it back during that window".  That is false for the hold window itself:
    the sticks are clamped only while the arm engine holds ``rc_hold`` — the
    writer forces ``rc_live = False`` under that clamp and nowhere else
    (``skills/writer/main_merged_writer.py:2719-2742``: the RC height-suspend
    is gated on ``rc_live``, and only the ``rc_hold`` / shutdown / autopilot
    branches zero it) — and the pick engine RELEASES the clamp when the pick
    completes (``skills/arm/pick_engine_v3.py``,
    ``_hold_rc_writer(arm, False, 'pick complete (step 5)')``).  So inside the
    window a stick push SUSPENDS the squat.  DERIVED from source; the box's
    behaviour is UNVERIFIABLE-FROM-HERE.  The window is bounded: the writer's
    own ``height_hold_s`` lifetime re-stands the body on the clock alone, and
    this lane gives the body back on its next retreat.

    NOT A DETECTOR.  This says what the lane IS holding; it makes no claim to
    observe the operator.  The lane cannot see a takeover, which is why the
    deleted ``TAKEOVER_VERBS`` table is not replaced by a guess (D3).

    ``lockout_s`` is the window the caller ACTUALLY honoured — for a hold entry
    that is ``DescentContext.holding``'s return (``min(hold_s, the writer's own
    remaining hold)``), never the un-capped request.  A non-finite or negative
    value reads as 0.0 rather than propagating a bad number into operator text.
    """
    window = (float(lockout_s)
              if _finite_num(lockout_s) and float(lockout_s) > 0.0 else 0.0)
    return ("HOLD WINDOW: the body is held low for up to %.1fs from now. The "
            "operator's sticks are clamped only while the arm engine holds "
            "`rc_hold` (during an arm motion; the writer console escape for "
            "that is `hold_rc off`); the engine releases that clamp when the "
            "pick completes, so inside this window a stick push suspends the "
            "squat. The writer's own `height_hold_s` lifetime re-stands the "
            "body at its deadline regardless, and the lane gives the body back "
            "on its next retreat" % window)


def slew_margin_s(law: Any) -> float:
    """The descent's OWN consumption of the writer's clock, in seconds.

    The writer's height reference is RATE LIMITED
    (:data:`HEIGHT_SLEW_RATE_M_S`, ``height_slew_rate`` in
    ``skills/writer/config/merge_config.yaml``), so by the time the guard reads
    the writer's remaining hold the clock is ALREADY down by the slew time of
    the move it just made: the full clamp band at that rate is ~1.3 s
    (merge_config.yaml:57-60: "covers the 0.19 m nominal->min span in ~1.3 s").
    That margin is NOT added to the guard's bar — the bar is the operation
    budget ALONE.  It is what the refusal ADDS to the ``height_hold_s`` it tells
    the operator to SET, and the measured D9 defect was exactly that arithmetic:
    a ``height_hold_s`` set to the bare bar does NOT dispatch, because ``rem`` at
    the gate is that value minus this margin, so the value that clears the bar is
    ``operation_budget_s + this margin``.

    DERIVED, never invented: the band is ``law.max_descent_m`` — the writer's
    own clamp band, the widest move this lane can command — divided by the
    writer's own slew rate.  A law that cannot supply a usable band yields 0.0
    (no margin), which degrades the guard to the pre-margin behaviour rather
    than refusing every dispatch over an arithmetic the caller cannot check.
    """
    try:
        band = float(law.max_descent_m)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if not _finite_num(band) or band <= 0.0:
        return 0.0
    return band / HEIGHT_SLEW_RATE_M_S


# ──────────────────────────────────────────────────── what the law is fed with
class LaneLaw:
    """The four law inputs, threaded in — never restated (REUSE LAW).

    ``tabletop_min_z`` is the envelope's ``pick_min_height_m`` (the authority);
    the two ``height_*`` numbers are the WRITER's own clamp band; ``hold_tol_m``
    is the operator's hold tolerance.  ``max_descent_m`` is DERIVED here —
    ``height_stand_m - height_min_m`` — exactly as ``brain.descent_law``
    requires, so no band number exists anywhere in this file as a literal and
    widening the writer's band moves the rescue floor with no code change.

    ``timeout_s`` is the confirm window threaded in from config
    (``embodiment.envelope.descent_confirm_timeout_s``).  It DEFAULTS to
    :data:`DESCENT_CONFIRM_TIMEOUT_S`, so a config that omits the key behaves
    exactly as before; a value that is not a finite positive number falls back to
    that same default rather than shrinking the window to nothing.

    ``operation_budget_s`` is the declared lane operation budget, threaded the
    SAME way: it defaults to :data:`LANE_OPERATION_BUDGET_S`, a value that is
    not a finite positive number falls back to that default, and the gateway may
    override it from ``embodiment.envelope.lane_operation_budget_s`` (a stub
    config that never sets it still constructs — the lane keeps its own module
    default).  The guard that consumes it reads the writer's own reported ``rem``
    and REFUSES a descended dispatch below it — the bar is ``operation_budget_s``
    itself; the slew margin is added only to the ``height_hold_s`` the refusal
    names, never to the bar.

    A plain object rather than a ``NamedTuple``/``dataclass``: this crosses from
    the laptop to a robot whose Python is older than the laptop's, and it needs
    no behaviour beyond one derived value.
    """

    __slots__ = ("tabletop_min_z", "height_stand_m", "height_min_m",
                 "hold_tol_m", "timeout_s", "operation_budget_s")

    def __init__(self, tabletop_min_z: float, height_stand_m: float,
                 height_min_m: float, hold_tol_m: float = HOLD_TOL_M,
                 timeout_s: float = DESCENT_CONFIRM_TIMEOUT_S,
                 operation_budget_s: float = LANE_OPERATION_BUDGET_S):
        self.tabletop_min_z = tabletop_min_z
        self.height_stand_m = height_stand_m
        self.height_min_m = height_min_m
        self.hold_tol_m = hold_tol_m
        window = None
        if _finite_num(timeout_s):
            window = float(timeout_s)
        self.timeout_s = (window if window is not None and window > 0.0
                          else float(DESCENT_CONFIRM_TIMEOUT_S))
        budget = None
        if _finite_num(operation_budget_s):
            budget = float(operation_budget_s)
        self.operation_budget_s = (
            budget if budget is not None and budget > 0.0
            else float(LANE_OPERATION_BUDGET_S))

    @property
    def max_descent_m(self) -> float:
        """The writer clamp's own band: ``height_stand_m - height_min_m``."""
        return self.height_stand_m - self.height_min_m

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return ("LaneLaw(tabletop_min_z=%r, height_stand_m=%r, height_min_m=%r,"
                " hold_tol_m=%r, timeout_s=%r, operation_budget_s=%r)"
                % (self.tabletop_min_z, self.height_stand_m,
                   self.height_min_m, self.hold_tol_m, self.timeout_s,
                   self.operation_budget_s))


def usable(law: Any) -> bool:
    """True iff ``law`` describes a real, non-empty descent band.

    A degenerate band (non-finite, zero-width, or inverted) must make the lane
    INERT rather than raise: a misconfigured envelope degrades to today's
    honest refusal, never to a crash and never to a silently widened workspace
    (plan §8).  The caller that routes a target into the lane
    (``GatewayNode._low_lane_claim``) consults this first, so a broken law can
    never route a target into the lane.
    """
    if law is None:
        return False
    try:
        stand = law.height_stand_m
        hmin = law.height_min_m
        floor = law.tabletop_min_z
    except AttributeError:
        return False
    return (_finite_num(stand) and _finite_num(hmin) and _finite_num(floor)
            and stand > hmin)


def height_command(z: float, law: LaneLaw) -> float:
    """The writer ``height`` command for a target at torso-frame ``z``.

    ``clamp(height_stand_m - d(z), height_min_m, height_stand_m)`` — plan §4.
    The command is clamped to the STANDING reference at the top, so a target
    that needs no descent produces the identity command, and the lane can
    never command the body higher than standing.
    """
    d = descent_law.descent_for_z(z, law.max_descent_m)
    return min(max(law.height_stand_m - d, law.height_min_m), law.height_stand_m)


def descent_negligible(z: Any, law: Any) -> bool:
    """True iff the body descent target ``z`` needs is within
    :data:`NO_DESCENT_BAND_M` — i.e. the skill runs AT STANDING.

    FAIL CLOSED to "not negligible" (False) for a non-finite ``z`` or a law
    that is not :func:`usable`: those cases keep every pre-existing refusal.
    """
    if not _finite_num(z) or not usable(law):
        return False
    return descent_law.descent_for_z(z, law.max_descent_m) <= NO_DESCENT_BAND_M


def deep_floor_refusal(z: Any, law: LaneLaw) -> str:
    """The lane-ON refusal for a target below the rescue floor.

    NAMES the floor it refused against, because "below the floor" is only
    actionable if the operator can see where the floor is and read off how much
    band a wider writer clamp would buy.  The floor and the budget are both
    DERIVED (``brain.descent_law.rescue_floor``) — no number here is a literal.

    THE BUDGET IS NAMED FOR WHAT IT IS.  ``law.max_descent_m`` is
    ``height_stand_m - height_min_m`` — the WRITER'S CLAMP BAND, i.e. a COMMAND
    delta in ``height_cmd`` units, NOT the body's measured descent (that is a
    different, simulated quantity: plan §4's correction, 0.907 realized per
    commanded at the clamp).  Saying "the body's available descent" would be the
    exact conflation plan §4 corrects, at the one moment an operator decides
    whether to widen the band — so this clause names the command band and marks
    it SIMULATED.  The honesty boundary is appended like every other lane text.
    """
    if not usable(law) or not _finite_num(z):
        return honest("ERROR:low-workspace pick/place refused: target height is "
                      "not usable (z=%r)" % (z,))
    floor = descent_law.rescue_floor(law.tabletop_min_z, law.max_descent_m)
    return honest(
        "ERROR:low-workspace pick/place refused: z=%.3f is below the rescue "
        "floor %.3f m (tabletop floor %.3f m minus the writer's %.3f m height "
        "clamp band — a COMMAND delta in height_cmd units, SIMULATED "
        "(MuJoCo GR00T-Balance), not a measured body descent) - floor/ground "
        "pick refused, no ground-pick skill exists"
        % (z, floor, law.tabletop_min_z, law.max_descent_m))


# ────────────────────────────────────────────────────── where the lease lives
class DescentContext:
    """The shared body-height state both skills ask (plan §6 D2).

    ONE instance lives on the ``GatewayNode`` — NOT a module singleton (which
    would leak across gateway instances and reconnects) and not per-connection
    (body height is process-global, and the low pick -> low place pair spans
    two separate calls).

    ``height_cmd`` is the last CONFIRMED reference, never the requested command:
    that is what makes the fail-closed precondition real rather than assumed.
    ``effective_z`` is the torso-frame height the engine was actually handed.
    ``hold_deadline`` is a MONOTONIC deadline, so a wall-clock step cannot
    extend a hold.

    ``writer_stand_m`` is the writer's OWN reported standing reference, read
    from a ``status`` poll and used for every frame shift thereafter (see
    :meth:`stand_m`).  The envelope's ``law.height_stand_m`` is a FALLBACK for
    the first command only: the two numbers drift, and a drift biases
    ``z_engine`` by exactly its size.

    A CONFIRMED DESCENT IS CHECKED, NOT DRIVEN.  ``height <m>`` is a LIFETIME
    on the writer: it arms a monotonic deadline and re-stands the body on the
    CLOCK alone whether or not this process is alive, and a ``status`` poll does
    not refresh it.  This context holds NO thread that re-issues the command —
    the writer's own ``height_hold_s`` (a config value) is set long enough to
    cover a whole lane operation, so a renewal thread is unnecessary, and a
    latched verdict that outlived a refusal is exactly the class of defect that
    removal eliminates.  The live gate is instead a FRESH ``status`` poll taken
    immediately before each dispatch (:func:`probe_active`) plus the fail-closed
    lifetime guard (:func:`_lifetime_refusal`).  Every piece of writer traffic —
    the lane's own polls and commands — is serialized through ``writer_lock``,
    which is NEVER held across an engine call.
    """

    __slots__ = ("state", "height_cmd", "effective_z", "hold_deadline",
                 "writer_stand_m", "writer_lock", "cancel_mono",
                 "run_start_mono", "last_retreat_was_noop", "stepped")

    def __init__(self) -> None:
        self.state: str = IDLE
        self.height_cmd: Optional[float] = None
        self.effective_z: Optional[float] = None
        self.hold_deadline: Optional[float] = None
        self.writer_stand_m: Optional[float] = None
        self.writer_lock = threading.Lock()
#: Monotonic stamps of the last operator cancel and of the run start: a cancel
#: counts only when stamped at or after the run began.
        self.cancel_mono: Optional[float] = None
        self.run_start_mono: Optional[float] = None
#: Set when a confirmed retreat found the writer already standing. reset() does
#: not clear it, so the next reply can say who raised the body.
        self.last_retreat_was_noop: bool = False
#: True while a pre-descent stance step may be installed. reset() keeps it so the
#: next retreat can still drop the step.
        self.stepped: bool = False

    def reset(self) -> None:
        """Back to ``IDLE`` with no body height outstanding.

        ``cancel_mono`` is DELIBERATELY NOT cleared here (deviation from the
        R4 L2 spec text, stated): ``reset`` runs inside a run's own retreat
        (e.g. the rise before a re-descend), and clearing there would WIPE a
        cancel that landed during that rise, so the run would go on to descend
        and dispatch the engine the operator had just stopped.  Stickiness is
        prevented by the run stamp instead (:meth:`cancelled_this_run`).
        """
        self.state = IDLE
        self.height_cmd = None
        self.effective_z = None
        self.hold_deadline = None
        self.writer_stand_m = None

    def begin_run(self, now_monotonic: float) -> None:
        """Stamp the start of ONE lane run (the gateway calls this under the
        lane lock, before the sequence runs)."""
        self.run_start_mono = float(now_monotonic)

    def request_cancel(self, now_monotonic: float) -> None:
        """Record an operator cancel at ``now_monotonic``."""
        self.cancel_mono = float(now_monotonic)

    def cancelled_this_run(self) -> bool:
        """True iff a cancel was stamped at or after the current run began.

        A context whose run was never stamped (every direct caller of the
        sequences, every legacy test) answers False, so behaviour without the
        gateway's :meth:`begin_run` is unchanged.
        """
        c, r = self.cancel_mono, self.run_start_mono
        return _finite_num(c) and _finite_num(r) and float(c) >= float(r)

    def in_flight(self) -> bool:
        """True while a body height is outstanding (the refusal predicate).

        A context that is not ``IDLE`` — including one parked in ``RETREATING``
        by an UNCONFIRMED retreat — means the body may not be where the caller
        thinks it is, so the next lane skill must refuse rather than act.
        """
        return self.state != IDLE

    def stand_m(self, fallback: float) -> float:
        """The frame a shift is measured in: the writer's OWN stand, else
        ``fallback``.

        The ``fallback`` is the envelope's ``height_stand_m`` and is honest
        only for the FIRST command, before any poll has reported the writer's
        own reference — every later shift reads the writer's number.
        """
        if _finite_num(self.writer_stand_m):
            return float(self.writer_stand_m)
        return float(fallback)

    def holding(self, now_monotonic: float, hold_s: float,
                writer_rem_s: Optional[float] = None) -> float:
        """Enter ``HOLDING`` for ``min(hold_s, writer_rem_s)`` seconds.

        The cap is the WRITER's own reported remaining hold.  The body re-stands
        when the writer's lifetime expires no matter what this window says, so a
        window reaching past that expiry would promise a hold the body cannot
        keep.  A ``None`` (or non-finite) ``writer_rem_s`` leaves ``hold_s`` as
        the window — the caller is expected to have refused the hold already in
        that case (see :func:`probe_active`).

        RETURNS THE WINDOW IT ACTUALLY HONOURED.  The caller MUST report this
        number, not its own ``hold_s``: the two differ whenever the writer's
        remaining lifetime is the shorter one, and reporting the uncapped
        ``hold_s`` would claim a hold the body will not keep (measured overclaim
        in round 2: claimed 5.0 s, honoured 0.400 s).  ``hold_deadline`` is
        stamped from the same window in the caller's clock, so the number
        returned and the deadline the next-command rule reads are one fact.
        """
        window = max(0.0, float(hold_s))
        if writer_rem_s is not None and _finite_num(writer_rem_s):
            window = min(window, max(0.0, float(writer_rem_s)))
        self.state = HOLDING
        self.hold_deadline = now_monotonic + window
        return window

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return ("DescentContext(state=%s, height_cmd=%r, effective_z=%r, "
                "hold_deadline=%r, writer_stand_m=%r)"
                % (self.state, self.height_cmd, self.effective_z,
                   self.hold_deadline, self.writer_stand_m))


# ──────────────────────────────────────────────────────────────── the log table
class DescentPhaseLog(_place_target.PhaseLog):
    """``place_target.PhaseLog`` with a descent-specific header.

    Overrides ``header()`` ONLY.  Every other behaviour — a row's elapsed time
    MEASURED against the previous row, the total carried on the VERDICT row,
    the wrap width — is inherited, so the repo keeps exactly ONE table
    renderer.  Render with ``place_target.render_table``; publish with
    ``place_target.status_message``.
    """

    def header(self) -> str:
        return "LOW-WORKSPACE  skill=%s  side=%s" % (self.label, self.side or "?")


# ───────────────────────────────────────────────────── the writer-socket verbs
def _issue(writer_verb: Callable[[str], str], line: str,
           lock: Optional[Any] = None) -> Tuple[bool, str]:
    """Send one writer line -> ``(accepted, reply_text)``.

    A raise from the socket layer is a REFUSAL, never a pass: ``_send_unix_verb``
    raises on a timeout or a closed socket, and an unanswered ``height`` command
    is not a descent.  The reply must START with ``DONE`` (the writer's height
    verb answers ``DONE`` on success and ``ERROR:<why>`` otherwise —
    ``skills/writer/main_merged_writer.py:475-479``).

    ``lock`` is the context's writer lock.  It is held for THIS call and no
    longer: the call is the traffic, so serializing around it is the point, and
    holding it any wider would leave the writer socket held across the lane's
    own other polls.  Never pass it with an engine call outstanding — the engine
    (``do_pick`` / ``do_place``) is a separate device and must not queue behind
    writer traffic.
    """
    try:
        if lock is None:
            reply = writer_verb(line)
        else:
            with lock:
                reply = writer_verb(line)
    except Exception as exc:  # noqa: BLE001 - a dead socket is a refusal
        return False, "%s: %s" % (type(exc).__name__, exc)
    text = str(reply or "").strip()
    return text.startswith("DONE"), text


def _safe_status(status_verb: Callable[[], str],
                 lock: Optional[Any] = None) -> Optional[str]:
    """One status poll -> the line, or ``None`` when the call itself failed.

    A failed poll is NOT a parse failure: it is retried inside the confirm
    window, and only a window that expires with nothing usable refuses.
    ``lock`` is the context's writer lock, held for this call only.
    """
    try:
        if lock is None:
            reply = status_verb()
        else:
            with lock:
                reply = status_verb()
    except Exception:  # noqa: BLE001 - treated exactly like an unparseable poll
        return None
    return reply if isinstance(reply, str) else None


def _clip(text: Any, limit: int = 120) -> str:
    """A writer status line, collapsed onto one line for an operator detail."""
    body = " ".join(str(text or "").split())
    return body if len(body) <= limit else body[:limit] + "..."


def _writer_stand(status_verb: Callable[[], str],
                  lock: Optional[Any] = None) -> Optional[float]:
    """The writer's OWN standing reference from a single ``status`` poll.

    The writer reports ``stand`` ONLY in the INACTIVE form — the form it emits
    while it is tracking nothing.  During an ACTIVE descent no poll carries it,
    so it must be read BEFORE the descent is commanded and carried from there
    (``DescentContext.writer_stand_m``).

    ``None`` — an unparseable poll, or a writer already tracking a height —
    means the caller falls back to the envelope's ``height_stand_m``, which is
    honest for the first command only.
    """
    parsed = descent_law.parse_height_status(_safe_status(status_verb, lock))
    if parsed is None or parsed.get("active") or parsed.get("suspended"):
        return None
    stand = parsed.get("stand")
    return float(stand) if _finite_num(stand) else None


def probe_active(status_verb: Callable[[], str], target_m: float,
                 *, tol_m: float = HEIGHT_CONFIRM_TOL_M,
                 lock: Optional[Any] = None
                 ) -> Tuple[bool, Optional[float], str]:
    """One SYNCHRONOUS ``status`` poll: is the writer still driving
    ``target_m``?  ``(ok, rem_s, detail)``.

    The lane's LAST LINE before an engine dispatch.  A descent is confirmed by
    a poll, but ``height <m>`` is a LIFETIME on the writer: it can expire, be
    suspended by an RC takeover, or be re-issued elsewhere between the
    confirmation and the moment the arm is actually commanded.  Dispatching
    against a body nobody is holding is the collision this lane exists to
    prevent, so this refuses rather than retries.

    ``rem_s`` is the writer's own remaining hold FROM THE SAME POLL — one poll,
    one frame, one clock — and is what bounds the hold window
    (:meth:`DescentContext.holding`).  It is ``None`` when the writer reports no
    parseable ``rem``; the caller may still act on ``ok``, but must not extend a
    hold past what the poll showed.
    """
    raw = _safe_status(status_verb, lock)
    parsed = descent_law.parse_height_status(raw)
    if parsed is None:
        return False, None, honest(
            "the writer's status is not a height report (got %s) - the body "
            "height cannot be vouched for" % (_clip(raw) or "no reply"))
    ref = parsed.get("ref")
    if parsed.get("suspended"):
        return False, None, honest(
            "an RC override SUSPENDED the squat at ref=%.3f - the lane no "
            "longer owns the body height" % ref)
    if not parsed.get("active") or parsed.get("target") is None:
        return False, None, honest(
            "the writer is no longer driving a height (status: %s)"
            % _clip(raw))
    if ref is None or abs(ref - target_m) > tol_m:
        return False, None, honest(
            "the writer's reference %.3f has left the held height %.3f "
            "(tol %.3f)" % (ref, target_m, tol_m))
    rem = parsed.get("rem")
    return (True, (float(rem) if _finite_num(rem) else None),
            "writer still active at ref=%.3f (rem=%s)" % (ref, _clip(raw)))


def _lane_budget(law: Any) -> float:
    """The DECLARED lane operation budget for ``law``, in seconds.

    ONE read of ONE name.  ``LaneLaw.operation_budget_s`` is the threaded field
    (config key ``embodiment.envelope.lane_operation_budget_s`` — see
    :data:`LANE_OPERATION_BUDGET_S` for the whole threading), and its
    constructor already falls back to that module constant for a value that is
    not a finite positive number.  This repeats the fallback only so a law
    object that never ran that constructor — a stub, a hand-built duck type —
    still yields the DECLARED budget instead of an ``AttributeError`` or a zero
    bar.  It is the number the pre-dispatch guard (:func:`_lifetime_refusal`,
    which adds the slew margin) and the post-dispatch check
    (:func:`_post_dispatch_body_check`) both grade against; nothing here
    invents a second budget.
    """
    budget = getattr(law, "operation_budget_s", LANE_OPERATION_BUDGET_S)
    if _finite_num(budget) and float(budget) > 0.0:
        return float(budget)
    return float(LANE_OPERATION_BUDGET_S)


def _continued_lockout_s(ctx: Any, now_monotonic: Any) -> float:
    """The RESIDUAL of the hold a next-command rule answered ``HOLD`` for.

    A ``NEXT_HOLD`` answer means the lane CONTINUES a hold it already entered,
    so the window the operator is subject to is what is LEFT of it, not the
    window it was entered with.  Read from ``ctx.hold_deadline`` — the same
    monotonic deadline the rule itself compared against — so the number
    reported and the number the rule acted on are one fact.  ``0.0`` for a
    context with no deadline (nothing to continue) or a non-finite clock; the
    rule cannot answer ``NEXT_HOLD`` past the deadline, so in the reachable
    case this is a positive window.
    """
    deadline = getattr(ctx, "hold_deadline", None)
    if not _finite_num(deadline) or not _finite_num(now_monotonic):
        return 0.0
    return max(0.0, float(deadline) - float(now_monotonic))


def _lifetime_refusal(rem_s: Any, budget_s: Any, verb: str,
                      slew_margin_s: Any = 0.0) -> Optional[str]:
    """The fail-closed budget guard: ``None`` to proceed, else the refusal.

    ``height <m>`` is a LIFETIME on the writer, and the lane does NOT renew it,
    so a dispatched action needs the writer's own reported remaining hold to
    cover the whole thing.  This is the guard that REPLACES a renewal thread:
    instead of re-issuing a short clock in the background, it REFUSES to
    operate on one and names the config key that fixes it.

    THE BAR IS ``budget``, AND THE REFUSAL ALSO QUOTES THE SLEW MARGIN.
    ``budget`` is :data:`LANE_OPERATION_BUDGET_S` (or
    ``LaneLaw.operation_budget_s``, which config threads into it) — the
    ENGINE-CALL budget, the same number :func:`_post_dispatch_body_check`
    grades the MEASURED engine elapsed against, so ``rem`` must cover it whole.

    ``slew_margin_s`` (:func:`slew_margin_s`) is NOT added to that bar, and
    adding it would be a double count: the descent's own slew+confirm is
    dispatched BEFORE this reads ``rem``, so the writer's own clock has already
    deducted it (this function reads ``rem`` after ``_descend_step`` returns).
    That is exactly why a ``height_hold_s`` equal to the bare budget does not
    dispatch (D9) — and the remedy for D9 is the VALUE TO SET, never a higher
    bar: a gate at ``budget + margin`` would refuse a body the writer is
    provably still holding, and would demand ``budget + 2 * margin`` of
    ``height_hold_s``, which lengthens the operator lockout 1:1 for no safety
    gain.  So the refusal quotes the observed ``rem``, the required bar, the
    budget, the margin, and the value to SET — the bar plus the margin, since
    that margin is spent before this gate reads the clock — so the operator
    does not have to rediscover it by bisection.

    REFUSES on every value that is not a usable number at or above the bar:
    an absent or unparseable ``rem`` (``None`` — no parseable fragment, a
    suspended hold, a dead socket), a non-finite value, and a NEGATIVE one.
    The same discipline applies to the budget: a missing, non-finite or
    non-positive ``budget_s`` falls back to :data:`LANE_OPERATION_BUDGET_S`, so
    a bad override can never silently lift the bar to zero, and a bad
    ``slew_margin_s`` degrades to 0.0 rather than to a guess.  Nothing is
    guessed and nothing defaults to ok — an unknown lifetime is not a long
    lifetime, and the permissive direction here is a body that re-stands
    mid-operation.
    """
    if _finite_num(budget_s) and float(budget_s) > 0.0:
        budget = float(budget_s)
    else:
        budget = float(LANE_OPERATION_BUDGET_S)
    margin = (float(slew_margin_s)
              if _finite_num(slew_margin_s) and float(slew_margin_s) > 0.0
              else 0.0)
    # The bar is the budget alone: `rem` is read after the descent's own
    # slew+confirm, so the writer's clock has already deducted the margin.
    required = budget
    if _finite_num(rem_s) and float(rem_s) >= required:
        return None
    observed = ("%.1fs" % float(rem_s) if _finite_num(rem_s)
                else "absent/unreadable (%r)" % (rem_s,))
    return honest(
        "refusing to dispatch %s: the writer's height lifetime is too short "
        "for a lane operation (rem=%s, required >= %.1fs = LANE_OPERATION_"
        "BUDGET_S %.1fs; slew_margin %.1fs of the SAME clock is already spent "
        "by the descent's own slew+confirm before this reads `rem`) — so set "
        "`height_hold_s` in "
        "skills/writer/config/merge_config.yaml to at least %.1fs so ONE "
        "`height` command covers the whole descent->pick->place"
        % (verb, observed, required, budget, margin, required + margin))


def hold_unusable_reason(body_rem: Any, law: Any) -> Optional[str]:
    """``None`` when a hold entered now could be reused by a matching place,
    else the honest reason it is NOT offered (R4 HRC-1).

    A matching place re-runs the lifetime guard at its own dispatch
    (:func:`_lifetime_refusal` with the bar :func:`_lane_budget`), so a hold
    entered with the writer's remaining lifetime already BELOW that bar can
    never be reused: the place would refuse it.  MEASURED on the box (brief):
    ``height_hold_s`` 66.0 vs ``lane_operation_budget_s`` 60.0, holds offered at
    45.6 s / 21.1 s rem, every one refused at the place.  So the hold is not
    offered and the pick retreats instead.

    DECIDABILITY.  Only the LIFETIME half can be decided here: the place's own
    drop point (and whether it classifies DESCEND) is built only at place time
    (``brain/agent/tools.py``) and is not an input of :func:`pick_sequence`.
    The text of the hold row says so instead of guessing.
    """
    budget = _lane_budget(law)
    if _finite_num(body_rem) and float(body_rem) >= budget:
        return None
    observed = (("%.1fs" % float(body_rem)) if _finite_num(body_rem)
                else "absent/unreadable")
    return honest("hold NOT offered: the writer reports %s of height lifetime "
                  "left and a matching place needs rem >= %.1fs "
                  "(lane_operation_budget_s) at its own dispatch, so its "
                  "lifetime guard would refuse it" % (observed, budget))


def _rise_phrase(ctx: Any, default: str) -> str:
    """``default`` ("rose to" / "stood back up to"), or the honest phrase when
    the last CONFIRMED retreat found the body ALREADY standing (R4 HRC-6/7)."""
    if getattr(ctx, "last_retreat_was_noop", False) is True:
        return "was already standing at"
    return default


def _post_dispatch_body_check(status_verb: Callable[[], str],
                              held_ref: Any, *, elapsed_s: Any, budget_s: Any,
                              lock: Optional[Any] = None
                              ) -> Tuple[bool, Optional[float],
                                         List[Tuple[str, str]], str]:
    """``(ok, rem_s, rows, summary)`` — the POST-DISPATCH body check (D4).

    A guard in FRONT of a dispatch only says the body was low when the arm was
    COMMANDED.  The seconds that follow are when the writer's own lifetime can
    expire, an RC takeover can suspend the squat, or the engine call can outrun
    the budget this lane declared — and until this round the lane's sole
    observable of that was conditional on the grasp SUCCEEDING, so a
    mid-operation re-stand was byte-indistinguishable from a normal run.  This
    is that observable, and BOTH sequences take it, on every path: the
    grasp-succeeded and the grasp-FAILED alike.

    ONE FRESH SYNCHRONOUS POLL, not a latched flag.  ``probe_active`` is the
    same gate the pre-dispatch side uses, so "still holding" means the same
    thing on both ends of the operation — the writer is ACTIVE at the reference
    this lane confirmed, within :data:`HEIGHT_CONFIRM_TOL_M` — and the ``rem``
    it returns is the writer's OWN remaining hold from that same poll.

    TWO ANOMALY CLASSES, ONE HANDLING.  A writer that is no longer holding the
    reference (re-stood on its own clock, suspended by an RC takeover, or
    unreachable) raises :data:`BODY_RESTOOD`; an engine call whose MEASURED
    elapsed outran the declared budget raises :data:`BUDGET_OVERRUN`.  Both are
    the same claim — an operation this lane cannot vouch for — and both are
    returned as a LOUD ``(phase, detail)`` row so the operator reads which one
    fired, what was measured, and the writer's own words.  The phase names are
    distinct from the generic ``hold``/``grasp``/``verify`` rows on purpose: a
    re-stand must not have to be inferred from a row that also means "the hold
    was declined".

    FAIL CLOSED on the values as well as the state.  An absent or non-finite
    ``held_ref`` cannot be compared against anything, so it raises the same
    class; an absent or non-finite ``elapsed_s`` cannot be shown to be WITHIN
    the budget, so it raises the overrun rather than passing by default.  A
    missing, non-finite or non-positive ``budget_s`` falls back to
    :data:`LANE_OPERATION_BUDGET_S` (see :func:`_lane_budget`), never to a bar
    of zero.  ``summary`` is the SAME anomalies as ONE suffix-free clause for a
    reply or a verdict — the rows carry the writer's prose, which already ends
    with the honesty boundary, and splicing a boundary-terminated string into
    the middle of a longer one is how a text ends up claiming a boundary it no
    longer ends on.
    """
    held = float(held_ref) if _finite_num(held_ref) else None
    if held is None:
        held_ok, held_rem, held_detail = False, None, honest(
            "the run never established a held reference to check against")
    else:
        held_ok, held_rem, held_detail = probe_active(status_verb, held,
                                                      lock=lock)
    budget = (float(budget_s)
              if _finite_num(budget_s) and float(budget_s) > 0.0
              else float(LANE_OPERATION_BUDGET_S))
    measured = ("%.1fs" % float(elapsed_s) if _finite_num(elapsed_s)
                else "absent/unreadable (%r)" % (elapsed_s,))
    rows: List[Tuple[str, str]] = []
    notes: List[str] = []
    if not held_ok:
        notes.append("%s - the writer was not still holding %s when the engine "
                     "returned (the engine call measured %s)"
                     % (BODY_RESTOOD,
                        "%.3f m" % held if held is not None else "a height",
                        measured))
        rows.append(("body_restood", honest("%s; %s"
                                           % (notes[-1], held_detail))))
    # The fresh poll above, not the clock, is the bound: a writer still holding
    # the reference held it for the whole call, so an overrun fails only when the
    # poll cannot vouch for the body or the elapsed time is unreadable.
    _over = not (_finite_num(elapsed_s) and float(elapsed_s) <= budget)
    if _over and held_ok and _finite_num(elapsed_s):
        print("[lane] B8j-hf14: %s - the engine call measured %s against %.1fs, "
              "but the writer still held %.3f m when it returned (rem=%s): "
              "not an anomaly, the engine verdict stands"
              % (BUDGET_OVERRUN, measured, budget, held, held_rem),
              flush=True)
        _over = False
    if _over:
        notes.append("%s - the engine call measured %s against the declared "
                     "lane operation budget %.1fs"
                     % (BUDGET_OVERRUN, measured, budget))
        rows.append(("budget", honest(
            "%s - the engine call measured %s against the declared lane "
            "operation budget %.1fs (embodiment.envelope.lane_operation_budget_s)"
            % (BUDGET_OVERRUN, measured, budget))))
    return (not rows), held_rem, rows, "; ".join(notes)


def _downgrade_verdict(summary: str, engine_verdict: Optional[str]) -> str:
    """The ANOMALY verdict: the lane's own ``VERDICT:error``, never a success.

    A mid-operation re-stand must never be REPORTED as a success, and the
    folded ``_do_pick``/``_do_place`` verdict is the engine's claim about the
    arm alone — it cannot see the body.  So the lane OWNS the downgrade: the
    string returned here does not start with ``VERDICT:success``, which is what
    every consumer in this repo reads (``place_target.verdict_row`` -> the
    VERDICT row; ``GatewayNode._shape_reply`` -> ``ok``), so no caller can
    render this run as a pass.

    THE ENGINE'S VERDICT IS EVIDENCE, NOT SOMETHING TO ERASE.  It is carried
    inside the evidence clause, so the operator still sees exactly what the
    verifier said about the grasp; only the LABEL is the lane's.

    THE PREFIX IS THE VERIFIER'S OWN GRAMMAR.  ``VERDICT:error|Evidence:`` is
    what ``brain/verifier.py`` emits and what ``GatewayNode._shape_reply``
    slices a ``reason`` out of, so a downgraded verdict surfaces there as
    ``ok: false`` with the anomaly as the reason rather than as an unreadable
    string.  The honesty boundary is appended by :func:`_honest_verdict` on the
    way out, like every other verdict this module returns.
    """
    evidence = str(summary or "").strip()
    if engine_verdict:
        evidence = ("%s (engine verdict was: %s)" % (evidence, engine_verdict)
                    if evidence else str(engine_verdict))
    return "VERDICT:error|Evidence:(low-workspace) %s" % evidence


def _judge_descent(parsed: Optional[dict], target: float, stand: float,
                   min_descent: float, tol: float) -> Tuple[bool, str]:
    """``(confirmed, detail)`` for one parsed height fragment.

    Confirmed requires ALL FOUR: the writer is ACTIVELY tracking; its reference
    has reached the height THIS LANE COMMANDED (``target``); the reference has
    left standing by more than ``min_descent``; and the comparison is made in
    the writer's own reported frame (``stand``).

    ``target`` is the COMMANDED height, not the writer's own intermediate
    ``target``: the writer's internal target can agree with its reference on a
    move that was never requested (a clamp, a concurrent re-issue), and a
    confirmation that reads it back would confirm the writer's arithmetic
    instead of the lane's command.

    The magnitude conjunct is not a formality.  ``STAND_EPS_M`` alone admits a
    band only ~1 cm wide: the reference moves just past epsilon, the confirm
    band still calls it a match, and a "descent" that never lowered the body
    enough to matter reads as CONFIRMED.
    """
    if parsed is None:
        return False, "no parseable height fragment in the writer status"
    ref = parsed.get("ref")
    if parsed.get("suspended"):
        return False, ("descent SUSPENDED by an RC override at ref=%.3f"
                       % ref)
    if not parsed.get("active") or parsed.get("target") is None:
        if parsed.get("stand") is not None:
            return False, ("descent was a no-op (writer height band not "
                           "widened): height=%.2f(stand %.2f)"
                           % (ref, parsed["stand"]))
        return False, "writer status carries no tracked height target"
    if abs(ref - target) > tol:
        return False, ("ref %.3f has not reached the commanded %.3f (tol %.2f)"
                       % (ref, target, tol))
    if abs(ref - stand) <= min_descent:
        return False, ("ref %.3f sits %.3f m from standing (%.3f) - a descent "
                       "must clear %.3f m"
                       % (ref, abs(ref - stand), stand, min_descent))
    return True, "ref=%.3f commanded=%.3f (<= %.3f)" % (ref, target, tol)


def _judge_retreat(parsed: Optional[dict], tol: float) -> Tuple[bool, str]:
    """``(confirmed, detail)`` for one parsed height fragment, retreating.

    Confirmed requires the INACTIVE form (the writer is tracking nothing) with
    the reference back at the standing reference it itself reports.  Reading
    the standing value off the fragment rather than re-supplying it keeps the
    two ends of the comparison from the same emitter.
    """
    if parsed is None:
        return False, "no parseable height fragment in the writer status"
    ref = parsed.get("ref")
    if parsed.get("suspended"):
        return False, ("writer holds a SUSPENDED squat at ref=%.3f" % ref)
    if parsed.get("active"):
        return False, ("still tracking a height: ref=%.3f target=%s"
                       % (ref, parsed.get("target")))
    stand = parsed.get("stand")
    if stand is None:
        return False, "writer status carries no standing reference"
    if abs(ref - stand) > tol:
        return False, ("ref %.3f has not returned to stand %.3f (tol %.2f)"
                       % (ref, stand, tol))
    return True, "ref=%.3f stand=%.3f" % (ref, stand)


def _poll(status_verb: Callable[[], str], judge: Callable[[Optional[dict], float], Tuple[bool, str]],
          tol: float, *, timeout_s: float, interval_s: float,
          sleep: Callable[[float], None],
          monotonic: Callable[[], float],
          lock: Optional[Any] = None) -> Tuple[bool, Optional[float], str]:
    """Poll ``status`` until ``judge`` confirms or the window expires.

    The FIRST poll happens before any sleep, so a writer that is already in the
    commanded state confirms with no latency at all.  The iteration count is
    capped as well as the clock, so a caller-supplied clock that never advances
    cannot spin forever.  ``lock`` serializes each poll against the lane's own
    other writer traffic; it is taken per poll, never across the loop.
    """
    interval = float(interval_s)
    deadline = monotonic() + max(0.0, float(timeout_s))
    max_polls = 2
    if interval > 0.0:
        max_polls = int(math.ceil(max(0.0, float(timeout_s)) / interval)) + 2
    last_detail = "the writer was never polled"
    for _ in range(max(max_polls, 1)):
        parsed = descent_law.parse_height_status(_safe_status(status_verb, lock))
        confirmed, detail = judge(parsed, tol)
        last_detail = detail
        if confirmed:
            return True, parsed.get("ref"), detail
        if monotonic() >= deadline:
            break
        sleep(interval)
    return False, None, last_detail


def poll_descent(status_verb: Callable[[], str], target_m: float, stand_m: float,
                 *, timeout_s: float = DESCENT_CONFIRM_TIMEOUT_S,
                 tol_m: float = HEIGHT_CONFIRM_TOL_M,
                 min_descent_m: float = MIN_CONFIRMED_DESCENT_M,
                 interval_s: float = DESCENT_POLL_INTERVAL_S,
                 sleep: Optional[Callable[[float], None]] = None,
                 monotonic: Optional[Callable[[], float]] = None,
                 lock: Optional[Any] = None
                 ) -> Tuple[bool, Optional[float], str]:
    """``(confirmed, ref, detail)`` — poll until the descent is CONFIRMED.

    ``target_m`` is the COMMANDED height and ``stand_m`` the writer's own
    nominal standing reference (``DescentContext.writer_stand_m``, read from a
    poll taken before the command was issued); both are needed because the
    writer's ACTIVE fragment reports the current reference and its own target
    but not the nominal it started from.  ``timeout_s`` defaults to the module
    constant so a caller that never populates ``LaneLaw.timeout_s`` behaves
    exactly as before.
    """
    return _poll(status_verb,
                 lambda parsed, tol: _judge_descent(parsed, target_m, stand_m,
                                                    float(min_descent_m), tol),
                 float(tol_m), timeout_s=timeout_s, interval_s=interval_s,
                 sleep=sleep or time.sleep, monotonic=monotonic or time.monotonic,
                 lock=lock)


def poll_retreat(status_verb: Callable[[], str],
                 *, timeout_s: float = DESCENT_CONFIRM_TIMEOUT_S,
                 tol_m: float = HEIGHT_CONFIRM_TOL_M,
                 interval_s: float = DESCENT_POLL_INTERVAL_S,
                 sleep: Optional[Callable[[float], None]] = None,
                 monotonic: Optional[Callable[[], float]] = None,
                 lock: Optional[Any] = None
                 ) -> Tuple[bool, Optional[float], str]:
    """``(confirmed, ref, detail)`` — poll until standing is CONFIRMED."""
    return _poll(status_verb, lambda parsed, tol: _judge_retreat(parsed, tol),
                 float(tol_m), timeout_s=timeout_s, interval_s=interval_s,
                 sleep=sleep or time.sleep, monotonic=monotonic or time.monotonic,
                 lock=lock)


# ─────────────────────────────────────────── the operator's next-command rule
def next_command_action(ctx: DescentContext, verb: str, z: Any,
                        law: LaneLaw, now_monotonic: float,
                        tol_m: Optional[float] = None,
                        args: Any = None) -> Tuple[str, str]:
    """``(action, reason)`` for a command arriving while a height may be held.

    The operator's rule (plan §6, D2), applied when the NEXT command arrives —
    the previous skill cannot see the future, so this is the only place the
    hold can be decided:

      * a lane skill needing the SAME height, inside the hold window -> HOLD
      * a RELEASE-ONLY place (``place []`` — no coordinates at all) arriving
        inside a live hold -> HOLD, and only then: the body is already at a
        confirmed height and the drop aims at no target height, so there is
        nothing an "unknown target" could mis-frame, and rising to drop from
        standing is strictly worse than dropping from the height the body is
        already confirmed at
      * a lane skill needing a DIFFERENT height, or past the window, or a lane
        skill whose target is high/unknown -> rise first
      * a ``navigate`` / ``cmd_set`` -> rise AND CONFIRM first (a walk while a
        descent is held is squat-walking, which is unmeasured)
      * an arm-only command -> rise best-effort, warn loudly on failure
      * anything non-motion (observe / status / image / query_user / ...) ->
        untouched, because a read cannot collide with a lowered body

    THERE IS NO ``NEXT_CLEAR`` AND NO TAKEOVER VERB SET (D3).  The lane cannot
    observe an operator takeover: nothing dispatches an ``estop``/``rc``/
    ``rc_takeover``/``suspend`` verb through this gateway, and the writer's own
    suspend is UNREACHABLE while the arm engine holds the body (D2), so a
    "forget the hold" branch could never fire honestly.  What the lane does
    instead is report the lockout it holds (:func:`lockout_notice`).

    ``z`` is the target's TORSO-FRAME height (the raw detection), NOT a body
    height: the height this target NEEDS is derived here via
    :func:`height_command`, so the caller never has to know the ladder.

    ``args`` is the lane skill's own argument list, when the caller has it.
    ``None`` — the default — means NO INFORMATION, so every pre-existing caller
    keeps this rule's original answer byte for byte; it is read ONLY by the
    release-only clause below, through :func:`release_only_args`.

    ``now_monotonic`` MUST be the same clock ``DescentContext.hold_deadline``
    was written from (``time.monotonic``) — a wall-clock value compared against
    a monotonic deadline would silently extend or truncate the hold across a
    clock step.

    A context that is ``IDLE`` — and a law that is not :func:`usable` — both
    answer :data:`NEXT_NONE`: with no descent outstanding the lane has nothing
    to say about the command, and that keeps the lane-off path inert.
    """
    name = str(verb or "")
    if ctx is None or ctx.state == IDLE or not usable(law):
        return NEXT_NONE, "no descent in flight"
    if name in LANE_SKILL_VERBS:
        if not _finite_num(z):
            # Release-only place inside a live hold: it carries no coordinates, so
            # no target height can be mis-placed, and rising first is strictly
            # worse. Narrowed to a place-family verb, empty args, a live hold.
            held = ctx.height_cmd
            if (name in PLACE_FAMILY_VERBS and release_only_args(args)
                    and ctx.state == HOLDING and held is not None
                    and ctx.hold_deadline is not None
                    and now_monotonic < ctx.hold_deadline):
                return NEXT_HOLD, (
                    "release-only place: no target height to be framed by, and "
                    "the body is already held at %.3f m inside the live window "
                    "(%.1fs left) — drop from the held height"
                    % (held, ctx.hold_deadline - now_monotonic))
            return NEXT_RETREAT_BEST_EFFORT, ("target height absent — no body "
                                              "height can be held for an unknown target")
        if z >= descent_law.COMFORT_Z:
            return NEXT_RETREAT_BEST_EFFORT, ("this target needs no descent — "
                                              "rise before acting on it")
        need = height_command(z, law)
        tol = law.hold_tol_m if tol_m is None else float(tol_m)
        held = ctx.height_cmd
        if (ctx.state == HOLDING and held is not None
                and ctx.hold_deadline is not None
                and now_monotonic < ctx.hold_deadline
                and descent_law.same_height(need, held, tol)):
            return NEXT_HOLD, ("held %.3f m is within %.3f m of the %.3f m this "
                               "target needs" % (held, tol, need))
        return NEXT_REDESCEND, ("needed %.3f m differs from the held %s m (or "
                                "the hold window lapsed) — rise, then descend"
                                % (need, "%.3f" % held if held is not None
                                   else "none"))
    if name in BODY_BLOCKING_VERBS:
        return NEXT_RETREAT_BLOCKING, ("squat-walking is unmeasured — standing "
                                       "must be CONFIRMED before a walk")
    if name in BEST_EFFORT_VERBS:
        return NEXT_RETREAT_BEST_EFFORT, ("arm-only command — rise best-effort, "
                                          "warn loudly on failure")
    return NEXT_NONE, "non-motion command — the lane leaves it alone"


# ───────────────────────────────────────────────────────── the two sequences
def _refuse(log: DescentPhaseLog, detail: str,
            stamp: Callable[[], float]) -> Tuple[str, Any]:
    """Close a run on a fail-closed refusal (engine NEVER called)."""
    log.finish("refuse", detail, stamp())
    return ("ERROR:low-workspace %s" % detail), None


def _descend_step(ctx: DescentContext, log: DescentPhaseLog, law: LaneLaw,
                  z: float, *, writer_verb: Callable[[str], str],
                  status_verb: Callable[[], str],
                  stamp: Callable[[], float],
                  clock: Callable[[], float],
                  sleep: Callable[[float], None]
                  ) -> Tuple[bool, Optional[float], str]:
    """``height <cmd>`` then confirm it.  ``(ok, confirmed_ref, detail)``.

    Writes TWO rows on the happy path — ``descend`` (the writer accepted the
    command) and ``confirm`` (the reference actually left standing) — because
    those are two independently falsifiable facts and collapsing them would
    hide which one failed.

    THE FRAME IS PROBED FIRST.  The writer reports its own standing reference
    only in the INACTIVE status form, and only while it is tracking nothing;
    during an ACTIVE descent no poll carries it.  So one ``status`` poll is
    taken BEFORE the command and its ``stand`` is carried on the context — the
    frame shift, the confirmation and the reply then all read that ONE number
    instead of the envelope's (which drifts).  The envelope value is the
    fallback only when that probe yields nothing, i.e. for a first command.

    A FAILED DESCENT PARKS THE CONTEXT (D7).  ``ctx.state`` is set to
    ``DESCENDING`` BEFORE the writer is touched, so the writer-rejected and the
    not-confirmed return paths both leave a context that is NOT ``IDLE`` — the
    mirror of ``_retreat_step``, which parks ``RETREATING`` before ``height
    off``.  A failure here means the body may be moving (a lost reply is not a
    proof the command never landed), and a context that is not ``IDLE`` makes
    every subsequent consumer refuse rather than act: blocking verbs answer
    ``NEXT_RETREAT_BLOCKING`` and a lane skill must take a CONFIRMED rise
    first.  Leaving ``IDLE`` would be a context that has FORGOTTEN an unknown
    body height, which is the one thing this lane must never do.
    """
    target = height_command(z, law)
    stand = _writer_stand(status_verb, ctx.writer_lock)
    if stand is None:
        stand = float(law.height_stand_m)
    ctx.writer_stand_m = stand
    # Park the state before touching the writer: `height <m>` may have been
    # accepted with the reply lost, and a failure here means the body may be
    # moving. A non-IDLE context makes every later consumer refuse, not act.
    ctx.state = DESCENDING
    ok, reply = _issue(writer_verb, "height %.3f" % target, ctx.writer_lock)
    if not ok:
        detail = honest("descent refused: writer rejected `height %.3f` (%s)"
                        % (target, reply or "no reply"))
        log.add("descend", "refuse", detail, stamp())
        return False, None, detail
    # The writer accepted the command. Confirmation below is a separate fact, and
    # collapsing the two would hide which one failed.
    log.add("descend", "ok", honest("height %.3f commanded (writer: %s)"
                                    % (target, reply)), stamp())
    confirmed, ref, cd = poll_descent(status_verb, target, stand,
                                      timeout_s=law.timeout_s,
                                      sleep=sleep, monotonic=clock,
                                      lock=ctx.writer_lock)
    if not confirmed:
        detail = honest("descent NOT confirmed: %s" % cd)
        log.add("confirm", "refuse", detail, stamp())
        return False, None, detail
    ctx.state = DESCENDING          # already parked; restated with ref
    ctx.height_cmd = ref            # the confirmed ref, not the command
    # The body stays down on the writer's own `height_hold_s` clock; nothing in
    # this lane renews it.
    detail = honest("height %.3f -> %.3f confirmed %s"
                    % (stand, ref, cd))
    log.add("confirm", "ok", detail, stamp())
    return True, ref, detail


def _retreat_step(ctx: DescentContext, log: DescentPhaseLog, *,
                  writer_verb: Callable[[str], str],
                  status_verb: Callable[[], str],
                  stamp: Callable[[], float],
                  clock: Callable[[], float],
                  sleep: Callable[[float], None]
                  ) -> Tuple[bool, Optional[float], str]:
    """``height off`` then confirm standing.  ``(ok, ref, detail)``.

    WARN-AND-HOLD on failure (plan §7): the loud stderr line, NO further motion
    dispatched, and the context stays NON-``IDLE`` so the next lane skill
    refuses rather than act on a body height nobody can vouch for.  The
    writer's ``height_hold_s`` watchdog and its shutdown ascend are the
    independent backstops.

    NOTHING ELSE CAN RE-ISSUE THE DESCENT.  There is no renewal thread to stop
    first (the body is held by the writer's own ``height_hold_s`` lifetime and
    by nothing else), so the ``height off`` below is the only command racing it
    and it is sent under :attr:`DescentContext.writer_lock` like every other.
    """
    ctx.state = RETREATING
    ctx.last_retreat_was_noop = False
    # One pre-poll: the writer's watchdog may already have re-stood the body, so
    # reading before `height off` lets the detail say who raised it. The command
    # is still sent: it is idempotent and clears any latch the poll cannot see.
    pre = descent_law.parse_height_status(_safe_status(status_verb,
                                                       ctx.writer_lock))
    was_standing, _pre_detail = _judge_retreat(pre, HEIGHT_CONFIRM_TOL_M)
    ok, reply = _issue(writer_verb, "height off", ctx.writer_lock)
    if not ok:
        detail = honest("retreat refused: writer rejected `height off` (%s)"
                        % (reply or "no reply"))
        log.add("retreat", "refuse", detail, stamp())
        _loud(detail)
        return False, None, detail
    log.add("retreat", "ok", honest("height off accepted"), stamp())
    confirmed, ref, cd = poll_retreat(status_verb, sleep=sleep, monotonic=clock,
                                      lock=ctx.writer_lock)
    if not confirmed:
        detail = honest("%s - body height may still be low; HOLD (%s)"
                        % (RETREAT_UNCONFIRMED, cd))
        _loud(detail)
        log.add("confirm", "refuse", detail, stamp())
        return False, None, detail
    if was_standing:
        detail = honest("%s: the writer already reported the body standing "
                        "BEFORE this lane sent `height off` (it re-stood on "
                        "its own height_hold_s clock or an earlier rise); this "
                        "lane did NOT raise it: %s" % (ALREADY_STANDING, cd))
    else:
        detail = honest("rise confirmed: %s" % cd)
    log.add("confirm", "ok", detail, stamp())
    if ctx.stepped:
        # The stance step is dropped after the rise, never before: relaxing the
        # stance snap at depth is the riskier order. A refused `step off` never
        # turns a confirmed rise into RETREATING; the next retreat retries it.
        s_ok, s_reply = _issue(writer_verb, "step off", ctx.writer_lock)
        if s_ok:
            ctx.stepped = False
            log.add("step_off", "ok", honest("step off accepted after the "
                                             "confirmed rise (writer: %s)"
                                             % s_reply), stamp())
        else:
            s_detail = honest("step off REFUSED after the confirmed rise (%s) "
                              "- the stance step may still be installed; the "
                              "next retreat or pre-step retries `step off`"
                              % (s_reply or "no reply"))
            log.add("step_off", "refuse", s_detail, stamp())
            _loud(s_detail)
    ctx.reset()
    ctx.last_retreat_was_noop = bool(was_standing)
    return True, ref, detail


# ────────────────────────────── optional pre-descent stance step
_STEP_PHASE_RE = re.compile(
    r"step=(IDLE|SHIFT|HOLD|LIFT|PLACE|RECENTRE|DONE|ABORT)(?![A-Za-z])")
_STEP_NUM = r"([+-]?(?:\d+(?:\.\d*)?|\.\d+|nan|inf))"


def parse_step_status(text: Any) -> Optional[Dict[str, Any]]:
    """The writer's ``step status`` reply -> ``{phase, leg, cm, shift,
    abort_reason}``, or ``None`` for anything that is not a STEP fragment.

    The grammar is the writer's own ``_step_status``
    (``skills/writer/main_merged_writer.py``): ``STEP:step=IDLE shift=<s>``,
    ``step=SHIFT(<tgt>)``, ``step=HOLD(...)``, ``step=LIFT(<leg> K=..)``,
    ``step=PLACE|RECENTRE(<leg> <cm>cm)``, ``step=DONE(<leg> <cm>cm
    shift=<s>)``, ``step=ABORT(<reason>)``, each followed by the honesty
    suffix.  A SHIFT or DONE fragment whose numbers do not parse is ``None``
    (unparseable, never guessed).
    """
    if not isinstance(text, str):
        return None
    body = text.strip()
    if not body.startswith("STEP:"):
        return None
    body = body[len("STEP:"):].strip()
    if body.endswith(HONESTY_SUFFIX):
        body = body[:-len(HONESTY_SUFFIX)].rstrip()
    m = _STEP_PHASE_RE.match(body)
    if m is None:
        return None
    phase = m.group(1)
    rest = body[m.end():]
    inner = None
    if rest.startswith("("):
        close = rest.rfind(")")
        if close < 0:
            return None
        inner = rest[1:close].strip()
    out: Dict[str, Any] = {"phase": phase, "leg": None, "cm": None,
                           "shift": None, "abort_reason": None}
    try:
        if phase == "IDLE":
            mm = re.search(r"shift=" + _STEP_NUM, rest)
            if mm:
                out["shift"] = float(mm.group(1))
        elif phase == "SHIFT":
            out["shift"] = float(inner)
        elif phase == "DONE":
            mm = re.match(r"([lr])\s+" + _STEP_NUM + r"cm\s+shift=" + _STEP_NUM
                          + r"$", inner or "")
            if mm is None:
                return None
            out["leg"] = mm.group(1)
            out["cm"] = float(mm.group(2))
            out["shift"] = float(mm.group(3))
        elif phase == "ABORT":
            out["abort_reason"] = inner if inner else "unstated"
        elif phase in ("LIFT", "PLACE", "RECENTRE") and inner:
            out["leg"] = inner.split()[0]
            mm = re.search(_STEP_NUM + r"cm", inner)
            if mm:
                out["cm"] = float(mm.group(1))
    except (TypeError, ValueError):
        return None
    return out


def _step_query(writer_verb: Callable[[str], str], lock: Optional[Any]
                ) -> Optional[str]:
    """One ``step status`` round-trip -> the raw reply, ``None`` on a raise."""
    try:
        if lock is None:
            reply = writer_verb("step status")
        else:
            with lock:
                reply = writer_verb("step status")
    except Exception:  # noqa: BLE001 - a dead socket reads as unparseable
        return None
    return reply if isinstance(reply, str) else None


def _pre_step(ctx: DescentContext, log: DescentPhaseLog, *,
              writer_verb: Callable[[str], str],
              stamp: Callable[[], float],
              clock: Callable[[], float],
              sleep: Callable[[float], None],
              legs: Tuple[str, ...] = PRE_STEP_LEGS,
              shift_deg: float = PRE_STEP_SHIFT_DEG,
              cm: float = PRE_STEP_CM,
              timeout_s: float = PRE_STEP_GO_TIMEOUT_S) -> Tuple[bool, str]:
    """Widen the stance BEFORE the descent, on the writer's own step verbs.
    ``(ok, detail)``.

    Per leg: ``step shift <+/-deg>`` (reply must be exactly ``DONE``), ``step
    status`` POLLED until the writer reports a phase in
    :data:`PRE_STEP_GO_READY_PHASES` (the settled ``IDLE``), every poll showing
    a shift of the right sign and at least :data:`PRE_STEP_MIN_SHIFT_DEG` (a
    clamped-to-zero shift is a SHUT band), bounded by
    :func:`pre_step_shift_timeout_s` (HIGH-B: the real writer refuses ``go``
    with "step busy: SHIFT" while the shift slews), ``step go <leg> <cm>``
    (``DONE`` = STARTED only), then ``step
    status`` polled until ``DONE(<leg> <cm>cm ...)`` with ``cm > 0``.  The
    achieved width is the writer's OWN fragment, quoted verbatim — the lane
    never states a total separation and never calls it measured.

    FAIL CLOSED.  Any ERROR, a shut band, an ``ABORT(...)``, an unparseable or
    ``IDLE`` status after ``go``, or a poll that outlives ``timeout_s`` sends
    ``step off`` (best effort) and returns ``(False, detail)``; the caller
    refuses the WHOLE descent, so ``height`` is never sent and the engine is
    never called.  ``ctx.stepped`` is set BEFORE the first ``step shift`` (a
    failed or timed-out step may have moved the body) and cleared only by a
    ``step off`` the writer answered ``DONE``.
    """
    lock = ctx.writer_lock

    def _off_after_failure(why: str) -> Tuple[bool, str]:
        o_ok, o_reply = _issue(writer_verb, "step off", lock)
        if o_ok:
            ctx.stepped = False
            log.add("step_off", "ok", honest("step off accepted after the "
                                             "refused step (writer: %s)"
                                             % o_reply), stamp())
        else:
            o_detail = honest("step off REFUSED after the refused step (%s) - "
                              "the stance step may still be installed"
                              % (o_reply or "no reply"))
            log.add("step_off", "refuse", o_detail, stamp())
            _loud(o_detail)
        return False, honest("pre-descent step refused, descent NOT "
                             "commanded: %s" % why)

    if ctx.stepped:
        o_ok, o_reply = _issue(writer_verb, "step off", lock)
        if not o_ok:
            detail = honest("pre-descent step refused: a previous stance step "
                            "could not be dropped (`step off` -> %s); descent "
                            "NOT commanded" % (o_reply or "no reply"))
            log.add("step_off", "refuse", detail, stamp())
            _loud(detail)
            return False, detail
        ctx.stepped = False
        log.add("step_off", "ok", honest("step off accepted before a new "
                                         "pre-descent step (writer: %s)"
                                         % o_reply), stamp())
    confirmed: List[str] = []
    for leg in legs:
        sign = 1.0 if leg == "l" else -1.0
        shift_line = "step shift %+.3f" % (sign * abs(float(shift_deg)))
        ctx.stepped = True
        ok, reply = _issue(writer_verb, shift_line, lock)
        if not ok or reply != "DONE":
            log.add("step_shift", "refuse",
                    honest("`%s` refused by the writer (%s)"
                           % (shift_line, reply or "no reply")), stamp())
            return _off_after_failure("`%s` -> %s"
                                      % (shift_line, reply or "no reply"))
        # Poll until the writer may legally take `step go`: SHIFT while the slew
        # settles, then IDLE, and any other phase answers "step busy". Both forms
        # must carry a shift of the right sign and at least the minimum.
        s_timeout = pre_step_shift_timeout_s(shift_deg)
        s_interval = float(DESCENT_POLL_INTERVAL_S)
        s_deadline = clock() + s_timeout
        s_polls = int(math.ceil(s_timeout / s_interval)) + 2
        ready_raw = None
        installed = None
        why = ("the shift never settled to IDLE within %.2fs (|%.1f| deg at "
               "%.1f deg/s)" % (s_timeout, float(shift_deg), PRE_STEP_SLEW_DEG_S))
        for _ in range(max(s_polls, 1)):
            st_raw = _step_query(writer_verb, lock)
            st = parse_step_status(st_raw)
            if st is None:
                why = ("step status unparseable after the shift (%s)"
                       % (_clip(st_raw) or "no reply"))
                break
            ph = st["phase"]
            installed = st.get("shift")
            if ph in ("SHIFT", "IDLE"):
                if (not _finite_num(installed)
                        or sign * float(installed) < PRE_STEP_MIN_SHIFT_DEG):
                    why = ("step shift clamped to %s - stance band shut/too "
                           "small (writer: %s)"
                           % (installed if installed is not None
                              else "an unreadable value",
                              _clip(st_raw) or "no reply"))
                    break
                if ph in PRE_STEP_GO_READY_PHASES:
                    ready_raw = st_raw
                    break
            elif ph == "ABORT":
                why = "step ABORTED by the writer during the shift: %s" % (
                    st["abort_reason"])
                break
            else:
                why = ("unexpected step phase %s while the shift settles (%s)"
                       % (ph, _clip(st_raw)))
                break
            if clock() >= s_deadline:
                why = ("the shift never settled to IDLE within %.2fs (last: %s)"
                       % (s_timeout, _clip(st_raw)))
                break
            sleep(s_interval)
        if ready_raw is None:
            log.add("step_shift", "refuse", honest(why), stamp())
            return _off_after_failure(why)
        log.add("step_shift", "ok",
                honest("`%s` settled IDLE shift=%+.1f (writer-reported)"
                       % (shift_line, float(installed))), stamp())
        go_line = "step go %s %s" % (leg, "%g" % float(cm))
        ok, reply = _issue(writer_verb, go_line, lock)
        if not ok or reply != "DONE":
            log.add("step_go", "refuse",
                    honest("`%s` refused by the writer (%s)"
                           % (go_line, reply or "no reply")), stamp())
            return _off_after_failure("`%s` -> %s"
                                      % (go_line, reply or "no reply"))
        log.add("step_go", "ok", honest("`%s` STARTED (writer: DONE = started, "
                                        "not completed)" % go_line), stamp())
        interval = float(DESCENT_POLL_INTERVAL_S)
        deadline = clock() + max(0.0, float(timeout_s))
        max_polls = int(math.ceil(max(0.0, float(timeout_s)) / interval)) + 2
        done_raw = None
        why = "the step never reported DONE within %.1fs" % float(timeout_s)
        for _ in range(max(max_polls, 1)):
            raw = _step_query(writer_verb, lock)
            st = parse_step_status(raw)
            if st is None:
                why = "step status unparseable (%s)" % (_clip(raw) or "no reply")
                break
            ph = st["phase"]
            if ph == "DONE":
                if st["leg"] == leg and _finite_num(st["cm"]) and st["cm"] > 0.0:
                    done_raw = raw
                else:
                    why = ("step reported DONE for the wrong leg or no width "
                           "(%s)" % _clip(raw))
                break
            if ph == "ABORT":
                why = "step ABORTED by the writer: %s" % st["abort_reason"]
                break
            if ph == "IDLE":
                why = "step status went IDLE before DONE (%s)" % _clip(raw)
                break
            if clock() >= deadline:
                why = ("the step never reported DONE within %.1fs (last: %s)"
                       % (float(timeout_s), _clip(raw)))
                break
            sleep(interval)
        if done_raw is None:
            log.add("step_confirm", "refuse", honest(why), stamp())
            return _off_after_failure(why)
        frag = done_raw.strip()[len("STEP:"):].strip()
        log.add("step_confirm", "ok",
                honest("leg %s step confirmed by the writer's own status: %s"
                       % (leg, frag)), stamp())
        if frag.endswith(HONESTY_SUFFIX):
            frag = frag[:-len(HONESTY_SUFFIX)].rstrip()
        confirmed.append(frag)
    return True, honest("pre-descent step done: %s" % "; ".join(confirmed))


