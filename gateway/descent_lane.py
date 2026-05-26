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
