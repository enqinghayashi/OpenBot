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
