"""
brain/hooks.py — the Hook pipeline: pre-hooks (confirm gate + safety
checks) MUST run before any motion, post-hooks (verifier) after a tool.

Stdlib only.  Platform law (STUART facts): confirmation is required before
ANY motion, and the pre-hook CANNOT be skipped by the agent — enforcement
lives in the agent loop: a pre-hook failure means the motion tool is NOT
called, period.  This module supplies the pipeline and the standard
built-in hooks; it cannot (and does not try to) make the loop honest —
that is the agent loop's job.

FIXED CONTRACT (this module and the agent wiring must agree):
    HookResult(ok: bool, state: str, reason: str = "")
        state is one of HOOK_STATES:
            pre_ok / pre_denied               — from run_pre()
            post_ok / post_pending / post_fail — from run_post()
    Hook(pre_hooks=[...], post_hooks=[...])
        .run_pre(context)              -> list[HookResult]
             pre hooks are `(context: Context) -> HookResult`; ALL must
             pass for motion.
        .run_post(action, observation) -> HookResult   (single result)
             post hooks are `(action, observation) -> HookResult`.
             `action` is the pending action (PendingAction / ToolCall /
             free-text line) with the raw tool reply attached as
             `.tool_result` (or `.result`) — a missing reply verifies
             conservatively as in_progress, never success.
             ok == (verdict.status == 'success'); 'in_progress' ->
             post_pending (ok=False, evidence 're-observe'); 'failure' ->
             post_fail.  run_post is THE single verification path: the
             agent feeds the returned HookResult's reason text back to
             the model.
    default_hooks(observation_source=None, ...) -> Hook
        ships the verifier post-hook (observation_source: an
        ObservationSupplier — a Callable returning WorldObservation —
        used when run_post receives no observation).

Built-ins:
    confirm_gate_hook(confirm=None)   — human gate; default prompts
                                        input() y/n; in a non-interactive
                                        shell it FAILS (EOF -> no motion).
    safety_checks_hook(envelope, freshness_s, now_fn) — deterministic
                                        checks from the platform facts.
    verifier_post_hook(get_obs=None, verifier=None) — maps the action's
                                        tool name through _CONTRACT_FOR_VERB
                                        onto a CONTRACTS key; a missing
                                        mapping or missing contract yields
                                        post_fail 'no contract for <verb>'
                                        (NEVER silent ok).  The SkillVerifier
                                        is constructed ONCE at hook-
                                        construction and held for the hook's
                                        lifetime (the navigate same-epoch
                                        guard is instance state).  Lazy-
                                        imports brain.verifier /
                                        brain.tool_contract (sibling
                                        modules); absent -> post_fail
                                        loudly.

MOTION_TOOLS is a SUPERSET of the brain tool names AND the platform motion
verbs (writer / grip / pick engine).  The bias rule: a false-positive
motion asks for confirmation (safe); a missed motion verb skips the gate
(unsafe) — the set errs toward inclusion.  Query verbs (q/status/list/
armstate/state/ready) and 'stop' (immediate halt — always allowed, no
gate) are deliberately excluded.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from .context import Context
from .embodiment import Embodiment

# lazy import, like gateway
# broken import -> lane FORCED OFF
# honest refusal, no boot crash
try:
    from .descent_law import effective_pick_floor
except Exception as _exc:  # noqa: BLE001 — degrade, never raise
    print("[descent] descent_law import FAILED (%r) — low-workspace lane "
          "FORCED OFF" % (_exc,), flush=True)

    def effective_pick_floor(envelope):
        """Lane-OFF floor: the envelope's own ``pick_min_height_m`` UNCHANGED.

        The fallback used only when brain/descent_law.py is absent.  It
        reproduces the law's switch-off return exactly, so the pick-height
        refusal below stays today's honest capability-block.
        """
        return envelope.pick_min_height_m

if TYPE_CHECKING:
    from .observation import WorldObservation

# motion = brain tools + platform verbs
# superset by design
# see skills/docs
MOTION_TOOLS = frozenset({
    # brain tools
    "pick", "place", "navigate", "open_hand", "close_hand",
    # arms move = motion -> gate + checks
    "pose",
    # spin = FSM yaw sweep -> gate + checks
    "turn_around",
    # platform verbs: writer / grip / pick
    "pick_left",
    "lift", "tuck", "carry", "carry_left", "neutral", "home",
    "payload", "arm", "cmd_set", "drop",
    "open", "close", "open_left", "close_left", "open_both", "close_both",
    "close_adaptive",
})

HOOK_STATES = frozenset({
    "pre_ok", "pre_denied", "post_ok", "post_pending", "post_fail",
})


@dataclass
class HookResult:
    """One hook's verdict under the fixed contract.

    `ok` is the pass/fail bit; `state` is one of HOOK_STATES; `reason`
    carries the honest human/LLM-readable detail (verdict lines for the
    verifier post-hook).
    """

    ok: bool
    state: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.state not in HOOK_STATES:
            raise ValueError(
                f"invalid HookResult state {self.state!r}; must be one of "
                f"{sorted(HOOK_STATES)}"
            )

    @classmethod
    def pre_ok(cls, reason: str = "") -> "HookResult":
        return cls(True, "pre_ok", reason)

    @classmethod
    def pre_denied(cls, reason: str = "") -> "HookResult":
        return cls(False, "pre_denied", reason)

    @classmethod
    def post_ok(cls, reason: str = "") -> "HookResult":
        return cls(True, "post_ok", reason)

    @classmethod
    def post_pending(cls, reason: str = "") -> "HookResult":
        return cls(False, "post_pending", reason)

    @classmethod
    def post_fail(cls, reason: str = "") -> "HookResult":
        return cls(False, "post_fail", reason)

    @property
    def passed(self) -> bool:
        """Compatibility alias for older readers; prefer `ok`."""
        return self.ok

    @property
    def message(self) -> str:
        """Compatibility alias for older readers; prefer `reason`."""
        return self.reason

    def to_string(self) -> str:
        return f"HOOK:{self.state.upper()}|ok={self.ok}|{self.reason}"


@dataclass
class Hook:
    """Pre/post hook pipeline under the fixed contract.

    run_pre MUST gate motion in the agent loop (a pre_denied result means
    the motion tool is NOT called).  run_post is THE single verification
    path for a tool call.
    """

    pre_hooks: List[Callable[[Context], HookResult]]
    post_hooks: List[Callable[[Any, Any], HookResult]]

    def run_pre(self, context: Context) -> List[HookResult]:
        """Run every pre hook (each receives the Context); ALL must pass."""
        return [h(context) for h in self.pre_hooks]

    def run_post(self, action: Any, observation: Any = None) -> HookResult:
        """Run every post hook; return the single aggregate verdict.

        First post_fail wins, else first post_pending, else the last
        post_ok.  With no post hooks configured the honest answer is
        post_pending (nothing was verified — never claim success).
        """
        if not self.post_hooks:
            return HookResult.post_pending("re-observe; no post hooks configured")
        first_fail: Optional[HookResult] = None
        first_pending: Optional[HookResult] = None
        last_ok: Optional[HookResult] = None
        for hook in self.post_hooks:
            result = hook(action, observation)
            if result.state == "post_fail" and first_fail is None:
                first_fail = result
            elif result.state == "post_pending" and first_pending is None:
                first_pending = result
            elif result.state == "post_ok":
                last_ok = result
        if first_fail is not None:
            return first_fail
        if first_pending is not None:
            return first_pending
        if last_ok is not None:
            return last_ok
        return HookResult.post_pending("re-observe; no post hook produced a verdict")

    @staticmethod
    def all_pass(results: List[HookResult]) -> bool:
        return all(r.ok for r in results)

    @staticmethod
    def failures(results: List[HookResult]) -> List[HookResult]:
        return [r for r in results if not r.ok]


# --------------------------------------------------------------------------
# action shapes
# Context / PendingAction / ToolCall / free text
# -> (tool, args, display, tool_result)
# --------------------------------------------------------------------------


def _action_tool(ctx_or_action: Any) -> Optional[str]:
    """The pending tool/verb name, whatever the shape of the input:
    Context (.pending_action / .pending_action_text), PendingAction
    (.tool), ToolCall (.name), or a free-text line (first token)."""
    if ctx_or_action is None:
        return None
    if isinstance(ctx_or_action, str):
        first = ctx_or_action.strip().split()
        return first[0] if first else None
    pending = getattr(ctx_or_action, "pending_action", None)
    if pending is not None:
        return _action_tool(pending)
    text = getattr(ctx_or_action, "pending_action_text", None)
    if text:
        first = str(text).strip().split()
        return first[0] if first else None
    tool = getattr(ctx_or_action, "tool", None) or getattr(ctx_or_action, "name", None)
    return str(tool) if tool else None


def _action_args(ctx_or_action: Any) -> Dict[str, Any]:
    if ctx_or_action is None or isinstance(ctx_or_action, str):
        return {}
    pending = getattr(ctx_or_action, "pending_action", None)
    if pending is not None:
        return _action_args(pending)
    args = getattr(ctx_or_action, "args", None) or getattr(ctx_or_action, "arguments", None)
    return dict(args) if isinstance(args, dict) else {}


#: hand-only verbs, no scene geometry
#: executor sends the platform verb
#: brain names too, direct callers
_HAND_ONLY_TOOLS = frozenset({
    "open_hand", "close_hand",
    "open", "close", "open_left", "close_left", "open_both", "close_both",
})


def _args_carry_xyz(args: Dict[str, Any]) -> bool:
    """True when a pending action carries a workspace coordinate (any of x, y,
    z with a non-None value): such a place consumes scene geometry."""
    return any(args.get(k) is not None for k in ("x", "y", "z"))


def _action_display(ctx_or_action: Any) -> str:
    """Human-readable description for the confirm prompt."""
    if ctx_or_action is None:
        return ""
    if isinstance(ctx_or_action, str):
        return ctx_or_action
    pending = getattr(ctx_or_action, "pending_action", None)
    if pending is not None:
        return _action_display(pending)
    if hasattr(ctx_or_action, "pending_action_text"):
        return getattr(ctx_or_action, "pending", None) or ""  # Context shape
    disp = getattr(ctx_or_action, "display", None)
    if callable(disp):
        try:
            return disp()
        except TypeError:
            pass
    return str(ctx_or_action)


def _action_tool_result(ctx_or_action: Any) -> str:
    """The raw tool reply attached to the action (`.tool_result` or
    `.result`); '' when absent — the verifier maps '' to in_progress
    ('no tool reply recorded'), never success."""
    if ctx_or_action is None or isinstance(ctx_or_action, str):
        return ""
    result = getattr(ctx_or_action, "tool_result", None)
    if result is not None:
        return str(result)
    result = getattr(ctx_or_action, "result", None)
    if result is not None:
        return str(result)
    pending = getattr(ctx_or_action, "pending_action", None)
    if pending is not None:
        return _action_tool_result(pending)
    return ""


def _is_motion(ctx_or_action: Any) -> bool:
    tool = _action_tool(ctx_or_action)
    return tool is not None and tool in MOTION_TOOLS


# --------------------------------------------------------------------------
# The pre-hooks that ship in this module
# --------------------------------------------------------------------------


def confirm_gate_hook(
    confirm: Optional[Callable[[Any], bool]] = None,
    require_confirm: bool = True,
) -> Callable[[Context], HookResult]:
    """Human confirm gate: motion tools require explicit confirmation.

    `confirm(context) -> bool` overrides the default interactive prompt
    (tests and auto-mode inject a stub).  Default prompts `input()`; on
    EOF/empty answer the hook FAILS — non-interactive is no-motion.
    Non-motion (query) tools pass without prompting.

    `require_confirm=False` is the OPERATOR'S OWN SWITCH (directive
    2026-09-26: *"i want to remove the confirmation, it does not need to ask
    shall i proceed, it should just do it immediately, so is pick and
    place"*).  It makes motion proceed WITHOUT asking.  Two things it
    deliberately does NOT do:

      * it does NOT claim a human confirmed.  The HookResult says plainly
        that the gate was disabled by config and no human confirmed — the
        audit trail must not manufacture consent that never happened, or the
        post-hoc record of "who approved this motion" becomes a lie;
      * it does NOT disable anything else.  The safety_checks_hook still
        runs (freshness, envelope, scene), the verifier post-hook still
        runs, and STOP/CANCEL are untouched — they were never part of this
        gate.  What is given up is exactly one thing: the pause between a
        spoken command and the robot moving.

    Default True so the pause is what you get unless someone turns it off
    on purpose.
    """

    def _default_confirm(ctx_or_action: Any) -> bool:
        display = _action_display(ctx_or_action) or "(pending action)"
        try:
            answer = input(f"Confirm motion: {display}? [y/N] ").strip().lower()
        except EOFError:
            return False
        return answer in ("y", "yes")

    gate = confirm if confirm is not None else _default_confirm

    def hook(context: Context) -> HookResult:
        if not _is_motion(context):
            return HookResult.pre_ok("no motion pending; gate skipped")
        if not require_confirm:
            # not "confirmed"
            # approved unasked
            return HookResult.pre_ok(
                "motion APPROVED UNASKED: the human confirm gate is disabled "
                "(confirm_motion: false); no human confirmed "
                f"{_action_display(context)}"
            )
        if gate(context):
            return HookResult.pre_ok(f"confirmed: {_action_display(context)}")
        return HookResult.pre_denied(f"human did NOT confirm: {_action_display(context)}")

    return hook


def safety_checks_hook(
    envelope: Optional[object] = None,
    freshness_s: Optional[float] = None,
    now_fn: Optional[Callable[[], float]] = None,
    observation_source: Optional[Callable[[], object]] = None,
) -> Callable[[Context], HookResult]:
    """Deterministic pre-motion safety checks (platform facts).

    Checks (all must pass for motion tools):
      1. A world observation exists at all.
      2. Detections are fresh (obs.fresh AND age <= freshness_s) — a dead
         detection node is stale, and stale detections never drive motion.
         The age law is evaluated at MOTION time: when the context's
         snapshot has aged past the budget it is re-read once through
         `observation_source` (the same ObservationSupplier the verifier
         post-hook uses — both lanes already hand it to default_hooks)
         before the law applies; a re-read that is also stale still
         denies honestly.  Without a supplier the original denial stands
         (B3: the spoken confirm gate blocks INSIDE the pre-pipeline, so
         a normal confirm round alone would age any pre-gate snapshot
         past freshness_s — the wire is not the thing that went stale).
         CARVE-OUT: this law does NOT govern navigate — it is a
         search-capable verb whose own walk FSM does its own detection
         (SEARCHING window, stop-on-find, stall law, obstruction freeze,
         NOT_FOUND terminal) and consumes NO brain-side geometry at motion
         time, so "walk to X (not currently visible)" dispatches rather
         than refusing on feed liveness.  navigate keeps the confirm gate
         and the no-world-observation refusal; only freshness/age relax.
      3. pick/pick_left z is inside the tabletop envelope — a floor pick
         is a capability-block: refuse honestly, never attempt.
    Non-motion tools pass.  `envelope`/`freshness_s` default to
    Embodiment.defaults().
    """
    emb = Embodiment.defaults() if envelope is None else envelope
    fresh_budget = emb.perception.freshness_s if freshness_s is None else freshness_s
    now = now_fn or time.time

    def hook(context: Context) -> HookResult:
        if not _is_motion(context):
            return HookResult.pre_ok("no motion pending; checks skipped")
        world = context.world
        if (
            world is None
            or not world.fresh
            or now() - world.detections_ts > fresh_budget
        ):
            # gate ran already
            # snapshot may be old, wire fine
            # re-read, then judge the re-read
            re_obs: Optional[object] = None
            if observation_source is not None:
                try:
                    re_obs = observation_source()
                except Exception:  # noqa: BLE001 — a wire hiccup denies, never crashes
                    re_obs = None
            if (
                re_obs is not None
                and getattr(re_obs, "fresh", False)
            ):
                re_age = max(0.0, now() - re_obs.detections_ts)
                if re_age <= fresh_budget:
                    context.world = re_obs
                    return HookResult.pre_ok(
                        "safety checks passed (wire re-observed at motion "
                        f"time, age {re_age:.1f}s)"
                    )
            if re_obs is not None:
                world = re_obs  # deny uses the new read
        if world is None:
            return HookResult.pre_denied(
                "no world observation; cannot confirm scene before motion"
            )
        tool = _action_tool(context)
        # navigate carve-out
        # freshness/age only
        # FSM detects
        # (SEARCHING, stop-on-find,
        # stall, obstruction freeze,
        # NOT_FOUND)
        # no brain geometry
        # gate + no-world refusal bind
        if tool == "navigate":
            return HookResult.pre_ok(
                "navigate dispatch allowed (search-capable verb; its walk FSM "
                "does its own detection); safety checks passed"
            )
        age = max(0.0, now() - world.detections_ts)
        # hand-only carve-out
        # open/close + bare place: no geometry
        # xyz place: law on
        # gate still binds
        if (not world.fresh or age > fresh_budget) and (
            tool in _HAND_ONLY_TOOLS
            or (tool == "place" and not _args_carry_xyz(_action_args(context)))
        ):
            return HookResult.pre_ok(
                f"{tool} dispatch allowed (hand-only motion, no scene geometry "
                "consumed); safety checks passed"
            )
        if not world.fresh:
            return HookResult.pre_denied(
                f"stale detections (fresh=False, ts={world.detections_ts:.1f}); "
                "re-observe before motion"
            )
        if age > fresh_budget:
            return HookResult.pre_denied(
                f"stale detections (age {age:.1f}s > {fresh_budget:.1f}s); "
                "re-observe before motion"
            )
        args = _action_args(context)
        if tool in ("pick", "pick_left") and "z" in args:
            try:
                z = float(args["z"])
            except (TypeError, ValueError):
                return HookResult.pre_denied(
                    f"pick z is not numeric ({args['z']!r}); cannot safety-check the height"
                )
            # floor = shared law
            # lane off: min unchanged
            # (refusal byte-identical)
            # lane on: rescue floor
            # ceiling untouched
            floor = effective_pick_floor(emb.envelope)
            if z < floor:
                # torso frame: z=0 at torso link
                if getattr(emb.envelope, "low_workspace_lane", False) is True:
                    return HookResult.pre_denied(
                        f"floor pick refused: workspace z={z:.2f} (torso frame) "
                        f"< descent floor {floor:.2f} m — below the lane-armed "
                        "low-workspace descent budget; a SIMULATED bound, not "
                        "a measured body capability — refusing honestly"
                    )
                return HookResult.pre_denied(
                    f"floor pick refused: workspace z={z:.2f} (torso frame) "
                    f"< min {emb.envelope.pick_min_height_m:.2f} m — the engine "
                    "is tabletop-only; no ground-pick skill exists "
                    "(capability-block)"
                )
            if z > emb.envelope.pick_max_height_m:
                return HookResult.pre_denied(
                    f"pick out of reach: z={z:.2f} > max {emb.envelope.pick_max_height_m:.2f} m"
                )
        return HookResult.pre_ok("safety checks passed")

    return hook


# --------------------------------------------------------------------------
# The post-hooks that ship in this module
# --------------------------------------------------------------------------

# platform verbs -> CONTRACTS keys
# (pick/place/navigate/open_hand/close_hand)
# no contract = FAIL
_CONTRACT_FOR_VERB = {
    "open": "open_hand",
    "close": "close_hand",
    "open_left": "open_hand",
    "close_left": "close_hand",
    "open_both": "open_hand",
    "close_both": "close_hand",
}

#: hint charset
#: verify_pick's _resolve_target set,
#: brain/verifier.py:
#: target_label\s*=\s*([A-Za-z0-9_.\-]+)
#: same as gateway
#: _HINT_LABEL_CHARS
#: why: 'red cup' sent verbatim,
#: regex cut it to 'red',
#: nothing matched, pick read SUCCESS,
#: cup still on the table
#: outside set -> no hint, never truncated
#: gated on
#: optional_tools.hint_charset_refusal
#: exactly True
#: off = verbatim, defect included
_HINT_LABEL_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")


def _runtime_target_hint(action: Any, refuse_charset: bool = False) -> Optional[str]:
    """``target_label=<label>`` for THIS call, or None when it cannot be honest.

    ``refuse_charset`` (batch-4 D10a switch, default OFF): when True a label
    outside ``_HINT_LABEL_CHARS`` earns no hint at all; when False the hint
    is ``"target_label=%s" % cand`` verbatim (base behaviour).

    WHY THIS EXISTS (2026-09-27, MEASURED on Stuart).  ``verify_pick`` names
    the picked object through ``_resolve_target`` (brain/verifier.py:752),
    which parses the target out of ``contract.context``.  That string comes
    from the STATIC ``CONTRACTS`` table (brain/tool_contract.py) — a prose
    description of the pick engine that carries no ``target=``/``label=``
    hint — so the verifier NEVER saw the object the call actually named.
    ``_target_present`` then answered None on any non-empty scene and
    ``verify_pick`` could only ever return ``in_progress`` 're-observe;
    cannot confirm object gone (no target ref/label in contract context and
    scene non-empty)'.

    That is NOT a timing defect: the verdict is identical whenever it is
    taken, so a pick could not verify success at ANY moment.  MEASURED: that
    exact string was the tool result the model received at 18:49:43,103 —
    one second after the pick that in fact SUCCEEDED (engine_grasp.log:
    ``closure=full delta=7200.0 blocked=True``) — and the brain, told
    "re-observe", re-observed, found the bottle still on the table (the arm
    was still moving), and never called ``place``.  Zero place calls in
    either run.

    FOUR PROPERTIES, each deliberate:

    * LABEL, never a bare ``target=<ref>``.  ``_target_present`` returns None
      for a ref hint when no detection carries a ref — and perception does
      not emit refs (``det.ref`` is never set, see
      [[openbot-label-contract-ref-and-matcher]]) — so a ref hint would
      reinstate exactly the permanent in_progress this fixes.

    * The ``_<n>`` stem is applied to WHICHEVER field carried the ref, and
      only as a FALLBACK: the value exactly as given is matched first, so a
      detection genuinely labelled ``bottle_1`` still beats its own stem and
      is never re-pointed at a different object.  (Before 2026-09-28 the stem
      was confined to the ``ref`` branch, so the same string arriving in
      ``label`` — the shape ``scene_brief``'s own output invites, since it
      prints the ref as the line's first token — silently earned no hint.)

    * The hint is supplied ONLY when the PRE-action observation already
      showed that label (``action.world`` — the snapshot the executor took
      before the tool was called, and never overwritten).  A hint therefore
      means "this label was on the table BEFORE the call", so its absence
      from the post-action packet is real evidence the object left the
      source.  A label that was never seen earns NO hint and the existing
      honest in_progress stands: a mis-spelled label cannot be turned into
      a success by this function.

    * The ``CONTRACTS`` entry is never mutated — the caller gets a
      ``dataclasses.replace`` copy, so one call's target cannot leak into
      the next (the table is module-level shared state).
    """
    pending = getattr(action, "pending_action", None)
    args = getattr(pending, "args", None)
    if not isinstance(args, dict):
        return None
    raw = str(args.get("label") or args.get("ref") or "").strip()
    if not raw:
        return None
    pre = getattr(action, "world", None)
    dets = getattr(pre, "detections", None) or []
    seen = {(getattr(d, "label", None) or "").lower() for d in dets}
    # scene_brief refs: 'bottle_1'
    # stem = label
    # ref can land in label or ref -
    # prints it as first token
    # exact first, stem fallback
    # a real 'bottle_1' never re-pointed
    for cand in (raw, re.sub(r"_\d+$", "", raw)):
        if cand and cand.lower() in seen:
            if refuse_charset is True and not set(cand) <= _HINT_LABEL_CHARS:
                # refused outright
                # never truncated
                return None
            return "target_label=%s" % cand
    # not seen -> no proof it left
    return None


