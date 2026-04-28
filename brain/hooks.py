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
