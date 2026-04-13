"""ToolExecutor — G1Agent's tool-call execution machinery, extracted from
brain/agent/agent.py (OpenBot v4 L1 slice 1).  G1Agent._handle_tool_call is
now a one-line delegation to ToolExecutor.execute(); the result text is
byte-identical to the pre-extraction agent by design (same bodies, same
strings, same semantics — the only signature change is the entry point,
``_handle_tool_call(self, call)`` -> ``execute(self, name, args)``, and the
result is carried out as a ToolOutcome whose ``text`` is exactly what the
caller emits).  A fresh executor is built per run() so the per-utterance
failure-budget reset happens at the same point in run()'s sequence as
before, while the persistent SkillVerifier keeps living in the hooks
(navigate-epoch guard survives).  Stdlib-only: imports from the brain
packages and brain.agent (fastpath, tools) — never third-party modules.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..context import Context, PendingAction
from ..hooks import HookResult, MOTION_TOOLS
from ..observation import WorldObservation
from ..self_state import SELF_STATE_MARK
from ..verifier import Verdict, containment_predrop_snapshot
from .tools import envelope_to_str, fail

log = logging.getLogger(__name__)

# how far the in-call re-observe loop goes.
# containment wants TWO CONSECUTIVE fresh
# in-region frames, ONE dropped_at epoch,
# inside RE_OBSERVE_WINDOW_S (brain/verifier.py).
# a new place re-stamps the epoch,
# so re-running the tool never builds the pair.
# one observation never certifies.
# so: re-observe INSIDE one call.  settle
# spacing keeps each frame in the window,
# packet inside POST_DROP_FRESHNESS_S when judged.
# past the cap: honest in_progress,
# never silence, never a made-up success.
_RE_OBSERVE_SETTLE_S = 1.4
_MAX_RE_OBSERVES = 2

# failure-exhaustion prefix.  one source of truth:
# brain/agent/agent.py imports it from here
# (can't import back — circular)
COMPLETE_FAIL_PREFIX = "I could not complete the task: "

#: the pick reply carries this during a fetch
#: (tools.py _FETCH_NEXT_STEP).  walk home after
_FETCH_MARK = "NEXT_STEP=fetch_return"
#: place handover opened the hand in place
#: (tools.py Tools.HANDOVER_OPEN_MARK).  judge as open
_HANDOVER_OPEN_MARK = "HANDOVER=open"

# MOTION_TOOLS speaks the writer/grip verbs;
# 'open'/'close' are the platform verbs behind
# open_hand/close_hand.  place_at_spot -> 'place'
# (engine 'drop', same release-only motion),
# so gate, safety and verifier treat it as place
_PLATFORM_VERB = {
    "pick": "pick",
    "place": "place",
    "place_at_spot": "place",
    "navigate": "navigate",
    "open_hand": "open",
    "close_hand": "close",
}

_VERDICT_RE = re.compile(r"VERDICT:(success|failure|in_progress)\|Evidence:(.*)", re.DOTALL)
_REASON_RE = re.compile(r"\|Reason:(.*)$", re.DOTALL)


@dataclass
class ToolOutcome:
    """One tool execution's outcome, shaped for the caller's branching.

    ``text`` is the exact result text the caller emits (verdict line or
    envelope string — byte-identical to the pre-extraction
    _handle_tool_call return).  ``kind``/``reason`` carry the
    _classify_result classification of that text, and ``terminal`` is True
    when the result exhausted the failure budget (the caller then emits
    the COMPLETE_FAIL_PREFIX text, which the executor also mirrors on its
    own ``_terminal_text`` slot).
    """

    text: str
    kind: str
    reason: str = ""
    terminal: bool = False


class ToolExecutor:
    """The tool-call execution machinery (extracted verbatim from G1Agent;
    see the module docstring).  Carries the same attribute names the
    original method bodies reference — ``_tools`` (the per-run registry
    snapshot), ``scene_graph``, ``hooks``, ``backend``, ``embodiment``,
    ``failure_budget``, ``_failure_budget_left``, ``_terminal_text`` — so
    the moved bodies resolve ``self.X`` exactly as before.  Fresh per
    run(): the per-utterance budget reset lives here.
    """

    def __init__(
        self,
        *,
        registry: Any,
        hooks: Any,
        scene_graph: Any,
        backend: Any,
        embodiment: Any,
        failure_budget: int,
    ) -> None:
        self.registry = registry
        self.hooks = hooks
        self.scene_graph = scene_graph
        self.backend = backend
        self.embodiment = embodiment
        self.failure_budget = int(failure_budget)
        # per-utterance state: a fresh executor per run()
        # resets the failure budget; G1Agent nils _executor
        # at the top of run()
        self._failure_budget_left = self.failure_budget
        self._terminal_text: Optional[str] = None
        # the tool set this run runs against — a registry
        # snapshot (a mapping with .get; ToolRegistry has
        # none).  G1Agent's run() pushes its own here before
        # any execute(); `registry` is the lazy builder's
        # fresh snapshot — an early execute() still resolves
        # something honest
        self._tools = registry

    # ---------------------------------------------------- running one tool call

    def execute(self, name: str, args: dict) -> ToolOutcome:
        """PRE-hooks -> tool -> POST run_post; returns the result text the
        LLM/human sees (verdict line or envelope string)."""
        # carry the tool name for _finish_tool_result's log.
        # one executor per run(), execute() never re-enters —
        # just a per-call stamp
        self._current_tool = name
        if name == "stop":
            return self._finish_tool_result(envelope_to_str(self._stop(args)))

        tool = self._tools.get(name)
        if tool is None:
            return self._finish_tool_result(
                envelope_to_str(fail(f"unknown or invalid tool call: {name!r}"))
            )
        violations = tool.validate_args(args)
        if violations:
            detail = "; ".join(violations)
            return self._finish_tool_result(
                envelope_to_str(fail(f"invalid arguments: {detail}"))
            )
        # prepare resolves coords/refs BEFORE the gate.
        # the safety hook sees the resolved z (floor-pick
        # refusal), confirm names the concrete action
        prepared, refusal = tool.prepare(args)
        if refusal is not None:
            return self._finish_tool_result(envelope_to_str(refusal))

        obs = self._observe()
        # the verifier only sees POST-tool observations, but
        # containment's substitution blocker needs the
        # PRE-DROP scene — this read, before run_pre and
        # tool.call.  stash it in the prepared args; place
        # sticks it on the reply (predrop_state/predrop_obj)
        # as the verifier's anchor.  nothing else is touched
        # (only place prepare writes into_ref/into_label)
        if prepared.get("into_ref") or prepared.get("into_label"):
            prepared["_predrop_containment"] = containment_predrop_snapshot(
                obs, prepared.get("into_label"), prepared.get("label"))
        # scene graph keeps the PRE-action scene so
        # apply_verdict can still find the ref after a
        # confirmed success (post-tool obs may not have it)
        if (
            self.scene_graph is not None
            and obs is not None
            and obs.fresh
            and obs.detections_ts > 0
        ):
            self.scene_graph.update(obs)
        context = Context.empty()
        context.world = obs
        context.pending_action = PendingAction(
            tool=self._platform_verb(name),
            args=prepared,
            description=self._describe_action(name, prepared),
        )
        # run_pre is still the confirm gate, unchanged and
        # unskippable: ALL pre results must pass or the
        # motion tool never runs.  HookResult as-is
        results = self.hooks.run_pre(context)
        if not all(r.ok for r in results):
            blocked = "; ".join(r.reason or r.state for r in results if not r.ok)
            return self._finish_tool_result(
                envelope_to_str(fail(f"pre-hook blocked motion: {blocked}"))
            )

        env = tool.call(prepared)
        post_obs = self._observe()
        if (
            self.scene_graph is not None
            and post_obs is not None
            and post_obs.fresh
            and post_obs.detections_ts > 0
        ):
            self.scene_graph.update(post_obs)
        verb = self._platform_verb(name)
        if verb in MOTION_TOOLS:
            # THE one verification path (fixed Hook contract):
            # the verifier lives in hooks.run_post; the agent never
            # calls SkillVerifier itself.  run_post sees the RAW
            # reply on context.tool_result (hooks read .tool_result
            # / .result), parsed the way the gateway lane does:
            # 'DONE:contact', 'DONE ...; NAV_EPOCH=...', 'ERROR:'.
            # query tools aren't motion — envelope reply only
            context.tool_result = self._tool_result_text(env)
            if verb == "place" and _HANDOVER_OPEN_MARK in context.tool_result:
                # a handover place opens the hand in place, no drop:
                # judge it as an open, measured-open
                context.pending_action = PendingAction(
                    tool="open", args=prepared,
                    description=context.pending_action.description)
            result = self.hooks.run_post(context, post_obs)
            # place and pick re-judge the same reply against newer fresh
            # packets: the first verdict can land before the decisive frame,
            # and a retry would re-run the motion instead.
            if verb in ("place", "pick") and result.state == "post_pending":
                result = self._reobserve_post(context, result, post_obs)
            result_text = self._hook_result_to_text(result)
            if self.scene_graph is not None:
                _verdict = self._hook_result_to_verdict(result)
                if self._apply_verdict_takes_label():
                    self.scene_graph.apply_verdict(
                        _verdict, name, prepared.get("ref"),
                        label=prepared.get("label"), args=prepared,
                    )
                else:  # legacy 3-arg signature
                    self.scene_graph.apply_verdict(
                        _verdict, name, prepared.get("ref"))
                _ss = getattr(self.scene_graph, "self_state", None)
                if _ss is not None:
                    _ss.record(_verdict, name, prepared)
            # fetch pick: walk back to the start point right
            # here, AFTER judging — the verdict saw the table,
            # not the walk home.  the voice model ignored the
            # "call return_home" hint in 1 of 2 runs.
            if (verb == "pick" and result.state != "post_fail"
                    and _FETCH_MARK in (context.tool_result or "")):
                result_text = self._auto_return(result_text)
        else:
            result_text = envelope_to_str(env)
        return self._finish_tool_result(result_text)

    def _auto_return(self, text: str) -> str:
        """B8l-hf4: run return_home after a fetch pick; append its outcome and
        the next step.  Never raises; its 'ERROR:' is reworded so the pick's
        own classification is untouched (any 'ERROR:' marks a reply failed)."""
        tool = self._tools.get("return_home") if self._tools is not None else None
        if tool is None:
            return text
        log.info("fetch pick: walking back to the start point (auto return_home)")
        try:
            env = tool.call({})
        except Exception as exc:  # noqa: BLE001 -- the pick result stands either way
            env = fail("return_home raised: %s" % exc)
        rt = envelope_to_str(env).replace("ERROR:", "failed:")
        if isinstance(env, dict) and env.get("success"):
            return (text + "\nAUTO_RETURN: done - " + rt + "\nNEXT: you are back at "
                    "the start point facing the person who asked: place with "
                    "into='person' now (navigate to 'person' first if they are "
                    "out of reach).")
        return (text + "\nAUTO_RETURN: did not finish - " + rt + "\nNEXT: you still "
                "hold the object; tell the person, and call return_home again "
                "when the way is clear.")

    def _apply_verdict_takes_label(self) -> bool:
        try:
            import inspect
            params = inspect.signature(self.scene_graph.apply_verdict).parameters
        except (TypeError, ValueError, AttributeError):
            return False
        return "label" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _finish_tool_result(self, text: str) -> ToolOutcome:
        """Classify the result and apply failure-budget bookkeeping."""
        kind, reason = self._classify_result(text)
        # those SPOKEN failure sentences are reason-free on
        # purpose — operator: "failed, try again".  but `reason`
        # only lived here and got dropped: two DICTATED
        # refusals ('floor pick refused: workspace z=... <
        # min -0.35 m', 'pick engine not configured
        # (engine_cmd empty)') and every engine/verifier error
        # read the same in the transcript.  a post-mortem then
        # had to read source (it cost a whole multi-agent
        # trace).  log it where kind and reason get computed —
        # the reason is already in the text, verbatim.
        if kind in ("failure", "unknown"):
            log.warning(
                "tool %s -> %s: %s",
                getattr(self, "_current_tool", "?"),
                kind,
                text.replace("\n", " ")[:400],
            )
        self._apply_budget(kind, reason)
        # expose self-state on every reply except the
        # offline-parsed query tools (classification used RAW)
        _ss = getattr(self.scene_graph, "self_state", None)
        _line = _ss.render() if _ss is not None else ""
        if _line and getattr(self, "_current_tool", None) not in (
                "scene_brief", "resolve_ref"):
            text = text + SELF_STATE_MARK + _line[len("SELF_STATE: "):]
        return ToolOutcome(
            text=text,
            kind=kind,
            reason=reason or "",
            terminal=self._terminal_text is not None,
        )

    def _platform_verb(self, name: str) -> str:
        return _PLATFORM_VERB.get(name, name)

    @staticmethod
    def _describe_action(name: str, args: dict) -> str:
        if name == "pick":
            return f"pick up {(args.get('label') or '?')}"
        if name == "place":
            # an into:<container> place is a containment request:
            # the gate must name the container ('place bottle into
            # bin', never 'at bin').  no-into keeps the old wording
            into = (args.get("into") or "").strip()
            if into:
                return f"place {(args.get('label') or '?')} into {into}"
            return f"place {(args.get('label') or '?')} at {(args.get('destination') or '?')}"
        if name == "place_at_spot":
            return (
                f"place {(args.get('label') or '?')} at spot "
                f"{(args.get('spot') or '?')}"
            )
        if name == "navigate":
            return f"navigate to {(args.get('target') or '?')}"
        if name in ("open_hand", "close_hand"):
            verb = "open" if name == "open_hand" else "close"
            return f"{verb} {(args.get('side') or 'right')} hand"
        return f"{name} {args}"

    @staticmethod
    def _tool_result_text(env: dict) -> str:
        """The raw tool reply the verifier must judge — mirrors the
        gateway's own lane (gateway_node feeds SkillVerifier the reply
        string verbatim, annotations included).  Success envelopes carry
        the raw slave reply in ``value`` ('DONE:contact', 'DONE pick
        executed; grasp_verified=...', 'DONE walk triggered: ...;
        NAV_EPOCH=...'); failure envelopes serialize to 'ERROR: <reason>'
        — the verifier treats any ERROR: marker anywhere in the reply as
        definitive failure.  NEVER 'OK: <value>': the verifier's reply
        classifier only reads DONE:/ERROR:/Q:/STATE:-prefixed lines, so an
        OK:-wrapped reply would classify as 'unknown' and can never
        confirm success."""
        if not isinstance(env, dict):
            return ""
        if env.get("success"):
            value = env.get("value")
            return "" if value is None else str(value)
        return f"ERROR: {env.get('reason') or 'unknown error'}"

