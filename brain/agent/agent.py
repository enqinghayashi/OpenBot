"""G1Agent — the tool-calling loop that drives the deterministic G1 skills
through a gateway backend, with the worker-core Hook pipeline and verifier.

Per-turn flow (fast path first):
    run(text)
      -> classify_tool_call(text)            (brain.agent.fastpath, zero LLM)
           -> _run_fast_path: human-confirmed single action + post-hook
      -> _llm_loop: Qwen3 tool-calling chat, per-turn CURRENT SCENE
           injection, tool calls handled by _handle_tool_call
      _handle_tool_call(name, args):
        PRE-hooks (worker-core): human confirm gate THEN safety checks on
            RESOLVED args (pick coordinates resolved from fresh detections
            BEFORE the gate, so the safety hook sees z) — a pre-hook failure
            means the motion tool is NEVER called.
        tool.call(augmented_args)             (envelope result)
        POST: re-observe -> scene graph update -> hooks.run_post(pending-
            action Context with the RAW tool reply attached (context.
            tool_result, exactly what the fixed Hook contract documents),
            post observation) — THE single verification path: the agent
            never calls SkillVerifier directly, the verifier lives in the
            post-hook (fixed Hook contract, assignment agent-hook-wiring).
            The returned HookResult(ok, state, reason) is rendered
            back into the verdict-shaped line the model and the budget
            classifier read: post_ok -> VERDICT:success, post_pending ->
            VERDICT:in_progress 're-observe...' (conservative, NEVER
            success), post_fail -> VERDICT:failure + reason.  The agent
            NEVER self-grades success; only a VERDICT:success line proves
            world success.

Failure budget: a VERDICT:failure or ERROR: envelope decrements the budget,
a genuine world-goal VERDICT:success resets it, and in_progress/neutral
never move it (an 'OK:' utility-tool envelope is NEUTRAL — a
scene_brief/query_user success must not erase accumulated failures); once
exhausted, the NEXT failure terminates the task with 'I could not complete
the task: <reason>'.  Tool-call count is capped at max_tool_calls.

Single-flight: run() returns BUSY_TEXT while another run is in progress.

The system prompt ships in prompts/system.txt (this module loads and caps
it).  The brain-sibling modules are imported EAGERLY so a shared-contract
drift is a compile-time error (project convention, docs/design_rationale.md).

CONFIG-SEAM (round-5 config-steering): G1Agent.from_config — the ONE
bootstrap the 'openbot-agent' CLI uses — constructs the validated
``EmbodimentConfig`` (brain/config_loader.py) from the ``embodiment:``
segment and INJECTS it into the RUNNING hooks and tools (no more
hardcoded ``Embodiment()`` defaults in the shipped path):
  - ``agent.embodiment`` holds the injected object; the pre-motion safety
    hook (default_hooks envelope) reads the CONFIGURED
    pick_min_height_m / pick_max_height_m (+ perception.freshness_s as
    the freshness-budget fallback when the agent segment is silent).
  - the tool registry gets ``embodiment=...`` so the pick placeholder
    clamps/offsets with the CONFIGURED max_reach_m /
    grip_center_standoff_m and place_at_spot resolves the CONFIGURED
    base_relative.key_positions.
  - absent ``embodiment:`` segment -> EmbodimentConfig defaults (parity
    with the brain/embodiment.py defaults) — the object graph is uniform,
    never silently conflicting.

v4 L1 slice 1: the tool-call execution machinery described above (and the
result/budget/fastpath-summary helpers it calls) now lives in
``brain/agent/tool_executor.py``; ``G1Agent._handle_tool_call`` is a
one-line delegation with byte-identical result text by design.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config_loader import EmbodimentConfig
from ..context import Context
from ..hooks import Hook, default_hooks
from ..observation import WorldObservation, default_world_observation
from ..scene_graph import SceneGraph
from .fastpath import classify_tool_call
from .optional_tools import switch_on
from .llm import (
    ChatResult,
    ChatTransportError,
    ToolCall,
    ToolCallParseError,
    from_config as llm_from_config,
)
# re-export: tool_executor owns it
# tests read it here
from .tool_executor import COMPLETE_FAIL_PREFIX, ToolExecutor, _PLATFORM_VERB
from .tools import ToolRegistry, fail

BUSY_TEXT = "The agent is busy handling another request; try again shortly."
BUSY_LLM_TEXT = "The model is busy handling another request; try again shortly."

#: prompt char cap
#: why 6400?
#: 6000 -> 6400
#: old lines +383
#: md5 (old file) 485a671d6a3d06b37da03735e87d601a
#: '2. HAND IT TO ME'
#: 5915..5989 -> 6298..6372
#: 6000 cut THREE SEQUENCES
#: FETCH recipe
#: 28 chars margin
#: 6000 impossible
#: ~380 chars pinned
#: (test_b5_brain_task_flow.py)
#: relational pin catches displacement
SYSTEM_PROMPT_MAX_CHARS = 6400
MAX_KEPT_HISTORY_MESSAGES = 14  # first has the scene, then last N
MAX_TOOL_RESULT_CHARS = 2000

_JSON_REPAIR_MESSAGE = (
    "Your previous output was not valid JSON. Return exactly one tool call "
    "as valid JSON — nothing else."
)


def _backend_from_gateway_segment(gateway_seg: Any) -> Any:
    """Build the gateway backend the ``gateway:`` config segment demands
    (round-4: G1Agent.from_config's ONE bootstrap wires the real client).

    transport 'mock' -> _LaptopMockBackend (in-process; the laptop never
    connects anywhere and refuses every verb honestly); transport 'real'
    -> GatewayCallerBackend(host, port) — the socket client to the
    gateway node's tools server (the production Stuart path); any OTHER
    transport value raises an honest ConfigError — never guess a
    transport.  A 'real' host that is a placeholder ('TODO-IP' /
    'localhost-TODO' / '') raises ConfigError from the client BEFORE any
    socket is opened ('gateway.host is a placeholder; fill it in
    preflight (D0)').
    """
    # lazy on purpose
    # brain_client -> tools -> __init__
    # -> this module
    # top-level import would circle back
    from gateway.brain_client import (  # noqa: PLC0415
        ConfigError,
        DEFAULT_PORT,
        GATEWAY_CALL_TIMEOUT_S,
        GatewayCallerBackend,
    )

    if gateway_seg is None:
        gateway_seg = {}
    if not isinstance(gateway_seg, dict):
        raise ConfigError(
            "config error: 'gateway' must be a mapping, got %s"
            % type(gateway_seg).__name__
        )
    transport = str(gateway_seg.get("transport") or "mock").strip().lower()
    if transport == "mock":
        return _LaptopMockBackend()
    if transport == "real":
        host = str(gateway_seg.get("host") or "").strip()
        # falsy port = key absent
        # 0/None/'' -> default port
        # no port 0
        port = gateway_seg.get("port") or DEFAULT_PORT
        try:
            port = int(port)
        except (TypeError, ValueError):
            raise ConfigError(
                "config error: gateway.port must be an int, got %r" % (port,)
            ) from None
        # placeholder host -> ConfigError
        # no quiet connect
        # timeout_s: SECOND constructor
        # same client
        # 30 s default would make
        # the pick motion-wait INERT
        # DONE ~3 s in, arm ~33 s
        # imported, not repeated
        # two bounds drift
        return GatewayCallerBackend(host=host, port=port,
                                    timeout_s=GATEWAY_CALL_TIMEOUT_S)
    raise ConfigError(
        "config error: gateway.transport must be 'mock' or 'real', got %r"
        % transport
    )


class G1Agent:
    """One agent: LLM client + tool registry + hook pipeline (pre-gate and
    the verifier post-hook, THE single verification path)."""

    def __init__(
        self,
        llm: Any,
        tool_registry: Any,
        backend: Any = None,
        *,
        confirm: Optional[Callable[[Context], bool]] = None,
        hooks: Optional[Hook] = None,
        scene_graph: Optional[SceneGraph] = None,
        embodiment: Optional[Any] = None,
        max_tool_calls: int = 25,
        failure_budget: int = 3,
        detection_freshness_s: float = 2.0,
        llm_timeout_s: float = 12.0,
        max_llm_attempts: int = 2,
        max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS,
        prompt_path: Optional[str] = None,
    ) -> None:
        if llm is None:
            raise ValueError("G1Agent requires an LLMClient")
        if tool_registry is None:
            raise ValueError("G1Agent requires a ToolRegistry")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if failure_budget < 0:
            raise ValueError("failure_budget must be >= 0")
        self.llm = llm
        self.tools = tool_registry
        self.backend = backend
        # det-conf floor
        # graph built here
        # takes the injected
        # perception.det_conf_floor
        # explicit scene_graph keeps its own
        # no embodiment -> 0.0
        # filter off
        # 0.35 from config
        self.scene_graph = scene_graph if scene_graph is not None else SceneGraph(
            conf_floor=getattr(
                getattr(embodiment, "perception", None), "det_conf_floor", 0.0
            ),
            # optional_tools via the registry
            # (from_config passes switches=...)
            # absent = OFF
            target_disambiguation=switch_on(
                getattr(tool_registry, "_optional_switches", None),
                "target_disambiguation"),
        )
        self.max_tool_calls = int(max_tool_calls)
        self.failure_budget = int(failure_budget)
        self.detection_freshness_s = float(detection_freshness_s)
        self.llm_timeout_s = float(llm_timeout_s)
        self.max_llm_attempts = int(max_llm_attempts)
        self.max_tool_result_chars = int(max_tool_result_chars)
        # injected embodiment
        # config-seam: loader's EmbodimentConfig,
        # or anything with .envelope/.perception/
        # .base_relative
        # None -> documented defaults
        self.embodiment = embodiment
        # default pipeline
        # gate before motion
        # verifier POST-hook after
        # (run_post -> HookResult)
        # pass hooks= to swap
        # confirm= honoured when default_hooks
        # takes it (scripted tests)
        # safety hook gets the injected envelope
        # (pick_min/max_height_m, freshness)
        # + detection_freshness_s at runtime
        # not Embodiment() defaults
        self.hooks = (
            hooks
            if hooks is not None
            else default_hooks(
                observation_source=self._observe,
                confirm=confirm,
                envelope=self.embodiment,
                freshness_s=self.detection_freshness_s,
                # same switches as above
                # absent = OFF = verbatim hint
                hint_charset_refusal=switch_on(
                    getattr(tool_registry, "_optional_switches", None),
                    "hint_charset_refusal"),
            )
        )
        self._system_prompt, self._prompt_truncated = self._load_system_prompt(prompt_path)

        # state for one run
        self._busy = False
        self._history: List[dict] = []
        self._task_text = ""
        # per-run executor
        # built lazily
        # run() nulls it; each utterance
        # gets a fresh one
        self._executor: Optional[ToolExecutor] = None
        self._terminal_text: Optional[str] = None
        self._failure_budget_left = self.failure_budget
        self._tool_call_count = 0
        self.transcript: List[dict] = []

    # --------------------------------------------------------- getting set up

    @staticmethod
    def _load_system_prompt(prompt_path: Optional[str]) -> tuple[str, bool]:
        """Load prompts/system.txt (capped at SYSTEM_PROMPT_MAX_CHARS)."""
        if prompt_path:
            path = Path(prompt_path)
        else:
            path = Path(__file__).parent / "prompts" / "system.txt"
        if not path.is_absolute():
            path = Path(__file__).parent / path
        if not path.is_file():
            raise FileNotFoundError(f"system prompt not found: {path}")
        text = path.read_text(encoding="utf-8").strip()
        truncated = len(text) > SYSTEM_PROMPT_MAX_CHARS
        if truncated:
            text = text[:SYSTEM_PROMPT_MAX_CHARS]
        return text, truncated

    # --------------------------------------------------------- the public API

    def run(self, text: str) -> str:
        """One task utterance. Single-flight: BUSY_TEXT while another runs."""
        if self._busy:
            return BUSY_TEXT
        self._busy = True
        try:
            # fresh executor per utterance
            # budget reset here, same spot as before
            # built on first state touch
            self._executor = None
            self._task_text = (text or "").strip()
            self._history = []
            self._terminal_text = None
            self._failure_budget_left = self.failure_budget
            self._tool_call_count = 0
            # one tool set for the whole run
            # snapshot(), not get()
            self._tools = self.tools.snapshot()
            if not self._task_text:
                return "I did not understand the request."
            call = classify_tool_call(self._task_text)
            if call is not None:
                return self._run_fast_path(call)
            return self._llm_loop()
        finally:
            self._busy = False
            self.transcript = [dict(m) for m in self._history]

    # --------------------------------------------------------- the fast lane

