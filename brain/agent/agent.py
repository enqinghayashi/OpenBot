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

    def _run_fast_path(self, call: ToolCall) -> str:
        """Deterministic lane: no LLM involved; full hook/verifier pipeline."""
        if call.name in ("pick", "place") and self._scene_is_empty():
            return "I can't see any objects right now."
        result = self._handle_tool_call(call)
        kind, reason = self._classify_result(result)
        self._history = [
            {"role": "user", "content": self._task_text},
            {"role": "assistant", "content": result},
        ]
        return self._fastpath_summary(call.name, call.arguments or {}, kind, reason)

    def _fastpath_summary(
        self, name: str, args: dict, kind: str, reason: Optional[str]
    ) -> str:
        """Delegation (v4-L1 extraction): the real summarizer lives in
        ToolExecutor (brain/agent/tool_executor.py); the phrasing is
        byte-identical."""
        return self._get_executor()._fastpath_summary(name, args, kind, reason)

    # ------------------------------------------------------------ the LLM lane

    def _llm_loop(self) -> str:
        """Qwen3 tool-calling loop with per-turn CURRENT SCENE injection."""
        self._history = [
            {"role": "user", "content": self._render_first_message()}
        ]
        while True:
            if self._terminal_text is not None:
                return self._terminal_text
            if self._tool_call_count >= self.max_tool_calls:
                return (
                    COMPLETE_FAIL_PREFIX
                    + f"reached the per-task tool-call limit ({self.max_tool_calls})"
                )
            self._history[0] = {
                "role": "user",
                "content": self._render_first_message(),
            }
            reply = self._chat_once()
            if reply is None or reply.dropped:
                return BUSY_LLM_TEXT if reply and reply.dropped else "The LLM produced no response."
            if reply.tool_calls:
                self._tool_call_count += len(reply.tool_calls)
                self._history.append(
                    {
                        "role": "assistant",
                        "content": reply.text or "",
                        "tool_calls": [
                            self._as_openai_call(c) for c in reply.tool_calls
                        ],
                    }
                )
                for call in reply.tool_calls:
                    result = self._handle_tool_call(call)
                    if self._terminal_text is not None:
                        return self._terminal_text
                    self._history.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            "content": self._truncate(result),
                        }
                    )
                self._compact_history()
                continue
            final = (reply.text or "").strip()
            return final if final else "The model returned an empty response."

    def _chat_once(self) -> Optional[ChatResult]:
        """One LLM call with timeout + repair attempts (max_llm_attempts).

        Each attempt runs on its OWN single-worker executor: a decode
        slower than ``llm_timeout_s`` abandons its worker (executor shut
        down with ``wait=False``; the daemon thread keeps decoding until
        the client call finishes) and the NEXT attempt starts on a fresh
        worker instead of queueing behind the stuck one.  With the old
        shared executor the abandoned future occupied the only worker, so
        every retry queued behind it and double-timed-out — a slow decode
        could permanently wedge retries.  Tradeoff: the abandoned decode
        still holds the LLM client's single-flight lock, so a retry that
        starts while it is still decoding is DROPPED by the client
        (BUSY_LLM_TEXT) rather than queued — the wedge is never permanent,
        and the retry runs whenever the first call finishes inside the
        timeout window.  A fresh executor per attempt costs one thread
        spawn per attempt (bounded by max_llm_attempts).

        Backend exceptions NEVER escape run(): any exception raised by the
        future (ModelLoadError on a missing/invalid model_path, transport
        failures, anything else) is recorded and routed through the same
        failure text, so the agent never raises backend exceptions to the
        caller.
        """
        last_error: Optional[Exception] = None
        for _attempt in range(1, self.max_llm_attempts + 1):
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="agent-llm"
            )
            try:
                future = executor.submit(
                    self.llm.chat, self._history, self.tools.get_specs()
                )
                try:
                    reply = future.result(timeout=self.llm_timeout_s)
                except concurrent.futures.TimeoutError:
                    last_error = TimeoutError(
                        f"LLM chat timed out after {self.llm_timeout_s:g}s"
                    )
                    self._history.append(
                        {
                            "role": "user",
                            "content": (
                                f"You did not respond within {self.llm_timeout_s:g}s. "
                                "Continue the task now."
                            ),
                        }
                    )
                    continue
                except ToolCallParseError as exc:
                    last_error = exc
                    self._history.append({"role": "user", "content": _JSON_REPAIR_MESSAGE})
                except ChatTransportError as exc:
                    last_error = exc  # transient, retry as-is
                except Exception as exc:  # noqa: BLE001 — never crash run()
                    last_error = exc  # blew up, retry, then fail text
                else:
                    # drop after an earlier failure = its
                    # tail: abandoned decode still holds
                    # the single-flight lock
                    # 'busy' = symptom
                    # surface the real last_error
                    # busy text only for a first drop
                    if reply is not None and reply.dropped and last_error is not None:
                        return ChatResult(
                            text=(
                                f"LLM chat failed after "
                                f"{self.max_llm_attempts} attempts: {last_error}"
                            )
                        )
                    return reply
            finally:
                # wait=False: never block a hung handler
                # daemon worker (3.9+) finishes the call
                executor.shutdown(wait=False)
        return ChatResult(
            text=f"LLM chat failed after {self.max_llm_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _as_openai_call(call: ToolCall) -> dict:
        return {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments or {}),
            },
        }

    def _compact_history(self) -> None:
        """Drop middle turns; keep the scene-carrying first message."""
        if len(self._history) <= MAX_KEPT_HISTORY_MESSAGES + 1:
            return
        self._history = [
            self._history[0],
            *self._history[-(MAX_KEPT_HISTORY_MESSAGES):],
        ]

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_result_chars:
            return text
        return text[: self.max_tool_result_chars] + "\n...[truncated]"

    # -------------------------------------------------- handling the tool calls

    def _handle_tool_call(self, call: ToolCall) -> str:
        """One-line delegation (v4-L1 extraction): the whole PRE-hooks ->
        tool -> POST pipeline lives in ToolExecutor
        (brain/agent/tool_executor.py); the result text is byte-identical
        by design."""
        return self._get_executor().execute(call.name, dict(call.arguments or {})).text

    # -- wiring up the ToolExecutor -------------------------------------------

    def _get_executor(self) -> ToolExecutor:
        """The per-run ToolExecutor, built lazily from self attributes
        (run() resets ``_executor`` to None, so each utterance builds a
        fresh one — the per-utterance failure-budget reset lives in the
        executor at the same point in run()'s sequence as before).

        NOTE: the executor-backed attribute proxies below
        (``_terminal_text`` / ``_failure_budget_left`` / ``_tools``) route
        through this getter, so touching one before the first run() builds
        a THROWAWAY executor (one extra registry snapshot — harmless; no
        production path touches them before run() sets its state)."""
        if self._executor is None:
            self._executor = ToolExecutor(
                registry=self.tools.snapshot(),
                hooks=self.hooks,
                scene_graph=self.scene_graph,
                backend=self.backend,
                embodiment=self.embodiment,
                failure_budget=self.failure_budget,
            )
        return self._executor

    @property
    def _terminal_text(self) -> Optional[str]:
        """Per-run terminal text, proxied to the current ToolExecutor (the
        executor owns the budget/terminal state after the v4-L1
        extraction; run() and _llm_loop keep reading this attribute
        unchanged)."""
        return self._get_executor()._terminal_text

    @_terminal_text.setter
    def _terminal_text(self, value: Optional[str]) -> None:
        self._get_executor()._terminal_text = value

    @property
    def _failure_budget_left(self) -> int:
        """Per-run failure budget, proxied to the current ToolExecutor
        (the agent-side surface the tests exercise; the real bookkeeping
        lives on the executor's slot)."""
        return self._get_executor()._failure_budget_left

    @_failure_budget_left.setter
    def _failure_budget_left(self, value: int) -> None:
        self._get_executor()._failure_budget_left = value

    @property
    def _tools(self) -> Any:
        """Per-run registry snapshot, proxied to the current ToolExecutor
        (run()'s ``self._tools = self.tools.snapshot()`` feeds the
        executor's execute())."""
        return self._get_executor()._tools

    @_tools.setter
    def _tools(self, value: Any) -> None:
        self._get_executor()._tools = value

    @staticmethod
    def _hook_result_to_text(result: "HookResult") -> str:
        """Delegation (v4-L1 extraction): the real renderer lives in
        ToolExecutor; kept here because the test surface calls it on the
        class."""
        return ToolExecutor._hook_result_to_text(result)

    @classmethod
    def _classify_result(cls, text: str) -> tuple[str, Optional[str]]:
        """Delegation (v4-L1 extraction): the real classifier lives in
        ToolExecutor; kept here because the fast path and the tests call
        it on the agent."""
        return ToolExecutor._classify_result(text)

    def _apply_budget(self, kind: str, reason: Optional[str]) -> None:
        """Delegation (v4-L1 extraction): the real bookkeeping lives in
        ToolExecutor, whose per-run budget state this agent proxies
        (``_failure_budget_left`` / ``_terminal_text``)."""
        self._get_executor()._apply_budget(kind, reason)

    # ------------------------------------------------------- reading the world

    def _observe(self) -> Optional[WorldObservation]:
        """One guarded observation read (never raises; None = no source)."""
        if self.backend is None:
            return None
        try:
            obs = self.backend.observe()
        except Exception:  # noqa: BLE001 — a gateway hiccup must not crash
            return None
        if obs is None or not isinstance(obs, WorldObservation):
            return None
        return obs

    def _scene_is_empty(self) -> bool:
        """True when no fresh detection and no persistent refs exist
        (event-driven silence == empty scene, never 'no data')."""
        obs = self._observe()
        if obs is None:
            return True
        if not obs.fresh or obs.detections_ts <= 0:
            return True
        if obs.detections:
            return False
        if self.scene_graph is not None and self.scene_graph.refs:
            return False
        return True

    def _scene_block(self) -> str:
        """Per-turn CURRENT SCENE text (scene graph brief or raw detections)."""
        obs = self._observe()
        if obs is None:
            return "SCENE: not available (no observation source)"
        if obs.detections_ts <= 0 or not obs.fresh:
            return (
                f"SCENE: STALE detections (ts={obs.detections_ts:.1f}, "
                f"fresh={obs.fresh}); do NOT plan motion on this"
            )
        if self.scene_graph is not None:
            self.scene_graph.update(obs)
            return self.scene_graph.scene_brief()
        if not obs.detections:
            return "SCENE: empty (detections are event-driven; silence == empty)"
        lines = []
        for d in obs.detections:
            ref = f"[{d.ref}]" if d.ref else ""
            lines.append(
                f"- {d.label}{ref} at ({d.x:.2f}, {d.y:.2f}, {d.z:.2f}) "
                f"conf={d.conf:.2f}"
            )
        return "\n".join(lines)

    def _render_first_message(self) -> str:
        scene = self._scene_block()
        return f"{self._system_prompt}\n\nCURRENT SCENE:\n{scene}\n\n{self._task_text}"

    # --------------------------------------------- bootstrapping from config

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        tool_registry: Optional[Any] = None,
        backend: Any = None,
        *,
        scene_graph: Optional[SceneGraph] = None,
        prompt_path: Optional[str] = None,
        **overrides: Any,
    ) -> "G1Agent":
        """Build a G1Agent from a config_loader.load() dict — the ONE
        bootstrap the 'openbot-agent' CLI uses.

        Wiring (all from the parsed config dict):
          ``llm:``      segment -> ``llm.from_config`` (backend 'mock' ->
                        MockLLM on the laptop; 'server' ->
                        LlamaServerClient; 'real' -> LlamaCppClient)
          ``agent:``    segment -> max_tool_calls / failure_budget /
                        detection_freshness_s / llm_timeout_s /
                        max_llm_attempts / max_tool_result_chars
          ``embodiment:`` segment -> the validated EmbodimentConfig
                        (brain/config_loader.py) — the loader's OWN
                        schema, ``EmbodimentConfig.from_section`` — which
                        is INJECTED into the running hooks (safety
                        envelope) and the tool registry (pick placeholder
                        clamps/standoffs + place_at_spot key_positions).
                        An absent segment -> EmbodimentConfig defaults
                        (parity with brain/embodiment.py's defaults).
                        detection_freshness_s falls back to the
                        embodiment's perception.freshness_s when the
                        agent segment is silent about it (the config.yaml
                        parity convention: the embodiment value is the
                        fallback authority, the agent value wins when
                        present).
          ``gateway:``  segment -> the backend, via
                        ``_backend_from_gateway_segment`` (round-4):
                        transport 'mock' -> _LaptopMockBackend: on-the-fly
                        empty WorldObservation snapshots (silence ==
                        empty) and an honest error envelope for every
                        verb — the laptop can never claim a world action
                        succeeded; transport 'real' ->
                        GatewayCallerBackend(host, port) — the socket
                        client to the gateway node's tools server (the
                        production Stuart path, no longer a silent
                        default); any OTHER transport value raises an
                        honest ConfigError (never guess a transport).
                        A 'real' host that is a placeholder ('TODO-IP' /
                        'localhost-TODO' / '') raises ConfigError —
                        the D0 preflight must fill it in.

        Unknown/missing keys fall back to the G1Agent constructor defaults;
        an explicitly passed `backend` wins over the gateway segment;
        `tool_registry` is used as-is (and wired to `backend` and
        `embodiment` via attach() when given); an explicitly passed
        `embodiment` kwarg wins over the constructed one; extra kwargs
        (confirm/hooks/...) pass through to __init__.
        """
        seg = cfg.get("agent") or {}
        if not isinstance(seg, dict):
            seg = {}
        # loader's schema
        # same dataclass for ConfigDict
        # or hand-built dict
        # absent -> defaults
        embodiment = EmbodimentConfig.from_section(cfg.get("embodiment"))
        if "detection_freshness_s" in seg:
            detection_freshness_s = float(seg["detection_freshness_s"])
        else:
            detection_freshness_s = float(embodiment.perception.freshness_s)
        llm_client = llm_from_config(cfg.get("llm") or {})
        if backend is None:
            backend = _backend_from_gateway_segment(cfg.get("gateway"))
        if tool_registry is None:
            registry = ToolRegistry(backend=backend, embodiment=embodiment,
                                    switches=cfg.get("optional_tools"))
        else:
            registry = tool_registry
            if backend is not None:
                registry.attach(backend=backend)
            registry.attach(embodiment=embodiment)
        kwargs = dict(overrides)
        kwargs.setdefault("embodiment", embodiment)
        return cls(
            llm=llm_client,
            tool_registry=registry,
            backend=backend,
            scene_graph=scene_graph,
            max_tool_calls=int(seg.get("max_tool_calls", 25)),
            failure_budget=int(seg.get("failure_budget", 3)),
            detection_freshness_s=detection_freshness_s,
            llm_timeout_s=float(seg.get("llm_timeout_s", 12.0)),
            max_llm_attempts=int(seg.get("max_llm_attempts", 2)),
            max_tool_result_chars=int(
                seg.get("max_tool_result_chars", MAX_TOOL_RESULT_CHARS)
            ),
            prompt_path=prompt_path,
            **kwargs,
        )


class _LaptopMockBackend:
    """Laptop-dev backend (the ``G1Agent.from_config`` default): on-the-fly
    empty WorldObservation snapshots and honest refusal for every verb.

    The agent loop reads the scene from ``observe()`` — detections are
    event-driven, silence == empty, so an empty fresh observation is the
    correct laptop scene (the SCENE block reads 'empty', never 'not
    available').  Every motion/user/stop verb refuses with an honest error
    envelope: the laptop has no robot, and nothing here may pretend a
    world action succeeded.
    """

    def observe(self) -> WorldObservation:
        return default_world_observation(fresh=True, detections_ts=time.time())

    @staticmethod
    def _refuse(name: str) -> dict:
        return fail(f"laptop mock backend — {name} refused (no robot connected)")

    def pick(self, args: dict) -> dict:
        return self._refuse("pick")

    def place(self, args: dict) -> dict:
        return self._refuse("place")

    def navigate(self, args: dict) -> dict:
        return self._refuse("navigate")

    def open_hand(self, args: dict) -> dict:
        return self._refuse("open_hand")

    def close_hand(self, args: dict) -> dict:
        return self._refuse("close_hand")

    def query_user(self, args: dict) -> dict:
        return self._refuse("query_user")

    def notify_user(self, args: dict) -> dict:
        return self._refuse("notify_user")

    def stop(self, args: dict) -> dict:
        return self._refuse("stop")
