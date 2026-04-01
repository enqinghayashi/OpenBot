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


