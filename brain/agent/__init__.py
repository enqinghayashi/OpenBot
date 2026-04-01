"""brain.agent package — the G1 agentic layer: fast path, LLM clients,
tool layer and the tool-calling loop.

Owns:
    fastpath   — deterministic regex classify_intent (openbot-thesis family),
                 maps a matched intent to a ToolCall without touching the LLM.
    llm        — LLMClient protocol + LlamaCppClient (lazy llama_cpp),
                 LlamaServerClient (OpenAI-compatible sidecar, stdlib urllib)
                 and MockLLM (scripted, for tests/laptop dev).
    tools      — Tool / ToolRegistry (pick, place, navigate, open_hand,
                 close_hand, resolve_ref, scene_brief, query_user,
                 notify_user) with the result-envelope dict contract.
    agent      — G1Agent loop: fastpath -> else LLM -> tool call ->
                 PRE-hooks (brain.hooks confirm gate + safety checks) -> run
                 -> POST-hook (brain.verifier, conservative) -> the VERDICT
                 line is fed back as the tool result; failure budget
                 halt-on-exhaustion; per-turn scene-brief injection.

Exports (binding list):
    G1Agent, Tool, ToolRegistry, MOTION_TOOLS (brain-level motion set),
    LLMClient, ChatResult, ToolCall, LlamaCppClient, LlamaServerClient,
    MockLLM, classify_intent, classify_tool_call, the envelope helpers
    ``ok`` / ``fail`` / ``envelope_to_str`` and the tool-layer exceptions.

Invariant: importing this package pulls zero third-party modules —
``llama_cpp`` is imported lazily inside ``LlamaCppClient.load()``, and the
brain-sibling modules (brain.hooks / brain.verifier / brain.scene_graph /
brain.tool_contract / brain.observation, owned by worker-core) are imported
eagerly by brain.agent.agent so a shared-contract drift is a compile-time
error, never a silent behavior change.
"""

from .agent import G1Agent
from .fastpath import classify_intent, classify_tool_call
from .llm import (
    ChatResult,
    ChatTransportError,
    LLMClient,
    LlamaCppClient,
    LlamaServerClient,
    MockLLM,
    ModelLoadError,
    ToolCall,
    ToolCallParseError,
)
from .tools import (
    MOTION_TOOLS,
    Tool,
    ToolNameConflict,
    ToolNameInvalid,
    ToolRegistry,
    UnknownTool,
    envelope_to_str,
    fail,
    ok,
)

__all__ = [
    "G1Agent",
    "Tool",
    "ToolRegistry",
    "MOTION_TOOLS",
    "LLMClient",
    "ChatResult",
    "ToolCall",
    "LlamaCppClient",
    "LlamaServerClient",
    "MockLLM",
    "ChatTransportError",
    "ToolCallParseError",
    "ModelLoadError",
    "classify_intent",
    "classify_tool_call",
    "ok",
    "fail",
    "envelope_to_str",
    "ToolNameConflict",
    "ToolNameInvalid",
    "UnknownTool",
]
