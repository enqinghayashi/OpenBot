"""
brain/context_assembler.py — ContextAssembler: prompt assembly with the
three lifetime sections, progressive tool disclosure, and n_ctx-aware
truncation that always leaves a note when it trims.

Stdlib only.  Replaces the old _build_system_prompt + scene injection seam:

    resident    = system prompt + resident extras (memory/embodiment dumps)
                  + tool defs (name + one-liner; FULL BODY ONLY ON DEMAND
                  via disclose(), after which the body replaces the
                  one-liner for the rest of the session)
    refreshed   = scene brief (scene_graph.brief(), identity-stable) +
                  world observation one-liner (dropped first under pressure)
    accumulated = history (oldest dropped first) + task notes + the
                  truncation note (always LAST so the LLM sees what it
                  lost)

Budget model (documented heuristic, not a tokenizer): tokens ~ chars/4
(default chars_per_token=4.0).  Budget = n_ctx - max_output_tokens; the
agent wires n_ctx/max_tokens from config llm.* (4096/256 laptop,
2048/256 stuart-profile).  Truncation priority under budget pressure:

    1. resident   — never trimmed, except disclosed full bodies are
                    dropped first; only in an extreme overflow is the
                    system prompt hard-truncated (noted).
    2. refreshed  — drop world one-liner, then scene brief.
    3. accumulated— drop history turns oldest-first, then task-note lines
                    oldest-first.

`build()` is a pure function of its inputs plus the assembler's resident
state; `disclose()` mutates resident state (that is its purpose).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from .context import Context, ContextSection, ToolSummary
from .observation import WorldObservation

DEFAULT_SYSTEM_PROMPT = (
    "You are the OpenBot embodied agent on a Unitree G1. "
    "You act through the tool verbs below. "
    "Motion only after confirmation. "
    "Never attempt a floor/ground pick: the pick engine is tabletop-only. "
    "Navigation truth is /walk_status == REACHED; never trust odom distance. "
    "Silence from detections means an empty scene."
)


class ContextAssembler:
    """Builds resident/refreshed/accumulated Context with budget control."""

    def __init__(
        self,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Optional[List[ToolSummary]] = None,
        n_ctx: int = 4096,
        max_output_tokens: int = 256,
        chars_per_token: float = 4.0,
        resident_extras: Optional[List[str]] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools: Dict[str, ToolSummary] = {}
        for t in tools or []:
            self.tools[t.name] = t
        self.disclosed: set = set()  # tools whose body we've switched on
        self.n_ctx = int(n_ctx)
        self.max_output_tokens = int(max_output_tokens)
        self.chars_per_token = float(chars_per_token)
        self.resident_extras: List[str] = list(resident_extras or [])
        self.last_truncation_note: str = ""
        self.last_report: Dict[str, object] = {}

    # --- poking at the resident state ---------------------------------------

    def add_tool(self, tool: ToolSummary) -> None:
        self.tools[tool.name] = tool

