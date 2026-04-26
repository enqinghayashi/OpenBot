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

    def disclose(self, name: str) -> str:
        """Reveal a tool's full body (progressive disclosure).

        Returns the body text.  Once disclosed, the body replaces the
        one-liner in the resident section for the rest of the session.
        Unknown tool -> ValueError.
        """
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"no such tool: {name!r}")
        self.disclosed.add(name)
        return tool.full_body

    # --- sorting out the budget -----------------------------------------------

    def budget_tokens(self, text: str) -> int:
        return max(1, int(math.ceil(len(text) / self.chars_per_token)))

    @property
    def budget_chars(self) -> int:
        """Character budget for the whole prompt (output reserved)."""
        tokens = self.n_ctx - self.max_output_tokens
        if tokens <= 0:
            return 0
        return int(tokens * self.chars_per_token)

    # --- putting the prompt together --------------------------------------------

    def build(
        self,
        *,
        scene_brief: str = "",
        world: Optional[WorldObservation] = None,
        history: Optional[List[str]] = None,
        task_notes: Optional[object] = None,
        scene_brief_fn: Optional[Callable[[], str]] = None,
    ) -> Context:
        """Assemble the Context for one turn.

        scene_brief: pre-rendered scene-graph brief string, OR
        scene_brief_fn: zero-arg callable producing it (mock-friendly).
        history: prior conversation turns, oldest first.
        task_notes: TaskNotes-like with .render()/.notes (brain.memory).
        """
        if scene_brief_fn is not None and not scene_brief:
            scene_brief = scene_brief_fn()

        ctx = Context.empty()
        budget = self.budget_chars
        notes: List[str] = []
        dropped = {"disclosed_bodies": 0, "history": 0, "task_notes": 0,
                   "world_desc": 0, "scene_brief": 0, "system_chars": 0}

        # --- resident (never cut, apart from disclosed bodies) ----------------
        resident_lines: List[str] = [self.system_prompt]
        resident_lines.extend(self.resident_extras)
        for name in self.tools:
            t = self.tools[name]
            resident_lines.append(t.full_line() if name in self.disclosed else t.resident_line())
        resident_text = "\n".join(resident_lines)

        if len(resident_text) > budget and self.disclosed:
            # shed disclosed bodies first - on-demand anyway
            kept = [self.system_prompt] + list(self.resident_extras)
            for name in self.tools:
                t = self.tools[name]
                kept.append(t.resident_line())  # one-liner survives
            dropped["disclosed_bodies"] = len(self.disclosed)
            resident_text = "\n".join(kept)
        if len(resident_text) > budget:
            dropped["system_chars"] = len(resident_text) - budget
            resident_text = resident_text[:budget]
        ctx.resident.add(resident_text)
        remaining = budget - len(resident_text)

        # --- refreshed (lose world detail first, then the scene brief) ---------
        refreshed_lines: List[str] = []
        if scene_brief:
            refreshed_lines.append("[scene] " + scene_brief)
        if world is not None:
            refreshed_lines.append("[world] " + world.describe())
        for line in refreshed_lines:
            if remaining <= 0 or len(line) > remaining:
                if line.startswith("[scene] "):
                    dropped["scene_brief"] += len(line)
                else:
                    dropped["world_desc"] += len(line)
                continue
            ctx.refreshed.add(line)
            remaining -= len(line)

        # --- accumulated (oldest goes first, note always last) -----------------
        hist = list(history or [])
        note_lines: List[str] = []
        if task_notes is not None:
            note_lines = list(getattr(task_notes, "notes", None) or
                              getattr(task_notes, "render", lambda: "")().split("\n") or [])
            note_lines = [n for n in note_lines if n]

        # history, oldest first
        for turn in hist:
            if remaining <= 0 or len(turn) > remaining:
                dropped["history"] += 1
                continue
            ctx.accumulated.add("[h] " + turn)
            remaining -= len(turn) + 4
        # task notes, oldest first
        for n in note_lines:
            if remaining <= 0 or len(n) > remaining:
                dropped["task_notes"] += 1
                continue
            ctx.accumulated.add("[note] " + n)
            remaining -= len(n) + 7

        # note goes in whenever anything dropped
        if sum(dropped.values()) > 0:
            note = (
                f"[context trimmed: budget {budget // max(1, int(self.chars_per_token))} tok; "
                f"dropped {dropped['disclosed_bodies']} tool body/bodies, "
                f"{dropped['history']} history turn(s), {dropped['task_notes']} task note(s), "
                f"{dropped['world_desc']} world-detail chars, "
                f"{dropped['scene_brief']} scene-brief chars, "
                f"{dropped['system_chars']} system chars]"
            )
            self.last_truncation_note = note
            if remaining > 0 and len(note) <= remaining:
                ctx.accumulated.add(note)
            else:
                # note takes over accumulated - we never trim silently
                ctx.accumulated.content = [note]
            remaining -= len(note)
        else:
            self.last_truncation_note = ""

        ctx.world = world
        self.last_report = {
            "budget_chars": budget,
            "resident_chars": ctx.resident.chars,
            "refreshed_chars": ctx.refreshed.chars,
            "accumulated_chars": ctx.accumulated.chars,
            "dropped": dropped,
        }
        return ctx
