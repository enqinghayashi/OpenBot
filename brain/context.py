"""
brain/context.py — the assembled-context data model (context lifetimes).

Stdlib only.  The three lifetime sections (Thea SS3.2 framing, see
docs/design_rationale.md):

    resident    — system prompt + tool definitions; constant across turns.
    refreshed   — scene brief + world observation; rebuilt every turn.
    accumulated — history + task notes; grows within a task.

The assembly logic (budgeting, progressive tool disclosure, truncation)
lives in brain/context_assembler.py — `ContextAssembler` is re-exported
here lazily so both import paths work:

    from brain.context import ContextAssembler
    from brain.context_assembler import ContextAssembler
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .observation import WorldObservation


@dataclass
class ContextSection:
    """One lifetime bucket of prompt text."""

    name: str  # 'resident' | 'refreshed' | 'accumulated'
    lifetime: str  # 'system' | 'per-turn' | 'task'
    content: List[str] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.content.append(text)

    def render(self) -> str:
        return "\n".join(self.content)

    @property
    def chars(self) -> int:
        return sum(len(t) for t in self.content)

    def __len__(self) -> int:
        return self.chars


@dataclass
class ToolSummary:
    """Progressive tool disclosure entry.

    Resident carries name + one-liner for every tool; `full_body` is
    revealed only when the agent calls disclose(name) on the assembler.
    """

    name: str
    one_liner: str
    full_body: str = ""

    def resident_line(self) -> str:
        return f"- {self.name}: {self.one_liner}"

    def full_line(self) -> str:
        return f"- {self.name}: {self.full_body}"


@dataclass
class PendingAction:
    """Structured description of the motion the agent is about to request.

    Set `context.pending_action` before run_pre() so the confirm gate and
    safety checks can inspect tool + args without parsing free text.
    """

    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    description: str = ""  # what the human reads in the confirm prompt

