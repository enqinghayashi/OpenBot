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

