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


