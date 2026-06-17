"""Adapters between the brain's tool shapes and the Realtime wire shape.

The OpenAI Realtime GA session advertises FLAT function entries at
``session.tools`` - ``[{"type": "function", "name": ..., ...}]`` - while
``brain/agent/tools.py`` produces the nested Chat-Completions shape
``[{"type": "function", "function": {"name", "description", "parameters"}}]``
(OpenBot protocol reference, chat/cloud/events.py build_session_update:
"GA shape: flat function entries ... at session.tools, not the Chat
Completions nested {"function": {}}").  :func:`flatten_specs` is the seam
between the two, preserving registration order.

Pure stdlib; malformed input raises ``ValueError`` so callers can map it
to a loud fallback (the session then runs with no tools rather than with a
silently broken tool list).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


