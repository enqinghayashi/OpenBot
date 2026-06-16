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


def flatten_specs(specs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten ``brain/agent/tools.py`` specs into the Realtime wire shape.

    Input (nested, as produced by ToolRegistry.get_specs)::

        [{"type": "function", "function": {"name": n, "description": d,
                                           "parameters": p}}, ...]

    Output (flat, order preserved)::

        [{"type": "function", "name": n, "description": d, "parameters": p}, ...]

    ``None``/empty input yields ``[]``.  Any item that is not the nested
    function shape raises ``ValueError`` (callers map that to a loud
    fallback).
    """
    if not specs:
        return []
    if not isinstance(specs, list):
        raise ValueError("tool specs must be a list, got {!r}".format(type(specs).__name__))

    flat = []  # type: List[Dict[str, Any]]
    for index, item in enumerate(specs):
        if not isinstance(item, dict):
            raise ValueError("tool spec {} is not a dict: {!r}".format(index, item))
        if item.get("type") != "function":
            raise ValueError(
                "tool spec {} has type {!r}, expected 'function'".format(
                    index, item.get("type")
                )
            )
        function = item.get("function")
        if not isinstance(function, dict):
            raise ValueError("tool spec {} has no 'function' object".format(index))
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "tool spec {} has no usable 'name' (got {!r})".format(index, name)
            )
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError(
                "tool spec {} 'parameters' must be a JSON-schema dict, got "
                "{!r}".format(index, type(parameters).__name__)
            )
        description = function.get("description") or ""
        flat.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters if parameters is not None else {},
            }
        )
    return flat
