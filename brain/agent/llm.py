"""LLM clients for the G1 brain agent loop.

Three clients behind one protocol (:class:`LLMClient`):

- :class:`LlamaCppClient` — primary; wraps an in-process ``llama_cpp.Llama``
  instance (Qwen3-4B GGUF on the laptop GPU / Stuart CPU).  ``llama_cpp``
  is imported LAZILY inside ``load()`` so importing this module (and the
  whole ``brain.agent`` package) works with zero third-party imports.
- :class:`LlamaServerClient` — alternative; talks to a llama.cpp / llama-
  server OpenAI-compatible sidecar over HTTP with stdlib ``urllib`` only.
- :class:`MockLLM` — scripted responses for tests and laptop dev.

Tool calling: native Qwen3 structured ``tool_calls`` through the chat
template (primary); the native path also parses inline ``<|tool_call|>``
segments and extracts a balanced JSON object from the reply text; the
schema-in-prompt JSON mode (``tool_mode="json"``) carries exactly one
auto-retry ("your previous output was not valid JSON").  The plain-text
skeleton mode (``tool_mode="skeleton"``, the Plan-B R0' default for new
usage) teaches a literal JSON skeleton with PLACEHOLDER values per tool
(never real example content — output-injection priming) and parses the
model's plain-text reply deterministically; the Qwen3 think-block finding
means detection runs on the RAW content before think stripping.  There are
no grammar constraints — validation lives in the agent loop.

``chat()`` never raises for recoverable model-output problems (malformed
tool-call JSON etc.): those become an empty :class:`ChatResult`, which the
agent's repair path handles.  Transport/model failures may raise; the agent
wraps them (its per-call timeout turns them into the same repair path).

Single-flight (shared by all three clients): llama.cpp is not thread-safe
for concurrent decodes and a sidecar must not be hammered, so a ``chat()``
call that arrives while one is in flight is DROPPED — it returns an empty
:class:`ChatResult` (agent repair path) and increments ``dropped_calls``.
The agent's per-call timeout can abandon a worker mid-decode; the drop
prevents that worker from racing the next call (the crash fix).

Stuart platform facts baked into the defaults: laptop n_ctx 4096 / GPU
offload; the robot profile (n_ctx 4096, n_gpu_layers 0, longer timeout) is
set by config.stuart.yaml + the loader clamp (2048 -> 4096 2026-08-29, the
D4 prompt-overflow fix), not hard-coded here.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# call-id minting
# ---------------------------------------------------------------------------

_MODULE_CALL_ID_SEQ = itertools.count(1)


def _module_call_id(prefix: str) -> str:
    """One process-unique tool-call id (stateless-parser fallback).

    Root cause of the C1 400 (repo tasks #145/#149): a compile-time
    constant id (``skeleton_1``) was stamped on EVERY parsed skeleton
    call, so a conversation's SECOND DISTINCT call collided with the
    first.  The session's ``_emitted_outputs`` dedup map is never
    cleared and the transcript history is only length-trimmed — id
    reuse can never be made safe by a mid-conversation reset, so any
    minted id must stay unique for the whole process lifetime.  Clients
    inject a per-client :class:`_CallIdFactory` (fresh client == fresh
    sequence — client construction, i.e. process start in the shipped
    entrypoints, is the only sound reset boundary); this module counter
    is the safe default for stateless parser call sites.
    """
    return f"{prefix}_{next(_MODULE_CALL_ID_SEQ)}"


