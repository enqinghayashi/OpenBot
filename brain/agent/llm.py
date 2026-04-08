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


class _CallIdFactory:
    """Per-client monotonic tool-call-id mint (the C1 fix).

    ``__call__(prefix)`` returns ``prefix_N`` where N never repeats for
    this client's lifetime, so every tool call a client parses in one
    conversation carries a distinct id and the outgoing history always
    has exactly one tool message per assistant tool_call_id.
    """

    def __init__(self) -> None:
        self._seq = itertools.count(1)

    def __call__(self, prefix: str) -> str:
        return f"{prefix}_{next(self._seq)}"

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool call proposed by the LLM (or the fast path)."""

    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    """One LLM response: either plain text, a list of tool calls, or both.

    A fully malformed response has both empty/None — the agent treats that
    as a parse failure (repair path).  ``dropped`` marks a single-flight
    drop (the input was discarded while another call was in flight).
    """

    text: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    dropped: bool = False
    #: truncated
    #: finish_reason 'length' / 'max_tokens'
    #: True = budget ran out
    #: False = check the model class
    truncated: bool = False


@runtime_checkable
class LLMClient(Protocol):
    """Protocol the G1Agent loop calls.

    ``messages`` are OpenAI-style dicts: system/user/assistant/tool roles,
    assistant messages may carry ``tool_calls``, tool messages carry
    ``tool_call_id`` (the Qwen3 chat template renders both).  ``tools`` is
    ``ToolRegistry.get_specs()`` output.
    """

    def chat(self, messages: list, tools: list) -> ChatResult:
        """Run one chat completion.  Never raises for recoverable output
        problems; transport failures may raise (the agent wraps them)."""
        ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolCallParseError(ValueError):
    """A tool-call payload could not be parsed (bad arguments JSON etc.).

    Raised internally by the parsers; the clients catch it and return an
    empty ChatResult (the agent's repair path), incrementing
    ``parse_failures``.
    """


class ModelLoadError(RuntimeError):
    """The model file or the ``llama_cpp`` runtime could not be set up."""


class ChatTransportError(RuntimeError):
    """Transport-level failure (HTTP errors, backend kwargs rejected, ...)."""


# ---------------------------------------------------------------------------
# dealing with think blocks (Qwen3 emits <|thinking|> ... <|/thinking|>)
# ---------------------------------------------------------------------------

_THINK_BLOCK_RES = [
    re.compile(r"<\|thinking\|>.*?(?:\|/thinking\|>|$)", re.DOTALL),
    re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL),
]

# no think block
# Qwen3's template turns thinking on by default; this line shuts it off
# a think block eats the CPU generation budget and breaks parsing
_NO_THINK_DIRECTIVE = (
    "\n\nIMPORTANT: You are in direct-answer mode. Do NOT emit a <think> "
    "block — answer immediately in plain text."
)

_INLINE_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>(.*?)<\|/tool_call\|>", re.DOTALL
)

# skeleton retry
# one auto-retry, then the empty-ChatResult repair path
_SKELETON_RETRY_MESSAGE = (
    'Your previous output was not a valid tool call. Respond with exactly '
    'one JSON object of the form {"tool": "<tool name>", "args": {<arguments>}} '
    "and nothing else."
)


def _skeleton_placeholder(name: str, ptype: Any) -> Any:
    """A literal placeholder value for one parameter (never real content —
    output-injection priming, the Escapement finding)."""
    if ptype in ("number", "integer"):
        return "<number>"
    if ptype == "boolean":
        return "<true|false>"
    return f"<{name}>"  # use the param name


def _build_skeleton_prompt(tools: List[dict]) -> str:
    """Serialize the tool specs into a literal-JSON skeleton instruction.

    One skeleton per tool with PLACEHOLDER values only (``<label>``,
    ``<number>``, ...) — never real example content.  The model answers in
    plain text; the deterministic parser (``_parse_skeleton_tool_call``)
    validates the tool name against the provided specs and the args shape
    code-side.  No grammar constraints anywhere (Plan-B R0': the hard-schema
    json-mode Constraint Tax).  Empty tool list -> empty string (the caller
    falls back to the native text path).
    """
    lines: List[str] = []
    for spec in tools:
        fn = spec.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters") if isinstance(fn, dict) else None
        properties = params.get("properties") if isinstance(params, dict) else None
        args: Dict[str, Any] = {}
        if isinstance(properties, dict):
            for pname, pmeta in properties.items():
                ptype = pmeta.get("type") if isinstance(pmeta, dict) else None
                args[pname] = _skeleton_placeholder(pname, ptype)
        lines.append(json.dumps({"tool": name, "args": args}, sort_keys=True))
    if not lines:
        return ""
    header = (
        "Choose exactly ONE tool for this turn. Respond with exactly one "
        'JSON object of the form {"tool": "<tool name>", "args": {<arguments>}} '
        "and NOTHING ELSE — no explanations, no markdown, no text before or "
        "after. The skeletons below show the SHAPE of the arguments only; "
        "fill each placeholder with the real value from CURRENT SCENE. "
        "Tool skeletons:"
    )
    return header + "\n" + "\n".join(f"- {s}" for s in lines)


def _parse_skeleton_tool_call(
    raw: str,
    valid_names: set,
    next_call_id: Optional[Callable[[str], str]] = None,
) -> Optional[ToolCall]:
    """Deterministically extract one skeleton-mode tool call from plain text.

    Detection runs on the RAW content BEFORE think stripping (the Qwen3
    think-block finding: tool calls can be emitted INSIDE a think block,
    and stripping first would destroy them); when the raw pass yields
    nothing, the think-stripped text is tried so reasoning braces cannot
    hijack the balanced-JSON scan.  Validates the tool name against the
    provided specs and the args shape (a dict).  Returns None on any
    failure — the caller's exactly-one-retry then the empty-ChatResult
    repair path takes over.

    ``next_call_id`` (optional) is the caller's id mint — a per-client
    :class:`_CallIdFactory` or the :func:`_module_call_id` default.  The
    parsed call's id is minted HERE, at parse time, never a compile-time
    constant: the C1 400 root cause was every parsed call being stamped
    ``skeleton_1``, which collided the moment a conversation carried two
    distinct tool calls.
    """
    mint = next_call_id if next_call_id is not None else _module_call_id
    for text in (raw, _strip_think(raw)):
        if not text:
            continue
        maybe = _extract_first_balanced_json(text)
        if maybe is None:
            continue
        try:
            data = json.loads(maybe)
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("tool")
        args = data.get("args")
        if isinstance(name, str) and name in valid_names and isinstance(args, dict):
            return ToolCall(id=mint("skeleton"), name=name, arguments=args)
    return None


def _find_think_blocks(text: str) -> List[str]:
    """Return every think-block match found in ``text`` (empty if none)."""
    if not text:
        return []
    found: List[str] = []
    for pattern in _THINK_BLOCK_RES:
        found.extend(pattern.findall(text))
    return found


def _strip_think(text: str) -> str:
    """Remove think blocks from ``text``."""
    if not text:
        return text
    for pattern in _THINK_BLOCK_RES:
        text = pattern.sub("", text)
    return text.strip()


def _extract_first_balanced_json(text: str) -> Optional[str]:
    """Extract the first balanced ``{...}`` region from ``text``.

    Returns the exact substring (including the outer braces) or None.
    Braces inside string literals and escapes are respected; an unbalanced
    region (truncated tail) returns None.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _parse_tool_calls_payload(
    payload: Any,
    next_call_id: Optional[Callable[[str], str]] = None,
) -> List[ToolCall]:
    """Normalize an OpenAI-style ``message["tool_calls"]`` list to ToolCall.

    Accepts both the OpenAI shape (``{"id", "type", "function": {name,
    arguments}}``) and the flatter llama.cpp shape (``{"id", "name",
    "arguments"}``).  ``arguments`` may be a JSON string or a parsed dict.
    Raises :class:`ToolCallParseError` on malformed payloads.  Provider-
    supplied ids are kept verbatim; MISSING ids are synthesized through
    the caller's mint (``next_call_id``, default :func:`_module_call_id`)
    as process-unique ``call_<n>`` ids — never a per-payload restarting
    counter, which would re-collide across id-less payloads in one
    conversation (the C1 defect family).
    """
    mint = next_call_id if next_call_id is not None else _module_call_id
    if not isinstance(payload, list):
        raise ToolCallParseError(f"tool_calls is not a list: {type(payload).__name__}")
    calls: List[ToolCall] = []
    for n, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ToolCallParseError(
                f"tool_calls[{n}] is not an object: {type(item).__name__}"
            )
        name = item.get("name")
        if name is None:
            function = item.get("function")
            if isinstance(function, dict):
                name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ToolCallParseError(f"tool_calls[{n}] missing a function name")
        raw_args = item.get("arguments")
        if raw_args is None and isinstance(item.get("function"), dict):
            raw_args = item["function"].get("arguments")
        if isinstance(raw_args, str):
            try:
                arguments = json.loads(raw_args)
            except ValueError as exc:
                raise ToolCallParseError(
                    f"tool_calls[{n}] {name!r} arguments are not valid JSON: {exc}"
                ) from exc
        elif isinstance(raw_args, dict):
            arguments = raw_args
        else:
            raise ToolCallParseError(
                f"tool_calls[{n}] {name!r} arguments missing or malformed"
            )
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = mint("call")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def _parse_inline_tool_calls(
    content: str,
    next_call_id: Optional[Callable[[str], str]] = None,
) -> tuple[str, Optional[List[ToolCall]]]:
    """Extract raw ``<|tool_call|>`` segments from text (non-native builds).

    Returns ``(content_without_segments, calls_or_None)``.  Raises
    :class:`ToolCallParseError` if a segment contains invalid JSON.  Ids
    are minted through ``next_call_id`` (default :func:`_module_call_id`)
    — never a per-parse restarting ``inline_<n>`` counter, which would
    collide across two inline-only turns of one conversation (the C1
    defect family).
    """
    if not content or "<|tool_call|>" not in content:
        return content, None
    mint = next_call_id if next_call_id is not None else _module_call_id
    calls: List[ToolCall] = []
    remaining: List[str] = []
    cursor = 0
    for match in _INLINE_TOOL_CALL_RE.finditer(content):
        remaining.append(content[cursor: match.start()])
        segment = match.group(1).strip()
        try:
            data = json.loads(segment)
        except ValueError as exc:
            raise ToolCallParseError(
                f"inline tool_call segment is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise ToolCallParseError("inline tool_call segment is not an object")
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise ToolCallParseError(
                "inline tool_call segment needs 'name' (str) and 'arguments' (dict)"
            )
        calls.append(ToolCall(id=mint("inline"), name=name, arguments=arguments))
        cursor = match.end()
    remaining.append(content[cursor:])
    return "".join(remaining).strip(), calls or None


def _parse_openai_chat_response(
    body: dict,
    next_call_id: Optional[Callable[[str], str]] = None,
) -> ChatResult:
    """Parse a chat-completion response body into a :class:`ChatResult`.

    Handles the OpenAI shape ``{"choices": [{"message": {...}}]}``.  Strips
    think blocks from any text.  Raises :class:`ToolCallParseError` for
    malformed tool-call payloads (callers convert that to an empty result).
    ``next_call_id`` passes the caller's id mint through to payload
    parsing (synthesized ids only; provider ids are kept verbatim).
    """
    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolCallParseError(f"response body has no choices[0].message: {exc}") from exc
    text: Optional[str] = None
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        text = _strip_think(raw_content)
    calls: Optional[List[ToolCall]] = None
    if message.get("tool_calls"):
        calls = _parse_tool_calls_payload(message["tool_calls"], next_call_id)
    return ChatResult(text=text, tool_calls=calls)


# ---------------------------------------------------------------------------
# single flight
# ---------------------------------------------------------------------------


class _SingleFlight:
    """Non-blocking mutual exclusion: one chat in flight per client.

    A ``chat()`` while one is decoding is DROPPED (empty ChatResult) and
    counted in ``dropped_calls`` — llama.cpp is not thread-safe for
    concurrent decodes and the agent's per-call timeout can abandon a
    worker mid-decode, so the next call must never overlap it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.dropped_calls: int = 0

    def acquire(self) -> bool:
        if not self._lock.acquire(blocking=False):
            self.dropped_calls += 1
            return False
        return True

    def release(self) -> None:
        self._lock.release()


# ---------------------------------------------------------------------------
# LlamaCppClient
# ---------------------------------------------------------------------------


class LlamaCppClient:
    """In-process llama.cpp client (laptop GPU / Stuart CPU).

    Parameters:
    - ``model_path``: explicit GGUF path (required for the real backend).
    - ``n_ctx``: context window (4096 laptop; Stuart 4096 via loader clamp —
      2048 -> 4096 2026-08-29, D4 prompt-overflow fix).
    - ``n_gpu_layers``: ``"all"`` (CUDA) or an int (0 on Stuart).
    - ``temperature`` / ``max_tokens``: sampling.
    - ``tool_mode``: ``"auto"`` (probe at load), ``"native"`` (structured
      tool_calls), ``"json"`` (schema-in-prompt fallback), ``"skeleton"``
      (literal-JSON-skeleton prompt + deterministic parse — the Plan-B R0'
      default for new usage: no grammar constraints, 2-4x faster decode).
    - ``chat_format``: ``"auto"`` (let the GGUF's embedded template win) or
      an explicit value (e.g. ``"chatml"`` on builds without a qwen3
      formatter).
    - ``timeout_s``: informational per-call budget (the agent enforces it).
    - ``_backend``: test seam — an object with ``create_chat_completion``
      replacing the real ``llama_cpp.Llama`` (skips import/load).

    ``chat()`` never raises for recoverable output problems; malformed
    tool_calls yield an empty ChatResult and increment ``parse_failures``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int | str = "all",
        temperature: float = 0.2,
        max_tokens: int = 256,
        tool_mode: str = "auto",
        chat_format: str = "auto",
        enable_thinking: bool = False,
        timeout_s: float = 12.0,
        _backend: Any = None,
    ) -> None:
        if tool_mode not in ("auto", "native", "json", "skeleton"):
            raise ValueError(
                f"tool_mode {tool_mode!r} must be 'auto', 'native', 'json' "
                "or 'skeleton'"
            )
        self.model_path = model_path
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = n_gpu_layers
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.tool_mode = tool_mode
        self.chat_format = chat_format
        self.enable_thinking = bool(enable_thinking)
        self.timeout_s = float(timeout_s)
        self._backend = _backend  # test hook
        self._backend_injected = _backend is not None

        self._resolved_chat_format: Optional[str] = None
        self._loaded = False
        self.tool_mode_actual: Optional[str] = None  # probe writes this
        self._flight = _SingleFlight()
        #: per-client ids
        #: never repeat here
        #: a fresh client restarts
        self._call_ids = _CallIdFactory()
        self.think_blocks_seen: int = 0
        self.parse_failures: int = 0
        self.chat_calls: int = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def dropped_calls(self) -> int:
        return self._flight.dropped_calls

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> None:
        """Construct the backend; idempotent.  Lazy-imports ``llama_cpp``.

        Raises :class:`ModelLoadError` with an actionable message (never a
        bare traceback).
        """
        if self._loaded:
            return
        if self._backend is None:
            if not self.model_path:
                raise ModelLoadError(
                    "no model_path given and no backend injected; pass an "
                    "explicit GGUF path (config llm.model_path)"
                )
            self._backend = self._build_llama(self.model_path)
        if self._backend_injected:
            self._resolved_chat_format = (
                None if self.chat_format == "auto" else self.chat_format
            )
        if self.tool_mode == "auto":
            self.tool_mode_actual = (
                "native" if self._backend_injected else self._probe_tool_support()
            )
        else:
            self.tool_mode_actual = self.tool_mode
        logger.info(
            "LLM loaded (n_ctx=%d, gpu_layers=%s, tool_mode=%s)",
            self.n_ctx, self.n_gpu_layers, self.tool_mode_actual,
        )
        self._loaded = True

    def close(self) -> None:
        """Release the model (frees VRAM); idempotent."""
        self._backend = None
        self._loaded = False

    def _build_llama(self, model_path: str) -> Any:
        try:
            import llama_cpp
        except ImportError as exc:
            raise ModelLoadError(
                "llama-cpp-python is not installed. Install the build that "
                "matches your device (CUDA on the laptop, CPU wheel on "
                "Stuart): e.g. CMAKE_ARGS=\"-DGGML_CUDA=on "
                "-DCMAKE_CUDA_ARCHITECTURES=89\" pip install llama-cpp-python"
            ) from exc
        kwargs: Dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": -1 if self.n_gpu_layers == "all" else int(self.n_gpu_layers),
            # pin all the cores
            # unset n_threads took the default:
            # ~3.5-4 tok/s on 8 cores
            # llama.cpp clamps it safely
            # the box is ours during a decode
            "n_threads": max(1, (os.cpu_count() or 4)),
            "verbose": False,
        }
        if self.chat_format and self.chat_format != "auto":
            kwargs["chat_format"] = self.chat_format
        return llama_cpp.Llama(**kwargs)

    # -- LLMClient ------------------------------------------------------------

    def chat(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """One completion; never raises on bad output.  Single-flight drop."""
        if not self._flight.acquire():
            logger.warning("LLM call dropped: a previous call is still decoding")
            return ChatResult(dropped=True)
        try:
            self.load()
            self.chat_calls += 1
            if self.tool_mode_actual == "json":
                return self._chat_json(messages, tools)
            if self.tool_mode_actual == "skeleton":
                return self._chat_skeleton(messages, tools)
            return self._chat_native(messages, tools)
        finally:
            self._flight.release()

    # -- native tool-calling path ----------------------------------------------

    def _chat_native(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Structured tool calls via the Qwen3 chat template.

        Parse chain: structured ``tool_calls`` -> inline ``<|tool_call|>``
        segments -> balanced-JSON extraction -> plain text -> empty result
        (repair).  Malformed payloads become an empty result and increment
        ``parse_failures``.
        """
        body = self._backend_call(messages, tools)
        raw_content = self._raw_message_content(body)
        if raw_content:
            self.think_blocks_seen += len(_find_think_blocks(raw_content))
        try:
            result = _parse_openai_chat_response(body, self._call_ids)
        except ToolCallParseError as exc:
            logger.warning("tool-call parse failed: %s", exc)
            self.parse_failures += 1
            return ChatResult()
        if result.tool_calls:
            return result
        if raw_content:
            cleaned, inline_calls = _parse_inline_tool_calls(
                raw_content, self._call_ids
            )
            if inline_calls:
                return ChatResult(text=result.text, tool_calls=inline_calls)
            if result.text is None and cleaned:
                maybe = _extract_first_balanced_json(cleaned)
                if maybe:
                    try:
                        data = json.loads(maybe)
                    except ValueError as exc:
                        logger.warning("balanced-JSON extraction failed: %s", exc)
                        self.parse_failures += 1
                        return ChatResult()
                    if isinstance(data, dict) and isinstance(data.get("name"), str):
                        arguments = data.get("arguments")
                        if isinstance(arguments, dict):
                            return ChatResult(
                                tool_calls=[
                                    ToolCall(
                                        id=self._call_ids("extracted"),
                                        name=data["name"],
                                        arguments=arguments,
                                    )
                                ],
                            )
        return result  # empty -> repair

    def _backend_call(self, messages: List[dict], tools: List[dict]) -> dict:
        """Call the backend, tolerating kwargs/format drift across versions.

        Some llama-cpp-python builds reject ``enable_thinking`` (Qwen3
        template input) or unknown chat-format values; each failure degrades
        one notch: drop ``enable_thinking``, then the explicit
        ``chat_format``.  Transport errors propagate (the agent wraps).
        """
        translated = self._translate_messages(messages)
        base: Dict[str, Any] = {
            "messages": translated,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        attempts: List[Dict[str, Any]] = [
            {**base, "enable_thinking": self.enable_thinking},
            base,
        ]
        last_exc: Optional[Exception] = None
        for attempt in attempts:
            try:
                if self._resolved_chat_format:
                    body = self._backend.create_chat_completion(
                        chat_format=self._resolved_chat_format, **attempt
                    )
                else:
                    body = self._backend.create_chat_completion(**attempt)
                return body if isinstance(body, dict) else dict(body)
            except TypeError as exc:
                last_exc = exc
                if self._resolved_chat_format:
                    self._resolved_chat_format = None
                    attempts.append(attempt)  # no chat_format
                    continue
                continue
        raise ChatTransportError(f"backend rejected completion kwargs: {last_exc}")

    @staticmethod
    def _translate_messages(messages: List[dict]) -> List[dict]:
        """Deep-copy messages so the caller's list is never mutated."""
        return [dict(m) for m in messages]

    @staticmethod
    def _raw_message_content(body: dict) -> Optional[str]:
        try:
            content = body["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError):
            return None
        return content if isinstance(content, str) else None

    # -- schema-in-prompt JSON path (fallback) ----------------------------------

    def _chat_json(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Schema-in-prompt JSON mode: one auto-retry, then empty result."""
        instruction = self._build_json_prompt(tools)
        prompt_messages: List[dict] = []
        inserted = False
        for m in messages:
            prompt_messages.append(dict(m))
            if m.get("role") == "user" and not inserted:
                prompt_messages[-1]["content"] = (
                    f"{m.get('content') or ''}\n\n{instruction}"
                )
                inserted = True
        if not inserted:
            prompt_messages.append({"role": "user", "content": instruction})
        for attempt in (0, 1):
            body = self._backend_call(prompt_messages, [])
            raw = self._raw_message_content(body)
            if raw is None:
                self.parse_failures += 1
                return ChatResult()
            self.think_blocks_seen += len(_find_think_blocks(raw))
            cleaned = _strip_think(raw)
            maybe = _extract_first_balanced_json(cleaned)
            if maybe:
                try:
                    data = json.loads(maybe)
                except ValueError as exc:
                    logger.warning("json-mode parse failed: %s", exc)
                    data = None
                if isinstance(data, dict) and isinstance(data.get("name"), str):
                    arguments = data.get("arguments")
                    if isinstance(arguments, dict):
                        return ChatResult(
                            tool_calls=[
                                ToolCall(
                                    id=self._call_ids("json"),
                                    name=data["name"],
                                    arguments=arguments,
                                )
                            ],
                        )
            if attempt == 0:
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. Respond with a "
                            'single JSON object of the form {"name": "<tool>", '
                            '"arguments": {<params>}} and nothing else.'
                        ),
                    }
                )
        self.parse_failures += 1
        return ChatResult()

    # -- plain-text skeleton path (the default) ----------------------------------

    def _chat_skeleton(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Literal-JSON-skeleton mode: one auto-retry, then empty result.

        The prompt teaches one skeleton per tool (placeholder values only —
        never real content, the output-injection priming rule); the plain-
        text reply is parsed deterministically with think-block detection on
        the RAW content first (the Qwen3 finding), the tool name is
        validated against the provided specs, and the args must be a dict.
        Any failure -> exactly one retry (_SKELETON_RETRY_MESSAGE), then the
        empty ChatResult the agent's repair path handles (and
        ``parse_failures`` increments).  Empty tool list -> the native text
        path (nothing to call anyway).
        """
        valid_names = {
            spec["function"]["name"]
            for spec in tools
            if isinstance(spec.get("function"), dict)
            and isinstance(spec["function"].get("name"), str)
        }
        if not valid_names:
            return self._chat_native(messages, tools)
        instruction = _build_skeleton_prompt(tools)
        prompt_messages: List[dict] = []
        inserted = False
        for m in messages:
            prompt_messages.append(dict(m))
            if m.get("role") == "user" and not inserted:
                prompt_messages[-1]["content"] = (
                    f"{m.get('content') or ''}\n\n{instruction}"
                )
                inserted = True
        if not inserted:
            prompt_messages.append({"role": "user", "content": instruction})
        for attempt in (0, 1):
            body = self._backend_call(prompt_messages, [])
            raw = self._raw_message_content(body)
            if raw is None:
                self.parse_failures += 1
                return ChatResult()
            self.think_blocks_seen += len(_find_think_blocks(raw))
            call = _parse_skeleton_tool_call(raw, valid_names, self._call_ids)
            if call is not None:
                return ChatResult(tool_calls=[call])
            if attempt == 0:
                prompt_messages.append(
                    {"role": "user", "content": _SKELETON_RETRY_MESSAGE}
                )
        self.parse_failures += 1
        return ChatResult()

    @staticmethod
    def _build_json_prompt(tools: List[dict]) -> str:
        """Serialize the tool specs into prompt text for json mode."""
        lines = [
            "Available tools. You MUST respond with exactly one JSON object of "
            'the form {"name": "<tool name>", "arguments": {<params>}}. '
            "Do not include any other text."
        ]
        for spec in tools:
            fn = spec.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            lines.append(f"- {name}: {desc} params={json.dumps(params)}")
        return "\n".join(lines)

    # -- capability probe --------------------------------------------------------

    def _probe_tool_support(self) -> str:
        """Decide native vs json mode with one toy tool call."""
        toy_tool = [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "Probe tool. Returns OK.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
        messages = [{"role": "user", "content": "What tools do you have?"}]
        try:
            body = self._backend_call(messages, toy_tool)
            result = _parse_openai_chat_response(body)
            if result.tool_calls:
                return "native"
            raw = self._raw_message_content(body)
            if raw and _parse_inline_tool_calls(raw)[1]:
                return "native"
        except (ChatTransportError, ToolCallParseError) as exc:
            logger.warning("probe failed (%s); falling back to json mode", exc)
        return "json"

    # -- token estimation --------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Estimate ``text``'s token count via the real tokenizer when loaded.

        Falls back to the chars/4 heuristic (+4 overhead) before load —
        keeps the agent's compaction working even for a client that never
        loaded (mock LLMs, offline tests).
        """
        if not self.loaded:
            return max(1, len(text) // 4) + 4
        try:
            tokens = self._backend.tokenize(text.encode("utf-8"))
            return max(1, len(tokens)) + 4
        except Exception:  # noqa: BLE001 — tokenizer hiccups degrade to heuristic
            return max(1, len(text) // 4) + 4


# ---------------------------------------------------------------------------
# sidecar client
# ---------------------------------------------------------------------------


class LlamaServerClient:
    """Alternative client for a llama.cpp / llama-server OpenAI-compatible
    HTTP sidecar.  ``base_url`` defaults to ``http://127.0.0.1:8080/v1``
    (llama-server default).  Implemented with stdlib ``urllib.request`` —
    no extra dependency; the fallback path for Stuart if the CPU wheel
    cannot do in-process Qwen3 tool calls.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "qwen3-4b-instruct",
        temperature: float = 0.2,
        max_tokens: int = 256,
        enable_thinking: bool = False,
        tool_mode: str = "skeleton",
        timeout_s: float = 12.0,
        _transport: Optional[Callable] = None,
    ) -> None:
        if tool_mode not in ("native", "json", "skeleton"):
            raise ValueError(
                f"tool_mode {tool_mode!r} must be 'native', 'json' or 'skeleton'"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.tool_mode = tool_mode
        self.timeout_s = float(timeout_s)
        self._transport = _transport  # test hook
        self._flight = _SingleFlight()
        #: per-client ids
        self._call_ids = _CallIdFactory()
        self.think_blocks_seen: int = 0
        self.parse_failures: int = 0
        self.chat_calls: int = 0

    @property
    def dropped_calls(self) -> int:
        return self._flight.dropped_calls

    def chat(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """POST /chat/completions; parse deterministically per tool_mode."""
        if not self._flight.acquire():
            logger.warning("server chat dropped: a previous call is in flight")
            return ChatResult(dropped=True)
        try:
            self.chat_calls += 1
            if self.tool_mode == "skeleton":
                return self._chat_skeleton(messages, tools)
            if self.tool_mode == "json":
                return self._chat_json(messages, tools)
            return self._chat_native(messages, tools)
        finally:
            self._flight.release()

    # -- tool modes -----------------------------------------------------------

    def _chat_native(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Structured ``tool_calls`` via the sidecar's OpenAI-compatible
        endpoint (the sidecar applies its own chat template)."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.enable_thinking:
            payload["enable_thinking"] = True
        body = self._post("/chat/completions", payload)
        try:
            result = _parse_openai_chat_response(body, self._call_ids)
        except ToolCallParseError as exc:
            logger.warning("server response parse failed: %s", exc)
            self.parse_failures += 1
            return ChatResult()
        raw = self._raw_content(body)
        if raw:
            self.think_blocks_seen += len(_find_think_blocks(raw))
        return result

    def _chat_skeleton(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Literal-JSON-skeleton mode (same deterministic contract as
        LlamaCppClient._chat_skeleton): one auto-retry, then the empty
        ChatResult repair path."""
        valid_names = {
            spec["function"]["name"]
            for spec in tools
            if isinstance(spec.get("function"), dict)
            and isinstance(spec["function"].get("name"), str)
        }
        if not valid_names:
            return self._chat_native(messages, tools)
        instruction = _build_skeleton_prompt(tools)
        prompt_messages = self._inject_instruction(messages, instruction)
        for attempt in (0, 1):
            body = self._post("/chat/completions", self._payload(prompt_messages, []))
            raw = self._raw_content(body)
            if raw is None:
                self.parse_failures += 1
                return ChatResult()
            self.think_blocks_seen += len(_find_think_blocks(raw))
            call = _parse_skeleton_tool_call(raw, valid_names, self._call_ids)
            if call is not None:
                return ChatResult(tool_calls=[call])
            if attempt == 0:
                prompt_messages.append(
                    {"role": "user", "content": _SKELETON_RETRY_MESSAGE}
                )
        self.parse_failures += 1
        return ChatResult()

    def _chat_json(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Schema-in-prompt JSON mode (name/arguments shape; mirrors
        LlamaCppClient._chat_json): one auto-retry, then empty result."""
        instruction = LlamaCppClient._build_json_prompt(tools)
        prompt_messages = self._inject_instruction(messages, instruction)
        for attempt in (0, 1):
            body = self._post("/chat/completions", self._payload(prompt_messages, []))
            raw = self._raw_content(body)
            if raw is None:
                self.parse_failures += 1
                return ChatResult()
            self.think_blocks_seen += len(_find_think_blocks(raw))
            cleaned = _strip_think(raw)
            maybe = _extract_first_balanced_json(cleaned)
            if maybe:
                try:
                    data = json.loads(maybe)
                except ValueError as exc:
                    logger.warning("server json-mode parse failed: %s", exc)
                    data = None
                if isinstance(data, dict) and isinstance(data.get("name"), str):
                    arguments = data.get("arguments")
                    if isinstance(arguments, dict):
                        return ChatResult(
                            tool_calls=[
                                ToolCall(
                                    id=self._call_ids("json"), name=data["name"],
                                    arguments=arguments,
                                )
                            ],
                        )
            if attempt == 0:
                prompt_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was not valid JSON. Respond with a "
                            'single JSON object of the form {"name": "<tool>", '
                            '"arguments": {<params>}} and nothing else.'
                        ),
                    }
                )
        self.parse_failures += 1
        return ChatResult()

    @staticmethod
    def _inject_instruction(messages: List[dict], instruction: str) -> List[dict]:
        """Copy messages with the tool-mode instruction appended to the
        first user message (or as a trailing user message when none)."""
        prompt_messages: List[dict] = []
        inserted = False
        for m in messages:
            prompt_messages.append(dict(m))
            if m.get("role") == "user" and not inserted:
                prompt_messages[-1]["content"] = (
                    f"{m.get('content') or ''}\n\n{instruction}"
                )
                inserted = True
        if not inserted:
            prompt_messages.append({"role": "user", "content": instruction})
        return prompt_messages

