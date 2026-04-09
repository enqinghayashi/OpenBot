"""Cloud LLM client — the Plan-B R4 rung (OpenAI-compatible HTTP endpoint).

Drop-in :class:`LLMClient` for a hosted OpenAI-compatible
``/chat/completions`` endpoint (the plan's R4 ladder: Claude Haiku 4.5 /
Gemini 3.7-Flash / GPT-5.4-mini via their OpenAI-compat surfaces).

Shape — deliberate deviations from the OpenBot example (grounding read:
``How_LLM_IMPLEMENTED_ON_G1_RUNNABLE_EXAMPLE/OpenBot/src/g1_voice/``):

- OpenBot's ``chat/cloud/backend.py`` is an OpenAI *Realtime* websocket
  session (``response.output_item.done`` events, session.update handshake);
  we use the plain OpenAI chat-completions HTTP shape because the brain's
  protocol seam (brain/agent/llm.py) is already OpenAI-style dicts —
  no Realtime event translation, no Anthropic-Messages layer.
- Their client sends ``Authorization: Bearer <key>`` from config; we keep
  the same header style but the key is read from ``os.environ`` per
  request and NEVER lives in config or the repo (deploy-secret rule).
- Their ``server/api.py`` + ``server/hub.py`` websocket console fan-out is
  not needed here: the brain<->gateway lane is the existing TCP 3900
  socket; this client is a drop-in ``chat()`` object, not a server.
- Their HTTP lib: none (Realtime websocket); ours: stdlib ``urllib`` only,
  mirroring :class:`LlamaServerClient` (llm.py) — no third-party import.

``stream`` is ALWAYS False: the DeepSeek streaming tool-call corruption
finding (plan Research result 3, V3) makes non-streaming the safe shape
for tool calls.

Request-side vision (L3 2026-08-31): ``chat(..., image_jpeg_b64=...)``
attaches ONE jpeg frame to the LAST user message as an OpenAI multimodal
content part (``[{"type": "text", ...}, {"type": "image_url",
"image_url": {"url": "data:image/jpeg;base64,<b64>"}}]``).  With no image
the payload is byte-identical to the pre-L3 shape.  Message ``content``
may be a string OR a parts list on both the request and the response side
(``_inject_instruction`` and ``_attach_image`` preserve both).  A
response ``content`` that arrives as a parts list is parsed — text parts
are joined, image parts are counted in ``images_seen``.

History round-trip (L3): the OpenAI-style history ALREADY carries
assistant ``tool_calls`` and ``role: "tool"`` results (``tool_call_id``)
— ``_payload`` sends ``messages`` verbatim, and ``_inject_instruction``
only copies messages and appends instruction text to the first user
message, so the round-trip passes through untouched.  Trace:
``G1Agent._llm_loop`` (brain/agent/agent.py) appends ``{"role":
"assistant", "content": ..., "tool_calls": [...]}`` and ``{"role":
"tool", "tool_call_id": <id>, "name": ..., "content": ...}``;
CloudLLMClient forwards both unchanged (native mode) or with only the
instruction text appended (skeleton/json modes).  No new history shape
was needed — /chat/completions defines tool results as ``role: "tool"``
(the Responses-API ``function_call_output`` part style does not apply to
this endpoint).

Prose fallback (the D4 parser fix): ``prose_fallback=True`` makes the
skeleton/json modes return the model's plain text as ``ChatResult.text``
when no tool call parses instead of the empty repair-path result (native
mode already returns text).  The raw extracted text is logged at INFO;
transport errors and malformed-body semantics are unchanged — an error
is never swallowed into empty prose.  Default ``False``: the agent
REPL's behavior is unchanged.

REST endpoint compatibility (measured 2026-08-31 against the request
shape below — POST ``{base_url}/chat/completions``, headers
``Content-Type: application/json`` + ``Authorization: Bearer <env key>``,
body ``{"model", "messages", "tools", "temperature", "max_tokens",
"stream": false}``):

  (a) OpenAI ``/v1/chat/completions`` — compatible as-is.
  (b) Gemini OpenAI-compat endpoint — ``https://generativelanguage.
      googleapis.com/v1beta/openai/chat/completions``; compatible as-is
      (Bearer with a Google AI Studio key; OpenAI-shape ``tools`` and
      ``image_url`` data-URI parts supported).
  (c) Groq — ``https://api.groq.com/openai/v1``; compatible for this
      shape (Bearer; tools/tool_calls supported) but TEXT-ONLY (no
      ``image_url`` parts — the image path is provider-gated at deploy),
      rejects ``logprobs``/``logit_bias``/``top_logprobs``, and
      ``temperature`` 0 is converted to 1e-8 (we default 0.2).
      ``messages[].name`` IS sent on tool messages: the agent lane
      (brain/agent/agent.py) and the transcript lane
      (voice/transcript.py) append ``"name"`` to ``role: "tool"``
      messages and CloudLLMClient forwards it verbatim.  Groq's
      OpenAI-compat docs (console.groq.com/docs/openai) list
      ``messages[].name`` under "currently not supported" fields that
      return HTTP 400 when supplied, with no documented carve-out for
      tool messages — so tool-call traffic against Groq carries a real
      400 risk on that field (a previous version of this note claimed
      "we send none", which was wrong).
  (d) Anthropic has TWO official surfaces, both valid:
      1. The native Messages API — NOT what CloudLLMClient speaks:
         ``api.anthropic.com/v1/messages``, auth via ``x-api-key`` +
         ``anthropic-version: 2023-06-01`` headers (no Bearer), a body
         with ``system`` hoisted out of messages, tools as ``{"name",
         "description", "input_schema"}``, assistant tool calls as
         ``tool_use`` content blocks, tool results as ``role: "user"``
         messages with ``tool_result`` blocks, images as ``{"type":
         "image", "source": {"type": "base64", ...}}`` blocks, and a
         ``{"content": [...], "stop_reason": ...}`` response (no
         ``choices[0].message``).  The :class:`AnthropicMessagesClient`
         adapter in this module speaks this shape behind the same
         ``chat()`` interface (native ``tool_use`` mode only); it was
         implemented for the REST bake-off column and is the
         production-grade surface (full feature set: thinking, prompt
         caching, PDFs, structured outputs).
      2. The official OpenAI-compat layer — ``https://api.anthropic.com/
         v1/``, documented by Anthropic at
         platform.claude.com/docs/en/api/openai-sdk (the same document
         was previously served at docs.anthropic.com/en/api/openai-sdk).
         Auth is ``Authorization: Bearer`` with an ``ANTHROPIC_API_KEY``,
         chat is POST ``/v1/chat/completions`` returning the OpenAI
         ``choices[0].message`` shape, and it supports ``tools``,
         assistant ``tool_calls``, ``role: "tool"`` messages with
         ``tool_call_id``, and user ``image_url`` content parts — the
         exact shape CloudLLMClient already speaks (Bearer auth,
         ``_payload``, ``_attach_image``).  CloudLLMClient works against
         it AS-IS: point ``base_url`` at ``https://api.anthropic.com/v1``
         and the tools/image/tool-result round-trip passes through
         unchanged (measured against the documented field tables:
         system/developer messages are hoisted into one initial system
         message; ``name`` on tool messages is ignored; ``temperature``
         > 1 is capped at 1; ``logprobs``/``logit_bias``/``top_logprobs``
         are ignored, not rejected).  NO ADAPTER IS NEEDED for this
         surface.  Anthropic's own caveat: the compat layer is
         "primarily intended to test and compare model capabilities,
         and is not considered a long-term or production-ready solution
         for most use cases" — prefer the native Messages shape (item 1)
         for anything beyond evaluation.

Reachability probe (the health_probe contract): ``CloudLLMClient.probe()``
and :class:`AnthropicMessagesClient` ``probe()`` do a SYNCHRONOUS,
bounded (2 s default) GET of the provider's model-list endpoint and
return ``bool(any 2xx)``.  No model tokens are consumed and it NEVER
raises (False on timeout / HTTP error / connection refused / any
non-2xx).  The transcript lane's health_probe() (voice/transcript.py)
and the app's Director probe use it so a provider failover is
recoverable instead of terminal.

``enable_thinking`` (a DeepSeek/Qwen3 extension) is forwarded to the
endpoint only when True — OpenAI/Gemini/Groq do not define it, so the
default False is the safe value on every surface.

``chat()`` never raises for recoverable model-output problems (malformed
tool-call JSON etc.): those become an empty :class:`ChatResult` (the
agent's repair path) and increment ``parse_failures``.  Transport failures
(HTTP errors, a missing API-key env var) raise :class:`ChatTransportError`;
the agent wraps them.  Single-flight drop semantics are shared with the
other clients (``dropped_calls``).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from .llm import (
    ChatResult,
    ChatTransportError,
    ToolCall,
    ToolCallParseError,
    LlamaCppClient,
    _CallIdFactory,
    _SKELETON_RETRY_MESSAGE,
    _SingleFlight,
    _build_skeleton_prompt,
    _extract_first_balanced_json,
    _find_think_blocks,
    _module_call_id,
    _parse_skeleton_tool_call,
    _parse_tool_calls_payload,
    _strip_think,
)

logger = logging.getLogger(__name__)

# probe timeout: far below the chat timeout so failover
# decides fast and a stuck host cannot stall the caller.
_PROBE_TIMEOUT_S = 2.0


def _probe_http_get(url: str, headers: Dict[str, str], timeout: float) -> bool:
    """GET ``url``; True iff the response status is any 2xx.

    The probe contract is "never raise": timeouts, connection refusals,
    HTTP errors (HTTPError is a URLError subclass) and any other
    exception raised by the GET — including the ValueError a malformed
    (scheme-less) base_url triggers — all return False and are logged
    at DEBUG.  Request construction and urlopen both sit inside the
    try, so the bare except covers every raise site.  The response
    body is never read — the check costs no model tokens.
    """
    try:
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            code = resp.getcode()
    except Exception as exc:  # noqa: BLE001 — the probe contract is never-raise
        logger.debug("probe GET %s failed: %s", url, exc)
        return False
    return code is not None and 200 <= code < 300


def _parse_cloud_response(
    body: dict,
    next_call_id: Optional[Callable[[str], str]] = None,
) -> ChatResult:
    """OpenAI chat-completion parse with the multimodal content stub.

    Mirrors ``brain.agent.llm._parse_openai_chat_response``; additionally
    accepts ``content`` as a list of parts (the OpenAI multimodal shape):
    text parts are joined, image parts are recognized-but-unused (the
    client counts them in ``images_seen``).  Think blocks are stripped
    from text.  Raises :class:`ToolCallParseError` for malformed bodies
    (callers convert that to an empty result).  ``next_call_id`` passes
    the caller's id mint through to payload parsing (synthesized ids
    only; provider ids are kept verbatim).
    """
    try:
        choice = body["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise ToolCallParseError(
            f"response body has no choices[0].message: {exc}"
        ) from exc
    text: Optional[str] = None
    raw_content = message.get("content")
    if isinstance(raw_content, str):
        text = _strip_think(raw_content)
    elif isinstance(raw_content, list):
        parts = [
            part.get("text")
            for part in raw_content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
        if parts:
            text = _strip_think("\n".join(parts))
    calls: Optional[List[ToolCall]] = None
    if message.get("tool_calls"):
        calls = _parse_tool_calls_payload(message["tool_calls"], next_call_id)
    return ChatResult(text=text, tool_calls=calls)


def _prose_without_skeleton(raw: Optional[str]) -> Optional[str]:
    """Prose with the skeleton JSON removed, or None when nothing remains.

    In skeleton mode the model is told to answer with a literal JSON object,
    so on a SUCCESSFUL parse the message content usually *is* that object.
    Handing it back as ``ChatResult.text`` made the transcript lane speak it:
    ``voice/transcript.py:854-863`` runs ``if text: ... self._speak(text)``
    unconditionally, independent of whether a tool call was also returned.
    With the shipped ``prose_fallback: true`` (config.cloud.yaml:86,
    config.voice.yaml:57) that means the operator hears

        {"tool": "say", "args": {"text": "I see a uniform, flat crimson field...

    read ALOUD on every good turn — LIVE-OBSERVED 2026-09-02 against
    deepseek-v4-flash-vision-exp.  Stripping the object leaves only genuine
    surrounding prose, and None when the reply was pure skeleton, so the
    lane speaks nothing extra.  Reuses the parser's own balanced scanner so
    the two cannot disagree about where the object ends.
    """
    if not raw:
        return None
    text = _strip_think(raw)
    blob = _extract_first_balanced_json(text)
    if blob:
        text = text.replace(blob, " ", 1)
    text = text.strip()
    return text or None
