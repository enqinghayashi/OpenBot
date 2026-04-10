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


class _CloudHTTPError(Exception):
    """Internal: an HTTP error that still carries the provider's body.

    ``urllib.error.HTTPError`` stringifies to just "HTTP Error 400: Bad
    Request"; the body — the part naming the offending parameter — is only
    available via ``.read()`` and is lost once the handler returns.  This
    carries it so ``_post`` can both report it and act on it.  Never escapes
    the client: ``_post`` converts it to ``ChatTransportError``.
    """

    def __init__(self, status: int, reason: str, detail: str,
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(f"{status} {reason}: {detail}")
        self.status = status
        self.reason = reason
        self.detail = detail
        self.cause = cause


_OPENAI_HOSTS = ("api.openai.com",)
#: OpenAI reasoning families (gpt-5.x, o1/o3/o4) reject
#: ``max_tokens`` and require ``max_completion_tokens``.
_MAX_COMPLETION_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _token_param_for(base_url: str, model: str) -> str:
    """Name of the completion-budget field for this provider + model.

    OpenAI-hosted reasoning models require ``max_completion_tokens`` and
    return HTTP 400 on ``max_tokens``.  Everything else — DeepSeek, the
    Gemini OpenAI-compat endpoint, older OpenAI ids, and any other
    OpenAI-shaped wire — still requires ``max_tokens``.  Scoped by HOST as
    well as model id so that a DeepSeek/Gemini model whose name happens to
    start with an OpenAI prefix is not misrouted.
    """
    host = (base_url or "").split("//")[-1].split("/")[0].lower()
    if not any(host == h or host.endswith("." + h) for h in _OPENAI_HOSTS):
        return "max_tokens"
    name = (model or "").lower()
    if any(name.startswith(p) for p in _MAX_COMPLETION_PREFIXES):
        return "max_completion_tokens"
    return "max_tokens"


def _temperature_for(base_url: str, model: str, fallback: float) -> Optional[float]:
    """Temperature to send to this provider + model, or None to OMIT the field.

    OpenAI-hosted reasoning-class ids (the same family as
    ``_MAX_COMPLETION_PREFIXES``) reject ANY temperature but the default 1 —
    LIVE-VERIFIED 2026-09-02 (bake-off, gpt-5.6-luna): HTTP 400
    "Unsupported value: 'temperature' does not support 0.2 with this
    model. Only the default (1) value is supported."  Omitting the field
    is the API default (1).  Everything else keeps the caller's
    temperature.  Same HOST-scoping rule as ``_token_param_for`` so a
    non-OpenAI model sharing a name prefix is not misrouted.
    """
    host = (base_url or "").split("//")[-1].split("/")[0].lower()
    if not any(host == h or host.endswith("." + h) for h in _OPENAI_HOSTS):
        return fallback
    name = (model or "").lower()
    if any(name.startswith(p) for p in _MAX_COMPLETION_PREFIXES):
        return None
    return fallback


def _finish_reason(body: dict) -> Optional[str]:
    """``choices[0].finish_reason``, or None when the wire omits it.

    Read so that a TRUNCATED response can never be mistaken for a decision.
    See ``_truncated`` for why this matters.
    """
    try:
        return body["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _truncated(body: dict) -> bool:
    """True when the provider stopped because the budget ran out.

    LIVE-VERIFIED 2026-09-02 (deepseek-v4-flash-vision-exp, skeleton mode,
    real billed calls): a reasoning model spends completion budget on
    reasoning_tokens BEFORE emitting the answer, so at our shipped
    ``max_tokens: 256`` a response can stop mid-JSON:

        max_tokens=190 -> finish_reason "length", reasoning 136,
          content '{"tool": "say", "args": {"text": "The image shows a solid,'
        max_tokens=230 -> finish_reason "length", reasoning 174,
          content '{"args": {"text": "I see a uniform, solid red square...'

    That fragment does not parse as a tool call, so before this check it fell
    through to the prose fallback and was SPOKEN to the operator as the final
    answer — raw JSON read aloud, ``status: completed``, ``parse_failures``
    never incremented, no execution, no confirm gate, no verifier.  A textbook
    false success, on the deployed transcript lane.

    A truncation is not a reply.  Callers must treat it as a failure.
    """
    return _finish_reason(body) == "length"


def _count_image_parts(body: dict) -> int:
    """Count image content parts in a response (vision-staging stub)."""
    try:
        content = body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return 0
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for part in content
        if isinstance(part, dict) and part.get("type") in ("image_url", "image")
    )


class CloudLLMClient:
    """OpenAI-compatible cloud client (stdlib urllib; key from env).

    Parameters:
    - ``base_url``: the OpenAI-compatible endpoint root, e.g.
      ``https://api.openai.com/v1`` (required — never guessed).
    - ``model``: the deployed model name (required — never guessed).
    - ``api_key_env``: the name of the env var holding the API key
      (default ``LLM_API_KEY``).  The key is read per request; a
      missing/empty var raises :class:`ChatTransportError` before
      anything is sent.
    - ``tool_mode``: ``"skeleton"`` (default; literal-JSON-skeleton prompt
      + deterministic parse), ``"json"`` (schema-in-prompt fallback) or
      ``"native"`` (structured ``tool_calls`` passthrough).
    - ``enable_thinking``: forwarded to the endpoint only when True
      (provider-specific; disabled by default — think tokens inflate
      latency and are a parse-failure source).
    - ``timeout_s``: per-request urllib timeout (default 120).
    - ``_transport``: test seam — a callable ``(request, timeout) -> body``
      replacing ``urllib.request.urlopen``.
    - ``prose_fallback``: when True, the skeleton/json modes return the
      model's plain text as ``ChatResult.text`` when no tool call parses
      (the D4 prose fallback — INFO-logged) instead of the empty
      repair-path result.  Default False: existing callers and the agent
      REPL behave exactly as before.

    ``chat(..., image_jpeg_b64=..., image_media_type=...)`` optionally
    attaches ONE jpeg frame to the last user message as an OpenAI
    ``image_url`` content part; with no image the payload is
    byte-identical to the pre-L3 shape.

    Counters mirror :class:`LlamaServerClient` for parity: ``chat_calls``,
    ``parse_failures``, ``think_blocks_seen``, ``dropped_calls``, plus
    ``images_seen`` (image parts counted on request response content).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.2,
        max_tokens: int = 256,
        enable_thinking: bool = False,
        tool_mode: str = "skeleton",
        timeout_s: float = 120.0,
        _transport: Optional[Callable] = None,
        prose_fallback: bool = False,
    ) -> None:
        if not base_url:
            raise ValueError("CloudLLMClient requires base_url (llm.base_url)")
        if not model:
            raise ValueError("CloudLLMClient requires model (llm.model)")
        if tool_mode not in ("native", "json", "skeleton"):
            raise ValueError(
                f"tool_mode {tool_mode!r} must be 'native', 'json' or 'skeleton'"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        # an empty name falls back to the default env var, so
        # configs may leave it blank.
        self.api_key_env = api_key_env or "LLM_API_KEY"
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # completion budget field is provider-specific: OpenAI
        # reasoning models reject max_tokens, while DeepSeek and
        # the Gemini compat endpoint take it. _post retries once.
        self._token_param = _token_param_for(self.base_url, self.model)
        # temperature is provider-specific too: reasoning-class
        # ids reject anything but the default 1, so the field is
        # omitted for them; other models keep the caller's value.
        self.enable_thinking = bool(enable_thinking)
        self.tool_mode = tool_mode
        self.timeout_s = float(timeout_s)
        self._transport = _transport  # test seam: overrides the HTTP call
        self.prose_fallback = bool(prose_fallback)
        self._flight = _SingleFlight()
        #: per-client tool-call id mint: fixed ids collide
        #: across turns and leave dangling tool_calls (HTTP 400
        #: on the next payload), so ids stay unique per process.
        self._call_ids = _CallIdFactory()
        self.think_blocks_seen: int = 0
        self.parse_failures: int = 0
        self.chat_calls: int = 0
        self.images_seen: int = 0
        #: replies cut off by the completion budget (finish_reason
        #: "length"; "max_tokens" on Anthropic), counted apart
        #: from parse_failures.
        self.truncations: int = 0

    @property
    def dropped_calls(self) -> int:
        return self._flight.dropped_calls

    def chat(
        self,
        messages: List[dict],
        tools: List[dict],
        *,
        image_jpeg_b64: Optional[str] = None,
        image_media_type: str = "image/jpeg",
    ) -> ChatResult:
        """POST /chat/completions (stream: False); parse per tool_mode.

        ``image_jpeg_b64`` (optional): attach ONE jpeg frame to the LAST
        user message as an OpenAI ``image_url`` content part (see
        ``_attach_image``); ``image_media_type`` defaults to
        ``image/jpeg``.  When no image is given the payload is
        byte-identical to the pre-L3 shape.

        Never raises for recoverable output problems; transport failures
        (HTTP errors, missing API key) raise :class:`ChatTransportError`.
        Single-flight drop while another call is in flight.
        """
        if not self._flight.acquire():
            logger.warning("cloud chat dropped: a previous call is in flight")
            return ChatResult(dropped=True)
        try:
            self.chat_calls += 1
            if image_jpeg_b64:
                messages = self._attach_image(
                    messages, image_jpeg_b64, image_media_type
                )
            if self.tool_mode == "skeleton":
                return self._chat_skeleton(messages, tools)
            if self.tool_mode == "json":
                return self._chat_json(messages, tools)
            return self._chat_native(messages, tools)
        finally:
            self._flight.release()

    # -- tool modes -----------------------------------------------------------

    def _chat_native(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Structured ``tool_calls`` passthrough (cloud provider templates)."""
        body = self._post("/chat/completions", self._payload(messages, tools))
        try:
            result = _parse_cloud_response(body, self._call_ids)
        except ToolCallParseError as exc:
            logger.warning("cloud response parse failed: %s", exc)
            self.parse_failures += 1
            return ChatResult()
        self._count(body)
        return result

    def _chat_skeleton(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Literal-JSON-skeleton mode: one auto-retry, then empty result.

        Same deterministic contract as LlamaCppClient._chat_skeleton:
        think-block detection on the RAW content first (the Qwen3 finding),
        tool name validated against the provided specs, args must be a
        dict.  ``stream`` stays False (the DeepSeek streaming tool-call
        corruption finding).
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
        prompt_messages = self._inject_instruction(messages, instruction)
        truncated = False
        for attempt in (0, 1):
            body = self._post(
                "/chat/completions", self._payload(prompt_messages, [])
            )
            truncated = _truncated(body)
            if truncated:
                self.truncations += 1
            content = self._message_content(body)
            raw = self._content_text(content)
            if raw is None:
                self.parse_failures += 1
                return ChatResult(truncated=truncated)
            self.think_blocks_seen += len(_find_think_blocks(raw))
            self.images_seen += _count_image_parts(body)
            call = _parse_skeleton_tool_call(raw, valid_names, self._call_ids)
            if call is not None:
                if truncated:
                    # budget ran out but a complete, validated
                    # tool object parsed; one call per turn, so
                    # that is the answer, unlike the no-call case.
                    logger.warning(
                        "cloud response truncated (finish_reason=length) but a "
                        "complete tool call parsed; executing it. Raise "
                        "llm.max_tokens (currently %d) if this recurs.",
                        self.max_tokens,
                    )
                if self.prose_fallback:
                    # return the call plus any prose around the
                    # skeleton object, not the object itself,
                    # which the transcript lane would speak.
                    return ChatResult(
                        tool_calls=[call], text=_prose_without_skeleton(raw),
                        truncated=truncated,
                    )
                return ChatResult(tool_calls=[call], truncated=truncated)
            if attempt == 0:
                prompt_messages.append(
                    {"role": "user", "content": _SKELETON_RETRY_MESSAGE}
                )
        if truncated:
            # a budget-cut reply is not prose and must not reach
            # the fallback below: the transcript lane speaks it and
            # marks the turn completed, with no execution or check.
            logger.error(
                "cloud response truncated (finish_reason=length) with no "
                "parseable tool call; refusing prose fallback on a partial "
                "reply. Raise llm.max_tokens (currently %d) — reasoning "
                "models spend this budget before they answer.",
                self.max_tokens,
            )
            return ChatResult(truncated=True)
        if self.prose_fallback:
            # no tool call parsed, so fall back to plain text.
            prose = self._prose_text(content)
            if prose:
                logger.info(
                    "cloud prose fallback (no tool call parsed): %s", prose
                )
                return ChatResult(text=prose)
        self.parse_failures += 1
        return ChatResult()

    def _chat_json(self, messages: List[dict], tools: List[dict]) -> ChatResult:
        """Schema-in-prompt JSON mode (name/arguments shape): one auto-
        retry, then the empty ChatResult repair path."""
        instruction = LlamaCppClient._build_json_prompt(tools)
        prompt_messages = self._inject_instruction(messages, instruction)
        for attempt in (0, 1):
            body = self._post(
                "/chat/completions", self._payload(prompt_messages, [])
            )
            content = self._message_content(body)
            raw = self._content_text(content)
            if raw is None:
                self.parse_failures += 1
                return ChatResult()
            self.think_blocks_seen += len(_find_think_blocks(raw))
            self.images_seen += _count_image_parts(body)
            cleaned = _strip_think(raw)
            maybe = _extract_first_balanced_json(cleaned)
            if maybe:
                try:
                    data = json.loads(maybe)
                except ValueError as exc:
                    logger.warning("cloud json-mode parse failed: %s", exc)
                    data = None
                if isinstance(data, dict) and isinstance(data.get("name"), str):
                    arguments = data.get("arguments")
                    if isinstance(arguments, dict):
                        if self.prose_fallback:
                            return ChatResult(
                                tool_calls=[
                                    ToolCall(
                                        id=self._call_ids("json"),
                                        name=data["name"],
                                        arguments=arguments,
                                    )
                                ],
                                text=self._prose_text(content),
                            )
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
        if self.prose_fallback:
            # no tool call parsed, so fall back to plain text.
            prose = self._prose_text(content)
            if prose:
                logger.info(
                    "cloud prose fallback (no tool call parsed): %s", prose
                )
                return ChatResult(text=prose)
        self.parse_failures += 1
        return ChatResult()

    # -- helper methods -------------------------------------------------------

    @staticmethod
    def _inject_instruction(messages: List[dict], instruction: str) -> List[dict]:
        """Copy messages with the tool-mode instruction appended to the
        first user message (or as a trailing user message when none).

        ``content`` may be a string or an OpenAI multimodal parts list
        (the L3 image shape): a parts list gets the instruction appended
        as an extra ``text`` part; a string keeps the historical
        ``"<content>\\n\\n<instruction>"`` concat (byte-identical payloads
        when no image is used).  The caller's list is never mutated.
        """
        prompt_messages: List[dict] = []
        inserted = False
        for m in messages:
            prompt_messages.append(dict(m))
            if m.get("role") == "user" and not inserted:
                content = m.get("content")
                if isinstance(content, list):
                    prompt_messages[-1]["content"] = list(content) + [
                        {"type": "text", "text": instruction}
                    ]
                else:
                    prompt_messages[-1]["content"] = (
                        f"{content or ''}\n\n{instruction}"
                    )
                inserted = True
        if not inserted:
            prompt_messages.append({"role": "user", "content": instruction})
        return prompt_messages

    @staticmethod
    def _attach_image(
        messages: List[dict],
        image_jpeg_b64: str,
        image_media_type: str = "image/jpeg",
    ) -> List[dict]:
        """Copy messages with ONE jpeg frame attached to the LAST user
        message as an OpenAI ``image_url`` content part.

        The target message's content becomes a parts list (existing parts
        kept, the image part appended; a string content becomes
        ``[{"type": "text", ...}, <image part>]``).  When no user message
        exists a new ``role: "user"`` message carrying only the image
        part is appended.  The caller's list is never mutated.
        ``image_jpeg_b64`` may be raw base64 or a full ``data:`` URI (the
        prefix is stripped defensively).
        """
        data = image_jpeg_b64.split(",", 1)[-1]
        part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{image_media_type};base64,{data}",
            },
        }
        out = [dict(m) for m in messages]
        for m in reversed(out):
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, list):
                    m["content"] = list(content) + [part]
                elif content:
                    m["content"] = [{"type": "text", "text": content}, part]
                else:
                    m["content"] = [part]
                return out
        out.append({"role": "user", "content": [part]})
        return out

