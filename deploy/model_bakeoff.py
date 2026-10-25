#!/usr/bin/env python3
"""model_bakeoff.py — OpenBot v4 model bake-off harness (plan section 6).

A thesis-grade measurement tool: every vision-capable candidate is measured
across BOTH brain lanes — the Realtime speech-to-speech WebSocket lane and
the REST multimodal text brain + local STT/TTS lane — on the SAME fixed
turn protocol, the SAME fixed ground-truth scene set, and the SAME tool
contract, then ranked by cost-efficiency ($/turn at equal-or-better
accuracy).  Cross-lane parity is deliberate: a lane is a wire, not a
score; the deployed pin (brain.lane + model, config.voice.yaml) is decided
from the ranked table, never from lane preference (docs/BAKEOFF.md).

THE HONESTY LAW (holds for every column): usage and timing are read from
what the lane actually exposes — never assumed constants:

  * REST lane (brain/agent/llm_cloud.py — CloudLLMClient /
    AnthropicMessagesClient): usage comes from the RAW response bodies,
    captured through the clients' documented ``_transport`` test seam
    (``callable(request, timeout) -> body`` — the seam IS the client's
    own public surface).  ``usage.prompt_tokens`` includes image tokens;
    ``usage.prompt_tokens_details.image_tokens`` is reported as the
    text/image split when the provider reports it, else the split is
    "Not reported".  The REST lane never streams (``stream: False`` is a
    documented law), so TTFT == the full round trip (voice/transcript.py
    documents this equality; it is a measured identity, not a guess).
  * Realtime lane (voice/realtime.py — RealtimeClient): usage comes from
    the ``response.done`` usage payload (``input_tokens`` /
    ``output_tokens`` and the ``*_token_details`` audio/text splits) —
    the L8 cost-ledger exposure voice/session.py serves the app with.
  * Prices come from the deploy/prices.json sidecar (recorded
    2026-08-31; entries with ``verified: false`` have no recorded
    per-token rate in the repo and the harness REFUSES to invent one:
    cost is "Not reported" with the re-verify instruction).
  * A run that fails is COUNTED in the reliability denominator
    (runs_total == runs_requested; every attempted run gets a row with
    its status).  It is never silently dropped, never retried into
    success.
  * "Not reported" is used ONLY for genuinely-unmeasurable cells (a
    failed run's timing/cost, a veto-flagged text-only row's vision-QA,
    an unverified price).

Module-level import surface is STDLIB-ONLY.  The brain (CloudLLMClient /
AnthropicMessagesClient) and voice (RealtimeClient / flatten_specs)
imports are LAZY, inside the runners, so ``--list`` works with zero
dependencies on any laptop.

CLI:

  --list [--format json|csv]      the 11-candidate manifest (zero deps)
  --candidate <id|all>            run one candidate (or all 11)
  --runs N                        per-candidate run count (default 20)
  --timeout SEC                   per-run bound (default: lane default;
                                  a run exceeding it fails with 'timeout')
  --out PATH                      results file (.json or .csv by extension)
  --format json|csv               force the output format
  --prices PATH                   price sidecar override (default
                                  deploy/prices.json)
  --stub-gateway URL              dev/probe seam: override the base_url
                                  for REST candidates (also
                                  BAKEOFF_STUB_GATEWAY env).  Echoed in
                                  the output — never a secret.
  --image-dir DIR                 real scene photos (jpeg), round-robin
                                  per run; without it a deterministic
                                  synthetic image is used per scene
                                  (dev/probe mode — per-run rows are
                                  marked synthetic-image).
  --audio WAV                     realtime lane: feed this PCM16 24 kHz
                                  mono WAV instead of text input.

Exit codes: 0 = the run completed (per-run failures are DATA, not exit
errors); 1 = harness error (out path unwritable, bad format); 2 =
REFUSAL (unknown candidate id, bad arguments).  Never exits nonzero
because a candidate failed runs — that is the measurement.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import io
import json
import math
import os
import re
import struct
import sys
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# candidates
#
# "repo-verified" means the model is backed in-repo; a "plan-matrix" id
# is not, so check it against the provider model listing before a run.
# veto rows carry no image on their wire: vision-QA is "Not reported"
# and the rest is measured.
# ---------------------------------------------------------------------------

#: wire -> provider base_url, keyed by wire id
WIRE_BASE_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic-compat": "https://api.anthropic.com/v1",
    "anthropic-native": "https://api.anthropic.com",  # client appends /v1/messages
    "gemini-compat": "https://generativelanguage.googleapis.com/v1beta/openai",
    "deepseek": "https://api.deepseek.com/v1",  # OpenAI-compat root
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",  # OpenAI-compat mode root
    "openai-realtime": "wss://api.openai.com/v1/realtime",
}


def _candidate(
    cid: str,
    model: str,
    lane: str,
    wire: str,
    key_env: str,
    profile_delta: Dict[str, Any],
    note: str,
    veto: str = "",
    baseline: bool = False,
) -> Dict[str, Any]:
    """One matrix row.  ``profile_delta`` is the config_loader overrides
    the harness validates the candidate's deploy pin with (the
    config-seam check in run mode)."""
    return {
        "id": cid,
        "model": model,
        "lane": lane,
        "wire": wire,
        "base_url": WIRE_BASE_URLS[wire],
        "key_env": key_env,
        "profile_delta": profile_delta,
        "note": note,
        "veto": veto,  # '' = image-capable; otherwise the veto reason
        "baseline": baseline,  # the default the ranking compares against
    }


CANDIDATES: Dict[str, Dict[str, Any]] = {
    # ---------------- REALTIME lane - speech-to-speech, OpenAI Realtime ----
    "gpt-realtime-2.1-mini": _candidate(
        "gpt-realtime-2.1-mini", "gpt-realtime-2.1-mini", "realtime",
        "openai-realtime", "OPENAI_API_KEY",
        {"brain": {"lane": "realtime"},
         "realtime": {"model": "gpt-realtime-2.1-mini"}},
        "SEALED default (config.voice.yaml realtime.model); the ranking "
        "baseline. repo-verified.",
        baseline=True,
    ),
    "gpt-realtime-2.1": _candidate(
        "gpt-realtime-2.1", "gpt-realtime-2.1", "realtime",
        "openai-realtime", "OPENAI_API_KEY",
        {"brain": {"lane": "realtime"},
         "realtime": {"model": "gpt-realtime-2.1"}},
        "full-size 2.1 family. plan-matrix id — verify day-of.",
    ),
    "gpt-realtime-1.5": _candidate(
        "gpt-realtime-1.5", "gpt-realtime-1.5", "realtime",
        "openai-realtime", "OPENAI_API_KEY",
        {"brain": {"lane": "realtime"},
         "realtime": {"model": "gpt-realtime-1.5"}},
        "prior-gen full. plan-matrix id — verify day-of.",
    ),
    # ---------------- REST lane - chat-completions / native Messages -------
    "gpt-5.4-mini": _candidate(
        "gpt-5.4-mini", "gpt-5.4-mini", "rest", "openai", "OPENAI_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "gpt-5.4-mini"}},
        "sealed rest placeholder (config.voice.yaml rest.model); "
        "battle-tested alternate, ~$0.0042/decision. repo-verified.",
        baseline=True,
    ),
    "gpt-5.6-luna": _candidate(
        "gpt-5.6-luna", "gpt-5.6-luna", "rest", "openai", "OPENAI_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "gpt-5.6-luna"}},
        "cheapest current-gen OpenAI vision tier, $0.20/$1.20 per 1M, "
        "135 image tokens at 424x240. repo-verified.",
    ),
    "claude-haiku-4-5-20251001.compat": _candidate(
        "claude-haiku-4-5-20251001.compat", "claude-haiku-4-5-20251001",
        "rest", "anthropic-compat", "ANTHROPIC_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "claude-haiku-4-5-20251001",
                  "base_url": "https://api.anthropic.com/v1"}},
        "the R4 pick (config.cloud.yaml pins it); OpenAI-compat surface "
        "(Bearer auth, no adapter). repo-verified.",
    ),
    "claude-haiku-4-5-20251001.native": _candidate(
        "claude-haiku-4-5-20251001.native", "claude-haiku-4-5-20251001",
        "rest", "anthropic-native", "ANTHROPIC_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "claude-haiku-4-5-20251001",
                  "base_url": "https://api.anthropic.com"}},
        "AnthropicMessagesClient — 'implemented for the REST bake-off "
        "column' (llm_cloud.py); the production-grade surface "
        "(x-api-key, tool_use blocks). repo-verified.",
    ),
    "claude-sonnet-5": _candidate(
        "claude-sonnet-5", "claude-sonnet-5", "rest", "anthropic-native",
        "ANTHROPIC_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "claude-sonnet-5",
                  "base_url": "https://api.anthropic.com"}},
        "accuracy tier: $2/$10 per 1M, pricing FAQ 'permanent'. "
        "repo-verified (PLAN_B.md price table).",
    ),
    "gemini-3.7-flash": _candidate(
        "gemini-3.7-flash", "gemini-3.7-flash", "rest", "gemini-compat",
        "GEMINI_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "gemini-3.7-flash",
                  "base_url": (
                      "https://generativelanguage.googleapis.com/"
                      "v1beta/openai")}},
        "budget profile: $0.75/$3.75 INTRO — doubles 2027-01-01; "
        "thought_signature round-trip gotcha (thinking disabled). "
        "repo-verified.",
    ),
    "deepseek-v4-flash-vision-exp": _candidate(
        "deepseek-v4-flash-vision-exp", "deepseek-v4-flash-vision-exp",
        "rest", "deepseek", "DEEPSEEK_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "deepseek-v4-flash-vision-exp",
                  "base_url": "https://api.deepseek.com/v1"}},
        "~$0.0008 off-peak / $0.0017 peak per decision — measured on the "
        "PLAIN id deepseek-v4-flash (text-only baseline, 2026-09-01), NOT "
        "transferable to this row; thinking ON by default "
        "(reasoning_content gotcha — thinking disabled). VETO LIFTED "
        "2026-09-02 by operator decision — image parts ride this wire; the "
        "PLAIN id is TEXT-ONLY (image part -> HTTP 400 'This model does "
        "not support image'), so this row uses the vision id "
        "deepseek-v4-flash-vision-exp (live-verified 2026-09-02). "
        "vision-exp price unmeasured — 'Not reported' until a day-of "
        "check. base_url path /v1 unverified — verify day-of.",
    ),
    "qwen3.7-plus": _candidate(
        "qwen3.7-plus", "qwen3.7-plus", "rest", "qwen", "QWEN_API_KEY",
        {"brain": {"lane": "transcript"},
         "rest": {"model": "qwen3.7-plus",
                  "base_url": (
                      "https://dashscope.aliyuncs.com/compatible-mode/v1")}},
        "multimodal, BFCL-v4 72.9% — the CN exception 'if ever needed'; "
        "CN-native veto for thesis scene images — measured text-only + "
        "marked. repo-verified.",
        veto="CN-native (Chinese native veto for thesis scene images)",
    ),
}


# ---------------------------------------------------------------------------
# scenes
#
# fixed order. each scene has a fixed prompt and expected answer; the
# prompt never holds the answer (colour lives only in the image).
# real photos are used when an image dir is given, otherwise a synthetic
# PNG of the expected colour, and that row is marked synthetic-image.
# ---------------------------------------------------------------------------

SCENES: List[Dict[str, Any]] = [
    {"id": "scene-a", "object": "bottle", "surface": "wooden table",
     "prompt": ("In the attached image there is one bottle on a wooden "
                "table. What color is the bottle? Call report_scene with "
                "the bottle's color and its object type."),
     "expected": ["red"], "synthetic_rgb": (200, 30, 30)},
    {"id": "scene-b", "object": "mug", "surface": "white desk",
     "prompt": ("In the attached image there is one mug on a white desk. "
                "What color is the mug? Call report_scene with the mug's "
                "color and its object type."),
     "expected": ["green"], "synthetic_rgb": (20, 160, 40)},
    {"id": "scene-c", "object": "plate", "surface": "grey counter",
     "prompt": ("In the attached image there is one plate on a grey "
                "counter. What color is the plate? Call report_scene with "
                "the plate's color and its object type."),
     "expected": ["blue"], "synthetic_rgb": (30, 60, 200)},
    {"id": "scene-d", "object": "box", "surface": "the floor",
     "prompt": ("In the attached image there is one box on the floor. "
                "What color is the box? Call report_scene with the box's "
                "color and its object type."),
     "expected": ["yellow"], "synthetic_rgb": (220, 190, 20)},
    {"id": "scene-e", "object": "cup", "surface": "metal table",
     "prompt": ("In the attached image there is one cup on a metal "
                "table. What color is the cup? Call report_scene with the "
                "cup's color and its object type."),
     "expected": ["black"], "synthetic_rgb": (30, 30, 30)},
]

#: the single tool contract every run carries, matching the deployed
#: decision turn.
TOOL_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_scene",
        "description": "Report the object type and its color observed in "
                       "the scene image.",
        "parameters": {
            "type": "object",
            "properties": {
                "object": {"type": "string"},
                "color": {"type": "string"},
            },
            "required": ["object", "color"],
        },
    },
}

VALID_TOOL_NAMES = {TOOL_SPEC["function"]["name"]}
REQUIRED_TOOL_ARGS = ["object", "color"]

DEFAULT_RUNS = 20            # at least 20 attempts; failures are counted, never retried
REST_TIMEOUT_S = 120.0       # 120 s; must match the deployed rest.timeout_s
#: REST completion budget, must match the deployed rest.max_tokens;
#: budget parity is checked at run time.
#: a 256-token cap truncated reasoning-heavy lanes (16/20 runs); a cheap
#: bench must never outrank the lane.
REST_MAX_TOKENS = 1024
REALTIME_TIMEOUT_S = 60.0    # realtime per-turn bound
REALTIME_CONNECT_S = 40.0    # open bound (retry ladder sleeps 8s)
MAX_TOOL_CONTINUATIONS = 3   # tool rounds allowed per turn
NOT_REPORTED = "Not reported"


# ---------------------------------------------------------------------------
# price sidecar
# ---------------------------------------------------------------------------


def default_prices_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.json")


def load_prices(path: str) -> Dict[str, Any]:
    """Load the prices sidecar (plain JSON).  Raises ValueError on a
    malformed file (loud build — a corrupted price table must never
    silently produce numbers)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("prices file %r has no 'models' mapping" % path)
    for key, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError("prices entry %r must be an object" % key)
    return models


def _price(entry: Dict[str, Any], key: str) -> Optional[float]:
    """A price entry field, or None when absent/null (never coerced)."""
    value = entry.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# synthetic scene image
# PNG built from zlib/struct only
# ---------------------------------------------------------------------------


def _solid_png(rgb: tuple, size: int = 48) -> bytes:
    """A solid-color PNG (pure stdlib zlib/struct) — the deterministic
    dev/probe image; the hardware run uses real camera photos."""
    row = b"\x00" + bytes(rgb) * size
    raw = row * size

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def scene_image(scene: Dict[str, Any], image_dir: Optional[str]) -> Dict[str, Any]:
    """(jpeg_b64, media_type, is_synthetic) for one scene.  --image-dir
    photos win (scene-a.jpg ... scene-e.jpg / .jpeg); otherwise the
    synthetic solid-color PNG of the scene's expected color."""
    if image_dir:
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = os.path.join(image_dir, scene["id"] + ext)
            if os.path.isfile(candidate):
                with open(candidate, "rb") as fh:
                    raw = fh.read()
                media = "image/png" if ext == ".png" else "image/jpeg"
                return {"b64": base64.b64encode(raw).decode("ascii"),
                        "media": media, "synthetic": False}
    return {"b64": base64.b64encode(_solid_png(scene["synthetic_rgb"])).decode("ascii"),
            "media": "image/png", "synthetic": True}


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


def _p95(values: List[float]) -> Optional[float]:
    """The 95th percentile: sorted rank ceil(0.95*n) (1-based), i.e. the
    largest value for n <= 20.  Documented definition — thesis-grade."""
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, int(math.ceil(0.95 * len(ordered))) - 1)
    return ordered[idx]


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _fmt(value: Optional[float], unit: str = "") -> Any:
    """A measured number or the literal 'Not reported'."""
    if value is None:
        return NOT_REPORTED
    return "{:.6f}{}".format(value, unit)


# ---------------------------------------------------------------------------
# REST runner
# clients imported lazily inside the runner
# ---------------------------------------------------------------------------


def _transport_factory():
    """The documented ``_transport`` seam wrapper: perform the client's
    urllib POST and capture the raw response body (usage lives there).
    A FRESH wrapper per run keeps usage attribution exact (a hung run's
    late body can never leak into a later run's span)."""
    captured: List[bytes] = []

    def transport(request: Any, timeout: float) -> bytes:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            body = resp.read()
        captured.append(body)
        return body

    return transport, captured


def _vision_scored_text(tool_args: Optional[Dict[str, Any]],
                        prose: Optional[str]) -> str:
    """The text vision-QA is scored against: the tool call's reported
    fields FIRST, prose second.

    LIVE-VERIFIED 2026-09-02: every SCENES prompt says "Call report_scene
    with the <object>'s color", so a well-behaved model puts the colour in
    the call ARGUMENTS and returns no prose at all.  Scoring prose alone
    (the pre-fix behaviour) therefore recorded 0/N vision accuracy for a
    model that answered perfectly, while a model that ignored the
    instruction and rambled would have scored 1/N -- the metric was
    inverted against correct tool-calling behaviour.  Both lanes share
    this one definition so they cannot drift apart.
    """
    parts: List[str] = []
    if isinstance(tool_args, dict):
        for key in REQUIRED_TOOL_ARGS:          # = ["object", "color"]
            value = tool_args.get(key)
            if isinstance(value, str):
                parts.append(value)
    if prose:
        parts.append(prose)
    return " ".join(parts).lower()


def _rest_run(candidate: Dict[str, Any], scene: Dict[str, Any],
              image: Optional[Dict[str, Any]], prices: Dict[str, Any],
              timeout_s: float, stub_base: Optional[str],
              media_override: Optional[str]) -> Dict[str, Any]:
    """One REST decision turn (one chat call, fresh client — the
    deployed transcript lane's per-turn unit)."""
    from brain.agent.llm_cloud import (  # lazy: --list needs zero deps
        AnthropicMessagesClient,
        ChatTransportError,
        CloudLLMClient,
    )

    base = stub_base or candidate["base_url"]
    key_env = candidate["key_env"]
    transport, captured = _transport_factory()
    common = dict(model=candidate["model"], api_key_env=key_env,
                  timeout_s=timeout_s, _transport=transport)
    if candidate["wire"] == "anthropic-native":
        client = AnthropicMessagesClient(base_url=base, **common)
    else:
        # same shape as the deployed lane:
        # skeleton tools and prose fallback.
        client = CloudLLMClient(base_url=base, tool_mode="skeleton",
                                prose_fallback=True, temperature=0.2,
                                max_tokens=REST_MAX_TOKENS,
                                enable_thinking=False,
                                **common)

    messages = [
        {"role": "system", "content": "You are a scene-analysis agent. "
                                      "Answer the scene question and call "
                                      "report_scene."},
        {"role": "user", "content": scene["prompt"]},
    ]
    # veto rows are measured text-only, with no image on their wire.
    attach_image = image if (image is not None and not candidate["veto"]) else None
    media_type = (media_override or attach_image["media"]
                  if attach_image else "image/jpeg")
    record: Dict[str, Any] = {
        "run": None, "scene": scene["id"], "status": "error",
        "ttft_s": None, "e2e_s": None, "cost_usd": None,
        "usage": None, "tool_call": None, "vision_qa": None,
        "answer": None, "note": "",
    }
    t0 = time.monotonic()
    try:
        result = client.chat(messages, [TOOL_SPEC],
                             image_jpeg_b64=(attach_image["b64"] if attach_image else None),
                             image_media_type=media_type)
    except ChatTransportError as exc:
        message = str(exc)
        if re.search(r"\b429\b|rate.limit", message, re.I):
            status = "rate_limit"
        elif (re.search(r"timed out|timeout", message, re.I)
              or isinstance(exc.__cause__, TimeoutError)):
            # urllib timeout fired: socket.timeout is an OSError, so it
            # arrives wrapped in URLError -> ChatTransportError. still a
            # per-run timeout, still counted.
            status = "timeout"
        else:
            status = "transport_error"
        record["status"] = status
        record["note"] = message[:300]
        record["e2e_s"] = time.monotonic() - t0
        return record
    except Exception as exc:  # noqa: BLE001 - any client failure is a failed run
        message = str(exc)
        # client-side timeout can slip past the wrap.
        # same classification.
        if re.search(r"timed out|timeout", message, re.I) or isinstance(
                exc, TimeoutError):
            record["status"] = "timeout"
        else:
            record["status"] = "error"
        record["note"] = message[:300]
        record["e2e_s"] = time.monotonic() - t0
        return record
    record["e2e_s"] = time.monotonic() - t0
    # no streaming on this lane: TTFT is the whole round trip.
    record["ttft_s"] = record["e2e_s"]

    if result.dropped:
        record["status"] = "dropped"
        record["note"] = "single-flight drop (another call in flight)"
        return record
    if not result.text and not result.tool_calls:
        # empty response is never fine. truncated = completion budget ran
        # out; not truncated = the model returned nothing.
        record["status"] = "empty_response"
        record["note"] = (
            "the model returned neither prose nor a tool call"
            + (" (truncated: completion budget ran out)" if result.truncated
               else " (not truncated)")
        )
        return record
    record["status"] = "ok"

    # ---- usage: straight from the raw bodies this run's POSTs produced ----
    usage_bodies = [json.loads(b) for b in captured]
    usage = _sum_rest_usage(usage_bodies)
    record["usage"] = usage
    record["cost_usd"] = _rest_cost(candidate, usage, prices)

    # ---- tool-call correctness: well-formed and runs with no retry/veto ----
    calls = result.tool_calls or []
    if not calls:
        record["tool_call"] = "no_call"
        record["note"] += "no tool call; "
    else:
        call = calls[0]
        args = call.arguments if isinstance(call.arguments, dict) else {}
        well_formed = (call.name in VALID_TOOL_NAMES
                       and isinstance(call.arguments, dict))
        schema_ok = all(key in args for key in REQUIRED_TOOL_ARGS)
        if not well_formed:
            record["tool_call"] = "parse_fail"
            record["note"] += "malformed tool call (name/args); "
        elif not schema_ok:
            record["tool_call"] = "veto"
            record["note"] += "tool call vetoed (missing required args); "
        else:
            # a run with no retry and no veto:
            # the call counts as executed.
            record["tool_call"] = "ok"
            record["note"] += "tool call executed; "

    # ---- vision-QA accuracy on the fixed scene ----------------------------
    answer = (result.text or "").strip() or None
    record["answer"] = answer
    if candidate["veto"]:
        # unmeasurable: no image on this wire
        record["vision_qa"] = None
    else:
        first_args = None
        if calls and isinstance(calls[0].arguments, dict):
            first_args = calls[0].arguments
        lowered = _vision_scored_text(first_args, answer)
        record["vision_qa"] = int(
            any(token.lower() in lowered for token in scene["expected"])
        )
    return record


def _sum_rest_usage(bodies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Aggregate usage over the run's response bodies (an internal retry
    POST is still a billed part of the turn).  None when NO body carried
    usage — the honest 'Not reported' state (llm_cloud.py exposes no
    usage of its own; absent wire usage is absent, never guessed)."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
              "image_tokens": None, "text_input_tokens": None,
              "audio_input_tokens": None, "text_output_tokens": None,
              "audio_output_tokens": None, "cached_tokens": None,
              "entries": 0}
    any_usage = False
    for body in bodies:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            continue
        any_usage = True
        totals["entries"] += 1
        # two wire dialects: prompt_tokens / completion_tokens
        # (OpenAI-compat) vs input_tokens / output_tokens (Anthropic).
        # read both, assume nothing.
        totals["input_tokens"] += int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            _accum(totals, "image_tokens", details.get("image_tokens"))
            _accum(totals, "cached_tokens", details.get("cached_tokens"))
            _accum(totals, "audio_input_tokens", details.get("audio_tokens"))
            _accum(totals, "text_input_tokens", details.get("text_tokens"))
        details_out = usage.get("completion_tokens_details")
        if isinstance(details_out, dict):
            _accum(totals, "audio_output_tokens", details_out.get("audio_tokens"))
            _accum(totals, "text_output_tokens", details_out.get("text_tokens"))
    if not any_usage:
        return None
    return totals


def _accum(totals: Dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        n = int(value)
    except (TypeError, ValueError):
        return
    totals[key] = n if totals[key] is None else totals[key] + n


def _rest_cost(candidate: Dict[str, Any], usage: Optional[Dict[str, Any]],
               prices: Dict[str, Any]) -> Optional[float]:
    """$/turn from REAL usage × the recorded sidecar rates.  None (the
    honest 'Not reported') when usage is absent or the price entry is
    unverified — a number computed from an unverified price would be a
    fabrication."""
    if usage is None:
        return None
    entry = prices.get(candidate["model"])
    if not isinstance(entry, dict) or not entry.get("verified"):
        return None
    input_rate = _price(entry, "input_per_1k")
    output_rate = _price(entry, "output_per_1k")
    if input_rate is None or output_rate is None:
        return None
    cost = (usage["input_tokens"] * input_rate
            + usage["output_tokens"] * output_rate) / 1000.0
    return cost


# ---------------------------------------------------------------------------
# realtime runner
# client and websockets imported lazily
# ---------------------------------------------------------------------------


def _realtime_run(candidate: Dict[str, Any], scene: Dict[str, Any],
                  image: Optional[Dict[str, Any]], prices: Dict[str, Any],
                  timeout_s: float, stub_base: Optional[str],
                  audio_wav: Optional[str],
                  media_override: Optional[str]) -> Dict[str, Any]:
    """One Realtime decision turn.  LIVE-VERIFIED 2026-09-02 from the
    laptop against api.openai.com (gpt-realtime-2.1-mini): wss connect,
    session config, image attach and tool call all confirmed on the wire.
    The earlier "not live-verifiable, py_compile-checked only" note was
    what hid a NameError here — `prices` was referenced in the async core
    but never passed to it, so a SUCCESSFUL turn was recorded as a failed
    run (status flipped ok -> error by the generic handler).  py_compile
    cannot catch a free-variable NameError; only a real call can."""
    from voice.adapters import flatten_specs  # lazy import
    from voice.realtime import RealtimeAuthError, RealtimeClient  # lazy import

    base = stub_base or candidate["base_url"]
    if stub_base:
        # stub-gateway seam: realtime only speaks
        # ws(s), so normalize the http(s) probe URL.
        # REST keeps the http URL.
        base = base.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    record: Dict[str, Any] = {
        "run": None, "scene": scene["id"], "status": "error",
        "ttft_s": None, "e2e_s": None, "cost_usd": None,
        "usage": None, "tool_call": None, "vision_qa": None,
        "answer": None, "note": "",
    }
    try:
        asyncio.run(_realtime_async_run(
            candidate, scene, image, prices, timeout_s, base, audio_wav,
            media_override, record, flatten_specs, RealtimeClient,
            RealtimeAuthError))
    except asyncio.TimeoutError:
        record["status"] = "timeout"
        record["note"] = "realtime turn exceeded the per-run bound"
    except RealtimeAuthError as exc:
        record["status"] = "auth_error"
        record["note"] = str(exc)[:300]
    except Exception as exc:  # noqa: BLE001 - any lane failure is a failed run
        record["status"] = "error"
        record["note"] = str(exc)[:300]
    return record


async def _realtime_async_run(candidate, scene, image, prices, timeout_s, base,
                              audio_wav, media_override, record,
                              flatten_specs, RealtimeClient,
                              RealtimeAuthError):  # noqa: E999 (py3.9+)
    """The async core of one realtime run: one client, one connect, one
    decision turn (tool continuations capped), response.done usage
    captured from the wire."""
    client = RealtimeClient(
        base_url=base,
        model=candidate["model"],
        api_key=os.environ.get(candidate["key_env"], ""),
        tools=flatten_specs([TOOL_SPEC]),
    )
    done = asyncio.Event()
    opened = asyncio.Event()
    usage_accum: List[Dict[str, Any]] = []
    status = {"value": "completed"}
    first_output_ts: List[Optional[float]] = [None]
    tool_items: List[Dict[str, Any]] = []
    scored_args: List[Dict[str, Any]] = []
    prose: List[str] = []
    last_prose_ts: List[Optional[float]] = [None]

    def on_open(name: str, payload: Any) -> None:
        opened.set()

    def on_done(name: str, payload: Any) -> None:
        response = payload.get("response") or {}
        status["value"] = str(response.get("status") or "completed")
        usage = response.get("usage")
        if isinstance(usage, dict):
            usage_accum.append(usage)
        done.set()

    def on_output_item(name: str, payload: Any) -> None:
        if first_output_ts[0] is None:
            first_output_ts[0] = time.monotonic()
        item = payload.get("item") or {}
        if item.get("type") == "function_call":
            tool_items.append(item)
        elif item.get("type") == "message":
            content = item.get("content") or []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "").strip()
                    if text:
                        prose.append(text)
                        last_prose_ts[0] = time.monotonic()

    def on_audio_delta(name: str, payload: Any) -> None:
        if first_output_ts[0] is None:
            first_output_ts[0] = time.monotonic()

    def on_error(name: str, payload: Any) -> None:
        message = str(payload.get("message") or "")
        status["value"] = "error:" + message[:200]

    client.on_event("open", on_open)
    client.on_event("response.done", on_done)
    client.on_event("response.output_item.done", on_output_item)
    client.on_event("response.audio.delta", on_audio_delta)
    client.on_event("response.output_audio.delta", on_audio_delta)
    client.on_event("error", on_error)

    loop_task = asyncio.ensure_future(client.run_forever())
    task_done = asyncio.Event()

    def _on_task_done(_task: asyncio.Future) -> None:
        task_done.set()

    loop_task.add_done_callback(_on_task_done)
    t0 = time.monotonic()
    try:
        # race the open event against the connect task: an immediate
        # rejection (401/402/403/404) must show in ms, not after the whole
        # handshake bound (that one is for the retry ladder).
        open_wait = asyncio.ensure_future(opened.wait())
        done_wait = asyncio.ensure_future(task_done.wait())
        finished, _ = await asyncio.wait(
            {open_wait, done_wait}, return_when=asyncio.FIRST_COMPLETED,
            timeout=REALTIME_CONNECT_S)
        if done_wait in finished and not open_wait.done():
            exc = loop_task.exception()
            if exc is not None:
                raise exc
            record["status"] = "error"
            record["note"] = "realtime connect task ended before the socket opened"
            return
        if not open_wait.done():
            record["status"] = "connect_error"
            record["note"] = ("realtime socket never opened within the "
                              "%ds open bound" % REALTIME_CONNECT_S)
            return
        # ---- the actual decision turn ----------------------------------------
        if audio_wav:
            await client.send_audio(_wav_pcm16(audio_wav))
        elif image is not None and not candidate["veto"]:
            await client.send_image(image["b64"])
        if not audio_wav:
            await client.send_input_text(scene["prompt"])
        await client.create_response()
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
        # ---- tool continuations, same bound as the deployed turn -----
        for _ in range(MAX_TOOL_CONTINUATIONS):
            if not tool_items:
                break
            item = tool_items.pop(0)
            call = _parse_realtime_call(item)
            if call is None:
                record["tool_call"] = "parse_fail"
                continue
            name, call_id, args = call
            if name not in VALID_TOOL_NAMES or not isinstance(args, dict):
                record["tool_call"] = "parse_fail"
                continue
            if not all(key in args for key in REQUIRED_TOOL_ARGS):
                record["tool_call"] = "veto"
            else:
                if record["tool_call"] in (None, "no_call"):
                    record["tool_call"] = "ok"
                # keep the accepted call's fields: vision-QA scores
                # them, and the prompt asks for the colour there so
                # prose is usually empty.
                scored_args.append(args)
            done.clear()
            await client.send_function_call_output(
                call_id, json.dumps({"ok": True}))
            await client.create_response()
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        record["e2e_s"] = time.monotonic() - t0
        if first_output_ts[0] is not None:
            record["ttft_s"] = first_output_ts[0] - t0
        if status["value"].startswith("error:"):
            record["status"] = "error"
            record["note"] = status["value"]
            return
        if status["value"] == "failed":
            # wire terminal state (status 'failed'): the model rejected
            # the turn. still a failed run, still in the denominator.
            record["status"] = "error"
            record["note"] = "response.status == 'failed' on the wire"
            return
        if not usage_accum:
            # no usage on the wire: the honest 'Not reported' state.
            record["status"] = "ok"
            record["note"] += "usage Not reported on the wire; "
            return
        record["status"] = "ok"
        usage = _sum_realtime_usage(usage_accum)
        record["usage"] = usage
        record["cost_usd"] = _realtime_cost(candidate, usage, prices)
        if record["tool_call"] is None:
            record["tool_call"] = "no_call"
        answer = " ".join(prose).strip() or None
        record["answer"] = answer
        if candidate["veto"]:
            record["vision_qa"] = None
        else:
            lowered = _vision_scored_text(
                scored_args[0] if scored_args else None, answer)
            record["vision_qa"] = int(
                any(token.lower() in lowered for token in scene["expected"]))
    except asyncio.TimeoutError:
        record["status"] = "timeout"
        record["note"] = "realtime turn exceeded the per-run bound"
    finally:
        try:
            await client.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        loop_task.cancel()
        try:
            await loop_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass


def _parse_realtime_call(item: Dict[str, Any]):
    """A realtime function_call item -> (name, call_id, args) or None."""
    name = item.get("name")
    call_id = item.get("call_id")
    raw_args = item.get("arguments")
    if not isinstance(name, str) or not isinstance(raw_args, str):
        return None
    try:
        args = json.loads(raw_args)
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    return name, str(call_id or ""), args


def _wav_pcm16(path: str) -> bytes:
    """Strip a WAV header -> PCM16 mono payload (24 kHz expected).  The
    mic pump's byte stream is what the realtime lane uploads; a WAV file
    is the deterministic stand-in."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data.startswith(b"RIFF") or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file: %s" % path)
    # find the 'data' chunk
    pos = 12
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if cid == b"data":
            return data[pos + 8: pos + 8 + size]
        pos += 8 + size + (size & 1)
    raise ValueError("WAV has no data chunk: %s" % path)


def _sum_realtime_usage(usage_accum: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the response.done usage payloads of the turn's
    responses.  The audio/text splits are read from ``*_token_details``
    exactly as the wire reports them."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
              "image_tokens": None, "text_input_tokens": None,
              "audio_input_tokens": None, "text_output_tokens": None,
              "audio_output_tokens": None, "cached_tokens": None,
              "entries": 0}
    for usage in usage_accum:
        totals["entries"] += 1
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)
        details = usage.get("input_token_details")
        if isinstance(details, dict):
            _accum(totals, "audio_input_tokens", details.get("audio_tokens"))
            _accum(totals, "text_input_tokens", details.get("text_tokens"))
            _accum(totals, "cached_tokens", details.get("cached_tokens"))
        details_out = usage.get("output_token_details")
        if isinstance(details_out, dict):
            _accum(totals, "audio_output_tokens", details_out.get("audio_tokens"))
            _accum(totals, "text_output_tokens", details_out.get("text_tokens"))
    return totals


def _realtime_cost(candidate: Dict[str, Any], usage: Dict[str, Any],
                   prices: Dict[str, Any]) -> Optional[float]:
    """$/turn from realtime usage (audio/text token splits × the recorded
    audio/text rates).  None (honest 'Not reported') when the price entry
    is unverified or the needed split/rate is absent — never a guessed
    number."""
    entry = prices.get(candidate["model"])
    if not isinstance(entry, dict) or not entry.get("verified"):
        return None
    audio_in = _price(entry, "audio_input_per_1k")
    audio_out = _price(entry, "audio_output_per_1k")
    text_in = _price(entry, "text_input_per_1k")
    text_out = _price(entry, "text_output_per_1k")
    has_splits = (usage["audio_input_tokens"] is not None
                  and usage["text_input_tokens"] is not None
                  and usage["audio_output_tokens"] is not None
                  and usage["text_output_tokens"] is not None)
    if has_splits:
        if None in (audio_in, audio_out, text_in, text_out):
            return None
        cost = (usage["text_input_tokens"] * text_in
                + usage["audio_input_tokens"] * audio_in
                + usage["text_output_tokens"] * text_out
                + usage["audio_output_tokens"] * audio_out) / 1000.0
        return cost
    # no audio/text split on the wire: only a
    # flat rate could price it, and none is
    # recorded -> 'Not reported'.
    return None


# ---------------------------------------------------------------------------
# config-seam check (profile-delta)
# ---------------------------------------------------------------------------


def _check_config_seam(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the candidate's profile-delta against the real loader:
    the deploy pin (brain.lane + model) must LOAD.  Degrades to
    'Not reported' outside the repo (lazy import — --list never gets
    here)."""
    try:
        from brain.config_loader import load as _cfg_load  # lazy import
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "ok": None,
                "note": "config_loader unimportable: %s" % exc}
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, "config.yaml")
    if not os.path.isfile(cfg_path):
        return {"checked": False, "ok": None,
                "note": "config.yaml not found at %s — run from the repo" % cfg_path}
    # deployed profile per lane: rest/transcript uses the cloud profile,
    # realtime uses the voice profile.
    profile = "voice" if candidate["lane"] == "realtime" else "cloud"
    try:
        cfg = _cfg_load(cfg_path, profile=profile,
                        overrides=candidate["profile_delta"])
    except Exception as exc:  # noqa: BLE001
        return {"checked": True, "ok": False,
                "note": "profile-delta rejected by config_loader (%s): %s"
                        % (profile, exc)}
    if candidate["lane"] == "realtime":
        ok = cfg.realtime.model == candidate["model"]
        note = "realtime.model == %r" % cfg.realtime.model
    else:
        ok = cfg.rest.model == candidate["model"]
        note = "rest.model == %r" % cfg.rest.model
    return {"checked": True, "ok": ok, "note": note}


def _check_budget_parity() -> Dict[str, Any]:
    """B5 honesty pin: the bench must rank with the SAME completion budget
    the deployed transcript lane ships (config.cloud.yaml
    rest.max_tokens).  #120 measured 16/20 truncation of deepseek-vision
    (a 512-token reference skeleton + reasoning) at the 256 ceiling — the
    re-run above is at REST_MAX_TOKENS, so a shipped config that reverts
    to 256 would silently rank a different game.  Degrades to 'Not
    reported' outside the repo, same as _check_config_seam."""
    try:
        from brain.config_loader import load as _cfg_load  # lazy import
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "ok": None,
                "note": "config_loader unimportable: %s" % exc}
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(repo_root, "config.yaml")
    if not os.path.isfile(cfg_path):
        return {"checked": False, "ok": None,
                "note": "config.yaml not found at %s — run from the repo"
                        % cfg_path}
    try:
        cfg = _cfg_load(cfg_path, profile="cloud")
    except Exception as exc:  # noqa: BLE001
        return {"checked": True, "ok": False,
                "note": "cloud profile rejected by config_loader: %s" % exc}
    shipped = getattr(cfg.rest, "max_tokens", None)
    return {"checked": True, "ok": shipped == REST_MAX_TOKENS,
            "note": "rest.max_tokens == %r (config.cloud.yaml)" % shipped}


# ---------------------------------------------------------------------------
# runner plumbing
# ---------------------------------------------------------------------------


def _run_one_candidate(candidate: Dict[str, Any], runs: int, timeout: float,
                       prices: Dict[str, Any], prices_path: str,
                       stub_base: Optional[str], image_dir: Optional[str],
                       audio_wav: Optional[str]) -> Dict[str, Any]:
    """N runs of one candidate.  Every attempted run gets a row; a failed
    run is COUNTED in the reliability denominator, never retried into
    success."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)  # so lazy brain/voice imports resolve

    runs_out: List[Dict[str, Any]] = []
    for i in range(runs):
        scene = SCENES[i % len(SCENES)]
        image = scene_image(scene, image_dir) if image_dir else scene_image(scene, None)
        if candidate["lane"] == "realtime":
            record = _realtime_run(candidate, scene, image, prices, timeout,
                                   stub_base, audio_wav, None)
        else:
            record = _rest_run(candidate, scene, image, prices, timeout,
                               stub_base, None)
        record["run"] = i
        runs_out.append(record)

    return _summarize(candidate, runs_out, prices, prices_path, stub_base)


def _summarize(candidate, runs_out, prices, prices_path,
               stub_base) -> Dict[str, Any]:
    ok = [r for r in runs_out if r["status"] == "ok"]
    failed = [r for r in runs_out if r["status"] != "ok"]
    fail_reasons: Dict[str, int] = {}
    for r in failed:
        fail_reasons[r["status"]] = fail_reasons.get(r["status"], 0) + 1

    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    e2e = [r["e2e_s"] for r in ok if r["e2e_s"] is not None]
    costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
    tool_ok = sum(1 for r in ok if r["tool_call"] == "ok")
    tool_no_call = sum(1 for r in ok if r["tool_call"] == "no_call")
    tool_failed = sum(1 for r in ok if r["tool_call"] not in ("ok", "no_call", None))
    vision_measured = [r for r in ok if r["vision_qa"] is not None]
    vision_correct = sum(1 for r in vision_measured if r["vision_qa"] == 1)

    usage_agg: Dict[str, Any] = {}
    for r in ok:
        usage = r.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            usage_agg[key] = usage_agg.get(key, 0) + int(usage.get(key) or 0)
        for key in ("image_tokens", "text_input_tokens", "audio_input_tokens",
                    "text_output_tokens", "audio_output_tokens", "cached_tokens"):
            value = usage.get(key)
            if value is not None:
                usage_agg[key] = usage_agg.get(key, 0) + int(value)

    entry = prices.get(candidate["model"])
    price_state = {
        "path": prices_path,
        "verified": bool(isinstance(entry, dict) and entry.get("verified")),
    }

    summary: Dict[str, Any] = {
        "candidate": candidate["id"],
        "model": candidate["model"],
        "lane": candidate["lane"],
        "wire": candidate["wire"],
        "key_env": candidate["key_env"],
        "veto": candidate["veto"],
        "baseline": candidate["baseline"],
        "note": candidate["note"],
        "stub_gateway": stub_base if stub_base else None,
        "runs_requested": len(runs_out),
        "runs_total": len(runs_out),
        "runs_succeeded": len(ok),
        "runs_failed": len(failed),
        "fail_reasons": fail_reasons,
        "rate_limits": fail_reasons.get("rate_limit", 0),
        # REST: 0 by construction (urllib is one-shot, no reconnect code).
        # realtime: the retry ladder (1,2,4,8,8,8,8,8 s) is internal to the
        # client and nothing public observes it, so a count would be guesswork.
        "reconnects": (0 if candidate["lane"] == "rest" else NOT_REPORTED),
        "ttft_s": {"measured_on": len(ttft), "of": len(ok),
                   "avg": _fmt(_mean(ttft)), "p95": _fmt(_p95(ttft))},
        "e2e_s": {"measured_on": len(e2e), "of": len(ok),
                  "avg": _fmt(_mean(e2e)), "p95": _fmt(_p95(e2e))},
        "cost_per_turn_usd": _fmt(_mean(costs)),
        "cost_per_turn_p95_usd": _fmt(_p95(costs)),
        "cost_measured_on": len(costs),
        "cost_of": len(ok),
        "cost_basis_tokens": usage_agg,
        "tool_call": {"ok": tool_ok, "failed": tool_failed,
                      "no_call": tool_no_call, "of": len(ok)},
        "vision_qa_accuracy": (
            _fmt(_mean([float(r["vision_qa"]) for r in vision_measured]))
            if vision_measured else NOT_REPORTED),
        "vision_qa": {"correct": vision_correct, "measured": len(vision_measured),
                      "of": len(ok)},
        "config_seam": _check_config_seam(candidate),
        "budget_parity": _check_budget_parity(),
        "prices": price_state,
        "runs": runs_out,
    }
    if candidate["veto"]:
        summary["vision_qa_not_reported_reason"] = (
            "veto-flagged text-only candidate (no image part on this wire): "
            + candidate["veto"])
    if summary["cost_per_turn_usd"] == NOT_REPORTED:
        summary["cost_not_reported_reason"] = (
            "no verified per-token rate recorded for %r in %s (recorded "
            "2026-08-31) — re-verify day-of against the provider price "
            "page and fill the entry (docs/BAKEOFF.md re-verify "
            "instruction)" % (candidate["model"], prices_path))
    if not ttft:
        summary["ttft_s"]["avg"] = NOT_REPORTED
        summary["ttft_s"]["p95"] = NOT_REPORTED
        summary["ttft_not_reported_reason"] = "no successful runs measured"
    if not e2e:
        summary["e2e_s"]["avg"] = NOT_REPORTED
        summary["e2e_s"]["p95"] = NOT_REPORTED
        summary["e2e_not_reported_reason"] = "no successful runs measured"
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_manifest(fmt: str) -> int:
    rows = [CANDIDATES[key] for key in sorted(CANDIDATES)]
    if fmt == "json":
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    if fmt == "csv":
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["id", "lane", "wire", "model", "key_env",
                         "veto", "baseline", "note"])
        for c in rows:
            writer.writerow([c["id"], c["lane"], c["wire"], c["model"],
                             c["key_env"], c["veto"], str(c["baseline"]),
                             c["note"]])
        sys.stdout.write(out.getvalue())
        return 0
    width = max(len(c["id"]) for c in rows) + 2
    print("OpenBot v4 model bake-off — 11-candidate manifest "
          "(plan section 6; %d rows)" % len(rows))
    print("=" * 118)
    for c in rows:
        veto = ("VETO: " + c["veto"]) if c["veto"] else "vision-capable"
        flag = " [BASELINE]" if c["baseline"] else ""
        print("%-*s %-8s %-17s %-24s %s%s" % (
            width, c["id"], c["lane"], c["wire"], c["key_env"], veto, flag))
        print("%*s %s" % (width, "", c["note"]))
    print("=" * 118)
    print("key env vars are read from the environment at run time — "
          "never printed, never passed in args")
    return 0


def _write_output(summary: Dict[str, Any], out_path: str, fmt: str) -> None:
    if fmt == "json":
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")
    else:
        with open(out_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run", "scene", "status", "ttft_s", "e2e_s",
                             "cost_usd", "tool_call", "vision_qa", "answer",
                             "note"])
            for r in summary["runs"]:
                writer.writerow([
                    r["run"], r["scene"], r["status"],
                    _fmt(r["ttft_s"], " s") if r["ttft_s"] is not None
                        else NOT_REPORTED,
                    _fmt(r["e2e_s"], " s") if r["e2e_s"] is not None
                        else NOT_REPORTED,
                    _fmt(r["cost_usd"]) if r["cost_usd"] is not None
                        else NOT_REPORTED,
                    r["tool_call"] if r["tool_call"] is not None
                        else NOT_REPORTED,
                    r["vision_qa"] if r["vision_qa"] is not None
                        else NOT_REPORTED,
                    r["answer"] if r["answer"] is not None else NOT_REPORTED,
                    r["note"].strip(),
                ])


def _print_summary(summary: Dict[str, Any]) -> None:
    c = summary["candidate"]
    print("")
    print("candidate %s (%s lane, wire %s)%s" % (
        c, summary["lane"], summary["wire"],
        " — stub gateway " + summary["stub_gateway"] if summary["stub_gateway"]
        else ""))
    print("  runs: %d total = %d succeeded + %d failed "
          "(fail reasons: %s; rate limits %d; reconnects %s)" % (
              summary["runs_total"], summary["runs_succeeded"],
              summary["runs_failed"],
              summary["fail_reasons"] or "{}",
              summary["rate_limits"], summary["reconnects"]))
    ttft = summary["ttft_s"]
    e2e = summary["e2e_s"]
    print("  TTFT:  avg %s, p95 %s (measured on %d/%d runs)"
          % (ttft["avg"], ttft["p95"], ttft["measured_on"], ttft["of"]))
    print("  E2E:   avg %s, p95 %s (measured on %d/%d runs)"
          % (e2e["avg"], e2e["p95"], e2e["measured_on"], e2e["of"]))
    print("  cost/turn: avg %s USD, p95 %s USD (measured on %d/%d runs; "
          "token basis %s)" % (
              summary["cost_per_turn_usd"], summary["cost_per_turn_p95_usd"],
              summary["cost_measured_on"], summary["cost_of"],
              json.dumps(summary["cost_basis_tokens"])))
    tc = summary["tool_call"]
    print("  tool-call correctness: %d ok / %d failed / %d no-call (of %d)"
          % (tc["ok"], tc["failed"], tc["no_call"], tc["of"]))
    print("  vision-QA accuracy: %s (%d/%d measured)" % (
        summary["vision_qa_accuracy"],
        summary["vision_qa"]["correct"], summary["vision_qa"]["measured"]))
    seam = summary["config_seam"]
    if seam.get("checked"):
        print("  config seam (profile-delta): %s — %s" % (
            "ok" if seam.get("ok") else "REJECTED", seam.get("note", "")))
    else:
        print("  config seam: %s (%s)" % (NOT_REPORTED, seam.get("note", "")))
    parity = summary.get("budget_parity", {})
    if parity.get("checked"):
        print("  budget parity (bench == shipped): %s — %s" % (
            "ok" if parity.get("ok") else "MISMATCH",
            parity.get("note", "")))
    else:
        print("  budget parity: %s (%s)" % (
            NOT_REPORTED, parity.get("note", "")))
    if summary["cost_per_turn_usd"] == NOT_REPORTED:
        print("  NOTE: %s" % summary.get("cost_not_reported_reason", ""))
    if summary.get("veto"):
        print("  NOTE: veto-flagged — %s" % summary.get(
            "vision_qa_not_reported_reason", summary["veto"]))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="model_bakeoff.py",
        description="OpenBot v4 model bake-off harness (plan section 6): "
                    "measures every vision-capable candidate across both "
                    "brain lanes on the same fixed scene set and tool "
                    "protocol, ranked by $/turn at equal-or-better "
                    "accuracy.  Honesty law: usage/timing read from the "
                    "wire, prices from deploy/prices.json, failed runs "
                    "counted in the reliability denominator.",
    )
    parser.add_argument("--list", action="store_true",
                        help="print the 11-candidate manifest and exit "
                             "(zero dependencies)")
    parser.add_argument("--candidate", default=None,
                        help="candidate id from --list, or 'all'")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help="runs per candidate (default %d)" % DEFAULT_RUNS)
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-run bound in seconds (default: lane "
                             "default — rest 120, realtime 60)")
    parser.add_argument("--out", default=None,
                        help="results file (.json or .csv by extension)")
    parser.add_argument("--format", choices=("json", "csv"), default=None,
                        help="output format (default: by --out extension, "
                             "else json)")
    parser.add_argument("--prices", default=None,
                        help="price sidecar path (default: deploy/prices.json)")
    parser.add_argument("--stub-gateway", default=None,
                        help="dev/probe seam: override the REST base_url "
                             "(also BAKEOFF_STUB_GATEWAY env); echoed in "
                             "output, never a secret")
    parser.add_argument("--image-dir", default=None,
                        help="real scene photos (scene-a.jpg ... scene-e.jpg); "
                             "default: deterministic synthetic images")
    parser.add_argument("--audio", default=None,
                        help="realtime lane: PCM16 24 kHz mono WAV to "
                             "upload instead of text input")
    args = parser.parse_args(argv)
    if args.list and args.candidate:
        parser.error("--list and --candidate are mutually exclusive")
    if args.candidate is None and not args.list:
        parser.error("either --list or --candidate <id|all> is required")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if args.list:
        fmt = args.format or "table"
        return _print_manifest(fmt)

    stub_base = args.stub_gateway or os.environ.get("BAKEOFF_STUB_GATEWAY") or None
    if args.format is None:
        fmt = "csv" if (args.out or "").endswith(".csv") else "json"
    else:
        fmt = args.format
    if args.out is None:
        parser = argparse.ArgumentParser()
        parser.error("--out is required when running a candidate")
        return 2
    if fmt not in ("json", "csv"):
        sys.stderr.write("model_bakeoff: unsupported format %r\n" % fmt)
        return 1

    prices_path = args.prices or default_prices_path()
    try:
        prices = load_prices(prices_path)
    except (OSError, ValueError) as exc:
        sys.stderr.write("model_bakeoff: REFUSAL: prices sidecar failed: %s\n" % exc)
        return 2

    if args.candidate == "all":
        ids = sorted(CANDIDATES)
    else:
        if args.candidate not in CANDIDATES:
            sys.stderr.write(
                "model_bakeoff: REFUSAL: unknown candidate %r (see --list)\n"
                % args.candidate)
            return 2
        ids = [args.candidate]

    results: List[Dict[str, Any]] = []
    for cid in ids:
        candidate = CANDIDATES[cid]
        timeout = args.timeout or (
            REALTIME_TIMEOUT_S if candidate["lane"] == "realtime"
            else REST_TIMEOUT_S)
        summary = _run_one_candidate(candidate, args.runs, timeout, prices,
                                     prices_path, stub_base, args.image_dir,
                                     args.audio)
        _print_summary(summary)
        results.append(summary)

    if len(results) == 1:
        document = results[0]
    else:
        document = {"candidates": results, "count": len(results)}

    try:
        _write_output(document, args.out, fmt)
    except OSError as exc:
        sys.stderr.write("model_bakeoff: could not write %s: %s\n"
                         % (args.out, exc))
        return 1
    sys.stdout.write("model_bakeoff: wrote %s (%s, %d runs per candidate)\n"
                     % (args.out, fmt, args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
