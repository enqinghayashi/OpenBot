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


