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


