"""
brain/config_loader.py — the ONE shared stdlib config parser and profile
loader for OpenBot.  No PyYAML (brain/ cores carry no third-party deps).

This module is the single place that:
  1. parses config.yaml (a documented YAML SUBSET — see below), and
  2. strips trailing inline comments ('key: value  # comment') at the
     VALUE level.  The comment-strip fix lives HERE, once; every module
     that reads config-shaped values goes through it
     (embodiment.from_config_file delegates to load()).

API
---
load(path, profile=None, overrides=None) -> ConfigDict
    Parse `path` (the base file).  The ACTIVE profile is the `profile:`
    key in the base file when `profile` is None, else the given profile.
    For any active profile other than 'laptop' the file
    `config.<profile>.yaml` next to the base file is deep-merged on top
    (mappings merge recursively; scalars/lists from the override
    replace).  A MISSING profile override file raises FileNotFoundError —
    silently running the laptop defaults on a non-laptop profile is not
    acceptable.  The ACTIVE profile is validated to be one of
    'laptop'|'cloud'|'voice'|'offline' (a typo'd profile name raises
    ValueError).  v4 (2026-08-31) SUPERSEDES the stuart tier: the
    'stuart' / 'stuart-cuda' profiles and the local-LLM ladder are
    REMOVED from the allowlist — load(profile='stuart') raises a
    ValueError naming the replacement (config.voice.yaml, profile
    'voice' = the cloud Realtime brain; config.offline.yaml, profile
    'offline' = the regex brain, no network).  'cloud' = the Plan-B R4
    off-robot LLM rung (config.cloud.yaml), still active.  'voice' = the
    cloud Realtime WebSocket brain + local voice lane (config.voice.yaml);
    'offline' = the fully offline regex lane (config.offline.yaml — no
    API key, no network, ever).
    `overrides` (optional) is a nested mapping of section overrides
    (e.g. {"llm": {"n_ctx": 512}}) deep-merged LAST — explicit caller
    intent (tests, runtime knobs) — so it wins over the profile file
    (the v4 clamp pass is GONE with the superseded stuart tier — there
    is no post-merge clamp to fight).  It can never change the resolved
    profile.

    RUNNABILITY GATE (round 4; v4-trimmed 2026-08-31): after the merge a
    post-check (_validate_runnable) rejects configs that LOAD but cannot
    RUN:
        - profile 'cloud' with gateway.transport 'real' and an unfilled
          placeholder gateway.host ('', 'TODO', 'TODO-IP', ...) raises
          ConfigError naming gateway.host — the D0 preflight fills the
          address; launching a TODO host would be a false 'config ok'
          before a connection that cannot exist.  The voice/offline
          profiles ship 127.0.0.1:3900 (the on-robot gateway) and need
          NO API key; there is deliberately NO key gate — the voice
          lane's cloud realtime session reads OPENAI_API_KEY at runtime
          (a runtime failure, not a config error), and offline never
          needs one.
        - a non-bool value for the safety-adjacent
          embodiment.perception.silence_means_empty key raises
          ConfigError (never coerced — bool(1) or bool('false') would
          silently flip the flag).
    ConfigError subclasses ValueError, so every pre-existing
    'except ValueError' caller keeps working.

    The return value is a ConfigDict: a dict (so legacy .get() /
    cfg['llm'] access keeps working — embodiment.from_config_file reads
    cfg.get('embodiment')) that ALSO exposes every section as a
    validated READ-ONLY (frozen) dataclass attribute:
        cfg.llm        -> LlmConfig
        cfg.gateway    -> GatewayConfig
        cfg.skills     -> SkillsConfig
        cfg.agent      -> AgentConfig
        cfg.embodiment -> EmbodimentConfig
        cfg.brain      -> BrainConfig      (v4 lane switch)
        cfg.realtime   -> RealtimeConfig   (v4, brain.lane 'realtime')
        cfg.rest       -> RestConfig       (v4, brain.lane 'transcript')
        cfg.voice      -> VoiceConfig      (v4 voice lane)
        cfg.vision     -> VisionConfig     (v4 camera lane)
        cfg.profile    -> str (the resolved ACTIVE profile)
    Section construction is schema-strict: an UNKNOWN key in any section
    raises ValueError naming the section and the offending key (typo
    safety — 'n_cxt' can never silently fall back to a default), and
    value types are validated (an int field rejects 'abc'; a bool field
    rejects 'false' as a string — a bool('false') == True flip would be
    dangerous).

strip_inline_comment(text) -> str
    The value-level trailing-comment rule: a '#' preceded by whitespace
    ends the value (the YAML rule); a '#' inside a leading-quoted string
    ('...' / "...") is literal.  Called by the parser on every value;
    also public so tests and any future config-shaped parser reuse
    exactly one implementation.

Supported YAML subset (exactly the shape config.yaml uses):
    - nested mappings by indentation (`key:` headers, 2-space bodies)
    - scalars: int / float / bool (true|false|yes|no) / bare string
    - inline float lists `[a, b, c]`
    - full-line comments and trailing inline comments after values
    - a value that is empty after comment-stripping means the key is
      ABSENT (the consumer's defaults apply — same convention as the
      documented 'missing keys fall back to defaults')
    - quoted strings keep any ' #' inside the quotes

Anything outside the subset is ignored (never guessed).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional

# a '#' preceded by whitespace starts a comment
# 'x#y' stays literal
_COMMENT_RE = re.compile(r"\s+#")


