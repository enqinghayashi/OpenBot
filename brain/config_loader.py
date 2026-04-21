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


def strip_inline_comment(text: str) -> str:
    """Strip a trailing inline comment from a raw value.

    '0.20   # workspace floor z' -> '0.20'.  A leading-quoted value
    ('...' or "...") keeps any ' #' that lives inside the quotes:
    "'a # b'  # c" -> "'a # b'".  Quotes are only honored at the START of
    the value, so a stray apostrophe inside a bare string ('it's fine')
    never swallows the comment.  A value that IS a comment (starts with
    '#' — e.g. the rest of 'key: # note') strips to ''.
    """
    value = text.strip()
    if not value:
        return text
    if value.startswith("#"):
        return ""
    if value[0] in ("'", '"'):
        quote = value[0]
        j = 1
        while j < len(value):
            if value[j] == quote:
                if j + 1 < len(value) and value[j + 1] == quote:
                    j += 2  # a doubled quote
                            # is an escaped quote
                    continue
                break
            j += 1
        end = j + 1 if j < len(value) else len(value)
        head = value[:end]
        tail = value[end:]
        m = _COMMENT_RE.search(tail)
        return (head + tail[: m.start()]) if m else head
    m = _COMMENT_RE.search(value)
    return value[: m.start()] if m else value


class ConfigError(ValueError):
    """Raised for config values that are structurally valid but cannot RUN.

    Two gates raise this today (round 4; v4-trimmed 2026-08-31 — the
    stuart placeholder-host gate died with the superseded stuart tier):
      1. profile 'cloud' with gateway.transport 'real' and an unfilled
         placeholder gateway.host ('', 'TODO', 'TODO-IP', ...) — the D0
         preflight fills the address; loading a TODO host would be a
         false 'config ok' before a launch that cannot connect;
      2. a non-bool value for the safety-adjacent
         embodiment.perception.silence_means_empty key (see _as_bool) —
         silence_means_empty MUST stay a real bool, never coerced.

    The message always names the offending key.  Subclasses ValueError so
    every pre-existing 'except ValueError' caller keeps working.
    """


def _parse_scalar(text: str) -> Any:
    t = text.strip().strip('"').strip("'")
    if t == "":
        return ""
    low = t.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def _parse_list(text: str) -> List[Any]:
    inner = text.strip()
    if inner.startswith("["):
        if not inner.endswith("]"):
            # line-oriented: a flow list must close on one line,
            # or the open '[' is kept and the continuation lines
            # are dropped, leaving a corrupt first entry
            raise ConfigError(
                "config error: a flow list must open and close on one line "
                "— got %r. This parser is line-oriented; put the whole "
                "list on a single line (a list wrapped across lines would "
                "otherwise be truncated silently, keeping the '[' and "
                "dropping every continuation line)." % inner
            )
        inner = inner[1:-1]
    items = [x for x in (p.strip() for p in inner.split(",")) if x]
    return [_parse_scalar(x) for x in items]


