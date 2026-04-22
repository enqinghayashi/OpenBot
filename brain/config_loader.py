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


def _has_deeper(lines: List[str], start: int, parent_indent: int) -> bool:
    """True when an indented (non-comment) line follows at `start` — i.e.
    the current key headers a nested block rather than a null value."""
    j = start
    while j < len(lines):
        nxt = lines[j]
        nxt_stripped = nxt.lstrip()
        if nxt_stripped == "" or nxt_stripped.startswith("#"):
            j += 1
            continue
        return (len(nxt) - len(nxt_stripped)) > parent_indent
    return False


def _collect_nested(lines: List[str], start: int, parent_indent: int) -> "tuple[List[str], int]":
    """Collect the deeper-indented lines of a nested block; returns
    (block lines, index just past the block)."""
    j = start
    nested: List[str] = []
    while j < len(lines):
        nxt = lines[j]
        nxt_stripped = nxt.lstrip()
        if nxt_stripped == "" or nxt_stripped.startswith("#"):
            j += 1
            continue
        nxt_indent = len(nxt) - len(nxt_stripped)
        if nxt_indent <= parent_indent:
            break
        nested.append(nxt.rstrip("\n"))
        j += 1
    return nested, j


def _build_nested(lines: List[str], indent: Optional[int]) -> Dict[str, Any]:
    """Recursively build a dict from indented `key: value` lines."""
    out: Dict[str, Any] = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.lstrip()
        if stripped == "" or stripped.startswith("#"):
            i += 1
            continue
        cur_indent = len(raw) - len(stripped)
        if indent is not None and cur_indent <= indent:
            break  # the caller's block ends at this dedent
        key, sep, rest_raw = stripped.partition(":")
        key = key.strip()
        if not sep or not key:
            i += 1
            continue
        rest_stripped = rest_raw.strip()
        if rest_stripped == "" or rest_stripped.startswith("#"):
            # empty rest is a block header or an absent key;
            # a commented-out value with no deeper lines is absent,
            # so the consumer default applies
            if rest_stripped.startswith("#") and not _has_deeper(lines, i + 1, cur_indent):
                i += 1
                continue
            nested, j = _collect_nested(lines, i + 1, cur_indent)
            out[key] = _build_nested(nested, indent=cur_indent)
            i = j
        else:
            rest = strip_inline_comment(rest_raw).strip()
            if rest == "":
                i += 1  # no value after comment strip
                continue
            if rest.startswith("["):
                out[key] = _parse_list(rest)
            else:
                out[key] = _parse_scalar(rest)
            i += 1
    return out


def _parse_text(text: str) -> Dict[str, Any]:
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    return _build_nested(lines, indent=None)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Mappings merge recursively; scalars/lists from `overrides` replace."""
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _profile_path(path: str, profile: str) -> str:
    return os.path.join(os.path.dirname(path), "config." + profile + ".yaml")


# ---------------------------------------------------------------------------
# Section schemas
# one frozen dataclass per config section; defaults match the
# consumers, so a missing key falls back to the same value, and
# from_section rejects unknown keys and wrong types
# ---------------------------------------------------------------------------


def _section_dict(section: Optional[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ValueError(
            f"config error: section {name!r} must be a mapping, got "
            f"{type(section).__name__}"
        )
    return dict(section)


def _check_keys(d: Dict[str, Any], allowed: set, name: str) -> None:
    unknown = sorted(k for k in d if k not in allowed)
    if unknown:
        raise ValueError(
            f"config error: unknown key(s) {unknown} in section {name!r} "
            f"(typo?) -- allowed keys: {sorted(allowed)}"
        )


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"config error: {name} must be an int, got bool {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"config error: {name} must be an int, got {value!r}") from None


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"config error: {name} must be a number, got bool {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"config error: {name} must be a number, got {value!r}"
        ) from None


def _as_str(value: Any, name: str) -> str:
    return str(value)


def _as_bool(value: Any, name: str) -> bool:
    """Bool coercion that NEVER guesses and NEVER inverts: only a real
    Python bool is accepted.

    In particular numeric 0/1 and the string 'false' are hard
    ConfigErrors — bool(1) or bool('false') == True would silently flip
    a safety-adjacent flag like embodiment.perception.silence_means_empty.
    (The YAML-subset parser already maps true/false/yes/no to Python
    bools, so every legit file value passes; this guard is what a
    hand-built overrides mapping or a future YAML dialect hits.)"""
    if isinstance(value, bool):
        return value
    raise ConfigError(
        f"config error: {name} must be a bool, got {value!r} — bool "
        f"coercion never guesses (a numeric 0/1 or the string 'false' "
        f"could silently invert a safety-adjacent flag)"
    )


def _as_gpu_layers(value: Any, name: str) -> int | str:
    """n_gpu_layers: an int, or the string 'all' (CUDA full offload)."""
    if isinstance(value, bool):
        raise ValueError(
            f"config error: {name} must be an int or 'all', got bool {value!r}"
        )
    if isinstance(value, str):
        if value == "all":
            return value
        try:
            return int(value)
        except ValueError:
            raise ValueError(
                f"config error: {name} must be an int or 'all', got {value!r}"
            ) from None
    if isinstance(value, (int, float)):
        return int(value)
    raise ValueError(
        f"config error: {name} must be an int or 'all', got {value!r}"
    )
