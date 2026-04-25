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


def _as_backoffs(value: Any, name: str) -> List[int]:
    """reconnect_backoffs: a NON-EMPTY list of non-negative ints — either
    the inline `[a, b, c]` list form (what the YAML-subset parser
    resolves) or a comma-separated string ('1,2,4,8').  Every entry must
    be a real int: bool and float are REJECTED, never coerced
    (int(1.5) == 1 would silently reshape the retry ladder), and a
    negative backoff is a contract violation.  An empty list is rejected
    on BOTH branches too — a session with zero retries is not a config
    value."""
    items: List[Any]
    if isinstance(value, str):
        items = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(
            f"config error: {name} must be a list of ints or a "
            f"comma-separated string, got {value!r}"
        )
    if not items:
        raise ValueError(
            f"config error: {name} must be a non-empty list of ints, "
            f"got {value!r}"
        )
    out: List[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            # comma-separated strings still need an int parse;
            # bools and floats are rejected rather than coerced
            if isinstance(item, str):
                try:
                    n = int(item)
                except ValueError:
                    raise ValueError(
                        f"config error: {name} entries must be "
                        f"non-negative integers, got {item!r}"
                    ) from None
            else:
                raise ValueError(
                    f"config error: {name} entries must be non-negative "
                    f"integers, got {item!r}"
                )
        else:
            n = item
        if n < 0:
            raise ValueError(
                f"config error: {name} entries must be non-negative "
                f"integers, got {n}"
            )
        out.append(n)
    return out


@dataclass(frozen=True)
class LlmConfig:
    """The `llm:` segment.  Field `temp` maps from the yaml key
    `temperature` (the bare `temp` yaml key is accepted as an alias);
    `temperature` is also available as a read-only property alias.
    Defaults mirror brain/agent/llm.py's LlamaCppClient defaults.

    PLAN-B keys (2026-08-30, R4 cloud rung) — consumed by
    brain/agent/llm.py from_config's 'cloud' backend branch:
    base_url / model / api_key_env / timeout_s / tool_mode /
    enable_thinking.  tool_mode semantics (the profile ladder):
      - laptop (base config.yaml) default 'skeleton' — the Plan-B R0'
        recommendation (plain-text JSON skeleton + deterministic parse,
        measured 9-10/10 and 2-4x faster than json-mode, whose grammar
        constraint is the hard-schema Constraint Tax 91.5->48.0%
        executable at 1.5B);
      - stuart PINS 'json' in config.stuart.yaml — the D4 deployed
        json-mode run must not silently change when the base default
        moves to 'skeleton' (historical: the stuart profile is
        SUPERSEDED 2026-08-31 and can no longer load — see load());
      - cloud PINS 'skeleton' in config.cloud.yaml — the R4 rung."""

    backend: str = "mock"  # mock = stub
                           # real = llama-cpp; cloud = cloud API
    model_path: str = ""
    n_ctx: int = 4096
    n_gpu_layers: int | str = "all"  # 'all' (CUDA) or an int (0 = CPU-only)
    max_tokens: int = 256
    temp: float = 0.2
    base_url: str = ""  # OpenAI-compatible
                        # cloud endpoint
    model: str = ""  # cloud model id
    api_key_env: str = ""  # env var NAME
                           # never the key itself in config
    timeout_s: float = 12.0  # cloud/sidecar timeout
                             # cloud profile pins 120
    tool_mode: str = "skeleton"  # 'skeleton' | 'json'
                                 # skeleton = plain JSON + parse
    enable_thinking: bool = False  # thinking off
                                   # in every profile

    @property
    def temperature(self) -> float:
        """Read-only alias for `temp` (the yaml key name)."""
        return self.temp

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "LlmConfig":
        d = _section_dict(section, "llm")
        _check_keys(
            d,
            {"backend", "model_path", "n_ctx", "n_gpu_layers", "max_tokens",
             "temperature", "temp", "base_url", "model", "api_key_env",
             "timeout_s", "tool_mode", "enable_thinking"},
            "llm",
        )
        temp = d.get("temperature", d.get("temp", 0.2))
        tool_mode = d.get("tool_mode", "skeleton")
        if tool_mode not in ("skeleton", "json"):
            raise ValueError(
                f"config error: llm.tool_mode {tool_mode!r} must be "
                f"'skeleton' (Plan-B R0'/R4 plain-text JSON + deterministic "
                f"parse) or 'json' (D4 grammar-constrained json-mode)"
            )
        return cls(
            backend=_as_str(d.get("backend", "mock"), "llm.backend"),
            model_path=_as_str(d.get("model_path", ""), "llm.model_path"),
            n_ctx=_as_int(d.get("n_ctx", 4096), "llm.n_ctx"),
            n_gpu_layers=_as_gpu_layers(
                d.get("n_gpu_layers", "all"), "llm.n_gpu_layers"
            ),
            max_tokens=_as_int(d.get("max_tokens", 256), "llm.max_tokens"),
            temp=_as_float(temp, "llm.temperature"),
            base_url=_as_str(d.get("base_url", ""), "llm.base_url"),
            model=_as_str(d.get("model", ""), "llm.model"),
            api_key_env=_as_str(d.get("api_key_env", ""), "llm.api_key_env"),
            timeout_s=_as_float(d.get("timeout_s", 12.0), "llm.timeout_s"),
            tool_mode=tool_mode,
            enable_thinking=_as_bool(
                d.get("enable_thinking", False), "llm.enable_thinking"
            ),
        )


@dataclass(frozen=True)
class GatewayConfig:
    """The `gateway:` segment.  Host/port defaults are neutral placeholders
    (the gateway-side consumer lives in gateway/; the deployed config.yaml
    always carries the real endpoints)."""

    transport: str = "mock"  # mock = in-process
                             # real = socket to the gateway
    host: str = ""
    port: int = 0
    #: bind host
    #: server-side only; host above stays client-dial only
    #: 0.0.0.0 binds every interface; 127.0.0.1 hides the
    #: unauthenticated motion socket but blocks every LAN dial
    bind_host: str = "0.0.0.0"

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "GatewayConfig":
        d = _section_dict(section, "gateway")
        _check_keys(d, {"transport", "host", "port", "bind_host"}, "gateway")
        return cls(
            transport=_as_str(d.get("transport", "mock"), "gateway.transport"),
            host=_as_str(d.get("host", ""), "gateway.host"),
            port=_as_int(d.get("port", 0), "gateway.port"),
            bind_host=_as_str(d.get("bind_host", "0.0.0.0"),
                              "gateway.bind_host"),
        )


@dataclass(frozen=True)
class SkillsConfig:
    """The `skills:` segment (protocol endpoints).  Socket defaults are the
    STUART-VERIFIED platform endpoints (writer /tmp/merged_writer.sock,
    grip /tmp/grip_server.sock).  engine_cmd is DUAL-FORM: the dataclass
    DEFAULT (key absent) is the engine-off verb form; a file key = its spawn."""

    writer_sock: str = "/tmp/merged_writer.sock"
    grip_sock: str = "/tmp/grip_server.sock"
    #: engine sock
    #: must match the entrypoint's SOCK_PATH
    #: or connect-first degrades to spawn-always
    engine_sock: str = "/tmp/pick_engine.sock"
    engine_cmd: str = "pick x y z nx ny nz orientation height_m"

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "SkillsConfig":
        d = _section_dict(section, "skills")
        _check_keys(d, {"writer_sock", "grip_sock", "engine_sock", "engine_cmd"},
                    "skills")
        return cls(
            writer_sock=_as_str(
                d.get("writer_sock", "/tmp/merged_writer.sock"), "skills.writer_sock"
            ),
            grip_sock=_as_str(
                d.get("grip_sock", "/tmp/grip_server.sock"), "skills.grip_sock"
            ),
            engine_sock=_as_str(
                d.get("engine_sock", "/tmp/pick_engine.sock"), "skills.engine_sock"
            ),
            engine_cmd=_as_str(
                d.get("engine_cmd", "pick x y z nx ny nz orientation height_m"),
                "skills.engine_cmd",
            ),
        )


@dataclass(frozen=True)
class AgentConfig:
    """The `agent:` segment.  Defaults mirror G1Agent.__init__ defaults.

    RUNTIME CONSUMER (round-5 config-steering): every field is read by
    brain/agent/agent.py ``G1Agent.from_config`` and wired into the
    running agent (max_tool_calls / failure_budget /
    detection_freshness_s / llm_timeout_s / max_llm_attempts /
    max_tool_result_chars -> G1Agent.__init__; detection_freshness_s ->
    brain/hooks.py safety_checks_hook freshness budget via
    default_hooks(freshness_s=...))."""

    max_tool_calls: int = 25
    failure_budget: int = 3
    detection_freshness_s: float = 2.0
    llm_timeout_s: float = 12.0
    max_llm_attempts: int = 2
    max_tool_result_chars: int = 2000

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "AgentConfig":
        d = _section_dict(section, "agent")
        _check_keys(
            d,
            {"max_tool_calls", "failure_budget", "detection_freshness_s",
             "llm_timeout_s", "max_llm_attempts", "max_tool_result_chars"},
            "agent",
        )
        return cls(
            max_tool_calls=_as_int(d.get("max_tool_calls", 25), "agent.max_tool_calls"),
            failure_budget=_as_int(d.get("failure_budget", 3), "agent.failure_budget"),
            detection_freshness_s=_as_float(
                d.get("detection_freshness_s", 2.0), "agent.detection_freshness_s"
            ),
            llm_timeout_s=_as_float(
                d.get("llm_timeout_s", 12.0), "agent.llm_timeout_s"
            ),
            max_llm_attempts=_as_int(
                d.get("max_llm_attempts", 2), "agent.max_llm_attempts"
            ),
            max_tool_result_chars=_as_int(
                d.get("max_tool_result_chars", 2000), "agent.max_tool_result_chars"
            ),
        )


@dataclass(frozen=True)
class EnvelopeConfig:
    """`embodiment.envelope` — STUART-verified platform facts (defaults
    match brain/embodiment.py's Envelope)."""

    pick_min_height_m: float = -0.35  # torso frame
                                      # z = height above torso link;
                                      # tabletop z ~[0.00,+0.07],
                                      # ground ~-0.75
    pick_max_height_m: float = 1.20
    max_reach_m: float = 0.90
    grip_center_standoff_m: float = 0.0  # must stay 0.0
                                         # 0.37 offsets the wrist target and picks the wrong spot;
                                         # no engine reads this field
    min_arm_dur_s: float = 1.5  # <dur> clamp
                                # shortest allowed arm motion

    # ── descent lane
    # all switches default off; False makes effective_pick_floor
    # return pick_min_height_m unchanged. The four law inputs
    # come from config, with no literals in the law itself
    low_workspace_lane: bool = False  # master switch
                                      # false = tabletop-only
                                      # strict bool: "false"/0 refuses
                                      # consumer: effective_pick_floor
    height_stand_m: float = 0.74  # upper bound of the
                                  # writer's height_stand
    height_min_m: float = 0.55  # clamp floor
                                # descent budget = stand - min; the writer keeps the
                                # band shut (min == max == 0.74) until it is armed
    hold_tol_m: float = 0.05  # same-height tol
                              # hold iff |h_need - h_cur| <= it
    descend_confirm_timeout_s: float = 3.0  # confirm poll (s)
    lane_operation_budget_s: float = 30.0  # lane budget (s)
                                           # matches the descent lane's own budget;
                                           # brain never imports gateway
    pre_descent_step: bool = False  # stance step
                                    # widen on the writer's `step` verbs before descending;
                                    # strict bool, defaults off
                                    # consumer: _pre_descent_step_flag
    left_hand_in_service: bool = False  # left Dex3 not in service
                                        # strict bool; False makes the tool registry
                                        # refuse arm commands for 'left'

    # ── pc switches
    # strict bools; all False means no pointcloud client, no
    # /pc_arm publisher and no /pc_result subscriber. Consumer:
    # resolve_pc_switches; pc_support_split is not allowlisted
    pc_shadow: bool = False  # log only
                             # non-decisive; the rim is requested and logged,
                             # never applied
    pc_hold_check: bool = False  # decisive
                                 # a negative check only downgrades
    pc_place_rim: bool = False  # applied
                                # the rim applies only when the check passes and
                                # the pose matches and the row is fresh


@dataclass(frozen=True)
class PerceptionConfig:
    """`embodiment.perception` (defaults match brain/embodiment.py)."""

    freshness_s: float = 2.0
    silence_means_empty: bool = True  # event-driven
                                      # silence == empty scene
    frame: str = "torso"  # frame axes
                          # x right, y fwd, z up
    # conf floor
    # brain-only: below it no detection becomes a scene-graph
    # reference. The gateway still emits raw detections, so
    # filtering upstream can make a table object read 'grasped'
    det_conf_floor: float = 0.35


@dataclass(frozen=True)
class BaseRelativeConfig:
    """`embodiment.base_relative` — named positions relative to base/torso.
    The key_positions map itself is read-only (mappingproxy).  Defaults
    mirror brain/embodiment.py's BaseRelative (grip_center_m +
    camera_offset_m) so an ABSENT segment yields the same named spots
    across the loader and dataclass lanes (round-5 parity: place_at_spot
    resolves the same defaults either way)."""

    key_positions: Mapping[str, List[float]] = field(default_factory=lambda: {
        "grip_center_m": [0.0, 0.37, 0.0],  # standoff
                                            # from the grip centre
        "camera_offset_m": [0.0, 0.0, 0.0],  # torso-frame
    })

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "key_positions", MappingProxyType(dict(self.key_positions))
        )


@dataclass(frozen=True)
class EmbodimentConfig:
    """The `embodiment:` capability segment — mirrors brain/embodiment.py's
    schema exactly (envelope / perception / base_relative).

    RUNTIME CONSUMER (round-5 config-steering): brain/agent/agent.py
    ``G1Agent.from_config`` constructs this dataclass via
    ``EmbodimentConfig.from_section`` and INJECTS it into the running
    hooks (safety_checks_hook envelope — pick_min/max_height_m,
    perception.freshness_s fallback) and the tool registry (pick
    placeholder max_reach_m / grip_center_standoff_m + place_at_spot
    base_relative.key_positions lookup).  Gateway-side keys
    (min_arm_dur_s / silence_means_empty / frame) are consumed by the
    gateway cluster's round-5 runtime-embodiment wiring (gateway/
    gateway_node.py + gateway/assemble.py)."""

    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    base_relative: BaseRelativeConfig = field(default_factory=BaseRelativeConfig)

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "EmbodimentConfig":
        d = _section_dict(section, "embodiment")
        _check_keys(d, {"envelope", "perception", "base_relative"}, "embodiment")

        env = _section_dict(d.get("envelope"), "embodiment.envelope")
        _check_keys(
            env,
            {"pick_min_height_m", "pick_max_height_m", "max_reach_m",
             "grip_center_standoff_m", "min_arm_dur_s",
             # entry first
             # _check_keys rejects unknown keys, so every key a
             # config file may set must be listed here
             "low_workspace_lane", "height_stand_m", "height_min_m",
             "hold_tol_m", "descend_confirm_timeout_s",
             # same rule for the lane budget
             "lane_operation_budget_s",
             # left-hand key, same rule
             "left_hand_in_service",
             # pc entries, same rule
             # pc_support_split is deliberately absent, so it cannot
             # be enabled from config. Keep these before
             # pre_descent_step: the documented order is pinned
             "pc_shadow", "pc_hold_check", "pc_place_rim",
             # stance key, same rule
             "pre_descent_step"},
            "embodiment.envelope",
        )
    # validate, do not substitute
    # a NaN budget makes every `elapsed > budget` false, so the
    # lane is unbounded; 0 or negative refuses every dispatch
    # and either way the file disagrees with the lane
        lane_budget = _as_float(
            env.get("lane_operation_budget_s", 30.0),
            "embodiment.envelope.lane_operation_budget_s",
        )
        if not (math.isfinite(lane_budget) and lane_budget > 0.0):
            raise ValueError(
                "config error: embodiment.envelope.lane_operation_budget_s "
                f"must be a finite positive number of seconds, got "
                f"{lane_budget!r} — refusing to guess a budget for the lane"
            )
        envelope = EnvelopeConfig(
            pick_min_height_m=_as_float(
                env.get("pick_min_height_m", -0.35), "embodiment.envelope.pick_min_height_m"
            ),
            pick_max_height_m=_as_float(
                env.get("pick_max_height_m", 1.20), "embodiment.envelope.pick_max_height_m"
            ),
            max_reach_m=_as_float(
                env.get("max_reach_m", 0.90), "embodiment.envelope.max_reach_m"
            ),
            grip_center_standoff_m=_as_float(
                env.get("grip_center_standoff_m", 0.0),
                "embodiment.envelope.grip_center_standoff_m",
            ),
            min_arm_dur_s=_as_float(
                env.get("min_arm_dur_s", 1.5), "embodiment.envelope.min_arm_dur_s"
            ),
            # strict bool
            # _as_bool accepts a real Python bool only; 'false', 0 or 1
            # is a ConfigError, never a silent motion enable. The law
            # inputs stay plain finite floats and the law fails closed
            # on a non-finite band
            low_workspace_lane=_as_bool(
                env.get("low_workspace_lane", False),
                "embodiment.envelope.low_workspace_lane",
            ),
            height_stand_m=_as_float(
                env.get("height_stand_m", 0.74), "embodiment.envelope.height_stand_m"
            ),
            height_min_m=_as_float(
                env.get("height_min_m", 0.55), "embodiment.envelope.height_min_m"
            ),
            hold_tol_m=_as_float(
                env.get("hold_tol_m", 0.05), "embodiment.envelope.hold_tol_m"
            ),
            descend_confirm_timeout_s=_as_float(
                env.get("descend_confirm_timeout_s", 3.0),
                "embodiment.envelope.descend_confirm_timeout_s",
            ),
            lane_operation_budget_s=lane_budget,
            # strict bool
            # like low_workspace_lane: 'true'/0/1 is a
            # ConfigError, never a silent motion enable
            pre_descent_step=_as_bool(
                env.get("pre_descent_step", False),
                "embodiment.envelope.pre_descent_step",
            ),
            left_hand_in_service=_as_bool(
                env.get("left_hand_in_service", False),
                "embodiment.envelope.left_hand_in_service",
            ),
            # strict bools too
            # a string or number is a ConfigError, never a
            # silent enable next to pick/place
            pc_shadow=_as_bool(
                env.get("pc_shadow", False),
                "embodiment.envelope.pc_shadow",
            ),
            pc_hold_check=_as_bool(
                env.get("pc_hold_check", False),
                "embodiment.envelope.pc_hold_check",
            ),
            pc_place_rim=_as_bool(
                env.get("pc_place_rim", False),
                "embodiment.envelope.pc_place_rim",
            ),
        )

        perc = _section_dict(d.get("perception"), "embodiment.perception")
        _check_keys(
            perc,
            {"freshness_s", "silence_means_empty", "frame", "det_conf_floor"},
            "embodiment.perception",
        )
        # finite only
        # a NaN floor makes `d.conf < conf_floor` false, so it
        # looks configured and filters nothing
        conf_floor = _as_float(
            perc.get("det_conf_floor", 0.35), "embodiment.perception.det_conf_floor"
        )
        if not math.isfinite(conf_floor):
            raise ValueError(
                "config error: embodiment.perception.det_conf_floor must be "
                f"finite, got {conf_floor!r}"
            )
        perception = PerceptionConfig(
            freshness_s=_as_float(
                perc.get("freshness_s", 2.0), "embodiment.perception.freshness_s"
            ),
            silence_means_empty=_as_bool(
                perc.get("silence_means_empty", True),
                "embodiment.perception.silence_means_empty",
            ),
            frame=_as_str(perc.get("frame", "torso"), "embodiment.perception.frame"),
            det_conf_floor=conf_floor,
        )

        br = _section_dict(d.get("base_relative"), "embodiment.base_relative")
        _check_keys(br, {"key_positions"}, "embodiment.base_relative")
        raw_kp = br.get("key_positions")
        if raw_kp is None:
            # dataclass default
            # an absent key yields grip_center_m and
            # camera_offset_m, matching the embodiment defaults;
            # an explicit {} stays an empty map
            base_relative = BaseRelativeConfig()
        else:
            if not isinstance(raw_kp, Mapping):
                raise ValueError(
                    "config error: embodiment.base_relative.key_positions must be a "
                    f"mapping, got {type(raw_kp).__name__}"
                )
            key_positions: Dict[str, List[float]] = {}
            for name, pos in raw_kp.items():
                if (
                    not isinstance(pos, list)
                    or not all(
                        isinstance(x, (int, float)) and not isinstance(x, bool)
                        for x in pos
                    )
                ):
                    raise ValueError(
                        "config error: embodiment.base_relative.key_positions"
                        f"[{name!r}] must be a list of numbers, got {pos!r}"
                    )
                key_positions[str(name)] = [float(x) for x in pos]
            base_relative = BaseRelativeConfig(key_positions=key_positions)
        return cls(
            envelope=envelope,
            perception=perception,
            base_relative=base_relative,
        )


@dataclass(frozen=True)
class BrainConfig:
    """The `brain:` segment — the v4 lane switch (OpenBot v4 rebuild).

    lane:
      - 'realtime'   — the cloud OpenAI Realtime WebSocket brain
        (RealtimeConfig below).  The SEALED default; the user pins BOTH
        model AND lane after the hardware bake-off (plan 6);
      - 'transcript' — the REST chat-completions brain (RestConfig below);
      - 'offline'    — the OfflineBrain regex brain (brain/agent/
        offline_brain.py): NO LLM, NO API key, NO network — ever.

    confirm_motion (operator directive 2026-09-26): whether a MOTION tool
    pauses for a spoken yes before it runs.  False executes on the command.
    This disables the ASK only — the safety pre-hooks, the verifier
    post-hook and STOP/CANCEL are untouched, and the hook records each
    unasked motion as APPROVED UNASKED so the audit trail never claims a
    human consent that did not happen.  Default True: a config that never
    sets the key keeps the pause.
    """

    lane: str = "realtime"
    confirm_motion: bool = True

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "BrainConfig":
        d = _section_dict(section, "brain")
        _check_keys(d, {"lane", "confirm_motion"}, "brain")
        lane = _as_str(d.get("lane", "realtime"), "brain.lane")
        if lane not in ("realtime", "transcript", "offline"):
            raise ValueError(
                f"config error: brain.lane {lane!r} must be one of "
                f"'realtime' (cloud Realtime WebSocket), 'transcript' "
                f"(REST chat-completions) or 'offline' (regex brain, "
                f"no network)"
            )
        raw = d.get("confirm_motion", True)
        if not isinstance(raw, bool):
            # no guessing
            # the parser maps true/false/yes/no to a bool; a str or int
            # here is 'maybe'/'on'/'2', or an overrides dict carrying
            # 'false', which reads truthy and leaves the gate on
            raise ValueError(
                f"config error: brain.confirm_motion {raw!r} must be a YAML "
                f"boolean (true/false), not {type(raw).__name__} — refusing "
                f"to guess whether the motion confirm gate is on or off"
            )
        return cls(lane=lane, confirm_motion=raw)


@dataclass(frozen=True)
class RealtimeConfig:
    """The `realtime:` segment — the OpenAI Realtime WebSocket brain
    (brain.lane 'realtime').  Every value is the SEALED marketplace
    default until the bake-off pin (plan 6) pins model AND lane after
    hardware testing.

    reconnect_backoffs: seconds to wait per reconnect attempt — the i-th
    entry is the i-th retry's backoff; 8 entries = 8 retries max, after
    which the session gives up.  Accepted as an inline list
    `[1, 2, 4, 8, ...]` (what the YAML-subset parser resolves) or a
    comma-separated string '1,2,4,8,...'.
    """

    model: str = "gpt-realtime-2.1-mini"
    base_url: str = "wss://api.openai.com/v1/realtime"
    api_key_env: str = "OPENAI_API_KEY"  # env var NAME
                                         # never the key itself
    voice: str = "cedar"
    speed: float = 0.9
    sample_rate: int = 24000
    #: voice identity
    #: session.update `instructions`, sent once at connect;
    #: per-turn create_response instructions still win.
    #: Empty = field absent and the built-in default applies
    instructions: str = ""
    connect_timeout_s: float = 10.0
    ready_timeout_s: float = 10.0
    session_cap_min: int = 60
    reconnect_backoffs: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 8, 8, 8, 8]
    )

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "RealtimeConfig":
        d = _section_dict(section, "realtime")
        _check_keys(
            d,
            {"model", "base_url", "api_key_env", "voice", "speed",
             "sample_rate", "connect_timeout_s", "ready_timeout_s",
             "session_cap_min", "reconnect_backoffs", "instructions"},
            "realtime",
        )
        return cls(
            model=_as_str(d.get("model", "gpt-realtime-2.1-mini"), "realtime.model"),
            base_url=_as_str(
                d.get("base_url", "wss://api.openai.com/v1/realtime"),
                "realtime.base_url",
            ),
            api_key_env=_as_str(
                d.get("api_key_env", "OPENAI_API_KEY"), "realtime.api_key_env"
            ),
            voice=_as_str(d.get("voice", "cedar"), "realtime.voice"),
            instructions=_as_str(
                d.get("instructions", ""), "realtime.instructions"
            ),
            speed=_as_float(d.get("speed", 0.9), "realtime.speed"),
            sample_rate=_as_int(d.get("sample_rate", 24000), "realtime.sample_rate"),
            connect_timeout_s=_as_float(
                d.get("connect_timeout_s", 10.0), "realtime.connect_timeout_s"
            ),
            ready_timeout_s=_as_float(
                d.get("ready_timeout_s", 10.0), "realtime.ready_timeout_s"
            ),
            session_cap_min=_as_int(
                d.get("session_cap_min", 60), "realtime.session_cap_min"
            ),
            reconnect_backoffs=_as_backoffs(
                d.get("reconnect_backoffs", [1, 2, 4, 8, 8, 8, 8, 8]),
                "realtime.reconnect_backoffs",
            ),
        )


@dataclass(frozen=True)
class RestConfig:
    """The `rest:` segment — the transcript-lane REST chat-completions
    brain (brain.lane 'transcript'; the OpenAI-compatible
    /chat/completions endpoint, the same client family as
    brain/agent/llm_cloud.py).  api_key_env names the env var holding
    the key — the key itself NEVER lives in config files."""

    model: str = "gpt-5.4-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    tool_mode: str = "skeleton"  # skeleton | json
    timeout_s: float = 120.0
    #: prose fallback
    #: a prose-only reply comes back as ChatResult.text, which
    #: the voice lane speaks, not as the empty repair-path
    #: result. Consumer: CloudLLMClient.prose_fallback
    prose_fallback: bool = False
    #: budget
    #: 256-token cap, same shape as llm.max_tokens; the client
    #: applies the host dialect. The bake-off ranks with this cap,
    #: pinned so a bench cannot outrank the lane by truncation
    max_tokens: int = 256

    @classmethod
    def from_section(cls, section: Optional[Mapping[str, Any]]) -> "RestConfig":
        d = _section_dict(section, "rest")
        _check_keys(
            d,
            {"model", "base_url", "api_key_env", "tool_mode", "timeout_s",
             "prose_fallback", "max_tokens"},
            "rest",
        )
        tool_mode = d.get("tool_mode", "skeleton")
        if tool_mode not in ("skeleton", "json"):
            raise ValueError(
                f"config error: rest.tool_mode {tool_mode!r} must be "
                f"'skeleton' (plain-text JSON skeleton + deterministic "
                f"parse) or 'json' (grammar-constrained json-mode)"
            )
        tok_raw = d.get("max_tokens", 256)
        if isinstance(tok_raw, float) and not tok_raw.is_integer():
            raise ValueError(
                f"config error: rest.max_tokens {tok_raw!r} must be an "
                f"int (a fractional value would be silently truncated — "
                f"never a silent budget change, #120)"
            )
        max_tokens = _as_int(tok_raw, "rest.max_tokens")
        if max_tokens < 1:
            raise ValueError(
                f"config error: rest.max_tokens {max_tokens!r} must be >= 1 "
                f"(a 0/negative budget makes the provider return an empty "
                f"completion — a silent truncation, #120)"
            )
        return cls(
            model=_as_str(d.get("model", "gpt-5.4-mini"), "rest.model"),
            base_url=_as_str(
                d.get("base_url", "https://api.openai.com/v1"), "rest.base_url"
            ),
            api_key_env=_as_str(
                d.get("api_key_env", "OPENAI_API_KEY"), "rest.api_key_env"
            ),
            tool_mode=tool_mode,
            timeout_s=_as_float(d.get("timeout_s", 120.0), "rest.timeout_s"),
            prose_fallback=_as_bool(
                d.get("prose_fallback", False), "rest.prose_fallback"
            ),
            max_tokens=max_tokens,
        )
