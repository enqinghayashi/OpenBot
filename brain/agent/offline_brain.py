"""OfflineBrain — OpenBot's deterministic regex utterance brain (v4 L1
slice 2a).  No local LLM anywhere: every lane is a finite rule built on the
fastpath family (brain/agent/fastpath.py), and every motion lane executes
through the ToolExecutor (brain/agent/tool_executor.py) with the FULL
PRE-hook confirm gate + POST-hook verifier pipeline.  The verifier stays
the ONLY success authority — online and offline alike.

OPENBOT-THESIS DISCIPLINE: this module builds its OWN strings and lanes on the
OpenBot fastpath primitives (halt token-scan, heard-name strip, word-number
normalization, cardinal stripping, trailing-filler strip, multi-clause
veto, and the shared NAV/PICK/PLACE/HAND patterns).  Nothing is copied
from openbot-thesis files: fastpath itself is the ported family and the reuse
boundary — we import its helpers BY NAME and never re-implement them
differently.  All spoken strings below are OpenBot's own (byte-fidelity
with the existing agent/executor strings where they exist; new strings only
where the lane is new, e.g. place_at_spot and the declines).

Decision tree (strict order; the halt law is mirrored from the agent: the
halt check runs BEFORE any name strip):
  1. explicit halt lane on the RAW text (fastpath halt predicate);
  2. leading-vocative strip (the robot's spoken name's hey/hi/hello
     <name>, <name>, and <name> forms — the name is voice-app-provided
     via the constructor or read off the embodiment, with the fastpath
     heard-name strip as the fallback when there is no name; a partial
     KWS 'he' is never a vocative) -> word-number normalization ->
     trailing-filler strip (fastpath order), then the halt re-check
     ('stewart, stop');
  3. simple keyword + LAST-WORD object lane (fastpath.simple_keyword_intent)
     — one action keyword among {put,pick,place,grab,walk,go,find}, the
     object is the last CONTENT word of the sentence, everything else is
     stripped; verb->action: pick/grab -> pick, put/place -> place (drop in
     place), walk/go/find -> navigate (USER DIRECTIVE 2026-09-08) ->
     hands (open/close) -> see-question (SEE keywords AND the generic
     'is the <object> on the <place>' question form) -> query-pattern
     decline -> DEFAULT_DECLINE.  No keyword yields the short decline
     'fail, try again' — the brain never fabricates a payload.

Contract:
- NO MODEL BACKSTOP: this brain has no LLM behind it.  If the tree does
  not match, it declines — it never falls through to a model.
- HONEST-DECLINE POLICY: an unrecognized/ambiguous utterance, a question
  beyond the tool set, or an unknown place-at spot is refused with a
  truthful capability statement (DEFAULT_DECLINE / QUERY_DECLINE /
  unknown-spot reply) — never a fabricated answer and never a guessed
  position.
- VERDICT-AWARE SPEAKING: a motion lane's spoken reply comes from
  ToolExecutor._fastpath_summary (the same phrasing the online fast path
  emits, byte-identical) keyed on the executor's classification of the
  tool result.  VERDICT:success is the ONLY success phrasing; anything
  else phrases in-progress/not-confirmed/failure.  This module never
  invents success phrasing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .fastpath import (
    CLOSE_HAND_RE,
    HALT_PHRASE_STARTS,
    NAV_PATTERNS,
    OPEN_HAND_RE,
    PICK_PATTERNS,
    POSE_RE,
    PLACE_DOWN_RE,
    PLACE_INTO_RE,
    S_VERB_HEADS,
    TRAILING_FILLERS,
    _NAV_FILLERS,
    _NAV_POLITE,
    _SIMPLE_ACTION_MAP,
    _halt_intent,
    _normalize_for_halt,
    _normalize_pick_label,
    _normalize_word_numbers,
    _strip_cardinals,
    _strip_heard_name,
    _strip_trailing_fillers,
    multi_clause_veto,
    pose_from_text,
    simple_keyword_intent,
)
from .llm import ToolCall
from .tool_executor import ToolExecutor, ToolOutcome

# our tool-call id
# not 'fastpath_1'
OFFLINE_TOOL_CALL_ID = "offline_1"

# ---------------------------------------------------------------------------
# Lane patterns plus the constants that go with them
# ---------------------------------------------------------------------------

# wake word leaks in
# 'hey stewart ...'
# fastpath takes
# a leading 's-word'
# greeting forms pin it
# bare: any s-name
# whisper's comma ok ([,\s]+)
# name-pin guards
_GREETING_VOCATIVE_RE = re.compile(
    r"^(?:hey|hi|hello)[,\s]+([^\s]+?)[,\s]+", re.IGNORECASE
)
_BARE_NAME_RE = re.compile(r"^([^\s,]+?)[,\s]+")
# terminal case
# greeting head, or token
# (whisper adds a period)
# [,\s]+ again: comma ok
_GREETING_HEAD_RE = re.compile(r"^(?:hey|hi|hello)[,\s]+", re.IGNORECASE)
_BARE_NAME_HEAD_RE = re.compile(r"^([^\s,]+?)[,\s]*$")

# ---------------------------------------------------------------------------
# Speech-stop lane ('stop talking') — the user's way out
# ---------------------------------------------------------------------------

# kills SPEECH
# never motion
# prompt-loop escape
# raw text: vocative hits
# before motion halt
STOP_TALKING_RE = re.compile(
    r"\b(?:stop\s+(?:talking|speaking|repeating)|be\s+quiet|shut\s+up)\b",
    re.IGNORECASE,
)

# lane off: stop, cut
# 'stop halt freeze stand'
# never match (halt words)
# 'shut up' no; 'shut down' yes
# before STOP_TALKING_RE
TURN_OFF_RE = re.compile(
    r"\b(?:turn\s+(?:it\s+|yourself\s+)?off|switch\s+off|power\s+(?:it\s+)?(?:off|down)"
    r"|shut\s+down|shutdown|go\s+to\s+sleep|sleep\s+now)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Turning the scene_brief data lines into something speakable
# ---------------------------------------------------------------------------

# one raw line per ref
# ("bottle_1: bottle x=0.18 y=0.53 z=0.01 conf=0.94 0.0s")
# says outcome.text
# so we read our own log
# out loud
# restructured; data raw
_SCENE_BRIEF_LINE = re.compile(
    r"^(?P<ref>\S+): (?P<label>[A-Za-z0-9_-]+) x=(?P<x>-?\d+\.\d+) "
    r"y=(?P<y>-?\d+\.\d+) z=(?P<z>-?\d+\.\d+) conf=(?P<conf>\d+\.\d+)"
    r"(?P<parts>.*)$"
)
_SCENE_STALE = re.compile(r"STALE (?P<age>\d+)s")
# enveloped: "OK: <value>"
# / "ERROR: <reason>"
# strip it or no match
_SCENE_ENVELOPE = re.compile(r"^(?:OK|ERROR): ")


