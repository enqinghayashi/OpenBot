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


def scene_brief_prose(text: str) -> str:
    """Restructure the scene_brief DATA form into one spoken sentence.

    The incoming text is the tool's RESULT TEXT (envelope form: the first
    line carries the "OK: " prefix).  Every non-empty line must then parse
    as the brief's own line form, else the text is returned BYTE-IDENTICAL
    (the SCENE: special forms — unavailable / empty — and any foreign brief
    are already honest prose; never a guessed restructure).  The numbers ARE
    the measured values; nothing is added or smoothed.  Flags stay honest:
    IN-HAND / IN-CONTAINER (task-inferred) / STALE are spelled out, never
    dropped.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return text
    first = _SCENE_ENVELOPE.sub("", lines[0], count=1)
    rest = [first] + lines[1:]
    parsed = []
    for ln in rest:
        m = _SCENE_BRIEF_LINE.match(ln)
        if m is None:
            return text  # not the data form — as-is
        parsed.append(
            (m.group("label"), float(m.group("x")), float(m.group("y")),
             (m.group("parts") or "").strip())
        )
    n = len(parsed)
    clauses = []
    for label, x, y, flags in parsed:
        side = "right" if x >= 0.0 else "left"
        clause = "The %s is about %.2f metres to the %s and %.2f metres ahead" % (
            label, abs(x), side, y)
        if "IN-HAND" in flags:
            clause += "; it is in my hand"
        if "IN-CONTAINER" in flags:
            clause += "; it is marked in the container (task-inferred)"
        m_stale = _SCENE_STALE.search(flags)
        if m_stale:
            clause += "; its position was seen %s seconds ago" % m_stale.group("age")
        clauses.append(clause)
    return "I see %d object%s. %s." % (
        n, "" if n == 1 else "s", "; ".join(clauses))

# 'put the bottle at the shelf'
# 'at' only
# into/on live in PLACE_INTO_RE
PLACE_AT_SPOT_RE = re.compile(
    rf'^{_NAV_POLITE}(?:put|place|set)\s+(?:the\s+)?(.+?)'
    rf'\s+at\s+(?:the\s+)?(.+)$',
    re.IGNORECASE,
)

# 'put the bottle in the bin'
# narrower than PLACE_INTO
#   verbs put|place
#   preps in|into
# (on/onto = surface)
# no legacy steal
# TOOL layer does it
# unknown -> NOT_DETECTED
# a drop needs demonstrated containment
# veto both groups
PLACE_INTO_CONTAINER_RE = re.compile(
    rf'^{_NAV_POLITE}(?:put|place)\s+(?:the\s+)?(.+?)'
    rf'\s+(?:in|into)\s+(?:the\s+)?(.+)$',
    re.IGNORECASE,
)

# -> scene_brief lane
# before the decline
# apostrophes first
# ("what's" -> "whats")
# hence the entries
SEE_KEYWORDS = (
    "see",
    "look at",
    "what is in the scene",
    "what can you see",
    "do you see",
    "is there",
    "what is on the table",
    "what do you see",
    # contracted forms only
    "whats in the scene",
    "whats on the table",
)

SEE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in SEE_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# 'is the bottle on the table?'
# catches the generic question
# fronted question auxiliary
# commands can't match
# multi-clause vetoed too
SEE_QUESTION_RE = re.compile(
    r"^(?:is|are|was)\s+(?:the|there)\s+(?:a|an|the\s+)?"
    r"(.+?)\s+(?:on|in|next to|beside|at)\s+(?:the\s+)?(.+?)\s*[?!.]*$",
    re.IGNORECASE,
)

# no answer -> decline
# no model backstop
# never invented
QUERY_PATTERN_RE = re.compile(
    r"\b(?:what|why|how|when|who|where|explain)\b|tell me about",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Spoken replies (OpenBot's own strings; byte-for-byte the same as the agent
# wherever the lane exists online)
# ---------------------------------------------------------------------------

# same bytes as G1Agent.run() (agent.py)
EMPTY_REPLY = "I did not understand the request."

# no capability speech
QUERY_DECLINE = "fail, try again."

# no match -> same terse decline
DEFAULT_DECLINE = "fail, try again."

# stripped first
# wake greetings leak in
# plus lead-ins
# filler only
# never a verb, a halt word,
# or an object label
_LEAD_GREETINGS = frozenset({
    "hey", "hi", "hello", "yo", "okay", "ok", "so", "well",
})
# butchered 'hey <name>'
# mis-hearings we saw
# exact -> _strip_vocative
# unlisted: nothing
# a wrong one dies
_NAME_MISHEARD_VARIANTS = {
    "stewart": ("stuart", "steal it", "still it", "still work", "steve with"),
}
_LEAD_COMMAND_PHRASES = ("do it", "go ahead", "go on")


def _strip_lead_fillers(text: str, name: Optional[str] = None) -> str:
    """Strip a bounded leading run of greeting/lead-in filler heads.

    Live 2026-09-07 (offline lane): the transcripts 'Hey hey, do it. Pick up
    the cup.' and 'Hey, hey, do it. Pick up the bottle.' reach this brain
    with the wake-word greeting AND a 'do it' lead-in in front of a perfectly
    canonical pick — and the '^'-anchored verb patterns decline them.  This
    step removes only RECOGNIZED filler heads so the action verb is exposed:

      'hey hey, do it. pick up the cup'  ->  'pick up the cup'

    Safety: a filler head is stripped ONLY when a non-empty remainder
    follows.  A bare greeting ('hey hey', 'hello there') returns unchanged
    and declines — it is NOT the silent bare-summon path (that is reserved
    for the embodied NAME).  A wrong-name utterance ('hey susan pick the
    bottle') strips the greeting but 'susan' survives as a non-filler, so the
    pick pattern still declines — the vocative strip's arbitrary-word guard
    is preserved.
    """
    name_variants = _NAME_MISHEARD_VARIANTS.get((name or "").lower(), ())
    original = text
    while True:
        t = text.strip()
        if not t:
            return original
        stripped = False
        m = re.match(r"^([A-Za-z']+)[\s,]+(.*)$", t)
        if m and m.group(1).lower().rstrip(".,!?") in _LEAD_GREETINGS:
            text = m.group(2).strip()
            stripped = True
        else:
            low = t.lower()
            for phrase in _LEAD_COMMAND_PHRASES + name_variants:
                if low.startswith(phrase) and (
                    len(low) == len(phrase) or low[len(phrase)] in " ,.!?"
                ):
                    text = t[len(phrase):].lstrip(" ,.!?").strip()
                    stripped = True
                    break
        if not stripped:
            return t if t else original
        if not text.strip():
            return original


# ---------------------------------------------------------------------------
# guards for the simple keyword + last-word lane
# ---------------------------------------------------------------------------
# simple lane scans
# the WHOLE utterance
# that's how 'Oh no, stay with it.
# Pick up the cup.' works
# and it acts on what
# rich lanes refuse
# two guards:
#   clause head: the verb
#     opens utterance/sentence
#     not a continuation
#     ('hey susan pick the bottle')
#   payload veto: multi_clause_veto
#     after the verb
# trailing prep/article
# is never an object
_SIMPLE_CLAUSE_BREAK_RE = re.compile(r"[.!?;:]\s*$")
_SIMPLE_NON_OBJECTS = frozenset({
    "the", "a", "an", "this", "that", "these", "those",
    "in", "into", "on", "onto", "at", "to", "for", "with", "from", "of",
    "and", "then", "or", "but", "by", "over", "under", "near",
})


def _simple_lane_payload(cleaned_l: str) -> Optional[str]:
    """Text AFTER the simple lane's action verb, or None when the verb does
    not HEAD a clause (see the guard note above)."""
    toks = cleaned_l.split()
    for i, tok in enumerate(toks):
        if tok.lower().strip("'") in _SIMPLE_ACTION_MAP:
            lead = " ".join(toks[:i])
            if lead and not _SIMPLE_CLAUSE_BREAK_RE.search(lead):
                return None
            return " ".join(toks[i + 1:])
    return None


