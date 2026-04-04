r"""Deterministic regex fast path — ``classify_intent`` family ported from
``openbot-thesis-open/src/llm_node/llm_node/llm_node.py`` (the intent-
classification block, which ran on the G1 in openbot-thesis), adapted to the
OpenBot brain tool set.

The port keeps the source's constants and helpers verbatim where they
survive (``_NAV_POLITE``, ``NAV_PATTERNS``, ``PICK_PATTERNS``,
``TRAILING_FILLERS``, ``NAME_PREFIX_RE``, word-number normalization).  The
source's ``move_relative`` / turn family is dropped — the OpenBot brain
exposes no relative-locomotion tool — and replaced by single-action
patterns for the brain tool set::

    pick | place | navigate 'X' | turn around | open/close hand | stop

Behavior contract (source semantics preserved):
- Returns one intent dict (action + resolved fields) or None.
- Explicit halt is a safety lane: the utterance is normalized
  (punctuation -> spaces) and the FIRST up-to-4 tokens scanned for any
  halt word ('stop', 'halt', 'freeze', 'stand', 'brake', 'wait', 'quit'),
  checked BEFORE any other lane — comma-punctuated phrasings ('stop,
  please', 'please, stop', 'stand still, please') all halt; a negated
  halt ('don't stop') falls through to the normal lanes.
- Heard-name stripping refuses halt-start tokens AND documented verb
  heads: 'set the cup down' is a place (PLACE_DOWN_RE / PLACE_INTO_RE
  advertise (put|set)), never 'set' being a name; real heard names
  ('stewart, ...', 'susan ...') still strip.
- Nav patterns are checked BEFORE pick patterns so "go to X" / "take me
  to X" never classify as picks.
- ``to`` is OPTIONAL for the walk-class verbs (2026-09-06, user
  canonical forms): "go to chair", "walk chair" and "navigate chair"
  all map to navigate target 'chair'.  The '(?:\s+to)?' class is
  deliberate — "move chair" / "head chair" classify too — while the
  required-target group still refuses a bare verb with no object.
  A filler-only target ('go to' / '... to the') is skipped via
  ``_NAV_FILLERS``: 'to'/'the' alone is never a marker.
- ``"get"`` is deliberately excluded from pick verbs (too easily confused
  with "get me to X").
- ``"pick"`` is accepted WITHOUT "up" at the utterance start ("pick the
  bottle" maps to a pick call).  The fast path is a matcher, not a judge:
  a phrase that is not a real object ("pick a rose") still maps to a pick
  call whose label never resolves — the tool refuses honestly with
  NOT_DETECTED; nothing here can claim world success.
- Trailing fillers are stripped from targets; word numbers normalize
  ("two cups" -> "2 cups").
- Cardinal number words/digits are stripped from LABEL candidates before
  extraction ("the two cups" -> label "cups"): the count is a
  disambiguation hint carried in the intent/arguments as ``count``, never
  part of the label (a detection label is 'cup', never '2 cups').

``classify_tool_call(text)`` maps a matched intent straight to a
:class:`~brain.agent.llm.ToolCall` — the fast path NEVER touches the LLM
(no model, no chat, no IO).  Honesty contract: this module is a *matcher*,
not a judge.  A match means "the utterance looks like this single action" —
the agent still runs the full PRE-hook (human confirm gate + safety checks)
and POST-hook (verifier) pipeline on the resulting tool call; nothing here
can claim world success.

Round 8 (multi-clause veto): every single-action lane that captures a
*following payload* (pick label, place label/destination, nav target)
refuses a SECOND clause.  ``multi_clause_veto`` splits the captured payload
on /[.!?]+|[,;]+|\band\b|\bthen\b/ and vetoes when some part after the
first contains any non-``TRAILING_FILLERS`` token — "pick up the remote,
put it in the green plate" is a TWO-step task that must fall through to
the LLM lane (classify returns None), NEVER fabricate the garbage pick
label 'remote, put it in the green plate' (which then honestly refuses
NOT_DETECTED and can never reach the LLM).  Sentence terminators split too:
whisper run-on transcripts ("Pick up the bottle. Let's see what...", the
2026-09-06 live E2 case) veto on the non-filler second sentence rather
than riding into the pick label.  Filler-only continuations
('please', 'thanks', 'now', 'then') never veto.  The leading 'please' of
``_NAV_POLITE`` also tolerates a comma ('please, pick up the cup') so the
comma form stays fastpath like the space form.  The explicit-halt lane is
untouched and runs FIRST — 'stop, set the cup down' is still a halt.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .llm import ToolCall

# latency gate: well under 50 ms
# LLM never on this lane
FASTPATH_MAX_MS = 50

# =============================================================================
# intent classification (the openbot-thesis family)
# =============================================================================

_NAV_POLITE = (
    r"(?:please[\s,]+|can\s+you\s+|could\s+you\s+|would\s+you\s+|"
    r"let'?s\s+|lets\s+)?"
)

# no marker from filler targets: 'go to'
# must not yield target 'to'
_NAV_FILLERS = frozenset({"to", "the", "a", "an", "me", "us", "there", "here"})

# seek verbs: "find backpack and walk to it".
# 'find' and 'locate' open this nav lane.
_NAV_SEEK = r'(?:find|locate)'

NAV_PATTERNS = [
    # 'find [me] the backpack' - seek verb alone,
    # else the motion pattern grabs 'me'.
    re.compile(
        rf'^{_NAV_POLITE}{_NAV_SEEK}\s+(?:me\s+|us\s+)?(?:the\s+)?(.+)$',
        re.IGNORECASE,
    ),
    # nav verbs + compound seek tail ('go find X'):
    # without it 'go find the backpack' labels
    # 'find the backpack' - NOT_FOUND.
    re.compile(
        rf'^{_NAV_POLITE}(?:go|walk|move|head|navigate)(?:\s+to)?'
        rf'(?:\s+{_NAV_SEEK}(?:\s+(?:me|us))?)?'
        rf'\s+(?:the\s+)?(.+)$',
        re.IGNORECASE,
    ),
    re.compile(
        rf'^{_NAV_POLITE}(?:take|bring)\s+(?:me|us)\s+to\s+(?:the\s+)?(.+)$',
        re.IGNORECASE,
    ),
]

# "pick up the bottle" / "grab the cup"
# optional 'up' and the '-ing' form. whisper
# glues 'pick up' -> 'pickup', so the particle
# takes a hyphen. a bare 'up'/'it' is no
# object; "get" is left out (cf. "get me to X").
PICK_PATTERNS = [
    re.compile(
        rf'^{_NAV_POLITE}(?:pick|grab|fetch)(?:ing|bing)?(?:\s*-?\s*up)?'
        rf'\s+(?:the\s+|a\s+|an\s+)?(.+)$',
        re.IGNORECASE,
    ),
]

TRAILING_FILLERS = {
    'now', 'please', 'thanks', 'thank', 'you',
    'asap', 'then', 'first', 'next', 'okay', 'ok',
}

NAME_PREFIX_RE = re.compile(
    r'^(?:he\s*)?(s\w*)[,\.\s]+',
    re.IGNORECASE,
)

# never strip a halt-phrase start as a name
HALT_PHRASE_STARTS = frozenset({"stop", "halt", "freeze", "stand"})

# grammar heads too: 'set the cup down' is a
# place, not a name (PLACE_* advertise put|set)
S_VERB_HEADS = frozenset({"set"})

# NAME_PREFIX_RE would chop 'spin around' to
# 'around', so 'spin' is a protected head.
TURN_AROUND_HEADS = frozenset({"spin"})

# halt words: any in the FIRST 4 normalized
# tokens. token scanning is what makes
# 'stop, please' halt.
# safety-critical: never drop a halt.
HALT_WORDS = frozenset(
    {"stop", "halt", "freeze", "stand", "brake", "wait", "quit",
     # 'cancel' / 'abort': "stop doing what it is
     # doing". single words, so the negation guard
     # covers them.
     "cancel", "abort"}
)

#: multi-word halts. not HALT_WORDS (single
#: tokens), not a new negation: 'never mind'
#: halts *because of* 'never', which
#: :data:`_HALT_NEGATIONS` would suppress.
#: matched on space-joined tokens, sound only
#: because callers pass _normalize_for_halt.
HALT_PHRASES_RE = re.compile(r"\b(?:never\s*mind|forget\s+(?:it|about\s+it))\b")

# negations: don't stop, never stop, do not
_HALT_NEGATIONS = frozenset({"don't", "dont", "never"})


def _normalize_for_halt(text: str) -> list:
    """Punctuation -> spaces, lowercased, whitespace collapsed; returns the
    token list the halt lane scans.  Normalizing BEFORE any matching is
    what makes comma-punctuated halt phrasings ('stop, please', 'please,
    stop', 'stand still, please') halt — the comma no longer separates the
    halt verb from the rest of the utterance."""
    norm = re.sub(r"[.,!?]+", " ", text).lower()
    return re.sub(r"\s+", " ", norm).strip().split()


def _halt_word_negated(tokens: list, i: int) -> bool:
    """True when tokens[i] (a halt word) is negated by 'don't' / 'dont' /
    'never' (within 2 tokens) or the 'do not' pair directly before it."""
    if i >= 1 and tokens[i - 1] in _HALT_NEGATIONS:
        return True
    if i >= 2:
        if tokens[i - 2] in _HALT_NEGATIONS:
            return True
        if tokens[i - 1] == "not" and tokens[i - 2] == "do":
            return True
    return False


def _halt_intent(tokens: list) -> bool:
    """Token-scan halt: a multi-word halt PHRASE anywhere in the utterance,
    or any halt word among the FIRST UP-TO-4 tokens of the normalized
    utterance, unless negated ('don't stop' is skipped and falls
    through to the normal lanes instead of halting).

    The phrase lane is position-independent on purpose.  The 4-token
    window exists so that an incidental 'wait' deep inside a task
    sentence ('pick the bottle and wait there') is not a halt; 'never
    mind' / 'forget it' carry no such ambiguity — they are complete
    retractions whenever they appear.
    """
    if HALT_PHRASES_RE.search(" ".join(tokens)):
        return True
    for i in range(min(4, len(tokens))):
        if tokens[i] in HALT_WORDS and not _halt_word_negated(tokens, i):
            return True
    return False


def _strip_heard_name(text: str) -> str:
    """Strip a leading heard-name prefix ('stewart,', 'susan '), refusing
    any token that starts an explicit halt phrase or heads a documented
    single-action grammar.

    ``NAME_PREFIX_RE`` matches any leading s-word + separator, which would
    eat 'stop the robot' / 'stop now' / 'stop please' as if 'stop' were a
    heard name.  A token in HALT_PHRASE_STARTS ('stop', 'halt', 'freeze',
    'stand') is therefore never stripped — the halt lane pre-empts name
    handling.  A token in S_VERB_HEADS ('set' — the head of the
    PLACE_DOWN_RE / PLACE_INTO_RE place grammars) is never stripped
    either: 'set the cup down' is a place, not 'set' being a name, and
    'set' must not fall to the LLM path.  A token in TURN_AROUND_HEADS
    ('spin' — the head of TURN_AROUND_RE) is never stripped for the same
    reason: 'spin around' is the turn-around grammar, not 'spin' being a
    name.  Returns the stripped text
    (unchanged when no name prefix matches, or when the candidate is a
    halt word or a verb head)."""
    m = NAME_PREFIX_RE.match(text)
    if not m:
        return text
    if m.group(1).lower() in HALT_PHRASE_STARTS:
        return text
    if m.group(1).lower() in S_VERB_HEADS:
        return text
    if m.group(1).lower() in TURN_AROUND_HEADS:
        return text
    return text[m.end():]

# ---- turning number words into digits ----
_WORD_UNITS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
               'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
_WORD_TEENS = {'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
               'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
               'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
_WORD_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fourty': 40,
              'fifty': 50, 'sixty': 60, 'seventy': 70,
              'eighty': 80, 'ninety': 90}
_WORD_FRACS = {'half': 0.5, 'quarter': 0.25}


