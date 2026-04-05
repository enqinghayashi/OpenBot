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


def _normalize_word_numbers(text: str) -> str:
    """Convert spoken numbers ('one', 'twenty five', 'half') into digits."""
    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        raw = tokens[i]
        m = re.match(r"^([A-Za-z']+)([,.!?;:]*)$", raw)
        if not m:
            out.append(raw); i += 1; continue
        word, suffix = m.group(1).lower(), m.group(2)

        if word in _WORD_TENS:
            base = _WORD_TENS[word]
            if i + 1 < len(tokens):
                m2 = re.match(r"^([A-Za-z']+)([,.!?;:]*)$", tokens[i + 1])
                if m2 and m2.group(1).lower() in _WORD_UNITS \
                        and _WORD_UNITS[m2.group(1).lower()] != 0:
                    base += _WORD_UNITS[m2.group(1).lower()]
                    suffix = m2.group(2)
                    i += 1
            out.append(f'{base}{suffix}')
        elif word in _WORD_TEENS:
            out.append(f'{_WORD_TEENS[word]}{suffix}')
        elif word in _WORD_UNITS:
            out.append(f'{_WORD_UNITS[word]}{suffix}')
        elif word in _WORD_FRACS:
            out.append(f'{_WORD_FRACS[word]:g}{suffix}')
        else:
            out.append(raw)
        i += 1
    return ' '.join(out)


# ---- pulling cardinal numbers out of LABEL candidates ----
# digitize words BEFORE matching ('the 2 cups')
# _strip_cardinals drops digit and word cardinals:
# label is 'cup', never '2 cups'; count is a hint.
_CARDINAL_WORDS = {
    word: number
    for table in (_WORD_UNITS, _WORD_TEENS, _WORD_TENS)
    for word, number in table.items()
}
_DIGIT_TOKEN_RE = re.compile(r"^[0-9]+$")


def _strip_cardinals(label: str) -> tuple[str, Optional[int]]:
    """Remove standalone cardinal number words/digits from a label candidate.

    Returns ``(cleaned_label, count_or_None)``.  Digit tokens ('2 cups')
    and word cardinals ('two cups') are removed; digits inside words
    ('bottle2') are kept.  The count (first cardinal found) is a
    disambiguation hint — never part of the label.
    """
    count: Optional[int] = None
    kept = []
    for token in label.split():
        bare = token.lower().strip(".,!?")
        number = _CARDINAL_WORDS.get(bare)
        if number is None and _DIGIT_TOKEN_RE.match(bare):
            number = int(bare)
        if number is not None:
            if count is None:
                count = number
            continue
        kept.append(token)
    return " ".join(kept).strip(), count


# =============================================================================
# OpenBot action patterns (single action; borrowed from the family)
# =============================================================================

# "put the bottle in the bin" / "set the mug down"
# label is non-greedy: first in/into/on/onto
PLACE_INTO_RE = re.compile(
    rf'^{_NAV_POLITE}(?:put|place|set)\s+(?:the\s+)?(.+?)'
    rf'\s+(?:in|into|on|onto)\s+(?:the\s+)?(.+)$',
    re.IGNORECASE,
)
# "set the cup down" - release right there.
PLACE_DOWN_RE = re.compile(
    rf'^{_NAV_POLITE}(?:put|set)\s+(?:the\s+)?(.+?)\s+down\s*$',
    re.IGNORECASE,
)

_HAND_PATTERN = (
    rf'^{_NAV_POLITE}(?:open|close)\s+(?:your\s+)?(?:the\s+)?'
    rf'(left|right|both)?\s*(?:hand|hands|gripper|grippers)?\s*$'
)
OPEN_HAND_RE = re.compile(_HAND_PATTERN.replace('open|close', 'open'), re.IGNORECASE)
CLOSE_HAND_RE = re.compile(_HAND_PATTERN.replace('open|close', 'close'), re.IGNORECASE)

# ---------------------------------------------------------------------------
# named arm poses, matched before the simple-
# keyword lane so 'go neutral' is not a marker
# hunt. longest-first, word-boundary anchored.
_POSE_TAIL = r"(?:\s+(?:pose|posture|position|up))?"
# one shared lead-in, outside each branch:
# inside, 'go ready pose' and 'get ready'
# would not both match.
_POSE_LEAD = (
    r"(?:(?:go|get|be|stand|assume|return|back)(?:\s+(?:to|into))?\s+)?"
)
# 'tuck' has no engine verb; both word orders
# sound natural. every part optional.
_POSE_TUCK = (
    r"tuck(?:\s+(?:"
    r"(?:your\s+|the\s+|my\s+)?(?:arms?|hands?)(?:\s+(?:in|up|back))?"
    r"|(?:in|up|back)(?:\s+(?:your\s+|the\s+|my\s+)?(?:arms?|hands?))?"
    r"))?"
)
# 'arms down' ("i want the home posture ... by
# having the arms down") names a pose by
# DESCRIBING it - pose_from_text can't see it.
#
# leads stay inside: they need a body part, so
# no collision with the 'go'->navigate lane.
#
# SAFETY: tokens[0] halt words swallow commands
# ('Stand back up.'). nothing here starts with
# one: arms/hands/put/lower/drop/relax/let.
# 'stand' IS a halt word, so 'stand arms down' is out.
_ARM = r"(?:arms?|hands?)"
_POS = r"(?:\s+(?:your\s+|the\s+|my\s+)?)?"          # optional possessive
_DIR = (
    r"(?:down"
    r"|by\s+(?:your\s+|the\s+|my\s+)?sides?"
    r"|to\s+(?:your\s+|the\s+|my\s+)?sides?"
    r"|at\s+(?:your\s+|the\s+|my\s+)?sides?"
    r")"
)
_POSE_DOWN = (
    # explicit '+' everywhere: a bare literal after
    # a "+ name" chain is a syntax error.
    # lower/drop/relax carry direction - particle optional.
    r"(?:(?:lower|drop|relax)" + _POS + _ARM + r"(?:\s+" + _DIR + r")?"
    # put/let imply no direction - particle required.
    + r"|(?:put|let)" + _POS + _ARM + r"\s+" + _DIR
    # no verb: noun + particle stands alone.
    + r"|" + _ARM + r"\s+" + _DIR
    + r")"
)
#: same source POSE_RE embeds, compiled again
#: so pose_from_text resolves it without a copy.
_POSE_DOWN_RE = re.compile(_POSE_DOWN, re.IGNORECASE)
POSE_RE = re.compile(
    rf"^{_NAV_POLITE}{_POSE_LEAD}(?:"
    rf"{_POSE_TUCK}"
    rf"|{_POSE_DOWN}"
    rf"|(?:the\s+)?ready{_POSE_TAIL}"
    rf"|(?:the\s+)?neutral{_POSE_TAIL}"
    rf"|(?:the\s+)?home{_POSE_TAIL}"
    rf")\s*$",
    re.IGNORECASE,
)


def pose_from_text(text: str) -> Optional[str]:
    """The pose name a phrase names, or None.

    Returns one of 'tuck' | 'ready' | 'neutral' | 'home' — the brain's pose
    vocabulary, NOT the engine's verbs (the backend owns that mapping).
    """
    m = POSE_RE.match(text or "")
    if m is None:
        return None
    said = m.group(0).lower()
    # key off the matched TEXT, not the pattern
    for name in ("tuck", "ready", "neutral", "home"):
        if re.search(rf"\b{name}\b", said):
            return name
    # 'put your arms down' = home, via _POSE_DOWN
    if _POSE_DOWN_RE.search(said):
        return "home"
    return None


# ---------------------------------------------------------------------------
# turn-around: "just the simple turn around
# sweep". the SWEEP is the walk FSM's; the
# brain just names it, no LLM.
#
# before NAV_PATTERNS: specific grammar first
#
# CLOSED alias set, never a free capture -
# resolved via ``known_labels.resolve_alias``.
# a stray tail is refused HERE, so no noun
# becomes a person to hunt.
#
# the extra 'please' allows 'could you please
# turn around'; widening _NAV_POLITE hits all lanes.
_TURN_AROUND_TARGET = r"(?:me|myself|i|us|human|operator|person|someone|somebody)"
TURN_AROUND_RE = re.compile(
    rf"^{_NAV_POLITE}(?:please[\s,]+)?"
    # 'turn around' / 'turnaround' / 'turn-around' /
    # 'turn round' / 'spin around' / 'look around'.
    r"(?:turn(?:\s*-\s*|\s+)?around|turn\s+round|spin\s+around|look\s+around)"
    # optional '... to/and look/find for <target>' tail.
    rf"(?:\s+(?:and\s+|to\s+)?(?:look|search|hunt|find)?\s*(?:for)?\s*"
    rf"(?:the\s+|a\s+)?({_TURN_AROUND_TARGET}))?"
    r"\s*[.!?,;]*\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# simple keyword + LAST-WORD model:
# only these verbs, OBJECT = last word, rest
# stripped. deliberately DUMB: put/place ->
# place at 'here', never containment inference.
# ---------------------------------------------------------------------------
_SIMPLE_ACTION_MAP = {
    'pick': 'pick', 'grab': 'pick',
    'put': 'place', 'place': 'place',
    'walk': 'navigate', 'go': 'navigate', 'find': 'navigate',
}
_SIMPLE_TRAILING_PHRASES = frozenset({"for me", "for us", "to me"})


def simple_keyword_intent(text: str) -> Optional[dict]:
    if not text:
        return None
    toks = [t for t in re.split(r"[^0-9A-Za-z']+", text) if t]
    while toks:
        tail2 = " ".join(toks[-2:]).lower()
        if tail2 in _SIMPLE_TRAILING_PHRASES:
            toks = toks[:-2]
            continue
        if toks[-1].lower() in TRAILING_FILLERS or toks[-1].lower() in ("me", "us"):
            toks.pop()
            continue
        break
    if not toks:
        return None
    verb = None
    for t in toks:
        low = t.lower()
        if low in _SIMPLE_ACTION_MAP:
            verb = low
            break
    if verb is None:
        return None
    obj = toks[-1].lower().strip("'")
    if not obj or obj == verb or obj in TRAILING_FILLERS or obj in ("me", "us", "it", "up"):
        return None
    return {"action": _SIMPLE_ACTION_MAP[verb], "object": obj}


def _strip_trailing_fillers(target: str) -> str:
    target = target.strip().rstrip('.!?,').strip()
    tokens = target.split()
    while tokens and tokens[-1].lower().strip('.,!?') in TRAILING_FILLERS:
        tokens.pop()
    # re-strip: a filler may sit behind punctuation
    # ('the tray, please'), and a label must not carry it.
    return ' '.join(tokens).rstrip('.,!?').strip()


def _normalize_pick_label(label: str) -> str:
    """Strip 'up'/'it' filler tokens from a pick label candidate.

    A LEADING 'up' and TRAILING 'it'/'up' tokens are filler, not objects:
    'pick it up' -> nothing, 'grab up the cup' -> 'the cup'.  Returns ''
    when nothing remains — the caller must treat an empty label as
    NOT_DETECTED (honest refusal; never a pick call without a label)."""
    tokens = label.split()
    while tokens and tokens[0].lower().strip('.,!?') == 'up':
        tokens.pop(0)
    # trailing phrases are filler ("... for me")
    # 'for my friend' keeps its tail - honest refusal.
    while True:
        if len(tokens) >= 2 and ' '.join(tokens[-2:]).lower().strip('.,!?') in (
            "for me", "for us", "to me",
        ):
            tokens = tokens[:-2]
            continue
        if tokens and tokens[-1].lower().strip('.,!?') in ('it', 'up'):
            tokens.pop()
            continue
        break
    return ' '.join(tokens)


# second clause: terminator, comma/semicolon,
# 'and' or 'then'. filler-only tails never veto;
# whisper run-ons ('Pick up the bottle. Let's see
# what...') must not ride in.
_MULTI_CLAUSE_SPLIT_RE = re.compile(
    r'[.!?]+|[,;]+|\band\b|\bthen\b',
    re.IGNORECASE,
)


def multi_clause_veto(capture_text: str) -> bool:
    """True when a captured payload continues into a SECOND clause.

    Operates on the RAW captured text — the label/object/destination
    capture AFTER the action head (``m.group(1)`` / ``m.group(2)``), not
    the whole utterance.  Splits on /[.!?]+|[,;]+|\band\b|\bthen\b/
    (case-insensitive) and vetoes iff SOME part after the first contains
    any token NOT in ``TRAILING_FILLERS`` (parts are tokenized with the
    same punctuation-normalized split the halt lane uses).  Filler-only
    continuations never veto: 'pick up the cup, please', 'put the cup on
    the tray, please' (filler tail), a trailing 'and'/'then' with NO
    continuation ('pick up the cup and'), and a sentence-terminal
    period ('pick up the cup.' — second part is empty) all split to an
    empty or filler-only second part.  A sentence-joined second sentence
    ('pick up the cup. turn it off') vetoes — never a garbage label.
    """
    parts = _MULTI_CLAUSE_SPLIT_RE.split(capture_text)
    for part in parts[1:]:
        tokens = _normalize_for_halt(part)
        if any(token not in TRAILING_FILLERS for token in tokens):
            return True
    return False


def classify_intent(text: str) -> Optional[dict]:
    """Returns an intent dict if a deterministic single action matches, else None.

    Action set (OpenBot adaptation): ``stop`` | ``navigate_to_marker`` |
    ``pick_up`` | ``place`` | ``open_hand`` | ``close_hand`` | ``pose`` |
    ``turn_around``.
    Nav is checked before pick (source order) so "go to X" / "take me to X"
    never classify as picks; "get" is deliberately excluded from pick verbs.
    The POSE lane runs before nav for the opposite reason — 'go neutral' is a
    posture, and nav would read it as a marker named 'neutral'.

    Halt precedence: the explicit-halt lane runs on the NORMALIZED
    utterance (punctuation -> spaces) BEFORE any heard-name stripping,
    scanning the FIRST up-to-4 tokens for any halt word; it is re-checked
    after stripping ('stewart, stop' -> 'stop'); the stripper additionally
    refuses halt words, so 'stop the robot' can never be eaten as a name.
    """
    raw = text.strip()
    # halt lane #1: normalized BEFORE anything else.
    # a comma ('stop, please' / 'please, stop') must
    # never break the match - halt comes FIRST.
    # a negated halt ('don't stop') falls through.
    if _halt_intent(_normalize_for_halt(raw)):
        return {'action': 'stop'}

    cleaned = _strip_heard_name(raw)
    cleaned = _normalize_word_numbers(cleaned)
    cleaned_l = cleaned.lower().rstrip('.!?,').strip()

    tokens = cleaned_l.split()
    while tokens and tokens[-1] in TRAILING_FILLERS:
        tokens.pop()
    cleaned_l = ' '.join(tokens)

    # halt lane #2, after name stripping ('stewart,
    # stop' -> 'stop'). a leading name can't conceal
    # a halt. plain 'stop ...' already hit #1.
    if _halt_intent(_normalize_for_halt(cleaned)):
        return {'action': 'stop'}

    # -- named arm poses ------------------------------------------------
    # ahead of nav and place: 'go home' would be a
    # marker hunt, 'put your arms down' a place.
    if POSE_RE.match(cleaned_l):
        return {'action': 'pose', 'name': pose_from_text(cleaned_l)}

    # -- the turn-around lane (see TURN_AROUND_RE) -------------------------
    # before nav, like the pose lane
    m = TURN_AROUND_RE.match(cleaned_l)
    if m:
        return {'action': 'turn_around', 'target': m.group(1)}

    for pat in NAV_PATTERNS:
        m = pat.match(cleaned)
        if m:
            if multi_clause_veto(m.group(1)):
                # two-step ('kitchen, then bathroom') - drop to LLM
                continue
            target = _strip_trailing_fillers(m.group(1))
            if not target:
                continue
            if all(tok.lower() in _NAV_FILLERS for tok in target.split()):
                # filler-only target ('go to') - skip
                continue
            return {'action': 'navigate_to_marker', 'target': target}

    for pat in PICK_PATTERNS:
        m = pat.match(cleaned)
        if m:
            if multi_clause_veto(m.group(1)):
                # 'remote, put it in the green plate' = TWO steps
                continue
            raw_target = _strip_trailing_fillers(m.group(1))
            # 'up'/'it' are never a label; empty = honest refusal
            target, count = _strip_cardinals(_normalize_pick_label(raw_target))
            if target:
                intent = {'action': 'pick_up', 'target': target}
                if count is not None:
                    intent['count'] = count
                return intent

    m = PLACE_INTO_RE.match(cleaned)
    if m and not (multi_clause_veto(m.group(1)) or multi_clause_veto(m.group(2))):
        # two-step place - drop to the LLM. label group
        # catches multi-object, destination the rest.
        label, count = _strip_cardinals(_strip_trailing_fillers(m.group(1)))
        label = re.sub(r'\s+down$', '', label).strip()
        destination = _strip_trailing_fillers(m.group(2))
        if label and destination:
            intent = {'action': 'place', 'label': label, 'destination': destination}
            if count is not None:
                intent['count'] = count
            return intent
    m = PLACE_DOWN_RE.match(cleaned)
    if m and not multi_clause_veto(m.group(1)):
        # multi-object place-down is two-step - drop to LLM
        label, count = _strip_cardinals(_strip_trailing_fillers(m.group(1)))
        if label:
            intent = {'action': 'place', 'label': label, 'destination': 'here'}
            if count is not None:
                intent['count'] = count
            return intent

    # no payload captured, so no multi-clause veto here
    m = OPEN_HAND_RE.match(cleaned_l)
    if m:
        return {'action': 'open_hand', 'side': m.group(1) or 'right'}
    m = CLOSE_HAND_RE.match(cleaned_l)
    if m:
        return {'action': 'close_hand', 'side': m.group(1) or 'right'}

    return None


# =============================================================================
# intent -> ToolCall (still no LLM involved at all)
# =============================================================================


