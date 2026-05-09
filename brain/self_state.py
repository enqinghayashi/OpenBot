"""brain/self_state.py - the robot's label-keyed memory of what it holds/did.

BRAIN B3a (2026-10-03).  Operator diagnosis (run19): "the current OpenBot do
not have contextual memories".  ``SelfState`` is the ONE place that records
what the robot believes about itself, written ONLY from verifier verdicts via
``record`` (never from model text, never from a raw tool reply).

HONESTY: ``holding`` is INFERRED.  No in-hand sensor exists (the verifier's
pick success is lane-up + target-gone), so ``confidence`` is a CATEGORY
('inferred'), never a number, and every rendered line says "may be".

BATCH 5 (2026-10-04, run21 F2/F4/F5 + run22).  Two inputs that are NOT
verdicts and never write a hand belief: the operator-request TURN
(``note_utterance``, fed by the voice app's utterance hook; utterance_seq 0 =
the hook never ran) and the PENDING PICK (``note_pick_attempt``, from
pick.prepare, because a prepare-time NOT_DETECTED never reaches ``record``).
They scope three pick guards (walk-only turn, task done after a place, a
detection older than the last cancel) and the navigate -> pick next step.
One MEASURED-grip write: ``repick_refusal(grip=...)`` clears an UNKNOWN /
UNCONFIRMED hand when a fresh right-hand read proves it empty.

Lifetime: lives on the SceneGraph (one per voice process), so it survives a
realtime reconnect and a per-run ToolExecutor rebuild; it is LOST on a
process restart.  stdlib + brain.known_labels at import time (the B6 grip
check imports brain.verifier lazily: verifier imports this module); py3.8-safe.
"""
from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Optional

from brain import known_labels

SELF_STATE_MARK = "\nSELF_STATE: "
_PREFIX = "SELF_STATE: "
#: release marker
#: set on an in_progress place once the release chain passed
#: (hand open, measured)
#: the verifier imports this module, so no import here
#: matched verbatim downstream, so keep the wording
PLACE_RELEASED_MARK = "BUT THE OBJECT WAS RELEASED"


# cancel phases
# CANCELLED:<where> names the phase
# values: arm_move, guarded_move, hand_close, sequence_hold
# none of them proves the hand never closed, so this stays
# empty; add one only with engine-side proof
PRE_CLOSE_WHERE = frozenset()  # type: frozenset

_CANCEL_WHERE_RE = re.compile(r"CANCELLED:\s*([A-Za-z0-9_]+)")
# both spellings
# the verifier writes '(phase <where>)'
# the raw reply is not part of the verdict line
_PHASE_RE = re.compile(r"\(phase\s+([A-Za-z0-9_?]+)\)")
_HAND_STATE_RE = re.compile(r"hand_state=(empty|unknown)")


def _cancel_phase_unproven(blob: str) -> bool:
    """True when the text NAMES a cancel phase that is not a proven pre-close
    phase: such a cancel can never be 'empty', whatever else the text says."""
    for rx in (_CANCEL_WHERE_RE, _PHASE_RE):
        for w in rx.findall(blob):
            if cancel_hand_state(w) != "empty":
                return True
    return False


def _reason_field(text: str) -> str:
    """The ``Reason:`` field of a full verdict line, else the text itself."""
    i = text.rfind("|Reason:")
    return text[i + len("|Reason:"):].strip() if i >= 0 else text.strip()


def _is_cancel(reason: str, evidence: str) -> bool:
    """A cancelled pick, on either the short or the full-verdict-line shape."""
    if _reason_field(reason).startswith("cancelled_by_operator"):
        return True
    blob = reason + "\n" + evidence
    return ("cancelled_by_operator" in blob
            or _CANCEL_WHERE_RE.search(blob) is not None)


def parse_cancel_where(text: Any) -> Optional[str]:
    """The ``<where>`` of an engine ``CANCELLED:<where>`` reply, or None."""
    m = _CANCEL_WHERE_RE.search(str(text or ""))
    return m.group(1) if m else None


def cancel_hand_state(where: Optional[str]) -> str:
    """'empty' only for a PROVEN pre-close phase; EVERYTHING else (including
    an unparseable or unknown phase) is 'unknown'."""
    return "empty" if where in PRE_CLOSE_WHERE else "unknown"


def _norm(label: Any) -> str:
    if not label:
        return ""
    return str(known_labels.resolve_alias(str(label))).strip().lower()


def _words(s: str) -> list:
    return re.findall(r"[a-z0-9_]+", s)


# ---- operator-request classifier ----
#: no-hook guard
#: re-picks land 34 s and 41 s after the place
#: 60 s covers both
POST_PLACE_GUARD_S = 60.0
#: pending pick TTL
#: a navigate sweep gives up after ~73 s, doubled plus margin
PENDING_PICK_TTL_S = 180.0

#: cross-body refusal
#: the engine's self-collision token:
#: `ERROR:self_collision verb=.. side=.. x=..m y=..m ... stage=..`
_SELF_COLLISION_RE = re.compile(r"\bself_collision\b")
_SC_FIELDS_RE = re.compile(
    r"side=(left|right) x=(-?[\d.]+)m y=(-?[\d.]+)m .*?move_right_m=([\d.]+)")
#: cross-body TTL
#: no later call clears it, so it lapses on its own
CROSS_BODY_BLOCK_S = 120.0
_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.I)

#: hint window
#: unconfirmed pick: the hint stays alive this long
#: re-picks land 8-60 s after the pick
#: fires every time, not one-shot
UNCONFIRMED_ABSENT_S = 120.0
#: turn window
#: a place destination / release request stays usable this many turns
REQUEST_TURNS_LIVE = 1
#: release words
#: no bare 'hand': it also matches "with your right hand"
_RELEASE_ASK_RE = re.compile(
    r"\b(?:open(?:s|ed|ing)?|releas(?:e|es|ed|ing)|let(?:ting)?\s+(?:it\s+|that\s+)?go|"
    r"drop(?:s|ped|ping)?|give|gives|giving|hand\s+(?:it|that|them)\s+over|"
    r"(?:put|set)\s+(?:it|that|them)\s+down|fingers?)\b")
#: place target
#: "in / into / inside / on / onto <label>", as the operator named it
_DEST_PREP_RE = re.compile(r"\b(?:in|into|inside|on|onto)\b")
#: beside = landmark
#: "in front of the bowl" names no container: the arm can drop only
#: on or into an object, so that is no destination
_DEST_BESIDE_RE = re.compile(
    r"\b(?:next\s+to|beside|near(?:by)?|by|left\s+of|right\s+of|front\s+of|"
    r"behind|around|between|(?:side|back)\s+of|opposite)\b")


def _cross_body_detail(text: str) -> Dict[str, Any]:
    """side / cm from the engine line; {} when absent or not sane.  (No turn
    angle here: this module stays stdlib-light -- re/time/typing -- and the
    pick reason's own hint already names the angle.)"""
    out: Dict[str, Any] = {}
    m = _SC_FIELDS_RE.search(text or "")
    if not m:
        return out
    try:
        out["side"] = m.group(1)
        mv = float(m.group(4))
        if 0.0 < mv < float("inf"):
            out["cm"] = int(round(mv * 100.0))
    except (TypeError, ValueError):
        pass
    return out

_PICK_VERB_RE = re.compile(
    r"\b(?:pick(?:s|ing|ed)?|pickup|grab(?:s|bing|bed)?|"
    r"grasp(?:s|ing|ed)?|take|takes|taking|fetch(?:es|ing|ed)?|"
    r"gets|getting|get|bring(?:s|ing)?|hand(?:s|ing|ed)?|"
    r"gives|giving|give|lift(?:s|ing|ed)?|collect(?:s|ing|ed)?|"
    r"holds|holding|hold|retriev(?:e|es|ing|ed))\b")
_PLACE_VERB_RE = re.compile(r"\b(place|put|drop|set\s+down|release)\b")
#: fetch to me
#: "go get me a bottle", "hand me that". A directed 'get' is
#: neutralised to 'go' first
_FETCH_TO_ME_RE = re.compile(
    r"\b(?:get|fetch|bring|grab|give|hand|pass|deliver|carry|take|retrieve)\b"
    r"[^.?!]{0,80}?\b(?:me|us)\b"
    r"|\b(?:bring|fetch|retrieve|get|grab)\b[^.?!]{0,80}?\b(?:here|over)\b"
    r"|\bgo\s+(?:and\s+)?(?:get|fetch|grab|bring)\b"
    r"|\b(?:fetch|retrieve)\b"
    r"|\b(?:can|could|may)\s+(?:i|we)\s+(?:have|get)\b"
    r"|\bhand\s+(?:it|that|this|them)\s+over\b")
#: "take me to the bowl" = walk
_FETCH_NOT_RE = re.compile(
    r"\bgive\s+(?:me|us)\s+(?:a|one|some)\s+(?:second|sec|moment|minute|min|break|hand|"
    r"chance|try|go)\b|\b(?:take|carry|bring|get)\s+(?:me|us)\s+(?:to|towards?|closer|"
    r"over|back|home|there|here)\b|\bcarry\s+(?:me|us)\b")


def is_fetch_request(text: Any) -> bool:
    """B8l F7: True for a request to bring something to the speaker."""
    t = _GET_WALK_RE.sub("go ", str(text or "").lower())
    if _FETCH_TO_ME_RE.search(t) is None or _FETCH_NOT_RE.search(t) is not None:
        return False
    # transport, not a fetch
    pm = _PLACE_VERB_RE.search(t)
    if pm is not None and _DEST_PREP_RE.search(t, pm.end()) is not None \
            and re.search(r"\b(?:me|us)\b", t[pm.end():]) is None:
        return False
    return True
#: strong pick verbs: may drive navigate -> pick
#: narrower than _PICK_VERB_RE: 'take me to the bowl' and
#: 'hold on' must never become a pick. 'get' is not here:
#: a directed get is a walk, while grasp/retrieve/pickup
#: are real pick requests
_STRONG_PICK_VERB_RE = re.compile(
    r"\b(?:pick(?:s|ing|ed)?|pickup|grab(?:s|bing|bed)?|"
    r"grasp(?:s|ing|ed)?|fetch(?:es|ing|ed)?|bring(?:s|ing)?|"
    r"gives|giving|give|hand(?:s|ing|ed)?|retriev(?:e|es|ing|ed))\b")
#: walk phrase
#: walk verb, up to two fillers, then to / toward(s) / over; or
#: 'come here'. A gap word outside _WALK_FILLER ('move the bottle to
#: the bowl') makes it a transport, never a walk-only request
_WALK_RE = re.compile(
    r"\b(?:walk|go|come|move|head|get)\b(?P<gap>(?:\s+\w+){0,2}?)\s+"
    r"(?:to|towards?|over)\b|\bcome\s+here\b")
#: directed 'get'
#: a walk, not a pick. It is neutralised to 'go' before the verb
#: search; a bare 'get the bottle' is still a pick
_GET_WALK_RE = re.compile(
    r"\bget\s+(?:me\s+|us\s+)?(?=to\b|towards?\b|closer\b|near\b|over\b|"
    r"back\b|here\b|there\b)")
_WALK_FILLER = frozenset((
    "over", "back", "closer", "up", "right", "straight", "forward",
    "forwards", "with", "me", "us", "please", "now", "just", "on", "down",
    "there", "here", "away", "along", "all"))
#: nav recovery
#: pick failures whose recovery is navigate + the same pick:
#: NOT_DETECTED (prepare), the reach refusal, the reach hint text,
#: a transport failure
_NAV_RECOVERY_RE = re.compile(
    r"NOT_DETECTED|reach_ceiling|OUT OF REACH|transport:")
_VERB_WORDS = {"pick_place": "pick and place", "pick": "pick",
               "place": "place"}


def _walk_match(t: str):
    """The first WALK_RE match whose gap is all filler (a walk-only phrase),
    else None."""
    for m in _WALK_RE.finditer(t):
        gap = m.group("gap")
        if gap is None:  # 'come here'
            return m
        if all(w in _WALK_FILLER for w in gap.split()):
            return m
    return None


def classify_utterance(text: Any) -> str:
    """'pick_place' | 'pick' | 'place' | 'walk' | 'other' (deterministic).

    Any pick verb beats walk ('walk to the bottle and pick it up' is a pick);
    'walk' needs a walk-only phrase and no pick/place verb.  A DIRECTED 'get'
    is neutralised to 'go' for the verb search, so 'get to the bottle' and
    'get me closer to the bottle' are walks, while a bare 'get the bottle'
    stays a pick (review r1 MEDIUM)."""
    t = str(text or "").lower()
    tp = _GET_WALK_RE.sub("go ", t)
    pick = _PICK_VERB_RE.search(tp) is not None
    place = _PLACE_VERB_RE.search(tp) is not None
    if pick and place:
        return "pick_place"
    if pick:
        return "pick"
    if place:
        return "place"
    if _walk_match(t) is not None:
        return "walk"
    return "other"


def _label_spans(t: str, aliases: bool = True) -> list:
    """[(start, label)] for every known class named in ``t`` (an optional
    plural 's').  Aliases ('me' -> 'person') only when ``aliases`` and no
    class is named, so 'bring me the bottle' names the bottle."""
    classes = set(known_labels.TRACKED_CLASSES)
    try:
        classes |= set(known_labels.extra_classes())
    except Exception:  # noqa: BLE001 - extras are optional
        pass
    out = []
    for c in classes:
        c = str(c).strip().lower()
        if not c:
            continue
        for m in re.finditer(r"\b%ss?\b" % re.escape(c), t):
            out.append((m.start(), c))
    if not out and aliases:
        for a in known_labels.TARGET_ALIASES:
            if len(a) < 2:  # skip 'i'
                continue
            for m in re.finditer(r"\b%s\b" % re.escape(a), t):
                out.append((m.start(), known_labels.resolve_alias(a)))
    out.sort()
    return out


