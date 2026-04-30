"""The DETECTABLE-OBJECT VOCABULARY — what perception can actually publish.

WHY THIS MODULE EXISTS (operator directive, 2026-09-26)
------------------------------------------------------
*"there are currently known detectable objects label lists, so the ai agent
should be aware of what is detectable and will be detected in the navigate
task such that when interpreting human speech instruction, the ai
automatically selects the correct label ... by comparing the known
detectable object label lists"*

Measured before this module: `grep -rn "TRACKED_CLASSES|COCO_NAMES"`
across `brain/ voice/ gateway/ tests/` returned **NOTHING**.  The agent
interpreted every utterance with no idea which labels can exist, so it
could not tell "the mug is not something I can see" (a REAL capability
fact, worth saying out loud) from "the mug is not in front of me right
now" (a scene fact).  Those are different answers and the model had only
one of them available.

THE AUTHORITY IS THE PERCEPTION NODE, NOT THIS FILE
---------------------------------------------------
`skills/perception/yolo_node_SEG_version.py` carries TWO lists, and they
are NOT the same thing:

  * `COCO_NAMES`   — the model's full 80, used ONLY as its own index
                     lookup.  NOT a promise that all 80 are published.
  * `TRACKED_CLASSES` — the PUBLISH ALLOWLIST.  This is the real answer
                     to "what can the robot detect?", and it is 20 names.

Curated 2026-09-07 (the node's own comment): the 2026-09-06 additions were
mostly removed because ROOM-SCALE masks — `dining table`, `tv`, `couch`,
`bed` — cover the whole frame and fed ~96k points into the depth SVD,
which OOM'd the 15 GB box at 69 GiB.  Food and pets were dropped as well.
So the small list is a HARD-WON safety fact, not an arbitrary subset: the
absence of `dining table` is the reason the depth pipeline survives.

Because that node imports `rclpy` it CANNOT be imported here (or on the
laptop), so this file carries a COPY and `load_tracked_classes_from_seg_node()`
re-reads the original by AST parse.  The drift guard in
`tests/test_known_labels.py` asserts the two are equal, so this copy can
never silently diverge from what the robot actually publishes.

DO NOT widen this list here.  Widening it in the brain changes nothing on
the robot and would make the agent claim capability it does not have — the
node is the only place that decides, and a room-scale class must be
re-admitted there with the memory problem solved first.
"""

import ast
import os
import threading
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Set

#: copy of `TRACKED_CLASSES` in
#: skills/perception/yolo_node_SEG_version.py
#: 20 names - the drift guard pins them.
TRACKED_CLASSES: FrozenSet[str] = frozenset({
    'bottle', 'chair', 'cup', 'laptop', 'cell phone', 'book', 'keyboard',
    'mouse', 'remote', 'clock', 'vase', 'potted plant', 'person', 'apple',
    'backpack', 'sports ball',
    'bowl', 'suitcase', 'handbag', 'fork',
})

#: display order. pick first, room last.
#: cosmetic - the set is what counts.
_DISPLAY_ORDER: List[str] = [
    'bottle', 'cup', 'bowl', 'fork', 'apple',
    'book', 'keyboard', 'mouse', 'remote', 'cell phone',
    'laptop', 'clock', 'vase', 'potted plant',
    'backpack', 'suitcase', 'handbag', 'sports ball',
    'chair', 'person',
]


def sorted_classes() -> List[str]:
    """The tracked classes in display order (unknown extras appended last)."""
    known = [c for c in _DISPLAY_ORDER if c in TRACKED_CLASSES]
    extra = sorted(TRACKED_CLASSES.difference(known))
    return known + extra


def describe() -> str:
    """One line naming every detectable class, for the model's context."""
    return ("DETECTABLE (the only labels perception can ever report): "
            + ", ".join(sorted_classes()))


def is_detectable(label: str) -> bool:
    """True when ``label`` is a class the perception lane can publish.

    Case/whitespace-insensitive.  A colour-prefixed form ('red cup') is
    reduced to its class BEFORE the test, because the colour prefix is a
    per-instance display adornment, never a class of its own.
    """
    if not label:
        return False
    text = label.strip().lower()
    if text in TRACKED_CLASSES:
        return True
    # open-vocab extras. lane ON + node
    # confirmed, else never fires.
    if text in _EXTRA_CLASSES or EXTRA_ALIASES.get(text) in _EXTRA_CLASSES:
        return True
    try:
        from .observation import label_base_class  # local: keep import light
    except Exception:  # noqa: BLE001 — if the helper is missing we still must not break the test
        return False
    try:
        base = label_base_class(text)
        return (base in TRACKED_CLASSES or base in _EXTRA_CLASSES
                or EXTRA_ALIASES.get(base) in _EXTRA_CLASSES)
    except Exception:  # noqa: BLE001
        return False


#: spoken self-references ('me', 'myself', 'i', 'human', 'operator')
#: map to the tracked class 'person' - one word for the human, so
#: just language -> class; every target is tested as publishable.
TARGET_ALIASES: Dict[str, str] = {
    "me": "person",
    "myself": "person",
    "i": "person",
    "human": "person",
    "operator": "person",
}


def resolve_alias(label: str) -> str:
    """Map a self-reference ('me') to its tracked class ('person').

    IDENTITY for every other label, INCLUDING its original casing and
    whitespace: this sits on the target-resolution path for every tool, so it
    must not quietly rewrite labels it does not own.  Case- and
    whitespace-insensitive in the LOOKUP only -- the alias arrives from spoken
    language, where the casing is whatever the model typed.
    """
    if not label:
        return label
    key = label.strip().lower()
    hit = TARGET_ALIASES.get(key)
    if hit is not None:
        return hit
    # learned phrases ('trash can' ->
    # 'bin'). empty by default, so with
    # lane OFF this is still the identity.
    # leading article tolerated.
    if EXTRA_ALIASES:
        hit = EXTRA_ALIASES.get(key)
        if hit is None:
            parts = key.split()
            while parts and parts[0] in ("the", "a", "an"):
                parts.pop(0)
            hit = EXTRA_ALIASES.get(" ".join(parts))
        if hit is not None:
            return hit
    return label


# --- open-vocab EXTRAS (brain/look_for_tool.py) ---
#
# TRACKED_CLASSES above stays the literal
# the drift guard reads by AST.
#
# lane `optional_tools.open_vocab_lane`,
# default OFF. adds single-token labels
# ('bin', 'pencil_case') the node
# CONFIRMED, plus phrases pointing at
# them ('trash can' -> 'bin'). both
# containers EMPTY by default; only the
# lane fills them, so lane OFF behaves
# byte-for-byte as before (pinned by
# tests/test_known_labels.py).
#
# describe() skips extras: it renders the
# STATIC capability line.

#: phrase -> extra label.
#: EMPTY by default.
EXTRA_ALIASES: Dict[str, str] = {}

#: confirmed extra labels.
#: EMPTY by default.
_EXTRA_CLASSES: Set[str] = set()

_EXTRA_LOCK = threading.Lock()


def extra_classes() -> FrozenSet[str]:
    """Snapshot of the currently registered extra labels."""
    return frozenset(_EXTRA_CLASSES)


def register_extra_classes(names: Iterable[str],
                           aliases: Optional[Mapping[str, str]] = None) -> None:
    """Register confirmed extra labels and phrase aliases to them.

    A BASE class is never an extra (D4: the base vocabulary is always on and
    is the TRACKED string verbatim) -> ValueError, loud.  An alias whose
    target is not a registered extra -> ValueError (an alias must resolve to
    something perception can actually publish).  Alias keys that are base
    class names or self-reference aliases ('me') are skipped: those words
    already mean something on every tool path and must not be rewritten.
    """
    clean = []
    for name in names:
        n = str(name).strip().lower()
        if not n:
            continue
        if n in TRACKED_CLASSES:
            raise ValueError("%r is a base class, never an extra" % n)
        clean.append(n)
    with _EXTRA_LOCK:
        # validate all, then apply: a
        # ValueError leaves both containers
        # untouched, nothing half-registered.
        targets = set(_EXTRA_CLASSES) | set(clean)
        pending = {}
        for phrase, target in dict(aliases or {}).items():
            key = str(phrase).strip().lower()
            tgt = str(target).strip().lower()
            if tgt not in targets:
                raise ValueError("alias %r targets %r, which is not a registered "
                                 "extra" % (key, tgt))
            if not key or key == tgt or key in TRACKED_CLASSES \
                    or key in TARGET_ALIASES:
                continue
            pending[key] = tgt
        _EXTRA_CLASSES.update(clean)
        EXTRA_ALIASES.update(pending)


def unregister_extra_classes(names: Iterable[str]) -> None:
    """Drop extra labels AND every alias that points at them."""
    gone = {str(n).strip().lower() for n in names}
    with _EXTRA_LOCK:
        _EXTRA_CLASSES.difference_update(gone)
        for key in [k for k, v in EXTRA_ALIASES.items() if v in gone]:
            del EXTRA_ALIASES[key]


def clear_extra_classes() -> None:
    """Back to the default (empty) state.  Clears IN PLACE: callers that
    imported the containers by name keep seeing the live (empty) objects."""
    with _EXTRA_LOCK:
        _EXTRA_CLASSES.clear()
        EXTRA_ALIASES.clear()
