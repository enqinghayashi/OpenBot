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


