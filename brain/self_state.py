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


