"""ToolExecutor — G1Agent's tool-call execution machinery, extracted from
brain/agent/agent.py (OpenBot v4 L1 slice 1).  G1Agent._handle_tool_call is
now a one-line delegation to ToolExecutor.execute(); the result text is
byte-identical to the pre-extraction agent by design (same bodies, same
strings, same semantics — the only signature change is the entry point,
``_handle_tool_call(self, call)`` -> ``execute(self, name, args)``, and the
result is carried out as a ToolOutcome whose ``text`` is exactly what the
caller emits).  A fresh executor is built per run() so the per-utterance
failure-budget reset happens at the same point in run()'s sequence as
before, while the persistent SkillVerifier keeps living in the hooks
(navigate-epoch guard survives).  Stdlib-only: imports from the brain
packages and brain.agent (fastpath, tools) — never third-party modules.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..context import Context, PendingAction
from ..hooks import HookResult, MOTION_TOOLS
from ..observation import WorldObservation
from ..self_state import SELF_STATE_MARK
from ..verifier import Verdict, containment_predrop_snapshot
from .tools import envelope_to_str, fail

log = logging.getLogger(__name__)

# how far the in-call re-observe loop goes.
# containment wants TWO CONSECUTIVE fresh
# in-region frames, ONE dropped_at epoch,
# inside RE_OBSERVE_WINDOW_S (brain/verifier.py).
# a new place re-stamps the epoch,
# so re-running the tool never builds the pair.
# one observation never certifies.
# so: re-observe INSIDE one call.  settle
# spacing keeps each frame in the window,
# packet inside POST_DROP_FRESHNESS_S when judged.
# past the cap: honest in_progress,
# never silence, never a made-up success.
_RE_OBSERVE_SETTLE_S = 1.4
_MAX_RE_OBSERVES = 2

# failure-exhaustion prefix.  one source of truth:
# brain/agent/agent.py imports it from here
# (can't import back — circular)
COMPLETE_FAIL_PREFIX = "I could not complete the task: "

#: the pick reply carries this during a fetch
#: (tools.py _FETCH_NEXT_STEP).  walk home after
_FETCH_MARK = "NEXT_STEP=fetch_return"
#: place handover opened the hand in place
#: (tools.py Tools.HANDOVER_OPEN_MARK).  judge as open
_HANDOVER_OPEN_MARK = "HANDOVER=open"

# MOTION_TOOLS speaks the writer/grip verbs;
# 'open'/'close' are the platform verbs behind
# open_hand/close_hand.  place_at_spot -> 'place'
# (engine 'drop', same release-only motion),
# so gate, safety and verifier treat it as place
_PLATFORM_VERB = {
    "pick": "pick",
    "place": "place",
    "place_at_spot": "place",
    "navigate": "navigate",
    "open_hand": "open",
    "close_hand": "close",
}

_VERDICT_RE = re.compile(r"VERDICT:(success|failure|in_progress)\|Evidence:(.*)", re.DOTALL)
_REASON_RE = re.compile(r"\|Reason:(.*)$", re.DOTALL)


@dataclass
class ToolOutcome:
    """One tool execution's outcome, shaped for the caller's branching.

    ``text`` is the exact result text the caller emits (verdict line or
    envelope string — byte-identical to the pre-extraction
    _handle_tool_call return).  ``kind``/``reason`` carry the
    _classify_result classification of that text, and ``terminal`` is True
    when the result exhausted the failure budget (the caller then emits
    the COMPLETE_FAIL_PREFIX text, which the executor also mirrors on its
    own ``_terminal_text`` slot).
    """

    text: str
    kind: str
    reason: str = ""
    terminal: bool = False


