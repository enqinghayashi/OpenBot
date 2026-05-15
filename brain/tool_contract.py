"""Tool contracts: what each skill tool may/must do, how to verify it, and a precondition gate for the agent (stdlib only)."""
from __future__ import annotations  # py3.8: list[str]/dict[...] evaluate at import
                                    # -> 'type' object is not subscriptable
                                    # kills the brain driver AND gateway _verify
                                    # (lazy import, inside try/except) ->
                                    # every motion verdict decodes as VERDICT:error

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolContract:
    """Machine-readable contract for one skill tool (pick/place/navigate/open_hand/close_hand).

    Fields (exact order per shared contract):
      name            - CONTRACTS key, e.g. 'pick'
      preconditions   - statements that must hold BEFORE invocation; those with
                        keywords in _PRECONDITION_KEYWORDS are judgeable from a
                        scene_brief, the rest are runtime checks the agent does directly
      effects         - world-state changes the tool is expected to cause
      post_condition  - single truth statement the verifier must establish
      verify_query    - what the verifier queries/checks (in platform vocabulary)
      timeout_s       - upper bound the agent uses for watchdog/timeout planning
      context         - free-form platform facts an agent needs to drive the tool
    """

    name: str
    preconditions: list[str]
    effects: list[str]
    post_condition: str
    verify_query: str
    timeout_s: float
    context: str
