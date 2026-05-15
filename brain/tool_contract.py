"""Tool contracts: what each skill tool may/must do, how to verify it, and a precondition gate for the agent (stdlib only)."""
from __future__ import annotations  # py3.8: list[str]/dict[...] evaluate at import
                                    # -> 'type' object is not subscriptable
                                    # kills the brain driver AND gateway _verify
                                    # (lazy import, inside try/except) ->
                                    # every motion verdict decodes as VERDICT:error

import re
from dataclasses import dataclass


