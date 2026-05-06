"""brain/preflight.py — the preflight RESOURCE gate (S10 / plan D0).

Stdlib only; a PURE function of its inputs (no sockets, no ROS, no state).
The agent's scenario-local pre-hook calls this BEFORE any motion; a denial
means the run must NOT move (no engine pick, no writer verb, no locomotion).

PLAN THRESHOLDS — this module is the encoding of the plan's D0 resource
law; the numbers below come from the plan, never invented here:

  * available memory headroom  >= 1.5 GiB   (MIN_AVAILABLE_MEM_BYTES)
    The brain + gateway + agent must keep >= 1.5 GB of free RAM so a
    detection burst or a depth re-projection cannot OOM the box mid-task.
  * load average               < 6.0       (LOAD_AVG_MAX, the D2 gate)
    A 1-minute load average AT or ABOVE 6.0 is the D2 deny line: the CPU
    is saturated and a realtime robot lane must not start motion under it.
  * depth RSS                   <= 500 MiB   (DEPTH_RSS_MAX_BYTES)
    The Part-7.5 depth OOM design: the depth projector is the one heavy
    slave, capped at 500 MiB resident BEFORE the brain+slaves split (the
    brain and the slaves run as SEPARATE processes; the depth slave is the
    one that can blow up).
  * WRITER-NEVER-LARGEST (only when ``other_rss_bytes`` is supplied)
    The merged-writer process must be STRICTLY below the largest OTHER
    resident (the brain / slave split) — the writer must never be the
    biggest RSS consumer on the box, because the writer owns the
    locomotion lane (hold_rc / cmd_set / arm) and a swap-thrash THERE is a
    motion safety hazard.  When the caller omits ``other_rss_bytes`` the
    check is SKIPPED and the reason says so (no data to compare against —
    never guess).

REQUIRED INPUTS (fail-loud: a missing key is ``(False, "missing input:
<key>")`` — the caller must never run a motion on an incomplete resource
picture):
    available_mem_bytes   int      free RAM headroom in bytes
    writer_rss_bytes      int      merged-writer resident set in bytes
    load_avg              float    the 1-minute load average
    depth_rss_bytes       int      depth-projector RSS in bytes
    other_rss_bytes       dict     OPTIONAL {name -> bytes} of the OTHER
                                   processes (brain/slaves) for the
                                   writer-never-largest law.

OUTPUT: ``(ok, reason)``.  Every violation contributes ONE line to the
reason (joined with '; '); ok iff ZERO violations.  On ok the reason names
the evaluated resources and any skipped check.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

MIN_AVAILABLE_MEM_BYTES = int(1.5 * 1024 ** 3)  # no motion unless 1.5 GiB is free
LOAD_AVG_MAX = 6.0                              # 6.0+ denies
DEPTH_RSS_MAX_BYTES = int(500 * 1024 ** 2)      # depth cap 500 MiB, no OOM

_REQUIRED_KEYS = (
    "available_mem_bytes",
    "writer_rss_bytes",
    "load_avg",
    "depth_rss_bytes",
)


def _as_float(value: Any, key: str, violations: list) -> float:
    """Coerce a resource value to float; a non-numeric value is a violation."""
    try:
        return float(value)
    except (TypeError, ValueError):
        violations.append("%s must be numeric, got %r" % (key, value))
        return float("nan")


