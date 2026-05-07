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


def preflight_resource_gate(inputs: Dict[str, Any]) -> Tuple[bool, str]:
    """Pure resource gate: ``(ok, reason)`` for one resource snapshot.

    Required keys fail-loud when missing: ``(False, "missing input: <key>")``.
    ``other_rss_bytes`` is optional; when present the writer must be STRICTLY
    below its largest value (writer-never-largest).  Every violation is one
    line of the reason; ok iff zero violations.
    """
    for key in _REQUIRED_KEYS:
        if key not in inputs or inputs[key] is None:
            return False, "missing input: %s" % key

    violations: list = []

    available = _as_float(inputs["available_mem_bytes"], "available_mem_bytes",
                          violations)
    writer = _as_float(inputs["writer_rss_bytes"], "writer_rss_bytes",
                       violations)
    load = _as_float(inputs["load_avg"], "load_avg", violations)
    depth = _as_float(inputs["depth_rss_bytes"], "depth_rss_bytes", violations)

    # spare-RAM law
    if available < MIN_AVAILABLE_MEM_BYTES:
        violations.append(
            "available memory %.2f GiB < %.2f GiB (MIN_AVAILABLE_MEM_BYTES)"
            % (available / 1024 ** 3, MIN_AVAILABLE_MEM_BYTES / 1024 ** 3))

    if load >= LOAD_AVG_MAX:
        violations.append(
            "load average %.2f >= %.1f (LOAD_AVG_MAX, D2 gate)"
            % (load, LOAD_AVG_MAX))

    if depth > DEPTH_RSS_MAX_BYTES:
        violations.append(
            "depth RSS %.2f MiB > %.2f MiB (DEPTH_RSS_MAX_BYTES)"
            % (depth / 1024 ** 2, DEPTH_RSS_MAX_BYTES / 1024 ** 2))

    # writer-never-largest, only if other_rss_bytes is given
    # absent -> skip it; never guess a target
    writer_note: str = ""
    others = inputs.get("other_rss_bytes")
    if isinstance(others, dict) and others:
        other_values = [v for v in others.values()
                        if isinstance(v, (int, float)) and v is not None]
        if other_values:
            max_other = max(other_values)
            if not (writer < max_other):
                violations.append(
                    "writer RSS %.2f MiB is NOT strictly below the largest "
                    "other resident %.2f MiB (writer-never-largest law)"
                    % (writer / 1024 ** 2, max_other / 1024 ** 2))
            else:
                writer_note = (
                    "; writer-not-largest ok (largest other %.2f MiB)"
                    % (max_other / 1024 ** 2))
        else:
            writer_note = ("; writer-never-largest skipped (other_rss_bytes "
                           "carries no numeric bytes values)")
    else:
        writer_note = ("; writer-never-largest skipped (other_rss_bytes "
                       "absent)")

    if violations:
        return False, "; ".join(violations)

    ok_reason = (
        "resources within plan thresholds: available %.2f GiB, "
        "writer %.2f MiB, load %.2f, depth %.2f MiB"
        % (available / 1024 ** 3, writer / 1024 ** 2, load, depth / 1024 ** 2)
    )
    return True, ok_reason + writer_note
