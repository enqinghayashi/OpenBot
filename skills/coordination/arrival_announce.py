#!/usr/bin/env python3
"""
arrival_announce.py -- PURE law for async spoken-arrival on the NAV lane
(no rclpy, no socket IO beyond a path constant -- importable and
unit-testable with the stdlib only).

Purpose
-------
The walk coordinator (pick_walk_coordinator.py) publishes one
/navigate_status (std_msgs/String) human line per navigation:

    "REACHED <label>"     (once -- walk_node published REACHED)

Each REACHED is emitted AT MOST ONCE per navigation by the coordinator.
We announce arrival asynchronously by SPEAKING
"I have reached the <label>".  A thin rclpy relay
(skills/coordination/arrival_relay.py) subscribes /navigate_status,
maps a REACHED line through THIS module, and writes the final phrase to
the UNIX socket named by SOCKET_PATH -- which the voice process reads.

At-most-once contract (this module enforces the second, coarser tier):
  * tier 1 -- the coordinator already guarantees each navigation fires
    "REACHED <label>" at most once;
  * tier 2 -- ArrivalGate additionally suppresses a repeat announcement
    for the SAME label within cooldown_s of the last fired one (e.g. a
    retarget to an already-reached label), while a different label is
    always allowed.  So a human hears each arrival once, at most, per
    cooldown window.

The module is pure: no I/O, no clocks (the caller passes `now`), no
global mutable state.  All functions are deterministic in their inputs.
"""

from typing import Optional
import re

# voice reads arrival phrases here
SOCKET_PATH = "/tmp/arrival_announce.sock"

# "REACHED cup", case-insensitive.
# needs the whole line: a standalone
# REACHED hits, one buried in prose
# (an ERROR line) does not.
REACHED_RE = re.compile(r"^\s*REACHED\s+(.+?)\s*$", re.IGNORECASE)


def parse_arrival(status_text: Optional[str]) -> Optional[str]:
    """Return the trimmed, lowercased label of a REACHED arrival line.

    Returns None for empty/None input and for any line that is not a
    REACHED arrival.  The label is lowercased to match the coordinator's
    own lowercase labels (its REACHED lines already carry a lowercase
    label), keeping ArrivalGate keys canonical regardless of the casing
    a test or producer happens to use.
    """
    if not status_text:
        return None
    match = REACHED_RE.search(status_text)
    if match is None:
        return None
    label = match.group(1).strip().lower()
    return label if label else None


