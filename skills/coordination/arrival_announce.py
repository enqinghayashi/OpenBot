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


def speech_text(label: str) -> str:
    """The spoken phrase announcing arrival at `label`."""
    return "I have reached the " + label + "."


# walk-phase narration: each parsed
# phase maps to the phrase spoken for it.
#
# operator: "maybe have the
# robot reporting like doing a
# sweep search to find the object,
# then ohh stop, i saw the object,
# then walking towards it.
# sound like human."
#
# measured: walk lane fine,
# every phase on /navigate_status.
# voice lane silent until
# arrival. a 77 s search
# felt like a hang.
# same events, just spoken.
#
# the table is the only place
# wording lives. a phase in the
# parser but absent here is
# deliberately UNSPOKEN.
# parse_phase still reports it,
# phase_speech returns None.
# nothing becomes speech
# by accident.

#: phase -> phrase. "{label}" filled
#: from the coordinator's own line; a
#: template needing a label is skipped
#: when there is none.
#:
#: ASCII only, on purpose. these go
#: to piper with no sanitizer, so an
#: em dash or curly quote has no rule.
#: "Oh, I see it." keeps the beat, no
#: mispronunciation on the first search.
PHASE_PHRASES = {
    "searching":   "Let me sweep around and look for the {label}.",
    "detected":    "Oh, I see it.",
    "approaching": "Walking over to it now.",
    "reached":     "I have reached the {label}.",
    "not_found":   "I could not find the {label}.",
    "lost":        "I lost sight of it. Let me sweep again.",
}

#: phases whose template has a label.
#: explicit so the label-less ones
#: (approaching, lost) keep their phrase.
_LABELLED_PHASES = frozenset({"searching", "reached", "not_found"})

#: a phase line that ENDS a nav.
#: clears the once-per-nav set, so the
#: next nav speaks its own phases.
_TERMINAL_PHASES = frozenset({"reached", "not_found", "cancelled", "stopped"})

#: ordered: first match wins, so a
#: line with two keywords reads as the
#: specific one. every pattern is
#: anchored and single-line (`.` can't
#: span newlines). that keeps the
#: multi-line target-check TABLE -
#: anything after it can look like
#: anything - from getting spoken.
_PHASE_RES = (
    ("detected",    re.compile(r"^\s*DETECTED\b(?:\s+(.+?))?\s+ONCE\b", re.I)),
    ("not_found",   re.compile(r"^\s*NOT_FOUND\s+(.+?)\s*$", re.I)),
    ("reached",     re.compile(r"^\s*REACHED\s+(.+?)\s*$", re.I)),
    ("searching",   re.compile(r"^\s*SEARCHING\s+(.+?)\s*$", re.I)),
    ("lost",        re.compile(r"^\s*DETECTION LOST\b", re.I)),
    ("approaching", re.compile(r"^\s*APPROACHING\b", re.I)),
    # CANCELLED has a label
    # ("CANCELLED <label>").
    # STOPPED doesn't.
    # captured anyway, so a
    # reader doesn't think
    # CANCELLED is label-less.
    ("cancelled",   re.compile(r"^\s*CANCELLED\s*(.*?)\s*$", re.I)),
    ("stopped",     re.compile(r"^\s*STOPPED\b", re.I)),
)


def parse_phase(status_text: Optional[str]) -> Optional[tuple]:
    """(phase, label) for a walk-progress line, or None.

    The label is trimmed and lowercased like ``parse_arrival``'s, and is None
    for the phases whose line does not carry one (``approaching``, ``lost``,
    ``stopped``).  Returns None for anything else -- PENDING acks, the
    three-confirmation lines, the target-check table, ERROR/WARNING lines and
    every non-navigate line.
    """
    if not status_text:
        return None
    for phase, rx in _PHASE_RES:
        match = rx.match(status_text)
        if match is None:
            continue
        label = None
        if match.groups():
            raw = match.group(1)
            label = raw.strip().lower() if raw else None
            label = label or None
        return phase, label
    return None


def phase_speech(phase: str, label: Optional[str] = None) -> Optional[str]:
    """The spoken phrase for `phase`, or None when it is not spoken.

    A labelled template with no label returns None rather than speaking a
    sentence with a hole in it.
    """
    template = PHASE_PHRASES.get(phase)
    if template is None:
        return None
    if phase in _LABELLED_PHASES:
        if not label:
            return None
        return template.replace("{label}", label)
    return template


