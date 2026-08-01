#!/usr/bin/env python3
"""
arrival_relay.py -- thin rclpy bridge between /navigate_status and the
voice process (NAV lane; THIS is the only file that imports rclpy).

The walk coordinator publishes one human line per navigation EVENT on
/navigate_status ("SEARCHING <label>", "DETECTED <label> ONCE, ...",
"APPROACHING d=.. m", "REACHED <label>", "NOT_FOUND <label>").  This node
subscribes to that topic, maps each line through the PURE law module
arrival_announce.py, and writes the resulting spoken phrase as one
newline-terminated line to the UNIX STREAM socket at
arrival_announce.SOCKET_PATH.

PHASE NARRATION (operator directive 2026-09-28).  Until this change the
relay spoke ONLY the arrival, so a search that took 77 s -- the FSM
sweeping, finding, losing and re-finding the object -- was silent from the
operator's side and sounded like a hang.  The walk lane had been reporting
every phase correctly all along; nothing new is measured, only spoken:
"Let me sweep around and look for the {label}." -> "Oh -- I see it." ->
"Walking over to it now." -> "I have reached the {label}."  See
arrival_announce.PHASE_PHRASES / PhaseAnnouncer.

The voice process (a sibling worker owns voice/) reads phrases from that
socket -- it connects, reads, and reconnects; it does NOT run here.  The
at-most-once guarantee has two tiers (see arrival_announce.py): the
coordinator emits each event at most once per navigation, and the law's
gates additionally suppress a repeat inside their cooldown window.
Announcements are best-effort: if no voice client is connected, or a send
would block, the phrase is dropped -- the voice side reconnects for the
next one.

Run (sourced ROS2 env):
    python3 skills/coordination/arrival_relay.py
"""

import os
import socket
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# sibling module; import works from any CWD
try:
    import arrival_announce
except ImportError:
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import arrival_announce


class ArrivalRelay(Node):
    """Subscribe /navigate_status -> parse -> gate -> UNIX socket phrase."""

