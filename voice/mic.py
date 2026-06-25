"""Microphone sources: the G1's array multicast, a mock, and laptop audio.

Wire facts (OpenBot protocol reference, devices/g1/mic.py): the audio
board publishes PCM16LE / 16 kHz / mono, already beam-formed to one
channel, as 5120-byte UDP datagrams every 160 ms to multicast group
239.168.123.161:5555.  The socket must set ``SO_REUSEADDR`` before bind
(so the board's rebind does not bounce us) and ``IP_ADD_MEMBERSHIP`` to
join the group on the robot-facing interface.

``G1Mic`` binds in its constructor (the bind fails fast on the robot if
the interface is wrong), so laptop code must not instantiate it - use
``MockMic`` for tests and ``SounddeviceMic`` for laptop demos.  Without
``interface_ip`` the robot-facing interface is auto-detected by
``robot_interface_ip()`` (robot-subnet scan, never a hard-coded name);
a host with no such interface raises ``MicError`` BEFORE binding - the
mic must fail loudly, never join a default interface and hear nothing.
"""

from __future__ import annotations

import array
import fcntl
import logging
import math
import os
import socket
import struct
import threading
from typing import Iterable, Iterator, Optional

log = logging.getLogger(__name__)

MULTICAST_GROUP = "239.168.123.161"
MULTICAST_PORT = 5555
#: what the board really sends:
#: 160 ms of 16 kHz mono PCM16
G1_FRAME_BYTES = 5120
G1_SAMPLE_RATE = 16000
G1_CHANNELS = 1

#: robot's internal subnet. the board that
#: multicasts lives in there, so join on
#: whichever interface is in range - by
#: address, never by name. spec §5 says no
#: an interface name.
ROBOT_SUBNET_PREFIX = "192.168.123."
_SIOCGIFADDR = 0x8915


