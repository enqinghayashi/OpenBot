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


def robot_interface_ip() -> Optional[str]:
    """The IPv4 address of the interface on the robot subnet, or None.

    Joining a multicast group with ``INADDR_ANY`` picks the host's
    *default* interface — which is not the robot-facing one when the box
    also has wifi.  Measured 2026-09-05 on Stuart's box (default route
    via wlan0 192.168.0.x, robot subnet on eth0): a join via 0.0.0.0
    hears NOTHING on 239.168.123.161:5555 (0 frames/8 s); the same join
    passed 192.168.123.164 (eth0) yields the 160 ms/5120 B stream.  The
    reference (OpenBot devices/g1/connection.py) solves this by
    scanning interfaces for the robot-subnet address; ported with raw
    ioctl so the pin holds with stdlib only (no netifaces).
    """
    try:
        names = sorted(os.listdir("/sys/class/net"))
    except OSError:
        return None
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for name in names:
            try:
                packed = struct.pack("256s", name[:15].encode("utf-8"))
                address = socket.inet_ntoa(
                    fcntl.ioctl(probe.fileno(), _SIOCGIFADDR, packed)[20:24]
                )
            except OSError:
                continue
            if address.startswith(ROBOT_SUBNET_PREFIX):
                return address
        return None
    finally:
        probe.close()


class MicError(RuntimeError):
    """A microphone could not be opened or read."""


class G1Mic:
    """The G1 microphone array, as synchronous 5120-byte PCM16 frames.

    Constructor binds UDP multicast (``SO_REUSEADDR`` + ``IP_ADD_MEMBERSHIP``)
    so errors surface immediately.  ``iter_frames()`` yields exactly
    ``frame_bytes`` per frame, reassembling if a datagram ever splits a
    frame; ``close()`` unblocks a blocked ``iter_frames()`` and releases the
    socket.  All socket access is serialised by one lock, so a reader and
    ``close()`` may run on different threads.
    """

    def __init__(
        self,
        group: str = MULTICAST_GROUP,
        port: int = MULTICAST_PORT,
        interface_ip: Optional[str] = None,
        frame_bytes: int = G1_FRAME_BYTES,
        recv_timeout_s: float = 2.0,
    ) -> None:
        self.group = group
        self.port = port
        if interface_ip is None:
            interface_ip = robot_interface_ip()
            if interface_ip is None:
                raise MicError(
                    "no interface carries an address on the robot subnet "
                    "{}0/24 — the G1 mic multicast group is reachable only "
                    "on the robot-facing interface, and an INADDR_ANY join "
                    "would silently hear nothing (see G1Mic docstring). "
                    "Boot with the robot interface up, or pass "
                    "interface_ip explicitly.".format(ROBOT_SUBNET_PREFIX)
                )
        self.interface_ip = interface_ip
        self.frame_bytes = frame_bytes
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock = self._make_socket(recv_timeout_s)
        log.info(
            "g1-mic: joined %s:%d via %s",
            self.group,
            self.port,
            self.interface_ip,
        )

