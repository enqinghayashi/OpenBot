"""writer_arm_client.py — socket client for the merged writer (v2b).

Single-connection UNIX-socket client (mirrors grip_client.py): retry connect
at __init__, ONE request line -> ONE reply line, WriterError on ERROR: or
empty replies.  Verbs implemented for the pick engine:

    slew_start / wait_slew / slew   raw 14-DOF cosine slews (`arm` verb;
                                    DONE = slew STARTED, so wait_slew polls
                                    `armstate` until active == 0)
    pose_start / pose               named writer poses (ready/lift/tuck/...)
    freeze                          halt the in-flight slew at its current
                                    interpolated position (contact-stop)
    q / get_current_dual_arm_q      lowstate arm q slice (motors 15..28)
    armstate                        (state_name, active, rem_T)
    hold_rc on|off                  RC-stick interlock (stationary guarantee)
    payload on|off                  writer payload gravity-compensation flag

The engine uses this client as the drop-in replacement for
G1_29_ArmController: the tau argument of the old ctrl_dual_arm is DROPPED
(the writer's validated regime writes tau=0), and there is no arm_sdk DDS
writer anywhere on this process.
"""

import socket
import time

import numpy as np

SOCK_PATH = "/tmp/merged_writer.sock"
_REPLY_TIMEOUT = 5.0        # per request; slews run async
_SLEW_POLL = 0.02           # armstate poll interval while slewing
_DEFAULT_CONNECT_RETRIES = 5
_DEFAULT_RETRY_DELAY = 0.5


class WriterError(Exception):
    """A socket-level or writer-rejected (ERROR:) failure."""


class WriterArmClient:
    def __init__(self, sock_path=SOCK_PATH, connect_retries=_DEFAULT_CONNECT_RETRIES,
                 retry_delay=_DEFAULT_RETRY_DELAY):
        self.sock_path = sock_path
        self._sock = None
        self._file = None
        last = None
        for attempt in range(connect_retries):
            try:
                self._connect()
                return
            except (OSError, EOFError) as e:
                last = e
                if attempt < connect_retries - 1:
                    time.sleep(retry_delay)
        raise WriterError(
            f"connect to {sock_path} failed after {connect_retries} tries: {last}")

    def _connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(_REPLY_TIMEOUT)
        s.connect(self.sock_path)
        self._sock = s
        self._file = s.makefile("r", encoding="utf-8")

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    # --- under the hood ---

    def _cmd(self, line, timeout=_REPLY_TIMEOUT):
        """Send one request line; return the stripped reply line.

        Raises WriterError on an ERROR: reply or a dropped connection.
        """
        if self._sock is None:
            raise WriterError("not connected")
        self._sock.settimeout(timeout)
        self._sock.sendall(line.encode("utf-8") + b"\n")
        reply = self._file.readline()
        if not reply:
            raise WriterError(f"writer closed the connection on: {line!r}")
        reply = reply.strip()
        if reply.startswith("ERROR:"):
            raise WriterError(reply[len("ERROR:"):])
        return reply

    # --- raw slews, all 14 DOFs ---

