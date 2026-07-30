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

    def slew_start(self, target14, duration=None, tau=None):
        """Start a cosine slew toward an arbitrary 14-DOF target.

        Returns immediately (DONE = slew STARTED).  Duration follows the
        daemon's move_arm_smooth formula unless given (engine passes it for
        profile alignment).  tau (optional 14-vector) becomes the writer's
        arm gravity feedforward for this slew — the standalone-lane physics
        (sol_tauff = pin.rnea at the target pose), applied every 250 Hz.
        Completion: wait_slew() / armstate().
        """
        if len(target14) != 14:
            raise WriterError(f"slew_start needs 14 floats, got {len(target14)}")
        if tau is not None and len(tau) != 14:
            raise WriterError(f"slew_start tau needs 14 floats, got {len(tau)}")
        line = "arm " + " ".join(f"{v:.6f}" for v in target14)
        if duration is not None:
            line += f" {duration:.3f}"
        if tau is not None:
            line += " " + " ".join(f"{v:.6f}" for v in tau)
        self._cmd(line)

    def wait_slew(self, timeout=60.0, abort=None):
        """Poll armstate until the in-flight slew completes (active == 0).

        Raises WriterError on timeout (the writer keeps slewing; the caller
        may freeze() or keep waiting).

        ``abort`` is an optional zero-argument callable polled once per
        iteration (``_SLEW_POLL`` = 20 ms).  It is expected to RAISE when the
        move should be abandoned — the engine passes
        ``engine_cancel.check``, which raises ``Cancelled``.  The client does
        NOT freeze on the way out: the writer's slew is left running so the
        caller's unwind can freeze it at the position it had actually
        reached, which is the same contract the contact-stop path uses.
        ``Cancelled`` derives from ``BaseException``, so it passes through
        this method and through every ``except Exception`` above it.

        This is the single choke point that makes EVERY blocking arm move in
        the engine cancellable: all of them reach the writer through ``slew``
        / ``pose`` / ``wait_slew`` (pick_engine_v3.py has exactly two
        ``arm.slew*`` call sites, both here).
        """
        deadline = time.monotonic() + timeout
        while True:
            if abort is not None:
                abort()
            state, active, rem_T = self.armstate()
            if not active:
                return
            if time.monotonic() >= deadline:
                raise WriterError(
                    f"slew not complete after {timeout:.1f}s "
                    f"(state={state}, active={active}, rem_T={rem_T:.2f})")
            time.sleep(_SLEW_POLL)

    def slew(self, target14, duration=None, tau=None, abort=None):
        """slew_start + wait_slew: blocking move to an arbitrary 14-DOF
        target (the old move_arm_smooth)."""
        if abort is not None:
            # before the slew: if a cancel is already pending, don't start
            # the motion (writer would slew at a pose nobody picked until
            # the caller's unwind freezes it)
            abort()
        self.slew_start(target14, duration=duration, tau=tau)
        self.wait_slew(abort=abort)

