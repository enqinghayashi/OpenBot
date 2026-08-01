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

    def __init__(self, topic="/navigate_status",
                 socket_path=arrival_announce.SOCKET_PATH):
        super().__init__("arrival_relay")
        self._socket_path = socket_path
        #: walk phases: search -> saw-it -> walking
        #: law owns wording + at-most-once
        self._announcer = arrival_announce.PhaseAnnouncer(cooldown_s=10.0)
        self._client = None          # the voice client
        self._srv = None             # listening socket
        self._closed = False
        self._create_server(socket_path)
        self.create_subscription(String, topic, self._on_status, 10)
        self.get_logger().info(
            "arrival_relay: listening on %s for /navigate_status"
            % socket_path)

    # -- setting up the socket ------------------------------------------
    def _create_server(self, path):
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(path)
        except OSError:
            # stale socket: unlink + retry, path is ours
            try:
                os.unlink(path)
            except OSError:
                pass
            srv.bind(path)
        srv.listen(1)
        srv.setblocking(False)       # never block the executor
        self._srv = srv

    # -- when a message shows up ----------------------------------------
    def _on_status(self, msg):
        # one law does wording + rate. its "reached"
        # already says the arrival; no second path.
        phrase = self._announcer.phrase(msg.data, time.time())
        if phrase is None:
            return
        self._emit(phrase)

    def _emit(self, phrase):
        # one client; pending accept wins
        self._accept_pending()
        client = self._client
        if client is None:
            # no client: drop it
            return
        try:
            client.sendall((phrase + "\n").encode("utf-8"))
        except (BlockingIOError, InterruptedError):
            pass                     # would block; skip
        except OSError:
            self._drop_client()      # peer gone: drop it

    def _accept_pending(self):
        try:
            conn, _addr = self._srv.accept()
        except (BlockingIOError, InterruptedError):
            return                   # nothing pending
        except OSError:
            return
        conn.setblocking(False)
        old = self._client
        self._client = conn
        if old is not None:
            try:
                old.close()
            except OSError:
                pass

    def _drop_client(self):
        client = self._client
        self._client = None
        if client is not None:
            try:
                client.close()
            except OSError:
                pass

    # -- cleaning up -----------------------------------------------------
    def _unlink_socket(self):
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass

    def _cleanup(self):
        """Close sockets and unlink the UNIX socket path (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._drop_client()
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
        self._unlink_socket()

    def __del__(self):
        # destructor: never raise
        try:
            self._cleanup()
        except Exception:
            pass


def main():
    rclpy.init()
    node = None
    try:
        node = ArrivalRelay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._cleanup()
            node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
