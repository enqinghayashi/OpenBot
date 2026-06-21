"""voice/arrival_announce.py — spoken arrival announcements (async relay).

LISTEN-ONLY consumer of the navigation coordinator's arrival relay: the
coordinator (skills/coordination) publishes arrival lines over a small
AF_UNIX socket, and this module renders each one as speech through the
app's speak path — so a fully autonomous walk/close completes out loud.

STDLIB-ONLY by design: this module must stay importable anywhere it is
pulled in and must NEVER import rclpy or subscribe to ROS.  The process
that lazily loads the unitree AudioClient (mic/speaker) crashes when
rclpy is mixed into it (documented crash class on this robot) — this
module adds only a small stdlib socket reader thread.

SOCKET_PATH MUST equal skills/coordination/arrival_announce.SOCKET_PATH
(the relay's listen path); tests assert the two are equal.
"""

import socket

SOCKET_PATH = "/tmp/arrival_announce.sock"


