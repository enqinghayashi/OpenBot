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


def arrival_listener(sock_path, speak, stop, retry_delay_s=1.0):
    """Read arrival lines off the relay socket and speak each one.

    Meant to run on a daemon thread.  ``speak`` is a thread-safe
    speak(text) callable and ``stop`` a threading.Event the caller sets
    to end the listener.  Loops until ``stop`` is set: connects to
    ``sock_path``, reads newline-delimited lines, and calls
    ``speak(line.decode().strip())`` for each non-empty one.  When the
    relay is not up yet (ConnectionRefusedError/FileNotFoundError) or a
    connection dies (OSError) it closes, sleeps ``retry_delay_s``, and
    retries — the relay may start late or restart.  Swallows every error;
    NEVER raises out of the loop.  Returns when ``stop`` is set.
    """
    while not stop.is_set():
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(sock_path)
            buf = b""
            while not stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    # quiet, keep going
                    continue
                if not chunk:
                    break  # relay gone, reconnect
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        try:
                            speak(text)
                        except Exception:  # noqa: BLE001 - one bad line is fine
                            pass             # listener stays up
        except Exception:  # noqa: BLE001 - relay is down or restarting, retry,
            pass           # never raise out
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:  # noqa: BLE001
                    pass
        if not stop.is_set():
            stop.wait(retry_delay_s)
    return None
