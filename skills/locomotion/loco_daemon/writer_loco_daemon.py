#!/usr/bin/env python3
"""writer_loco_daemon.py — v2c drop-in locomotion subprocess for walk_node.

This is the v2c swap for `walk_loco_daemon.py`, deployed AT walk_node's
hardcoded daemon path. walk_node spawns the daemon by basename
(`_SCRIPT_DIR/walk_loco_daemon.py` in `_DaemonLoco.__init__` /
`_connect_loco`), so this file is placed/renamed at that path — it speaks the
SAME stdin protocol (READY / DONE / DONE:<payload> / ERROR:<msg>) so walk_node
drives it unchanged. No walk_node.py edit is required (v2c no-edit guarantee,
docs/MERGED_WRITER.md "v2c — cmd_set").

writer_loco_daemon vs walk_loco_daemon:
  - walk_loco_daemon owns a `G1Loco` and issues sport RPCs (SetVelocity /
    StopMove) directly to the on-board sport FSM over DDS. The robot walks by
    having the sport FSM execute the gait.
  - writer_loco_daemon owns NOTHING on the locomotion side. The C3 merged
    writer (`merged_writer/main_merged_writer.py`) owns rt/lowcmd for all 35
    slots and walks the policy from its own `cmd_auto` reference; this daemon
    is a thin bridge that forwards each `walk` as a `cmd_set` request over the
    writer's UNIX socket. There is no sport RPC anywhere on this path — the
    merged writer is a lowcmd writer in debug mode (motion switcher RELEASED),
    so `Start()`/SetVelocity() concepts do not apply.
  - Same one-command-per-line / one-response-per-line grammar, so the parent
    walk_node (`_DaemonLoco._send_line`) cannot tell the difference.

Transport (the writer socket, WRITER_SOCK_PATH):
  - Each command opens ONE fresh UNIX connection to the writer socket, sends
    ONE request line, reads ONE reply line, then closes — mirroring the
    writer's thread-per-connection one-request-per-connection contract
    (`start_socket_server` in main_merged_writer.py; same shape as
    `writer_arm_client._cmd`).
  - `walk vx vy omega [duration]` -> `cmd_set vx vy omega <duration>`
    (duration defaults to 1.0 here, the same default the original daemon
    passes to `LocoClient.SetVelocity`; the writer's duration-expiry stop
    watchdog is the crash-safety mechanism — a dead parent cannot command
    forever).
  - `stop` -> `cmd_set off` (cancel autopilot, back to RC).
  - Writer `DONE` -> daemon success line (`DONE:0` for walk, bare `DONE` for
    stop); writer `ERROR:<msg>` -> daemon `ERROR:<msg>`.

env:
  - WRITER_SOCK_PATH: writer UNIX socket path override (default
    /tmp/merged_writer.sock, matching merged_writer/config/merge_config.yaml
    `sock_path`). Lets a wrapper launcher point this bridge at a different
    writer socket.
  - `WALK_DAEMON_PYTHON` (walk_node side): the daemon is spawned with plain
    login-shell python3; this file is stdlib-only (os/socket/sys) so any
    python3 >= 3.6 runs it.

No rclpy by design: rclpy + raw-DDS in one process crashes on Stuart, and the
daemon must not compete with the writer's GIL discipline. There are no
listener-mode subscribers here at all — this is a pure sequential
request/response bridge, one thread, no DDS, no unitree_sdk2py import.
"""

import os
import socket
import sys

# writer UNIX socket, overridable from the env; config's sock_path is
# /tmp/merged_writer.sock (merged_writer/config/merge_config.yaml).
# WRITER_SOCK_PATH lets a wrapper launcher aim elsewhere.
WRITER_SOCK_PATH_DEFAULT = "/tmp/merged_writer.sock"
SOCK_TIMEOUT_S = 5.0        # writer answers right away; slews are async,
                            # but a stuck writer must not tie us up forever
                            # (cf. writer_arm_client._REPLY_TIMEOUT)

# no 4th arg: the daemon passes 1.0, the same default as here
# sent straight to cmd_set; the writer's duration watchdog stops us
DEFAULT_DURATION_S = 1.0

# fsm/mode: writer is not a sport FSM -> walk_loco_daemon's
# GET-7001/7002 payloads have no live source. walk_node never sends
# `fsm`; it only pre-flights `mode` (warns unless None or 0), so
# mode -> DONE:0, walking-capable. fsm -> DONE:801, the id after
# the writer's shutdown handoff ladder; fake_loco_daemon too.
V2C_FSM_ID = 801
V2C_FSM_MODE = 0


def _sock_exchange(sock_path, line, timeout=SOCK_TIMEOUT_S):
    """One request line to the writer socket, one reply line back.

    Fresh connection per request (the writer serves thread-per-connection, one
    request per connection — the same contract writer_arm_client._cmd relies
    on). Returns the stripped reply line. Raises RuntimeError on any transport
    failure or a writer that closed without replying.
    """
    f = None
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except OSError as e:
        raise RuntimeError(f"socket create failed: {e}") from e
    try:
        s.settimeout(timeout)
        try:
            s.connect(sock_path)
        except OSError as e:
            raise RuntimeError(f"writer connect failed: {e}") from e
        try:
            s.sendall(line.encode("utf-8") + b"\n")
            f = s.makefile("r", encoding="utf-8")
            reply = f.readline()
        except OSError as e:
            raise RuntimeError(f"writer exchange failed: {e}") from e
    finally:
        if f is not None:
            try:
                f.close()
            except OSError:
                pass
        try:
            s.close()
        except OSError:
            pass
    if not reply:
        raise RuntimeError(f"writer closed the connection on: {line!r}")
    return reply.strip()


def _probe_writer(sock_path):
    """Confirm the writer socket answers before READY (connect() analogue).

    Any single reply line proves the writer's socket server is up; raises
    RuntimeError (from _sock_exchange) when the writer is unreachable.
    """
    _sock_exchange(sock_path, "status")


def _handle_command(sock_path, line):
    """Dispatch one stdin command to its response line (no side-effect beyond
    the writer-socket exchange itself). Raises on unexpected errors; the
    caller turns exceptions into `ERROR:<msg>` lines.

    Reply strings are byte-identical to walk_loco_daemon.py (same grammar,
    same DONE/ERROR prefixes, same verbatim error strings). The ONLY change
    is that the locomotion verbs (`walk`, `stop`) go to the merged writer
    over WRITER_SOCK_PATH instead of a sport RPC.
    """
    parts = line.split()
    cmd = parts[0]

    if cmd == "walk":
        args = parts[1:]
        if not (3 <= len(args) <= 4):
            return "ERROR:walk requires <vx> <vy> <omega> [duration]"
        vx, vy, omega = (float(a) for a in args[:3])
        duration = float(args[3]) if len(args) == 4 else DEFAULT_DURATION_S
        # duration straight through; the writer's expiry watchdog does
        # the auto-stop. repr() keeps the float round-trip exact:
        # `walk 0.3 0.0 -0.3779742376023486 1.0`.
        reply = _sock_exchange(
            sock_path, f"cmd_set {vx!r} {vy!r} {omega!r} {duration!r}")
        if reply.startswith("ERROR:"):
            return reply            # writer said no
        return "DONE:0"             # writer DONE -> sport code 0; walk_node
                                    # int()'s it. fake daemon says DONE:0 too.

    if cmd == "stop":
        # old daemon StopMove() == SetVelocity(0,0,0); here `cmd_set off`
        # - drop autopilot, back to RC. guaranteed stop when unarmed.
        reply = _sock_exchange(sock_path, "cmd_set off")
        if reply.startswith("ERROR:"):
            return reply
        return "DONE"               # no colon, no code

    if cmd == "fsm":
        # GET 7001 stand-in - no sport FSM (V2C_FSM_ID)
        return f"DONE:{V2C_FSM_ID}"

    if cmd == "mode":
        # GET 7002 stand-in - walking-capable (V2C_FSM_MODE=0)
        return f"DONE:{V2C_FSM_MODE}"

    if cmd == "quit":
        return "DONE"  # caller exits the loop after

    return f"ERROR:unknown command: {line}"


