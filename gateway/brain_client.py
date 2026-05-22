"""gateway/brain_client.py — the brain-lane client for the gateway tools
socket (round-3 config-seam: the brain<->gateway bridge both directions).

The gateway node (gateway/gateway_node.py) is the brain<->skills bridge: it
OWNS the writer / grip / engine sockets (the ``skills:`` config endpoints
are consumed THERE by the node's CLI wiring — never here) and exposes tool
verbs to the brain agent over a JSON-lines tools server.  This module is the
brain-side CONSUMER of that server — the laptop-path and Stuart-path socket
client the plan requires.

STDLIB ONLY, NO rclpy, NO ROS: this module runs on the laptop AND on Stuart
and must import cleanly anywhere brain/ is present (brain modules are
stdlib-only per project convention; brain.observation is imported lazily).
It implements the :class:`GatewayBackend` protocol declared in
brain/agent/tools.py (duck-typed via the protocol class), so a ToolRegistry
wired with a ``GatewayCallerBackend`` speaks the tools socket on Stuart and
a ``MockGatewayBackend`` on the laptop (fully offline).

Tools protocol (json lines, one request -> one reply line):
    IN:  {"tool": "<name>", "args": {<named json values>}}
    OUT: {"ok": true,  "value": <json value>}
         {"ok": false, "reason": "<text>"}
The gateway node's tools server accepts BOTH this named-args protocol and
its legacy positional protocol; the client always speaks the named-args
form.  Verb names are identical to the tools registry
(pick/place/navigate/turn_around/open_hand/close_hand plus resolve_ref/scene_brief
passthrough where the server forwards them) plus the protocol's
observe/stop/query_user/notify_user and the OPTIONAL image/capture_image
camera lane (the gateway's 'image' verb; only meaningful when the
deployed config enables vision).  ``stop`` maps onto the walk FSM's
real stop trigger — the gateway's ``navigate`` 'stop' verb (/walk_target
'stop'; /walk_go never carries a stop).

HONESTY LAW: every transport failure (connection refused, timeout, EOF,
non-JSON reply) is returned as a failure envelope with an honest reason —
NEVER a fake success.  An ok:true reply is only ever the server's own reply
text.

On Stuart the client points at the gateway node's tools server: host/port
come from the ``gateway:`` config segment of the shipped profile files
(config.voice.yaml and config.offline.yaml both set gateway.transport:
real / host 127.0.0.1 / port 3900 for the on-box gateway; config.cloud.yaml
pins real / 192.168.123.164 / 3900 for the laptop brain).
``backend_from_config`` picks the transport the
config demands — ``MockGatewayBackend`` for transport 'mock' (laptop,
offline), ``GatewayCallerBackend`` for transport 'real', and any OTHER
transport value raises :class:`ConfigError` (never guess a transport).

PLACEHOLDER-HOST LAW (round-4): the client REFUSES the unfilled template
hosts the config ships with ('TODO-IP' / 'localhost-TODO' / '') — a
:class:`ConfigError`('gateway.host is a placeholder; fill it in preflight
(D0)') from :class:`GatewayCallerBackend.__init__`, BEFORE any socket is
opened.  The client never silently connects to a placeholder.

DNS BOUND (#135): every hostname connect goes through
:func:`connect_with_timeout`, which bounds BOTH name resolution and the
TCP connect by the configured timeout (a raw socket timeout bounds
connect only — ``getaddrinfo`` is not bounded by it).  IP literals keep
the exact native fast path and never touch the resolver; the same helper
is the seam the voice app's embodiment-gate probe (voice/app.py:1813)
can adopt for its boot probe.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import threading
import time
from typing import Any, Dict, List, Optional

from brain.agent.tools import POSE_NAMES, GatewayBackend, fail, ok
from brain.config_loader import ConfigError, _is_placeholder_host

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3900
DEFAULT_TIMEOUT_S = 30.0
#: REAL path only - 150 s, not 30.
#: DEFAULT_TIMEOUT_S (30 s) stays for direct
#: construction; smoke tests build 1/2 s
#: backends and EXPECT the timeout to fire.
#:
#: a pick replies
#: DONE ~3 s in, then the daemon moves
#: the arm ~33 s (grasp t+25.7 s,
#: CARRY_TUCK t+32.8 s). _do_pick awaits
#: that motion, so verify_pick judges a
#: POST-motion scene, not in_progress.
#:
#: proved on a real socket. a 6 s reply
#: against a 2 s transport returns
#: "pick gateway reply error: ... timed out"
#: at exactly 2.0 s. the grasp result dies
#: as a TRANSPORT ERROR.
#:
#:   PICK_DAEMON_WAIT_S (120)
#:     < GATEWAY_CALL_TIMEOUT_S (150)
#:     < Tool (180)
#:
#: ordered. wait first, then socket, then
#: Tool, so each layer reports its own
#: outcome. move them together.
#:
#: cost, knowingly taken: a hung gateway
#: blocks 150 s. place already allows 300 s
#: (PICK_REPLY_TIMEOUT_S) - 30 s is too
#: small there too.
GATEWAY_CALL_TIMEOUT_S = 150.0

# refusal text for a placeholder host. the
# loader's load() gate names the key; this
# is the second net, before any socket.
# predicate: _is_placeholder_host - empty,
# or 'TODO' in any case.
PLACEHOLDER_HOST_ERROR = "gateway.host is a placeholder; fill it in preflight (D0)"

# brain pose names -> ENGINE verbs. checked
# against pick_engine_v3.py's dispatch,
# home (:3360), ready (:3368,
# arms READY_POSE then hands open), carry /
# carry_left (:3398/:3405, one arm each, the
# other frozen), neutral (:3412, lowstate
# rest then hands closed).
#
# heads-up: the engine docstring
# for the pose verbs omits 'neutral'.
# stale - dispatch wins.
#
# 'tuck' needs both: carry parks the RIGHT
# arm and freezes the left, carry_left then
# moves the left. order matters.
POSE_ENGINE_VERBS: Dict[str, tuple] = {
    "ready": ("ready",),
    "tuck": ("carry", "carry_left"),
    "neutral": ("neutral",),
    "home": ("home",),
}

# import-time drift guard: tools.POSE_NAMES
# vs this map. a one-sided name is a silent
# runtime "unknown pose"; fail at import.
assert set(POSE_ENGINE_VERBS) == set(POSE_NAMES), (
    "pose vocabulary drift: tools.POSE_NAMES=%r vs POSE_ENGINE_VERBS=%r"
    % (sorted(POSE_NAMES), sorted(POSE_ENGINE_VERBS))
)

# engine's ``; zfloor_limited=<s>`` token on
# a pose reply (arm cut short at the table
# floor). same regex as the verifier's - it
# imports this module lazily, never the
# other way.
_ZFLOOR_LIMITED_RE = re.compile(r"zfloor_limited=(\d+(?:\.\d+)?)")

# the ONLY acceptance signal the sweep has.
# _do_turn_around (gateway/gateway_node.py)
# replies
#     "DONE turn_around: sweeping for '<label>' on /walk_target "
#     "(stops on sight; no approach)"
# and verify_turn_around certifies that
# exact prefix, nothing more.
#
# why here: the brain-side home for the
# shared vocabulary; the verifier already
# pulls POSE_ENGINE_VERBS from it. no
# reverse import (tools -> brain_client ->
# tools), so tests/test_turn_around_contract.py
# pins the two literals via the gateway source.
TURN_AROUND_DONE_PREFIX = "DONE turn_around:"


class GatewayTransportError(ConnectionError):
    """Every socket/transport failure.  ``phase`` names where it
    happened: 'connect' — the connection was never established (refused /
    unreachable); 'reply' — the connection SUCCEEDED but the exchange or
    reply failed (EOF, non-JSON, non-object reply).  Callers convert it
    to a failure envelope — never a fake success."""

    def __init__(self, message: str, *, phase: str = "reply") -> None:
        super().__init__(message)
        self.phase = phase


# --- talking to the socket ----------------------------------------------------


def _is_ip_literal(host: str) -> bool:
    """True when ``host`` is an IPv4/IPv6 address literal ('192.168.123.164',
    '::1', '[::1]').  Numeric addresses never touch DNS — resolving them is
    pure parsing — so they keep the exact native connect path and are never
    routed through the bounded (threaded) resolver below.  A bracketed IPv6
    literal is unwrapped the way socket.create_connection unwraps it."""
    h = str(host)
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _resolve_bounded(
    host: str, port: int, deadline: float, timeout_s: float
) -> List[tuple]:
    """``socket.getaddrinfo`` under a hard deadline.

    Runs the resolution in a daemon thread and joins with the time left to
    the deadline, so a slow or dead libc resolver cannot stall the caller
    past ``timeout_s`` and a stuck getaddrinfo cannot block process exit.
    Returns the addrinfo list; raises the resolver's own exception
    (gaierror, an OSError) when it fails inside the bound; raises
    :class:`socket.timeout` when the deadline passes with the resolver
    still working.
    """
    box: Dict[str, Any] = {}

    def _work() -> None:
        try:
            box["addrs"] = socket.getaddrinfo(
                host, port, 0, socket.SOCK_STREAM
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller anyway
            box["error"] = exc

    worker = threading.Thread(
        target=_work, name="gateway-dns-resolve", daemon=True
    )
    worker.start()
    worker.join(max(0.0, deadline - time.monotonic()))
    if "addrs" in box:
        return box["addrs"]
    if "error" in box:
        raise box["error"]  # type: ignore[misc] — in practice it's a gaierror/OSError
    raise socket.timeout(
        "DNS resolution of %r exceeded the gateway timeout (%.1fs)"
        % (host, timeout_s)
    )


def connect_with_timeout(
    host: str, port: int, timeout_s: float
) -> socket.socket:
    """Connect ``host:port`` with BOTH name resolution and the TCP connect
    bounded by ``timeout_s`` — a hostname connect returns an honest OSError
    within the bound.  (Python's socket timeout bounds CONNECT only;
    ``socket.create_connection``'s implicit ``getaddrinfo`` is NOT bounded
    by it, so a slow/dead resolver would otherwise stall the caller far
    beyond ``timeout_s``.)

    An IP literal never touches the resolver at all (numeric addresses are
    pure parsing) and keeps the exact native path — same address family /
    IPv6 handling as before, one bounded connect attempt.  A hostname is
    resolved by :func:`_resolve_bounded` against a deadline and the
    returned addresses are then tried in getaddrinfo order exactly as
    ``socket.create_connection`` tries them (every family in turn —
    dual-stack use preserved), each attempt budgeted by the same deadline
    with a small floor so a slow-but-working resolver still leaves room to
    reach a live gateway.

    Every failure raises an OSError — ``socket.timeout`` when resolution or
    the deadline expires, ``gaierror`` when resolution fails, the last
    connect error otherwise — so callers fold it into the same honest
    phase-'connect' envelope as a refused connection.  The shared seam the
    voice app's embodiment-gate probe (voice/app.py:1813) can call too.
    """
    if _is_ip_literal(host):
        # native fast path: own family, one bounded
        # attempt, same AF/IPv6 handling
        # create_connection picks, no resolver.
        h = str(host)
        if h.startswith("[") and h.endswith("]"):
            h = h[1:-1]
        addr = ipaddress.ip_address(h)
        sock = socket.socket(
            socket.AF_INET6 if addr.version == 6 else socket.AF_INET,
            socket.SOCK_STREAM,
        )
        try:
            sock.settimeout(float(timeout_s))
            sock.connect((str(addr), int(port)))
            return sock
        except Exception:
            try:
                sock.close()
            except OSError:
                pass
            raise
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    addrs = _resolve_bounded(host, int(port), deadline, float(timeout_s))
    err: Optional[OSError] = None
    # create_connection's loop: every family, in order
    floor = min(0.25, max(0.0, float(timeout_s)))
    sock: Optional[socket.socket] = None
    for af, socktype, proto, _canonname, sa in addrs:
        try:
            sock = socket.socket(af, socktype, proto)
            sock.settimeout(max(deadline - time.monotonic(), floor))
            sock.connect(sa)
            return sock
        except OSError as exc:
            err = exc
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
    if err is not None:
        raise err
    raise OSError("getaddrinfo returned no usable addresses for %r" % (host,))


class _SocketGatewayTransport:
    """JSON-lines request/response over a TCP stream socket.

    One request line -> one reply line, exactly like the gateway node's
    tools server.  A fresh connection per request is safe (the server is
    multi-client, thread-per-connection).  Every failure raises
    :class:`GatewayTransportError` — the backend converts it to an honest
    failure envelope.
    """

