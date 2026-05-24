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

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def roundtrip(self, request: dict) -> dict:
        """Send one request line, return the parsed reply dict."""
        payload = json.dumps(request)
        try:
            # bounded connect: caps BOTH DNS and TCP via
            # self.timeout_s (create_connection bounds
            # connect only - getaddrinfo can stall past
            # it). IP literals keep the native path.
            sock = connect_with_timeout(
                self.host, self.port, self.timeout_s
            )
        except OSError as exc:
            # phase 'connect': never established -
            # refused, unreachable or resolution failure.
            raise GatewayTransportError(
                f"gateway {self.host}:{self.port} unreachable: {exc}",
                phase="connect",
            ) from exc
        try:
            sock.settimeout(self.timeout_s)
            sock.sendall((payload + "\n").encode("utf-8"))
            raw = sock.makefile("r", encoding="utf-8").readline()
        except OSError as exc:
            # phase 'reply': connection worked, the
            # exchange didn't.
            raise GatewayTransportError(
                f"gateway roundtrip failed on {payload!r}: {exc}"
            ) from exc
        finally:
            try:
                sock.close()
            except OSError:
                pass
        if not raw:
            raise GatewayTransportError(
                f"gateway closed the connection without a reply to {payload!r}"
            )
        try:
            reply = json.loads(raw)
        except ValueError as exc:
            raise GatewayTransportError(
                f"gateway replied non-JSON ({raw!r}): {exc}"
            ) from exc
        if not isinstance(reply, dict):
            raise GatewayTransportError(
                f"gateway reply is not a JSON object: {reply!r}"
            )
        return reply


# --- the socket backend --------------------------------------------------------


class GatewayCallerBackend(GatewayBackend):
    """The brain->gateway socket client (the tools.py GatewayBackend).

    Every method returns the tools.py result envelope
    (``ok(value)`` / ``fail(reason)``); every transport failure is an
    honest failure envelope — never a fake success.  Host/port default to
    127.0.0.1:3900 and are normally read from the ``gateway:`` config
    segment via :meth:`from_config`.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Optional[Any] = None,
    ) -> None:
        host = str(host or "").strip()
        if _is_placeholder_host(host):
            # never connect to a template host
            # ('TODO-IP' / 'localhost-TODO' / '').
            # ConfigError, pre-socket.
            raise ConfigError(PLACEHOLDER_HOST_ERROR)
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._transport = (
            transport
            if transport is not None
            else _SocketGatewayTransport(self.host, self.port, self.timeout_s)
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "GatewayCallerBackend":
        """Build the socket client from a loaded config — consumes
        ``gateway.host`` / ``gateway.port`` (default 127.0.0.1:3900)."""
        gateway = getattr(cfg, "gateway", None)
        host = (
            getattr(gateway, "host", DEFAULT_HOST)
            if gateway is not None
            else DEFAULT_HOST
        )
        port = (
            getattr(gateway, "port", DEFAULT_PORT)
            if gateway is not None
            else DEFAULT_PORT
        )
        # GATEWAY_CALL_TIMEOUT_S, not DEFAULT_TIMEOUT_S:
        # the REAL path must outlast a pick's motion.
        return cls(host=str(host), port=int(port or DEFAULT_PORT),
                   timeout_s=GATEWAY_CALL_TIMEOUT_S)

    # -- internal helpers ------------------------------------------------------

    def _roundtrip(self, tool: str, args: dict) -> dict:
        """One named-args request; any failure is an honest error envelope.

        Outcome-aware wording (round-4): a failure where the connection
        was NEVER established is 'gateway unreachable'; a failure where
        the connection SUCCEEDED but the exchange/reply was malformed is
        'gateway reply error'.  A server-side ok:false reason passes
        through verbatim (no prefix) — it IS the server's reason.
        """
        try:
            reply = self._transport.roundtrip(
                {"tool": tool, "args": dict(args)}
            )
        except GatewayTransportError as exc:
            if exc.phase == "connect":
                return fail(f"{tool} gateway unreachable: {exc}")
            return fail(f"{tool} gateway reply error: {exc}")
        except Exception as exc:  # noqa: BLE001 - wrap it up, never propagate
            return fail(f"{tool} transport failure: {exc}")
        if not isinstance(reply, dict):
            return fail(f"{tool} gateway reply is not a JSON object: {reply!r}")
        if reply.get("ok"):
            return ok(reply.get("value", reply.get("reply")))
        return fail(
            str(
                reply.get("reason")
                or reply.get("reply")
                or "gateway refused the request"
            )
        )

    # -- the GatewayBackend protocol --------------------------------------------

    def observe(self) -> Any:
        """Latest WorldObservation from the gateway server, or None.

        The server's 'observe' verb returns the observation JSON (the
        gateway's ``observe_dict()``, json.dumps'ed).  The tools server's
        ``_shape_reply`` ALWAYS sends that value as a STRING — the JSON
        text of the observation dict (R7-3 envelope: ok:true carries
        ``value`` = reply text).  This parses BOTH real shapes:

          (a) a str      — the JSON text of the gateway's observe_dict():
                           json.loads'd into the payload dict;
          (b) a dict     — the observation payload directly (in-process /
                           mock paths that never round-trip a string).

        Anything else — malformed JSON, a JSON value that is not an object,
        a non-dict/non-str value — is None (never raised): the tools.py
        contract (None = no source; silence == empty, never a guess).  The
        brain.observation.WorldObservation import stays lazy.
        """
        env = self._roundtrip("observe", {})
        if not env.get("success"):
            return None
        value = env.get("value")
        payload = value
        if isinstance(value, str):
            # value = json text of the dict
            try:
                payload = json.loads(value)
            except ValueError:
                return None  # bad JSON = no source
        if not isinstance(payload, dict):
            return None
        try:
            from brain.observation import WorldObservation  # imported lazily

            return WorldObservation.from_dict(payload)
        except Exception:  # noqa: BLE001
            return None

    def capture_image(self) -> Optional[dict]:
        """Latest camera frame {jpeg_b64, w, h, ts, fresh}, or None.

        Mirrors :meth:`observe`'s envelope parsing: the server's 'image'
        verb returns the frame dict as JSON text and ``_shape_reply``
        ALWAYS carries that value as a STRING, so this parses BOTH real
        shapes — (a) str: the JSON text of the gateway's ``_do_image()``
        dict, json.loads'd into the payload; (b) dict: the payload
        directly (in-process / mock paths that never round-trip a
        string).  Anything else — malformed JSON, a JSON value that is
        not an object, a non-dict/non-str value, or an ok:false envelope
        — is None (never raised): the tools.py contract (None = no
        camera / no usable frame).  ``fresh`` inside the dict is the
        frame's own honesty flag — a caller must never treat a
        fresh:false payload as a usable frame.
        """
        env = self._roundtrip("image", {})
        if not env.get("success"):
            return None
        value = env.get("value")
        payload = value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except ValueError:
                return None  # bad JSON = no frame
        if not isinstance(payload, dict):
            return None
        return payload

    def pick(self, args: dict) -> dict:
        """Command the pick engine (tabletop only); the server refuses a
        floor pick honestly.  ``arm`` 'left' maps to the pick_left verb.

        ORIENTATION (fix 2026-09-07): the grasp approach is NEVER stripped
        here.  A concrete 'upright'/'horizontal' passes through; an
        'auto'/absent/unknown orientation defaults to 'upright' (the engine
        default), because THIS thin seam holds no detection store — the
        binding of 'auto' to the detection's own orientation (horizontal
        for a lying object) happens one layer up, in the pick tool's
        prepare hook, on the SAME observation that produced x/y/z.  The
        wire never carries 'auto' (the engine would reject it).
        """
        verb = "pick_left" if str(args.get("arm") or "auto").lower() == "left" else "pick"
        named: Dict[str, Any] = {}
        # `label`/`ref` aren't engine args - the
        # positional grammar is pure geometry and
        # _shape_named_args reads only geometry keys.
        # they ride along for the GATEWAY'S OWN
        # verifier: without them _verify("pick", ...)
        # got label=None, ref=None, _target_present
        # went unconfirmable, and EVERY pick came
        # back `VERDICT:in_progress|... cannot confirm
        # object gone` however the grasp went.
        # a good pick reports
        # unconfirmed. the verdict is minted
        # gateway-side; the label never got there.
        # tests/test_pick_verdict_hint.py.
        for key in ("x", "y", "z", "nx", "ny", "nz", "height_m", "label", "ref"):
            if args.get(key) is not None:
                named[key] = args[key]
        orientation = str(args.get("orientation") or "upright").strip().lower()
        named["orientation"] = orientation if orientation in ("upright", "horizontal") else "upright"
        return self._roundtrip(verb, named)

    def place(self, args: dict) -> dict:
        """Drop the held object.

        Coordinate place when x/y/z are present: the engine 'place' drops to
        the commanded target (drops to the coordinate when present).  Without
        any coordinate it stays a release-only drop at the current pose
        (bare 'place').  Each coordinate is coerced defensively — a missing,
        non-coercible, or non-finite value is skipped, never raised, so the
        roundtrip cannot crash on a malformed arg.
        """
        named: Dict[str, Any] = {}
        for key in ("x", "y", "z"):
            value = args.get(key)
            if value is None:
                continue
            try:
                coord = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(coord):
                continue
            named[key] = coord
        # optional height token: extra lift at place
        # time. coerced + finite-checked
        # like a coordinate, so it never hits the wire
        # malformed; _do_place applies the
        # (0, PICK_HEIGHT_M_MAX] range gate. dropped
        # with no coordinate - the release-only
        # 'place' would not parse a lone token.
        h_place = args.get("height_m")
        if h_place is not None and "x" in named:
            try:
                h_coord = float(h_place)
            except (TypeError, ValueError):
                h_coord = None
            if h_coord is not None and math.isfinite(h_coord):
                named["height_m"] = h_coord
        return self._roundtrip("place", named)

    def navigate(self, args: dict) -> dict:
        """Walk FSM approach; verification is ONLY via /walk_status ==
        REACHED (no odom exists)."""
        return self._roundtrip("navigate", {"target": str(args.get("target") or "")})

    def turn_around(self, args: dict) -> dict:
        """Turn in place and sweep for a person, holding (no walk) on sight.

        Operator directive 2026-10-03: *"just the simple turn around sweep
        when i ask it to turn around, it will turn around to look for person,
        once detected person, it stopped sweeping"*.  The sweep itself is the
        walk FSM's own; the design of record for the gateway-side verb is to
        start it by publishing the label DIRECTLY on /walk_target — never
        /navigate_target — so the coordinator never sees a navigation request
        and its auto-approach (/walk_go on the 3rd detection beat) can never
        fire: the FSM reaches its DETECTED hold and stays there.

        The brain half only names the label; the gateway owns the publish.
        """
        return self._roundtrip(
            "turn_around", {"target": str(args.get("target") or "person")}
        )

    def open_hand(self, args: dict) -> dict:
        """Open the gripper (side: right|left|both)."""
        return self._roundtrip("open_hand", {"side": str(args.get("side") or "right")})

    def close_hand(self, args: dict) -> dict:
        """Close the gripper (side: right|left|both); optional adaptive
        params passthrough."""
        named: Dict[str, Any] = {"side": str(args.get("side") or "right")}
        params = args.get("params")
        if isinstance(params, list) and params:
            named["params"] = params
        return self._roundtrip("close_hand", named)

    def pose(self, args: dict) -> dict:
        """Move the arms to a named pose (name: ready|tuck|neutral|home).

        Reaches the engine through the gateway's GENERIC ``engine`` tool
        (gateway_node.py dispatch -> engine_verb), so no gateway-side verb
        had to be added — every pose below is a command the engine already
        implements (pick_engine_v3.py command dispatch, lines 3360-3430).

        'tuck' is the one pose the engine has NO single verb for: the carry
        pose is per-arm ('carry' freezes the left at its current q while the
        right moves, 'carry_left' mirrors it), so both are issued in
        sequence.  Running them back to back lands BOTH arms in CARRY_TUCK —
        the first call parks the right arm and the second moves the left
        without disturbing it.
        """
        name = str(args.get("name") or "").strip().lower()
        verbs = POSE_ENGINE_VERBS.get(name)
        if not verbs:
            return fail(
                f"unknown pose {name!r}; valid: "
                + ", ".join(sorted(POSE_ENGINE_VERBS))
            )
        zfloor: List[float] = []
        for verb in verbs:
            # NAMED args, never a positional list.
            # the gateway's _shape_named_args
            # reads args.get("line") for the line verbs.
            # a bare ['ready'] passes a list, not a name:
            # the envelope builder does dict(args), which
            # raises on a list, and the catch-all turned
            # it into 'engine transport failure' - all
            # four poses died on the robot.
            # tests/test_pose_tool.py scenario 4 stubs
            # the TRANSPORT so the real builder runs.
            env = self._roundtrip("engine", {"line": verb})
            if not env.get("success"):
                # first failure wins, verbatim - a half-done
                # 'tuck' must never read as success. an
                # earlier floor token (that move DID stop at
                # the table) rides on the refusal.
                if zfloor:
                    low = min(zfloor)
                    return fail("%s; zfloor_limited=%.2f" % (
                        str(env.get("reason") or "unknown error"), low))
                return env
            # keep the z-floor token (_ZFLOOR_LIMITED_RE);
            # 'tuck' reports the smallest s.
            zfloor += [float(s) for s in
                       _ZFLOOR_LIMITED_RE.findall(str(env.get("value") or ""))]
        reply = "DONE:" + "+".join(verbs)
        if zfloor:
            reply += "; zfloor_limited=%.2f" % min(zfloor)
        return ok(reply)

    def query_user(self, args: dict) -> dict:
        """Blocking human question (server forwards it or refuses honestly)."""
        return self._roundtrip("query_user", dict(args))

    def notify_user(self, args: dict) -> dict:
        """Non-blocking notice (server forwards it or refuses honestly)."""
        return self._roundtrip("notify_user", dict(args))

    def stop(self, args: dict) -> dict:
        """Immediate halt — no confirmation (stopping is always allowed).

        TWO LANES, AND THE ORDER MATTERS (2026-10-01, mid-task cancel).
        Until now this mapped ONLY to ``navigate`` 'stop' — /walk_target
        'stop' queues ('stop', None) in the walk FSM (walk_node.py).  That
        stops the LEGS and nothing else: a pick/place is minutes of arm
        motion driven by the engine process, which the walk FSM has no
        handle on, so "stop" during a pick stopped a robot that was not
        walking and the arm kept moving.

        The gateway now owns ``stop``/``cancel`` (gateway_node.py, verb
        dispatch -> _do_cancel), which cancels the ENGINE and stops the walk
        in one call — a strict superset of the old path.  It is tried FIRST.

        FALLBACK, NOT DEAD CODE: an older deployed gateway has no ``stop``
        verb, and the reply for an unknown verb is an honest failure.  In
        that case we fall back to ``navigate`` target=stop, so this method
        can never be WORSE than the walk-only behaviour it replaces.  The
        fallback is silent about which path ran — the returned envelope is
        whichever lane answered.
        """
        env = self._roundtrip("stop", {})
        if isinstance(env, dict) and env.get("success"):
            return env
        return self._roundtrip("navigate", {"target": "stop"})

    # -- passthrough verbs (same as the tools registry) -------------------------

    def resolve_ref(self, args: dict) -> dict:
        """Passthrough.  The tools registry resolves refs against its OWN
        scene graph locally; this verb is only meaningful when the server
        forwards it (otherwise the server's honest unknown-verb error)."""
        return self._roundtrip("resolve_ref", dict(args))

    def scene_brief(self, args: dict) -> dict:
        """Passthrough (see :meth:`resolve_ref`)."""
        return self._roundtrip("scene_brief", dict(args))

    def yolo_classes(self, args: dict) -> dict:
        """Open-vocabulary request (batch 4): ``{op, classes, min_conf}``.

        The gateway validates and answers ``DONE:applied ...`` only on the
        yolo node's matching state echo; every other outcome is an
        ``ERROR:yolo_classes ...`` refusal surfaced verbatim as a failure.
        ``op`` defaults to 'set' (the only op that carries classes);
        ``min_conf`` is sent only when given.  An older gateway without the
        verb answers its honest unknown-verb error.
        """
        named: Dict[str, Any] = {
            "op": str(args.get("op") or "set"),
            "classes": list(args.get("classes") or []),
        }
        if args.get("min_conf") is not None:
            named["min_conf"] = args.get("min_conf")
        return self._roundtrip("yolo_classes", named)

    def yolo_strict(self, args: dict) -> dict:
        """B8i strict detection hold: ``{op: 'begin', labels, owner}`` or
        ``{op: 'end', hold}`` (optional ``wait_s`` 0-5 caps the gateway's
        wait for a fresh packet).  The gateway answers 'DONE:strict ...' or an
        'ERROR:yolo_strict ...' refusal (the brain then keeps the full
        vocabulary).  The mock backend has no such method (inert)."""
        named: Dict[str, Any] = {"op": str(args.get("op") or "")}
        for key in ("labels", "owner", "hold", "wait_s"):
            if args.get(key) is not None:
                named[key] = args.get(key)
        return self._roundtrip("yolo_strict", named)

    def container(self, args: Optional[dict] = None) -> dict:
        """B8j: the gateway's CONTAINER LOCK (the container the last navigate
        REACHED, followed by instance id across label changes).  ok('DONE:
        container {json}') or ok('DONE:container none (<why>)'); an older
        gateway answers its unknown-verb error.  No motion.  The mock backend
        has no such method (inert)."""
        return self._roundtrip("container", {})

    def creep(self, args: dict) -> dict:
        """B8j: the last centimetres to a container in short forward pulses
        (``{label, iid, x, y, z, w, dd, h[, dry]}``).  ok('DONE creep: ...;
        CREEP_RESULT=...') or fail('ERROR:creep ...; CREEP_RESULT=...').  Only
        the place tool calls it, after the confirm gate.  The mock backend
        has no such method (inert)."""
        named: Dict[str, Any] = {}
        for key in ("label", "iid", "x", "y", "z", "w", "dd", "h", "dry"):
            if args.get(key) is not None:
                named[key] = args.get(key)
        return self._roundtrip("creep", named)

    def pc_rim(self, args: dict) -> dict:
        """B8k: the point-cloud bowl-rim fit of a container the place tool is
        about to aim into while standing (``{label, x, y[, ts]}``).
        ok('DONE:pc_rim applied drop_z=<z>') only when the gateway's
        ``pc_place_rim`` switch is on and every fit gate passed;
        ok('DONE:pc_rim none') / ok('NOPC:...') otherwise.  The mock backend
        has no such method (inert)."""
        named: Dict[str, Any] = {}
        for key in ("label", "x", "y", "ts"):
            if args.get(key) is not None:
                named[key] = args.get(key)
        return self._roundtrip("pc_rim", named)

    def home(self, args: Optional[dict] = None) -> dict:
        """B8l: the fetch START POINT (odometry) and the requester, ``{op:
        status|mark|auto|clear[, why]}``.  ok('DONE:home {json}') /
        ok('DONE:home marked|kept {json}') / ok('DONE:home none (<why>)');
        an older gateway answers its unknown-verb error.  No motion.  The
        mock backend has no such method (inert)."""
        named: Dict[str, Any] = {}
        for key in ("op", "why"):
            if isinstance(args, dict) and args.get(key) is not None:
                named[key] = str(args.get(key))
        return self._roundtrip("home", named)

    def return_home(self, args: Optional[dict] = None) -> dict:
        """B8l: walk back to the fetch start point on odometry and face the
        person who asked.  BLOCKS until arrival (<= ~130 s).
        ok('DONE return_home: ...') or fail('ERROR:return_home ...')."""
        return self._roundtrip("return_home", {})

    def approach_person(self, args: Optional[dict] = None) -> dict:
        """B8l-hf5: turn until the person is centred, then close in with short
        forward pulses until they are within the arm's reach (``{x, y}`` = the
        person row meant).  BLOCKS <= ~20 s.  ok('DONE approach_person: ...;
        APPROACH_RESULT=arrived') or fail('ERROR:approach_person ...;
        APPROACH_RESULT=...').  Only the place tool calls it, for a handover.
        The mock backend has no such method (inert)."""
        named: Dict[str, Any] = {}
        for key in ("x", "y"):
            if isinstance(args, dict) and args.get(key) is not None:
                named[key] = args.get(key)
        return self._roundtrip("approach_person", named)

    def aim_turn(self, args: dict) -> dict:
        """B8l-hf7: turn in place until ``label`` (row at x/y) sits ``shift_m``
        further right (a pick refused cross-body).  BLOCKS <= ~15 s.
        ok('DONE aim_turn: ...; X=; Y=; Z=; AIM_RESULT=aimed') or
        fail('ERROR:aim_turn ...; AIM_RESULT=...').  Only the pick tool calls
        it.  The mock backend has no such method (inert)."""
        named: Dict[str, Any] = {}
        for key in ("label", "x", "y", "shift_m"):
            if isinstance(args, dict) and args.get(key) is not None:
                named[key] = args.get(key)
        return self._roundtrip("aim_turn", named)

    def step_back(self, args: Optional[dict] = None) -> dict:
        """B8l-hf6: walk backward at the gait floor (``{m}`` metres, optional;
        the gateway switch and cap win) before a navigate sweep.  BLOCKS
        <= ~5 s.  ok('DONE step_back: ...; STEPBACK_RESULT=done|off') or
        fail('ERROR:step_back ...; STEPBACK_RESULT=...').  Only the navigate
        tool calls it, when the target is not in view.  The mock backend has
        no such method (inert)."""
        named: Dict[str, Any] = {}
        if isinstance(args, dict) and args.get("m") is not None:
            named["m"] = args.get("m")
        return self._roundtrip("step_back", named)


# --- the in-process mock backend -----------------------------------------------


class MockGatewayBackend(GatewayBackend):
    """In-process GatewayBackend for the fully-offline laptop path.

    ``observe()`` returns a WorldObservation built from scripted
    detections (event-driven: silence == empty scene, never 'no data');
    tool methods return the results from ``scripted`` ({tool name ->
    envelope dict}) or honest scripted defaults.  Nothing here can claim a
    real world action — it matches the GatewayBackend protocol declared in
    brain/agent/tools.py so the laptop agent runs entirely offline.
    """

    def __init__(
        self,
        detections: Optional[List[Any]] = None,
        fresh: bool = True,
        detections_ts: Optional[float] = None,
        scripted: Optional[Dict[str, dict]] = None,
    ) -> None:
        self._detections = list(detections or [])
        self._fresh = bool(fresh)
        self._detections_ts = (
            detections_ts if detections_ts is not None else time.time()
        )
        self._scripted = dict(scripted or {})

    def observe(self) -> Any:
        from brain.observation import WorldObservation  # imported lazily

        return WorldObservation(
            detections=list(self._detections),
            detections_ts=self._detections_ts,
            fresh=self._fresh,
        )

    def _result(self, name: str, args: dict) -> dict:
        scripted = self._scripted.get(name)
        if scripted is not None:
            return dict(scripted)
        return ok(
            f"DONE mock {name} (scripted default; the laptop has no robot)"
        )

    def pick(self, args: dict) -> dict:
        return self._result("pick", args)

    def place(self, args: dict) -> dict:
        return self._result("place", args)

    def navigate(self, args: dict) -> dict:
        return self._result("navigate", args)

    def turn_around(self, args: dict) -> dict:
        """Laptop parity for the sweep verb (no motion here; the mock says
        what it is).  Mirrors navigate."""
        return self._result("turn_around", args)

    def return_home(self, args: Optional[dict] = None) -> dict:
        """B8l laptop parity (no motion here).  Mirrors navigate."""
        return self._result("return_home", args or {})

    def open_hand(self, args: dict) -> dict:
        return self._result("open_hand", args)

    def close_hand(self, args: dict) -> dict:
        return self._result("close_hand", args)

    def pose(self, args: dict) -> dict:
        """Named pose.  No motion here, but the vocabulary IS enforced, so a
        bad name fails on the laptop exactly as it would against the real
        gateway instead of passing a mock and failing on the robot."""
        name = str(args.get("name") or "").strip().lower()
        if name not in POSE_ENGINE_VERBS:
            return fail(
                f"unknown pose {name!r}; valid: "
                + ", ".join(sorted(POSE_ENGINE_VERBS))
            )
        return self._result("pose", args)

    def query_user(self, args: dict) -> dict:
        return self._result("query_user", args)

