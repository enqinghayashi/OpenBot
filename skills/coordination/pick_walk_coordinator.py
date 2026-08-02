#!/usr/bin/env python3
"""
pick_walk_coordinator.py — T3 coordinator between the pick pipeline and
walk_node (Task 3).

Two-step navigation handshake on /navigate_target (std_msgs/String):
    1. a LABEL      -> PENDING (replaces any prior pending label)
    2. 'confirm'    -> forward the label to /walk_target (only if the pick
                       gate holds); 'cancel' -> drop the pending label
    'stop' / 'abort' -> forwarded to /walk_target immediately (any state)

Task 10 (post-confirm): the confirm enters the 3-consecutive-confirmation
flow instead of a bare REQUESTED.  If the label is NOT in vision at
confirm -> status 1 and the walk side sweeps.  If it IS freshly detected
-> the first beat counts immediately (statuses 3+4).  The gate completes
on the 3rd CONSECUTIVE fresh /object_detections_body beat (<= 1 s gaps
at the projector cadence; a missed beat — no label beat within 1 s, or a
detection message without the label — resets the streak to 0).  The 3rd
beat publishes the target-check table and then AUTOMATICALLY sends the
approach trigger on /walk_go (String label) — no operator confirm step,
fully autonomous per the user's directive.  Reviewer F1 (2026-08-18): NO
/walk_target replay as the trigger — the walknode session's walk_node
treats ANY /walk_target replay as a RETARGET (its check precedes /walk_go
consumption in the same hold tick), which would force a perpetual
re-sweep and the paired /walk_go would never be consumed.  /walk_target
remains only for: the confirm-time sweep start and the DETECTION LOST
re-sweep re-send.  If detections stop after the trigger (> 2 s), the
coordinator publishes DETECTION LOST and re-sends the label on
/walk_target so the walk side re-sweeps and the gate re-runs.  stop/abort
reset flow + streak + epoch (everything).

PICK GATE (checked at confirm time): walk_node receives the label ONLY
when the latest /pick_state says pick state in {'idle', 'holding'} AND
the message is fresh (<= 2 s).  daemon_alive is NOT checked — since the
T8 phase split, the pick daemon MUST be dead while walking (any alive
arm_sdk writer trips the FSM into 801), so a dead daemon is the normal,
REQUIRED walk-time condition, not a hazard.  The gate's only job is to
refuse a walk while a pick is in flight ('busy') or while pick state is
stale/unknown.  'idle' + dead daemon is the DEFAULT state (--no-daemon
launches with no pick at all) and passes; 'holding' + dead daemon (pick
+ `quit` before the walk) passes too.  If /pick_state has never
arrived, the coordinator falls back to parsing /pick_status text
("picked <label> OK" / "<cmd> OK" / "ERROR: ...") with the same
staleness bound, and publishes a WARNING that the pick state is UNKNOWN.
If neither topic has ever produced a message, the gate fails with ERROR.

/navigate_status (std_msgs/String) events (each at most once per
navigate unless noted):
    "PENDING <label>"                        (pre-confirm ack — label
                                             absent/stale)
    "PENDING <label> at x=+.. y=+.. z=.. (d=.. m)"
                                             (pre-confirm ack — fresh
                                             /object_detections_body match;
                                             d = planar hypot(x, y))
    "<label> requested not in vision, Proceed to Sweep Search"
                                             (confirm with no fresh
                                             detection — walk side sweeps)
    "SEARCHING <label>"                      (walk side sweeping;
                                             transition-deduped, re-fires on
                                             each re-entry)
    "DETECTED <label> ONCE, EXIT Sweep Motion Search"
                                             (first fresh beat of a run)
    "WAIT FOR THREE CONSECUTIVE CONFIRMATION"
    "Three Confirmation Received, outputting the target check table"
                                             (3rd consecutive beat)
    <target-check table, multi-line>         (label, torso x/y/z, planar
                                             d = hypot(x, y), conf, trigger
                                             note) — follows the status
                                             above on /navigate_status
    "APPROACHING d=.. m"                     (walk_node distance heartbeat,
                                             deduped to >= 0.02 m changes
                                             or state transitions, only
                                             while navigating)
    "REACHED <label>"                        (once — walk_node published
                                             REACHED)
    "DETECTION LOST — re-entering sweep"     (post-trigger: no fresh beat
                                             for > DETECTION_LOST_GAP_S —
                                             label re-sent on /walk_target)
    "NOT_FOUND <label>"                      (once — walk_node exhausted its
                                             sweep and went IDLE; the watch
                                             ENDS here, with no stop and no
                                             `ERROR: walk timeout`)
    "CANCELLED <label>"
    "STOPPED"                  (stop/abort forwarded)
    "WARNING: <reason>"        (e.g. pick state unknown — /pick_status
                                fallback used)
    "ERROR: <reason>"          (gate failure, walk timeout, etc.)

Topics:
    sub /navigate_target, /pick_state, /pick_status, /walk_status,
        /object_detections_body
    pub /walk_target, /walk_go, /navigate_status

Run (Stuart, tv env — same as pick_node):
    /home/unitree/miniconda3/envs/tv/bin/python3 pick_walk_coordinator.py
"""

import importlib
import json
import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ── ref-shaped targets are resolved to plain labels ──────────────────────────
# brain may send 'bottle_3'
# lookups are VERBATIM keys
# here and in walk_node
# -> no match
# 77 s to NOT_FOUND
# 374/714 had it
# window 1790595203-1790595280
# 119 were plain 'bottle'
# gateway/place_target.py: same pair
# missing -> raise
_REF_TO_LABEL_MODULES = ("skills.perception.instance_labels",   # repo: skills/perception/instance_labels.py
                         "instance_labels")                    # robot: openbot/instance_labels.py


def _load_ref_to_label():
    """`ref_to_label` from whichever layout is present; raises if neither is."""
    last = None
    for name in _REF_TO_LABEL_MODULES:
        try:
            return importlib.import_module(name).ref_to_label
        except ImportError as exc:
            last = exc
    raise ImportError("instance_labels not importable (tried %s): %s"
                      % (", ".join(_REF_TO_LABEL_MODULES), last))


ref_to_label = _load_ref_to_label()

TOPIC_NAV_TARGET = '/navigate_target'
TOPIC_NAV_STATUS = '/navigate_status'
TOPIC_PICK_STATE = '/pick_state'
TOPIC_PICK_STATUS = '/pick_status'
TOPIC_WALK_TARGET = '/walk_target'
TOPIC_WALK_STATUS = '/walk_status'
TOPIC_OBJECT_DETECTIONS = '/object_detections_body'
WALK_GO_TOPIC = '/walk_go'   # approach trigger (String label)

PICK_STATE_STALE_S = 2.0
WALK_REACHED_TIMEOUT_S = 120.0
GATE_STATES = ('idle', 'holding')

# PENDING / echo knobs
# projector is EVENT-DRIVEN
# only publishes with detections
# stale entry = last ghost
# never a made-up guess
# 1.0 s guard
DETECTION_FRESH_S = 1.0    # fresh this long
DET_MIN_CONF = 0.20        # pick_node MIN_CONF
APPROACH_DEDUPE_M = 0.02   # min delta

# 3-beat gate
# after each sweep
# projector ~30 fps in view
# beats land inside 1 s
# gap > CONFIRM_BEAT_GAP_S
# msg without label
CONFIRM_BEAT_GAP_S = 1.0     # -> streak reset
CONFIRM_REQUIRED = 3         # max beat gap
# beats to finish
# post-trigger quiet
# LOST + resend
DETECTION_LOST_GAP_S = 2.0   # 2 missed beats at 1 s
# re-arm after a rollback
# /walk_go is ONE-SHOT
# counting stops
# FSM bails -> SEARCHING
# -> DETECTED hold
# only exit: fresh /walk_go
# nobody will send
# needs SILENCE: live object
# keeps beats coming
# froze twice, 'person' held 36 s
# until the 120 s timeout
# STALL_BUDGET = 2
# grace > GO_STALE_S 5.0 s
# + time to collect CONFIRM_REQUIRED beats
REARM_GRACE_S = 8.0


class PickWalkCoordinator(Node):
    def __init__(self):
        super().__init__('pick_walk_coordinator')

        self._nav_lock = threading.Lock()
        self._pending_label = None
        self._navigating = False       # sent, waiting for REACHED
        self._active_label = None
        self._nav_epoch = 0            # stop/abort bumps it; old watches exit

        # post-confirm flow
        #   idle: nothing
        #   sweeping: searching
        #   detecting: counting
        #   confirmed: sent
        self._flow = 'idle'
        self._confirm_count = 0        # consecutive fresh beats
        self._last_beat_ts = 0.0       # last beat (0 = none)
        self._last_walk_state = None   # SEARCHING dedupe

        # pick state (locked)
        self._pick_lock = threading.Lock()
        self._pick_state = None        # dict or None
        self._pick_state_ts = 0.0
        self._pick_status = None
        self._pick_status_ts = 0.0

        self._walk_state = None
        self._walk_state_ts = 0.0

        # NOT_FOUND latch
        # NOT_FOUND then IDLE
        # 1.25 ms apart
        # 1790595280.168 -> .169
        # 0.2 s poll misses it ~99%
        # subscription sees every frame
        self._walk_not_found_label = None
        self._walk_not_found_ts = 0.0

        # latest detections snapshot
        # label -> {x, y, z, conf, _t}
        # same as pick_node; locked
        self._det_lock = threading.Lock()
        self._latest_detections = {}

        # last APPROACHING distance
        self._last_approach_d = None

        self.create_subscription(String, TOPIC_NAV_TARGET,
                                 self._on_nav_target, 10)
        self.create_subscription(String, TOPIC_PICK_STATE,
                                 self._on_pick_state, 10)
        self.create_subscription(String, TOPIC_PICK_STATUS,
                                 self._on_pick_status, 10)
        self.create_subscription(String, TOPIC_WALK_STATUS,
                                 self._on_walk_status, 10)
        self.create_subscription(String, TOPIC_OBJECT_DETECTIONS,
                                 self._on_detections, 10)

        self._walk_pub = self.create_publisher(String, TOPIC_WALK_TARGET, 10)
        self._walk_go_pub = self.create_publisher(String, WALK_GO_TOPIC, 10)
        self._nav_status_pub = self.create_publisher(
            String, TOPIC_NAV_STATUS, 10)

        self._log('ready')

    # ── little logging helpers ────────────────────────────────────────────
    def _log(self, msg):
        self.get_logger().info(f'[coordinator] {msg}')

    def _publish_nav(self, text):
        msg = String()
        msg.data = text
        self._nav_status_pub.publish(msg)
        self._log(f'NAV_STATUS: {text}')

    # ── the subscriptions ─────────────────────────────────────────────────
    def _on_nav_target(self, msg):
        cmd = msg.data.strip().lower()
        with self._nav_lock:
            if cmd in ('stop', 'abort'):
                self._nav_epoch += 1   # kills any in-flight watch
                self._pending_label = None
                self._navigating = False
                self._active_label = None
                # reset gate + flow
                self._flow = 'idle'
                self._confirm_count = 0
                self._last_beat_ts = 0.0
                self._last_approach_d = None
                self._walk_pub.publish(String(data='stop'))
                self._publish_nav('STOPPED')
                return

            if cmd == 'confirm':
                if self._pending_label is None:
                    self._publish_nav('ERROR: nothing pending — '
                                      'send a label first')
                    return
                label = self._pending_label
                if self._navigating:
                    # replace in-flight
                    self._publish_nav(f'ERROR: already navigating to '
                                      f'"{self._active_label}" — send stop '
                                      f'first or wait for REACHED')
                    return
                if not self._pick_gate_ok():
                    return
                self._pending_label = None
                self._navigating = True
                self._active_label = label
                # epoch start
                # state latches
                # nothing clears it
                # across navs
                # so old REACHED survives
                # stale 22 s
                # 'REACHED bottle' 3 ms in
                # label None -> feeder bails
                # /walk_go never again
                # 1434 detections ignored until STOP
                # older = old epoch; stamp first
                since_ts = time.time()
                # sweep = /walk_target publish
                # 7-status ack
                self._walk_pub.publish(String(data=label))
                with self._det_lock:
                    det = self._latest_detections.get(label)
                fresh = (det is not None
                         and time.time() - det.get('_t', 0.0)
                         <= DETECTION_FRESH_S)
                if fresh:
                    # first beat counts
                    self._flow = 'detecting'
                    self._confirm_count = 1
                    self._last_beat_ts = time.time()
                    self._publish_nav(f'DETECTED {label} ONCE, EXIT '
                                      f'Sweep Motion Search')
                    self._publish_nav('WAIT FOR THREE CONSECUTIVE '
                                      'CONFIRMATION')
                else:
                    # absent -> sweep
                    self._flow = 'sweeping'
                    self._confirm_count = 0
                    self._last_beat_ts = 0.0
                    self._publish_nav(f'{label} requested not in vision, '
                                      f'Proceed to Sweep Search')
                threading.Thread(target=self._watch_walk,
                                 args=(since_ts,),
                                 daemon=True).start()
                return

            if cmd == 'cancel':
                self._pending_label = None
                self._publish_nav('CANCELLED')
                return

            # else: a label
            # un-mint REFs at ingress
            # keyed by label
            # so a ref finds nothing
            # even in plain view
            cmd = ref_to_label(cmd)
            if not cmd:
                return
            self._pending_label = cmd
            self._publish_pending(cmd)

    def _on_detections(self, msg):
        """Task 9 snapshot + Task 10 3-consecutive-confirmation gate.

        Mirrors pick_node._on_detections parsing (DET_MIN_CONF filter +
        _t freshness stamp) so the PENDING lookup agrees with what pick_node
        would see.

        Task 10: while the flow is in the WAITING phase ('sweeping' /
        'detecting'), each fresh beat of the ACTIVE label is counted.
        Consecutive means a gap <= CONFIRM_BEAT_GAP_S to the previous beat
        — a label beat after a longer gap, or a detection message that does
        not contain the label at all (projector is event-driven, so a
        published message lists exactly what is currently detected), breaks
        the streak: counter -> 0 and the next label beat restarts the run
        (statuses 3+4 re-fire).  The 3rd consecutive beat completes the
        gate -> _confirm_flow() (status 5 + table + automatic trigger).
        After the trigger ('confirmed') beats no longer count — they only
        feed _last_beat_ts so _watch_walk can detect the projector going
        silent (DETECTION LOST).
        """
        try:
            detections = json.loads(msg.data)
        except Exception:
            self._log(f'bad /object_detections_body: {msg.data!r}')
            return
        now = time.time()
        with self._det_lock:
            self._latest_detections = {
                d['label']: {**d, '_t': now} for d in detections
                if d.get('conf', 0) >= DET_MIN_CONF
            }
        with self._nav_lock:
            label = self._active_label
            flow = self._flow
        if label is None or flow == 'idle':
            return
        with self._det_lock:
            det = self._latest_detections.get(label)
        with self._nav_lock:
            if self._flow != flow:       # stop/abort snuck in
                return
            if flow == 'confirmed':
                # feeds the loss timer
                if det is not None:
                    self._last_beat_ts = now
                return
            if det is None:
                # missing label = miss
                # -> reset, next restarts
                if self._confirm_count != 0 or self._last_beat_ts != 0.0:
                    self._confirm_count = 0
                    self._last_beat_ts = 0.0
                return
            last = self._last_beat_ts
            if last == 0.0 or now - last > CONFIRM_BEAT_GAP_S:
                self._confirm_count = 1      # fresh run
                self._flow = 'detecting'
                streak_started = True
            else:
                self._confirm_count += 1
                streak_started = False
            self._last_beat_ts = now
            triggered = self._confirm_count >= CONFIRM_REQUIRED
        if streak_started:
            self._publish_nav(f'DETECTED {label} ONCE, EXIT '
                              f'Sweep Motion Search')
            self._publish_nav('WAIT FOR THREE CONSECUTIVE CONFIRMATION')
        if triggered:
            self._confirm_flow(det)

    def _publish_pending(self, label):
        """Task 9: PENDING status with detection details when fresh.

        Looks the label up in the latest /object_detections_body snapshot.
        Fresh (<= DETECTION_FRESH_S) -> "PENDING <label> at x=+.. y=+..
        z=.. (d=.. m)" on /navigate_status (d = PLANAR hypot(x, y), per
        researcher §9) plus a CONFIRM?-style box in the log.  Absent or
        stale (event-driven projector ghost) -> plain "PENDING <label>"
        with a "(no fresh detection)" note — never a fabricated position.
        """
        with self._det_lock:
            det = self._latest_detections.get(label)
        if det is not None and time.time() - det.get('_t', 0.0) <= DETECTION_FRESH_S:
            x, y, z = det.get('x', 0.0), det.get('y', 0.0), det.get('z', 0.0)
            d = math.hypot(x, y)   # planar, not 3D
            self._publish_nav(
                f'PENDING {label} at x={x:+.3f} y={y:+.3f} z={z:+.3f} '
                f'(d={d:.2f} m)')
            self.get_logger().info(
                '\n'
                f'  ╔══════════════════════════════════╗\n'
                f'  ║       PENDING TARGET CHECK        ║\n'
                f'  ╠══════════════════════════════════╣\n'
                f'  ║  Target: "{label}"\n'
                f'  ║  Torso frame:\n'
                f'  ║    x (lateral): {x:+.3f} m '
                f'({"right" if x > 0 else "left"})\n'
                f'  ║    y (forward): {y:+.3f} m\n'
                f'  ║    z (height):  {z:+.3f} m '
                f'{"(above)" if z > 0 else "(below)"} torso\n'
                f'  ║  Planar dist:  {d:.3f} m\n'
                f'  ║  Conf:         {det.get("conf", 0.0):.2f}\n'
                f'  ╠══════════════════════════════════╣\n'
                f'  ║  Send "confirm" to /navigate_target\n'
                f'  ╚══════════════════════════════════╝')
        else:
            self._publish_nav(f'PENDING {label} (no fresh detection)')
            self._log(f'{label}: no fresh detection on '
                      f'/object_detections_body (absent or stale > '
                      f'{DETECTION_FRESH_S:.0f}s — event-driven projector) — '
                      f'navigating without position info')

    # ── finishing the gate + approach trigger ───────────────────────────
    def _confirm_flow(self, det):
        """3rd consecutive beat: table first, THEN the automatic trigger.

        Publishes the 7-status status 5 text and the CONFIRM-style
        target-check table (label, torso x/y/z, planar d = hypot(x, y),
        conf, trigger note) on /navigate_status + a box in the log, then
        sends the approach trigger with NO operator step in between —
        the user's directive: walking starts only after the table, and the
        whole search-and-approach is autonomous after the single confirm.
        """
        with self._nav_lock:
            if self._flow not in ('sweeping', 'detecting'):
                return                      # stop/abort snuck in
            self._flow = 'confirmed'
            label = self._active_label
            if label is None:
                return
        x = det.get('x', 0.0)
        y = det.get('y', 0.0)
        z = det.get('z', 0.0)
        d = math.hypot(x, y)                # planar, not 3D
        conf = det.get('conf', 0.0)
        side = 'right' if x > 0 else 'left'
        self._publish_nav('Three Confirmation Received, outputting the '
                          'target check table')
        table = (
            f'  Target: "{label}"\n'
            f'  Torso frame:\n'
            f'    x (lateral): {x:+.3f} m ({side})\n'
            f'    y (forward): {y:+.3f} m\n'
            f'    z (height):  {z:+.3f} m '
            f'{"(above)" if z > 0 else "(below)"} torso\n'
            f'  Planar dist:  {d:.3f} m\n'
            f'  Conf:         {conf:.2f}\n'
            f'  Trigger: approach sent autonomously on {WALK_GO_TOPIC} — '
            f'no confirm step')
        self._publish_nav(table)
        self.get_logger().info(
            '\n'
            f'  ╔══════════════════════════════════╗\n'
            f'  ║    TARGET CHECK — 3 consecutive  ║\n'
            f'  ║         confirmations            ║\n'
            f'  ╠══════════════════════════════════╣\n'
            f'  ║  Target: "{label}"\n'
            f'  ║  Torso frame:\n'
            f'  ║    x (lateral): {x:+.3f} m ({side})\n'
            f'  ║    y (forward): {y:+.3f} m\n'
            f'  ║    z (height):  {z:+.3f} m\n'
            f'  ║  Planar dist:  {d:.3f} m\n'
            f'  ║  Conf:         {conf:.2f}\n'
            f'  ║  -> APPROACH TRIGGER SENT\n'
            f'  ║     ({WALK_GO_TOPIC}), fully autonomous\n'
            f'  ╚══════════════════════════════════╝')
        self._send_approach_trigger(label)

    def _send_approach_trigger(self, label):
        """Approach trigger: /walk_go ONLY (reviewer F1, 2026-08-18).

        The walknode session's walk_node subscribes /walk_go (String label,
        GO_STALE_S=5.0 freshness) and consumes it on the next hold tick
        (~1 s after this publish — well inside the freshness bound).  The
        /walk_target leg of the original dual-topic design is DELIBERATELY
        dropped: walk_node treats ANY /walk_target replay as a RETARGET —
        the retarget check precedes /walk_go consumption in the same hold
        tick — so a paired replay would force a perpetual re-sweep and the
        /walk_go would never be consumed.  /walk_target is only ever used
        for the sweep start (confirm) and the re-sweep (DETECTION LOST
        re-send), NEVER as the approach trigger.  Only called from
        _confirm_flow(), i.e. ONLY after the table has been published.
        """
        self._walk_go_pub.publish(String(data=label))
        self._log(f'approach trigger sent for {label!r} on {WALK_GO_TOPIC}')

    def _on_pick_state(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception:
            self._log(f'bad /pick_state: {msg.data!r}')
            return
        with self._pick_lock:
            self._pick_state = data
            self._pick_state_ts = time.time()

