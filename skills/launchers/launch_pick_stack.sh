#!/usr/bin/env bash
# launch_merge_pick_v2b.sh — v2b combined launcher: perception pipeline +
# non-owning pick ENGINE inside the merged-writer world (no phase split).
#
# Usage (Stuart) — the merged writer must ALREADY be running:
#   exec bash --norc --noprofile
#   source /home/unitree/leon/setup_openbot.sh
#   bash /home/unitree/leon/launch_merge_pick_v2b.sh [--trigger <label>]
#          [--navigate <label>] [--no-daemon]
#
# Nodes (in order):
#   T0 grip_server        (persistent dex3 hand writers — Task 8; tv python
#                          + scrubbed env; trip-law-exempt: SEPARATE dex3 MCU)
#   T1 openbot_camera_node (python3 $LEON_DIR/openbot_camera_node.py)
#   T2 yolo_node          (COCO gate; bbox_w/h published)
#   T3 depth_projector    (height_m signal)
#   T4 pick_node          (spawns the NON-OWNING tabletop_pick_place_engine.py:
#                          pinocchio IK + place + T5 sequence + guarded insert
#                          + grip verify; it serves /tmp/pick_engine.sock so
#                          the gateway can ATTACH to it for a place instead of
#                          spawning a second engine;
#                          the engine commands the merged writer over
#                          /tmp/merged_writer.sock — NO arm_sdk writer, NO
#                          second motor owner, structurally cannot trip)
#   T5 walk_node          (v2b: UNUSED — do NOT trigger; see WARNING below)
#   T6 pick_walk_coordinator  (v2b: UNUSED — auto-navigation is v2c)
#   T7 arrival_relay      (async spoken-arrival seam, track 214-d; subscribes
#                          /navigate_status, maps REACHED via the PURE
#                          arrival_announce law to /tmp/arrival_announce.sock;
#                          best-effort + deploy-gated — see the T7 spawn)
#
# v2b session (NO PHASE SPLIT, one continuous writer session):
#   writer up (owns rt/lowcmd + arms from capture) → this launcher →
#   operator RC walks to the table (writer Walk policy) → /arm_command pick
#   (engine: hold_rc clamp → ready → tuck → wp1-4 → guarded insert →
#   grasp verify → LIFT → CARRY_TUCK → hold_rc off) → RC walks AWAY with
#   the cup (engine still alive) → /arm_command drop → Ctrl-C everything.
#   The engine NEVER quits for the walk; the writer NEVER hands the arms
#   over.  MotionSwitcher stays RELEASED; the writer exits zero-torque at
#   shutdown (no SelectMode in C3b).
#
# WARNING (v2b): --trigger / --navigate drive walk_node / coordinator over
# sport RPC, which the released-switcher world refuses (error 3102, FSM
# 801 class) and a /walk_target WOULD SPAWN walk_loco_daemon = a SECOND
# rt/lowcmd writer = ABSOLUTE TRIP.  v2b walking is OPERATOR RC ONLY;
# autopilot (both directions) is v2c.  walk_node/coordinator launch
# harmlessly if never triggered.
#
# Flags:
#   --no-daemon        pass through to pick_node: stack boots with ZERO
#                      pick engine (writer alone owns everything; walk
#                      immediately).  Spawn the engine on demand via
#                      /arm_command "spawn_daemon".
#
# Node output goes to /tmp/pick_walk_logs/<name>.log — the ready-polls grep
# the logfiles (tailing /proc/PID/fd on a live tty never matches, which made
# the old polls false-negative).
#
# WARNING: requires the robot in motion mode R2+A (spotter on the kill
# switch). Do NOT run alongside launch_loco_pick.sh / launch_walk.sh.

# --- openbot route pin (user directive 2026-09-05) --------------------------
# Runs on EVERY OpenBot framework launch.  Idempotent, silent-on-404, safe:
#   (1) drop a DEAD eth0 default route ('default via 192.168.123.1 dev eth0' —
#       the #105-class shadow that broke box egress over a working wlan0).
#       The del names gateway + dev EXPLICITLY: a working wlan0 default is
#       never touched; absent route = silent 404 (rc 1 -> || true — REQUIRED
#       under `set -e`, or the 404 aborts the launcher).
#   (2) pin multicast (224.0.0.0/4) to eth0 so Unitree-SDK DDS SPDP discovery
#       and the voice mic group stay on the robot ethernet regardless of which
#       interface holds the default.  Re-running when the route already exists
#       errors 'File exists' — suppressed; any OTHER add failure warns loudly.
openbot_route_pin() {
  sudo ip route del default via 192.168.123.1 dev eth0 2>/dev/null || true
  if ! sudo ip route add 224.0.0.0/4 dev eth0 2>/dev/null; then
    ip route show 224.0.0.0/4 dev eth0 >/dev/null 2>&1 \
      || echo "openbot_route_pin: WARNING: cannot pin 224.0.0.0/4 dev eth0 (route absent; add needs root — check sudo)" >&2
  fi
}
openbot_route_pin

set -e

LEON_DIR="${LEON_DIR:-/home/unitree/leon/openbot}"
LOG_DIR=/tmp/pick_walk_logs
mkdir -p "$LOG_DIR"
TRIGGER_OBJ=""
NAVIGATE_OBJ=""
NO_DAEMON=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trigger)
            TRIGGER_OBJ="$2"
            shift 2
            ;;
        --navigate)
            NAVIGATE_OBJ="$2"
            shift 2
            ;;
        --no-daemon)
            NO_DAEMON=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--trigger object_label] [--navigate object_label]"
            echo "          [--no-daemon]"
            exit 1
            ;;
    esac
done

# ── T8 forensics batch: start guards ─────────────────────────────────────────
# Duplicate-stack guard: any alive pick ENGINE / grip server / pick node from
# another session must be resolved by the operator FIRST.  Note the engine is
# NON-OWNING (no arm_sdk writer, no trip risk on its own) — the guard exists
# so two pick_nodes cannot fight over the socket + grip server.
# Pattern requires a PYTHON invocation ('python.*<name>.py') so a stray
# diagnostic `pgrep -f pick_engine_v2.py` in another terminal cannot trip it.
# 2026-09-13: the pattern also carries `tabletop_pick_place_engine.py`, the
# entrypoint T4 now spawns (it wraps pick_engine_v3 and serves
# /tmp/pick_engine.sock for the gateway to attach to).
if pgrep -f 'python.*tabletop_pick_place_engine\.py|python.*pick_engine_v[23]\.py|python.*grip_server\.py|python.*pick_node\.py' >/dev/null 2>&1; then
    echo "[launcher] FATAL: an existing pick stack is already running:" >&2
    pgrep -af 'python.*tabletop_pick_place_engine\.py|python.*pick_engine_v[23]\.py|python.*grip_server\.py|python.*pick_node\.py' >&2 || true
    echo "[launcher] Kill it first, or this launch will fight over the" >&2
    echo "[launcher] writer socket + grip server socket." >&2
    exit 1
fi
# Double-launch guard: only one launcher per session (exclude ourselves).
if pgrep -f 'bash.*launch_merge_pick_v2b\.sh' | grep -v "^$$$" >/dev/null 2>&1; then
    echo "[launcher] FATAL: launch_merge_pick_v2b.sh is already running (another" >&2
    echo "[launcher] session/terminal). Close it first, then re-launch." >&2
    exit 1
fi

cleanup() {
    echo ""
    echo "[launcher] Shutting down all nodes..."
    kill $(jobs -p) 2>/dev/null
    # The pick engine is a CHILD of pick_node — killing the jobs list
    # ORPHANS it.  BENIGN in v2b (unlike the daemon): the engine owns NO
    # motors.  Kill it anyway so no stale socket client lingers.
    #
    # 2026-09-13: the "on stdin EOF its command loop ends and it exits" half
    # of that is NO LONGER TRUE.  The engine is now RESIDENT by design (the
    # gateway attaches to its socket instead of spawning one), so it must
    # survive a closed stdin — which means this pkill is the ONLY teardown
    # that stops it.  Keep both names: a pre-2026-09-13 engine may still be
    # running from an older stack.
    #
    # STOP IT GRACEFULLY FIRST.  The engine has NO SIGTERM/SIGINT handler
    # (pick_engine_v3.py registers none), and only the `quit` verb runs its
    # closeout — the writer RC-interlock release.  A bare pkill therefore
    # SIGTERMs a resident engine that may still hold RC, leaving the writer
    # interlock held for the next session.  'quit' over its own socket is the
    # clean stop; pkill stays as the backstop for a wedged engine.
    _engine_quit() {
        [ -S /tmp/pick_engine.sock ] || return 0
        python3 - <<'PYEOF' 2>/dev/null || true
import socket
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    s.settimeout(10.0)
    s.connect("/tmp/pick_engine.sock")
    f = s.makefile("rb")
    f.readline()                 # READY banner
    s.sendall(b"quit\n")
    f.readline()                 # DONE (closeout ran)
except OSError:
    pass
finally:
    s.close()
PYEOF
    }
    _engine_quit
    pkill -f 'tabletop_pick_place_engine.py|pick_engine_v[23].py' 2>/dev/null
    # T8: the persistent grip server is a launcher child too — teardown
    # must kill it so no dex3 writer survives a "killed" stack.
    pkill -f 'grip_server.py' 2>/dev/null
    wait 2>/dev/null
    echo "[launcher] Done."
}
# T8: HUP added — a closed terminal sends SIGHUP, which previously orphaned
# the daemon and grip_server.
trap cleanup EXIT INT TERM HUP

echo "=== Launching pick (non-owning engine) + writer world (NO PHASE SPLIT) ==="
echo "[launcher] Logs: $LOG_DIR"

# ── v2b pre-flight: the merged writer MUST already own the robot ────────────
if [ ! -S /tmp/merged_writer.sock ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║ WARNING: /tmp/merged_writer.sock NOT FOUND — the merged writer is    ║"
    echo "║ not running.  v2b order: START THE WRITER FIRST (it owns rt/lowcmd   ║"
    echo "║ and the arms from capture), then this launcher.  The pick engine     ║"
    echo "║ will fail-fast on connect if the writer is absent.                   ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
fi

# ── Task 7 pre-flight: fsm_mode must be 0 (gait NOT blocked) ───────────────
FSM_HELPER=""
for _cand in "$LEON_DIR/fsm_check.py" /home/unitree/leon/fsm_check.py \
             "$(dirname "$0")/fsm_check.py"; do
    if [ -f "$_cand" ]; then
        FSM_HELPER="$_cand"
        break
    fi
done
if [ -n "$FSM_HELPER" ] && [ -S /tmp/merged_writer.sock ]; then
    # v2b (2026-08-24): while the writer holds, the loco/sport query
    # services are dead BY DESIGN (switcher released -> ClientStub 3102) —
    # the fsm_mode query cannot answer (rc=2 every time).  fsm_mode=0 is
    # guaranteed by the writer's own CheckMode->ReleaseMode boot; re-run
    # the check only when the writer is NOT running.
    echo "[launcher] fsm pre-flight SKIPPED: writer owns the robot"
    echo "[launcher]   (switcher released -> loco query dead, expected;"
    echo "[launcher]    fsm_mode=0 guaranteed by the writer's ReleaseMode)"
elif [ -n "$FSM_HELPER" ]; then
    # set -e guard (found on gantry 2026-08-24): fsm_check exits 1 when
    # gait is blocked and 2 when unreadable — a bare command substitution
    # killed the launcher (EXIT trap -> cleanup) before it could report.
    FSM_OUT=$(python3 "$FSM_HELPER" 2>&1) || FSM_RC=$?
    echo "[launcher] fsm pre-flight ($FSM_HELPER):"
    echo "$FSM_OUT" | sed 's/^/[launcher]   /'
    if [ "$FSM_RC" -eq 1 ]; then
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║ WARNING: fsm_mode != 0 — GAIT IS BLOCKED.  Recover: MODE_RECOVERY.md.║"
        echo "║ Primary: RC L1+B -> L2+UP -> R2+A;  reboot = fallback.               ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
    elif [ "$FSM_RC" -eq 2 ]; then
        echo "[launcher] WARNING: fsm_mode UNREADABLE (loco not reachable) — "
        echo "           cannot verify gait state; check the loco process."
    fi
else
    echo "[launcher] WARNING: fsm_check.py not found — skipping FSM pre-flight"
fi

# T0 - grip_server (persistent dex3 hand writers — Task 8, FULL SPLIT:
# owns rt/dex3/{side}/cmd 100%; engine routes hand commands through it;
# NEVER opens the hands on boot).  Run with the tv python and the scrubbed
# env (DAEMON_ENV in pick_node) — the ROS-sourced shell env must not leak
# into the SDK writer.
GRIP_PYTHON=/home/unitree/miniconda3/envs/tv/bin/python3
echo "[launcher] T0: grip_server (persistent dex3 hand writers)..."
env -i HOME=/home/unitree USER=unitree \
    PATH=/home/unitree/miniconda3/envs/tv/bin:/home/unitree/miniconda3/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH= \
    LD_LIBRARY_PATH=/home/unitree/miniforge3/envs/pinocchio_env/lib:/home/unitree/miniconda3/envs/tv/lib \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    "$GRIP_PYTHON" "$LEON_DIR/grip_server.py" > "$LOG_DIR/grip.log" 2>&1 &
GRIP_PID=$!

# ── Ready poll: grip server (before pick_node ever starts) ────────────────
echo "[launcher] Waiting for grip server READY..."
GRIP_READY=0
for i in $(seq 1 120); do
    if grep -qE '^READY$' "$LOG_DIR/grip.log" 2>/dev/null; then
        GRIP_READY=1
        break
    fi
    if ! kill -0 $GRIP_PID 2>/dev/null; then
        echo "[launcher] ERROR: grip_server died during init — see $LOG_DIR/grip.log"
        GRIP_READY=0
        break
    fi
    sleep 0.5
done
if [ "$GRIP_READY" -eq 0 ]; then
    echo "[launcher] WARNING: grip server READY not detected within 60s — see $LOG_DIR/grip.log"
fi

# T1 - Camera
echo "[launcher] T1: openbot_camera_node..."
python3 "$LEON_DIR/openbot_camera_node.py" > "$LOG_DIR/camera.log" 2>&1 &
CAM_PID=$!

# T2 - YOLO (COCO gate; bbox_w/bbox_h for the height chain)
#
# INTERPRETER PINNED TO jp_env (2026-09-30).  Carried over from the box's copy
# of THIS file, which had the pin while the repo's had a bare `python3`.  It is
# kept because it is the only form PROVEN on the box: run9 ran through this
# launcher with the pin and produced detections (yolo.log carried the bottle and
# person frames).  Bare `python3` is unproven HERE.
#
# HONEST CAVEAT — the pin is an ANOMALY, not the house convention.  Measured
# across the box's five launchers that spawn yolo_node.py, `jp_env` appears in
# exactly one (this line); launch_pick_walk.sh:262, launch_loco_pick_robot.sh:60,
# launch_walk.sh:76, b1_relaunch.sh:26 and launch_pick_pipeline.sh:64 all use a
# bare `python3` — and the box kept launch_pick_walk.sh byte-identical to the
# repo's, bare form and all.  That argues the pin may be vestigial.  It stays
# until someone can check on the box, because the two ways to be wrong are not
# symmetric: shipping the pin when it was unnecessary still lands a working
# interpreter (jp_env has cv2/ultralytics), while shipping the bare form when
# the pin was needed kills yolo_node.py at import — into yolo.log, where the T2
# ready poll reads the failure as a slow start.
#
# The PYTHONPATH is the foxy site-packages the rclpy shim needs under jp_env.
echo "[launcher] T2: yolo_node..."
PYTHONPATH=/opt/ros/foxy/lib/python3.8/site-packages:/usr/lib/python3.8/dist-packages /home/unitree/leon/jp_env/bin/python3 "$LEON_DIR/yolo_node.py" > "$LOG_DIR/yolo.log" 2>&1 &
YOLO_PID=$!

# T3 - Depth projector (height_m signal)
echo "[launcher] T3: depth_projector_node..."
python3 "$LEON_DIR/depth_projector_node.py" > "$LOG_DIR/depth.log" 2>&1 &
DEPTH_PID=$!

# Give perception nodes time to start
sleep 2

# T4 - pick_node (spawns the NON-OWNING engine; no --upper-only in v2b —
# the upper/stock controller split is gone, there is exactly one engine)
# --no-daemon passthrough: skip the engine spawn entirely (writer alone at
# boot; spawn via /arm_command "spawn_daemon" later — see flags above).
PICK_EXTRA_ARGS=""
if [ "$NO_DAEMON" -eq 1 ]; then
    PICK_EXTRA_ARGS="$PICK_EXTRA_ARGS --no-daemon"
    echo "[launcher] T4: pick_node --no-daemon (ZERO pick engine; spawn on demand)..."
else
    echo "[launcher] T4: pick_node (engine loads pinocchio ~10s, connects to writer socket)..."
fi
python3 "$LEON_DIR/pick_node.py" $PICK_EXTRA_ARGS > "$LOG_DIR/pick.log" 2>&1 &
PICK_PID=$!

# T5 - walk_node (v2b: UNUSED — see header WARNING; never trigger, a
# /walk_target would spawn walk_loco_daemon = second rt/lowcmd writer = trip)
echo "[launcher] T5: walk_node (UNUSED in v2b — RC walking only)..."
python3 "$LEON_DIR/walk_node.py" > "$LOG_DIR/walk.log" 2>&1 &
WALK_PID=$!

# T6 - coordinator (v2b: UNUSED — auto-navigation is v2c)
echo "[launcher] T6: pick_walk_coordinator (UNUSED in v2b)..."
python3 "$LEON_DIR/pick_walk_coordinator.py" > "$LOG_DIR/coord.log" 2>&1 &
COORD_PID=$!

# T7 - arrival_relay (async spoken-arrival seam, track 214-d).  Subscribes
# /navigate_status (the coordinator publishes one "REACHED <label>" per
# navigation), maps it through the PURE arrival_announce law, and writes the
# spoken phrase to the /tmp/arrival_announce.sock UNIX socket that the voice
# process reads.  Must sit NEXT TO arrival_announce.py in $LEON_DIR (the
# import resolves from the relay's own script dir).  Best-effort: no ready
# poll to gate on, and no pkill — it is a plain background job that the
# cleanup() `kill $(jobs -p)` already reaps at teardown.
# DEPLOY-GATED (2026-09-08): the box is down — nothing runs now.  To arm the
# feature at deploy, ship arrival_relay.py + arrival_announce.py to $LEON_DIR
# beside pick_walk_coordinator.py; this spawn then brings the relay up with
# the coordinator.
echo "[launcher] T7: arrival_relay (async spoken arrival; deploy-gated)..."
python3 "$LEON_DIR/arrival_relay.py" > "$LOG_DIR/arrival_relay.log" 2>&1 &
RELAY_PID=$!

# ── Ready poll: pick engine ────────────────────────────────────────────────
# (skipped with --no-daemon: no engine is spawned at boot)
if [ "$NO_DAEMON" -eq 1 ]; then
    echo "[launcher] --no-daemon: skipping engine READY poll (spawn on demand via /arm_command \"spawn_daemon\")"
    PICK_READY=0
    for i in $(seq 1 120); do
        if grep -q "Pick node ready" "$LOG_DIR/pick.log" 2>/dev/null; then
            PICK_READY=1
            break
        fi
        if ! kill -0 $PICK_PID 2>/dev/null; then
            echo "[launcher] ERROR: pick_node died during init — see $LOG_DIR/pick.log"
            PICK_READY=0
            break
        fi
        sleep 0.5
    done
    if [ "$PICK_READY" -eq 0 ]; then
        echo "[launcher] WARNING: 'Pick node ready' not detected within 60s — see $LOG_DIR/pick.log"
    fi
else
    PICK_READY=0
    for i in $(seq 1 240); do
        if grep -qE "Pick daemon ready" "$LOG_DIR/pick.log" 2>/dev/null; then
            PICK_READY=1
            break
        fi
        if ! kill -0 $PICK_PID 2>/dev/null; then
            echo "[launcher] ERROR: pick_node died during init — see $LOG_DIR/pick.log"
            PICK_READY=0
            break
        fi
        sleep 0.5
    done
    if [ "$PICK_READY" -eq 0 ]; then
        echo "[launcher] WARNING: pick engine READY not detected within 120s — see $LOG_DIR/pick.log"
    fi
fi

# ── Ready poll: walk_node ─────────────────────────────────────────────────
echo "[launcher] Waiting for walk_node READY..."
WALK_READY=0
for i in $(seq 1 120); do
    if grep -q "WalkNode ready" "$LOG_DIR/walk.log" 2>/dev/null; then
        WALK_READY=1
        break
    fi
    if ! kill -0 $WALK_PID 2>/dev/null; then
        echo "[launcher] ERROR: walk_node died during init — see $LOG_DIR/walk.log"
        WALK_READY=0
        break
    fi
    sleep 0.5
done
if [ "$WALK_READY" -eq 0 ]; then
    echo "[launcher] WARNING: 'WalkNode ready' not detected within 60s — see $LOG_DIR/walk.log"
fi

# ── Ready poll: coordinator ───────────────────────────────────────────────
echo "[launcher] Waiting for coordinator READY..."
COORD_READY=0
for i in $(seq 1 60); do
    if grep -q "coordinator] ready" "$LOG_DIR/coord.log" 2>/dev/null; then
        COORD_READY=1
        break
    fi
    if ! kill -0 $COORD_PID 2>/dev/null; then
        echo "[launcher] ERROR: coordinator died during init — see $LOG_DIR/coord.log"
        COORD_READY=0
        break
    fi
    sleep 0.5
done
if [ "$COORD_READY" -eq 0 ]; then
    echo "[launcher] WARNING: coordinator 'ready' not detected within 30s — see $LOG_DIR/coord.log"
fi

echo ""
echo "=== All nodes running ==="
echo "  Grip:         PID $GRIP_PID   (log: $LOG_DIR/grip.log)"
echo "  Camera:       PID $CAM_PID   (log: $LOG_DIR/camera.log)"
echo "  YOLO:         PID $YOLO_PID"
echo "  Depth:        PID $DEPTH_PID"
echo "  Pick engine:  PID $PICK_PID"
echo "  Walk:         PID $WALK_PID   (v2b: unused)"
echo "  Coordinator:  PID $COORD_PID  (v2b: unused)"
echo "  ArrivalRelay: PID $RELAY_PID  (deploy-gated; log: $LOG_DIR/arrival_relay.log)"
echo ""
echo "v2b flow (NO PHASE SPLIT):"
echo "  RC-walk to the table, stop, then:"
echo "  Pick:   ros2 topic pub /pick_target std_msgs/msg/String \"{data: 'water bottle'}\" --once   # yolo label"
echo "  Confirm: ros2 topic pub /pick_target std_msgs/msg/String \"{data: 'confirm'}\" --once"
echo "  (/pick_state goes held:true -> writer CARRY_TUCK -> hold_rc off)"
echo "  RC-walk away with the cup (engine stays alive)."
echo "  Drop:   ros2 topic pub /arm_command std_msgs/msg/String \"{data: 'drop'}\" --once"
echo "  Quit:   ros2 topic pub /arm_command std_msgs/msg/String \"{data: 'quit'}\" --once"
echo "  Status: ros2 topic echo /pick_state"
echo ""

if [ -n "$TRIGGER_OBJ" ] || [ -n "$NAVIGATE_OBJ" ]; then
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║ FATAL-WARNING: --trigger/--navigate are v2c (autopilot), NOT v2b.    ║"
    echo "║ A /walk_target would spawn walk_loco_daemon = SECOND rt/lowcmd       ║"
    echo "║ writer = ABSOLUTE TRIP while the merged writer owns the robot.       ║"
    echo "║ v2b walking is OPERATOR RC ONLY. Aborting the trigger.               ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
fi

# Keep running until Ctrl+C
wait
