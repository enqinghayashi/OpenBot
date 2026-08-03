#!/usr/bin/env bash
# launch_pick_walk.sh — T3 combined launcher: pick pipeline (upper-only) +
# walk_node + pick_walk_coordinator in one terminal (Task 3).
#
# Usage (Stuart):
#   exec bash --norc --noprofile
#   source /home/unitree/leon/setup_openbot.sh
#   bash /home/unitree/leon/launch_pick_walk.sh [--trigger <label>]
#          [--navigate <label>] [--restore-pose] [--no-daemon]
#
# Nodes (in order):
#   T0 grip_server        (persistent dex3 hand writers — Task 8; tv python
#                          + the daemon's scrubbed env; survives the pick
#                          daemon's phase-split quit; killed only here in
#                          teardown)
#   T1 openbot_camera_node (python3 $LEON_DIR/openbot_camera_node.py)
#   T2 yolo_node          (COCO gate; bbox_w/h published)
#   T3 depth_projector    (height_m signal)
#   T4 pick_node --upper-only   (spawns pick daemon with the arms-only
#                                controller — legs/waist never written)
#   T5 walk_node          (spawns walk_loco_daemon on first walk)
#   T6 pick_walk_coordinator    (two-step /navigate_target -> /walk_target)
#
# Flags:
#   --trigger <label>  direct /walk_target publish once WalkNode is ready
#                      (legacy one-step; bypasses the pick gate)
#   --navigate <label> two-step through the coordinator: label + confirm
#                      (pick gate: state in {idle, holding} and fresh —
#                      daemon_alive NOT checked since T8 phase-split:
#                      walking REQUIRES the daemon dead, so idle+dead
#                      (--no-daemon default) and holding+dead (after the
#                      pick daemon's quit) both pass; unknown pick state
#                      → /pick_status fallback + visible WARNING.
#                      PENDING shows detection position + planar distance
#                      when fresh, "(no fresh detection)" otherwise)
#   --restore-pose     auto-run restore_arm_pose.py after the daemon's
#                      phase-split quit (watches pick.log for the 'Quit
#                      (phase split)' marker) — DEFAULT OFF; manual step
#                      is the default (user decision: easily switchable,
#                      and off = run nothing, zero impact)
#   --no-daemon        pass through to pick_node: stack boots with ZERO
#                      arm_sdk writers (no pick daemon) — robot can walk
#                      immediately after init. Spawn the daemon on demand
#                      via /arm_command "spawn_daemon" (guarded). Cycle:
#                      launch --no-daemon -> walk anytime -> spawn_daemon
#                      -> pick -> quit -> walk again. Daemon READY poll
#                      skipped (no daemon to wait for).
#
# Node output goes to /tmp/pick_walk_logs/<name>.log — the ready-polls grep
# the logfiles (tailing /proc/PID/fd on a live tty never matches, which made
# the old polls false-negative).  "Shutting down" after a Ctrl+C while a
# poll ran also made the old design print misleading "died during init"
# lines; the logfile polls are immune.
#
# WARNING: requires the robot in motion mode R2+A (spotter on the kill
# switch). Do NOT run alongside launch_loco_pick.sh / launch_walk.sh —
# this launcher IS the combined flow (object in hand, then walk).

set -e

LEON_DIR="${LEON_DIR:-/home/unitree/leon/openbot}"
LOG_DIR=/tmp/pick_walk_logs
mkdir -p "$LOG_DIR"
TRIGGER_OBJ=""
NAVIGATE_OBJ=""
RESTORE_POSE=0
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
        --restore-pose)
            RESTORE_POSE=1
            shift 1
            ;;
        --no-daemon)
            NO_DAEMON=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--trigger object_label] [--navigate object_label]"
            echo "          [--restore-pose] [--no-daemon]"
            exit 1
            ;;
    esac
done

# ── T8 forensics batch: start guards ─────────────────────────────────────────
# Both guards run BEFORE the trap is installed so a FATAL exit cannot run
# cleanup() (cleanup would pkill an orphaned daemon that a DIFFERENT session
# may legitimately be running — e.g. a pick mid-flow in another terminal).
# Duplicate-writer guard: any alive pick daemon / grip server / pick node
# from another session (e.g. SIGHUP orphans) must be resolved by the operator
# FIRST — a second stack would double-write arms (two arm_sdk writers =
# FSM-trip risk) and fight the grip_server's socket.
# Pattern requires a PYTHON invocation ('python.*<name>.py') so a stray
# diagnostic `pgrep -f pick_daemon_v2.py` or `grep` in another terminal
# cannot trip the guard (all stack nodes are launched via python3).
if pgrep -f 'python.*tabletop_pick_place_engine\.py|python.*pick_engine_v[23]\.py|python.*pick_daemon_v2\.py|python.*grip_server\.py|python.*pick_node\.py' >/dev/null 2>&1; then
    echo "[launcher] FATAL: an existing pick stack is already running:" >&2
    pgrep -af 'python.*tabletop_pick_place_engine\.py|python.*pick_engine_v[23]\.py|python.*pick_daemon_v2\.py|python.*grip_server\.py|python.*pick_node\.py' >&2 || true
    echo "[launcher] Kill it first (kill the orphan daemon + grip server, then" >&2
    echo "[launcher] redo RC R2+A per MODE_RECOVERY.md), or this launch will" >&2
    echo "[launcher] double-write the arms." >&2
    exit 1
fi
# Double-launch guard: only one launcher per session (exclude ourselves).
# Pattern requires a 'bash' invocation for the same diagnostic-safety reason.
if pgrep -f 'bash.*launch_pick_walk\.sh' | grep -v "^$$$" >/dev/null 2>&1; then
    echo "[launcher] FATAL: launch_pick_walk.sh is already running (another" >&2
    echo "[launcher] session/terminal). Close it first, then re-launch." >&2
    exit 1
fi

cleanup() {
    echo ""
    echo "[launcher] Shutting down all nodes..."
    kill $(jobs -p) 2>/dev/null
    # The pick daemon is a CHILD of pick_node — killing the jobs list
    # ORPHANS it.  An orphaned daemon keeps the arm_sdk writer alive,
    # which trips the FSM into mode 3 (RC sticks dead, gait blocked).
    # 2026-08-16: recovered without reboot only after the daemon was
    # truly dead (foreground Ctrl+C) + RC redo.  Kill it explicitly.
    # 2026-09-13: the pick engine is now `tabletop_pick_place_engine.py`
    # (pick_node's DAEMON_SCRIPT), NOT pick_daemon_v2.py — so the old pattern
    # could not see it.  It is RESIDENT by design (the gateway attaches to its
    # socket rather than spawning a second engine), and it survives stdin EOF,
    # so this teardown is the only thing that stops it.  Stop it GRACEFULLY
    # first: the engine registers no signal handler, and only the `quit` verb
    # runs the closeout that releases the writer RC interlock — a bare pkill
    # would leave RC held.
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
    # Legacy name kept: a pre-2026-09-13 stack may still have one running.
    pkill -f 'pick_daemon_v2.py' 2>/dev/null
    # T8: the persistent grip server is a launcher child too — teardown
    # must kill it so no dex3 writer survives a "killed" stack (same
    # orphan logic as the daemon; the server never trips the FSM — it is
    # a SEPARATE dex3 MCU — but a stray writer would fight a future start).
    pkill -f 'grip_server.py' 2>/dev/null
    wait 2>/dev/null
    echo "[launcher] Done."
}
# T8: HUP added — a closed terminal sends SIGHUP, which previously orphaned
# the daemon and grip_server (both run until pkill; an orphaned daemon is an
# alive arm_sdk writer in the walk phase = trip risk).
trap cleanup EXIT INT TERM HUP

echo "=== Launching pick (upper-only) + walk + coordinator pipeline ==="
echo "[launcher] Logs: $LOG_DIR"

# ── Task 7 pre-flight: fsm_mode must be 0 (gait NOT blocked) ───────────────
# 2026-08-16 incident: fsm_mode flipped 0 -> 3 MID-SESSION (two arm_sdk
# writers: manual tool with the STOCK 35-joint controller + the upper-only
# daemon).  Mechanism REVISED (researcher §7.1): sticks die on EVERY launcher
# start alone (no stock tool) — prime suspect = daemon startup ARM MOTION
# during R2+A; legs-written entries = AGGRAVATOR, not root cause.  Walk
# commands are then ACCEPTED but no gait runs.  ORDERING RULE (see
# MODE_RECOVERY.md): daemon startup arm motion COMPLETES to a steady neutral
# hold BEFORE R2+A engages; no arm transients during R2+A (P3-proven steady
# hold safe).
# WARN LOUDLY + suggest the recovery ladder — we do NOT block the launch:
# the operator may already be mid-recovery or running headless.
FSM_HELPER=""
for _cand in "$LEON_DIR/fsm_check.py" /home/unitree/leon/fsm_check.py \
             "$(dirname "$0")/fsm_check.py"; do
    if [ -f "$_cand" ]; then
        FSM_HELPER="$_cand"
        break
    fi
done
if [ -n "$FSM_HELPER" ]; then
    FSM_OUT=$(python3 "$FSM_HELPER" 2>&1)
    FSM_RC=$?
    echo "[launcher] fsm pre-flight ($FSM_HELPER):"
    echo "$FSM_OUT" | sed 's/^/[launcher]   /'
    if [ "$FSM_RC" -eq 1 ]; then
        echo "╔══════════════════════════════════════════════════════════════════════╗"
        echo "║ WARNING: fsm_mode != 0 — GAIT IS BLOCKED (walk commands will be      ║"
        echo "║ accepted but NO gait will run).  Recover: MODE_RECOVERY.md ladder.   ║"
        echo "║ Primary: RC L1+B -> L2+UP -> R2+A;  reboot = fallback.               ║"
        echo "║ StopMove / SetBalanceMode(0) = UNVERIFIED options (diagnostics only).║"
        echo "║ Verify with fsm_check.py until fsm_mode=0 BEFORE driving the robot.  ║"
        echo "╚══════════════════════════════════════════════════════════════════════╝"
    elif [ "$FSM_RC" -eq 2 ]; then
        echo "[launcher] WARNING: fsm_mode UNREADABLE (loco not reachable) — "
        echo "           cannot verify gait state; check the loco process."
    fi
else
    echo "[launcher] WARNING: fsm_check.py not found — skipping FSM pre-flight"
fi

# T0 - grip_server (persistent dex3 hand writers — Task 8, FULL SPLIT:
# owns rt/dex3/{side}/cmd 100%; daemon routes hand commands through it;
# NEVER opens the hands on boot; survives the pick daemon's phase-split
# quit so the grip holds through the walk).  Run with the tv python and
# the daemon's scrubbed env (DAEMON_ENV in pick_node) — the ROS-sourced
# shell env must not leak into the SDK writer.
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
# NOTE: Stuart-deployed names (scp targets), not the laptop source names.
echo "[launcher] T2: yolo_node..."
python3 "$LEON_DIR/yolo_node.py" > "$LOG_DIR/yolo.log" 2>&1 &
YOLO_PID=$!

# T3 - Depth projector (height_m signal)
echo "[launcher] T3: depth_projector_node..."
python3 "$LEON_DIR/depth_projector_node.py" > "$LOG_DIR/depth.log" 2>&1 &
DEPTH_PID=$!

# Give perception nodes time to start
sleep 2

# T4 - pick_node with --upper-only (daemon runs the arms-only controller)
# --no-daemon passthrough: skip the daemon spawn entirely (zero writers at
# boot; spawn via /arm_command "spawn_daemon" later — see flags above).
PICK_EXTRA_ARGS="--upper-only"
if [ "$NO_DAEMON" -eq 1 ]; then
    PICK_EXTRA_ARGS="$PICK_EXTRA_ARGS --no-daemon"
    echo "[launcher] T4: pick_node --upper-only --no-daemon (ZERO arm_sdk writers; spawn daemon on demand)..."
else
    echo "[launcher] T4: pick_node --upper-only (daemon loads pinocchio ~10s)..."
fi
python3 "$LEON_DIR/pick_node.py" $PICK_EXTRA_ARGS > "$LOG_DIR/pick.log" 2>&1 &
PICK_PID=$!

# T5 - walk_node (locomotion FSM; spawns walk_loco_daemon on first walk)
echo "[launcher] T5: walk_node..."
python3 "$LEON_DIR/walk_node.py" > "$LOG_DIR/walk.log" 2>&1 &
WALK_PID=$!

# T6 - coordinator (two-step navigate handshake + pick gate)
echo "[launcher] T6: pick_walk_coordinator..."
python3 "$LEON_DIR/pick_walk_coordinator.py" > "$LOG_DIR/coord.log" 2>&1 &
COORD_PID=$!

# T7 - arrival_relay (async SPOKEN WALK NARRATION, operator directive
# 2026-09-28: "sweeping to find it / oh I see it / walking over").  Subscribes
# /navigate_status, maps each line through the pure arrival_announce law, and
# writes the phrase to /tmp/arrival_announce.sock, which the voice process
# reads in its listener thread.
#
# Best-effort BY DESIGN: no ready poll to gate on, and no pkill — it is a plain
# background job reaped by cleanup()'s `kill $(jobs -p)`.  This launcher
# previously never started it (the relay was armed only by
# launch_merge_pick_v2b.sh:330, whose walk nodes are marked UNUSED), so the
# narration was silently dead in the walk flow — and a silent feature is
# indistinguishable from a broken one.
#
# REQUIRES arrival_relay.py + arrival_announce.py in $LEON_DIR beside
# pick_walk_coordinator.py: the relay imports the phrase law from its OWN
# script dir.  If they are missing this job dies quietly and ONLY the
# narration is lost — the walk is unaffected — so it must never gate the launch.
echo "[launcher] T7: arrival_relay (async spoken walk narration)..."
python3 "$LEON_DIR/arrival_relay.py" > "$LOG_DIR/arrival_relay.log" 2>&1 &
RELAY_PID=$!

# ── Ready poll: pick daemon ───────────────────────────────────────────────
# (skipped with --no-daemon: no daemon is spawned at boot — pick_node
# itself is up when its "Pick node ready" log line appears, and the
# daemon can be spawned on demand via /arm_command "spawn_daemon")
if [ "$NO_DAEMON" -eq 1 ]; then
    echo "[launcher] --no-daemon: skipping daemon READY poll (spawn on demand via /arm_command \"spawn_daemon\")"
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
        echo "[launcher] WARNING: pick daemon READY not detected within 120s — see $LOG_DIR/pick.log"
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

# ── --restore-pose watcher (DEFAULT OFF): after the daemon's phase-split
# quit, auto-restore the recorded arm posture (upper-only, one-shot, exits
# before any R2+A — trip-safe).  Watches pick.log for the daemon's quit
# stderr marker (pick_node's stderr logger prefixes '[daemon] ...').
if [ "$RESTORE_POSE" -eq 1 ]; then
    (
        while ! grep -q "Quit (phase split)" "$LOG_DIR/pick.log" 2>/dev/null; do
            sleep 1
        done
        sleep 2  # let the daemon finish release_arms + DONE + exit
        echo "[launcher] --restore-pose: restoring recorded arm posture..."
        "$GRIP_PYTHON" "$LEON_DIR/restore_arm_pose.py" \
            || echo "[launcher] --restore-pose: restore refused/failed — manual posture setup"
    ) &
fi

echo ""
echo "=== All nodes running ==="
echo "  Grip:         PID $GRIP_PID   (log: $LOG_DIR/grip.log)"
echo "  Camera:       PID $CAM_PID   (log: $LOG_DIR/camera.log)"
echo "  YOLO:         PID $YOLO_PID"
echo "  Depth:        PID $DEPTH_PID"
echo "  Pick (upper): PID $PICK_PID"
echo "  Walk:         PID $WALK_PID"
echo "  Coordinator:  PID $COORD_PID"
echo "  Relay:        PID $RELAY_PID   (log: $LOG_DIR/arrival_relay.log)"
echo ""
echo "Two-step navigate:"
echo "  ros2 topic pub /navigate_target std_msgs/msg/String \"{data: 'chair'}\" --once"
echo "  ros2 topic pub /navigate_target std_msgs/msg/String \"{data: 'confirm'}\" --once"
echo "Stop/abort:  ros2 topic pub /navigate_target std_msgs/msg/String \"{data: 'stop'}\" --once"
echo "Direct walk: ros2 topic pub /walk_target std_msgs/msg/String \"{data: 'chair'}\" --once"
echo "Statuses:    ros2 topic echo /navigate_status | /walk_status | /pick_state"
echo ""

if [ -n "$TRIGGER_OBJ" ]; then
    sleep 3
    echo "[launcher] Auto-triggering DIRECT walk toward '$TRIGGER_OBJ'..."
    ros2 topic pub /walk_target std_msgs/msg/String "{data: '$TRIGGER_OBJ'}" --once
fi

if [ -n "$NAVIGATE_OBJ" ]; then
    sleep 3
    echo "[launcher] Auto-navigating via coordinator toward '$NAVIGATE_OBJ'..."
    ros2 topic pub /navigate_target std_msgs/msg/String "{data: '$NAVIGATE_OBJ'}" --once
    sleep 1
    ros2 topic pub /navigate_target std_msgs/msg/String "{data: 'confirm'}" --once
fi

# Keep running until Ctrl+C
wait
