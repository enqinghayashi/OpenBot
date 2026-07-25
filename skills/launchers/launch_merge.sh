#!/usr/bin/env bash
# launch_merge.sh — C3 merged whole-body writer launch (pick-walk_skill merge_version)
#
# Env-scrubbed launch: flock + pgrep duplicate-writer guards, fsm pre-flight,
# onnxruntime pre-flight, gantry GATE prompt, trap -> SIGINT -> SHUTDOWN COMPLETE.
#
# Usage:
#   ./launch_merge.sh                  # stage 1 (policy active) — gantry default
#   ./launch_merge.sh --stage 0        # Stage 0 ownership probe: hold-only, NO policy
#   ./launch_merge.sh -- --no-rc       # extra args pass through to the writer
#
# Deploy map: /home/unitree/leon/merged_writer/ + policy/*.onnx (scp), this
# script at branch root; python = /home/unitree/miniconda3/envs/tv/bin/python3.

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

set -u
BRANCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$BRANCH_DIR/merged_writer"
LOCKFILE="${TMPDIR:-/tmp}/merged_writer.lock"
# Stuart deploy default: OpenBot (Leon's unified env), then tv, then PATH python
PYTHON="${MERGE_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for _p in /home/unitree/miniconda3/envs/OpenBot/bin/python3 \
            /home/unitree/miniconda3/envs/tv/bin/python3; do
    if [[ -x "$_p" ]]; then PYTHON="$_p"; break; fi
  done
fi
PYTHON="${PYTHON:-$(command -v python3 || echo python3)}"

STAGE="1"
WRITER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --) shift; WRITER_ARGS+=("$@"); break ;;
    *) WRITER_ARGS+=("$1"); shift ;;
  esac
done

echo "=== C3 merged writer launcher ==="
echo "branch: $BRANCH_DIR   stage: $STAGE   python: $PYTHON"

# --- duplicate writer guards ---
if [[ -e "$LOCKFILE" ]]; then
  LOCK_PID="$(cat "$LOCKFILE" 2>/dev/null || echo unknown)"
  if kill -0 "$LOCK_PID" 2>/dev/null; then
    echo "ABORT: writer already running (pid $LOCK_PID, lockfile $LOCKFILE)"
    exit 1
  fi
  echo "stale lockfile from dead pid $LOCK_PID — removing"
  rm -f "$LOCKFILE"
fi
PGRES="$(pgrep -f "main_merged_writer.py" || true)"
if [[ -n "$PGRES" ]]; then
  echo "ABORT: main_merged_writer.py already running: $PGRES"
  exit 1
fi

# --- fsm pre-flight (fsm_mode must be 0 / no sport FSM) ---
# candidate list mirrors launch_pick_walk.sh (deployed layout on Stuart)
FSM_CHECK=""
for _c in "$BRANCH_DIR/helpers/fsm_check.py" \
          "/home/unitree/leon/helpers/fsm_check.py" \
          "/home/unitree/leon/fsm_check.py" \
          "/tmp/fsm_check.py"; do
  if [[ -f "$_c" ]]; then FSM_CHECK="$_c"; break; fi
done
if [[ -n "$FSM_CHECK" ]]; then
  "$PYTHON" "$FSM_CHECK" || {
    echo "ABORT: fsm pre-flight failed (fsm_mode != 0 or unreachable) —"
    echo "       run the MODE_RECOVERY.md ladder BEFORE starting the writer"
    exit 1
  }
else
  echo "WARN: fsm_check.py not found — skipping fsm pre-flight (run it manually!)"
fi

# --- onnxruntime pre-flight (Stuart tv env must have it) ---
if ! "$PYTHON" -c "import onnxruntime" 2>/dev/null; then
  echo "ABORT: onnxruntime not importable with $PYTHON"
  echo "  install: /home/unitree/miniconda3/envs/tv/bin/pip install onnxruntime"
  exit 1
fi

# --- py_compile gate ---
"$PYTHON" -m py_compile "$PKG_DIR"/*.py || { echo "ABORT: py_compile failed"; exit 1; }

# --- gantry GATE ---
if [[ "${MERGE_SKIP_GATE:-0}" != "1" ]]; then
  echo ""
  echo "GATE: gantry ON? spotter ready? physical E-stop armed?"
  echo "      robot will take rt/lowcmd ownership on Enter."
  read -r -p "Press Enter to launch (Ctrl-C to abort)... " || exit 1
fi

cd "$PKG_DIR"
# setsid: writer leaves the terminal's process group, so a Ctrl-C reaches only
# the launcher, whose trap forwards ONE SIGINT. Without this, the terminal
# SIGINT + the trap's kill = double signal that killed the writer mid-shutdown
# (rc=130) before its zero-torque shutdown tail (2026-08-23 Stage 0).
trap 'echo "[launcher] signal — sending SIGINT to writer"; kill -INT "$W_PID" 2>/dev/null' INT TERM

# D1 (2026-08-25, C2 forensics): tee writer stdout/stderr — the C2 console was
# a live pts and is UNOBTAINABLE; the 1 s status line (poll_to/overruns/drops)
# + [Reader] take sample error + [fatal]/[signal] are the highest-value evidence.
# Pin the writer to 2 dedicated cores + negative nice (the shutdown tail
# inherits affinity + priority so it can preempt the perception stack).
WRITER_LOG="${TMPDIR:-/tmp}/pick_walk_logs/writer.log"
mkdir -p "$(dirname "$WRITER_LOG")"
env -u PYTHONPATH -u LD_LIBRARY_PATH -u ROS_DISTRO \
  taskset -c 2,3 nice -n -5 ionice -c2 -n0 \
  setsid "$PYTHON" main_merged_writer.py --stage "$STAGE" "${WRITER_ARGS[@]}" \
  >"$WRITER_LOG" 2>&1 &
W_PID=$!
echo "[launcher] writer pid $W_PID — log: $WRITER_LOG — waiting for SHUTDOWN COMPLETE"
wait "$W_PID"
RC=$?
echo "[launcher] writer exited rc=$RC (tail: $WRITER_LOG)"
exit $RC
