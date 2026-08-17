#!/usr/bin/env bash
# odom_ctl.sh -- operator control for the OpenBot odometry module.
#
# Verbs: preflight | start | stop | status | gate-start | gate-stop | capture <seconds>
# Never touches the robot, never edits a launcher, never starts the walk lane.
# Dry mode: ODOM_CTL_DRY=1 echoes launch/kill commands instead of running them
# (read-only ros2 probes still run, so a stub ros2 on PATH can drive a test).
set -u

TOPIC="/unitree/slam_mapping/odom"
ODOM_DIR="${ODOM_DIR:-/home/unitree/leon/odom}"
LEON_DIR="${LEON_DIR:-/home/unitree/leon/openbot}"
LOG_DIR="${LOG_DIR:-/tmp/pick_walk_logs}"
ODOM_PID="${ODOM_PID:-/tmp/odom_node.pid}"
GATE_PID="${GATE_PID:-/tmp/odom_gate.pid}"
DRY="${ODOM_CTL_DRY:-0}"
VERDICT=""

say() { echo "$*"; }

usage() {
  say "usage: odom_ctl.sh {preflight|start|stop|status|gate-start|gate-stop|capture <seconds>}"
}

source_env() {
  # the OpenBot env (the setup scripts use unset variables, so relax -u around them)
  set +u
  # shellcheck disable=SC1090
  source "${OPENBOT_ENV:-/home/unitree/leon/setup_openbot.sh}"
  set -u
}

preflight() {
  say "== preflight: is a vendor/other publisher on $TOPIC? (read-only) =="
  if [[ "${RMW_IMPLEMENTATION:-}" != rmw_cyclonedds_cpp || -z "${CYCLONEDDS_URI:-}" ]]; then
    say "WARNING: the OpenBot env is not sourced (RMW_IMPLEMENTATION / CYCLONEDDS_URI unset). If only /opt/ros/foxy/setup.bash is sourced, an EMPTY topic list is the known trap, not a dead stack. Source /home/unitree/leon/setup_openbot.sh first."
  fi
  local nodes info n out rc
  if ! command -v ros2 >/dev/null 2>&1; then
    VERDICT="UNKNOWN"
    say "ros2 not on PATH: cannot probe"
    say "VENDOR ODOM: UNKNOWN"
    return 0
  fi
  nodes="$(ros2 node list 2>/dev/null | awk 'tolower($0) ~ /slam|odom/')"
  say "nodes matching slam/odom: ${nodes:-<none>}"
  info="$(ros2 topic info -v "$TOPIC" 2>&1)"
  n="$(printf '%s\n' "$info" | sed -n 's/^Publisher count: *\([0-9][0-9]*\).*/\1/p' | head -n 1)"
  n="${n:-0}"
  say "publisher count on $TOPIC: $n"
  out="$(timeout 5 ros2 topic echo --once "$TOPIC" 2>/dev/null)"
  rc=$?
  if [ "$rc" -eq 0 ] && [ -n "$out" ]; then
    VERDICT="PUBLISHING"
  elif [ "$n" -gt 0 ]; then
    VERDICT="REGISTERED-BUT-SILENT"
  else
    VERDICT="ABSENT"
  fi
  say "VENDOR ODOM: $VERDICT"
}

launch() {  # launch <name> <script> <logfile> <pidfile> [env-prefix]
  local name="$1" script="$2" logf="$3" pidf="$4" envp="${5:-}"
  local cmd="${envp}nohup python3 $script > $logf 2>&1 &"
  if [ "$DRY" = "1" ]; then
    say "DRY: $cmd"
    return 0
  fi
  mkdir -p "$LOG_DIR"
  source_env
  if [ -n "$envp" ]; then
    PYTHONPATH="$LEON_DIR${PYTHONPATH:+:$PYTHONPATH}" nohup python3 "$script" > "$logf" 2>&1 &
  else
    nohup python3 "$script" > "$logf" 2>&1 &
  fi
  echo $! > "$pidf"
  say "$name started pid $(cat "$pidf") log $logf"
}

script_of() {  # script_of <name>: the script a pidfile of <name> must point at
  case "$1" in
    odom_node) echo "odom_node.py" ;;
    detection_gate) echo "detection_gate_node.py" ;;
    *) echo "$1" ;;
  esac
}

is_ours() {  # is_ours <pid> <script>: the live pid really runs <script> (B8k:
  # the stack launcher writes the pidfile too; after a crash or power loss the
  # pid may belong to an unrelated process - never signal that one)
  local cmd
  cmd="$(tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null)" || return 1
  [[ "$cmd" == *"$2"* ]]
}

stop_proc() {  # stop_proc <name> <pidfile>
  local name="$1" pidf="$2" pid i
  if [ ! -f "$pidf" ]; then
    say "$name: no pidfile $pidf (not started by this script)"
    return 0
  fi
  pid="$(cat "$pidf")"
  if [ "$DRY" = "1" ]; then
    say "DRY: kill -INT $pid"
    return 0
  fi
  if kill -0 "$pid" 2>/dev/null && ! is_ours "$pid" "$(script_of "$name")"; then
    say "$name: pid $pid is not $(script_of "$name") (stale pidfile) - removed, nothing signalled"
    rm -f "$pidf"
    return 0
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT "$pid"
    for i in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
  fi
  if kill -0 "$pid" 2>/dev/null; then
    say "$name: pid $pid STILL ALIVE after SIGINT"
    return 1
  fi
  rm -f "$pidf"
  say "$name: stopped (pid $pid gone)"
}

alive_line() {  # alive_line <name> <pidfile> <logfile>
  local name="$1" pidf="$2" logf="$3" pid
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null \
      && is_ours "$(cat "$pidf")" "$(script_of "$name")"; then
    pid="$(cat "$pidf")"
    say "$name: ALIVE pid $pid"
  else
    say "$name: not running"
  fi
  if [ -f "$logf" ]; then
    say "  last log line: $(tail -n 1 "$logf")"
  fi
}

case "${1:-}" in
  preflight)
    preflight
    ;;
  start)
    if [ "$DRY" != "1" ]; then
      # source the OpenBot env BEFORE the probe: a foxy-only shell reads an
      # empty topic list as ABSENT and would fail the guard open
      source_env
      if ! command -v ros2 >/dev/null 2>&1; then
        say "REFUSE: ros2 not found after sourcing the OpenBot env -- verdict UNKNOWN, not starting"
        exit 2
      fi
    fi
    preflight
    if [ "$VERDICT" != "ABSENT" ]; then
      say "REFUSE: vendor odom verdict is $VERDICT, not ABSENT -- not starting a second publisher on $TOPIC"
      exit 2
    fi
    launch odom_node "$ODOM_DIR/odom_node.py" "$LOG_DIR/odom.log" "$ODOM_PID"
    ;;
  stop)
    stop_proc odom_node "$ODOM_PID"
    ;;
  status)
    alive_line odom_node "$ODOM_PID" "$LOG_DIR/odom.log"
    alive_line detection_gate "$GATE_PID" "$LOG_DIR/odom_gate.log"
    sample="$(timeout 5 ros2 topic echo --once "$TOPIC" 2>/dev/null | head -n 12)"
    if [ -n "$sample" ]; then
      say "echo --once: RETURNED (header follows)"
      say "$sample"
    else
      say "echo --once: nothing within 5 s"
    fi
    ;;
  gate-start)
    launch detection_gate "$ODOM_DIR/detection_gate_node.py" "$LOG_DIR/odom_gate.log" "$GATE_PID" "PYTHONPATH=$LEON_DIR "
    ;;
  gate-stop)
    stop_proc detection_gate "$GATE_PID"
    ;;
  capture)
    secs="${2:-}"
    case "$secs" in
      ''|*[!0-9]*) say "usage: odom_ctl.sh capture <seconds 1..600>"; exit 2 ;;
    esac
    out="/tmp/odom_capture_$(date +%s).csv"
    if [ "$DRY" = "1" ]; then
      say "DRY: python3 $ODOM_DIR/capture_detections.py --seconds $secs --out $out"
    else
      source_env
      python3 "$ODOM_DIR/capture_detections.py" --seconds "$secs" --out "$out"
    fi
    ;;
  *)
    usage
    exit 2
    ;;
esac
