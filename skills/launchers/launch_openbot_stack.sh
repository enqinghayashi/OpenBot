#!/usr/bin/env bash
# launch_openbot_stack.sh — collapsed single-terminal OpenBot stack entry:
# route pin -> gantry gate -> merged writer -> v2b launcher chain (the pick
# engine ALWAYS spawns with the chain; --no-daemon is removed from this entry).
#
# Collapses the terminal-1 (writer) and terminal-2 (v2b chain) roles of the
# launch chain into ONE console.  Terminal-3 (gateway) and terminal-4 (voice)
# REMAIN separate live terminals: this script prints their exact commands when
# the v2b chain exits — it must not and does not own their stdin (the gateway's
# query_user reads stdin; the voice session is interactive).
#
# Flow:
#   1. route pin   : openbot_route_pin (defined below) — the FIRST action of
#                    the real flow: pin multicast 224.0.0.0/4 dev eth0 (DDS
#                    SPDP discovery + the voice mic group stay on the robot
#                    ethernet), then drop the DEAD 'default via
#                    192.168.123.1 dev eth0' (#105-class egress shadow; the
#                    wlan0 default is never touched — the del names gateway +
#                    dev explicitly; `|| true` because the 404 is the
#                    idempotent re-run case).
#   2. gantry gate : THIS script reads the physical GATE confirmation, then
#                    launch_merge.sh runs with MERGE_SKIP_GATE=1 (its own
#                    prompt is replaced by this one).
#   3. writer      : launch_merge.sh backgrounded under a SPAWN-ONLY `set -m`
#                    (job control, immediately undone with `set +m`), stdin
#                    /dev/null — the writer's in-process gate auto-arms on
#                    non-tty stdin (main_merged_writer.py:207 prints '[gate]
#                    stdin not a tty (launcher-gated) — arming immediately').
#                    WHY -m at the spawn and NOT setsid (measured 2026-09-05,
#                    e2e harness): bash SIG_IGNs SIGINT+SIGQUIT in async
#                    children of non-interactive shells, so launch_merge.sh
#                    would start with SIGINT ignored and its INT trap (the
#                    graceful zero-torque limp relay, stage 7) would silently
#                    never install — stage-7 Ctrl+C would then leave the
#                    writer holding the robot.  -m gives the writer job its
#                    own process group and keeps the launcher's SIGINT at
#                    default (disposition fixed at exec).  -m is then turned
#                    OFF: under permanent job control a trapped SIGINT during
#                    the stage-7 `wait` is deferred until the waited job
#                    exits — the forward would deadlock.  A terminal Ctrl+C
#                    during the v2b phase reaches only the FOREGROUND group
#                    (this script + the v2b tree), never the writer's group.
#                    (setsid itself is unusable: under -m the async child is
#                    already a PG leader, so setsid forks and $! would track
#                    the dead wrapper, not the launcher.)
#   4. writer poll : bounded 90 s poll of /tmp/pick_walk_logs/writer.log for
#                    '[sock] listening on /tmp/merged_writer.sock'.  The
#                    socket binds BEFORE the 250 Hz loop, so socket-exists ==
#                    armed-and-holding (main_merged_writer.py:277).  The log
#                    is truncated at the stage-3 spawn, so a SECOND+ launch
#                    within a boot cannot green on the PREVIOUS run's socket
#                    line.  Loud fail with log tail if never seen or the
#                    launcher dies.
#   5. v2b chain   : launch_merge_pick_v2b.sh FOREGROUND, "$@" passed through
#                    (no --no-daemon — the pick engine always spawns WITH
#                    the chain).  Its own traps own its tree;
#                    during this phase this script's INT trap is a NO-OP, so a
#                    terminal Ctrl+C kills only the v2b chain — exactly
#                    today's terminal-2 semantics.
#   6. next-steps  : when the chain exits, print the exact terminal-3 gateway
#                    and terminal-4 voice commands.
#   7. shutdown    : `wait` on the launch_merge.sh job, forwarding SIGINT to
#                    it ONLY at that point — its launcher trap then sends the
#                    writer the single SIGINT that runs the graceful
#                    zero-torque limp.  Never kill the writer mid-session
#                    off-gantry.
#
# NOTE on job control: this script runs `set -m` ONLY around the writer
# spawn, so the backgrounded launcher starts with SIGINT trappable while the
# final `wait` stays interruptible (see flow point 3).  All other signal
# semantics — no-op INT during the v2b phase, INT forwarded ONLY at the final
# wait — are unchanged.
#
# Usage:
#   bash launch_openbot_stack.sh [--dry-run]
#     --dry-run     print the stage plan and exit (NO pin / gate / launch)
#     everything else passes through to launch_merge_pick_v2b.sh
#     --no-daemon   REJECTED (removed 2026-09-06): the pick engine ALWAYS
#                   spawns with the v2b chain
#
# v2b chain prerequisite (loud-fails before anything starts if unmet): the
# OpenBot env — exec bash --norc --noprofile; source /home/unitree/leon/setup_openbot.sh
#
# Deploy map: /home/unitree/leon/ — a sibling of launch_merge.sh and
# launch_merge_pick_v2b.sh.  NOT in the manifest push set; ships per-file per
# DEPLOY_PROTOCOL (md5 + null-byte checks; no py_compile for shell scripts).

set -u
# Job control is enabled ONLY around the writer spawn below (set -m / set
# +m), never permanently — see the header note, flow point 3:
#   - WHY -m at the spawn: without it, bash SIG_IGNs SIGINT+SIGQUIT in every
#     async child of this non-interactive script — the backgrounded
#     launch_merge.sh would start with SIGINT ignored and its INT trap (the
#     graceful zero-torque limp relay in stage 7) would never install.  -m puts
#     the writer job in its own process group and keeps SIGINT at default
#     (its disposition is fixed at exec, so the parent may switch back).
#   - WHY set +m right after: under job control a trapped SIGINT received
#     while `wait`-ing on a background job is DEFERRED until that job
#     completes (measured 2026-09-05) — stage 7 would deadlock (the forward
#     only fires after the launcher exits, and the launcher only exits after
#     the forward).  With -m off, the final `wait` is interruptible again.
#   - setsid is NOT used: under -m the async child is already a PG leader,
#     so setsid would fork and $! would track the dead wrapper, not the
#     launcher.

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
# NOTE (W2, this script only): the pin CALL is stage 1 of the real flow below
# (right after the --dry-run short-circuit) — a dry run must not mutate routes
# or prompt sudo; the function body above is byte-identical to the three
# launcher copies.
openbot_route_pin() {
  sudo ip route del default via 192.168.123.1 dev eth0 2>/dev/null || true
  if ! sudo ip route add 224.0.0.0/4 dev eth0 2>/dev/null; then
    ip route show 224.0.0.0/4 dev eth0 >/dev/null 2>&1 \
      || echo "openbot_route_pin: WARNING: cannot pin 224.0.0.0/4 dev eth0 (route absent; add needs root — check sudo)" >&2
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRITER_LAUNCHER="$SCRIPT_DIR/launch_merge.sh"
V2B_LAUNCHER="$SCRIPT_DIR/launch_merge_pick_v2b.sh"
WRITER_LOG="${TMPDIR:-/tmp}/pick_walk_logs/writer.log"
SOCKET_LINE='[sock] listening on /tmp/merged_writer.sock'
WRITER_POLL_S=90

stage() { echo "[stack] $*"; }

# -- parse: --dry-run is ours; everything else goes to the v2b launcher ------
DRY_RUN=0
V2B_ARGS=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    --no-daemon)
      echo "[stack] FATAL: --no-daemon is REMOVED — the pick engine always spawns with the v2b chain." >&2
      exit 1 ;;
    *) V2B_ARGS+=("$a") ;;
  esac
done

if [ "$DRY_RUN" -eq 1 ]; then
  cat <<'EOF'
[stack --dry-run] stage plan:
  1. route pin  : openbot_route_pin — pin 224.0.0.0/4 dev eth0 (DDS SPDP
                  discovery + the voice mic group stay on the robot ethernet),
                  then drop the DEAD 'default via 192.168.123.1 dev eth0'
                  (#105-class egress shadow).  The working wlan0 default is
                  never touched.  Idempotent: a second launch re-runs
                  silently; any failure to pin warns loudly.
  2. gantry gate: operator physical GATE confirmation (gantry ON? spotter
                  ready? E-stop armed?) — replaces launch_merge.sh's prompt
  3. writer     : MERGE_SKIP_GATE=1 bash launch_merge.sh & under a SPAWN-
                  ONLY `set -m` / `set +m` (job control at the fork: the
                  writer job owns its process group and SIGINT stays
                  trappable for launch_merge.sh's graceful zero-torque relay; -m
                  off afterwards so the stage-7 `wait` stays interruptible —
                  see the header note; stdin /dev/null: the writer's in-
                  process gate auto-arms on non-tty stdin,
                  main_merged_writer.py:207)
  4. writer poll: bounded 90 s poll of /tmp/pick_walk_logs/writer.log for
                  '[sock] listening on /tmp/merged_writer.sock' — the log is
                  truncated at the stage-3 spawn, so a SECOND+ launch within
                  a boot cannot green on the previous run's socket line;
                  loud fail with log tail if never seen (socket binds BEFORE
                  the 250 Hz loop: socket-exists == armed-and-holding)
  5. v2b chain  : bash launch_merge_pick_v2b.sh <args> FOREGROUND
                  T0 grip_server -> T1 openbot_camera_node -> T2 yolo_node ->
                  T3 depth_projector_node -> T4 pick_node (spawns the
                  NON-OWNING pick engine WITH the chain — --no-daemon
                  removed from this entry) -> T5 walk_node + T6 coordinator
                  (idle, NEVER triggered — a /walk_target would spawn a
                  SECOND rt/lowcmd writer = ABSOLUTE TRIP) -> T7 arrival_relay
                  (async spoken arrival, track 214-d: subscribes
                  /navigate_status, maps REACHED lines through the PURE
                  arrival_announce law to the /tmp/arrival_announce.sock that
                  the terminal-4 voice process reads; best-effort, deploy-gated
                  — ship skills/coordination/arrival_{relay,announce}.py to
                  $LEON_DIR and set voice.arrival_announce to arm it)
  6. next-steps : after the chain exits, print the exact terminal-3 gateway
                  and terminal-4 voice commands — those MUST run in their own
                  live consoles (gateway query_user reads stdin; voice is the
                  interactive session; this script never owns their stdin)
  7. shutdown   : wait on the launch_merge.sh job; Ctrl+C forwards SIGINT to
                  it ONLY at that point -> its launcher trap -> the writer's
                  graceful zero-torque limp
[stack --dry-run] no changes made.
EOF
  exit 0
fi

# stage 1 — route pin: the FIRST action of the flow (see note by the def).
stage "route pin: openbot_route_pin (224.0.0.0/4 -> eth0; del dead eth0 default)"
openbot_route_pin

# -- v2b preflight (fail BEFORE anything starts; the chain needs ros2) -------
if ! command -v ros2 >/dev/null 2>&1; then
  echo "[stack] FATAL: 'ros2' not on PATH — the v2b chain needs the OpenBot ROS env." >&2
  echo "[stack]   Prep this shell first (v2b terminal prep):" >&2
  echo "[stack]     exec bash --norc --noprofile" >&2
  echo "[stack]     source /home/unitree/leon/setup_openbot.sh" >&2
  exit 1
fi

# stage 2 — gantry GATE (physical confirmation; replaces launch_merge.sh's
# prompt — launch_merge.sh below therefore runs with MERGE_SKIP_GATE=1).
echo ""
echo "GATE: gantry ON? spotter ready? physical E-stop armed?"
echo "      the robot will take rt/lowcmd ownership (merged writer) on Enter."
read -r -p "Press Enter to launch (Ctrl-C to abort)... " || exit 1

# stage 3 — merged writer, backgrounded under a SPAWN-ONLY `set -m` (job
# control): the writer job gets its own process group, so a terminal Ctrl+C
# during the v2b phase can never reach it — and launch_merge.sh starts with
# SIGINT trappable so its own INT trap -> the writer stop relay works in
# stage 7.  -m is turned OFF immediately after the fork (the launcher's
# dispositions are fixed at exec): the stage-7 `wait` must stay
# interruptible, and under permanent job control a trapped INT during `wait`
# is deferred until the waited job exits — the stage-7 forward would
# deadlock (see the set -u / set -m note above).  stdin /dev/null: the
# writer's in-process gate auto-arms on non-tty stdin.
stage "writer: MERGE_SKIP_GATE=1 bash $(basename "$WRITER_LAUNCHER") & (spawn-only -m PG; stdin /dev/null)"
# The stage-4 poll greps this log for the writer's socket line.  On a
# SECOND+ launch within a boot the log still holds the PREVIOUS run's line,
# and launch_merge.sh only truncates it when ITS writer subprocess spawns
# (after its guards / preflights) — the poll would false-green at ~0.5 s
# against the stale line.  Truncate NOW: only THIS run's lines can green
# the poll, and a launcher death pre-spawn still trips the poll's loud
# kill -0 FATAL with the log tail.
mkdir -p "$(dirname "$WRITER_LOG")"
: > "$WRITER_LOG"
set -m
MERGE_SKIP_GATE=1 bash "$WRITER_LAUNCHER" </dev/null &
MERGE_PID=$!
set +m

# From here until the final wait, this script's INT trap is a NO-OP: a
# terminal Ctrl+C must kill only the v2b tree (its own traps own its
# children), never this script (it still has to print the next-step commands
# and wait for the writer) and never the writer (separate setsid session).
trap ':' INT

# stage 4 — bounded poll for the writer socket (binds BEFORE the 250 Hz loop:
# socket-exists == armed and holding, main_merged_writer.py:277).
stage "polling $WRITER_LOG for the writer socket (up to ${WRITER_POLL_S}s)..."
WRITER_READY=0
for _i in $(seq 1 $((WRITER_POLL_S * 2))); do
  if ! kill -0 "$MERGE_PID" 2>/dev/null; then
    echo "[stack] FATAL: launch_merge.sh exited during the poll — the writer never came up." >&2
    echo "[stack] log tail ($WRITER_LOG):" >&2
    tail -n 30 "$WRITER_LOG" 2>/dev/null | sed 's/^/[writer.log] /' >&2 || true
    exit 1
  fi
  if grep -qF "$SOCKET_LINE" "$WRITER_LOG" 2>/dev/null; then
    WRITER_READY=1
    break
  fi
  sleep 0.5
done
if [ "$WRITER_READY" -eq 0 ]; then
  echo "[stack] FATAL: '$SOCKET_LINE' not seen in $WRITER_LOG within ${WRITER_POLL_S}s — writer not armed/holding." >&2
  echo "[stack] log tail:" >&2
  tail -n 30 "$WRITER_LOG" 2>/dev/null | sed 's/^/[writer.log] /' >&2 || true
  exit 1
fi
stage "writer socket live — writer armed and holding rt/lowcmd"

# stage 5 — v2b chain FOREGROUND.  Its EXIT/INT/TERM/HUP traps own its tree;
# this script just waits (no-op INT) and resumes when the chain exits
# (normally via the operator's Ctrl+C = rc 130).
V2B_DESC="${V2B_ARGS[*]:-}"
stage "v2b chain: bash $(basename "$V2B_LAUNCHER")${V2B_DESC:+ }$V2B_DESC (foreground)"
bash "$V2B_LAUNCHER" "${V2B_ARGS[@]}"
V2B_RC=$?
stage "v2b chain exited rc=$V2B_RC — the merged writer is still holding the robot"

# stage 6 — next-step commands for terminals 3 + 4 (console invariants: the
# gateway's query_user reads stdin and the voice session is interactive —
# BOTH must run in their own live terminals; this script never owns them).
cat <<'EOF'

=== stack: v2b chain done; the merged writer is STILL holding the robot ===
Open TWO more terminals on the box (deploy root /home/unitree/leon) and run:

  [Terminal 3 — gateway (OpenBot brain lane; needs the writer socket + grip
   socket + engine present; must follow the writer + v2b chain)]
  exec bash --norc --noprofile
  source /home/unitree/leon/setup_openbot.sh && cd /home/unitree/leon
  OPENBOT_PROFILE=offline python3 -m gateway.gateway_node --config config.yaml

  [Terminal 4 — voice app (sole tools client of the gateway)]
  bash voice/run_voice.sh
  # keyless offline lane instead:
  #   bash voice/run_voice.sh --config config.offline.yaml
  # (the cloud lane needs the key file /home/unitree/openbot/openai_key,
  #  chmod 600)

Ready-when:
  gateway: 'openbot_gateway ready — tools socket /tmp/openbot_gateway.sock
            (tcp port 3900)'
  voice:   app boots with its gateway boot probe green, KWS armed, mic
           frames arriving from the 239.168.123.161:5555 join, G1Speaker
           volume-100 boot gate

To end the writer session, Ctrl+C THIS terminal: SIGINT is forwarded to
launch_merge.sh ONLY at that point, whose launcher trap then sends the writer
the single SIGINT that runs the graceful zero-torque limp.
EOF

# stage 7 — final wait on the writer's launcher.  SIGINT is forwarded to
# launch_merge.sh ONLY here — never while the v2b chain owns the console and
# never directly to the writer (it is in its own job-controlled process
# group; killing it off-gantry would bypass the graceful zero-torque limp).
# ARMED-ONCE relay: the shutdown takes seconds (launcher trap -> writer
# zero-torque limp), and a SECOND Ctrl+C inside that window would
# double-signal the writer (rc=130 forensics on record).  Only the
# FIRST Ctrl+C forwards; later ones are absorbed.
_INT_FORWARDED=0
_forward_int() {
  if [ "$_INT_FORWARDED" -eq 1 ]; then
    echo "[stack] Ctrl+C (absorbed — SIGINT already forwarded once)"
    return 0
  fi
  _INT_FORWARDED=1
  echo "[stack] Ctrl+C — forwarding SIGINT to launch_merge.sh (graceful zero-torque limp)"
  kill -INT "$MERGE_PID" 2>/dev/null || true
}
trap _forward_int INT
stage "waiting on launch_merge.sh (Ctrl+C here = graceful zero-torque limp)"
while :; do
  wait "$MERGE_PID"
  RC=$?
  # A trapped SIGINT interrupts `wait` (rc > 128) while launch_merge.sh is
  # still completing the shutdown — re-wait until it actually exits.
  if [ "$RC" -gt 128 ] && kill -0 "$MERGE_PID" 2>/dev/null; then
    continue
  fi
  break
done
stage "launch_merge.sh exited rc=$RC — stack session complete"
exit "$RC"
