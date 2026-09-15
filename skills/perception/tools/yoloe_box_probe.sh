#!/usr/bin/env bash
# yoloe_box_probe.sh -- run the READ-ONLY S1 probe under every interpreter the box has.
#
# Batch 4, docs/YOLOE_BATCH4_ARCHITECTURE.md section 11.  OPERATOR-RUN ON THE BOX.
# Publishes nothing, moves nothing; writes only under $OUT_ROOT.  Exit 0 always.
#
#   bash yoloe_box_probe.sh [probe args...]
#   e.g. bash yoloe_box_probe.sh --assets-dir /home/unitree/leon/yoloe --bench --frame f.png
#        bash yoloe_box_probe.sh --record 30            (camera must be running)
#        bash yoloe_box_probe.sh --agree <out>/jp_env/frames --baseline-pt yolo11n-seg.pt
#
# Interpreters (each prints ONE JSON line, prefixed with its tag):
#   jp_env     today's yolo node interpreter, PYTHONPATH exactly as
#              launch_pick_stack.sh:301 sets it
#   python3    bare python3 with the caller's environment (launch_pick_walk.sh:262 form)
#   yoloe_env  the batch-4 venv over jp_env, only if it exists, same PYTHONPATH as jp_env
#
# Env knobs: OUT_ROOT (default ./yoloe_probe_<epoch>), LEON_DIR (default /home/unitree/leon/openbot).

PROBE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$PROBE_DIR/yoloe_box_probe.py"
OUT_ROOT="${OUT_ROOT:-$PWD/yoloe_probe_$(date +%s)}"
LEON_DIR="${LEON_DIR:-/home/unitree/leon/openbot}"
export LEON_DIR
FOXY_PP="/opt/ros/foxy/lib/python3.8/site-packages:/usr/lib/python3.8/dist-packages"
JP_PY="/home/unitree/leon/jp_env/bin/python3"
YOLOE_PY="/home/unitree/leon/yoloe_env/bin/python3"

mkdir -p "$OUT_ROOT" || { echo "[probe.sh] cannot create $OUT_ROOT"; exit 0; }
echo "[probe.sh] out root: $OUT_ROOT" >&2

run_one() {  # tag pythonpath-mode interpreter
  local tag="$1" ppmode="$2" py="$3"; shift 3
  if [ ! -x "$py" ] && ! command -v "$py" >/dev/null 2>&1; then
    echo "$tag {\"interpreter\": {\"tag\": \"$tag\", \"executable\": \"$py\"}, \"errors\": [\"interpreter missing\"]}"
    return 0
  fi
  local line
  if [ "$ppmode" = "foxy" ]; then
    line="$(PYTHONPATH="$FOXY_PP" "$py" "$PROBE" --tag "$tag" --out-dir "$OUT_ROOT/$tag" "$@" 2>"$OUT_ROOT/$tag.stderr.log")"
  else
    line="$("$py" "$PROBE" --tag "$tag" --out-dir "$OUT_ROOT/$tag" "$@" 2>"$OUT_ROOT/$tag.stderr.log")"
  fi
  echo "$tag $line"
  return 0
}

mkdir -p "$OUT_ROOT/jp_env" "$OUT_ROOT/python3" 2>/dev/null
run_one jp_env foxy "$JP_PY" "$@"
run_one python3 caller python3 "$@"
if [ -x "$YOLOE_PY" ]; then
  mkdir -p "$OUT_ROOT/yoloe_env" 2>/dev/null
  run_one yoloe_env foxy "$YOLOE_PY" "$@"
else
  echo "yoloe_env {\"interpreter\": {\"tag\": \"yoloe_env\", \"executable\": \"$YOLOE_PY\"}, \"errors\": [\"not present\"]}"
fi
exit 0
