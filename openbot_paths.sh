#!/usr/bin/env bash
# openbot_paths.sh: bash twin of openbot_paths.py. `source` it from a launcher.
# Precedence: explicit OPENBOT_* env var > new location (Unitree layout of 2026-10-09) > old location (kept as a symlink).
# Do not spell the live-tree path as a literal here (merge_to_live.py refuses it): ROOT comes from BASH_SOURCE.
OPENBOT_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_ob_pick() {  # $1 = env value, $2 = new location, $3 = old location
  if [ -n "$1" ]; then printf '%s' "$1"; elif [ -e "$2" ]; then printf '%s' "$2"; else printf '%s' "$3"; fi
}
_OB_UNITREE="$HOME/Unitree"
_OB_WORKSPACE="$_OB_UNITREE/OpenBot_workspace"   # was ~/Unitree/.wf
_OB_DATA="$_OB_UNITREE/OpenBot_data"
_OB_THIRD="$_OB_UNITREE/third_party"
OPENBOT_WF="$(_ob_pick "${OPENBOT_WF:-}" "$_OB_WORKSPACE" "$_OB_UNITREE/.wf")"
OPENBOT_STUART_RUN="$(_ob_pick "${OPENBOT_STUART_RUN:-}" "$_OB_DATA/robot_logs/stuart_run" "$_OB_UNITREE/stuart_run")"
OPENBOT_PLANNER_MODELS="$(_ob_pick "${OPENBOT_PLANNER_MODELS:-}" "$_OB_DATA/models/planner_models" "$_OB_UNITREE/planner_models")"
OPENBOT_SIM_EVIDENCE="${OPENBOT_SIM_EVIDENCE:-$HOME/sonic_kneel_video}"
OPENBOT_BM_GUARD="$(_ob_pick "${OPENBOT_BM_GUARD:-}" "$_OB_WORKSPACE/tools/bm_guard.sh" "$_OB_UNITREE/.wf/tools/bm_guard.sh")"
OPENBOT_BM_GUARD_SONIC="${OPENBOT_BM_GUARD_SONIC:-$HOME/sonic_kneel_video/tools/bm_guard.sh}"
OPENBOT_LEGACY="$_OB_UNITREE/OpenBot_archive"
# external dependencies (moved to third_party/ and research/; the old spellings are compatibility symlinks)
OPENBOT_DEPS="$(_ob_pick "${OPENBOT_DEPS:-}" "$_OB_THIRD/Dependencies_Libraries" "$_OB_UNITREE/Dependencies_Libraries")"
OPENBOT_GRAIL_SONIC="$(_ob_pick "${OPENBOT_GRAIL_SONIC:-}" "$_OB_THIRD/GRAIL/imports/SONIC/gear_sonic_deploy/src/g1" "$_OB_UNITREE/GRAIL/imports/SONIC/gear_sonic_deploy/src/g1")"
OPENBOT_LEROBOT="$(_ob_pick "${OPENBOT_LEROBOT:-}" "$_OB_UNITREE/research/video2contactdynamic/Lerobot_Framework" "$_OB_UNITREE/video2contactdynamic/Lerobot_Framework")"
export OPENBOT_ROOT OPENBOT_WF OPENBOT_STUART_RUN OPENBOT_PLANNER_MODELS OPENBOT_SIM_EVIDENCE OPENBOT_BM_GUARD \
       OPENBOT_BM_GUARD_SONIC OPENBOT_LEGACY OPENBOT_DEPS OPENBOT_GRAIL_SONIC OPENBOT_LEROBOT
