#!/usr/bin/env bash
# run_voice.sh — launch the OpenBot v4 voice app on the box (Orin NX,
# JetPack 5 / Ubuntu 20.04).  Run from the deploy root ON the box:
#   bash voice/run_voice.sh [voice/app.py args ...]
#
# Responsibilities (all secrets-local; the key NEVER lives in this repo):
#   1. cwd = the deploy root (the app resolves config.yaml, voice/models/*
#      and the piper voice against its cwd — the app's own documented
#      contract, voice/app.py: "run from the OpenBot repo root").
#   2. OpenAI key: read from a box-side chmod-600 file (default
#      /home/unitree/openbot/openai_key; override OPENBOT_KEY_FILE) and
#      exported as OPENAI_API_KEY into the app subprocess env only —
#      never echoed, never logged, never written to any file.
#      ONE-TIME SETUP on the box (the key is not in the repo):
#        mkdir -p /home/unitree/openbot
#        printf '%s\n' 'sk-...' > /home/unitree/openbot/openai_key
#        chmod 600 /home/unitree/openbot/openai_key
#        chown unitree:unitree /home/unitree/openbot/openai_key
#      (mode 400 is also accepted — the check is: owner-readable, and no
#      group/other bits.)
#   3. Preflight: a missing key file or missing voice models stops LOUDLY
#      (exit 2) before anything starts.  Models are fetched by
#      voice/fetch_models.sh.
#   4. The app runs under the box conda env (default 'voice' — the voice
#      phase conda env: RUNBOOK §13.3 creates env 'voice'; override with
#      OPENBOT_CONDA_ENV / OPENBOT_CONDA_ROOT) with
#      voice/requirements-box.txt installed.  ('tv' remains the
#      brain/gateway push interpreter per push_stuart.sh REMOTE_PY — a
#      separate env, not the voice app's.)
#   5. voice/models is prepended to PATH for the app subprocess: the
#      cascade resolves the whisper.cpp binary LAYOUT-FIRST at launch —
#      a bare bin_path ('whisper-server') is exec'd from the cwd-relative
#      voice/models/<name> when that file exists AND is executable, with
#      the bare-name PATH lookup only as the fallback (cascade.py
#      _resolve_bin; voice/cascade.py 'WIRE FACTS').  The prepend feeds
#      that fallback; cwd = the deploy root (see 1) makes the layout copy
#      win deterministically for both launch forms.
#   6. All CLI args are passed through to voice/app.py untouched; with no
#      --config the script defaults to config.voice.yaml (brain.lane
#      realtime).  The shipped lane profiles are config.voice.yaml
#      (brain.lane realtime) and config.offline.yaml (brain.lane offline);
#      the offline stage can be booted with --config config.offline.yaml.
#      The app's own config.yaml default is the laptop profile and fails
#      loudly by design.  -h/--help bypass the key/models preflight
#      (argparse answers with stdlib only).
#
# Env overrides: OPENBOT_KEY_FILE, OPENBOT_CONDA_ROOT, OPENBOT_CONDA_ENV,
#   OPENBOT_LOG_FILE (where to write the session log; '/dev/null' disables it).
#
# Session log (added 2026-09-26): every run is mirrored to
#   voice/logs/voice-YYYYmmdd-HHMMSS.log  (mode 600)
# with voice/logs/latest.log pointing at the newest one, so
#   tail -f voice/logs/latest.log
# always follows the current run.  The terminal output is unchanged (a tee).
# The newest 20 logs are kept.  It is REFUSED under `bash -x` — see the
# xtrace guard below.

set -euo pipefail

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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$REPO_ROOT/voice/app.py"
CONFIG_DEFAULT="config.voice.yaml"
MODEL_DIR="$REPO_ROOT/voice/models"

KEY_FILE="${OPENBOT_KEY_FILE:-/home/unitree/openbot/openai_key}"
CONDA_ROOT="${OPENBOT_CONDA_ROOT:-/home/unitree/miniconda3}"
CONDA_ENV="${OPENBOT_CONDA_ENV:-voice}"

die() { echo "run_voice.sh: REFUSAL: $*" >&2; exit 2; }

# argparse answers -h/--help before any config/key/models are needed.
HELP=0
for a in "$@"; do
  case "$a" in -h|--help) HELP=1 ;; esac
done

# --- output log (operator request 2026-09-26) -------------------------------
#   "bash voice/run_voice.sh ... add the output log file part for me such that
#    i can view the log"
#
# Everything below this line — the launcher's preflight AND the app's own
# output — is mirrored to a file, with the live terminal output UNCHANGED
# (a tee, not a redirect): the operator still watches the run, and can read it
# back afterwards.
#
#   default path : <repo>/voice/logs/voice-YYYYmmdd-HHMMSS.log
#   stable alias : <repo>/voice/logs/latest.log -> the newest log, so
#                  `tail -f voice/logs/latest.log` always follows the CURRENT run
#   override     : OPENBOT_LOG_FILE=/some/path.log
#   disable      : OPENBOT_LOG_FILE=/dev/null
#
# IT STARTS HERE, NOT AT THE APP, because the preflight refusals below —
# missing voice models, a missing or wrong-mode key file, an unresolvable
# lane — are exactly the failures worth reading back, and every one of them
# happens before the app process exists.
#
# WHY `exec > >(tee ...)` AND NOT `python3 ... | tee`: the difference that
# matters is the PID, not the status.  Measured 2026-09-26 — with the process
# substitution the launcher's OWN PID becomes the app (`ps -o comm= -p <pid>`
# reports the app), so `kill <pid>`, `timeout`, or a systemd unit that signals
# the launcher reaches the app.  With `exec ... | tee` the launcher's PID
# stays `bash` with the app and tee as its CHILDREN, so killing the launcher
# leaves the voice lane RUNNING as an orphan — on a robot, with the mic open.
# It is NOT the exit code: `set -o pipefail` at the top already makes a
# pipeline report the app's status (measured both ways — rc=7 with pipefail,
# rc=0 without).  pipefail is what protects the status; exec is what protects
# the PID.
# Honest residual: the app's status returns before tee has necessarily drained
# its last lines, so a caller that reads the log the instant the process exits
# can catch a truncated tail.  `tail -f`, and anyone reading it later, see the
# complete file.
#
# SAFETY: only stdout+stderr are captured.  The API key is EXPORTED into the
# app subprocess env and never echoed (see the key section), and nothing here
# prints the environment — so the key cannot reach this file through the
# stream.  The ONE way it could is the shell's own `-x` trace, which the guard
# below refuses outright rather than trusting anyone to remember.  Logging is
# strictly best-effort: if the file cannot be written the launcher warns and
# boots anyway.  A boot must never be blocked by its own log.
if [[ "$HELP" == "0" ]]; then
  LOG_FILE="${OPENBOT_LOG_FILE:-}"
  LOG_IS_DEFAULT=0
  if [[ -z "$LOG_FILE" ]]; then
    LOG_DIR="$REPO_ROOT/voice/logs"
    LOG_FILE="$LOG_DIR/voice-$(date +%Y%m%d-%H%M%S).log"
    LOG_IS_DEFAULT=1
  fi
  # XTRACE GUARD — why this log is REFUSED under `bash -x`.
  # The shell's own trace goes to stderr, which is exactly what this block
  # captures, and the trace prints every expansion.  Measured 2026-09-26 with
  # a probe key: under -x the raw value hits stderr THREE times, including
  # `+ export OPENAI_API_KEY=sk-...` and the bare `+ KEY_VALUE=sk-...`.
  # Capturing stderr would therefore write the API key in PLAINTEXT into a
  # log file on the robot.  Debugging is precisely what this log exists for,
  # so `bash -x` is a thing the operator will reach for — the guard has to be
  # here, not in a comment telling them not to.  The trace still reaches the
  # terminal, where nothing persists it; only the FILE is refused.
  if [[ "$-" == *x* ]]; then
    echo "run_voice.sh: xtrace is ON (bash -x) — NOT writing a log file: the \
shell trace prints the API key value, which must never be persisted to disk. \
The trace stays on this terminal only.  Re-run without -x to get the log." >&2
  # The launcher owns the default convention, so it creates that directory.
  # Without this the very FIRST run on a box fails the `: >>` probe below —
  # voice/logs does not exist in a fresh checkout — and the operator gets a
  # "cannot write the log file" warning instead of a log, on exactly the run
  # they wanted to read back.  A hand-set OPENBOT_LOG_FILE gets its parent
  # made too: the operator named that path, so a missing parent is a
  # directory to create, not a reason to refuse.  Either way the probe below
  # is still what decides — this only removes the avoidable failure.
  elif mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null; \
       : >>"$LOG_FILE" 2>/dev/null; then
    # `>>` (not touch) so we both prove writability AND never truncate a file
    # the operator pointed us at by hand.
    # 600: this file collects arbitrary app stderr, which can carry a bearer
    # token out of an HTTP error.  Nothing here needs it group/world readable.
    # This tightens a hand-set OPENBOT_LOG_FILE too — deliberate: tightening is
    # the safe direction, and a log nobody else can read is the intent.
    chmod 600 "$LOG_FILE" 2>/dev/null || true
    exec > >(tee -a "$LOG_FILE") 2>&1
    echo "run_voice.sh: logging to $LOG_FILE"
    if [[ "$LOG_IS_DEFAULT" == "1" ]]; then
      # A stable name for `tail -f`; the timestamped files stay as history.
      #
      # THIS ALIAS IS THE ONE PIECE OF THE CONTRACT THE OPERATOR TYPES
      # (`tail -f voice/logs/latest.log`), so it is the one piece that must
      # NOT fail silently.  Measured 2026-10-01: when latest.log already
      # exists as a DIRECTORY, `ln -sfn` refuses ("cannot overwrite
      # directory") — and a bare `2>/dev/null || true` swallowed it.  The run
      # then looked completely healthy (timestamped log created, "logging to
      # ..." printed, app booted) while the alias stayed a directory and
      # `tail -f` died with "Is a directory".  Diagnosing that from the
      # outside is near-impossible, because every visible signal is green.
      #
      # So: clear a foreign occupant explicitly, and if the link still cannot
      # be made, SAY SO on stderr instead of pretending.
      if [[ -L "$LOG_DIR/latest.log" ]]; then
        :                       # our symlink (possibly dangling) — ln -sfn replaces it
      elif [[ -e "$LOG_DIR/latest.log" ]]; then
        # Not a symlink: something else owns this name (a stray mkdir, a
        # hand-made copy).  A dir is the measured case.  Remove it if empty;
        # otherwise move it aside rather than delete data we do not own.
        if rmdir "$LOG_DIR/latest.log" 2>/dev/null; then
          echo "run_voice.sh: removed a stray EMPTY DIRECTORY at \
'$LOG_DIR/latest.log' left by an earlier run (it blocked the log alias)." >&2
        elif mv "$LOG_DIR/latest.log" \
             "$LOG_DIR/latest.log.blocked-$(date +%Y%m%d-%H%M%S)" 2>/dev/null; then
          echo "run_voice.sh: moved a non-symlink occupant of \
'$LOG_DIR/latest.log' aside to 'latest.log.blocked-*' so the log alias can be \
created (it blocked the alias).  Nothing was deleted." >&2
        fi
      fi
      if ! ln -sfn "$LOG_FILE" "$LOG_DIR/latest.log" 2>/dev/null; then
        echo "run_voice.sh: WARNING: could NOT create the log alias \
'$LOG_DIR/latest.log' — it still holds: $(ls -ld "$LOG_DIR/latest.log" 2>&1).  \
The timestamped log IS being written: $LOG_FILE  (read that path directly; \
'tail -f' on the alias will not work until the occupant is cleared)." >&2
      fi
      # Retention: the newest 20 session logs, oldest removed.  Anchored three
      # ways — the directory, the voice-*.log name, and -type f — so neither a
      # hand-set OPENBOT_LOG_FILE nor the latest.log symlink can be caught by
      # it.  (find, not `ls`: same reason shellcheck SC2012 exists, and it lets
      # the symlink be excluded by type rather than by name alone.)
      find "$LOG_DIR" -maxdepth 1 -type f -name 'voice-*.log' \
        -printf '%T@\t%p\n' 2>/dev/null \
        | sort -rn | tail -n +21 | cut -f2- | xargs -r rm -f 2>/dev/null || true
    fi
  else
    echo "run_voice.sh: WARNING: cannot write the log file '$LOG_FILE' — \
continuing WITHOUT a log (the boot is not blocked by its own logging)." >&2
  fi
fi

# -- which lane is being booted? ---------------------------------------------
# The offline lane is the architecture's no-network fallback: regex brain, no
# LLM, no API key, ever (config.offline.yaml).  This preflight used to demand
# the OpenAI key file for EVERY invocation, so `run_voice.sh --config
# config.offline.yaml` on a keyless box died with "REFUSAL: OpenAI API key
# file not found" — telling the operator to create a credential that lane
# never reads, and hard-blocking the very path this script's own header
# advertises (found + reproduced 2026-09-02).  `python3 voice/app.py --config
# config.offline.yaml` booted the same lane keyless the whole time, so the
# launcher was strictly worse than calling the app directly.
#
# The FIRST fix for that read `brain.lane` out of the YAML with sed.  Review
# defeated it twice, and both defeats had one cause: sed was a SECOND,
# divergent implementation of a merge the app already does.  Measured
# 2026-09-02:
#   config.offline.yaml --profile voice -> lane 'realtime'  (NEEDS a key)
#   config.offline.yaml --profile cloud -> lane 'transcript'(NEEDS a key)
# The sed saw `lane: offline` in the file, announced "no key required",
# skipped the gate — and the app then deep-merged the profile on top and
# booted a lane that dials out.  Quoted or padded spellings (`lane: 'offline'`)
# defeated the regex the other way, re-blocking a keyless lane.
#
# So the lane is no longer PARSED here; it is RESOLVED by the app's own
# loader, below, after conda activate.  One implementation, no divergence.
SEL_CONFIG=""
SEL_PROFILE=""
PREV=""
for a in "$@"; do
  case "$a" in
    --config=*)  SEL_CONFIG="${a#--config=}" ;;
    --profile=*) SEL_PROFILE="${a#--profile=}" ;;
    *)
      [[ "$PREV" == "--config"  ]] && SEL_CONFIG="$a"
      [[ "$PREV" == "--profile" ]] && SEL_PROFILE="$a"
      ;;
  esac
  PREV="$a"
done
[[ -n "$SEL_CONFIG" ]] || SEL_CONFIG="$CONFIG_DEFAULT"
case "$SEL_CONFIG" in /*) ;; *) SEL_CONFIG="$REPO_ROOT/$SEL_CONFIG" ;; esac

if [[ "$HELP" == "0" ]]; then
  # -- voice models (installed by voice/fetch_models.sh) -------------------
  # No interpreter needed, so this runs before conda.  The KEY gate does
  # need one (it asks the app's loader which lane is being booted) and so
  # lives below, after activate.
  MISSING=()
  [[ -d "$MODEL_DIR/vosk-model-small-en-us" ]] || MISSING+=("$MODEL_DIR/vosk-model-small-en-us (KWS model)")
  [[ -f "$MODEL_DIR/ggml-tiny.en-q5_1.bin" ]]  || MISSING+=("$MODEL_DIR/ggml-tiny.en-q5_1.bin (whisper STT model)")
  [[ -x "$MODEL_DIR/whisper-server" ]]         || MISSING+=("$MODEL_DIR/whisper-server (whisper.cpp server)")
  [[ -x "$MODEL_DIR/piper" && -d "$MODEL_DIR/espeak-ng-data" ]] \
                                               || MISSING+=("$MODEL_DIR/piper + espeak-ng-data (Piper TTS)")
  [[ -f "$MODEL_DIR/en_US-ryan-medium.onnx" && -f "$MODEL_DIR/en_US-ryan-medium.onnx.json" ]] \
                                               || MISSING+=("$MODEL_DIR/en_US-ryan-medium.onnx(+.json) (Piper voice)")
  [[ -f "$REPO_ROOT/en_US-ryan-medium.onnx" && -f "$REPO_ROOT/en_US-ryan-medium.onnx.json" ]] \
                                               || MISSING+=("$REPO_ROOT/en_US-ryan-medium.onnx(+.json) (deploy-root piper-voice copy)")
  if [[ "${#MISSING[@]}" -gt 0 ]]; then
    MISSING_MSG=""
    for m in "${MISSING[@]}"; do MISSING_MSG+="$m, "; done
    MISSING_MSG="${MISSING_MSG%, }"
    die "voice models are incomplete — run voice/fetch_models.sh first.  \
Missing: $MISSING_MSG"
  fi
fi

# -- conda env --------------------------------------------------------------
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  die "conda not found at $CONDA_ROOT (override with OPENBOT_CONDA_ROOT).  \
The voice app must run under a py3.8 box env with \
voice/requirements-box.txt installed."
fi
# conda's shell functions abort the shell on unbound variables under
# `set -u` (bash aborts even inside an `if !` guard) — the known conda
# + set -u incompatibility.  -u is suspended only around source/activate
# and re-enabled immediately after.
set +u
# shellcheck disable=SC1090,SC1091  # conda.sh path is a runtime override
source "$CONDA_SH"
if ! conda activate "$CONDA_ENV" >/dev/null 2>&1; then
  set -u
  die "conda env '$CONDA_ENV' not found under $CONDA_ROOT (override with \
OPENBOT_CONDA_ENV).  Create it and install the voice deps:  \
conda create -n $CONDA_ENV python=3.8 && conda activate $CONDA_ENV && \
pip install -r \"$REPO_ROOT/voice/requirements-box.txt\""
fi
set -u
PY="$(command -v python3)"
case "$PY" in
  "$CONDA_ROOT"/*) ;;
  *) die "conda activate '$CONDA_ENV' did not put its python first on \
PATH (got: $PY) — check OPENBOT_CONDA_ROOT / OPENBOT_CONDA_ENV." ;;
esac

# -- lane resolution: ask the APP's loader, never a second parser -----------
# One implementation of the merge, so the launcher and the app can never
# disagree about which lane is booting (see the note above the arg scan).
# Runs under the conda python, i.e. the same interpreter the app will use.
cd "$REPO_ROOT"
NEEDS_KEY=1
LANE=""
if [[ "$HELP" == "0" ]]; then
  LANE="$(OPENBOT_CFG="$SEL_CONFIG" OPENBOT_PROF="$SEL_PROFILE" python3 - <<'PY' 2>/dev/null || true
import os, sys
sys.path.insert(0, os.getcwd())
from brain.config_loader import load
prof = os.environ.get("OPENBOT_PROF") or None
print(load(os.environ["OPENBOT_CFG"], prof).brain.lane)
PY
)"
  case "$LANE" in
    offline)
      NEEDS_KEY=0
      echo "run_voice.sh: lane 'offline' — no API key required (no network)." >&2
      ;;
    realtime|transcript)
      : # dials out; the key gate below applies
      ;;
    *)
      # Resolution failed — a broken/absent config, or a loader error.  Do
      # NOT guess: demanding a key here would resurrect the very hard block
      # this section exists to remove, and the app's own loader is about to
      # print the authoritative diagnostic.  Warn, skip the key gate, let
      # the app fail with the real reason.
      echo "run_voice.sh: WARNING: could not resolve brain.lane from \
'$SEL_CONFIG'${SEL_PROFILE:+ (--profile $SEL_PROFILE)} — skipping the API-key \
preflight.  The app will report the underlying config error." >&2
      NEEDS_KEY=0
      ;;
  esac
fi

# -- key file (B6.4 no-secrets law; the STUART_PASS box-file pattern) -------
# Only the lanes that actually dial out: realtime (WebSocket) and transcript
# (REST).  The offline lane never reads it.
if [[ "$HELP" == "0" && "$NEEDS_KEY" == "1" ]]; then
  if [[ ! -f "$KEY_FILE" ]]; then
    die "OpenAI API key file not found: $KEY_FILE (lane '$LANE' dials out).  \
Create it ONCE on the box:  \
mkdir -p /home/unitree/openbot && printf '%s\n' 'sk-...' > \"$KEY_FILE\" && \
chmod 600 \"$KEY_FILE\" (chown unitree:unitree \"$KEY_FILE\").  Override the \
path with OPENBOT_KEY_FILE.  The key itself never lives in this repo.  \
For the no-network lane run with --config config.offline.yaml (no key needed)."
  fi
  MODE="$(stat -c %a "$KEY_FILE" 2>/dev/null || true)"
  if [[ -z "$MODE" || $(( 8#$MODE & 8#077 )) -ne 0 || $(( 8#$MODE & 8#400 )) -eq 0 ]]; then
    die "API key file $KEY_FILE must be owner-readable with NO group/other \
bits (chmod 600; 400 also accepted) — current mode: '${MODE:-unreadable}'.  \
Fix: chmod 600 \"$KEY_FILE\""
  fi
fi

# -- key -> app env only -----------------------------------------------------
# Skipped entirely for the offline lane: nothing downstream reads it, and
# reading a file that need not exist would re-introduce the hard block.
if [[ "$HELP" == "0" && "$NEEDS_KEY" == "1" ]]; then
  KEY_VALUE="$(<"$KEY_FILE")"          # command substitution strips trailing newlines
  # ${VAR//[[:space:]]/} strips EVERY whitespace class, not just literal
  # spaces: a tab-only or CR-only file is empty too, and an all-whitespace
  # key exported as OPENAI_API_KEY would fail at the provider with an
  # opaque 401 instead of here with a clear reason.
  [[ -n "${KEY_VALUE//[[:space:]]/}" ]] || die "API key file $KEY_FILE is empty (or only whitespace)."
  export OPENAI_API_KEY="$KEY_VALUE"   # never echoed / logged anywhere
  unset KEY_VALUE
fi

# -- route pin (user directive 2026-09-05) -----------------------------------
# Invoked only on a REAL boot: everything above this point is either a
# no-op --help path (HELP=1 — argparse answers in the app, nothing boots)
# or a refusal die path (missing models / conda / key, exit 2), so the pin
# can never mutate host routes for an invocation that will not launch the
# app.  Runs before the app subprocess starts, so the DDS + mic-group
# multicast pin is in place for the boot.
if [[ "$HELP" == "0" ]]; then
  openbot_route_pin
fi

# -- run ---------------------------------------------------------------------
cd "$REPO_ROOT"
# Bare-name PATH fallback for the cascade's whisper-bin resolution: a bare
# bin_path is exec'd layout-first from the cwd-relative voice/models/<name>
# (when present + executable), else resolved on this PATH (cascade.py
# _resolve_bin).  Run with cwd = the deploy root: layout wins either way.
export PATH="$MODEL_DIR:$PATH"
HAS_CONFIG=0
for a in "$@"; do
  case "$a" in --config|--config=*) HAS_CONFIG=1 ;; esac
done
if [[ "$HAS_CONFIG" == "1" ]]; then
  exec python3 "$APP" "$@"
else
  exec python3 "$APP" --config "$CONFIG_DEFAULT" "$@"
fi
