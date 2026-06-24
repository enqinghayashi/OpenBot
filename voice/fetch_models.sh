#!/usr/bin/env bash
# fetch_models.sh — download + verify the OpenBot v4 voice models and
# binaries into voice/models/ (gitignored and excluded from the deploy
# manifest by deploy/make_manifest.py — weights never ship with the kit).
#
# Target: the box (Orin NX 16 GB, JetPack 5 = Ubuntu 20.04, aarch64).
# Run it ON the box from the deploy root:  bash voice/fetch_models.sh
#
# What it installs (paths are the config contract — voice/config*.yaml):
#   voice/models/vosk-model-small-en-us/   KWS wake-word model (Vosk)
#   voice/models/ggml-tiny.en-q5_1.bin     whisper.cpp STT model (B2 decision 2026-09-04)
#   voice/models/whisper-server            whisper.cpp server (BUILT HERE)
#   voice/models/piper  + libs + espeak-ng-data   Piper TTS binary
#   voice/models/en_US-ryan-medium.onnx(+.json)  Piper voice — REUSE-FIRST
#   <deploy root>/en_US-ryan-medium.onnx(+.json) copy: the shipped configs'
#                                          piper_voice is the BARE filename
#                                          'en_US-ryan-medium.onnx',
#                                          resolved against the app's cwd
#                                          (the deploy root) — these copies
#                                          are what piper actually loads today
#                                          (USER DIRECTIVE 2026-09-04: the
#                                          box's OWN voice model is reused —
#                                          REUSE-FIRST source now the deploy-root
#                                          copy, i.e. Leon-folder-only paths
#                                          ($REPO_ROOT); the download is a
#                                          fresh-box fallback)
#
# WHY whisper.cpp is built from source here (verified 2026-08-31): every
# published prebuilt aarch64 server binary (whisper-bin-ubuntu-arm64.tar.gz
# from v1.8.7 and the b4938 nightly) requires GLIBC_2.34 (checked with
# readelf); JetPack 5 ships glibc 2.31, so those binaries die at load.
# Building on the box (cmake) yields a native binary that runs, and
# BUILD_SHARED_LIBS=OFF yields a SELF-CONTAINED whisper-server (no
# libwhisper.so.1 / libggml.so.0 to carry — the shared builds bake the
# absolute build-tree paths into the binary's RPATH, which die with the
# temp build dir; caught by the idempotency re-run).  Requires cmake + a
# C++ compiler on the box (JetPack 5 ships gcc/make; cmake can be added
# with `pip install cmake` in the box env if absent).
#
# Idempotency: an existing artifact is SKIPPED only when it passes its
# own check (sha256 for every artifact we could pin, plus an exec smoke
# for the two binaries); a mismatched or truncated artifact is re-fetched
# (never silently reused).  Any mismatch ends with exit != 0 and a loud
# REFUSAL on stderr.
#
# Env overrides:
#   OPENBOT_FETCH_ARCH=aarch64   (default: uname -m — the kit REFUSES
#                                 non-aarch64 hosts because the piper
#                                 binary is aarch64-only; the override
#                                 exists for dry-testing the script on
#                                 another arch — exec smokes are skipped)
#   OPENBOT_MODEL_DIR=<dir>      (default: <repo>/voice/models)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${OPENBOT_MODEL_DIR:-$REPO_ROOT/voice/models}"
CACHE_DIR="$MODEL_DIR/.cache"
NPROC="$(nproc 2>/dev/null || echo 4)"

fail() { echo "fetch_models.sh: REFUSAL: $*" >&2; exit 1; }
warn() { echo "fetch_models.sh: WARNING: $*" >&2; }

# --- arch gate ------------------------------------------------------------
ARCH="${OPENBOT_FETCH_ARCH:-$(uname -m)}"
case "$ARCH" in
  aarch64|arm64) ;;
  *)
    fail "this kit targets the Orin NX box (aarch64); host arch is '$(uname -m)'.  \
The piper prebuilt binaries are aarch64-only.  To dry-test the script on another \
arch set OPENBOT_FETCH_ARCH=aarch64 (downloads and verifies everything; the \
binary exec smokes are skipped)."
    ;;
esac

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "required tool '$1' not found on PATH.  On the box (JetPack 5 / Ubuntu 20.04): \
sudo apt install $1   (or for cmake: pip install cmake in the box conda env)."
  fi
}
for t in curl sha256sum stat tar; do require_cmd "$t"; done
require_cmd unzip

mkdir -p "$MODEL_DIR" "$CACHE_DIR"

# --- helpers ---------------------------------------------------------------
# fetch_verify <url> <expected_size> <expected_sha256> <dest> -> prints
# SKIP (dest already verified) or OK (freshly downloaded + verified);
# loud fail (exit 1) on any size/sha mismatch.  Downloads to dest.part
# so a failed fetch never leaves a half file at the final path.
fetch_verify() {
  local url="$1" want_size="$2" want_sha="$3" dest="$4"
  if [[ -f "$dest" ]]; then
    local got
    got="$(sha256sum "$dest" | awk '{print $1}')"
    if [[ "$got" == "$want_sha" ]]; then
      echo "SKIP"
      return 0
    fi
    warn "existing $dest has sha256 $got != pinned $want_sha — re-fetching"
    rm -f "$dest"
  fi
  curl -fL --retry 3 --retry-delay 2 -o "$dest.part" "$url" || {
    rm -f "$dest.part"
    fail "download failed: $url"
  }
  local size got
  size="$(stat -c %s "$dest.part")"
  got="$(sha256sum "$dest.part" | awk '{print $1}')"
  if [[ "$size" != "$want_size" || "$got" != "$want_sha" ]]; then
    rm -f "$dest.part"
    fail "downloaded $url failed verification (size $size != $want_size, \
sha256 $got != $want_sha).  The upstream file may have changed — update the \
pinned size/sha256 in this script, then re-run."
  fi
  mv "$dest.part" "$dest"
  echo "OK"
}

TMPDIRS=()
cleanup() { for d in "${TMPDIRS[@]}"; do rm -rf "$d"; done; }
trap cleanup EXIT

S_VOSK="-"; S_GGML="-"; S_WSRV="-"; S_PIPER="-"; S_ALBA="-"

# --- 1. Vosk KWS model (68 MB class; measured 41,205,931 bytes) ------------
# config.voice.yaml / config.offline.yaml: kws_model_dir voice/models/vosk-model-small-en-us
VOSK_ZIP="$CACHE_DIR/vosk-model-small-en-us-0.15.zip"
VOSK_DIR="$MODEL_DIR/vosk-model-small-en-us"
if [[ -f "$VOSK_DIR/conf/mfcc.conf" ]]; then
  S_VOSK="SKIP"
else
  S_VOSK="$(fetch_verify \
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip" \
    41205931 30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498 \
    "$VOSK_ZIP")"
  rm -rf "$VOSK_DIR" "$MODEL_DIR/vosk-model-small-en-us-0.15"
  unzip -q -o "$VOSK_ZIP" -d "$MODEL_DIR"
  mv "$MODEL_DIR/vosk-model-small-en-us-0.15" "$VOSK_DIR"
  [[ -f "$VOSK_DIR/conf/mfcc.conf" ]] || fail "vosk model extraction did not produce $VOSK_DIR/conf/mfcc.conf"
fi

# --- 2. whisper.cpp STT model (tiny.en q5_1, 31 MB class; measured 32,166,155)
# B2 gate decision 2026-09-04: small.en was 21.7 s/inference on this CPU,
# tiny.en-q5_1 measures 1.52 s warm via the app's own WhisperSubmit path.
# cascade.py resolves the model itself: voice/models/ggml-<model>.bin
S_GGML="$(fetch_verify \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin" \
  32166155 c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b \
  "$MODEL_DIR/ggml-tiny.en-q5_1.bin")"

# --- 3. whisper.cpp server (built from source — see header) ----------------
# cascade.py launches it as the bare 'whisper-server' via PATH; run_voice.sh
# puts voice/models on PATH for the app subprocess.
WSRV="$MODEL_DIR/whisper-server"
if [[ -x "$WSRV" ]] && "$WSRV" --help >/dev/null 2>&1; then
  S_WSRV="SKIP"
else
  if [[ -x "$WSRV" ]]; then
    warn "existing $WSRV is not runnable (wrong arch / stale build) — rebuilding"
    rm -f "$WSRV"
  fi
  require_cmd cmake
  command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1 || \
    fail "no C compiler (cc/gcc) on PATH — whisper.cpp must be built on the box (see header)"
  WS_SRC="$CACHE_DIR/whisper.cpp-1.8.7.tar.gz"
  fetch_verify \
    "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v1.8.7.tar.gz" \
    8910569 0b988ba5053cfa720f6d399f3f21885b01c4222178be435ca2272d6872717554 \
    "$WS_SRC" >/dev/null
  TMP="$(mktemp -d)"
  TMPDIRS+=("$TMP")
  tar xzf "$WS_SRC" -C "$TMP"
  ( cd "$TMP/whisper.cpp-1.8.7" \
      && cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      && cmake --build build --config Release --target whisper-server -j"$NPROC" ) \
    || fail "whisper.cpp build failed (see output above)"
  cp "$TMP/whisper.cpp-1.8.7/build/bin/whisper-server" "$WSRV"
  chmod +x "$WSRV"
  "$WSRV" --help >/dev/null 2>&1 || fail "built $WSRV failed its --help smoke"
  S_WSRV="BUILT"
fi

# --- 4. Piper TTS binary (aarch64 prebuilt, glibc 2.17-only: box-safe) -----
# config.voice.yaml / config.offline.yaml: piper_bin voice/models/piper.
# The tarball's piper/ dir installs FLAT (binary + $ORIGIN libs +
# espeak-ng-data as siblings) so the RUNPATH $ORIGIN resolution works.
PIPER_TGZ="$CACHE_DIR/piper_linux_aarch64.tar.gz"
if [[ -x "$MODEL_DIR/piper" && -f "$MODEL_DIR/libpiper_phonemize.so.1" \
      && -d "$MODEL_DIR/espeak-ng-data" ]]; then
  S_PIPER="SKIP"
else
  S_PIPER="$(fetch_verify \
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz" \
    26004717 fea0fd2d87c54dbc7078d0f878289f404bd4d6eea6e7444a77835d1537ab88eb \
    "$PIPER_TGZ")"
  TMP="$(mktemp -d)"
  TMPDIRS+=("$TMP")
  tar xzf "$PIPER_TGZ" -C "$TMP"
  cp -r "$TMP/piper/." "$MODEL_DIR/"
  rm -rf "$MODEL_DIR/pkgconfig"
  chmod +x "$MODEL_DIR/piper"
  case "$(uname -m)" in
    aarch64|arm64)
      "$MODEL_DIR/piper" --help >/dev/null 2>&1 \
        || fail "installed piper failed its --help smoke"
      ;;
    *)
      warn "exec smoke for piper skipped (test arch override on '$(uname -m)')"
      ;;
  esac
fi

# --- 5. Piper voice (en_US ryan medium, 63 MB class; measured 63,201,294) ---
# REUSE-FIRST (USER DIRECTIVE 2026-09-04: "have all the audio voice model
# that were already on Stuwart, we can use the same voice audio model if not
# yet used"; DIRECTIVE 2026-09-04 evening: box paths are Leon-folder-only —
# /home/unitree/leon): the reuse source is the box's own DEPLOY-ROOT copy
# ($REPO_ROOT/en_US-ryan-medium.onnx), verified byte-identical to upstream
# piper-voices v1.0.0 (sha below).  When that copy matches the pin it is
# COPIED, never re-downloaded; a box without it falls back to the
# upstream download so this script stays portable.
# Piper needs the .onnx.json sidecar NEXT TO THE MODEL (cascade.py PiperTTS
# passes only --model; piper derives <model>.json), so BOTH files ship to
# MODEL_DIR and to the deploy root.
RYAN="$MODEL_DIR/en_US-ryan-medium.onnx"
RYAN_JSON="$MODEL_DIR/en_US-ryan-medium.onnx.json"
RYAN_SRC="$REPO_ROOT/en_US-ryan-medium.onnx"
RYAN_SRC_JSON="$REPO_ROOT/en_US-ryan-medium.onnx.json"
RYAN_SHA="abf4c274862564ed647ba0d2c47f8ee7c9b717d27bdad9219100eb310db4047a"
RYAN_JSON_SHA="44034c056cb15681b2ad494307c7f3f2e4499d1253c700c711fa0a4607ffe78d"
S_RYAN="-"
if [[ -f "$RYAN" && -f "$RYAN_JSON" ]] \
   && [[ "$(sha256sum "$RYAN" | awk '{print $1}')" == "$RYAN_SHA" ]] \
   && [[ "$(sha256sum "$RYAN_JSON" | awk '{print $1}')" == "$RYAN_JSON_SHA" ]]; then
  S_RYAN="SKIP"
elif [[ -f "$RYAN_SRC" && -f "$RYAN_SRC_JSON" ]] \
   && [[ "$(sha256sum "$RYAN_SRC" | awk '{print $1}')" == "$RYAN_SHA" ]] \
   && [[ "$(sha256sum "$RYAN_SRC_JSON" | awk '{print $1}')" == "$RYAN_JSON_SHA" ]]; then
  cp "$RYAN_SRC" "$RYAN"
  cp "$RYAN_SRC_JSON" "$RYAN_JSON"
  S_RYAN="REUSED"
else
  S_RYAN="$(fetch_verify \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx" \
    63201294 "$RYAN_SHA" "$RYAN")"
  fetch_verify \
    "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json" \
    4883 "$RYAN_JSON_SHA" "$RYAN_JSON" >/dev/null
fi
# The shipped configs' piper_voice is the bare filename 'en_US-ryan-medium.onnx'
# and PiperTTS passes it to `piper --model` verbatim — resolved against the
# app's cwd (the deploy root); piper loads the '<voice>.json' from the SAME
# directory.  Keep the contract-working copies in sync.
ROOT_RYAN="$REPO_ROOT/en_US-ryan-medium.onnx"
ROOT_RYAN_JSON="$REPO_ROOT/en_US-ryan-medium.onnx.json"
if [[ -f "$ROOT_RYAN" && -f "$ROOT_RYAN_JSON" ]] \
   && [[ "$(sha256sum "$ROOT_RYAN" | awk '{print $1}')" == "$RYAN_SHA" ]] \
   && [[ "$(sha256sum "$ROOT_RYAN_JSON" | awk '{print $1}')" == "$RYAN_JSON_SHA" ]]; then
  :
else
  cp "$RYAN" "$ROOT_RYAN"
  cp "$RYAN_JSON" "$ROOT_RYAN_JSON"
fi
# The superseded alba voice (pre-2026-09-04 pin) is orphaned now that
# configs + preflight resolve ryan — drop its copies rather than let a
# 63 MB stale sidecar linger next to the real voice.
rm -f "$MODEL_DIR/en_GB-alba-medium.onnx" "$REPO_ROOT/en_GB-alba-medium.onnx"

# --- summary ----------------------------------------------------------------
TOTAL="$(du -sm "$MODEL_DIR" 2>/dev/null | awk '{print $1}')"
echo "fetch_models.sh: ALL MODELS READY — vosk=$S_VOSK whisper-model=$S_GGML \
whisper-server=$S_WSRV piper=$S_PIPER voice=$S_RYAN (voice/models total ${TOTAL} MiB; \
deploy-root piper-voice copies $(test -f "$ROOT_RYAN" && test -f "$ROOT_RYAN_JSON" && echo present || echo MISSING))"
