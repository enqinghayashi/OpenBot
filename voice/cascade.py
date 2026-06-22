"""cascade: the offline voice lane — utterance audio -> whisper.cpp -> brain
-> piper (L2, plan §3 cascade + §5b offline lane).

Position in the v4 speech protocol (voice/__init__.py architecture, offline
lane):

    G1 mic frames  ->  EnergyGate  ->  OfflineChain.run_pipeline(pcm16)
                                          |  whisper.cpp server (subprocess,
                                          |  POST /inference, 127.0.0.1:8082)
                                          v
                                     OfflineBrain.new_utterance(text,
                                                                typed=False)
                                          |  (brain/agent/offline_brain.py —
                                          |  the regex brain, no LLM)
                                          v
                                     OfflineOutcome.reply  (verdict-aware
                                          |  phrasing — success only ever
                                          |  from the post-hook verifier)
                                          v
                                     piper (subprocess) -> PCM16 -> G1Speaker

The chain is a pure orchestrator between the three stages: it owns NO
intent logic, NO verification and NO success authority.  Every boundary is
injectable — ``submit`` (whisper), ``brain``, ``tts`` (piper) — so the app
hands in real ones and tests hand in fakes.

--------------------------------------------------------------------------
THE LAWS (each with its implementation pointer)
--------------------------------------------------------------------------

L1. EVERY STAGE HAS A TIMEOUT; there is no unbounded wait anywhere.
    * whisper server launch: ``whisper_launch_timeout_s`` (default 30 s)
      — after a Popen the server must accept a probe within it, else the
      stage fails (WhisperSubmit._ensure_server);
    * one whisper POST: ``whisper_submit_timeout_s`` (default 15 s) — the
      urllib request timeout (WhisperSubmit._post);
    * brain: ``brain_timeout_s`` (default 30 s) — new_utterance runs in a
      daemon worker thread joined with this bound (OfflineChain._run_brain).
      On expiry the chain raises stage 'brain' and REFUSES the thread's
      eventual outcome — it is never spoken and never treated as a decline
      (an internal crash or stall is a chain failure, not a canned reply);
    * piper: ``piper_tts_timeout_s`` (default 20 s) — ONE
      ``communicate(timeout=...)`` bounds spawn + model load + synthesis
      together (PiperTTS.__call__).
    All four are constructor args; these defaults are the documented
    production values.

L2. WHISPER FAILURE -> SPOKEN CANNED DECLINE, and NO brain call.  A whisper
    error (HTTP != 200, connection refused, timeout, unparseable payload)
    and an EMPTY transcript both speak WHISPER_DECLINE — the chain's OWN
    string ("I could not hear you clearly — say that again."), never copied
    from anywhere: nothing was understood, so the honest reply asks for a
    repeat (OfflineChain.run_pipeline / _decline).  The decline itself goes
    through piper; when piper is dead too, L4 applies — no speech is
    possible at all.

L3. BRAIN OUTCOME -> ITS OWN REPLY, SPOKEN VERBATIM.  The brain's
    OfflineOutcome.reply is verdict-aware by construction: only the
    post-hook verifier's VERDICT:success ever phrases success, declines are
    the brain's own honest statements, and motion failures phrase as
    failures (offline_brain.py contract).  The chain speaks that text
    byte-for-byte and never rewrites it — verdict-aware motion replies and
    declines flow through unchanged.  A brain EXCEPTION or TIMEOUT is a
    chain failure (OfflineChainError stage 'brain'), never papered over
    with a decline string.

L4. PIPER FAILURE -> EXACTLY ONE RETRY -> NON-SPEECH SIGNAL, then
    OfflineChainError(stage='piper').  PIPER_RETRIES == 1 is a law, not a
    knob.  When the second attempt also fails, NO speech is possible, so
    the chain must make that failure explicit: the sink's LED is flashed
    red (``set_led`` when the sink has one; the G1Speaker has it), the
    standard log line PIPER_DEAD_LOG is written, and OfflineChainError(
    stage='piper') is raised (OfflineChain._speak / _signal_piper_dead).
    NEVER a fake empty success — an empty-piper output is treated exactly
    like a failure.

L5. THE CHAIN NEVER REPORTS SUCCESS.  ChainOutcome has deliberately NO
    success field: the executor is the only success authority and it
    speaks through the brain's verdict-aware reply.  The chain only
    annotates (``source``, ``brain_outcome``, the synthesized ``pcm``).

--------------------------------------------------------------------------
WIRE FACTS (the binaries are SUBPROCESSES, never imports)
--------------------------------------------------------------------------

* whisper.cpp is launched as a subprocess (the ``whisper-server`` binary,
  ``-m <model> --host 127.0.0.1 --port 8082``) only when no server is
  already answering on the url, and it is never imported: the submit
  callable POSTs each utterance as a WAV file (multipart ``file`` field,
  ``response_format=json``) to ``<url>`` — config.offline.yaml ships
  ``http://127.0.0.1:8082/inference``.  The server is probed before any
  launch, so an operator-started server (the deploy kit's own launch) is
  reused and no second process is ever started.  ``model_path`` is
  required for a launch — explicit, or the standard deploy layout
  ``voice/models/ggml-<model>.bin`` ONLY when that file exists; the chain
  never guesses a model path.  The ``whisper-server`` BINARY is resolved
  the same layout-first way, at LAUNCH time only (never in the
  constructor): a ``bin_path`` containing a path separator is used as-is;
  a BARE name first tries the cwd-relative deploy layout
  ``voice/models/<bin_path>`` when that file exists AND is executable
  (os.path.isfile + os.access X_OK — an existence check, never a guess),
  falling back to the bare name (resolved on PATH) only when it does
  not.  Both launch forms — ``run_voice.sh`` and a bare
  ``python3 voice/app.py`` — run with cwd = the deploy root, so the
  layout copy is what gets exec'd deterministically.
* piper is launched per synthesis as a subprocess
  (``piper --model <voice> --output_raw``): the reply text on stdin, raw
  PCM16 mono on stdout.  Piper emits the VOICE MODEL's native sample rate
  (en_US-ryan-medium.onnx is 22050 Hz) — the chain annotates it on
  ChainOutcome.sample_rate and the SINK owns any resampling to the G1's
  16 kHz PlayStream contract (speaker.py).  The chain never resamples in
  stdlib; that is a documented deploy-kit wiring responsibility, not a
  hidden assumption.

Module-level import surface is stdlib-only (dataclasses/json/logging/os/
struct/subprocess/threading/time/urllib + brain.agent.offline_brain, itself
stdlib-only — verified): the binaries above are subprocesses and no
third-party library is ever imported, mirroring the voice package's lazy
pattern (voice/__init__.py).
"""

from __future__ import annotations

import json
import logging
import os
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

from brain.agent.offline_brain import OfflineOutcome

log = logging.getLogger("voice.cascade")

# ---------------------------------------------------------------------------
# constants - our production defaults
# ---------------------------------------------------------------------------

#: our own line
#: whisper died,
#: or heard nothing
#: -> just ask again
WHISPER_DECLINE = "I could not hear you clearly — say that again."

#: exactly one retry
#: a law, not a knob
#: then the no-speech signal
PIPER_RETRIES = 1

#: piper died twice:
#: no speech possible - LED red,
#: then raise stage='piper'
PIPER_DEAD_LOG = (
    "offline chain: piper synthesis failed twice — NO speech is possible; "
    "LED flashed red; OfflineChainError(stage='piper') raised"
)

#: our defaults
#: brain 30, launch 30,
#: submit 15, piper 20 -
#: all ctor args
BRAIN_TIMEOUT_S = 30.0
WHISPER_LAUNCH_TIMEOUT_S = 30.0
WHISPER_SUBMIT_TIMEOUT_S = 15.0
PIPER_TTS_TIMEOUT_S = 20.0

#: piper's raw rate,
#: sink resamples to
#: the G1's 16 kHz
PIPER_OUTPUT_RATE = 22050

#: default endpoint;
#: both configs
#: ship exactly this
WHISPER_URL = "http://127.0.0.1:8082/inference"


# ---------------------------------------------------------------------------
# what can go wrong
# ---------------------------------------------------------------------------


class WhisperError(RuntimeError):
    """The whisper.cpp stage failed (subprocess layer).  The chain maps
    every whisper failure onto a spoken decline (L2); this class is what
    the WhisperSubmit boundary raises."""


class PiperError(RuntimeError):
    """The piper stage failed (subprocess layer).  The chain maps piper
    failures onto one retry, then the non-speech signal + a raised
    OfflineChainError (L4); this class is what PiperTTS raises."""
