"""voice/app.py — the OpenBot v4 voice app (L2, plan §3 + §4 + §5b).

This is the L2 WIRING file: the app builds the lane the loaded config
demands and runs the full session stack end-to-end.  The LANE AUTHORITY
is the real :class:`voice.director.Director` (director.py — the only
director): the deprecated boolean-awake "default director" is gone.
Lane ONLINE/DEGRADING/REOPENING = the realtime backend; the app runs
the offline cascade exactly while the director says OFFLINE_CASCADE.
The director is fed the session events (turn bookkeeping + failover +
session_expired) and the lane's own loud-exit signal; it announces its
transitions through the app's speak path.  Lane switches are director
transitions only, applied between utterances at conversation boundaries
by the director's own rule — never mid-utterance.  KWS wake/sleep words
remain INPUT GATING only (asleep -> utterances dropped), never lane
state.

WIRING MAP (the three lanes build_brain_stack dispatches on)

    brain.lane 'realtime' (config.voice.yaml, the cloud lane):
        ToolRegistry + hooks + ToolExecutor + gateway backend (the SAME
        brain stack G1Agent.from_config builds, see build_brain_stack)
        + RealtimeClient(cfg.realtime, tools=flatten_specs(...))
        -> RealtimeBackend(client) -> VoiceSession(backend)
        The app is the director: session 'tool_call' events (not
        intercepted) execute on a worker thread through the SAME
        ToolExecutor (pre-hook confirm gate + post-hook verifier apply)
        and the result is answered via session.respond_tool_result.
        query_user two-phase = the session's OWN L9 intercept (the app
        never answers an intercepted call).  The app schedules
        backend.run_forever() as an asyncio task and feeds the session
        24 kHz chunks from the mic pump (echo guard + directed-mode KWS
        gate first).

    brain.lane 'transcript' (config.voice.yaml, the REST cloud lane):
        the same brain stack + CloudLLMClient(cfg.rest.base_url,
        cfg.rest.model, api_key_env=cfg.rest.api_key_env,
        tool_mode=cfg.rest.tool_mode, timeout_s=cfg.rest.timeout_s,
        prose_fallback=cfg.rest.prose_fallback)
        -> TranscriptBackend(client, tools=registry.get_specs() (the
        NESTED llm_cloud wire shape, validated+flattened eagerly by the
        backend — never tools=[]), submit=WhisperSubmit(url=
        cfg.voice.whisper_url, model=cfg.voice.whisper_model),
        tts=PiperTTS(bin_path=cfg.voice.piper_bin, voice=
        cfg.voice.piper_voice) — the SAME configured STT/TTS boundaries
        the offline cascade uses) (voice/transcript.py — the BrainBackend
        surface, the same VoiceSession the realtime lane drives) ->
        VoiceSession(backend).  The lane runs the SAME _run_realtime
        cloud session loop; the api-key law is the realtime lane's (the
        key is read from the env var NAME cfg.rest.api_key_env, never
        from the config file); the director's cfg carries the ACTUAL
        lane and its probe prefers the backend's own health_probe() (the
        transcript wire's reach check) over the realtime 'connected'
        flag.

    brain.lane 'offline' (config.offline.yaml, the no-network lane):
        the same brain stack + OfflineBrain(executor, embodiment,
        name=cfg.voice.name) -> OfflineChain(brain, submit=WhisperSubmit,
        tts=PiperTTS) -> EnergyGate -> KWS -> MockMic/MockSink (--audio
        mock) or G1/laptop audio.  The pump whispers ONCE per utterance
        (chain.handle_utterance, never run_pipeline — the app owns the
        whisper stage so the SAME transcript text feeds both the
        UtteranceSlot's current-transcript law and the chain) and speaks
        the cascade WHISPER_DECLINE on a whisper failure (cascade L2).

    brain.lane anything else -> ValueError (main returns rc != 0 with a
        loud stderr; nothing is opened, no box state changes).

voice_confirm (plan §4) is the hooks.py confirm gate for BOTH lanes:
the CURRENT transcript decides by affine/negative regex; ambiguous ->
ONE short spoken-confirm round (the app speaks "Shall I proceed?"
through the local TTS path — the model is NOT involved) blocked on the
NEXT captured utterance while mic/KWS stay live; timeout/negative/
ambiguous denies (deny bias).  The offline chain's brain stage is sized
to fit the round: brain_timeout_s = BRAIN_TIMEOUT_S + CONFIRM_ROUND_S.

query_user two-phase (plan §4): the cloud lane uses the session's L9
intercept verbatim.  The offline lane gets the app's question-speak
path (OfflineQuestionBackend): query_user SPEAKS the question and
captures the next utterance as the answer; notify_user speaks the
notice.  (The regex brain's decision tree has no query lane today, so
this is defensive wiring for the tool path.)

--audio mock swaps ONLY the audio/ASR/TTS boundaries (MockMic +
MockSink + MockKws + scripted whisper transcripts + scripted piper PCM)
— the EnergyGate, KWS law, OfflineChain laws, OfflineBrain, ToolExecutor
hooks, ToolRegistry, SceneGraph and gateway backend all run REAL.

Maintains the codebase invariants: stdlib-only at import (vosk /
sounddevice / websockets / unitree_sdk2py are imported lazily inside
the audio/backend builders); the G1 mic/speaker are never constructed
for laptop/mock modes; main() returns rc != 0 with a loud stderr on any
config/backend build failure and never touches the box first.
"""

from __future__ import annotations

import argparse
import array
import base64
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------- sort out the import root
# the repo root must be on sys.path before the first absolute import
# sys.path[0] is the script's directory, not the working directory
# idempotent, and a no-op when the module is imported as a package
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# arrival relay listener: stdlib only, no ROS client library
# the announcer runs on every lane, so no ROS dependency may leak in
from voice import arrival_announce

log = logging.getLogger("voice.app")

# ----------------------------------------------------------- the defaults

#: --config default
#: relative, so the app must be launched from the repo root
CONFIG_DEFAULT = "config.yaml"

#: --audio default: the shipped configs are all G1
DEFAULT_AUDIO = "g1"

#: G1 cadence: 5120 bytes / 16 kHz PCM16 = 160 ms
G1_FRAME_BYTES = 5120
G1_SAMPLE_RATE = 16000

#: dead-mic watchdog: a stopped stream is a dead lane, not a quiet room
#: the mic parks in its recv timeout forever, so only the frame age shows it
#: warns every MIC_STALL_WARN_EVERY_S; --once aborts rc=1 at the limit
MIC_STALL_LIMIT_S = 30.0
MIC_STALL_WARN_EVERY_S = 15.0

#: unused wake deadline: the wake contract is a latch, with no window
#: only open_listen_window's signature names it
LISTEN_WINDOW_S = 10.0

#: 'stop talking' mute window: after the cut the pump drops utterances
#: silently for this long (no whisper, no decline) so small noises cannot
#: re-raise the prompt; a wake word inside the window wins anyway
STOP_TALKING_MUTE_S = 20.0

# voice_confirm law strings: ours, not the model's
# the recogniser only yields what KWS/whisper heard, so keep the lists
# short and anchored; a negative wins over "yes, but don't" (deny bias)
CONFIRM_YES_RE = re.compile(
    r"\b(?:yes|yeah|sure|go ahead|do it|okay|ok|yep|affirmative|correct|please do)\b",
    re.IGNORECASE,
)
CONFIRM_NO_RE = re.compile(
    r"\b(?:no|nope|don't|dont|stop|not yet|never mind|cancel|abort)\b",
    re.IGNORECASE,
)
CONFIRM_QUESTION = "Shall I proceed?"
#: one spoken round per ambiguity
#: bounded; a timeout is a deny, never a silent yes
CONFIRM_ROUND_TIMEOUT_S = 20.0

#: the summon is a doorbell, not a command: KWS and ASR share the audio,
#: so a whispered wake phrase reaches the brain as a command; the wake
#: utterance is consumed, never whispered, and the lane greets instead
WAKE_GREETING = "Hi, I'm ready for instructions."

#: 'turn off' answers before it goes: close_listen_window ends in
#: stop_talking(0.0), which discards the sink queue, so the ack must play
#: first; the wait is bounded so a wedged speaker never blocks the latch
TURN_OFF_ACK = "Shutting down voice."
TURN_OFF_ACK_TIMEOUT_S = 5.0

#: chain brain stage must fit the confirm round: 30 s of brain plus 20 s of
#: round = 50 s; the total stays a bound, never an unbounded wait
BRAIN_TIMEOUT_S = 30.0

#: offline question lane answer wait
#: sits inside query_user's 25 s timeout
QUESTION_ROUND_TIMEOUT_S = 20.0

#: default mock script: wake, an ambiguous motion (one confirm round),
#: "yes", then a non-motion turn; --directed-mode gates on the kws
#: entries, else the transcripts drive every turn; --script overrides
DEFAULT_MOCK_SCRIPT = {
    "kws": ["hey stewart", None, None, "sleep"],
    "transcripts": ["open the right hand", "yes", "what can you see"],
}
#: loud tone per scripted utterance: the energy gate needs a peak above
#: min_peak to open
MOCK_LOUD_S = 0.8
MOCK_TONE_HZ = 440.0
MOCK_TONE_AMP = 0.3
#: quiet chunks before the first turn: the gate learns its floor from what
#: it has seen, so a loud first chunk becomes the floor and lifts the
#: threshold above the utterance (max(min_peak, floor_factor * floor))
MOCK_WARMUP_CHUNKS = 10


# ----------------------------------------------------- the utterance slot

