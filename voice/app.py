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

class UtteranceSlot:
    """Thread-safe handshake for voice_confirm + the offline question lane.

    The capture path calls :meth:`deliver` with every captured utterance
    TEXT: ``latest()`` always tracks the most recent transcript (the
    "current transcript" the regex law reads), and a text delivered
    while a round is active IS the round's answer.  A round calls
    :meth:`begin_round` before speaking and :meth:`wait_answer` after;
    the round is consumed exactly once and a timeout returns None.
    Thread-safe: the pump thread and the session loop thread both
    deliver, the chain worker / director worker wait.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest = ""
        self._round_active = False
        self._delivered: Optional[str] = None
        self._event = threading.Event()

    def latest(self) -> str:
        with self._lock:
            return self._latest

    def begin_round(self) -> None:
        with self._lock:
            self._round_active = True
            self._delivered = None
            self._event.clear()

    def end_round(self) -> None:
        with self._lock:
            self._round_active = False

    def deliver(self, text: str) -> bool:
        """Record the utterance text; True when a waiting round consumed
        it as its answer (the caller then must NOT route it anywhere
        else — the round's wait_answer returns it)."""
        with self._lock:
            self._latest = str(text or "")
            if self._round_active and self._delivered is None:
                self._delivered = self._latest
                self._event.set()
                return True
            return False

    @property
    def round_active(self) -> bool:
        with self._lock:
            return self._round_active

    def wait_answer(self, timeout_s: float) -> Optional[str]:
        self._event.wait(timeout_s)
        with self._lock:
            answer = self._delivered
            self._round_active = False
            return answer


def make_round_capture(slot: UtteranceSlot) -> Callable[[float], Optional[str]]:
    """A capture callable for the offline question lane: one spoken
    answer via the shared slot (never concurrent with a confirm round —
    the executor runs one tool at a time)."""

    def capture(timeout_s: float) -> Optional[str]:
        slot.begin_round()
        try:
            return slot.wait_answer(timeout_s)
        finally:
            slot.end_round()

    return capture


def make_voice_confirm(
    slot: UtteranceSlot,
    speak: Callable[[str], None],
    round_timeout_s: float = CONFIRM_ROUND_TIMEOUT_S,
) -> Callable[[Any], bool]:
    """The hooks.py confirm gate: ``confirm(context) -> bool`` — True =
    motion proceeds.

    Law (plan §4): the CURRENT transcript decides by affine/negative
    regex (OUR strings); ambiguous -> ONE short spoken-confirm round
    blocked on the NEXT captured utterance while mic/KWS stay live; the
    round is bounded and a timeout / negative / still-ambiguous answer
    DENIES (a gate that never moves without explicit spoken consent).
    The spoken question rides the app's local TTS->sink path — the model
    is not involved, so no response accounting is touched, and the
    answer is consumed by the gate (in the cloud lane it arrives as a
    dropped_utterance — never replayed into the conversation).
    """

    def confirm(context: Any) -> bool:  # noqa: ARG001 — hooks contract
        transcript = (slot.latest() or "").strip().lower()
        if CONFIRM_NO_RE.search(transcript):
            return False
        if CONFIRM_YES_RE.search(transcript):
            return True
        # ambiguous: one spoken round; a missing speak path denies, never a
        # silent yes
        if speak is None:
            return False
        slot.begin_round()
        try:
            speak(CONFIRM_QUESTION)
            answer = slot.wait_answer(round_timeout_s)
        finally:
            slot.end_round()
        if not answer or not answer.strip():
            return False  # timeout: deny
        answer_l = answer.strip().lower()
        if CONFIRM_NO_RE.search(answer_l):
            return False
        return bool(CONFIRM_YES_RE.search(answer_l))

    return confirm


# ------------------------------------------ the offline question-speak path

class OfflineQuestionBackend:
    """Offline-lane backend wrapper that SPEAKS user-facing questions and
    captures the spoken answer — the offline counterpart of the session's
    L9 two-phase intercept (plan §4).  Every other verb (observe, pick,
    place, ...) delegates to the wrapped gateway backend.

    ``speak``/``capture`` are app-installed (play_pcm + the
    UtteranceSlot round).  When they are None (the cloud lane, where the
    session's L9 intercept owns query_user), query_user/notify_user
    delegate to the wrapped backend's own channel.  The regex brain's
    decision tree has no query lane today, so this is defensive wiring
    for the tool path — the spoken shape is the voice-lane convention.
    """

    def __init__(
        self,
        backend: Any,
        speak: Optional[Callable[[str], None]] = None,
        capture: Optional[Callable[[float], Optional[str]]] = None,
        answer_timeout_s: float = QUESTION_ROUND_TIMEOUT_S,
    ) -> None:
        if backend is None:
            raise ValueError("OfflineQuestionBackend requires a backend")
        self._backend = backend
        self.speak = speak
        self.capture = capture
        self.answer_timeout_s = answer_timeout_s

    def set_voice(
        self,
        speak: Callable[[str], None],
        capture: Callable[[float], Optional[str]],
    ) -> None:
        """Install the spoken question path (the offline runner does this
        once the audio lane exists)."""
        self.speak = speak
        self.capture = capture

    def query_user(self, args: dict) -> dict:
        if self.speak is None or self.capture is None:
            return self._backend.query_user(args)
        question = str(args.get("question") or "")
        self.speak(question or "(no question text)")
        answer = self.capture(self.answer_timeout_s)
        if answer is None or not str(answer).strip():
            from brain.agent.tools import fail

            return fail(
                "no spoken answer within {:.0f}s".format(self.answer_timeout_s)
            )
        from brain.agent.tools import ok

        return ok(str(answer).strip())

    def notify_user(self, args: dict) -> dict:
        if self.speak is not None:
            text = str(args.get("text") or args.get("message") or "")
            if text:
                self.speak(text)
                from brain.agent.tools import ok

                return ok("spoken notice")
        return self._backend.notify_user(args)

    # -- everything else goes to the wrapped backend -----------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


# --------------------------------------------- all the little audio helpers

class PlayingLevel:
    """Current playback loudness for the echo guard (thread-safe).

    Set to the peak of the pcm handed to the sink; expires ~0.5 s after
    the audio would have finished, so a freshly-finished playback stops
    suppressing the mic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._peak = 0.0
        self._expires = 0.0

    def peak(self) -> float:
        with self._lock:
            if time.monotonic() > self._expires:
                return 0.0
            return self._peak

    def set_playing(self, pcm: bytes) -> None:
        with self._lock:
            self._peak = float(_peak_of_pcm16(pcm))
            self._expires = time.monotonic() + len(pcm) / 32000.0 + 0.5


def _peak_of_pcm16(pcm: bytes) -> float:
    """Max |int16| of the buffer (sampled every 8th sample — a playback
    loudness estimate, not an analysis)."""
    n = len(pcm) // 2
    if n <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: n * 2])
    peak = 0
    for i in range(0, n, 8):
        v = samples[i]
        if v < 0:
            v = -v
        if v > peak:
            peak = v
    return float(peak)


def _resample_pcm16(src: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample for 16-bit mono PCM (pure stdlib).

    Used for 16 kHz mic -> 24 kHz realtime wire and 22050 Hz piper ->
    16 kHz G1 sink.  Linear interpolation is the documented seam — the
    realtime ratio is exact (2:3) and the piper down-conversion is a
    clean band-limited source (speech), both well within its error bar.
    """
    if src_rate == dst_rate or not src:
        return src
    n = len(src) // 2
    samples = array.array("h")
    samples.frombytes(src[: n * 2])
    out_n = int(n * dst_rate / src_rate)
    out = array.array("h")
    step = src_rate / dst_rate  # source samples per output sample
    for i in range(out_n):
        pos = i * step
        lo = int(pos)
        frac = pos - lo
        hi = lo + 1 if lo + 1 < n else lo
        out.append(int(samples[lo] * (1.0 - frac) + samples[hi] * frac))
    return out.tobytes()


# ---------------------------------------------------------- argparse setup

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="openbot-voice",
        description=(
            "OpenBot v4 voice app / director: builds the lane the loaded "
            "config's brain.lane demands (realtime WebSocket cloud lane, "
            "transcript REST cloud lane or offline regex lane) and runs "
            "the full session stack."
        ),
    )
    parser.add_argument(
        "--config",
        default=CONFIG_DEFAULT,
        help="config file (default: %(default)s, from the repo root)",
    )
    parser.add_argument(
        "--profile",
        default=None,
        choices=("laptop", "cloud", "voice", "offline"),
        help="config profile merged on top of --config (default: the "
        "config's own profile)",
    )
    parser.add_argument(
        "--audio",
        choices=("g1", "laptop", "mock"),
        default=DEFAULT_AUDIO,
        help="audio transport: g1 = robot multicast mic + PlayStream "
        "speaker; laptop = sounddevice; mock = MockMic + MockSink + "
        "scripted turns (default: %(default)s)",
    )
    parser.add_argument(
        "--directed-mode",
        action="store_true",
        help="KWS wake/sleep words gate input; without it the energy "
        "gate alone drives sessions",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one turn (offline: one utterance answered; realtime: "
        "one turn_complete) then exit",
    )
    parser.add_argument(
        "--script",
        default=None,
        metavar="FILE",
        help="mock-script JSON ({kws: [...], transcripts: [...]}) "
        "overriding the default scripted turns (--audio mock only)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="log level (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-mock-body",
        action="store_true",
        help="acknowledge a SIMULATED body (gateway.transport 'mock'): "
        "pick/place/open/close report success WITHOUT the robot moving. "
        "Required to boot a mock body with a live mic, so a box operator "
        "can never get scripted 'DONE' replies by accident",
    )
    parser.add_argument(
        "--skip-gateway-probe",
        action="store_true",
        help="skip the boot-time TCP probe of the real gateway; the app "
        "will start even with the gateway down (and be blind until it is up)",
    )
    parser.add_argument(
        "--version", action="version", version="openbot-voice (OpenBot v4 L2)"
    )
    return parser.parse_args(argv)


def _setup_logging(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# ------------------------------------------------- building the brain stack

def _guarded_observe(backend: Any) -> Any:
    """ObservationSupplier for the hooks (never raises; None = no
    source)."""
    if backend is None:
        return None
    try:
        return backend.observe()
    except Exception:  # noqa: BLE001 — a gateway hiccup is no source
        return None


def _freshness_s(cfg: Any, embodiment: Any) -> float:
    """detection_freshness_s with the config.yaml parity convention:
    the embodiment's perception.freshness_s is the fallback authority
    (G1Agent.from_config does the same)."""
    seg = cfg.get("agent") or {}
    if isinstance(seg, dict) and seg.get("detection_freshness_s"):
        return float(seg["detection_freshness_s"])
    perception = getattr(embodiment, "perception", None)
    if perception is not None:
        return float(getattr(perception, "freshness_s", 10.0))
    return 10.0


def build_brain_stack(cfg: Any) -> Dict[str, Any]:
    """Build the shared brain stack (BOTH lanes) exactly the way
    G1Agent.from_config does:

        embodiment = EmbodimentConfig.from_section(cfg['embodiment'])
        backend    = gateway.brain_client.backend_from_config(cfg)
        registry   = ToolRegistry(backend=backend, embodiment=embodiment)
        scene_graph= SceneGraph(conf_floor=embodiment.perception.det_conf_floor)
        hooks      = default_hooks(observation_source=guarded observe,
                                   confirm=voice_confirm, envelope=embodiment,
                                   freshness_s=...)
        executor   = ToolExecutor(registry=registry.snapshot(), hooks=hooks,
                                   scene_graph=..., backend=...,
                                   embodiment=..., failure_budget=...)

    plus the voice lane's per-utterance budget reset (mirrors the fresh
    executor per run() that G1Agent uses — a voice session is a stream
    of utterances, each with its own budget).

    Returns a dict stack with: backend (the gateway backend, wrapped by
    question_backend), registry, scene_graph, hooks, executor, slot,
    confirm, question_backend, and — for the cloud lanes (realtime /
    transcript) — session (RealtimeBackend / TranscriptBackend); for
    the offline lane — brain and chain.  Pure: no I/O, no sockets, no
    box access.  Failure raises ConfigError/ValueError — main() turns
    it into rc != 0 + loud stderr.
    """
    from brain.agent.optional_tools import switch_on
    from brain.agent.tool_executor import ToolExecutor
    from brain.agent.tools import ToolRegistry
    from brain.config_loader import EmbodimentConfig
    from brain.hooks import default_hooks
    from brain.scene_graph import SceneGraph
    from gateway.brain_client import backend_from_config

    lane = cfg.brain.lane
    if lane not in ("realtime", "transcript", "offline"):
        raise ValueError(
            "config error: voice app supports brain.lane 'realtime' "
            "(cloud Realtime WebSocket), 'transcript' (REST "
            "chat-completions) or 'offline' (regex brain, no network), "
            "got {!r}".format(lane)
        )

    # -- mirror the G1Agent.from_config brain stack; every lane reads the
    # optional_tools switches
    embodiment = EmbodimentConfig.from_section(cfg.get("embodiment"))
    backend = backend_from_config(cfg)  # gateway.transport selects mock or real
    registry = ToolRegistry(backend=backend, embodiment=embodiment,
                            switches=cfg.get("optional_tools"))
    # detection-confidence floor: this lane builds its own graph, so plumb
    # it in here or low-confidence detections count as referents
    scene_graph = SceneGraph(
        conf_floor=embodiment.perception.det_conf_floor,
        # missing means off: bind to the freshest detection
        target_disambiguation=switch_on(cfg.get("optional_tools"),
                                        "target_disambiguation"),
    )
    # scene_brief/resolve_ref look the graph up at call time, so this late
    # wiring is enough; the same seam G1Agent.from_config uses
    registry.attach(scene_graph=scene_graph)

    slot = UtteranceSlot()
    # the runner installs the spoken path later, once the audio lane exists;
    # without it the confirm gate denies
    _speak_ref: List[Optional[Callable[[str], None]]] = [None]

    def _confirm_speak(text: str) -> None:
        fn = _speak_ref[0]
        if fn is not None:
            try:
                fn(text)
            except Exception:  # noqa: BLE001 — a dead TTS denies, never crashes
                log.exception("confirm round could not speak: %s", text)

    confirm = make_voice_confirm(slot, _confirm_speak)
    # confirm switch: brain.confirm_motion false removes the spoken round
    # it is a hooks/brain concern, so it lives in the 'brain' section;
    # absent means true, so a config that omits it keeps the pause
    _brain_seg = cfg.get("brain") or {}
    _require_confirm = True
    if isinstance(_brain_seg, dict) and "confirm_motion" in _brain_seg:
        _require_confirm = bool(_brain_seg.get("confirm_motion"))
    if not _require_confirm:
        log.info(
            "confirm gate DISABLED by config (brain.confirm_motion: false) — "
            "motion tools will execute without asking; the hook records each "
            "one as APPROVED UNASKED (no human confirmed)"
        )
    hooks = default_hooks(
        observation_source=lambda: _guarded_observe(backend),
        confirm=confirm,
        envelope=embodiment,
        freshness_s=_freshness_s(cfg, embodiment),
        require_confirm=_require_confirm,
        # missing means off: pass the hint verbatim
        hint_charset_refusal=switch_on(cfg.get("optional_tools"),
                                       "hint_charset_refusal"),
    )
    seg = cfg.get("agent") or {}
    failure_budget = (
        int(seg["failure_budget"]) if isinstance(seg, dict) and seg.get("failure_budget") else 3
    )
    executor = ToolExecutor(
        registry=registry.snapshot(),
        hooks=hooks,
        scene_graph=scene_graph,
        backend=backend,
        embodiment=embodiment,
        failure_budget=failure_budget,
    )

    # -- the question-speak path wraps the same backend -------------------
    question_backend = OfflineQuestionBackend(backend)
    registry.attach(backend=question_backend)
    executor.backend = question_backend  # observe and friends go through here

    stack: Dict[str, Any] = {
        "lane": lane,
        "embodiment": embodiment,
        "registry": registry,
        "scene_graph": scene_graph,
        "hooks": hooks,
        "executor": executor,
        "backend": question_backend,
        # raw gateway backend, not the wrapper: the camera seam needs
        # .capture_image(), which lives on the gateway client and is wired
        # in by _attach_frame_hook
        "gateway_backend": backend,
        "slot": slot,
        "confirm": confirm,
        "question_backend": question_backend,
        "speak_ref": _speak_ref,
    }

    if lane == "realtime":
        # -- wiring up the cloud lane ---------------------------------------
        from voice.adapters import flatten_specs
        from voice.realtime import RealtimeClient
        from voice.session import RealtimeBackend

        api_key = os.environ.get(cfg.realtime.api_key_env, "")
        if not api_key:
            raise ValueError(
                "config error: realtime lane needs the env var {!r} set "
                "(the API key is never in config files); pass --profile "
                "offline for the no-network lane".format(cfg.realtime.api_key_env)
            )
        client = RealtimeClient(
            base_url=cfg.realtime.base_url,
            model=cfg.realtime.model,
            api_key=api_key,
            tools=flatten_specs(registry.get_specs()),
            # the model's own identity, sent as session.update instructions;
            # "" means absent
            instructions=str(cfg.realtime.instructions),
            session_updated_timeout_s=float(cfg.realtime.connect_timeout_s),
            session_cap_s=float(cfg.realtime.session_cap_min * 60.0),
        )
        stack["session"] = RealtimeBackend(client)

    if lane == "transcript":
        # -- wiring the REST chat-completions lane ----------------------------
        # same BrainBackend surface the realtime lane drives; the api key is
        # read from the env var named by cfg.rest.api_key_env, never the file
        # tools: nested registry specs, never tools=[] (a malformed spec fails
        # loudly); STT/TTS mirror the offline cascade boundaries, and piper is
        # an absolute binary path rather than a PATH lookup
        from brain.agent.llm_cloud import CloudLLMClient
        from voice.cascade import PiperTTS, WhisperSubmit
        from voice.transcript import TranscriptBackend

        api_key = os.environ.get(cfg.rest.api_key_env, "")
        if not api_key:
            raise ValueError(
                "config error: transcript lane needs the env var {!r} set "
                "(the API key is never in config files); pass --profile "
                "offline for the no-network lane".format(cfg.rest.api_key_env)
            )
        client = CloudLLMClient(
            base_url=cfg.rest.base_url,
            model=cfg.rest.model,
            api_key_env=cfg.rest.api_key_env,
            tool_mode=cfg.rest.tool_mode,
            timeout_s=float(cfg.rest.timeout_s),
            prose_fallback=bool(cfg.rest.prose_fallback),
            max_tokens=int(cfg.rest.max_tokens),
        )
        stack["session"] = TranscriptBackend(
            client,
            tools=registry.get_specs(),
            submit=WhisperSubmit(
                url=cfg.voice.whisper_url, model=cfg.voice.whisper_model
            ),
            tts=PiperTTS(
                bin_path=cfg.voice.piper_bin, voice=cfg.voice.piper_voice
            ),
        )

    # -- the offline cascade, needed by every lane ---------------------------
    # the director owns lane authority: the cascade runs while it says
    # OFFLINE_CASCADE, configured or reached on a hard failure; construction
    # is lazy (no socket, no process)
    _build_offline_cascade(stack, cfg)
    return stack


def _build_offline_cascade(stack: Dict[str, Any], cfg: Any) -> None:
    """Build the offline cascade (brain + chain) on the SHARED brain
    stack and store it as stack['brain'] / stack['chain'].

    Built for BOTH lanes: the Director runs the cascade whenever it says
    OFFLINE_CASCADE, whether the process was configured offline or the
    realtime lane fell back.  Pure — WhisperSubmit/PiperTTS construct
    lazily and perform no I/O here (the mock audio lane swaps both
    boundaries later, keeping the cascade/brain/executor/gateway real)."""
    from brain.agent.offline_brain import OfflineBrain, STOP_TALKING_RE
    from voice.cascade import OfflineChain, PiperTTS, WhisperSubmit

    brain = OfflineBrain(
        executor=stack["executor"],
        embodiment=stack["embodiment"],
        name=cfg.voice.name,  # the VoiceConfig name
    )
    chain = OfflineChain(
        brain=brain,
        submit=WhisperSubmit(url=cfg.voice.whisper_url, model=cfg.voice.whisper_model),
        tts=PiperTTS(bin_path=cfg.voice.piper_bin, voice=cfg.voice.piper_voice),
        # the confirm round must fit inside the brain stage's bound
        brain_timeout_s=BRAIN_TIMEOUT_S + CONFIRM_ROUND_TIMEOUT_S,
    )
    stack["brain"] = brain
    stack["chain"] = chain


def _reset_utterance_budget(executor: Any) -> None:
    """Per-utterance budget reset (mirrors G1Agent.run()'s fresh
    executor): a voice session is a stream of utterances, each with its
    own failure budget."""
    executor._failure_budget_left = executor.failure_budget
    executor._terminal_text = None


# ---------------------------------------------------- mock bits and pieces

class MockWhisper:
    """Scripted whisper submit: one transcript per call, in order; '' when
    the script is exhausted (the chain speaks the canned decline, L2)."""

    def __init__(self, script: Optional[List[str]] = None) -> None:
        self.script = [str(s) for s in (script or [])]
        self.calls = 0

    def __call__(self, pcm16: bytes) -> str:  # noqa: ARG001 — scripted
        if not self.script:
            self.calls += 1
            return ""
        self.calls += 1
        return self.script.pop(0)


class MockPiper:
    """Scripted piper TTS: short zero-peak PCM (silence — the echo guard
    stands down, so scripted answers are never suppressed)."""

    def __init__(self, seconds: float = 0.1, output_rate: int = 22050) -> None:
        self.seconds = seconds
        self.output_rate = output_rate
        self.calls = 0

    def __call__(self, text: str) -> Tuple[bytes, int]:  # noqa: ARG001
        self.calls += 1
        n = max(1, int(self.seconds * self.output_rate))
        return (b"\x00\x00" * n, self.output_rate)


def _load_mock_script(path: Optional[str]) -> Dict[str, Any]:
    """The mock script (--script FILE, else the default).  Validates the
    shape: kws = per-utterance KWS answers, transcripts = whisper
    transcripts (one per non-wake/sleep utterance, in order), gap_s =
    extra quiet seconds between turns (optional, >= 0).  gap_s paces a
    test battery: a real answer's latency (measured 3-12 s) can exceed
    the default cadence (~4.7 s), and a tone landing mid-answer is a
    genuine barge-in the battery must never induce."""
    if path is None:
        return json.loads(json.dumps(DEFAULT_MOCK_SCRIPT))  # deep copy, don't share it
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("mock script must be a JSON object")
    kws = raw.get("kws", [])
    transcripts = raw.get("transcripts", [])
    if not isinstance(kws, list) or not isinstance(transcripts, list):
        raise ValueError("mock script kws/transcripts must be JSON lists")
    gap_s = raw.get("gap_s", 0.0)
    if type(gap_s) is not float and type(gap_s) is not int:
        raise ValueError("mock script gap_s must be a number (seconds)")
    if not 0.0 <= gap_s <= 3600.0:
        raise ValueError("mock script gap_s out of range: {}".format(gap_s))
    return {
        "kws": list(kws),
        "transcripts": [str(t) for t in transcripts],
        "gap_s": float(gap_s),
    }


def _mock_loud_frames(voice: Any) -> List[bytes]:
    """Loud 440 Hz tone frames (one turn's worth) for the mock script —
    the energy gate needs a peak above min_peak to open."""
    from voice.mic import MockMic

    mic = MockMic.sine(
        seconds=MOCK_LOUD_S,
        frequency=MOCK_TONE_HZ,
        amplitude=MOCK_TONE_AMP,
        sample_rate=int(getattr(voice, "sample_rate", G1_SAMPLE_RATE)),
        frame_bytes=G1_FRAME_BYTES,
    )
    return list(mic.iter_frames())


def _make_mock_frames(
    script: Dict[str, List[Any]], directed: bool, voice: Any
) -> List[bytes]:
    """Scripted G1-cadence frames: one energy-gated utterance per turn.

    Each turn = loud tone frames (open the gate) + quiet chunks
    (hangover closes it -> the utterance emits) + quiet chunks covering
    the gate's wall-clock cooldown, so the next turn's tone starts past
    the cooldown, + the script's gap_s quiet chunks (outpacing the
    wire's answer latency: a tone landing while a real answer is still
    rendering is a genuine barge-in — a battery must run slow, never
    fast).  The pump paces itself at the real cadence (mock only)."""
    turns = len(script["kws"]) if directed else len(script["transcripts"])
    frame_bytes = int(getattr(voice, "sample_rate", G1_SAMPLE_RATE) * 2 * 0.16)
    frame_bytes = max(1, frame_bytes)
    samples_per_frame = frame_bytes // 2
    frame_s = samples_per_frame / float(
        getattr(voice, "sample_rate", G1_SAMPLE_RATE)
    )
    bytes_per_s = samples_per_frame * 2 / frame_s
    hangover_chunks = (
        int(getattr(voice, "energy_hangover_s", 0.9) * bytes_per_s)
        // frame_bytes
        + 1
    )
    cooldown_chunks = max(
        1, int(getattr(voice, "energy_cooldown_s", 3.0) / frame_s) + 1
    )
    gap_chunks = max(0, int(round(float(script.get("gap_s", 0.0)) / frame_s)))
    quiet = b"\x00\x00" * samples_per_frame
    loud = _mock_loud_frames(voice)
    loud_per_turn = min(len(loud), max(1, int(round(MOCK_LOUD_S / frame_s))))
    out: List[bytes] = []
    # boot warm-up: quiet chunks, not loud, so the gate's noise floor stays
    # near 0 and min_peak wins; a loud first chunk would become the floor and
    # lift the threshold above the utterance
    out.extend([quiet] * MOCK_WARMUP_CHUNKS)
    for _ in range(max(1, turns)):
        turn = [quiet] * loud_per_turn
        turn.extend([quiet] * hangover_chunks)
        turn.extend([quiet] * cooldown_chunks)
        turn.extend([quiet] * gap_chunks)
        for i in range(loud_per_turn):
            turn[i] = loud[i % len(loud)]
        out.extend(turn)
    return out


# -------------------------------------------------------------- the runner

class Runner:
    """The running app: audio lane + transcript record + play path.

    Built AFTER the brain stack, so a config/backend failure never
    constructs a mic/speaker (no box state changes on failure)."""

    def __init__(
        self,
        cfg: Any,
        args: argparse.Namespace,
        stack: Dict[str, Any],
        script: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self.cfg = cfg
        self.args = args
        self.stack = stack
        self.script = script
        self.transcript: List[Dict[str, Any]] = []
        self.error_flag = False
        self.stop = threading.Event()
        self.play_lock = threading.Lock()
        self.level = PlayingLevel()
        # mic-liveness watchdog state: _last_frame_t is stamped per frame, so
        # a frame age past _stall_limit_s means a dead stream; _last_warn_t
        # 0.0 means never warned, later throttled to MIC_STALL_WARN_EVERY_S
        self._last_frame_t = time.monotonic()
        self._last_warn_t = 0.0
        self._stall_limit_s = MIC_STALL_LIMIT_S
        #: mute window end (monotonic); the pump drops frames while the clock is below it
        self._muted_until = 0.0
        #: the listen latch: input only while true
        #: a wake phrase latches it on with no expiry and 'turn off' latches
        #: it off until the next summon; boot state off, quiet until summoned
        self._listening = False
        self.mic: Any = None
        self.sink: Any = None
        self.kws: Any = None
        #: local stop spotter (realtime only; None = off)
        self.stop_spotter: Any = None
        self._build_audio()

    # -- setting up the audio lane -------------------------------------------

    def _build_audio(self) -> None:
        cfg = self.cfg
        args = self.args
        voice = cfg.voice
        self.directed = bool(args.directed_mode)

        if args.audio == "mock":
            from voice.keyword_spotting import MockKws
            from voice.mic import MockMic
            from voice.speaker import MockSink

            assert self.script is not None, "mock runner needs a script"
            self.sink = MockSink()
            frames = _make_mock_frames(self.script, self.directed, voice)
            self.mic = MockMic(frames, frame_bytes=G1_FRAME_BYTES)
            # the wake phrases come from the config's list, the sleep list
            # stays the module default; the pump reads kws.phrases and
            # kws.sleep_phrases on every lane
            self.kws = MockKws(
                script=self.script["kws"],
                phrases=list(getattr(voice, "wake_phrases", None) or ()),
            )
            chain = self.stack.get("chain")
            if chain is not None:
                # swap only ASR/TTS; cascade, brain, executor, hooks and
                # gateway stay real
                chain.submit = MockWhisper(self.script["transcripts"])
                chain.tts = MockPiper()
                chain.sink = self.sink  # the LED signal has no effect in mock mode
            from voice.transcript import TranscriptBackend
            backend = self.stack.get("session")
            if isinstance(backend, TranscriptBackend):
                # the transcript lane keeps real boundaries, but under
                # --audio mock it gets the same mocks: never a live whisper
                # endpoint or a real piper binary
                backend._submit = MockWhisper(self.script["transcripts"])
                backend._tts = MockPiper()
            self.mock = True
        elif args.audio == "laptop":
            from voice.mic import SounddeviceMic
            from voice.speaker import SounddeviceSink

            self.sink = SounddeviceSink(sample_rate=G1_SAMPLE_RATE)
            self.mic = SounddeviceMic(sample_rate=G1_SAMPLE_RATE)
            self._maybe_real_kws(voice)
            self.mock = False
        else:  # g1: robot multicast mic + PlayStream speaker
            from voice.mic import G1Mic
            from voice.speaker import G1Speaker

            self.sink = G1Speaker(
                app_name=voice.speaker_app_name,
                sample_rate=G1_SAMPLE_RATE,
                cap_bytes=int(voice.speaker_cap_bytes),
            )
            self.mic = G1Mic(
                group=voice.multicast_group,
                port=int(voice.multicast_port),
            )
            self._maybe_real_kws(voice)
            self.mock = False

        self.energy = self._build_energy_gate(voice)
        from voice.echo_guard import EchoGuard

        self.echo = EchoGuard(coupling=float(voice.echo_coupling))

    def _maybe_real_kws(self, voice: Any) -> None:
        # always built for a real audio lane: the wake gate is the summon
        # contract, not a --directed-mode option; consuming the wake utterance
        # instead would drop the summon and go deaf on a false 'sleep'
        from voice.keyword_spotting import StewardKws

        self.kws = StewardKws(
            model_dir=str(voice.kws_model_dir),
            phrases=list(voice.wake_phrases),
            cooldown_s=float(voice.kws_cooldown_s),
        )

    @staticmethod
    def _build_energy_gate(voice: Any) -> Any:
        from voice.energy_gate import EnergyGate

        return EnergyGate(
            sample_rate=int(voice.sample_rate),
            min_peak=int(voice.energy_min_peak),
            floor_factor=float(voice.energy_floor_factor),
            prefix_ms=int(voice.energy_prefix_s * 1000.0),
            hangover_ms=int(voice.energy_hangover_s * 1000.0),
            cooldown_s=float(voice.energy_cooldown_s),
        )

    # -- watching the mic for a dead stream ------------------------------------

    def _mic_watchdog(self) -> None:
        """Dead stream must never look like silence (Fix B).

        The pumps stamp :attr:`_last_frame_t` on every frame; when no
        frame arrived for ``_stall_limit_s`` the stream is dead (the
        Explore-app voice method toggled off, the robot powered down) —
        G1Mic then parks in its 2 s recv timeout forever, which a
        silent-app idle must never pass as a quiet room.  Non-once:
        loud ERROR every MIC_STALL_WARN_EVERY_S — multicast membership
        persists, so toggling the setting back ON recovers mid-run
        without a restart.  --once: bounded abort (rc=1) via the
        error_flag/stop gates, exactly like a pump failure.  Mock audio
        is exempt: a scripted pump ends by design and must never look
        stalled.
        """
        if self.args.audio == "mock":
            return
        age = time.monotonic() - self._last_frame_t
        if age <= self._stall_limit_s:
            return
        if self.args.once:
            self.set_error(
                "mic stall {:.0f}s: no 5555 frames (Explore app 'Voice "
                "interaction method' OFF or the robot not powered)".format(age)
            )
            return
        if time.monotonic() - self._last_warn_t >= MIC_STALL_WARN_EVERY_S:
            self._last_warn_t = time.monotonic()
            log.error("MIC STALL %.0fs: no 5555 frames (Explore app voice "
                      "method = WAKE UP MODE? robot powered?)", age)

    def set_error(self, message: str) -> None:
        """Loud app-level failure: error log + transcript event + the
        error_flag/stop gates (the bounded --once abort path)."""
        log.error("%s", message)
        self.record({"event": "mic_stall", "error": message})
        self.error_flag = True
        self.stop.set()

    # -- the play and speak path -----------------------------------------------

    def play_pcm(self, pcm: bytes, rate: int) -> None:
        """The ONE play path: playing-level tracking (echo guard) +
        piper 22050 Hz -> G1 16 kHz contract resample + sink, all under
        the play lock (the pump's whisper decline can race the chain
        worker's reply)."""
        if not pcm:
            return
        with self.play_lock:
            self.level.set_playing(pcm)
            pcm16 = _resample_pcm16(pcm, int(rate), G1_SAMPLE_RATE)
            try:
                self.sink.play(pcm16)
            except Exception:  # noqa: BLE001 — a dead speaker must not kill
                log.exception("sink.play failed")
                self.error_flag = True

    def is_muted(self) -> bool:
        """True while the stop-talking mute window is open: the pump
        drops completed utterances SILENTLY (no whisper, no speech — the
        small-noise decline loop cannot re-raise the prompt)."""
        return time.monotonic() < self._muted_until

