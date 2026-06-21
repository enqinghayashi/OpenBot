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

    def is_listening(self) -> bool:
        """True while the listen LATCH is ON: the lane takes input.

        The latch has NO expiry (operator directive 2026-09-20) — once
        'hey stewart' latches it on, the lane STAYS awake, and once
        'turn off' latches it off it STAYS asleep, until the other
        command.  Read by both pumps' wake gate."""
        return self._listening

    def open_listen_window(self, seconds: Optional[float] = None) -> None:
        """Wake: latch the lane ON (no expiry) and clear the stop-talking
        mute — an explicit summon is the user asking to talk.

        ``seconds`` is accepted and IGNORED: it is the vestige of the
        retired ``LISTEN_WINDOW_S`` deadline, kept in the signature so
        every existing call site works unchanged.  A caller that passes
        one gets a debug line, never a silent behaviour change."""
        if seconds is not None:
            log.debug("open_listen_window: ignoring seconds=%r (latch)", seconds)
        self._listening = True
        self._muted_until = 0.0

    def close_listen_window(self) -> None:
        """'turn off': latch the lane OFF until the next summon, and cut
        current speech NOW (operator: *"turn off turns it off so it
        stopped talking"*).

        This IS a state that sticks — deliberately, and it is the whole
        point of the operator's contract (*"only 'hello stuwart' wakes
        it up again without no window limitation"*).  Recovery is the
        wake phrase, which the KWS hears even while latched off (the
        wake path runs BEFORE the pump's listen gate)."""
        self._listening = False
        # mute_s=0.0: cut the sink with no mute window; the latch is the real
        # off switch, and a 20 s mute on top would keep the lane deaf
        self.stop_talking(0.0)

    def stop_talking(self, mute_s: float = STOP_TALKING_MUTE_S) -> None:
        """The 'stop talking' command's effect: cut current speech NOW
        (the sink discards its paced queue and ends the stream), open
        the echo guard's playing window, and mute the pump for
        ``mute_s``.  Safe to call from the pump thread (play_lock is the
        speaker path's own — sink.stop is the G1Speaker's own lock)."""
        self.level.set_playing(b"")
        try:
            self.sink.stop()
        except Exception:  # noqa: BLE001 — a dead speaker must never break the mute
            log.exception("sink.stop failed")
        self._muted_until = time.monotonic() + mute_s

    def wait_speech_drained(self, timeout_s: float) -> bool:
        """Block until the sink has PLAYED everything queued, bounded.

        Exists so 'turn off' can be audible: its acknowledgement would
        otherwise be discarded a few milliseconds after being queued by
        ``close_listen_window`` -> ``stop_talking(0.0)``.

        Only the G1 sink exposes a drain (speaker.py:327).  The laptop
        and mock sinks are synchronous — their ``play`` returns once the
        audio is out — so a missing method means there is nothing left
        to wait for, NOT a failure.  Returns True when the queue is
        empty (or unknowable), False only on a real timeout/error; the
        caller must treat both the same way, because the latch is the
        contract and the ack is a courtesy.
        """
        drain = getattr(self.sink, "wait_until_drained", None)
        if drain is None:
            return True
        try:
            return bool(drain(timeout_s))
        except Exception:  # noqa: BLE001 — a dead sink must not block the latch
            log.exception("wait_until_drained failed")
            return False

    def speak(self, text: str) -> None:
        """The confirm/question spoken path (local TTS -> sink; the model
        is never involved).  Every spoken line lands in the transcript."""
        self.record({"event": "spoken", "text": str(text)})
        self.note_spoken(text)
        tts = self._tts()
        try:
            pcm, rate = tts(text)
        except Exception:  # noqa: BLE001 — a dead TTS denies, never crashes
            log.exception("tts failed for %r", text)
            return
        self.play_pcm(pcm, rate)

    def note_spoken(self, text: Any) -> None:
        """B8j-hf19: tell the stop spotter what the robot is about to say
        (its own 'stopped' / 'cancelled' must never read as the operator's
        stop).  Never raises."""
        sp = self.stop_spotter
        if sp is None or not text:
            return
        try:
            sp.note_spoken(str(text))
        except Exception:  # noqa: BLE001
            log.exception("stop spotter note_spoken failed")

    def _tts(self) -> Callable[[str], Tuple[bytes, int]]:
        chain = self.stack.get("chain")
        if chain is not None:
            return chain.tts  # offline lane's TTS (real or mock)
        from voice.cascade import PiperTTS

        return PiperTTS(
            bin_path=str(self.cfg.voice.piper_bin),
            voice=str(self.cfg.voice.piper_voice),
        )

    # -- who plays the cloud reply audio ----------------------------------------

    def _on_audio_delta(self, name: str, payload: dict) -> None:  # noqa: ARG001
        """The session's ``audio`` events, played (Fix A).

        Both cloud lanes release reply audio as session ``audio`` events
        — base64 PCM16 24 kHz chunks (session.py:1426/1443: the realtime
        wire's response.output_audio.delta AND the transcript lane's
        piper re-emit, transcript.py _speak).  Nothing consumed them, so
        a cloud reply was never played and PlayingLevel never armed —
        the silent-robot defect.  Playing here funnels through the ONE
        play path (:meth:`play_pcm` — it sets PlayingLevel BEFORE
        sink.play, so EchoGuard admits nothing the robot is saying) with
        the same resampler/rate the chain-worker reply uses (1222/1595);
        the session already holds the 0.6 s lead + 60 s cap (L6), so
        every delta is real reply audio.
        """
        raw = str(payload.get("delta") or payload.get("data") or "")
        if not raw:
            return
        try:
            pcm24 = base64.b64decode(raw)
        except Exception:  # noqa: BLE001 — a bad chunk must not kill the lane
            log.error("audio: bad delta (base64) %d bytes", len(raw))
            return
        self.play_pcm(pcm24, 24000)

    def _on_spoken_text(self, name: str, payload: dict) -> None:  # noqa: ARG001
        """Cloud-lane text ledger (audio_transcript): the reply text the
        model is speaking lands in the transcript record like the local
        speak path's ``spoken`` entries — one entry per FINAL spoken
        event (session.py:1450 fires partial deltas with final=False;
        :1458 the final transcript).  No audio duty."""
        if not payload.get("final"):
            return
        text = str(payload.get("text") or payload.get("transcript") or "")
        if not text:
            return
        self.record({"event": "spoken", "text": text})
        log.info("cloud spoken: %r", text)

    # -- whisper and transcript handling -----------------------------------------

    def whisper(self, utt: bytes) -> str:
        """Transcribe one utterance (the pump owns the whisper stage;
        the chain receives TEXT via handle_utterance).

        Returns the transcript ('' = nothing heard).  A whisper FAILURE
        is recorded as a whisper_error event; the PUMP speaks the cascade
        WHISPER_DECLINE for both the failure and the empty transcript (L2
        — the app owns the whisper stage, so the decline the chain would
        have spoken is the app's to speak; never a silent skip)."""
        chain = self.stack.get("chain")
        submit = chain.submit if chain is not None else None
        if submit is None:
            return ""
        try:
            return str(submit(utt) or "")
        except Exception as exc:  # noqa: BLE001 — a whisper failure is a decline
            log.warning("whisper failed: %s", exc)
            self.record({"event": "whisper_error", "error": str(exc)})
            return ""

    def _speak_decline(self) -> None:
        from voice.cascade import WHISPER_DECLINE

        self.speak(WHISPER_DECLINE)
        self.record({"event": "whisper_decline", "reply": WHISPER_DECLINE})

    def record(self, entry: Dict[str, Any]) -> None:
        self.transcript.append(entry)

    # -- capturing an answer ---------------------------------------------------

    def capture(self, timeout_s: float) -> Optional[str]:
        return make_round_capture(self.stack["slot"])(timeout_s)

    def install_voice(self) -> None:
        """Wire the spoken paths the stack deferred: the confirm round's
        speak + the question lane's speak/capture.  Installed for BOTH
        lanes: the session's L9 intercept owns query_user while the
        realtime lane is live (the brain stack is not in use), and the
        cascade — which the director may run in either process — needs
        the spoken question path whenever it is the active lane."""
        self.stack["speak_ref"][0] = self.speak
        self.stack["question_backend"].set_voice(self.speak, self.capture)


# -------------------------------------------------------- the offline runner

# the KWS window and the ASR window are the same audio, so whisper returns
# the wake phrase at the head of the transcript; a captured vocative would
# defeat the offline grammar's fixed-phrase match, so one leading vocative
# is trimmed here before the chain
# the name is optional because those phonemes get mangled, and a bare summon
# trims to '' and drops silently: never answer your own wake word


def _wake_name_from_phrases(phrases: Optional[List[str]]) -> Optional[str]:
    """The embodied name from the first '<greeting> <name>' wake phrase.

    The KWS engine's own phrase list is the authority (voice.kws
    StewardKws).  Returns None for a phrase set no greeting head starts
    (a 'sleep'-only set), so the trim degrades to head-only stripping.
    """
    for p in phrases or ():
        m = re.match(r"^(?:hey|hi|hello)\s+(\S+)$", p, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _wake_names_from_phrases(phrases: Optional[List[str]]) -> List[str]:
    """EVERY embodied name the wake phrases carry, in first-seen order.

    A phrase set can hold more than one SPELLING of the name — the
    shipped 'stewart' plus the operator's 'stuwart' (directive
    2026-09-20) — and the wake-leak trim must strip whichever spelling
    the recogniser actually produced.

    ``_wake_name_from_phrases`` returns only the FIRST, which is correct
    for a single-spelling set but leaves every other spelling glued to
    the command: 'hello stuwart' would trim to 'stuwart' instead of '',
    so the bare summon would reach the brain as the command 'stuwart',
    match no lane, and be answered with the decline — the robot
    answering its own wake word, which is the exact failure this trim
    exists to prevent.
    """
    names: List[str] = []
    for p in phrases or ():
        m = re.match(r"^(?:hey|hi|hello)\s+(\S+)$", p, re.IGNORECASE)
        if m:
            name = m.group(1)
            if name not in names:
                names.append(name)
    # longest first: a shorter spelling that prefixes a longer one must not
    # win the alternation
    return sorted(names, key=len, reverse=True)


def make_wake_leak_re(name: Any) -> "re.Pattern[str]":
    """Build the leading wake-leak regex for the given embodied name(s).

    ``name`` is optional: whisper often drops the name of the wake
    phrase — 'Hey, steal it!' is the wake 'hey stewart' plus the command
    'steal it!' — so the head alone must strip.  Punctuation after the
    head/name is tolerated (whisper closes vocatives with commas and
    periods: 'Hello, Stewart.').

    ``name`` may be a single string, a LIST of accepted spellings (see
    ``_wake_names_from_phrases``), or None for a head-only pattern.  The
    string and None forms are unchanged — the list form only widens the
    alternation.
    """
    if name is None:
        names: List[str] = []
    elif isinstance(name, str):
        names = [name]
    else:
        names = [n for n in name if n]
    name_part = (
        "(?:" + "|".join(re.escape(n) for n in names) + ")" if names else ""
    )
    return re.compile(
        # whisper closes vocatives with any punctuation ('Hello, Stewart!'),
        # so the class is [,.!?] on both slots; a comma-period class would
        # leave the '!' glued to the head
        rf"^(?:hey|hi|hello)[,.!?]?\s*(?:{name_part})?[,.!?]?\s*",
        re.IGNORECASE,
    )


def trim_wake_leak(text: str, wake_re: "re.Pattern[str]") -> str:
    """Strip ONE leading wake/vocative from an asr transcript.

    Only text that STARTS 'hey/hi/hello' is touched; a name-head without
    a greeting ('stewart, ...') passes through — that form is layer 1
    (the brain's name-pinned strip).
    """
    m = wake_re.match(text)
    return text[m.end():] if m else text


def _build_stop_spotter(runner: "Runner", session: Any,
                        fire_stop: Callable[[str], None], audio_mode: str) -> Any:
    """B8j-hf19: build + start the local stop spotter, or return None.

    Shares the wake word's Vosk model (one model in memory).  Off for mock
    audio, without a real wake-word engine, or with OPENBOT_STOP_SPOTTER=0.
    Its actions are the existing ones: the motion stop is ``fire_stop`` (the
    same gateway stop the spoken route sends), 'stop talking' is
    ``runner.stop_talking``, and every hit first cuts the speech and drops
    the rest of the reply (``session.stop_playback_now``)."""
    from voice.stop_spotter import StopSpotter, make_vosk_recognizer, spotter_mode

    mode = spotter_mode()
    kws = runner.kws
    if mode == "off":
        log.info("stop spotter: OFF (OPENBOT_STOP_SPOTTER=0)")
        return None
    if audio_mode == "mock" or kws is None or not hasattr(kws, "_ensure_engine"):
        log.info("stop spotter: not built (no real wake-word engine to share)")
        return None

    def factory() -> Any:
        kws._ensure_engine()
        return make_vosk_recognizer(kws._model, G1_SAMPLE_RATE)

    def cut(text: str) -> None:  # noqa: ARG001
        runner.stop_talking(0.0)          # cut the sink, no mute window
        if session is not None and hasattr(session, "stop_playback_now"):
            session.stop_playback_now()

    def stop(text: str) -> None:
        cut(text)
        fire_stop(text)

    def stop_talking(text: str) -> None:
        cut(text)
        runner.stop_talking()
        runner.record({"event": "stop_talking", "text": text, "via": "stop_spotter"})

    def speaking() -> bool:
        probe = getattr(runner.sink, "is_speaking", None)
        if probe is not None:
            try:
                if probe():
                    return True
            except Exception:  # noqa: BLE001
                pass
        return runner.level.peak() > 0.0

    spotter = StopSpotter(factory, cut, stop, stop_talking, speaking,
                          mode=mode, record=runner.record)
    runner.stop_spotter = spotter
    if session is not None:
        session.on_event("spoken", lambda _n, p: runner.note_spoken(
            (p or {}).get("text") or (p or {}).get("transcript") or ""))
    spotter.start()
    log.info("stop spotter: ON (mode=%s, words=%s)", mode, "/".join(
        __import__("voice.stop_spotter", fromlist=["STOP_WORDS"]).STOP_WORDS))
    return spotter


def _start_arrival_announcer(runner: "Runner") -> None:
    """Start the arrival-announce listener thread when the config opts in.

    Reads the gate DEFENSIVELY: cfg.voice.arrival_announce (bool, a
    config_loader sibling lands it with code default False) may not exist
    yet — getattr defaults keep app.py safe before that edit lands.
    Returns None (a no-op) when the flag is off/missing; otherwise starts
    ONE daemon thread whose body reads arrival lines off the coordinator's
    AF_UNIX relay and speaks each one through runner.speak (thread-safe
    under play_lock; the "spoken" transcript entry it records is the
    desired audit trail).  When the relay is down the thread just retries
    its connect until runner.stop is set — safe on either lane.
    """
    voice_cfg = getattr(runner.cfg, "voice", None)
    if not bool(getattr(voice_cfg, "arrival_announce", False)):
        return None
    threading.Thread(
        target=arrival_announce.arrival_listener,
        args=(arrival_announce.SOCKET_PATH, runner.speak, runner.stop),
        daemon=True,
        name="arrival-announcer",
    ).start()
    return None


def _run_offline(
    cfg: Any,
    args: argparse.Namespace,
    stack: Dict[str, Any],
    script: Optional[Dict[str, List[Any]]] = None,
) -> int:
    from brain.agent.offline_brain import STOP_TALKING_RE, TURN_OFF_RE
    from voice.cascade import OfflineChainError
    from voice.director import OFFLINE_CASCADE, Director

    chain = stack["chain"]
    runner = Runner(cfg, args, stack, script)
    runner.install_voice()
    mic = runner.mic
    energy = runner.energy
    echo = runner.echo
    kws = runner.kws
    wake_leak_re = make_wake_leak_re(
        # every spelling, not just the first: the config can carry more than
        # one, and first-only would leave the other stuck on the command
        _wake_names_from_phrases(getattr(kws, "phrases", None))
    )
    slot = stack["slot"]
    work_q: "queue.Queue[Any]" = queue.Queue()
    frame_s = G1_FRAME_BYTES / 2 / float(cfg.voice.sample_rate)
    once_done = threading.Event()

    # -- the director, the only lane authority
    # 'offline' starts at OFFLINE_CASCADE and stays: no lane signal or tick
    # reaches this process and both exits need a tick, so no transition can
    # fire and the cascade runs the whole time
    chain_busy = False  # turn boundary (a reply is in flight)
    chain_lock = threading.Lock()

    def _boundary_open() -> bool:
        with chain_lock:
            return not chain_busy

    director = Director(
        backend=None,
        # no wire here, nothing to probe
        probe=lambda: False,
        # announcements via the app's speak path, not the brain
        speak=runner.speak,
        # nothing to reset: no session, no transitions
        reset=lambda: None,
        boundary_source=_boundary_open,
        cfg={"lane": "offline"},
    )

    def _feed_director(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            director.on_event(name, payload)
        except Exception:  # noqa: BLE001 — one bad event must not kill the lane
            log.exception("director event %r failed", name)

    def chain_worker() -> None:
        nonlocal chain_busy
        while not runner.stop.is_set():
            try:
                text = work_q.get(timeout=0.2)
            except Exception:  # noqa: BLE001 — empty after timeout
                continue
            if text is None:
                break
            with chain_lock:
                chain_busy = True
            _feed_director("gate_changed", {"open": False})  # the turn is opening
            try:
                _reset_utterance_budget(stack["executor"])
                outcome = chain.handle_utterance(text)
            except OfflineChainError as exc:
                log.error("offline chain failed (stage=%s): %s", exc.stage, exc)
                runner.record(
                    {"event": "chain_error", "stage": exc.stage, "error": str(exc)}
                )
            except Exception:  # noqa: BLE001 — one bad utterance must not kill
                log.exception("offline chain failed for %r", text)
                runner.record({"event": "chain_error", "error": "unexpected"})
            else:
                _pcm = outcome.pcm or b""
                log.info(
                    "brain reply: source=%s text=%r pcm=%dB",
                    outcome.source,
                    (outcome.reply_text or "")[:120],
                    len(_pcm),
                )
                runner.record(
                    {
                        "event": "reply",
                        "text": outcome.reply_text,
                        "source": outcome.source,
                        "pcm_bytes": len(_pcm),
                    }
                )
                runner.play_pcm(_pcm, outcome.sample_rate)
                log.info(
                    "playback queued: %d ms (%d Hz)",
                    len(_pcm) * 1000 // 2 // (outcome.sample_rate or 16000),
                    outcome.sample_rate or 16000,
                )
            with chain_lock:
                chain_busy = False
            _feed_director("turn_complete", {"lane": "offline"})  # boundary open again
            if args.once:
                once_done.set()  # turn done, answered or refused

    def pump_worker() -> None:
        try:
            for frame in mic.iter_frames():
                if runner.stop.is_set():
                    break
                # stamp the mic-liveness clock (mock too; the watchdog
                # skips mock audio anyway)
                runner._last_frame_t = time.monotonic()
                if args.audio == "mock":
                    time.sleep(frame_s)  # pace the script at the G1 cadence
                if not echo.admit(frame, runner.level.peak()):
                    continue
                utt = energy.feed(frame)
                if not utt:
                    continue
                # the wake gate: input only while the latch is on
                # a wake match latches it on and the utterance keeps going; a
                # sleep match latches it off and is consumed
                # no expiry, and the wake path runs before the gate, so the
                # summon is always heard even on a lane left listening
                if kws is not None:
                    try:
                        phrase = kws.feed(utt)
                    except Exception as exc:  # noqa: BLE001 — a dead KWS stops the lane
                        log.error("kws failed: %s", exc)
                        runner.error_flag = True
                        runner.stop.set()
                        return
                    if phrase is not None:
                        if phrase in kws.sleep_phrases:
                            if slot.round_active:
                                # a sleep phrase can be the confirm answer
                                # itself; closing the window here would swallow
                                # it and time the round out, so an active round
                                # owns the next utterance
                                log.info(
                                    "kws: sleep (%r) while confirm/query "
                                    "round active — forwarding as the answer",
                                    phrase,
                                )
                                runner.record(
                                    {
                                        "event": "kws",
                                        "action": "sleep_during_round",
                                        "phrase": phrase,
                                    }
                                )
                            else:
                                runner.close_listen_window()
                                log.info("kws: sleep (%r)", phrase)
                                runner.record(
                                    {"event": "kws", "action": "sleep", "phrase": phrase}
                                )
                                continue
                        if phrase in kws.phrases:
                            was_awake = runner.is_listening()
                            runner.open_listen_window()
                            log.info("kws: wake (%r)", phrase)
                            runner.record(
                                {"event": "kws", "action": "wake", "phrase": phrase}
                            )
                            # the summon is consumed, never whispered: the wake
                            # utterance is the same audio, so falling through
                            # would feed whisper's guess to the brain as a
                            # command and the wake-leak trim cannot catch it;
                            # greet instead
                            if slot.round_active:
                                # an active round owns that next utterance (the
                                # same law as the sleep branch); fall through so
                                # the answer reaches the slot
                                pass
                            else:
                                if not was_awake:
                                    # greet only on off -> on: a second 'hey
                                    # stewart' is a nudge, not a fresh arrival
                                    runner.speak(WAKE_GREETING)
                                continue
                    if not runner.is_listening() and not slot.round_active:
                        continue  # asleep: only a summon gets through
                if runner.is_muted():
                    # mute window: drop the utterance silently (no whisper,
                    # no decline, no speech) so the prompt loop cannot
                    # re-raise; answering resumes after the window
                    log.debug("muted: dropping completed utterance")
                    continue
                if director.state != OFFLINE_CASCADE:
                    # no cascade because the director left the offline lane;
                    # that cannot happen here, since no lane signal or tick is
                    # ever fed, so the state stays pinned
                    log.debug(
                        "director state %s — the cascade lane does not run",
                        director.state,
                    )
                    continue
                text = runner.whisper(utt)
                log.info("asr: %r", text)
                # leading wake-leak trim: the ASR window is the KWS window, so
                # the wake phrase leads the transcript and the grammar must
                # never inherit it; a bare summon trims to '' and drops
                # silently rather than answering its own wake word
                trimmed = trim_wake_leak(text, wake_leak_re)
                if trimmed != text:
                    log.debug("wake-leak trimmed: %r -> %r", text, trimmed)
                if text.strip() and not trimmed.strip():
                    log.info("summon-only: bare wake phrase, silent")
                    continue
                text = trimmed
                if TURN_OFF_RE.search(text):
                    # 'turn off': latch off (only a summon reopens) and cut
                    # speech now, checked before stop-talking because the
                    # families overlap ('shut down' vs 'shut up') and before
                    # slot delivery so a pending round times out instead
                    log.info(
                        "turn-off: acknowledging, then latching OFF (%r)",
                        text[:80],
                    )
                    runner.record({"event": "turn_off", "text": text})
                    # ack first, latch second, wait in between: flipping these
                    # two discards the ack and nothing is heard
                    runner.speak(TURN_OFF_ACK)
                    runner.wait_speech_drained(TURN_OFF_ACK_TIMEOUT_S)
                    runner.close_listen_window()
                    continue
                if STOP_TALKING_RE.search(text):
                    # 'stop talking': cut speech now and mute the pump; nothing
                    # is routed to the chain and there is no decline, else the
                    # prompt comes right back
                    log.info("stop-talking: cutting speech + muting 20 s (%r)", text[:80])
                    runner.stop_talking()
                    runner.record({"event": "stop_talking", "text": text})
                    continue
                if not text.strip():
                    # empty transcript: speak the decline, never skip silently;
                    # the app owns the whisper stage, so the decline is its to
                    # speak
                    log.info("empty transcript -> speaking decline")
                    runner._speak_decline()
                    continue
                runner.record({"event": "utterance", "text": text})
                if slot.deliver(text):
                    # a round is waiting: this utterance is the answer, never
                    # a command for the chain
                    runner.record({"event": "answer", "text": text})
                    continue
                work_q.put(text)
                if args.once:
                    # one utterance answered: stop after the reply
                    once_done.wait()
                    runner.stop.set()
        except Exception:  # noqa: BLE001 — a dead mic ends the lane loudly
            log.exception("mic pump failed")
            runner.error_flag = True
            runner.stop.set()
        finally:
            runner.stop.set()  # script ran out / the lane died

    worker = threading.Thread(target=chain_worker, daemon=True, name="offline-chain")
    pump = threading.Thread(target=pump_worker, daemon=True, name="mic-pump")
    worker.start()
    pump.start()
    # arrival announcements, opt-in via cfg.voice.arrival_announce
    _start_arrival_announcer(runner)

    try:
        while not runner.stop.is_set():
            # watchdog on the idle tick: a parked mic yields
            # nothing, so only this loop can spot the frame age
            runner._mic_watchdog()
            time.sleep(0.1)
    except KeyboardInterrupt:
        runner.stop.set()
        log.info("interrupted — closing cleanly")
    worker.join(timeout=2.0)
    try:
        runner.mic.close()
        runner.sink.close()
    except Exception:  # noqa: BLE001 — teardown must not raise
        log.exception("audio close failed")
    rc = 1 if runner.error_flag else 0
    if rc == 0 and args.once and not once_done.is_set():
        # same false-success gate as the realtime runner:
        # a --once that completed no turn is a failure
        _die("--once completed no turn: the offline lane never answered "
             "(0 events). Exit code 1 so a smoke run cannot score a dead "
             "lane as success.")
        rc = 1
    if args.audio == "mock":
        _print_transcript(runner.transcript)
    return rc


def _print_transcript(transcript: List[Dict[str, Any]]) -> None:
    """The mock run's transcript report (stdout — the bake-off / verify
    consumer reads it)."""
    print("TRANSCRIPT ({} events):".format(len(transcript)))
    for entry in transcript:
        event = entry.get("event", "?")
        if event == "utterance":
            print("  utterance: {!r}".format(entry.get("text")))
        elif event == "answer":
            print("  confirm/query answer: {!r}".format(entry.get("text")))
        elif event == "reply":
            print("  reply ({0}): {1!r}".format(entry.get("source"), entry.get("text")))
        elif event == "kws":
            print("  kws {0}: {1!r}".format(entry.get("action"), entry.get("phrase")))
        elif event == "spoken":
            print("  spoken: {!r}".format(entry.get("text")))
        elif event == "whisper_decline":
            print("  whisper decline: {!r}".format(entry.get("reply")))
        elif event == "whisper_error":
            print("  whisper error: {!r}".format(entry.get("error")))
        elif event == "chain_error":
            print(
                "  chain error (stage={0}): {1!r}".format(
                    entry.get("stage"), entry.get("error")
                )
            )
        else:
            print("  {0}: {1!r}".format(event, entry))


# -------------------------------------------------------- the realtime runner

def _attach_frame_hook(session: Any, stack: Dict[str, Any], cfg: Any) -> bool:
    """L10: wire the gateway camera into the session's per-turn frame seam.

    THE SEAM EXISTED ON BOTH SIDES AND WAS NEVER CONNECTED (found 2026-09-02).
    ``VoiceSession.set_frame_hook`` (voice/session.py:740) and the gateway's
    ``capture_image()`` (gateway/brain_client.py:286) both shipped, both were
    tested — but the ONLY caller of either was ``tests/``, which does not
    ship.  So on the box ``session._frame_hook`` was always None, every
    decision turn took the ``envelope is None`` branch, and the model was
    told "no frame source" on EVERY turn regardless of camera health.  The
    configs advertise ``vision.enabled: true`` (config.voice.yaml:84,
    config.cloud.yaml:89, config.offline.yaml:47) and the gateway really does
    serve the frame, so the operator had a vision-capable robot that answered
    "I can't see anything" forever.

    This is a PASS-THROUGH by design — every honesty rule already lives at
    one end or the other, and re-implementing any of them here would be a
    place for them to drift:

    * ``capture_image()`` returns None for a dead socket, an ok:false
      envelope, or malformed JSON — never a guess (its docstring is the
      contract: None = no camera / no usable frame).
    * The gateway stamps ``fresh`` itself against ``vision.max_age_s``, and
      ``_prepare_frame_for_turn`` attaches the image ONLY when
      ``fresh is True`` and ``jpeg_b64`` is present, otherwise telling the
      model it is blind.  A stale frame is therefore never passed off as live.
    * ``vision.min_interval_s`` is the GATEWAY's encode throttle
      (gateway_node.py:126), applied server-side on the camera worker.  The
      session already calls this hook at most once per decision turn
      (``_frame_attached_this_turn``), so throttling again here would only
      add a second, divergent clock.

    Returns True when the hook was attached, False when the camera lane is
    off or unavailable — the caller logs which, so "no frame" is always
    attributable to a named cause rather than silence.
    """
    vision = getattr(cfg, "vision", None)
    if not bool(getattr(vision, "enabled", False) if vision is not None else False):
        log.info("camera lane OFF (vision.enabled is not true) — the model is "
                 "told it is blind on every turn, by configuration")
        return False

    backend = stack.get("gateway_backend")
    capture = getattr(backend, "capture_image", None)
    if not callable(capture):
        # the mock backend has capture_image too (it returns None honestly),
        # so a missing verb means an older backend
        log.warning("camera lane ON in config but backend %s has no "
                    "capture_image() — the model stays blind",
                    type(backend).__name__)
        return False

    def _frame_hook() -> Optional[Dict[str, Any]]:
        """One gateway 'image' roundtrip per decision turn, or None."""
        try:
            return capture()
        except Exception:  # noqa: BLE001 — a dead camera must not kill the turn
            log.exception("camera capture failed — this turn is imageless")
            return None

    session.set_frame_hook(_frame_hook)
    log.info("camera lane ON — gateway 'image' verb wired to the session "
             "frame seam (topic %s, max_age %.1fs)",
             getattr(vision, "topic", "?"), getattr(vision, "max_age_s", -1.0))
    return True


def _run_realtime(
    cfg: Any,
    args: argparse.Namespace,
    stack: Dict[str, Any],
    script: Optional[Dict[str, List[Any]]] = None,
) -> int:
    """The cloud lane: asyncio loop + mic pump thread + director workers.

    The session owns the query_user two-phase (L9) and all turn laws;
    the app executes non-intercepted tool calls on a worker thread
    through the SAME ToolExecutor the offline lane uses (confirm gate +
    verifier apply) and answers via respond_tool_result.

    THE DIRECTOR is the lane authority (director.py — the only
    director).  It starts ONLINE (lane 'realtime') and is fed the
    session events (turn bookkeeping + failover + session_expired) plus
    the lane's own loud-exit signal; it announces transitions through
    the app's speak path.  Lane switches happen ONLY as director
    transitions at conversation boundaries — never mid-utterance: the
    pump reads director.state once per completed utterance and routes
    that utterance to the realtime wire (ONLINE/DEGRADING/REOPENING) or
    the offline cascade (OFFLINE_CASCADE), which is built for this
    process too.  KWS wake/sleep words stay INPUT GATING only.
    """
    import asyncio

    # the command grammar this lane's cascade branch searches; both patterns
    # must be imported here or the first cascade utterance raises NameError
    from brain.agent.offline_brain import STOP_TALKING_RE, TURN_OFF_RE
    # the motion halt law, imported rather than re-spelled: the offline lane
    # and the fastpath classifier scan these exact functions, so 'stop' means
    # one thing everywhere
    from brain.agent.fastpath import _halt_intent, _normalize_for_halt, _strip_heard_name
    from voice.cascade import OfflineChainError
    from voice.director import OFFLINE_CASCADE, Director
    from voice.session import VoiceSession

    runner = Runner(cfg, args, stack, script)
    runner.install_voice()
    session = VoiceSession(stack["session"])
    _attach_frame_hook(session, stack, cfg)
    slot = stack["slot"]
    chain = stack["chain"]
    work_q: "queue.Queue[Any]" = queue.Queue()
    frame_s = G1_FRAME_BYTES / 2 / float(cfg.voice.sample_rate)
    once_done = threading.Event()

    # -- the director, the only lane authority
    # backend=session gives the default reset; the boundary source is the
    # session's own half-duplex gate
    # the probe must match the lane's wire: sync health_probe() on the
    # transcript lane, the client's 'connected' flag on the realtime lane
    # probing the wrong one pins the lane in OFFLINE_CASCADE -> REOPENING
    # forever, announcing every ~8 s; a raising probe is logged, not raised
    chain_busy = False  # cascade turn boundary (a reply is in flight)
    chain_lock = threading.Lock()

    def _boundary_open() -> bool:
        if not session.gate_open:
            return False
        with chain_lock:
            return not chain_busy

    def _lane_probe() -> bool:
        backend = stack["session"]
        health = getattr(backend, "health_probe", None)
        if callable(health):
            return bool(health())
        return bool(getattr(getattr(backend, "_client", None), "connected", False))

    director = Director(
        backend=session,
        probe=_lane_probe,
        speak=runner.speak,  # announcements use the app's speak path
        boundary_source=_boundary_open,
        # the actual lane: 'realtime' or 'transcript', whichever this
        # process runs
        cfg={"lane": stack["lane"]},
    )

    def _feed_director(name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            director.on_event(name, payload)
        except Exception:  # noqa: BLE001 — one bad event must not kill the lane
            log.exception("director event %r failed", name)

    def _stop_intent(text: str) -> bool:
        """True when the utterance is an explicit MID-TASK HALT.

        The law is fastpath's, not a second regex written here.  Raw text
        is tested first so a bare 'stop' halts before any name handling;
        the vocative-stripped text is tested second so a LEADING NAME
        cannot push the halt word out of the scan window ('stewart, could
        you stop' -> 'could you stop' -> 'stop' is inside the first four
        tokens only after the strip).

        KNOWN LIMIT, deliberately not papered over: the window is the
        first four tokens, so a halt word buried deeper survives only if
        everything before it is one strippable name ('hey stewart, could
        you stop' still misses — 'hey' is not stripped).  Widening it is
        a change to the ONE halt law in fastpath, which every lane shares;
        it must not be widened here alone, or 'stop' starts meaning
        different things on different lanes.
        """
        if _halt_intent(_normalize_for_halt(text)):
            return True
        stripped = _strip_heard_name(text)
        return stripped != text and _halt_intent(_normalize_for_halt(stripped))

    def _fire_stop(text: str) -> None:
        """Deliver a mid-task stop to the gateway, OFF the dispatch thread.

        THE OPERATOR ASK (2026-10-01, verbatim): "i want the voice delivery
        too, such that i can tell the robot to stop doing what it is doing
        using natural language speaking, stop the search, stop what you
        looking for, stop picking up, stop placing, stop whatever you
        currently doing, 'cancel'".

        TWO BYPASSES, BOTH DELIBERATE:

        * NOT through ``stack['executor']``.  ``ToolExecutor`` documents
          that it is never re-entered (its ``_current_tool`` and budget
          bookkeeping are shared), and the pick we are interrupting is
          living inside it at this very moment — a second ``execute()``
          would corrupt that task's state instead of interrupting it.
        * NOT through the session's tool path.  ``session._on_function_call``
          REFUSES any call once tool results are flowing
          (voice/session.py:1173-1179), which is precisely the state a
          long pick is in.  That is why a spoken 'stop' could never have
          worked by asking the model to call the registered ``stop`` tool.

        The gateway is the right seam: it is thread-per-connection
        (gateway/gateway_node.py:4195-4210) and its cancel opens a
        SEPARATE engine connection (``cancel_engine`` -> ``_send_unix_verb``),
        so this lands while the pick is still moving rather than behind it.
        ``_do_cancel`` covers all three lanes in one call — the engine
        (pick/place), the walk FSM, and the search that rides it.  A stop
        that arrives when nothing is running is refused by the engine
        ("no task running") rather than armed for the NEXT task, so a late
        stop is a no-op, never a trap.
        """
        def work() -> None:
            try:
                out = stack["gateway_backend"].stop({})
                log.info("mid-task stop delivered (%r) -> %s",
                         text[:60], str(out)[:200])
                runner.record({"event": "stop", "text": text,
                               "result": str(out)[:200]})
            except Exception:  # noqa: BLE001 — never a silent zero
                log.exception("mid-task stop FAILED (%r)", text[:60])
                runner.record({"event": "stop", "text": text,
                               "error": "delivery failed"})

        threading.Thread(target=work, daemon=True,
                         name="mid-task-stop").start()

    def on_utterance(name: str, payload: dict) -> None:
        # serves both user_text (gate open) and dropped_utterance (gate shut)
        # the drop policy: a pending motion call keeps the gate shut, so the
        # confirm answer turns up here, consumed and never replayed
        # that is why a mid-task stop lives here: with a pick in flight, 'stop'
        # arrives as dropped_utterance, and this is the only subscription that
        # sees the text
        text = str(payload.get("text") or "")
        # 'turn off' on the session path: the KWS grammar is wake-only (the
        # default sleep phrase list is empty) and there is no local transcript
        # outside the cascade, so this is the only spot this lane can hear it;
        # latch off and cut speech, a pending round just times out
        if TURN_OFF_RE.search(text):
            log.info("turn-off: latching the lane OFF (%r)", text[:80])
            runner.close_listen_window()
            runner.record({"event": "turn_off", "text": text})
            return
        # order is load-bearing: 'stop talking' also begins with the halt
        # word 'stop', so checking the halt first would turn a request for
        # quiet into a motion halt; the speech stop is checked first
        if STOP_TALKING_RE.search(text):
            log.info("stop-talking: cutting speech (%r)", text[:80])
            runner.stop_talking()
            runner.record({"event": "stop_talking", "text": text})
            return
        if _stop_intent(text):
            _fire_stop(text)
            return
        # record the operator's words for the place-destination fallback and
        # the open_hand guard; the registry and the executor share the scene
        # graph, so this is the same self state the tools read
        # deliberately the narrow hook, not note_utterance, which also arms the
        # walk-only and task-done refusals; a failure never breaks delivery
        _ss = getattr(stack.get("scene_graph"), "self_state", None)
        if _ss is not None:
            try:
                _ss.note_operator_words(text)
            except Exception:  # noqa: BLE001 - the words record never blocks the turn
                log.exception("operator-words record failed (%r)", text[:60])
        slot.deliver(text)

    def on_tool_call(name: str, payload: dict) -> None:
        if payload.get("intercepted"):
            return  # intercepted: the session owns the two-phase answer
        call_id = str(payload.get("call_id") or "")
        tool_name = str(payload.get("name") or "")
        arguments = payload.get("arguments") or {}
        # one INFO line per tool call, with the arguments, so the place label
        # is readable; logging never blocks the call
        try:
            log.info("tool call %s %s", tool_name,
                     json.dumps(arguments, sort_keys=True, default=str)[:400])
        except Exception:  # noqa: BLE001 - logging never blocks a tool call
            log.info("tool call %s (arguments not printable)", tool_name)

        def work() -> None:
            try:
                _reset_utterance_budget(stack["executor"])
                outcome = stack["executor"].execute(tool_name, arguments)
                result_text = outcome.text
            except Exception as exc:  # noqa: BLE001 — never a silent zero
                log.exception("director tool %r failed", tool_name)
                result_text = "ERROR: director tool execution failed: {}".format(exc)
            session.respond_tool_result(call_id, result_text)

        threading.Thread(target=work, daemon=True, name="director-tool").start()

    session.on_event("user_text", on_utterance)
    session.on_event("dropped_utterance", on_utterance)
    session.on_event("tool_call", on_tool_call)
    # cloud reply audio: the session emits 'audio' events on both lanes (the
    # realtime wire's audio deltas and the transcript lane's piper re-emit)
    # plus the reply text as 'spoken'
    # play_pcm arms PlayingLevel before the sink, so the echo guard never
    # admits the robot's own voice
    session.on_event("audio", runner._on_audio_delta)
    session.on_event("spoken", runner._on_spoken_text)
    # always-on local stop: 'stop' / 'cancel' is heard even while the robot
    # talks
    _build_stop_spotter(runner, session, _fire_stop, args.audio)
    # director events straight off session events: turn bookkeeping keeps
    # the boundary, failover and session_expired are lane signals
    for _event in (
        "gate_changed", "tool_call", "query_waiting", "answer",
        "turn_complete", "failover", "session_expired",
    ):
        session.on_event(_event, _feed_director)

    if args.once:

        def _once(name: str, payload: dict) -> None:  # noqa: ARG001
            once_done.set()

        session.on_event("turn_complete", _once)

    def chain_worker() -> None:
        nonlocal chain_busy
        while not runner.stop.is_set():
            try:
                text = work_q.get(timeout=0.2)
            except Exception:  # noqa: BLE001 — empty after timeout
                continue
            if text is None:
                break
            with chain_lock:
                chain_busy = True
            _feed_director("gate_changed", {"open": False})  # the turn is opening
            try:
                _reset_utterance_budget(stack["executor"])
                outcome = chain.handle_utterance(text)
            except OfflineChainError as exc:
                log.error("offline chain failed (stage=%s): %s", exc.stage, exc)
                runner.record(
                    {"event": "chain_error", "stage": exc.stage, "error": str(exc)}
                )
            except Exception:  # noqa: BLE001 — one bad utterance must not kill
                log.exception("offline chain failed for %r", text)
                runner.record({"event": "chain_error", "error": "unexpected"})
            else:
                _pcm = outcome.pcm or b""
                log.info(
                    "brain reply: source=%s text=%r pcm=%dB",
                    outcome.source,
                    (outcome.reply_text or "")[:120],
                    len(_pcm),
                )
                runner.record(
                    {
                        "event": "reply",
                        "text": outcome.reply_text,
                        "source": outcome.source,
                        "pcm_bytes": len(_pcm),
                    }
                )
                runner.note_spoken(outcome.reply_text)  # the robot's own reply
                runner.play_pcm(_pcm, outcome.sample_rate)
                log.info(
                    "playback queued: %d ms (%d Hz)",
                    len(_pcm) * 1000 // 2 // (outcome.sample_rate or 16000),
                    outcome.sample_rate or 16000,
                )
            with chain_lock:
                chain_busy = False
            _feed_director("turn_complete", {"lane": "offline"})  # boundary open again
            if args.once:
                once_done.set()  # turn done, answered or refused

    def pump_worker() -> None:
        kws = runner.kws
        try:
            for frame in runner.mic.iter_frames():
                if runner.stop.is_set():
                    break
                # stamp the mic-liveness clock (mock too; the watchdog
                # skips mock audio anyway)
                runner._last_frame_t = time.monotonic()
                # every raw frame hits the stop spotter first, before the
                # echo guard or the mute law can drop it
                if runner.stop_spotter is not None:
                    runner.stop_spotter.submit(frame)
                if args.audio == "mock":
                    # pace the script at the G1 cadence: the gate's
                    # wall-clock cooldown needs real time
                    time.sleep(frame_s)
                if not runner.echo.admit(frame, runner.level.peak()):
                    continue
                # the energy gate drives sessions without --directed-mode:
                # only completed utterances reach the lane, then the server
                # VAD closes the turn
                utt = runner.energy.feed(frame)
                if not utt:
                    continue
                # the wake gate: input only while the latch is on
                # a wake match latches it on and the same utterance keeps
                # going; a sleep match latches it off and is eaten
                # no expiry: without a local whisper outside the cascade,
                # 'turn off' arrives through the session transcript instead
                if kws is not None:
                    try:
                        phrase = kws.feed(utt)
                    except Exception as exc:  # noqa: BLE001
                        log.error("kws failed: %s", exc)
                        runner.error_flag = True
                        runner.stop.set()
                        return
                    if phrase is not None:
                        if phrase in kws.sleep_phrases:
                            runner.close_listen_window()
                            log.info("kws: sleep (%r)", phrase)
                            runner.record(
                                {"event": "kws", "action": "sleep", "phrase": phrase}
                            )
                            continue
                        if phrase in kws.phrases:
                            runner.open_listen_window()
                            log.info("kws: wake (%r)", phrase)
                            runner.record(
                                {"event": "kws", "action": "wake", "phrase": phrase}
                            )
                    if not runner.is_listening():
                        continue  # asleep: only a summon gets through
                if runner.is_muted():
                    # mute window: drop the utterance silently (no whisper,
                    # no decline, no speech) so the prompt loop cannot
                    # re-raise; answering resumes after the window
                    log.debug("muted: dropping completed utterance")
                    continue
                if director.state == OFFLINE_CASCADE:
                    # the director says offline, so this runs the cascade;
                    # the state is read once per completed utterance, never
                    # mid-utterance
                    text = runner.whisper(utt)
                    if TURN_OFF_RE.search(text):
                        # 'turn off' on the cascade path, this lane's only
                        # local transcript: latch off, cut speech, nothing
                        # routed onward
                        log.info("turn-off: latching the lane OFF (%r)", text[:80])
                        runner.close_listen_window()
                        runner.record({"event": "turn_off", "text": text})
                        continue
                    if STOP_TALKING_RE.search(text):
                        # 'stop talking': cut speech now and mute the pump;
                        # nothing is routed to the chain and there is no
                        # decline, else the prompt comes right back
                        runner.stop_talking()
                        runner.record({"event": "stop_talking", "text": text})
                        continue
                    if not text.strip():
                        # empty transcript: speak the decline, never skip
                        # silently
                        runner._speak_decline()
                        continue
                    runner.record({"event": "utterance", "text": text})
                    if slot.deliver(text):
                        # a round is waiting: this utterance is the
                        # answer, never a command
                        runner.record({"event": "answer", "text": text})
                        continue
                    work_q.put(text)
                    if args.once:
                        once_done.wait()
                        runner.stop.set()
                    continue
                # realtime wire: resample 16 kHz up to 24 kHz
                chunk = _resample_pcm16(utt, G1_SAMPLE_RATE, 24000)
                session.upload_mic_audio(chunk)
        except Exception:  # noqa: BLE001 — a dead mic ends the lane loudly
            log.exception("mic pump failed")
            runner.error_flag = True
            runner.stop.set()
        finally:
            log.info("mic pump ended")
            runner.stop.set()

    def _lane_task_done(task: Any) -> None:
        """The lane's loud-exit signal: run_forever returning with an
        exception (auth error, retry exhaustion) is the director's
        documented 'hard' lane signal; a deliberate close/cancel feeds
        nothing."""
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            _feed_director("hard", {
                "detail": "{}: {}".format(type(exc).__name__, exc),
            })

    async def amain() -> None:
        session.start()
        lane_task = asyncio.ensure_future(stack["session"].run_forever())
        lane_task.add_done_callback(_lane_task_done)
        pump = threading.Thread(target=pump_worker, daemon=True, name="mic-pump")
        chain = threading.Thread(target=chain_worker, daemon=True, name="realtime-chain")
        pump.start()
        chain.start()
        # arrival announcements, opt-in via cfg.voice.arrival_announce
        _start_arrival_announcer(runner)
        try:
            while not runner.stop.is_set():
                if args.once and once_done.is_set():
                    break
                # watchdog on the idle tick: a parked mic yields
                # nothing, so only this loop can spot the frame age
                runner._mic_watchdog()
                # idle tick: the director's lazy timers evaluate here,
                # at a conversation boundary, never by wall time
                _feed_director("tick", {})
                await asyncio.sleep(0.2)
            # --once + mock: scripted input can finish before the lane has
            # answered, and a slow but alive wire needs longer than the pump
            # budget, so wait for turn_complete, the lane's terminal signal
            # (OFFLINE_CASCADE) or a horizon tied to the retry ladder
            # real audio never ends on its own, hence mock-only
            if (
                args.once
                and args.audio == "mock"
                and not once_done.is_set()
                and not runner.error_flag
            ):
                from voice.realtime import RETRY_LADDER_S

                deadline = time.monotonic() + float(sum(RETRY_LADDER_S))
                while not once_done.is_set() and not runner.error_flag:
                    if director.state == OFFLINE_CASCADE:
                        # the lane gave up by itself (retries exhausted or
                        # failover), so no answer is coming; the gate below
                        # decides the exit code
                        break
                    if time.monotonic() >= deadline:
                        break
                    _feed_director("tick", {})
                    await asyncio.sleep(0.2)
        except KeyboardInterrupt:
            runner.stop.set()
        finally:
            runner.stop.set()
            session.close()
            try:
                lane_task.cancel()
            except Exception:  # noqa: BLE001 — teardown must not raise
                pass
        try:
            runner.mic.close()
            runner.sink.close()
        except Exception:  # noqa: BLE001 — teardown must not raise
            log.exception("audio close failed")

    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        runner.stop.set()
    rc = 1 if runner.error_flag else 0
    if rc == 0 and args.once and not once_done.is_set():
        # false-success gate: --once promises exactly one turn. An
        # unreachable wire retries at WARNING and never sets error_flag (only
        # the mic-pump and KWS paths do), so teardown can return 0 after
        # printing an empty transcript and score a dead lane as success
        _die("--once completed no turn: the lane never answered (0 events). "
             "Nothing was heard or replied to — check the gateway, the "
             "network wire and the model endpoint. Exit code 1 so a smoke "
             "run cannot score a dead lane as success.")
        rc = 1
    if args.audio == "mock":
        _print_transcript(runner.transcript)
    return rc


# ------------------------------------------------ main and the boot checks

def _die(message: str) -> None:
    """Loud stderr (the main() failure contract: rc != 0 and a human can
    see why)."""
    print("ERROR: {}".format(message), file=sys.stderr)


def _embodiment_gate(cfg: Any, args: argparse.Namespace) -> Optional[str]:
    """Refuse to boot a configuration that would LIE about the robot.

    Two boot-time false successes, both found 2026-09-02 and both
    reproduced on this laptop:

    **(a) real brain + simulated body.**  ``config.yaml`` resolves profile
    'laptop' -> ``gateway.transport: mock`` -> ``MockGatewayBackend``, whose
    verbs return ``ok('DONE mock pick ...')``.  ``brain.lane`` still defaults
    to 'realtime', so the app dials the REAL (billed) cloud endpoint and then
    reports every motion as done while the arm never moves.  Nothing read
    ``cfg.profile`` after the load, so an operator running the documented
    ``python3 voice/app.py --audio g1 --directed-mode`` on the box (no
    ``--config``) got exactly this.  A mock body is now an explicit choice.

    **(b) real body, gateway down.**  ``GatewayCallerBackend`` opens no
    socket at construction — connections are per tool call — and the only
    boot probe checked the CLOUD backend, never the local gateway.  So the
    app booted rc 0, every preflight green, with the robot lane dead;
    scene questions then answered "I don't see anything" (an empty scene is
    indistinguishable from a dead pipeline downstream).  A bounded TCP
    probe makes that state loud at boot instead of silent forever.

    Returns None to proceed, or the refusal message.
    """
    gateway = getattr(cfg, "gateway", None)
    # normalise exactly as the backend factory does: .strip().lower()
    # a quoted value like `transport: " mock "` keeps its padding through the
    # YAML-subset parser, so .lower() alone matches neither "mock" nor "real"
    # and the gate falls through both branches while the factory builds a
    # mock; keep these two sites byte-identical
    transport = str(getattr(gateway, "transport", "mock") or "mock").strip().lower()

    if transport == "mock":
        # a scripted body is fine for scripted runs and explicit development,
        # but never silently in front of a live mic
        if args.audio == "mock" or args.allow_mock_body:
            log.warning(
                "SIMULATED BODY: gateway.transport is 'mock' — pick/place/"
                "open/close will report success WITHOUT moving the robot "
                "(profile %r)", getattr(cfg, "profile", "?"))
            return None
        return (
            "refusing to start: the resolved config has a SIMULATED body "
            "(gateway.transport 'mock', profile {!r}) but a live audio lane "
            "({!r}).\n"
            "         Motion commands would report success while the robot "
            "never moves.\n"
            "         On the robot pass --config config.voice.yaml (or "
            "config.offline.yaml).\n"
            "         To use a scripted body deliberately, pass "
            "--allow-mock-body.".format(
                getattr(cfg, "profile", "?"), args.audio)
        )

    if transport != "real":
        return None
    if args.skip_gateway_probe:
        log.warning("gateway probe skipped by --skip-gateway-probe; the app "
                    "may start blind if the gateway is not up")
        return None

    host = str(getattr(gateway, "host", "") or "")
    port = int(getattr(gateway, "port", 0) or 0)
    if not host or not port:
        return None  # backend_from_config already refuses placeholder hosts

    from gateway.brain_client import connect_with_timeout

    try:
        # one bounded resolve + connect: a raw connect would bound only the
        # connect, not the DNS lookup, so a dead resolver could stall the boot
        # probe past 2 s
        with connect_with_timeout(host, port, 2.0):
            pass
    except OSError as exc:
        return (
            "refusing to start: the gateway at {}:{} is not accepting "
            "connections ({}).\n"
            "         The robot lane would be DEAD while the app looked "
            "healthy — scene questions answer 'I don't see anything' and "
            "motion is refused.\n"
            "         Start the gateway node first, then relaunch.\n"
            "         To start anyway (it will be blind until the gateway "
            "is up), pass --skip-gateway-probe.".format(host, port, exc)
        )
    log.info("gateway reachable at %s:%d", host, port)
    return None


