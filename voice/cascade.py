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


class OfflineChainError(RuntimeError):
    """One stage of the offline chain died.  ``stage`` names the dead
    stage: 'whisper' | 'brain' | 'piper'.  This is the ONLY exception the
    chain raises — a caller never receives a fake empty success when
    speech was impossible."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__("[{}] {}".format(stage, message))
        self.stage = stage


# ---------------------------------------------------------------------------
# what we hand back
# ---------------------------------------------------------------------------


@dataclass
class ChainOutcome:
    """One chain run's result.

    Deliberately NO success field (L5): the chain never reports success —
    the executor's verdicts reach the human only through the brain's
    verdict-aware reply, and the chain only annotates.  ``pcm`` is always
    real audio when an outcome is RETURNED (a piper failure raises
    OfflineChainError instead of returning); ``source`` names which stage
    produced the reply: 'brain' (an OfflineOutcome reply, decline or
    verdict-aware motion phrasing) | 'whisper_decline' (whisper stage
    failed — L2) | 'whisper_empty' (whisper heard nothing — L2) |
    'silent' (the brain's speech-stop lane, e.g. 'stop talking': pcm is
    EMPTY and nothing is spoken — the app's play path no-ops on it).
    """

    reply_text: str
    pcm: bytes
    sample_rate: int
    source: str
    brain_outcome: Optional[OfflineOutcome] = None


# ---------------------------------------------------------------------------
# submit boundary
# ---------------------------------------------------------------------------


def _wav_bytes(pcm16: bytes, sample_rate: int) -> bytes:
    """A complete PCM16 mono WAV (44-byte RIFF header + samples) — the
    shape the whisper.cpp /inference ``file`` field expects; the server
    infers the format from the header."""
    data_len = len(pcm16)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_len, b"WAVE",
        b"fmt ", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b"data", data_len,
    )
    return header + pcm16


def _silence_wav(sample_rate: int) -> bytes:
    """A ~0.1 s silence WAV — the tiny probe payload (any HTTP status
    means 'server up'; transport errors mean 'not up')."""
    samples = max(1600, sample_rate // 10)
    return _wav_bytes(b"\x00\x00" * samples, sample_rate)


def _decode_payload(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8", "replace"))
    except ValueError as exc:
        raise WhisperError("whisper /inference returned non-JSON: {}".format(exc))


class WhisperSubmit:
    """whisper.cpp ASR submit callable (the chain's default ``submit``).

    Keeps a whisper.cpp server answering on ``url``: lazily Popen'd on
    first use, or reused when one is already answering (probed first —
    an operator/deploy-kit-started server is never shadowed by a second
    process).  ``submit(pcm16)`` POSTs the utterance as a WAV and
    returns the transcript ('' = nothing heard).  All of this is lazy —
    the constructor performs no I/O and whisper.cpp is never imported.

    Construction is lazy and idempotent exactly like StewardKws' engine
    build (voice/kws.py pattern); failures raise WhisperError, which the
    chain turns into the spoken decline (L2).
    """

    def __init__(
        self,
        url: str = WHISPER_URL,
        model: str = "tiny.en-q5_1",  # 2026-09-04 pick, measured 1.52 s
        model_path: Optional[str] = None,
        bin_path: str = "whisper-server",
        sample_rate: int = 16000,
        launch_timeout_s: float = WHISPER_LAUNCH_TIMEOUT_S,
        submit_timeout_s: float = WHISPER_SUBMIT_TIMEOUT_S,
    ) -> None:
        """Pure state capture — NO I/O here (the constructor I/O-free
        invariant, pinned by test_constructor_defaults_and_boundaries):
        probing, ``model_path``/``bin_path`` resolution and the Popen all
        happen lazily on the first ``submit``, never in ``__init__``.

        ``url``: the whisper.cpp /inference endpoint to keep answering
        (probed before any launch; an operator-started server is reused).
        ``model``: the model name used to build the layout fallback
        ``voice/models/ggml-<model>.bin``.  ``model_path``: explicit
        model file — wins when given; ``None`` falls back at LAUNCH time
        to that layout file ONLY when it exists.  ``bin_path``: the
        whisper-server binary, resolved at LAUNCH time — a path
        containing a separator is used as-is; a BARE name first tries the
        cwd-relative deploy layout ``voice/models/<bin_path>`` when that
        file exists AND is executable, falling back to the bare name (on
        PATH) only when it does not.  ``sample_rate``: the PCM16 rate of
        submitted utterances.  ``launch_timeout_s`` / ``submit_timeout_s``:
        the L1 bounds for the launch window and one /inference POST."""
        self.url = url
        self.model = model
        self.model_path = model_path
        self.bin_path = bin_path
        self.sample_rate = sample_rate
        self.launch_timeout_s = launch_timeout_s
        self.submit_timeout_s = submit_timeout_s
        self._lock = threading.Lock()
        self._proc = None  # type: Optional[subprocess.Popen]

    def __call__(self, pcm16: bytes) -> str:
        """Transcribe one utterance (16 kHz mono PCM16) to text; '' when
        the server heard nothing (the chain then speaks the canned
        decline, L2)."""
        if not pcm16:
            return ""
        self._ensure_server()
        code, payload = self._post(
            _wav_bytes(pcm16, self.sample_rate), self.submit_timeout_s
        )
        if code != 200:
            raise WhisperError(
                "whisper /inference returned HTTP {}: {}".format(
                    code, payload[:300]
                )
            )
        data = _decode_payload(payload)
        text = data.get("text") if isinstance(data, dict) else None
        if text is None:
            raise WhisperError(
                "whisper /inference payload has no 'text': {!r}".format(
                    str(data)[:200]
                )
            )
        return str(text).strip()

    # ---------------------------------------------------- keeping the server alive

    def _ensure_server(self) -> None:
        """Make sure a whisper.cpp server answers on ``url`` (lazy, once):
        reuse one already answering, else Popen one and wait (bounded by
        ``launch_timeout_s``), else a loud WhisperError.  All under one
        lock; idempotent and cheap after the first call."""
        with self._lock:
            proc = self._proc
            if proc is not None and proc.poll() is None:
                return  # still up
            if self._probe_repeat(min(2.0, self.launch_timeout_s)):
                log.debug("offline chain: whisper server already answering on %s", self.url)
                return
            model = self._resolve_model()
            host = urlparse(self.url).hostname or "127.0.0.1"
            port = urlparse(self.url).port or 8082
            bin_path = self._resolve_bin()  # launch time only
            try:
                proc = subprocess.Popen(
                    [bin_path, "-m", model, "--host", host, "--port", str(port)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                raise WhisperError(
                    "cannot launch the whisper.cpp server ({!r}): {}".format(
                        bin_path, exc
                    )
                )
            self._proc = proc
            log.info(
                "offline chain: launched whisper.cpp server (pid %s, %s:%s, model %s)",
                proc.pid, host, port, model,
            )
            started = time.monotonic()
            deadline = started + self.launch_timeout_s
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = self._drain_stderr(proc)
                    raise WhisperError(
                        "whisper.cpp server exited {} at startup: {}".format(
                            proc.returncode, tail or "no stderr"
                        )
                    )
                if self._probe():
                    log.info(
                        "offline chain: whisper.cpp server ready in %.1fs",
                        time.monotonic() - started,
                    )
                    return
                time.sleep(0.25)
            # launch window expired: kill and reap the child
            # (bounded, like close()), drop self._proc so
            # no later submit trusts it; next one relaunches
            self._proc = None
            try:
                proc.kill()
                proc.wait(timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.debug(
                    "whisper server teardown after launch timeout raised: %s", exc
                )
            raise WhisperError(
                "whisper.cpp server did not accept connections within {} s "
                "(pid {} killed)".format(self.launch_timeout_s, proc.pid)
            )

    def _resolve_model(self) -> str:
        """The whisper.cpp model file the server is launched with.
        ``model_path`` wins; None falls back to the standard deploy
        layout ``voice/models/ggml-<model>.bin`` ONLY when that file
        exists (a check, never a guess); otherwise an explicit error."""
        if self.model_path is not None:
            return self.model_path
        candidate = os.path.join("voice", "models", "ggml-" + self.model + ".bin")
        if os.path.isfile(candidate):
            return candidate
        raise WhisperError(
            "no whisper.cpp model to launch: pass model_path= or place the "
            "model at {} (model={!r})".format(candidate, self.model)
        )

    def _resolve_bin(self) -> str:
        """The whisper.cpp ``whisper-server`` binary the server is
        launched with.  ``bin_path`` containing a path separator is used
        as-is (an explicit path always wins); a BARE name first tries the
        cwd-relative deploy layout ``voice/models/<bin_path>`` when that
        file exists AND is executable (os.path.isfile + os.access X_OK —
        an existence check, never a guess), falling back to the bare name
        (resolved on PATH) only when it does not.  Resolution happens at
        LAUNCH time only — the constructor performs no I/O — so both
        launch forms (``run_voice.sh`` and a bare ``python3 voice/app.py``,
        cwd = the deploy root) exec the layout copy deterministically."""
        if os.sep in self.bin_path:
            return self.bin_path
        candidate = os.path.join("voice", "models", self.bin_path)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        return self.bin_path

    def _probe(self) -> bool:
        """One probe POST: any HTTP status means the server is up;
        only a transport failure (refused/timeout) means it is not."""
        try:
            _code, _payload = self._post(
                _silence_wav(self.sample_rate), min(2.0, self.submit_timeout_s)
            )
            return True
        except WhisperError:
            return False

    def _probe_repeat(self, budget_s: float) -> bool:
        """Probe until one answers or ``budget_s`` elapses (bounded)."""
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            if self._probe():
                return True
            time.sleep(0.25)
        return False

    def _drain_stderr(self, proc: subprocess.Popen) -> str:
        """Best-effort stderr tail of a dead server (diagnostics only)."""
        try:
            _out, err = proc.communicate(timeout=2.0)
        except Exception:  # noqa: BLE001 - best-effort diagnostics
            return ""
        return (err or b"")[-300:].decode("utf-8", "replace").strip()

    def close(self) -> None:
        """Kill the server process THIS instance launched (never raises;
        an operator-started server is left alone)."""
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5.0)
            except Exception as exc:  # noqa: BLE001 - teardown must not raise
                log.debug("whisper server teardown raised: %s", exc)

    # -------------------------------------------------------- how the request goes out

    def _post(self, wav: bytes, timeout_s: float) -> Tuple[int, bytes]:
        """POST one WAV to /inference (multipart, response_format=json);
        returns (status, body).  Raises WhisperError only on TRANSPORT
        failures (refused/timeout) — any HTTP status comes back to the
        caller, so the probe can tell 'server up' (any status) from
        'server gone' (transport error)."""
        boundary = "----openbot-cascade-{:x}".format(int(time.monotonic() * 1e6))
        body = bytearray()
        body += b"--" + boundary.encode() + b"\r\n"
        body += b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        body += b"Content-Type: audio/wav\r\n\r\n"
        body += wav
        body += b"\r\n"
        for name, value in (("response_format", "json"), ("temperature", "0.0")):
            body += b"--" + boundary.encode() + b"\r\n"
            body += b'Content-Disposition: form-data; name="' + name.encode() + b'"\r\n\r\n'
            body += value.encode() + b"\r\n"
        body += b"--" + boundary.encode() + b"--\r\n"
        request = urllib.request.Request(
            self.url,
            data=bytes(body),
            headers={
                "Content-Type": "multipart/form-data; boundary={}".format(boundary)
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise WhisperError("whisper /inference request failed: {}".format(exc))


# ---------------------------------------------------------------------------
# piper TTS default
# ---------------------------------------------------------------------------


class PiperTTS:
    """Piper synthesis callable (the chain's default ``tts``): one
    subprocess per synthesis (``piper --model <voice> --output_raw`` —
    the reply text on stdin, raw PCM16 mono on stdout).  Piper is never
    imported — always a subprocess.  Returns (pcm, sample_rate); the
    rate is the VOICE MODEL's native rate and the sink owns resampling
    to the G1's 16 kHz (documented in the module docstring).

    ONE timeout bounds the whole stage — spawn + model load + synthesis
    together, a single ``communicate(timeout=...)`` — so there is no
    unbounded wait (L1).  Failures raise PiperError; the chain maps them
    onto the L4 retry law.
    """

    def __init__(
        self,
        bin_path: str = "piper",
        voice: str = "en_US-ryan-medium.onnx",
        output_rate: int = PIPER_OUTPUT_RATE,
        tts_timeout_s: float = PIPER_TTS_TIMEOUT_S,
    ) -> None:
        self.bin_path = bin_path
        self.voice = voice
        self.output_rate = output_rate
        self.tts_timeout_s = tts_timeout_s

    def __call__(self, text: str) -> Tuple[bytes, int]:
        """Synthesize one reply; (pcm, sample_rate) or PiperError."""
        try:
            proc = subprocess.Popen(
                [self.bin_path, "--model", self.voice, "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise PiperError(
                "cannot launch piper ({!r}): {}".format(self.bin_path, exc)
            )
        try:
            out, err = proc.communicate(
                input=(text or "").encode("utf-8"), timeout=self.tts_timeout_s
            )
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001 - teardown best effort
                pass
            raise PiperError(
                "piper synthesis exceeded {} s; process killed".format(
                    self.tts_timeout_s
                )
            )
        except OSError as exc:
            raise PiperError("piper pipe broke: {}".format(exc))
        if proc.returncode != 0:
            tail = (err or b"")[-500:].decode("utf-8", "replace").strip()
            raise PiperError(
                "piper exited {}: {}".format(proc.returncode, tail or "no stderr")
            )
        if not out:
            raise PiperError(
                "piper produced no audio for {!r}".format((text or "")[:60])
            )
        return out, self.output_rate


# ---------------------------------------------------------------------------
# runs the show
# ---------------------------------------------------------------------------


class OfflineChain:
    """The offline lane orchestration (plan §3 cascade + §5b offline
    lane): whisper submit -> OfflineBrain.new_utterance(text,
    typed=False) -> piper.

    All boundaries are injectable: ``submit`` (default: a
    :class:`WhisperSubmit`; any callable ``(pcm16: bytes) -> str``),
    ``brain`` (REQUIRED — anything with ``new_utterance(text, *,
    typed=False) -> OfflineOutcome``; the app builds the real
    OfflineBrain over its ToolExecutor/hooks), ``tts`` (default: a
    :class:`PiperTTS`; any callable ``(text: str) -> (pcm, sample_rate)``),
    ``sink`` (optional; used ONLY for the L4 LED signal).

    Entry points: :meth:`run_pipeline` (the full audio chain) and
    :meth:`handle_utterance` (an already-transcribed utterance).  See the
    module docstring for the laws (L1 timeouts, L2 whisper decline,
    L3 verdict flow-through, L4 piper retry, L5 no success reporting).
    """

    def __init__(
        self,
        brain: Any,
        *,
        submit: Optional[Callable[[bytes], str]] = None,
        tts: Optional[Callable[[str], Tuple[bytes, int]]] = None,
        sink: Any = None,
        whisper_decline: str = WHISPER_DECLINE,
        brain_timeout_s: float = BRAIN_TIMEOUT_S,
        whisper_launch_timeout_s: float = WHISPER_LAUNCH_TIMEOUT_S,
        whisper_submit_timeout_s: float = WHISPER_SUBMIT_TIMEOUT_S,
        piper_tts_timeout_s: float = PIPER_TTS_TIMEOUT_S,
    ) -> None:
        """Validate the four L1 stage timeouts and wire the boundaries.
        Defaults are the documented production constants (L1): brain
        ``BRAIN_TIMEOUT_S`` (30 s), whisper launch
        ``WHISPER_LAUNCH_TIMEOUT_S`` (30 s), whisper submit
        ``WHISPER_SUBMIT_TIMEOUT_S`` (15 s), piper
        ``PIPER_TTS_TIMEOUT_S`` (20 s)."""
        if brain is None or not callable(getattr(brain, "new_utterance", None)):
            raise ValueError(
                "OfflineChain requires a brain with "
                "new_utterance(text, *, typed=False) -> OfflineOutcome"
            )
        bad_timeouts = [
            name
            for name, value in (
                ("brain_timeout_s", brain_timeout_s),
                ("whisper_launch_timeout_s", whisper_launch_timeout_s),
                ("whisper_submit_timeout_s", whisper_submit_timeout_s),
                ("piper_tts_timeout_s", piper_tts_timeout_s),
            )
            if not value or value <= 0
        ]
        if bad_timeouts:
            raise ValueError(
                "stage timeouts must be positive (L1: no unbounded waits): "
                + ", ".join(bad_timeouts)
            )
        if whisper_decline is None or not str(whisper_decline).strip():
            raise ValueError("whisper_decline must be a non-empty string")
        self.brain = brain
        self.sink = sink
        self.whisper_decline = whisper_decline
        self.brain_timeout_s = brain_timeout_s
        self.submit = (
            submit
            if submit is not None
            else WhisperSubmit(
                launch_timeout_s=whisper_launch_timeout_s,
                submit_timeout_s=whisper_submit_timeout_s,
            )
        )
        self.tts = (
            tts if tts is not None else PiperTTS(tts_timeout_s=piper_tts_timeout_s)
        )

    # ------------------------------------------------------- the entry points

    def run_pipeline(self, pcm16: bytes) -> ChainOutcome:
        """The full audio chain on one utterance (16 kHz mono PCM16):
        whisper -> brain -> piper.  A whisper failure or an empty
        transcript speaks the canned decline with NO brain call (L2);
        anything else flows through handle_utterance."""
        try:
            text = self.submit(pcm16)
        except OfflineChainError:
            raise  # already a chain failure
        except Exception as exc:  # noqa: BLE001 - any whisper failure
            log.warning("offline chain: whisper stage failed: %s", exc)
            return self._decline(self.whisper_decline, "whisper_decline")
        if not str(text or "").strip():
            return self._decline(self.whisper_decline, "whisper_empty")
        return self.handle_utterance(text)

    def handle_utterance(self, text: str) -> ChainOutcome:
        """Brain + piper on an already-transcribed utterance.  The reply
        spoken is the OfflineOutcome's verdict-aware text, byte-for-byte
        (L3) — the chain never rewrites it."""
        outcome = self._run_brain(text)
        if outcome.silent:
            # speech-stop = no speech
            # ('stop talking')
            # _speak never returns empty PCM, so a
            # silent outcome skips piper; the app's
            # play_pcm no-ops on an empty buffer
            return ChainOutcome(
                reply_text="",
                pcm=b"",
                sample_rate=PIPER_OUTPUT_RATE,
                source="silent",
                brain_outcome=outcome,
            )
        pcm, rate = self._speak(outcome.reply)
        return ChainOutcome(
            reply_text=outcome.reply,
            pcm=pcm,
            sample_rate=rate,
            source="brain",
            brain_outcome=outcome,
        )

    # -------------------------------------------------------------- the stages

