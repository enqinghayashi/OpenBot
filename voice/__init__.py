"""voice: the OpenBot v4 speech protocol layer.

Architecture (mic -> brain -> speaker), left to right:

    G1 mic multicast  --5120 B frames-->  EnergyGate  --utterance-->  StewardKws
        |                                                    (wake / sleep phrases)
        |                                        |                          |
        |                                    wake fired                  asleep
        |                                        |                          |
        |                          +-------------+--------------+           v
        |                          |                            |      (sleep: stop)
        |                          v                            v
        |                 [cloud lane]                    [offline lane]
        |              RealtimeClient wss                whisper.cpp ASR
        |              (OpenAI Realtime GA)              -> OfflineBrain -> Piper TTS
        |              (brain.lane 'realtime')           (brain.lane 'offline')
        |                          |
        |              [transcript lane]
        |              REST chat-completions (voice/transcript.py,
        |              brain.lane 'transcript' — CloudLLMClient family)
        |                          |
        +---- EchoGuard <---------+--------- speaker level ----+
              (level bar: mic must beat coupling x playing level)

    PlayStream  <-  G1Speaker / SounddeviceSink / MockSink

Key measured box facts (OpenBot protocol reference, devices/g1/mic.py and
devices/g1/speaker.py, plus audio/speech_gate.py):

* Microphone: UDP multicast group 239.168.123.161:5555 (source
  192.168.123.161, the audio board); 5120-byte datagrams every 160 ms,
  exactly real time; PCM16LE / 16 kHz / mono, already beam-formed to one
  channel.  No on-board ASR is ever used: ``/audio_msg`` is never read
  (measured unusable in the OpenBot reference).
* Speaker: unitree_sdk2py ``AudioClient.PlayStream(app_name, stream_id,
  chunk)`` with app_name ``uwa_csse_voice``, PCM16 / 16 kHz / mono, chunks
  capped at 96000 bytes (3 s) per call - the only size known to be safe.
  ``PlayStop`` discards its result and always returns 0: it is a no-op in
  effect, never a reliable stop.
* Energy: measured on the G1, fan-only peaks 836-1122 vs speech peaks
  2700-14674 (B2 re-measure gate).

Module-level import surface is stdlib-only (socket/array/math/struct/
threading/json/queue/logging/asyncio); every third-party dependency
(websockets, vosk, sounddevice, unitree_sdk2py) is imported lazily inside
the class methods that need it, so this package imports and instantiates
its mocks on any laptop.
"""

__all__ = [
    "cascade",
    "session",
    "mic",
    "keyword_spotting",
    "speaker",
    "energy_gate",
    "echo_guard",
    "realtime",
    "transcript",
    "adapters",
    "app",
]
