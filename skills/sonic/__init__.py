"""skills.sonic — the SONIC Kinematic Planner / tracker lane (ADDITIVE).

This package is the single source of truth for the SONIC stack used by the
OpenBot merged writer lane. It is deliberately import-light:

  sonic_core      -- constants + pure-numpy helpers. numpy ONLY. No mujoco, no
                     onnxruntime at import time (onnxruntime is imported lazily
                     inside SonicStack.__init__).
  sonic_runtime   -- stateful, clock-injected control law. No DDS.
  sonic_satellite -- the RT process that owns its own clock and lowstate
                     subscription and talks to the writer over a unix socket.

NOTHING in this package may be imported by an rt/ process at 250 Hz. The writer
imports ONLY sonic_core, and only for static tables (see the guarded import in
skills/writer/main_merged_writer.py).

No side effects on import.
"""
