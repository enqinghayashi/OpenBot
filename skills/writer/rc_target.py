"""rc_target.py — rc command from lowstate.wireless_remote.

Uses the SDK Joystick class verbatim (unitree_sdk2py/utils/joystick.py,
wireless_remote[2:4] buttons, [4:8] lx, [8:12] rx, [12:16] ry, [20:24] ly).
Mapping + scale live in config (rc_mapping/rc_scale_*); signs calibrated on
gantry Stage 2.

Safety: inactivity > rc_inactivity_timeout s => cmd forced to zero
(guaranteed stop — mirrors the RC-inactive rule from the working pipeline).
"""

import time

import numpy as np
from unitree_sdk2py.utils.joystick import Joystick

STICK_NAMES = ("lx", "ly", "rx", "ry")


class RCTarget:
    def __init__(self, cfg):
        self.joy = Joystick()
        self.mapping = cfg["rc_mapping"]   # {"vx": "-ly", "vy": "lx", "omega": "rx"}
        self.scale = {
            "vx": float(cfg["rc_scale_vx"]),
            "vy": float(cfg["rc_scale_vy"]),
            "omega": float(cfg["rc_scale_omega"]),
        }
        self.timeout = float(cfg["rc_inactivity_timeout"])
        self.last_active = time.perf_counter()
        self.cmd = np.zeros(3, dtype=np.float32)

    def update(self, wireless_remote):
        """Decode one lowstate wireless_remote (40 x uint8). Returns cmd (3,)."""
        self.joy.extract(wireless_remote)

        # any stick out of deadzone = activity
        if any(abs(getattr(self.joy, n).data) > 0.05 for n in STICK_NAMES):
            self.last_active = time.perf_counter()
        elif time.perf_counter() - self.last_active > self.timeout:
            self.cmd[:] = 0.0
            return self.cmd

        for i, name in enumerate(("vx", "vy", "omega")):
            src = self.mapping[name]
            neg = src.startswith("-")
            val = getattr(self.joy, src.lstrip("-")).data
            self.cmd[i] = (-val if neg else val) * self.scale[name]
        return self.cmd

    def zero(self):
        self.cmd[:] = 0.0
        return self.cmd
