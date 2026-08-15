"""gate_core.py -- pure-stdlib spatial gate for /object_detections_body.

Drops same-label detections that jump farther than a physically plausible
distance since the last ACCEPTED one (the measured bimodal bearing: -1.5 deg
vs +20.9 deg at d=0.60 m, ~0.22-0.245 m lateral swing in 0.10 s).  Every other
label passes through UNCHANGED.  No rclpy.  Python-3.8-safe.

Opt-in only: nothing consumes the gated topic unless walk_node is restarted
with the --ros-args remap (see README.md).
"""

import json
import math

# walk_node.py copies; tests pin by regex:
VX_MAX_DEFAULT = 0.30          # = VX_WALK = 0.30
YAW_RATE_MAX_DEFAULT = 0.60    # = K_THETA*OMEGA_MAX = 1.5*0.40
DT_CAP_DEFAULT_S = 5.0         # = TARGET_JUMP_DT_S = 5.0
# INFERRED, not measured: doc swing ~0.22 m
# (0.245 derived) vs 0.198 m tick-to-tick;
# within 10% -> no calibration; re-tune off capture_detections.py
NOISE_FLOOR_DEFAULT_M = 0.10
REACQ_DEFAULT_S = 1.0          # INFERRED: matches the doc's >= 1 s dropout

STOP_WORDS = ('stop', 'abort', '')


def _num(v):
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _stamp(det):
    """Capture stamp as float, or None (the G18 _det_stamp rule,
    walk_node.py:1498-1511)."""
    return _num(det.get('stamp'))


def radius(noise_floor_m, vx_max, yaw_rate_max, d_last_m, dt_s):
    """Acceptance radius around the last accepted (x, y)."""
    return noise_floor_m + (vx_max + yaw_rate_max * d_last_m) * dt_s


class TargetGate(object):
    def __init__(self, noise_floor_m=NOISE_FLOOR_DEFAULT_M,
                 vx_max=VX_MAX_DEFAULT, yaw_rate_max=YAW_RATE_MAX_DEFAULT,
                 dt_cap_s=DT_CAP_DEFAULT_S, reacq_s=REACQ_DEFAULT_S):
        self.noise_floor_m = float(noise_floor_m)
        self.vx_max = float(vx_max)
        self.yaw_rate_max = float(yaw_rate_max)
        self.dt_cap_s = float(dt_cap_s)
        self.reacq_s = float(reacq_s)
        self.accepted = 0
        self.dropped = 0
        self.last_radius = 0.0
        self.reset()

    def reset(self):
        self._last_xy = None
        self._last_stamp = None
        self._last_recv = None

    def _dt(self, det, recv_t):
        s = _stamp(det)
        if s is not None and self._last_stamp is not None:
            return max(0.0, s - self._last_stamp)
        return max(0.0, recv_t - self._last_recv)

    def _take(self, det, recv_t):
        self._last_xy = (_num(det.get('x')), _num(det.get('y')))
        self._last_stamp = _stamp(det)
        self._last_recv = recv_t

