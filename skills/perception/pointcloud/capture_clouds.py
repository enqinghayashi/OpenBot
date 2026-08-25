#!/usr/bin/env python3
"""capture_clouds: a READ-ONLY, BOUNDED recorder of /object_pointcloud +
/object_detections_body into one .npz (+ a .json sidecar) for calibrating
pc_config.PcConfig on real data.

  python3 capture_clouds.py --out /home/unitree/leon/pc_capture/<scene>_<ts>.npz \
      --frames 60 --max-mb 48 --timeout-s 60 [--label bottle] [--body-baseline-s 5]

Bounds (refused above): --frames <= 300, --max-mb <= 128, --timeout-s <= 600,
--body-baseline-s <= 30.  Stops at the FIRST cap reached (frames, bytes,
timeout).  Peak RSS ~ 2 x max-mb (the final concatenate) + the rclpy baseline
(DERIVED).

Safety: publishes nothing; subscribes only to the two perception topics;
os.nice(10); never shells out and never runs the ros2 CLI rate tool (a rate
probe once OOM'd the box: FUTURE_WORK_target_signal_and_odometry.md :262-264).

Phase A (--body-baseline-s): subscribe to the BODY topic only and record its
inter-arrival gaps.  Phase B: subscribe to the cloud as well and keep
recording body gaps.  The sidecar reports both, so the operator can see whether
a cloud subscriber slows the projector (critique: callback p90 increase gate).

The .npz schema (all plain arrays, allow_pickle=False):
  pts (SUM N,3) float32, oid (SUM N,) uint32, frame_off (F+1,) int64,
  n_raw (F,) int64, t_rx_wall (F,) float64, t_rx_mono (F,) float64,
  hdr_stamp_ns (F,) int64, body_json (F,) str, body_t_rx_mono (F,) float64,
  body_prev_json (F,) str, body_prev_t_rx_mono (F,) float64, meta () str (JSON).
pc_core.load_capture(path) replays it through the SAME pair_cloud.

Python 3.8-safe.  rclpy is imported only inside main().
"""
import argparse
import json
import os
import sys
import time

import numpy as np

try:  # package import
    from . import pc_core as core
except ImportError:  # flat import, box layout
    import pc_core as core

MAX_FRAMES_CAP = 300
MAX_MB_CAP = 128
MAX_TIMEOUT_S = 600.0
MAX_BODY_BASELINE_S = 30.0

# projector constants as literals - we never import the projector
PROJECTOR_CONSTANTS = {
    "FX": 302.7, "FY": 302.8, "CX": 212.9, "CY": 121.1,
    "DEPTH_MIN": 0.3, "DEPTH_MAX": 6.0, "COLOR_W": 424, "COLOR_H": 240,
    "T_CAM_IN_TORSO": [0.0, 0.04764, 0.46268], "TILT_DEG": 48.0,
    "MAX_PTS_PER_OBJ": 2000, "MAX_CLOUD_POINTS": 20000,
}


class CaptureBuffer(object):
    """Bounded in-memory store.  add() refuses a frame that would cross either
    cap and marks the buffer full."""

    def __init__(self, max_frames, max_bytes):
        self.max_frames = int(max_frames)
        self.max_bytes = int(max_bytes)
        self.nbytes = 0
        self.full = False
        self.n_format_err = 0
        self._pts, self._oid = [], []
        self._cols = dict((k, []) for k in (
            "n_raw", "t_rx_wall", "t_rx_mono", "hdr_stamp_ns", "body_json", "body_t_rx_mono",
            "body_prev_json", "body_prev_t_rx_mono"))

    def __len__(self):
        return len(self._pts)

    def add(self, data, width, point_step, fields, is_bigendian, hdr_stamp_ns, t_mono, t_wall,
            bodies, height=1):
        """bodies: [(t_rx_mono, json_text), ...] newest first (<= 2 kept)."""
        if self.full or len(self._pts) >= self.max_frames:
            self.full = True
            return False
        try:
            pts, oid, _ = core.decode_cloud(data, width, point_step, fields, is_bigendian, height)
        except core.CloudFormatError:
            self.n_format_err += 1
            return False
        b0 = bodies[0] if len(bodies) > 0 else (float("nan"), "")
        b1 = bodies[1] if len(bodies) > 1 else (float("nan"), "")
        cost = pts.nbytes + oid.nbytes + 4 * (len(b0[1]) + len(b1[1])) + 64
        if self.nbytes + cost > self.max_bytes:
            self.full = True
            return False
        self.nbytes += cost
        self._pts.append(pts)
        self._oid.append(oid)
        c = self._cols
        c["n_raw"].append(int(width) * int(height))
        c["t_rx_wall"].append(float(t_wall))
        c["t_rx_mono"].append(float(t_mono))
        c["hdr_stamp_ns"].append(int(hdr_stamp_ns))
        c["body_json"].append(b0[1])
        c["body_t_rx_mono"].append(float(b0[0]))
        c["body_prev_json"].append(b1[1])
        c["body_prev_t_rx_mono"].append(float(b1[0]))
        if len(self._pts) >= self.max_frames:
            self.full = True
        return True

    def save(self, path, meta):
        F = len(self._pts)
        off = np.zeros(F + 1, dtype=np.int64)
        for k, p in enumerate(self._pts):
            off[k + 1] = off[k] + len(p)
        pts = np.concatenate(self._pts) if F else np.zeros((0, 3), np.float32)
        oid = np.concatenate(self._oid) if F else np.zeros((0,), np.uint32)
        c = self._cols
        arrays = {
            "pts": pts.astype(np.float32), "oid": oid.astype(np.uint32), "frame_off": off,
            "n_raw": np.asarray(c["n_raw"], np.int64),
            "t_rx_wall": np.asarray(c["t_rx_wall"], np.float64),
            "t_rx_mono": np.asarray(c["t_rx_mono"], np.float64),
            "hdr_stamp_ns": np.asarray(c["hdr_stamp_ns"], np.int64),
            "body_json": _str_array(c["body_json"]),
            "body_t_rx_mono": np.asarray(c["body_t_rx_mono"], np.float64),
            "body_prev_json": _str_array(c["body_prev_json"]),
            "body_prev_t_rx_mono": np.asarray(c["body_prev_t_rx_mono"], np.float64),
            "meta": np.asarray(json.dumps(meta)),
        }
        with open(path, "wb") as f:
            np.savez(f, **arrays)
        return arrays
