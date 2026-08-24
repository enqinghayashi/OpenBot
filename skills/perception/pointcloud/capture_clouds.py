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


