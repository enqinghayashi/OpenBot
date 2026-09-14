#!/usr/bin/env python3
"""yoloe_box_probe.py -- READ-ONLY S1 measurement probe for the YOLOE batch-4 lane.

Batch 4 (docs/YOLOE_BATCH4_ARCHITECTURE.md section 11).  Operator-run on the box;
never run by an agent against the robot.  py3.8, stdlib + OPTIONAL imports
(numpy / cv2 / torch / ultralytics / rclpy are each probed and reported, never
required).

What it does (each step is independent; a failing step is recorded, never fatal):
  0. provenance  md5 of the DEPLOYED node / projector vs the staging base md5s,
                 `match_staging_base: {"node": bool, "projector": bool}` and a
                 unified diff `deployed_vs_base_<file>.diff` in --out-dir when false.
  1. env         interpreter, ultralytics / torch / CUDA / TensorRT, YOLOE API,
                 clip importability, free RAM, nproc, 1-min load, file presence,
                 and `torch.load(<each .pt>, weights_only=False)` per checkpoint.
  2. --bench     the engine, then each .pt, on CPU and CUDA, at threads 1/2/4:
                 median/p90 ms, process CPU %, RSS; for YOLOE: bank_swap_ms.
  3. --record N  saves N frames of /camera/color/image_raw as PNGs (subscribe only).
     --agree DIR engine (or --baseline-pt) vs YOLOE per PNG: base-20 label-set
                 agreement, |dconf| median, extra-row count per seeded prompt; TWO
                 tiers (architecture section 17 M7): raw (GATE_CONF 0.20, reported)
                 and pick (--tier-conf, default 0.35, the G-AGREE gate tier), plus
                 <out-dir>/agree_frames.json with both detectors' box lists per frame.
  4. output      ONE JSON line on stdout + --out <file>; exit 0 ALWAYS.

Writes ONLY inside --out-dir (the ultralytics settings dir is redirected there via
YOLO_CONFIG_DIR, and YOLO_OFFLINE=1 stops network probes).  Publishes nothing,
calls no service, moves nothing.  The text encoder is NEVER invoked: a vocabulary
is set only as set_classes(names, embeddings) from a laptop-built bank NPZ.
"""
import argparse
import difflib
import glob
import hashlib
import json
import os
import re
import resource
import statistics
import struct
import sys
import time
import traceback
import zlib

PROBE_VERSION = '1.0'

# base md5s (BASE_MD5.json)
# hand-placed copies, no repo source:
# md5 = is it base?
BASE_NODE_MD5 = '8b1619e50724cecdd40aa759cd950b6e'        # md5 of yolo_node_seg.py
BASE_PROJECTOR_MD5 = '40e1757677a1b754f093e49358fd7e42'   # md5 of depth_projector_node.py
BASE_NODE_NAME = 'yolo_node_seg.py'
BASE_PROJECTOR_NAME = 'depth_projector_node.py'

DEFAULT_LEON_DIR = '/home/unitree/leon/openbot'           # from launch_pick_stack.sh:84
DEFAULT_ENGINE = '/home/unitree/yolo11n-seg.engine'      # see yolo_node_seg.py:236
DEFAULT_ASSETS_DIR = '/home/unitree/leon/yoloe'          # INFERRED
IMAGE_TOPIC = '/camera/color/image_raw'                  # see yolo_node_seg.py:258
FRAME_W, FRAME_H = 424, 240
GATE_CONF = 0.20                                         # see yolo_node_seg.py:96
IOU = 0.45
EXTRA_MIN_CONF = 0.35                                    # config.yaml det_conf_floor
PICK_TIER_CONF = 0.35                                    # < 0.35 never binds
AGREE_FRAMES_FILE = 'agree_frames.json'
EXTRAS_PER_CHUNK = 16                                    # cap: 16/chunk

# Base-20 = TRACKED_CLASSES verbatim
# spaces kept exactly as the detector emits them
BASE_20 = ('bottle', 'chair', 'cup', 'laptop', 'cell phone', 'book', 'keyboard',
           'mouse', 'remote', 'clock', 'vase', 'potted plant', 'person', 'apple',
           'backpack', 'sports ball', 'bowl', 'suitcase', 'handbag', 'fork')

# top-level keys, test pinned
# tests/unit/perception/test_yoloe_probe_schema.py
REQUIRED_KEYS = ('probe_version', 'interpreter', 'args', 'out_dir', 'provenance',
                 'match_staging_base', 'env', 'torch_load', 'banks', 'bench',
                 'record', 'agree', 'errors', 'elapsed_s')


# ------------------------------------------------------------ the stdout guard
# stdout: one line
# ultralytics/TensorRT -> stderr at FD
# + sys.stdout
_REAL_STDOUT_FD = None


def _guard_stdout():
    global _REAL_STDOUT_FD
    try:
        sys.stdout.flush()
    except Exception:
        pass
    _REAL_STDOUT_FD = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr


def _emit_line(line):
    data = (line + '\n').encode('utf-8')
    fd = _REAL_STDOUT_FD if _REAL_STDOUT_FD is not None else 1
    while data:
        n = os.write(fd, data)
        data = data[n:]


def _log(msg):
    sys.stderr.write('[probe] %s\n' % (msg,))
    sys.stderr.flush()


# ---------------------------------------------------------- small helpers
def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _exc(e):
    return '%s: %s' % (type(e).__name__, e)


def _rss_mb():
    """Current RSS (VmRSS) in MB; falls back to peak ru_maxrss."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _maxrss_mb():
    """This process's own peak (VmHWM).  ru_maxrss is inherited across execve on Linux
    (it can report the launching shell's/parent's peak), so it is only the fallback."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmHWM:'):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)


def _cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(q * len(sorted_vals)))
    return round(sorted_vals[i], 2)


def _parse_list(s, cast=str):
    return [cast(x.strip()) for x in str(s).split(',') if x.strip()]


# ---------------------------------------------------------- 0. provenance check
