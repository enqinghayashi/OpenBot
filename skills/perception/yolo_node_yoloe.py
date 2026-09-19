#!/usr/bin/env python3
"""
yolo_node_SEG_version.py — yolo11n-seg COCO gate + per-object segmentation masks.

Same COCO-gate semantics as yolo_node_COCO_GATE_version.py (stock yolo11n,
ALL 80 COCO classes admitted — allowlist derived from the model's own names,
GATE_CONF 0.20, 1 s confirmation, empty-scene silence by design), but the
model is **yolo11n-seg**, so every
confirmed detection additionally carries a per-object segmentation mask as a
COCO-style polygon (list of [x, y] int pairs in color-pixel coords, 424x240).

The ``mask`` field is ADDITIVE to /detections: consumers that read by key
(pick_node, walk_node) ignore it; depth_projector_node_SEG_version.py uses it
to deproject object-only pixels instead of the fixed 7x7 bbox patch.

2026-09-07 seam (colour + left/right perception, ported from the reference
pick-walk_skill yolo_full_gate tree): every confirmed detection ADDITIONALLY
carries a ``hue`` (float degrees [0,360) | None), ``color_name`` (one of the 8
instance_labels.HUE_BANDS names | None) and ``frame_side`` ('left'|'right').
Colours are read from sticky-mask pixels of the CURRENT frame only, via a
guarded cv2 rasterize -> HSV -> instance_labels.dominant_color() (cv2
unavailable / mask-less / unmeasurable -> honest None, never a crash).  Pure
colour/label/rank/grammar helpers live in the sibling stdlib module
``instance_labels.py`` (deployed co-located with this file on the box).

Weights: /home/unitree/yolo11n-seg.pt (scp alongside this file — the stock
yolo11n.pt detect weights do NOT include masks).

Deploy:
    scp this_file unitree@192.168.123.164:/home/unitree/leon/openbot/yolo_node.py
    scp yolo11n-seg.pt unitree@192.168.123.164:/home/unitree/yolo11n-seg.pt
    # kill yolo_node; relaunch via launch_pick_pipeline.sh
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import json
import threading
import time
import sys
import os
import math

from ultralytics import YOLO

# cv2 optional: colour leg
# absent -> None
try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

# ── instance_labels ─────────────────────
# plain, dir, root
# missing = fatal
# never class-only
try:
    import instance_labels  # form 1
except ImportError:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _import_err = None
    for _candidate in (_SCRIPT_DIR,
                       os.path.dirname(_SCRIPT_DIR),
                       os.path.dirname(os.path.dirname(_SCRIPT_DIR))):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        try:
            import instance_labels  # forms 2, 3
            break
        except ImportError as _e:
            _import_err = _e
    else:
        raise ImportError(
            'instance_labels.py is not importable: tried a direct import, '
            'this file\'s directory (%r), the perception parent and the repo '
            'root.  It must ship co-located with this node (box flat '
            'layout) or at the skills/perception root.  Underlying error: %s'
            % (_SCRIPT_DIR, _import_err))

# ── the gate numbers ────────────────────
# lockstep 0.20: pick_node + floor_pick_node
# bottle 0.26-0.40: 0.25 left holes
# 0.20 holds it
# empty = quiet on purpose
# (COCO 80-class negatives)
GATE_CONF = 0.20
CONFIRM_SECONDS = 1.0

# allowlist: 32 daily-life names
# 16 base + 16 extra
# food/pets dropped
# fixed at import
# not the names dict - membership
# filters below
COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

TRACKED_CLASSES = frozenset({
    # 16 base, 4 keeps
    # matches yolo_node_SEG_version.py
    # cut: wine glass, knife, spoon, umbrella,
    # scissors, teddy bear, toothbrush, hair drier
    # room masks (table, tv, couch, bed) push
    # ~96k pts into the depth SVD -> OOM
    'bottle', 'chair', 'cup', 'laptop', 'cell phone', 'book', 'keyboard',
    'mouse', 'remote', 'clock', 'vase', 'potted plant', 'person', 'apple',
    'backpack', 'sports ball',
    'bowl', 'suitcase', 'handbag', 'fork',
})


def classify_orientation(aspect_ratio):
    """Return 'upright' or 'horizontal' from bbox aspect ratio."""
    if aspect_ratio > 1.5:
        return 'horizontal'
    elif aspect_ratio < 0.7:
        return 'upright'
    return 'upright'  # can't tell: safer behind


def _mask_to_polygon(mask_xy_i):
    """Convert one ultralytics mask polygon (numpy (N,2)) to a JSON-ready list
    of [x, y] int pairs in color-pixel coords.  Returns None if absent or
    degenerate (< 3 points)."""
    if mask_xy_i is None:
        return None
    try:
        poly = [[int(round(float(p[0]))), int(round(float(p[1])))]
                for p in mask_xy_i]
        return poly if len(poly) >= 3 else None
    except Exception:
        return None
