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

# cv2 optional - colour leg only (BGR2HSV)
# gone -> colour fields None, no crash
try:
    import cv2
except Exception:  # noqa: BLE001  — no crash
    cv2 = None

# ── instance_labels ─────────────────────
# lives in instance_labels.py next door
# import: plain, this dir, parent, root
# missing -> hard startup error
try:
    import instance_labels  # try plain
except ImportError:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _import_err = None
    for _candidate in (_SCRIPT_DIR,
                       os.path.dirname(_SCRIPT_DIR),
                       os.path.dirname(os.path.dirname(_SCRIPT_DIR))):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        try:
            import instance_labels  # with dirs added
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

# ── Gate settings ───────────────────────
# 0.20, same as pick_node MIN_CONF
# bottle conf flickers 0.26-0.40
# 0.25 slid back into "confirming"
# -> multi-second gaps
# 0.20 holds it
# empty scene silent (COCO 80 negatives)
# ref: yolo_node_COCO_GATE_version.py
GATE_CONF = 0.20
CONFIRM_SECONDS = 1.0

# key = centroid grid cell, not class name
# 3 cups apart -> 3 candidates
# each confirms on its own
# not a tracker: static centroid stays put
# jump past INSTANCE_GRID_PX -> new instance
INSTANCE_GRID_PX = 30

# 32-name daily-life allowlist
# trimmed from the all-80 window
# 16 keeps + 16 tableware/furniture
# food + pets absent -> filter drops them
# fixed at import, not from model names
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
    # 16 originals + 4 keeps:
    # bowl, suitcase, handbag, fork
    # dropped room-scale masks (dining table,
    # tv, couch, bed): whole-frame -> ~96k pts
    # into the depth SVD -> 69 GiB OOM
    # rest (wine glass, knife, spoon, umbrella,
    # scissors, teddy bear, toothbrush, hair
    # drier) only added latency
    # COCO_NAMES stays full-80 (engine index)
    # TRACKED_CLASSES = publish allowlist
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
    return 'upright'  # ambiguous: approach behind


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


# ── Identity + colour label ─────────────
def _instance_key(class_name, cx, cy, grid_px=INSTANCE_GRID_PX):
    """Stable per-object identity key: f'{class_name}|{col},{row}' where
    (col, row) is the centroid's cell in a 30 px grid over the colour frame.
    class_name is part of the key so two DIFFERENT classes sharing a cell never
    merge; same-class objects separated by >= grid_px in x or y get distinct
    keys, so 3 cups at 3 positions map to 3 keys and each confirms on its own."""
    return '%s|%d,%d' % (class_name,
                         int(float(cx) // grid_px),
                         int(float(cy) // grid_px))


def _colour_label(color_name, class_name):
    """Published detection label: f'{color_name} {class_name}' ('red cup') when
    color_name is a REAL instance_labels HUE_BANDS band name, else the bare
    class_name ('cup').  A colour is NEVER faked: unknown/unmeasured colour
    falls back to the bare class.  The bare COCO class is always carried
    separately in the row's 'class_name' field for the pick chain."""
    if color_name and color_name in getattr(
            instance_labels, 'HUE_BAND_BY_NAME', ()):
        return '%s %s' % (color_name, class_name)
    return class_name


# ── Colour leg ──────────────────────────
COLOR_MAX_SAMPLE_PX = 4096  # strided sample cap


def _mask_rows_hsv(img_bgr, poly, max_px=COLOR_MAX_SAMPLE_PX):
    """Sample the CURRENT bgr frame inside mask polygon ``poly``, strided to
    <= max_px pixels, and return rows as float HSV with hue scaled to the
    plan's 0-360 degrees and S/V as 0-1 fractions
    (cv2.cvtColor -> H *= 2.0, S = S/255, V = V/255).  Returns None when cv2
    is unavailable, the polygon is unusable, or the mask rasterizes empty."""
    if cv2 is None or poly is None:
        return None
    try:
        h, w = img_bgr.shape[0], img_bgr.shape[1]
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        raster = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(raster, [pts], 1)
        ys, xs = np.where(raster > 0)
        n = ys.size
        if n == 0:
            return None
        if n > max_px:
            step = int(math.ceil(math.sqrt(float(n) / float(max_px))))
            ys = ys[::step]
            xs = xs[::step]
        rows_bgr = img_bgr[ys, xs]  # cvtColor rejects CV_64F
        hsv = cv2.cvtColor(
            rows_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(-1, 3).astype(np.float64)
        # cv2 hue 0-180 -> degrees 0-360
        hsv[:, 0] = hsv[:, 0] * 2.0
        hsv[:, 1] = hsv[:, 1] / 255.0
        hsv[:, 2] = hsv[:, 2] / 255.0
        return hsv
    except Exception as e:  # noqa: BLE001  — no crash
        print('[yolo] colour rasterize failed: %s' % (e,), file=sys.stderr)
        return None


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # ── YOLOE state (inert) ─────────────────
        self._yoloe_on = _env_on('OPENBOT_YOLOE')
        self._classes_on = _env_on('OPENBOT_YOLO_CLASSES')
        self._backend = None            # backend once live
        self._backend_name = 'yolo'
        self._fallback_reason = None
        self._fallback_due = False
        self._fallback_thread = None
        self._fatal = None              # worker sets; re-raised
        self._consec_errors = 0
        self._base_order = sorted(TRACKED_CLASSES)
        self._extras = {}               # extra -> prompt
        self._extra_conf_default = YOLOE_DEFAULT_EXTRA_CONF
        self._extra_conf = YOLOE_DEFAULT_EXTRA_CONF
        self._pending_vocab = None      # None unless OPENBOT_YOLO_CLASSES is ON
        self._vocab_mod = None
        self._vocab_dedupe = None
        self._classes_sub = None
        self._state_pub = None
        self._tracker = None            # InstanceTracker when ids on
        if _env_on('OPENBOT_INSTANCE_IDS'):
            self._init_instance_tracker()

        if self._yoloe_on:
            self._init_yoloe()
        else:
            # ── Load COCO-gate SEG model ────────────
            try:
                self._gate_model = YOLO(ENGINE_PATH)
                self.get_logger().info(
                    'COCO-gate SEG model loaded (yolo11n-seg.pt, 80 classes).')
            except Exception as e:
                self.get_logger().error(f'Gate model failed to load: {e}')
                raise

        self._tracked_classes = TRACKED_CLASSES
        # TRACKED_CLASSES unless vocab adds extras
        self._runtime_allow = TRACKED_CLASSES
        self.get_logger().info(
            f'Class allowlist CURATED: {len(self._tracked_classes)} '
            f'daily-life COCO classes admitted.')

        self._latest_frame = None
        self._lock = threading.Lock()
        self._busy = False

        # key -> {first_seen, last_seen, ...}
        self._candidates = {}

        self.sub = self.create_subscription(
            Image, '/camera/color/image_raw', self._on_image, 10)
        self.pub = self.create_publisher(String, '/detections', 10)
        self.create_timer(0.1, self._run_inference)
        if self._classes_on:
            self._init_classes_topic()
        self.get_logger().info(
            f'COCO-gate SEG node ready (AR orientation). '
            f'gate_conf>{GATE_CONF}, confirm>{CONFIRM_SECONDS}s')

    def _on_image(self, msg):
        with self._lock:
            self._latest_frame = msg

