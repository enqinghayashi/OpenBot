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


# ── the colour leg ───────────────────
COLOR_MAX_SAMPLE_PX = 4096  # strided sampling


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
        rows_bgr = img_bgr[ys, xs]  # uint8: no CV_64F
        hsv = cv2.cvtColor(
            rows_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)
        hsv = hsv.reshape(-1, 3).astype(np.float64)
        # hue 0-180 -> 0-360
        hsv[:, 0] = hsv[:, 0] * 2.0
        hsv[:, 1] = hsv[:, 1] / 255.0
        hsv[:, 2] = hsv[:, 2] / 255.0
        return hsv
    except Exception as e:  # noqa: BLE001
        print('[yolo] colour rasterize failed: %s' % (e,), file=sys.stderr)
        return None


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        # ── optional-lane state (inert) ──────
        self._yoloe_on = _env_on('OPENBOT_YOLOE')
        self._classes_on = _env_on('OPENBOT_YOLO_CLASSES')
        self._backend = None            # YoloeBackend live
        self._backend_name = 'yolo'
        self._fallback_reason = None
        self._fallback_due = False
        self._fallback_thread = None
        self._fatal = None              # worker sets it
        self._consec_errors = 0
        self._base_order = sorted(TRACKED_CLASSES)
        self._extras = {}               # label -> prompt
        self._extra_conf_default = YOLOE_DEFAULT_EXTRA_CONF
        self._extra_conf = YOLOE_DEFAULT_EXTRA_CONF
        self._pending_vocab = None      # None unless CLASSES=1
        self._vocab_mod = None
        self._vocab_dedupe = None
        self._classes_sub = None
        self._state_pub = None
        self._tracker = None            # only with IDS=1
        if _env_on('OPENBOT_INSTANCE_IDS'):
            self._init_instance_tracker()

        if self._yoloe_on:
            self._init_yoloe()
        else:
            # ── loading the gate model ────────────
            try:
                self._gate_model = YOLO(ENGINE_PATH)
                self.get_logger().info(
                    'COCO-gate SEG model loaded (yolo11n-seg.pt, 80 classes).')
            except Exception as e:
                self.get_logger().error(f'Gate model failed to load: {e}')
                raise

        # allowlist = the 32 above, not
        # the names dict; filter below
        self._tracked_classes = TRACKED_CLASSES
        # TRACKED + extras
        # the literal is fixed
        self._runtime_allow = TRACKED_CLASSES
        self.get_logger().info(
            f'Class allowlist CURATED: {len(self._tracked_classes)} '
            f'daily-life COCO classes admitted.')

        self._latest_frame = None
        self._lock = threading.Lock()
        self._busy = False

        # key -> {first_seen, last_seen, ...}
        # class name or iid (IDS=1)
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

