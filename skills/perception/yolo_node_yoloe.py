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

    def _run_inference(self):
        if self._fatal is not None:
            raise self._fatal           # no engine either
        if self._busy:
            return
        if self._pending_vocab is not None:      # None unless CLASSES=1
            self._apply_pending_vocab()          # before the frame
        if getattr(self, '_strict_sub', None) is not None:   # strict, if wired
            self._strict_step()                  # lease first
        with self._lock:
            msg = self._latest_frame
            self._latest_frame = None
        if msg is None:
            return

        self._busy = True
        try:
            img = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(
                msg.height, msg.width, 3)
            now = time.time()

            # ── stage 1: gate all classes ────────
            backend = self._backend
            if backend is None:
                gate_results = self._gate_model(
                    img, conf=GATE_CONF, iou=0.45, verbose=False)[0]
                names = None            # fixed engine: COCO names
            else:
                # YOLOE: box.cls is a PROMPT index
                # result names, never COCO_NAMES
                # (prompt 0 would be 'person')
                gate_results = backend.infer(img, conf=GATE_CONF, iou=0.45)
                names = getattr(gate_results, 'names', None) or backend.names
            suppressed = ()
            if names is not None:
                suppressed = self._yoloe_suppressed(gate_results, names)

            mask_xy = None
            if gate_results.masks is not None:
                mask_xy = gate_results.masks.xy

            tracked = None              # None unless IDS=1
            if self._tracker is not None:
                tracked = self._track_frame(gate_results, names, suppressed,
                                            now)

            for i, box in enumerate(gate_results.boxes):
                conf = float(box.conf)
                if conf < GATE_CONF:
                    continue
                class_id = int(box.cls)
                if names is None:
                    class_name = COCO_NAMES[class_id]
                else:
                    class_name = _name_at(names, class_id)
                    if i in suppressed:
                        continue
                if class_name not in self._runtime_allow:
                    continue
                if (names is not None and class_name not in TRACKED_CLASSES
                        and conf < self._extra_conf):
                    continue            # extra admit floor
                if tracked is not None:
                    if i not in tracked:
                        continue        # fused duplicate
                    iid, class_name, track_age = tracked[i]

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
                aspect_ratio = bbox_w / max(bbox_h, 1)
                orientation = classify_orientation(aspect_ratio)

                mask_poly = None
                if mask_xy is not None and i < len(mask_xy):
                    mask_poly = _mask_to_polygon(mask_xy[i])

                # ── colour + side ────────────────────
                # mask px, this frame only
                # -> HSV -> dominant_color
                # no cv2/mask: None
                # hue deg [0,360), color_name
                # 8 HUE_BANDS; side
                # from bbox centre vs midline
                hue, color_name = None, None
                if mask_poly is not None:
                    hsv_rows = _mask_rows_hsv(img, mask_poly)
                    if hsv_rows is not None:
                        # ``rows_hsv or ()`` guard bool()s
                        # its input - raises on a >1-el
                        # numpy array, so feed plain rows
                        _rows = (hsv_rows.tolist()
                                 if hasattr(hsv_rows, 'tolist') else hsv_rows)
                        _res = instance_labels.dominant_color(_rows)
                        hue = _res.get('hue')
                        if hue is not None:
                            _hint = _res.get('color_name_hint')
                            color_name = (_hint if _hint and _hint != 'unknown'
                                          else instance_labels.nearest_color_name(
                                              hue))
                        else:
                            color_name = None
                frame_side = instance_labels.frame_side(cx, msg.width)

                if tracked is None:
                    key = class_name    # one per class
                else:
                    key = iid           # rides its track

                if key not in self._candidates:
                    self._candidates[key] = {
                        'first_seen': now, 'last_seen': now,
                        'class_name': class_name,
                        'cx': cx, 'cy': cy, 'conf': conf,
                        'bbox_w': bbox_w, 'bbox_h': bbox_h,
                        'aspect_ratio': aspect_ratio,
                        'orientation': orientation,
                        'mask': mask_poly,
                        'hue': hue,
                        'color_name': color_name,
                        'frame_side': frame_side,
                    }
                    if tracked is not None:
                        self._candidates[key]['instance_id'] = iid
                        self._candidates[key]['track_age_s'] = track_age
                    self.get_logger().info(
                        f'Candidate: {class_name} conf={conf:.2f} '
                        f'orient={orientation} mask={"yes" if mask_poly else "no"} '
                        f'color={color_name} side={frame_side} — '
                        f'confirming for {CONFIRM_SECONDS}s...')
                else:
                    c = self._candidates[key]
                    c['last_seen'] = now
                    c['class_name'] = class_name
                    c['cx'] = cx
                    c['cy'] = cy
                    c['conf'] = conf
                    c['bbox_w'] = bbox_w
                    c['bbox_h'] = bbox_h
                    c['aspect_ratio'] = aspect_ratio
                    c['orientation'] = orientation
                    c['mask'] = mask_poly
                    c['hue'] = hue
                    c['color_name'] = color_name
                    c['frame_side'] = frame_side
                    if tracked is not None:
                        c['track_age_s'] = track_age

            stale = [l for l in self._candidates
                     if now - self._candidates[l]['last_seen'] > 2.0]
            for label in stale:
                del self._candidates[label]

            confirmed = []
            for key, c in self._candidates.items():
                duration = c['last_seen'] - c['first_seen']
                if duration >= CONFIRM_SECONDS:
                    d = {
                        'label': c['class_name'],
                        'conf': c['conf'],
                        'cx': c['cx'],
                        'cy': c['cy'],
                        'bbox_w': c['bbox_w'],
                        'bbox_h': c['bbox_h'],
                        'aspect_ratio': c['aspect_ratio'],
                        'orientation': c['orientation'],
                        'stamp': now,
                        # ── colour + side (always sent) ───────
                        'hue': c['hue'],
                        'color_name': c['color_name'],
                        'frame_side': c['frame_side'],
                    }
                    if c.get('mask') is not None:
                        d['mask'] = c['mask']
                    if 'instance_id' in c:
                        d['instance_id'] = c['instance_id']
                        d['track_age_s'] = c['track_age_s']
                    confirmed.append(d)

            if confirmed:
                out = String()
                out.data = json.dumps(confirmed)
                self.pub.publish(out)
                labels = [f'{d["label"]}({d["orientation"]})'
                          for d in confirmed]
                self.get_logger().info(f'Confirmed: {labels}')

            self._consec_errors = 0

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')
            if self._backend is not None:
                self._consec_errors += 1
                if self._consec_errors >= YOLOE_FALLBACK_AFTER:
                    self._fallback_due = True
        finally:
            self._busy = False
        if self._fallback_due:
            self._start_engine_fallback()

    # ── picking the backend (YOLOE=1) ────
    def _load_engine(self):
        """Load today's engine for a FALLBACK.  A missing engine is FATAL
        (logged, re-raised) -- never a silent node publishing nothing."""
        if not os.path.isfile(ENGINE_PATH):
            err = FileNotFoundError('engine missing: %s' % ENGINE_PATH)
            self.get_logger().error('[yoloe] FATAL no backend: %s' % err)
            raise err
        try:
            return YOLO(ENGINE_PATH)
        except Exception as e:
            self.get_logger().error('[yoloe] FATAL no backend: %s' % e)
            raise

    def _init_yoloe(self):
        weights, bank, threads, extra_conf, problems = _yoloe_env()
        for p in problems:
            self.get_logger().error('[yoloe] %s' % p)
        self._extra_conf_default = extra_conf
        self._extra_conf = extra_conf
        try:
            mod = _import_sibling('yoloe_backend')
            backend = mod.YoloeBackend(
                weights, bank, threads=threads,
                log=lambda s: self.get_logger().info(s),
                base=self._base_order)
            backend.load()
        except Exception as e:
            self._fallback_reason = str(e)
            self.get_logger().error(
                '[yoloe] FALLBACK to yolo11n-seg.engine: %s' % e)
            self._gate_model = self._load_engine()
            self._backend = None
            self._backend_name = 'yolo'
            return
        self._backend = backend
        self._gate_model = None
        self._backend_name = 'yoloe'
        self.get_logger().info(
            '[yoloe] backend yoloe: weights=%s bank=%s threads=%d '
            'extra_conf=%.2f' % (weights, bank, threads, extra_conf))

    # ── /yolo/classes (CLASSES=1) ────────
    def _init_classes_topic(self):
        try:
            self._vocab_mod = _import_sibling('vocab_state')
        except Exception as e:
            self.get_logger().error(
                '[yoloe] OPENBOT_YOLO_CLASSES disabled: vocab_state not '
                'importable (%s)' % e)
            return
        import rclpy.qos as _rq
        QoSProfile = _rq.QoSProfile
        # Foxy+ names, else
        # the long QoS* aliases
        DurabilityPolicy = getattr(_rq, 'DurabilityPolicy', None) or \
            getattr(_rq, 'QoSDurabilityPolicy')
        ReliabilityPolicy = getattr(_rq, 'ReliabilityPolicy', None) or \
            getattr(_rq, 'QoSReliabilityPolicy')
        HistoryPolicy = getattr(_rq, 'HistoryPolicy', None) or \
            getattr(_rq, 'QoSHistoryPolicy')
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._vocab_dedupe = self._vocab_mod.Dedupe()
        self._classes_sub = self.create_subscription(
            String, TOPIC_CLASSES, self._on_classes, qos)
        self._state_pub = self.create_publisher(
            String, TOPIC_CLASSES_STATE, qos)
        # VOLATILE: no replay on restart
        # gateway re-sends every 1 s
        self._strict = None             # {'labels', 'until'} while narrowed
        self._pending_strict = None
        self._strict_sub = None
        if callable(getattr(self._vocab_mod, 'parse_strict', None)):
            strict_qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                                    reliability=ReliabilityPolicy.RELIABLE,
                                    durability=DurabilityPolicy.VOLATILE)
            self._strict_dedupe = self._vocab_mod.Dedupe()
            self._strict_sub = self.create_subscription(
                String, '/yolo/strict', self._on_strict, strict_qos)
        else:
            self.get_logger().error('[strict] /yolo/strict disabled: '
                                    'vocab_state.py has no parse_strict')
        self.get_logger().info('[yoloe] %s subscribed, %s published '
                               '(backend %s)' % (TOPIC_CLASSES,
                                                 TOPIC_CLASSES_STATE,
                                                 self._backend_name))
        if self._fallback_reason is not None:
            self._publish_vocab_state(None, None,
                                      error=self._fallback_reason)

    def _on_classes(self, msg):
        # swap next tick
        with self._lock:
            req, err = self._vocab_mod.parse(msg.data, TRACKED_CLASSES)
            if self._pending_vocab is None:
                self._pending_vocab = []
            self._pending_vocab.append((req, err))

    def _publish_vocab_state(self, epoch, seq, dup=False, rejected=None,
                             swap_ms=None, error=None, extras=None,
                             model=None):
        if self._state_pub is None or self._vocab_mod is None:
            return None
        if extras is None:
            extras = list(self._extras)
        if model is None:
            model = (self._backend.model_name if self._backend is not None
                     else ENGINE_MODEL_NAME)
        state = self._vocab_mod.build_state(
            epoch, seq, dup, self._backend_name, model, self._base_order,
            extras, dict((lab, 'bank') for lab in extras),
            rejected or {}, self._extra_conf, swap_ms, time.time(), error)
        out = String()
        out.data = json.dumps(state)
        self._state_pub.publish(out)
        return state

    def _apply_pending_vocab(self):
        with self._lock:
            pending = self._pending_vocab
            self._pending_vocab = None
        for req, err in pending or ():
            try:
                self._handle_vocab_request(req, err)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(
                    '[yoloe] vocabulary request failed: %s' % e)

    def _handle_vocab_request(self, req, err):
        epoch, seq = req.get('epoch'), req.get('seq')
        if err is not None:
            self.get_logger().error('[yoloe] /yolo/classes rejected: %s'
                                    % err)
            self._publish_vocab_state(epoch, seq,
                                      rejected=req.get('rejected'),
                                      error=err)
            return
        if self._vocab_dedupe.is_dup(epoch, seq):
            self._publish_vocab_state(epoch, seq, dup=True, swap_ms=0.0)
            return
        op = req['op']
        if op == 'query':
            self._vocab_dedupe.commit(epoch, seq)
            self._publish_vocab_state(epoch, seq, swap_ms=0.0)
            return
        wanted = list(req['classes']) if op == 'set' else []
        rejected = dict(req['rejected'])
        if self._backend is None:
            for label, _ in wanted:
                rejected[label] = 'backend_fixed'
            self._vocab_dedupe.commit(epoch, seq)
            self._publish_vocab_state(
                epoch, seq, rejected=rejected, swap_ms=0.0,
                error=self._vocab_mod.FIXED_BACKEND_ERROR)
            return
        entries = [(b, b) for b in self._base_order] + wanted
        try:
            res = self._backend.set_vocab(entries)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('[yoloe] swap failed: %s' % e)
            self._publish_vocab_state(epoch, seq, rejected=rejected,
                                      error='%s' % (e,))
            return
        rejected.update(res.rejected)
        new_extras = {}
        for label, prompt in wanted:
            if label in res.resolved and label not in TRACKED_CLASSES:
                new_extras[label] = prompt
        with self._lock:
            self._extras = new_extras
            self._runtime_allow = frozenset(res.names)
            # left classes only
            # base untouched
            gone = [k for k, c in self._candidates.items()
                    if c['class_name'] not in self._runtime_allow]
            for k in gone:
                del self._candidates[k]
            if self._tracker is not None:
                # ids survive the swap
                self._tracker.drop_classes(self._runtime_allow)
            mc = req.get('min_conf')
            self._extra_conf = (mc if mc is not None
                                else self._extra_conf_default)
        self._vocab_dedupe.commit(epoch, seq)
        self._publish_vocab_state(epoch, seq, rejected=rejected,
                                  swap_ms=res.swap_ms)
        self.get_logger().info(
            '[yoloe] vocabulary seq=%s extras=%s rejected=%s swap_ms=%.1f '
            'dropped=%d' % (seq, ','.join(new_extras) or '-', rejected,
                            res.swap_ms, len(gone)))

    # ── /yolo/strict (CLASSES=1) ─────────
    # gateway narrows the WIDE vocab to a
    # few labels for one bounded phase
    # nothing here edits _extras /
    # _extra_conf: release or lease expiry
    # restores the operator's vocabulary
    def _on_strict(self, msg):
        # applied next tick
        with self._lock:
            req, err = self._vocab_mod.parse_strict(msg.data, TRACKED_CLASSES)
            if self._pending_strict is None:
                self._pending_strict = []
            self._pending_strict.append((req, err))

    def _wide_entries(self):
        """The full vocabulary, exactly as the last /yolo/classes swap built it."""
        return ([(b, b) for b in self._base_order]
                + list(self._extras.items()))

    def _publish_strict_state(self, epoch, seq, dup=False, rejected=None,
                              swap_ms=None, error=None):
        """A /yolo/classes_state for a strict request; carries 'strict' (the
        narrowed labels) only while the detector is narrowed."""
        if self._state_pub is None or self._vocab_mod is None:
            return None
        model = (self._backend.model_name if self._backend is not None
                 else ENGINE_MODEL_NAME)
        state = self._vocab_mod.build_state(
            epoch, seq, dup, self._backend_name, model, self._base_order,
            list(self._extras), dict((lab, 'bank') for lab in self._extras),
            rejected or {}, self._extra_conf, swap_ms, time.time(), error)
        if self._strict is not None:
            state['strict'] = list(self._strict['labels'])
        out = String()
        out.data = json.dumps(state)
        self._state_pub.publish(out)
        return state

    def _strict_step(self):
        """Every tick, BEFORE the frame: requests, wide changes, the lease."""
        with self._lock:
            pending = self._pending_strict
            self._pending_strict = None
        for req, err in pending or ():
            try:
                self._handle_strict_request(req, err)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error('[strict] request failed: %s' % e)
        if self._strict is None:
            return
        if self._backend is None or self._backend_name != 'yoloe':
            self._strict = None          # fixed engine has no strict set
            return
        if self._runtime_allow != frozenset(self._strict['labels']):
            self._strict_reapply('wide_changed')   # a /yolo/classes swap ran since
        if self._strict is not None and time.time() > self._strict['until']:
            self._strict_restore('lease_expired', None, None)

    def _strict_apply(self, entries):
        """Swap the backend to ``entries``; drop the candidates and tracks
        whose class left.  Returns (swap_ms, number of candidates dropped)."""
        res = self._backend.set_vocab(entries)
        with self._lock:
            allow = frozenset(res.names)
            self._runtime_allow = allow
            left = [k for k, c in self._candidates.items()
                    if c['class_name'] not in allow]
            for k in left:
                del self._candidates[k]
            if self._tracker is not None:
                self._tracker.drop_classes(allow)
        return res.swap_ms, len(left)

    def _handle_strict_request(self, req, err):
        epoch, seq = req.get('epoch'), req.get('seq')
        if err is not None:
            self.get_logger().error('[strict] /yolo/strict rejected: %s' % err)
            self._publish_strict_state(epoch, seq, error=err)
            return
        if self._strict_dedupe.is_dup(epoch, seq):
            self._publish_strict_state(epoch, seq, dup=True, swap_ms=0.0)
            return
        self._strict_dedupe.commit(epoch, seq)
        if self._backend is None or self._backend_name != 'yoloe':
            self._publish_strict_state(epoch, seq, swap_ms=0.0,
                                       error=self._vocab_mod.FIXED_BACKEND_ERROR)
            return
        if req['op'] == 'release':
            self._strict_restore('released', epoch, seq)
            return
        want = req['labels']
        entries = [(l, p) for l, p in self._wide_entries() if l in want]
        if not entries:
            self._publish_strict_state(epoch, seq, swap_ms=0.0,
                                       error='strict_no_label:%s' % ','.join(want))
            return
        labels = tuple(l for l, _ in entries)
        missing = dict((l, 'not_in_vocabulary') for l in want if l not in labels)
        until = time.time() + req['lease_s']
        if self._strict is not None and self._strict['labels'] == labels:
            self._strict['until'] = until          # renewal, no swap
            self._publish_strict_state(epoch, seq, rejected=missing, swap_ms=0.0)
            return
        try:
            swap_ms, n_left = self._strict_apply(entries)
        except Exception as e:  # noqa: BLE001
            self.get_logger().error('[strict] swap failed: %s' % e)
            self._publish_strict_state(epoch, seq,
                                       error='strict swap failed: %s' % (e,))
            return
        self._strict = {'labels': labels, 'until': until}
        self._publish_strict_state(epoch, seq, rejected=missing, swap_ms=swap_ms)
        self.get_logger().info(
            '[strict] vocabulary NARROWED seq=%s labels=%s missing=%s lease=%.1fs '
            'swap_ms=%.1f dropped=%d' % (seq, ','.join(labels),
                                         ','.join(missing) or '-',
                                         req['lease_s'], swap_ms, n_left))

    def _strict_restore(self, reason, epoch, seq):
        if self._strict is None:
            if epoch is not None:
                self._publish_strict_state(epoch, seq, swap_ms=0.0)
            return
        try:
            swap_ms, _n = self._strict_apply(self._wide_entries())
        except Exception as e:  # noqa: BLE001
            self._strict['until'] = time.time() + 1.0
            self.get_logger().error('[strict] restore FAILED (%s): %s - retry '
                                    'in 1 s' % (reason, e))
            return
        labels = self._strict['labels']
        self._strict = None
        self._publish_strict_state(epoch, seq, swap_ms=swap_ms)
        self.get_logger().info(
            '[strict] vocabulary RESTORED (%s) after %s extras=%s swap_ms=%.1f'
            % (reason, ','.join(labels), ','.join(self._extras) or '-', swap_ms))

    def _strict_reapply(self, reason):
        """A /yolo/classes swap ran while narrowed: narrow the NEW wide set."""
        want = self._strict['labels']
        entries = [(l, p) for l, p in self._wide_entries() if l in want]
        if not entries:
            self._strict = None
            self._publish_strict_state(None, None, swap_ms=0.0)
            self.get_logger().info('[strict] vocabulary RESTORED (%s): no strict '
                                   'label left' % reason)
            return
        try:
            swap_ms, n_left = self._strict_apply(entries)
        except Exception as e:  # noqa: BLE001
            self._strict = None
            self._publish_strict_state(None, None,
                                       error='strict re-narrow failed: %s' % (e,))
            self.get_logger().error('[strict] re-narrow failed (%s): %s'
                                    % (reason, e))
            return
        self._strict['labels'] = tuple(l for l, _ in entries)
        self._publish_strict_state(None, None, swap_ms=swap_ms)
        self.get_logger().info(
            '[strict] vocabulary RE-NARROWED (%s) labels=%s swap_ms=%.1f '
            'dropped=%d' % (reason, ','.join(self._strict['labels']), swap_ms,
                            n_left))

    # ── runtime fallback (10 errors) ─────
    def _start_engine_fallback(self):
        self._fallback_due = False
        self._busy = True
        self._backend_name = 'yolo'
        # (1) state FIRST - gateway cache and
        # verifier see extras go before rows
        # change. model pinned here, worker
        # clears _backend later.
        self._publish_vocab_state(
            None, None, extras=[], model=ENGINE_MODEL_NAME,
            error='fallback after %d consecutive inference errors'
            % YOLOE_FALLBACK_AFTER)
        with self._lock:
            self._extras = {}
            self._runtime_allow = TRACKED_CLASSES
            gone = [k for k, c in self._candidates.items()
                    if c['class_name'] not in TRACKED_CLASSES]
            for k in gone:
                del self._candidates[k]
            if self._tracker is not None:
                self._tracker.drop_classes(TRACKED_CLASSES)
        self.get_logger().error(
            '[yoloe] FALLBACK after %d consecutive inference errors; '
            'reloading %s on a worker thread' % (YOLOE_FALLBACK_AFTER,
                                                 ENGINE_PATH))
        # off-thread reload
        t = threading.Thread(target=self._reload_engine_worker,
                             name='yoloe-fallback', daemon=True)
        self._fallback_thread = t
        t.start()

    def _reload_engine_worker(self):
        try:
            model = self._load_engine()
        except Exception as e:  # noqa: BLE001
            self._fatal = e
            return
        with self._lock:
            self._gate_model = model
            self._backend = None
        self._busy = False
        self.get_logger().error('[yoloe] engine reloaded; vocabulary fixed '
                                'to the %d base classes'
                                % len(TRACKED_CLASSES))

    # ── stable instance ids (IDS=1) ──────
    def _init_instance_tracker(self):
        try:
            mod = _import_sibling('instance_tracker')
            self._tracker = mod.InstanceTracker(base=TRACKED_CLASSES)
        except Exception as e:  # noqa: BLE001
            self._tracker = None
            self.get_logger().error(
                '[ids] OPENBOT_INSTANCE_IDS disabled: instance_tracker not '
                'importable (%s) - class keying kept' % e)
            return
        self.get_logger().info(
            '[ids] instance ids ON: IoU>=%.2f class-agnostic, dup fusion '
            'IoU>=%.2f, evict %.1fs' % (self._tracker.iou_min,
                                        self._tracker.dup_iou,
                                        self._tracker.evict_s))

    def _track_frame(self, gate_results, names, suppressed, now):
        """{box index: (instance_id, majority class, track_age_s)} over the
        boxes that pass EXACTLY the main loop's admit filters; a box fused
        into a duplicate has no entry."""
        items = []
        for i, box in enumerate(gate_results.boxes):
            conf = float(box.conf)
            if conf < GATE_CONF:
                continue
            class_id = int(box.cls)
            if names is None:
                class_name = COCO_NAMES[class_id]
            else:
                class_name = _name_at(names, class_id)
                if i in suppressed:
                    continue
            if class_name not in self._runtime_allow:
                continue
            if (names is not None and class_name not in TRACKED_CLASSES
                    and conf < self._extra_conf):
                continue
            items.append((i, (class_name, box.xyxy[0].tolist(), conf)))
        # dup fusion uses the extra floor
        # tap logs the drops
        self._tracker.extra_floor = self._extra_conf
        self._tracker.tap = ((lambda m: self.get_logger().info(m))
                             if _env_on('OPENBOT_YOLOE_SCORE_TAP') else None)
        res = self._tracker.update_indexed([b for _, b in items], now)
        out = {}
        for k, iid, cls, _xyxy, _conf, _carried in res:
            age = self._tracker.age(iid)
            out[items[k][0]] = (iid, cls, round(age, 3))
        return out

    def _yoloe_suppressed(self, gate_results, names):
        """Indices removed by the cross-class IoU>=0.7 dedupe (YOLOE only),
        over the boxes that pass the same admit filters as the main loop."""
        items = []
        for i, box in enumerate(gate_results.boxes):
            conf = float(box.conf)
            if conf < GATE_CONF:
                continue
            class_name = _name_at(names, int(box.cls))
            if class_name not in self._runtime_allow:
                continue
            # same admit set - redundant under
            # base-first / conf-desc order; a
            # sub-floor extra only beats other
            # sub-floor extras
            # (test_yolo_node_yoloe_mutations)
            if class_name not in TRACKED_CLASSES and conf < self._extra_conf:
                continue
            items.append((i, class_name, conf, box.xyxy[0].tolist()))
        if not _env_on('OPENBOT_YOLOE_SCORE_TAP'):
            return cross_class_suppressed(items, extra_floor=self._extra_conf)
        # SCORE_TAP=1: node LOG only, no topic
        # every box >= 0.15 plus what the
        # admit filters and dedupe did;
        # engine runner logs top-3 raw
        # scores per class - join top-1
        drops = []
        sup = cross_class_suppressed(items, extra_floor=self._extra_conf,
                                     drops=drops)
        why = dict((d[0][0], d) for d in drops)
        lg = self.get_logger()
        admitted = set(it[0] for it in items)
        for i, box in enumerate(gate_results.boxes):
            conf = float(box.conf)
            if conf < 0.15:
                continue
            cn = _name_at(names, int(box.cls))
            if i in why:
                d = why[i]
                fate = ('DROPPED by box %d %s conf=%.3f IoU=%.2f (rank %d '
                        'vs %d)' % (d[1][0], d[1][1], d[1][2], d[2],
                                    d[3], d[4]))
            elif i in admitted:
                fate = 'kept'
            elif conf < GATE_CONF:
                fate = 'below gate %.2f' % GATE_CONF
            elif cn not in self._runtime_allow:
                fate = 'not in allowlist'
            else:
                fate = 'below extra floor %.2f' % self._extra_conf
            lg.info('[score_tap] box %d %s conf=%.3f xyxy=%s %s'
                    % (i, cn, conf,
                       [round(v) for v in box.xyxy[0].tolist()], fate))
        return sup


# ════════════════════════════════════════════════════════════════════════
# helpers after the class, so
# the code above stays byte-identical
# (call-time resolution)
ENGINE_PATH = '/home/unitree/yolo11n-seg.engine'

# ── env switches (read once) ──────────
# absent or any value but '1' = OFF
# all off -> plain node: engine above,
# COCO_NAMES names, TRACKED_CLASSES
# allowlist, one sub + one pub + timer
# (the plain node is the reference;
# test_yolo_node_yoloe_off.py pins it)
#   OPENBOT_YOLOE=1            YOLOE .pt + bank (yoloe_backend.py)
#   OPENBOT_YOLOE_WEIGHTS      the .pt path
#   OPENBOT_YOLOE_BANK         bank NPZ (default: beside the .pt)
#   OPENBOT_YOLOE_THREADS      torch.set_num_threads (default 2)
#   OPENBOT_YOLOE_EXTRA_CONF   EXTRA class floor (default 0.35);
#                              base classes stick to GATE_CONF
#   OPENBOT_YOLO_CLASSES=1     /yolo/classes + /yolo/classes_state
#   OPENBOT_INSTANCE_IDS=1     instance_tracker.py owns the key
#                              (one candidate per track); rows gain
#                              instance_id + track_age_s
YOLOE_DEFAULT_WEIGHTS = '/home/unitree/leon/yoloe/yoloe-26n-seg.pt'
YOLOE_DEFAULT_THREADS = 2
YOLOE_DEFAULT_EXTRA_CONF = 0.35       # same as config.yaml det_conf_floor
YOLOE_FALLBACK_AFTER = 10             # errors, ~1 s
CROSS_CLASS_DEDUPE_IOU = 0.7
TOPIC_CLASSES = '/yolo/classes'
TOPIC_CLASSES_STATE = '/yolo/classes_state'
ENGINE_MODEL_NAME = 'yolo11n-seg'


def _env_on(name, environ=None):
    env = os.environ if environ is None else environ
    return env.get(name) == '1'


def _yoloe_env(environ=None):
    """(weights, bank, threads, extra_conf, problems) from the environment.
    A malformed value falls back to its default and is REPORTED."""
    env = os.environ if environ is None else environ
    problems = []
    weights = env.get('OPENBOT_YOLOE_WEIGHTS') or YOLOE_DEFAULT_WEIGHTS
    bank = env.get('OPENBOT_YOLOE_BANK')
    if not bank:
        stem = os.path.basename(weights)
        if stem.endswith('.pt'):
            stem = stem[:-3]
        bank = os.path.join(os.path.dirname(weights),
                            'vocab_bank_%s.npz' % stem)
    threads = YOLOE_DEFAULT_THREADS
    raw = env.get('OPENBOT_YOLOE_THREADS')
    if raw:
        try:
            threads = int(raw)
            if threads < 1:
                raise ValueError('must be >= 1')
        except ValueError as e:
            problems.append('OPENBOT_YOLOE_THREADS=%r ignored (%s)' % (raw, e))
            threads = YOLOE_DEFAULT_THREADS
    extra_conf = YOLOE_DEFAULT_EXTRA_CONF
    raw = env.get('OPENBOT_YOLOE_EXTRA_CONF')
    if raw:
        try:
            extra_conf = float(raw)
            if not (GATE_CONF <= extra_conf <= 0.90):
                raise ValueError('outside [%.2f, 0.90]' % GATE_CONF)
        except ValueError as e:
            problems.append('OPENBOT_YOLOE_EXTRA_CONF=%r ignored (%s)'
                            % (raw, e))
            extra_conf = YOLOE_DEFAULT_EXTRA_CONF
    return weights, bank, threads, extra_conf, problems


def _import_sibling(name):
    """Import a stdlib module shipped co-located with this node (box flat
    layout) -- same search order as the instance_labels import above."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as first:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (here, os.path.dirname(here),
                     os.path.dirname(os.path.dirname(here))):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            try:
                return importlib.import_module(name)
            except ImportError:
                continue
        raise first


def _name_at(names, idx):
    """results.names lookup (dict or list); None when out of range."""
    try:
        if isinstance(names, dict):
            return names.get(idx)
        if 0 <= idx < len(names):
            return names[idx]
    except Exception:  # noqa: BLE001
        return None
    return None


def _iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
          - inter)
    return inter / ua if ua > 0.0 else 0.0


