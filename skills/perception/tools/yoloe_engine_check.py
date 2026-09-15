#!/usr/bin/env python3
"""yoloe_engine_check.py -- batch 6.2 on-robot acceptance for the YOLOE TensorRT engine.

READ-ONLY for the robot: it publishes nothing and moves nothing; it reads recorded PNG frames, runs the
backend in engine mode (yoloe_backend.YoloeBackend -> yoloe_trt_runner) exactly as the node would, and
writes one JSON result.  Run it from a clean shell with the node's interpreter and the ultralytics overlay:

  PYTHONPATH=/home/unitree/leon/yoloe_site:<foxy paths> /home/unitree/leon/jp_env/bin/python3 \\
    yoloe_engine_check.py --engine $Y/yoloe-26n-seg.engine --bank $Y/vocab_bank_yoloe-26n-seg.npz \\
      --node /home/unitree/leon/openbot/yolo_node.py --reference $Y/yoloe_reference_384x640.json \\
      --frames $Y/probe_e2c/jp_env/frames --frames $Y/probe_e2b/jp_env/frames \\
      --soak-s 600 --hz 10 --out $Y/engine_check.json

Three parts, every number printed and saved, NO verdict words (read them against the runbook):
  1. AGREEMENT with the laptop's exact YOLOE .pt results (FP32) on the same frames: per frame the
     node-level label set (conf >= 0.35 after the node's own cross-class dedupe), matched box IoU,
     |dconf|, and every flip with the reference confidence of the object involved (an FP16 engine flips
     borderline objects; the runbook judges flips by how far the object sits from a threshold).
  2. LATENCY of backend.infer() (letterbox + engine + post-processing) over --reps calls.
  3. SOAK at --hz for --soak-s seconds: errors, latency drift, GPU memory (torch.cuda) start/end/max.
Python 3.8.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
import time


def _node_helpers(node_path):
    """TRACKED_CLASSES, CROSS_CLASS_DEDUPE_IOU, _iou_xyxy, cross_class_suppressed from the node source
    (exec of those blocks only: the node module itself imports rclpy)."""
    lines = open(node_path).read().splitlines(True)
    ns = {}

    def block(pat):
        k = next(n for n, l in enumerate(lines) if re.match(pat, l))
        out = [lines[k]]
        for l in lines[k + 1:]:
            if l.strip() and not l[0].isspace() and not l.startswith(('})', ')')):
                break
            out.append(l)
        return ''.join(out)
    for pat in (r'TRACKED_CLASSES = ', r'CROSS_CLASS_DEDUPE_IOU = ', r'def _iou_xyxy', r'def cross_class_suppressed'):
        exec(block(pat), ns)
    return ns


def _dets(r):
    return [(r.names[int(c)], float(s), [float(v) for v in b.tolist()])
            for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


def _pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * q))] if s else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--node', required=True, help="the robot's yolo_node.py (dedupe rule + base order)")
    ap.add_argument('--reference', required=True, help='laptop .pt results JSON (yoloe_reference_*.json)')
    ap.add_argument('--frames', action='append', required=True)
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--soak-s', type=float, default=600.0)
    ap.add_argument('--hz', type=float, default=10.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ort-onnx', default=None,
                    help='LAPTOP REHEARSAL ONLY: run this ONNX through onnxruntime instead of TensorRT')
    a = ap.parse_args(argv)

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in (os.path.dirname(os.path.abspath(a.node)), here):
        if d not in sys.path:
            sys.path.insert(0, d)
    import cv2
    import torch
    import yoloe_backend as YB
    ns = _node_helpers(a.node)
    ccs, iou = ns['cross_class_suppressed'], ns['_iou_xyxy']
    ref = json.load(open(a.reference))
    vocab = ref['vocab']
    conf, iou_thr = ref['conf'], ref['iou']
    logs = []
    kw = {}
    if a.ort_onnx:                                   # laptop rehearsal - onnxruntime, not TRT
        import onnxruntime as ort
        import yoloe_trt_runner as TRR

        class _Ort(object):
            def __init__(self, path):
                self.s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.shapes = {i.name: tuple(i.shape) for i in self.s.get_inputs()}
            imgsz = property(lambda self: (self.shapes['images'][2], self.shapes['images'][3]))
            nmax = property(lambda self: self.shapes['tpe'][1])

            def infer(self, images, tpe):
                o0, o1 = self.s.run(None, {'images': images.numpy(), 'tpe': tpe.numpy()})
                return torch.from_numpy(o0), torch.from_numpy(o1)
        kw['engine_ctor'] = lambda p: TRR.YoloeEngineModel(_Ort(a.ort_onnx), torch, device='cuda:0')
        out_note = 'REHEARSAL: onnxruntime, not TensorRT'
    else:
        out_note = 'tensorrt'
    be = YB.YoloeBackend(a.engine, a.bank, base=vocab, log=logs.append, **kw)
    t0 = time.time()
    be.load()
    load_s = time.time() - t0
    for line in logs:
        print(line)
    res = be.set_vocab([(p, p) for p in vocab])
    out = {'runtime': out_note, 'engine': a.engine, 'load_s': round(load_s, 2), 'log': logs, 'vocab_active': res.names,
           'rejected': res.rejected, 'reference': a.reference}

    def node_view(D):
        items = [(k, l, c, b) for k, (l, c, b) in enumerate(D) if c >= 0.20]
        drop = set(ccs(items))
        return sorted(l for k, (l, c, b) in enumerate(D) if c >= 0.35 and k not in drop)

    # part 1: agreement
    frames = []
    for d in a.frames:
        frames += sorted(glob.glob(os.path.join(d, '*.png')))
    rows, ious, dconfs, flips, missing_ref = [], [], [], [], []
    for f in frames:
        # reference key: <probe dir>/<frame>.png
        key = '%s/%s' % (os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f)))),
                         os.path.basename(f))
        R = ref['frames'].get(key)
        if R is None:
            missing_ref.append(key)
            continue
        P = [(d['label'], d['conf'], d['xyxy']) for d in R]
        O = _dets(be.infer(cv2.imread(f), conf, iou_thr))
        for lp, cp, bp in P:
            m = [(iou(bp, bo), abs(co - cp)) for lo, co, bo in O if lo == lp]
            if m:
                x = max(m)
                ious.append(x[0])
                dconfs.append(x[1])
        vp, vo = node_view(P), node_view(O)
        if vp != vo:
            changed = sorted(set(vp) ^ set(vo))
            near = {l: [round(c, 3) for ll, c, _ in P + O if ll == l] for l in changed}
            flips.append({'frame': key, 'reference': vp, 'engine': vo, 'confs_of_changed': near})
        rows.append(key)
    out['agreement'] = {
        'frames': len(rows), 'frames_without_reference': missing_ref,
        'node_level_flips': len(flips), 'flips': flips,
        'box_iou_min': round(min(ious), 4) if ious else None,
        'box_iou_p05': round(_pct(ious, 0.05), 4) if ious else None,
        'dconf_median': round(statistics.median(dconfs), 4) if dconfs else None,
        'dconf_p95': round(_pct(dconfs, 0.95), 4) if dconfs else None,
        'dconf_max': round(max(dconfs), 4) if dconfs else None}
    if missing_ref or not rows:
        print('WARNING %d frame(s) had no reference entry and were NOT compared (e.g. %s); compared %d'
              % (len(missing_ref), missing_ref[:2], len(rows)))
    print('AGREEMENT frames=%d flips=%d box_iou_min=%s dconf_median=%s dconf_p95=%s dconf_max=%s'
          % (len(rows), len(flips), out['agreement']['box_iou_min'], out['agreement']['dconf_median'],
             out['agreement']['dconf_p95'], out['agreement']['dconf_max']))
    for fl in flips:
        print('  FLIP %s reference=%s engine=%s confs=%s' % (fl['frame'], fl['reference'], fl['engine'],
                                                             fl['confs_of_changed']))

    # part 2: latency
    imgs = [cv2.imread(f) for f in frames[:30]] or [None]
    lat = []
    for k in range(a.reps):
        t = time.perf_counter()
        be.infer(imgs[k % len(imgs)], conf, iou_thr)
        lat.append((time.perf_counter() - t) * 1000.0)
    out['latency_ms'] = {'reps': a.reps, 'median': round(statistics.median(lat), 2),
                         'p90': round(_pct(lat, 0.90), 2), 'max': round(max(lat), 2)}
    print('LATENCY reps=%d median=%.2f p90=%.2f max=%.2f ms' % (a.reps, out['latency_ms']['median'],
                                                                out['latency_ms']['p90'], out['latency_ms']['max']))

    # part 3: soak
    period = 1.0 / a.hz
    mem0 = torch.cuda.memory_allocated()
    mem_max, errors, n, slow = mem0, [], 0, []
    t_end = time.time() + a.soak_s
    t_next = time.time()
    first, last = [], []
    while time.time() < t_end:
        t = time.perf_counter()
        try:
            be.infer(imgs[n % len(imgs)], conf, iou_thr)
        except Exception as e:  # noqa: BLE001 - counted, never fatal
            errors.append('%s: %s' % (type(e).__name__, e))
        ms = (time.perf_counter() - t) * 1000.0
        (first if n < 100 else last).append(ms)
        if len(last) > 100:
            last.pop(0)
        if ms > 1000.0 * period:
            slow.append(round(ms, 1))
        n += 1
        mem_max = max(mem_max, torch.cuda.memory_allocated())
        t_next += period
        time.sleep(max(0.0, t_next - time.time()))
    mem1 = torch.cuda.memory_allocated()
    out['soak'] = {'seconds': a.soak_s, 'hz': a.hz, 'calls': n, 'errors': len(errors), 'error_samples': errors[:5],
                   'over_budget_calls': len(slow), 'median_first100_ms': round(statistics.median(first), 2) if first else None,
                   'median_last100_ms': round(statistics.median(last), 2) if last else None,
                   'gpu_mem_mb_start': round(mem0 / 1e6, 1), 'gpu_mem_mb_end': round(mem1 / 1e6, 1),
                   'gpu_mem_mb_max': round(mem_max / 1e6, 1)}
    s = out['soak']
    print('SOAK calls=%d errors=%d over_budget=%d median_first100=%s median_last100=%s gpu_mem_mb start=%s end=%s max=%s'
          % (s['calls'], s['errors'], s['over_budget_calls'], s['median_first100_ms'], s['median_last100_ms'],
             s['gpu_mem_mb_start'], s['gpu_mem_mb_end'], s['gpu_mem_mb_max']))
    with open(a.out, 'w') as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print('WROTE %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
