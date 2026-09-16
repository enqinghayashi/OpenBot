#!/usr/bin/env python3
"""yoloe_engine_diag.py -- batch 6.2: why does an engine return no detection?  READ-ONLY, one frame.

Prints which yoloe_trt_runner.py is imported (path + md5), then runs ONE recorded frame through the
engine with three vocabulary tensors and reports the raw output0 rows (before any post-processing):
  A  the runner's own tpe (what the backend sends: base 20 + the runner's spare-slot rule)
  B  spare slots = ZERO rows (the original runner)
  C  spare slots = copies of row 0 (the fix, built here independently of the runner)
For each: NaN count in conf / boxes / mask coefficients, max conf, rows with conf >= 0.35, the top 5
(conf, class slot), and whether output1 (mask prototypes) holds NaN.  No verdict words.
Then, on a real TensorRT engine, the copy-pad tensor once per TensorRT API (bindings = execute_v2,
tensors = execute_async_v3) with both output buffers pre-filled with the sentinel -7.0:
'unwritten' = how many output values still hold -7.0 after the run (the engine never wrote them).

  PYTHONPATH=/home/unitree/leon/yoloe_site:<foxy paths> /home/unitree/leon/jp_env/bin/python3 \\
    yoloe_engine_diag.py --engine $Y/yoloe-26n-seg.engine --bank $Y/vocab_bank_yoloe-26n-seg.npz \\
      --node-dir /home/unitree/leon/openbot --frame $Y/probe_e2c/jp_env/frames/frame_000.png
Python 3.8.
"""
import argparse
import hashlib
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--node-dir', required=True, help='directory holding yoloe_backend.py + yoloe_trt_runner.py')
    ap.add_argument('--frame', required=True)
    ap.add_argument('--reference', required=True, help='yoloe_reference_384x640.json (its base vocabulary)')
    ap.add_argument('--dump-dir', default=None, help='write images.bin + tpe.bin (raw float32, copy-pad) for trtexec')
    ap.add_argument('--ort-onnx', default=None, help='LAPTOP REHEARSAL ONLY: onnxruntime instead of TensorRT')
    a = ap.parse_args(argv)
    sys.path.insert(0, a.node_dir)
    import json
    import cv2
    import numpy as np
    import torch
    import yoloe_backend as YB
    import yoloe_trt_runner as TR
    print('RUNNER %s md5 %s' % (TR.__file__, hashlib.md5(open(TR.__file__, 'rb').read()).hexdigest()))
    print('BACKEND %s md5 %s' % (YB.__file__, hashlib.md5(open(YB.__file__, 'rb').read()).hexdigest()))
    base = list(json.load(open(a.reference))['vocab'])
    with np.load(a.bank, allow_pickle=False) as z:
        prompts = [str(p) for p in z['prompts'].tolist()]
        emb = np.asarray(z['embeds'], np.float32).reshape(-1, 512)
    idx = {p: i for i, p in enumerate(prompts)}
    kw = {}
    if a.ort_onnx:
        import onnxruntime as ort

        class _Ort(object):
            def __init__(self, path):
                self.s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                self.shapes = {i.name: tuple(i.shape) for i in self.s.get_inputs()}
            imgsz = property(lambda self: (self.shapes['images'][2], self.shapes['images'][3]))
            nmax = property(lambda self: self.shapes['tpe'][1])

            def infer(self, images, tpe):
                o0, o1 = self.s.run(None, {'images': images.numpy(), 'tpe': tpe.numpy()})
                return torch.from_numpy(o0), torch.from_numpy(o1)
        kw['engine_ctor'] = lambda p: TR.YoloeEngineModel(_Ort(a.ort_onnx), torch, device='cuda:0')
        print('REHEARSAL: onnxruntime, not TensorRT')
    be = YB.YoloeBackend(a.engine, a.bank, base=base, log=print, **kw)
    be.load()
    model = be._model
    runner, cap = model.runner, model.capacity
    rows = np.stack([emb[idx[b]] for b in base])
    n = len(base)
    zero = np.zeros((1, cap, 512), np.float32)
    zero[0, :n] = rows
    copy = np.repeat(rows[:1][None], cap, axis=1).astype(np.float32)
    copy[0, :n] = rows
    own = model._tpe.numpy() if hasattr(model._tpe, 'numpy') else np.asarray(model._tpe)
    print('TPE_A zero_rows=%d (of %d)' % (int((np.abs(own[0]).sum(-1) == 0).sum()), cap))
    img = cv2.imread(a.frame)
    model._post()
    x = model._letterbox(image=img)[:, :, ::-1].transpose(2, 0, 1)[None]
    x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32) / 255.0)
    print('INPUT %s min %.3f max %.3f' % (tuple(x.shape), float(x.min()), float(x.max())))
    if a.dump_dir:
        import os
        np.ascontiguousarray(x.numpy(), dtype='<f4').tofile(os.path.join(a.dump_dir, 'images.bin'))
        np.ascontiguousarray(copy, dtype='<f4').tofile(os.path.join(a.dump_dir, 'tpe.bin'))
        print('DUMPED %s/images.bin %s/tpe.bin' % (a.dump_dir, a.dump_dir))
    for tag, tpe in (('A runner-own', own), ('B zero-pad', zero), ('C copy-pad', copy)):
        o0, o1 = runner.infer(x, torch.from_numpy(np.ascontiguousarray(tpe)))
        if int(o0.shape[-1]) == 4 + cap + int(o1.shape[1]):           # raw head: host does top-k
            print('%-13s RAW output0 %s class-score max=%.4f'
                  % (tag, tuple(o0.shape), float(torch.nan_to_num(o0[0, :, 4:4 + n], nan=-1).max())))
            o0 = model.topk_rows(o0, n, cap, int(o1.shape[1]))
        o0 = o0[0].cpu()
        o1c = o1.cpu()
        print('%-13s STATS box|max|=%.4g cls_max=%.4g coef|max|=%.4g proto|max|=%.4g proto_std=%.4g'
              % (tag, float(o0[:, :4].abs().max()), float(o0[:, 5].max()), float(o0[:, 6:].abs().max()),
                 float(o1c.abs().max()), float(o1c.std())))
        c = o0[:, 4]
        top = sorted(((float(v), int(o0[k, 5])) for k, v in enumerate(c) if v == v), reverse=True)[:5]
        print('%-13s conf_nan=%d box_nan=%d coef_nan=%d proto_nan=%d max_conf=%s rows>=0.35=%d top5=%s'
              % (tag, int(torch.isnan(c).sum()), int(torch.isnan(o0[:, :4]).sum()), int(torch.isnan(o0[:, 6:]).sum()),
                 int(torch.isnan(o1).sum()), round(float(torch.nan_to_num(c, nan=-1).max()), 4),
                 int((c >= 0.35).sum()), [(round(v, 3), k) for v, k in top]))
    if hasattr(runner, '_ctx') and runner._ctx is not None:
        ctx, eng = runner._ctx, runner._engine
        for api in ('bindings', 'tensors'):
            if api == 'bindings' and not hasattr(eng, 'num_bindings'):
                print('API %-8s not offered by this TensorRT' % api)
                continue
            if api == 'tensors' and not hasattr(ctx, 'execute_async_v3'):
                print('API %-8s not offered by this TensorRT' % api)
                continue
            runner._legacy = api == 'bindings'
            sets = []
            try:
                if api == 'tensors':
                    sets = [ctx.set_tensor_address(nm, int(b.data_ptr())) for nm, b in runner._buf.items()]
                for nm in runner.OUTPUTS:
                    runner._buf[nm].fill_(-7.0)
                torch.cuda.synchronize()
                o0, o1 = runner.infer(x, torch.from_numpy(np.ascontiguousarray(copy)))
                unw0 = int((o0 == -7.0).sum())
                if int(o0.shape[-1]) == 4 + cap + int(o1.shape[1]):
                    o0 = model.topk_rows(o0, n, cap, int(o1.shape[1]))
                c = o0[0, :, 4].cpu()
                print('API %-8s set_tensor_address=%s unwritten out0=%d out1=%d max_conf=%s rows>=0.35=%d'
                      % (api, sets or '-', unw0, int((o1 == -7.0).sum()),
                         round(float(torch.nan_to_num(c, nan=-1).max()), 4), int((c >= 0.35).sum())))
            except Exception as e:  # noqa: BLE001 - we report it, never let it kill us
                print('API %-8s ERROR %s: %s' % (api, type(e).__name__, e))
        runner._legacy = hasattr(eng, 'num_bindings')
    r = be.infer(img, 0.20, 0.45)
    print('BACKEND_INFER dets=%d %s' % (len(r.boxes), [(r.names[int(k)], round(float(s), 3))
                                                       for k, s in zip(r.boxes.cls, r.boxes.conf)][:8]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
