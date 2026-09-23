#!/usr/bin/env python3
"""yoloe_trt_runner.py -- run the live-vocabulary YOLOE TensorRT engine (batch 6.2).

The engine is built ON THE ORIN from yoloe-26n-seg-live36-384x640.onnx (trtexec --fp16):
  inputs   images [1, 3, H, W] float (letterboxed, RGB, 0..1)
           tpe    [1, NMAX, 512] float (the vocabulary bank rows = YOLOE get_text_pe output;
                                        spare rows are COPIES of active rows, never zeros)
  outputs  output0 [1, A, 4 + NMAX + 32]  RAW head rows, one per anchor (A = 5040 at 384x640):
                   decoded box(4), sigmoid class scores(NMAX), mask coefficients(32)
           output1 [1, 32, H/4, W/4] mask prototypes
The top-k is done HERE, on the host, in torch (ultralytics' own agnostic get_topk_index): MEASURED on
the Orin 2026-10-05 (TensorRT 8.5, trtexec alone, FP16 and FP32) an engine with the in-graph top-k
returned correct boxes but conf and class columns of 0.0 on every row.  An engine whose output0 already
is end2end rows [1, 300, 6 + 32] is still accepted (no host top-k).
A vocabulary change only changes ``tpe``: no rebuild.  The shapes are read from the engine.

I/O buffers are CUDA tensors, exactly as ultralytics runs today's ``yolo11n-seg.engine`` on the
robot.  The binding API (num_bindings / execute_v2) is used whenever the engine has it -- the
same rule as ultralytics' AutoBackend, i.e. the path the robot's yolo11n engine runs on TensorRT 8.5;
only TensorRT >= 10 (which removed num_bindings) uses the tensor API (set_tensor_address /
execute_async_v3).  TensorRT 8.5 exposes BOTH; picking the tensor API there was MEASURED on the Orin
2026-10-05 to return an all-zero output0 (no detection on any frame, FP16 engine).
An ultralytics metadata header (4-byte little-endian length + JSON), if present, is skipped.

torch / tensorrt are imported lazily (both injectable for tests).  py3.8.
"""
import json
import os


def _score_tap_on():
    """OPENBOT_YOLOE_SCORE_TAP=1: log the raw per-class scores to the node log.
    Default off: no extra work, no extra output."""
    return os.environ.get('OPENBOT_YOLOE_SCORE_TAP') == '1'     # same rule as the node's _env_on


def _stderr_line(msg):
    import sys
    sys.stderr.write(msg + '\n')
    sys.stderr.flush()


class EngineUnavailable(Exception):
    """The engine cannot be loaded or run; the caller falls back."""


def split_ultralytics_header(blob):
    """(metadata dict | None, engine bytes).  A blob without a plausible header is returned whole."""
    if len(blob) > 8:
        n = int.from_bytes(blob[:4], byteorder='little')
        if 2 <= n <= min(len(blob) - 4, 1 << 20):
            head = blob[4:4 + n]
            if head[:1] == b'{':
                try:
                    return json.loads(head.decode('utf-8')), blob[4 + n:]
                except (UnicodeDecodeError, ValueError):
                    pass
    return None, blob


class TrtRunner(object):
    INPUTS = ('images', 'tpe')
    OUTPUTS = ('output0', 'output1')

    def __init__(self, engine_path, device='cuda:0', torch_mod=None, trt_mod=None, log=print):
        self.engine_path = engine_path
        self.device = device
        self.log = log
        self._torch = torch_mod
        self._trt = trt_mod
        self.meta = None
        self.shapes = {}
        self._buf = {}
        self._legacy = None
        self._ctx = None
        self._engine = None
        self._ptrs = None

    def load(self):
        if not os.path.isfile(self.engine_path):
            raise EngineUnavailable('engine not found: %s' % self.engine_path)
        torch = self._torch
        if torch is None:
            try:
                import torch  # noqa: F811 -- a lazy, heavy import that only works on the box
            except Exception as e:  # noqa: BLE001
                raise EngineUnavailable('torch not importable: %s' % (e,))
            self._torch = torch
        cuda = getattr(torch, 'cuda', None)
        if not str(self.device).startswith('cuda') or cuda is None or not cuda.is_available():
            raise EngineUnavailable('a TensorRT engine runs on the GPU only (device %r, CUDA %s)'
                                    % (self.device, 'unavailable' if cuda is None or not cuda.is_available()
                                       else 'available'))
        trt = self._trt
        if trt is None:
            try:
                import tensorrt as trt  # noqa: F811
            except Exception as e:  # noqa: BLE001
                raise EngineUnavailable('tensorrt not importable: %s' % (e,))
            self._trt = trt
        with open(self.engine_path, 'rb') as f:
            self.meta, blob = split_ultralytics_header(f.read())
        logger = trt.Logger(trt.Logger.WARNING)
        engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        if engine is None:
            raise EngineUnavailable('deserialize_cuda_engine returned None (built for another '
                                    'TensorRT version or device?)')
        ctx = engine.create_execution_context()
        self._engine, self._ctx = engine, ctx
        self._legacy = hasattr(engine, 'num_bindings')     # TensorRT < 10 (8.5 has both APIs)
        names = []
        if self._legacy:
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                names.append((i, name, tuple(engine.get_binding_shape(i)),
                              trt.nptype(engine.get_binding_dtype(i)), engine.binding_is_input(i)))
        else:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                names.append((i, name, tuple(engine.get_tensor_shape(name)),
                              trt.nptype(engine.get_tensor_dtype(name)),
                              engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT))
        got = sorted(n for _, n, _, _, _ in names)
        if got != sorted(self.INPUTS + self.OUTPUTS):
            raise EngineUnavailable('engine I/O %r, expected %r' % (got, sorted(self.INPUTS + self.OUTPUTS)))
        ptrs = [0] * len(names)
        for i, name, shape, npdt, _is_in in names:
            if any(int(d) < 1 for d in shape):
                raise EngineUnavailable('dynamic shape %r for %s: build the engine with static shapes'
                                        % (shape, name))
            tdt = torch.float16 if str(npdt) in ('float16', "<class 'numpy.float16'>") else torch.float32
            buf = torch.empty(tuple(int(d) for d in shape), dtype=tdt, device=self.device)
            self._buf[name] = buf
            self.shapes[name] = tuple(int(d) for d in shape)
            ptrs[i] = int(buf.data_ptr())
            if not self._legacy and ctx.set_tensor_address(name, int(buf.data_ptr())) is False:
                raise EngineUnavailable('set_tensor_address refused %s' % name)
        self._ptrs = ptrs
        self.log('[yoloe] tensorrt engine %s: images %s tpe %s api=%s'
                 % (os.path.basename(self.engine_path), self.shapes['images'], self.shapes['tpe'],
                    'bindings' if self._legacy else 'tensors'))
        return self

    @property
    def imgsz(self):
        return self.shapes['images'][2], self.shapes['images'][3]

    @property
    def nmax(self):
        return self.shapes['tpe'][1]

    def infer(self, images, tpe):
        """images [1,3,H,W] and tpe [1,NMAX,512] (numpy float32 or torch); returns (output0, output1)
        as float32 torch tensors on the runner's device.  Raises EngineUnavailable on a failed run."""
        torch = self._torch
        for name, arr in (('images', images), ('tpe', tpe)):
            t = arr if hasattr(arr, 'data_ptr') else torch.from_numpy(arr)
            if tuple(t.shape) != self.shapes[name]:
                raise ValueError('%s shape %r, engine wants %r' % (name, tuple(t.shape), self.shapes[name]))
            self._buf[name].copy_(t.to(self._buf[name].dtype))
        if self._legacy:
            ok = self._ctx.execute_v2(self._ptrs)
        else:
            stream = torch.cuda.current_stream(self.device)
            ok = self._ctx.execute_async_v3(stream.cuda_stream)
        torch.cuda.synchronize(self.device)
        if ok is False:
            raise EngineUnavailable('TensorRT execution failed')
        return tuple(self._buf[n].float().clone() for n in self.OUTPUTS)


class YoloeEngineModel(object):
    """Stands in for ultralytics' YOLOE model inside yoloe_backend.YoloeBackend when the weights are a
    TensorRT ``.engine``: the same ``set_classes(names, embeddings)`` and
    ``model(img, conf=, iou=, verbose=, device=)`` -> [Results] contract, so the backend's vocabulary,
    fallback and device rules stay one code path.

    Pre-processing is ultralytics' own for a static model (LetterBox to the engine size, auto=False,
    BGR->RGB, CHW, /255); post-processing is ultralytics' own SegmentationPredictor.postprocess on the
    end2end rows.

    SPARE SLOTS ARE COPIES, NEVER ZEROS.  The head L2-normalises every vocabulary row; an all-zero row
    is 0/(0+eps) and in FP16 the eps (1e-12) underflows to 0 -> NaN, and one NaN class score poisons the
    per-box max: MEASURED on the Orin 2026-10-05, the FP16 engine returned NO detection on 60/60 frames
    (FP32 was unaffected); reproduced on the laptop in torch FP16 (300/300 NaN rows; none with real
    rows).  So slot j >= N holds a copy of row j % N, and a detection on slot j is relabelled j % N:
    a copy scores exactly as its original, so the label is always the right one.  Laptop
    proof: ONNX of the same graph vs YOLOE .pt, 60 robot frames x 4 vocabularies, box IoU 1.0000,
    dconf 0.0001, mask IoU 1.0000 (.wf/b6/trt/t2live_opus/equiv_live.py)."""

