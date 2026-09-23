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
