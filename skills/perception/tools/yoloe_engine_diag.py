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


