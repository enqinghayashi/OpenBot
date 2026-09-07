#!/usr/bin/env python3
"""trt_add_ultralytics_meta.py -- Prepend ultralytics metadata header to TensorRT engine.

Python 3.8 stdlib only. Reads an engine built by trtexec and prepends the metadata
header that ultralytics' own TensorRT exporter writes.

The header format (from ultralytics/utils/export/engine.py:452-455):
    1. 4-byte little-endian signed integer: length of metadata JSON
    2. UTF-8 encoded JSON metadata string
    3. TensorRT engine binary data

Ultralytics metadata keys (from ultralytics/engine/exporter.py ~line 905-921):
    - task: 'detect' | 'segment' | 'pose' | 'obb' | 'classify'
    - model: model stem (e.g., 'yoloe-26n')
    - imgsz: [height, width]
    - names: {0: 'bottle', 1: 'chair', ...}
    - stride: model stride (typically 32)
    - etc.

Usage:
    trt_add_ultralytics_meta.py \
        --engine /path/to/raw.engine \
        --names bottle chair cup ... \
        --output /path/to/final.engine \
        [--task segment] [--imgsz 640 640] [--stride 32]

The output engine can be loaded by ultralytics YOLO(engine_path) and will have
the correct class names accessible.
"""
import argparse
import json
import struct
import sys


