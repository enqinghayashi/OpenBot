#!/usr/bin/env python3
"""yoloe_export_fixed_vocab.py -- Export YOLOE with fixed vocabulary to ONNX.

Creates an ONNX model with a fixed set of class names baked in (no dynamic vocabulary).
The exported ONNX has opset <= 17 for TensorRT 8.5 compatibility.

Usage:
    yoloe_export_fixed_vocab.py \
        --pt /path/to/yoloe-26n-seg.pt \
        --bank /path/to/vocab_bank_yoloe-26n-seg.npz \
        --vocab bottle chair cup ... \
        --output /path/to/model.onnx \
        [--imgsz 640]

Outputs:
    - model.onnx: ONNX model with fixed vocabulary
    - model.onnx.json: Sidecar with class names in order
    - Prints md5 and line counts to stdout
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def compute_md5(filepath):
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    return md5.hexdigest()
