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
