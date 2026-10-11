#!/usr/bin/env python3
"""obs_unit_test.py — laptop gate for the C3 obs stack (no robot, no SDK).

Validates the 516-D obs layout, history ordering, gate, and ONNX round-trip
against the verified GR00T constants. Exit 0 = green.

Usage:  python3 tools/obs_unit_test.py [--onnx-dir <dir>]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from obs_builder import ObsBuilder, FRAME_DIM, NUM_ACTIONS
from policy_runner import PolicyRunner

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
CFG = os.path.join(PKG, "config", "merge_config.yaml")

FAILS = []


