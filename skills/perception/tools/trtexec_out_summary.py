#!/usr/bin/env python3
"""trtexec_out_summary.py -- summarise a trtexec --exportOutput JSON for the YOLOE live engine.
Per output: shape, NaN count, |max|, std; for output0 also the conf column (4) max and rows >= 0.35.
Python 3.8, stdlib + numpy.   usage: python3 trtexec_out_summary.py out.json"""
import json
import sys

import numpy as np


