#!/usr/bin/env python3
"""trtexec_out_summary.py -- summarise a trtexec --exportOutput JSON for the YOLOE live engine.
Per output: shape, NaN count, |max|, std; for output0 also the conf column (4) max and rows >= 0.35.
Python 3.8, stdlib + numpy.   usage: python3 trtexec_out_summary.py out.json"""
import json
import sys

import numpy as np


def main(path):
    for o in json.load(open(path)):
        v = np.asarray(o['values'], np.float32).reshape([int(d) for d in o['dimensions'].split('x')]
                                                         if isinstance(o['dimensions'], str) else o['dimensions'])
        line = '%s shape=%s nan=%d |max|=%.4g std=%.4g' % (o['name'], v.shape, int(np.isnan(v).sum()),
                                                         float(np.nanmax(np.abs(v))), float(np.nanstd(v)))
        if o['name'] == 'output0':
            c = v.reshape(-1, v.shape[-1])[:, 4]
            line += ' conf_max=%.4f rows>=0.35=%d' % (float(np.nanmax(c)), int((c >= 0.35).sum()))
        print(line)


if __name__ == '__main__':
    main(sys.argv[1])
