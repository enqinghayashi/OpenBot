#!/usr/bin/env python3
"""capture_detections.py -- READ-ONLY discriminator for the bimodal bearing.

FUTURE_WORK_target_signal_and_odometry.md section 3.4 / priority 2: record
per-sample width/height/bearing from /object_detections_body to decide whether
the 22 deg bearing swing is MASK MERGING (w bimodal) or LABEL HOPPING /
projector jitter (w unimodal).  Per section 3.4: do NOT build fix (2) if the
answer is mask merging.

Subscribes only.  No publisher, no service call, no child processes.  Streams every
row to the CSV and flushes it: memory is O(1) while recording (the summary is
computed afterwards by re-reading the file).  Stdlib only at module level;
rclpy is imported lazily inside main().

    capture_detections.py --seconds 60 --out /tmp/x.csv [--label bottle]
    ... --from-stdin  (one JSON list per line, offline analysis)
"""

import argparse
import json
import math
import sys
import time

COLUMNS = ['t_recv', 'stamp', 'label', 'w', 'h', 'x', 'y', 'z',
           'yaw_err_deg', 'conf']
BODY_TOPIC = '/object_detections_body'
# inferred from the doc's 0.076 vs ~0.44 m example, not measured
# wider gap -> bimodal
WIDTH_GAP_M = 0.15
MAX_ROWS_DEFAULT = 20000


def _num(v):
    """float or None (bool / str / NaN / inf are not numbers here)."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def yaw_err_deg(x, y):
    """degrees(atan2(x, y)) wrapped to (-180, 180] -- the walk_node definition
    `yaw_err = _wrap_pi(math.atan2(x_b, y_b))` (walk_node.py:1394 / :1423).
    x = lateral (+right), y = forward."""
    a = math.degrees(math.atan2(x, y))
    a = (a + 180.0) % 360.0 - 180.0
    if a == -180.0:
        a = 180.0
    return a


def parse_message(json_text, recv_t):
    """-> list of row dicts, one per detection.  Raises ValueError on a
    malformed message (callers count and skip)."""
    data = json.loads(json_text)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError('not a detection list')
    rows = []
    for det in data:
        if not isinstance(det, dict):
            continue
        x, y = _num(det.get('x')), _num(det.get('y'))
        rows.append({
            't_recv': recv_t,
            'stamp': _num(det.get('stamp')),
            'label': det.get('label'),
            'w': _num(det.get('width_m')),
            'h': _num(det.get('height_m')),
            'x': x, 'y': y, 'z': _num(det.get('z')),
            'yaw_err_deg': (yaw_err_deg(x, y)
                            if x is not None and y is not None else None),
            'conf': _num(det.get('conf')),
        })
    return rows
