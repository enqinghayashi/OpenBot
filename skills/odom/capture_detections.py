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


def _cell(v):
    if v is None:
        return ''
    if isinstance(v, float):
        return repr(round(v, 6))
    s = str(v)
    return '"%s"' % s.replace('"', '""') if (',' in s or '"' in s) else s


def classify_width(ws):
    """Gap-based 1-D clustering verdict on the width samples."""
    ws = sorted(w for w in ws if w is not None)
    if not ws:
        return 'DISCRIMINATOR: no width samples -> INCONCLUSIVE'
    clusters = 1 + sum(1 for a, b in zip(ws, ws[1:]) if b - a > WIDTH_GAP_M)
    if clusters >= 2:
        return ('DISCRIMINATOR: w bimodal (>=2 clusters separated by >%.2f m) '
                '-> MASK MERGING' % WIDTH_GAP_M)
    return ('DISCRIMINATOR: w unimodal -> LABEL HOPPING / projector jitter')


class CsvSink(object):
    """Streams rows to the file; keeps only O(labels) counters."""

    def __init__(self, path, max_rows=MAX_ROWS_DEFAULT, label=None):
        self._fh = open(path, 'w')
        self._fh.write(','.join(COLUMNS) + '\n')
        self._fh.flush()
        self.max_rows = int(max_rows)
        self.label = label
        self.rows = 0
        self.malformed = 0
        self.per_label = {}

    @property
    def full(self):
        return self.rows >= self.max_rows

    def feed(self, json_text, recv_t):
        try:
            rows = parse_message(json_text, recv_t)
        except ValueError:
            self.malformed += 1
            return 0
        n = 0
        for r in rows:
            if self.full:
                break
            if self.label is not None and r['label'] != self.label:
                continue
            self._fh.write(','.join(_cell(r[c]) for c in COLUMNS) + '\n')
            self._fh.flush()
            self.rows += 1
            n += 1
            self.per_label[r['label']] = self.per_label.get(r['label'], 0) + 1
        return n

    def close(self):
        self._fh.close()


def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def summarize_csv(path):
    """Post-run summary (re-reads the file; the recording phase held no rows)."""
    per = {}
    ws_all = []
    total = 0
    with open(path) as fh:
        header = fh.readline().strip().split(',')
        iw, iy, il = header.index('w'), header.index('yaw_err_deg'), header.index('label')
        for line in fh:
            cells = line.rstrip('\n').split(',')
            if len(cells) < len(header):
                continue
            total += 1
            d = per.setdefault(cells[il], {'n': 0, 'w': [], 'yaw': []})
            d['n'] += 1
            if cells[iw] != '':
                d['w'].append(float(cells[iw]))
                ws_all.append(float(cells[iw]))
            if cells[iy] != '':
                d['yaw'].append(float(cells[iy]))
    lines = ['rows: %d' % total]
    for lab in sorted(per):
        d = per[lab]
        lines.append('label=%s n=%d' % (lab, d['n']))
        for name, vals in (('w', d['w']), ('yaw_err_deg', d['yaw'])):
            if vals:
                lines.append('  %s min/median/max = %.4f / %.4f / %.4f' % (
                    name, min(vals), _median(vals), max(vals)))
            else:
                lines.append('  %s: no samples' % name)
    lines.append(classify_width(ws_all))
    return '\n'.join(lines)


def run_stdin(sink, lines, clock, seconds):
    deadline = clock() + seconds
    for line in lines:
        if clock() >= deadline or sink.full:
            break
        line = line.strip()
        if line:
            sink.feed(line, clock())


def run_ros(sink, seconds, clock=time.time):
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node('capture_detections')
    node.create_subscription(
        String, BODY_TOPIC, lambda m: sink.feed(m.data, clock()), 10)
    deadline = clock() + seconds
    try:
        while rclpy.ok() and clock() < deadline and not sink.full:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


def main(argv=None):
    p = argparse.ArgumentParser(description='read-only detection capture')
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--max-rows', type=int, default=MAX_ROWS_DEFAULT)
    p.add_argument('--label', default=None)
    p.add_argument('--out', default='/tmp/odom_capture_%d.csv' % int(time.time()))
    p.add_argument('--from-stdin', action='store_true')
    a = p.parse_args(argv)
    if not 1 <= a.seconds <= 600:
        p.error('--seconds must be within 1..600')
    sink = CsvSink(a.out, max_rows=a.max_rows, label=a.label)
    try:
        if a.from_stdin:
            run_stdin(sink, sys.stdin, time.time, a.seconds)
        else:
            run_ros(sink, a.seconds)
    finally:
        sink.close()
    print('wrote %s (malformed messages skipped: %d)' % (a.out, sink.malformed))
    print(summarize_csv(a.out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
