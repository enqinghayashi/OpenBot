#!/usr/bin/env python3
"""load_probe.py - READ-ONLY load meter for the batch-6 A/B gates.

OPERATOR TOOL.  Run it on the robot while the stack is up.  Python 3.8, standard library
only (it reads /proc and one log file; no shell-outs, no psutil).  It writes NOTHING and
prints plain "KEY value" lines.  It prints no verdict: read the numbers against the gate.

USAGE:   python3 load_probe.py [--writer-log /tmp/pick_walk_logs/writer.log]
                                  [--window-s 60] [--sample-s 5]
                                  [--procs depth_projector_node.py,yolo_node.py,pick_engine_v3.py,
                                           gateway.gateway_node,main_merged_writer.py,pc_consumer_node.py]

WRITER (from the status lines "[t= 801.0] balance cmd=... mode=6 Tmax=69C poll_to=56
overruns=87813 ..." in the writer log; the file is block-buffered, so its newest line can
lag by up to ~40 s):
  writer_overruns_per_s_mean   (last overruns - first overruns) / (last t - first t) over the window
  writer_overruns_per_s_p50    per-interval rates delta(overruns)/delta(t), the median
  writer_overruns_per_s_p90    the same, 90th percentile
        percentile q = sorted rates[min(n-1, n*q)] (the index convention of yoloe_box_probe.py)
  writer_poll_to_delta         growth of poll_to over the window
  writer_Tmax_last             Tmax of the newest line, as the writer printed it
  writer_mode_machine          "mode=" of the newest line, as printed (the lowstate mode_machine)
  writer_lines_used            status lines inside the window
  writer_newest_t              t of the newest line
  writer_counter_resets        intervals left out because a counter went DOWN (writer restart)
The window is the status lines with t >= newest t - window-s, never reaching back across a
restart (t going backwards).  A line without its trailing newline (a half-flushed buffer) is
ignored.  With fewer than min(45, ceil(0.75 * window-s)) lines in the window (at least 2) the
tool prints "WARN writer log has fewer than N lines in the window" and NO statistics.

SYSTEM:
  loadavg_1m                   first field of /proc/loadavg, as printed
  MemAvailable_mb              MemAvailable from /proc/meminfo
PROCESSES: every process whose command line contains a pattern (pgrep -f style; this tool
and its parent processes never match), CPU sampled twice --sample-s apart:
  <pattern> pid <pid> cpu_pct_of_one_core <x> rss_mb <y> nthreads <n>
  <pattern> ABSENT             no matching process (a process that vanished mid-sample is skipped)
cpu_pct_of_one_core = 100 * (utime+stime ticks / SC_CLK_TCK) / elapsed seconds; 100 = one full
core, a multi-threaded process can exceed 100.
"""
import argparse
import math
import os
import re
import sys
import time

DEFAULT_WRITER_LOG = "/tmp/pick_walk_logs/writer.log"
DEFAULT_PROCS = ("depth_projector_node.py,yolo_node.py,pick_engine_v3.py,gateway.gateway_node,"
                 "main_merged_writer.py,pc_consumer_node.py")
TAIL_BYTES = 16 * 1024 * 1024
MIN_LINES_CAP = 45

_T_RE = re.compile(r"^\[t=\s*([0-9]+(?:\.[0-9]*)?)\]")
_OVERRUNS_RE = re.compile(r"\boverruns=([0-9]+)")
_POLL_TO_RE = re.compile(r"\bpoll_to=([0-9]+)")
_TMAX_RE = re.compile(r"\bTmax=(\S+)")
_MODE_RE = re.compile(r"\bmode=(\S+)")


# ------------------------------------------------------------- the writer log
def read_log_text(path):
    """The log text; only the last TAIL_BYTES of a huge file, cut at a line start."""
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        start = max(0, fh.tell() - TAIL_BYTES)
        fh.seek(start)
        data = fh.read()
    text = data.decode("utf-8", "replace")
    if start > 0:
        nl = text.find("\n")
        text = text[nl + 1:] if nl >= 0 else ""
    return text
