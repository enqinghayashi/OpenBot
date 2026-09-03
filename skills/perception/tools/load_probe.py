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


def parse_status_lines(text):
    """[(t, overruns, poll_to, Tmax token or None, mode token or None)] in file order.

    Only newline-terminated lines count: the last fragment of a block-buffered file may
    stop in the middle of a number.
    """
    rows = []
    for line in text.split("\n")[:-1]:
        m_t = _T_RE.match(line)
        m_ov = _OVERRUNS_RE.search(line)
        m_pt = _POLL_TO_RE.search(line)
        if not (m_t and m_ov and m_pt):
            continue
        m_tm = _TMAX_RE.search(line)
        m_md = _MODE_RE.search(line)
        rows.append((float(m_t.group(1)), int(m_ov.group(1)), int(m_pt.group(1)),
                     m_tm.group(1) if m_tm else None, m_md.group(1) if m_md else None))
    return rows


def select_window(rows, window_s):
    """The newest rows whose t is within window_s of the newest t, stopping at a restart."""
    if not rows:
        return []
    start = len(rows) - 1
    newest_t = rows[start][0]
    while start > 0:
        prev_t = rows[start - 1][0]
        if prev_t > rows[start][0]:
            break
        if newest_t - prev_t > window_s:
            break
        start -= 1
    return rows[start:]


def min_lines_for(window_s):
    if math.isinf(window_s):
        return MIN_LINES_CAP
    return min(MIN_LINES_CAP, max(2, int(math.ceil(0.75 * window_s))))


def percentile(sorted_vals, num, den):
    """sorted_vals[min(n-1, n*num/den)] with integer arithmetic (probe's _pct convention)."""
    n = len(sorted_vals)
    return sorted_vals[min(n - 1, (num * n) // den)]


def window_stats(win):
    """Statistics over the intervals of a window; an interval where a counter went down
    (a writer restart) is left out and counted."""
    rates, tot_dt, tot_ov, poll_delta, resets = [], 0.0, 0, 0, 0
    for a, b in zip(win, win[1:]):
        dt, d_ov, d_pt = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        if dt <= 0:
            continue
        if d_ov < 0 or d_pt < 0:
            resets += 1
            continue
        rates.append(d_ov / dt)
        tot_dt += dt
        tot_ov += d_ov
        poll_delta += d_pt
    return {"rates": rates, "mean": (tot_ov / tot_dt) if tot_dt > 0 else None,
            "poll_to_delta": poll_delta, "resets": resets}


def writer_report(path, window_s):
    lines = []
    err = None
    try:
        rows = parse_status_lines(read_log_text(path))
    except (OSError, IOError) as exc:
        rows, err = [], "cannot read %s (%s)" % (path, exc.strerror or exc)
    win = select_window(rows, window_s)
    need = min_lines_for(window_s)
    if len(win) < need:
        lines.append("WARN writer log has fewer than %d lines in the window (found %d, window "
                     "%s s%s)" % (need, len(win), repr(window_s), "; " + err if err else ""))
    else:
        st = window_stats(win)
        if not st["rates"]:
            lines.append("WARN writer log window has no usable interval (every interval was "
                         "a counter reset)")
        else:
            rates = sorted(st["rates"])
            lines.append("writer_overruns_per_s_mean %s" % repr(st["mean"]))
            lines.append("writer_overruns_per_s_p50 %s" % repr(percentile(rates, 50, 100)))
            lines.append("writer_overruns_per_s_p90 %s" % repr(percentile(rates, 90, 100)))
            lines.append("writer_poll_to_delta %d" % st["poll_to_delta"])
    if win:
        lines.append("writer_Tmax_last %s" % (win[-1][3] if win[-1][3] is not None else "NA"))
        lines.append("writer_mode_machine %s" % (win[-1][4] if win[-1][4] is not None else "NA"))
    lines.append("writer_lines_used %d" % len(win))
    if win:
        lines.append("writer_newest_t %s" % repr(win[-1][0]))
        lines.append("writer_counter_resets %d" % window_stats(win)["resets"])
    return lines


# ------------------------------------------------------------------- /proc bits
def read_stat(pid):
    """{"ticks", "ppid", "start"} from /proc/<pid>/stat, or None (gone / unreadable)."""
    try:
        with open("/proc/%d/stat" % pid, "rb") as fh:
            data = fh.read().decode("utf-8", "replace")
        rest = data[data.rindex(")") + 2:].split()
        return {"ticks": int(rest[11]) + int(rest[12]), "ppid": int(rest[1]),
                "start": rest[19]}
    except (OSError, IOError, ValueError, IndexError):
        return None


def read_status(pid):
    """(rss_mb, nthreads) from /proc/<pid>/status, or None."""
    rss_kb = threads = None
    try:
        with open("/proc/%d/status" % pid, "rb") as fh:
            for raw in fh.read().decode("utf-8", "replace").split("\n"):
                if raw.startswith("VmRSS:"):
                    rss_kb = int(raw.split()[1])
                elif raw.startswith("Threads:"):
                    threads = int(raw.split()[1])
    except (OSError, IOError, ValueError, IndexError):
        return None
    if rss_kb is None or threads is None:
        return None
    return rss_kb / 1024.0, threads


def own_pids():
    """This process and its ancestors (a wrapper such as timeout carries our arguments)."""
    seen = set([os.getpid()])
    cur = os.getpid()
    for _ in range(128):
        st = read_stat(cur)
        if st is None or st["ppid"] <= 0 or st["ppid"] in seen:
            break
        cur = st["ppid"]
        seen.add(cur)
    return seen


