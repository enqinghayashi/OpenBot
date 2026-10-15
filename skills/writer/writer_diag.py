#!/usr/bin/env python3
"""writer_diag.py — stdlib-only /proc watchdog for the merged writer.

Run:  python3 writer_diag.py <writer_pid> [log_path]
Every 1 s appends one line to the log (default /tmp/pick_walk_logs/writer_diag.log):
  writer CPU% (utime+stime delta vs /proc/stat total), per-thread top-3 CPU,
  rchar/wchar/rss, loadavg. NO DDS, NO /proc writes — cannot perturb the
  writer's lowstate measurement (do NOT add an independent lowstate reader as
  a diagnostic; the writer's own status line answers that question).
On writer exit: final summary line, exit 0.
"""

import os
import sys
import time


def read_stat(pid):
    """Whole-process cpu ticks from /proc/<pid>/stat (aggregates all threads)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read().rsplit(")", 1)[1].split()
        return {"state": s[0], "cpu": int(s[11]) + int(s[12])}
    except (OSError, IndexError, ValueError):
        return None


def read_threads(pid):
    """[(tid, state, cpu_ticks)] per thread — identifies the starving thread."""
    rows = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return rows
    for t in tids:
        try:
            with open(f"/proc/{pid}/task/{t}/stat") as f:
                s = f.read().rsplit(")", 1)[1].split()
            rows.append((t, s[0], int(s[11]) + int(s[12])))
        except (OSError, IndexError, ValueError):
            pass
    return rows


def read_io(pid):
    out = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                out[k.strip()] = int(v)
    except OSError:
        pass
    return out


def read_rss(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return line.split()[1]  # this one's in kB
    except OSError:
        pass
    return "?"


def cpu_total():
    try:
        with open("/proc/stat") as f:
            return sum(int(x) for x in f.readline().split()[1:])
    except (OSError, ValueError):
        return None
