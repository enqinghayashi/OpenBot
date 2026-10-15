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


def loadavg():
    try:
        with open("/proc/loadavg") as f:
            return " ".join(f.read().split()[:3])
    except OSError:
        return "? ? ?"


def main():
    if len(sys.argv) < 2:
        print("usage: writer_diag.py <pid> [log_path]", file=sys.stderr)
        return 1
    pid = int(sys.argv[1])
    log_path = sys.argv[2] if len(sys.argv) > 2 else \
        "/tmp/pick_walk_logs/writer_diag.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    prev = read_stat(pid)
    if prev is None:
        print(f"writer_diag: pid {pid} not found or gone", file=sys.stderr)
        return 1
    prev_total = cpu_total()
    prev_threads = dict((t, c) for t, s, c in read_threads(pid))
    prev_io = read_io(pid)
    rss = read_rss(pid)

    with open(log_path, "a") as log:
        log.write(f"--- writer_diag start pid={pid} "
                  f"{time.strftime('%F %T')} rss={rss}kB ---\n")
        log.flush()
        while True:
            time.sleep(1.0)
            cur = read_stat(pid)
            if cur is None:
                log.write(f"[diag] writer {pid} EXITED "
                          f"{time.strftime('%F %T')} (final rss={rss}kB)\n")
                log.flush()
                return 0
            now_total = cpu_total()
            dt_cpu = (now_total - prev_total) if (prev_total and now_total) else 0
            pct = 100.0 * (cur["cpu"] - prev["cpu"]) / dt_cpu if dt_cpu > 0 else 0.0
            threads = read_threads(pid)
            t3 = sorted(
                ((c - prev_threads.get(t, c), t, s)
                 for t, s, c in threads if t in prev_threads),
                reverse=True)[:3]
            io = read_io(pid)
            dw = io.get("write_bytes", 0) - prev_io.get("write_bytes", 0)
            dr = io.get("read_bytes", 0) - prev_io.get("read_bytes", 0)
            rss = read_rss(pid)
            tstr = "; ".join(
                f"{t}({s},{100.0 * d / dt_cpu:4.1f}%)" if dt_cpu > 0
                else f"{t}({s},?)" for d, t, s in t3)
            log.write(
                f"[diag] t={time.strftime('%H:%M:%S')} cpu={pct:5.1f}% "
                f"load={loadavg()} rss={rss}kB "
                f"r+={dr/1024.0:7.0f}KiB/s w+={dw/1024.0:7.0f}KiB/s "
                f"top-threads=[{tstr}]\n")
            log.flush()
            prev, prev_total, prev_threads, prev_io = cur, now_total, \
                dict((t, c) for t, s, c in threads), io


if __name__ == "__main__":
    sys.exit(main())
