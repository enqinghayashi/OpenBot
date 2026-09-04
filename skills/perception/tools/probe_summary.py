#!/usr/bin/env python3
"""probe_summary.py - print the gate fields of the S1 probe's JSON files.

OPERATOR TOOL (batch 6 port of b4_probe_summary.py).  Python 3.8, standard library only.
It reads; it never writes.

USAGE:   python3 probe_summary.py <probe root dir>

It looks for <probe root dir>/*/probe.json (what yoloe_box_probe.py / yoloe_box_probe.sh
write, one directory per interpreter tag) and, per file, prints

    <path of the file>
      match_staging_base = <json>
      env.<key> = <json>              scalars of the probe's env block, one level deeper
      env.<key>.<key2> = <json>       for dict values (sys_path left out)
      <bench|record|agree>.<path> = <json>     the KEYS below, wherever they sit in a dict
      errors = <count> <json of the first 5>

A missing key prints as null (match_staging_base) or is left out; it never stops the run.
A file that cannot be summarised (unreadable, empty, invalid JSON, or not a JSON object)
prints "UNREADABLE <path>" (the reason goes to stderr) and the run goes on with the next file.
When no probe.json is found: "NO_PROBE_JSON <probe root dir>".
"""
import glob
import json
import os
import sys

KEYS = ("median_ms", "p90_ms", "cpu_pct", "threads_effective", "load_ms", "rss_mb_before",
        "rss_mb_after", "maxrss_mb", "add_extras", "restore_base", "engine", "skipped", "error",
        "saved", "timed_out", "frames_dir", "n_frames", "label_set_agree_fraction_raw",
        "label_set_agree_fraction_pick", "dconf_median_pick", "base_extra_frames_fraction_pick",
        "prompts_fired", "extra_rows_total", "frames_file")


def walk(x, path, out):
    """Append "  <path> = <json>" for every KEYS entry found in the dict tree under x."""
    if isinstance(x, dict):
        for k in sorted(x):
            p = path + "." + str(k)
            if k in KEYS and not isinstance(x[k], dict):
                out.append("  %s = %s" % (p, json.dumps(x[k])))
            else:
                walk(x[k], p, out)


def summarize(r):
    """The summary lines (without the path line) of one probe result r (a dict)."""
    out = ["  match_staging_base = " + json.dumps(r.get("match_staging_base"))]
    e = r.get("env")
    if not isinstance(e, dict):
        e = {}
    for k in sorted(e):
        if k != "sys_path" and not isinstance(e[k], (dict, list)):
            out.append("  env.%s = %s" % (k, json.dumps(e[k])))
        elif isinstance(e[k], dict):
            for k2 in sorted(e[k]):
                if not isinstance(e[k][k2], (dict, list)):
                    out.append("  env.%s.%s = %s" % (k, k2, json.dumps(e[k][k2])))
    for sec in ("bench", "record", "agree"):
        walk(r.get(sec), sec, out)
    errs = r.get("errors")
    if not errs:
        errs = []
    elif not isinstance(errs, (list, tuple)):
        errs = [errs]
    out.append("  errors = %d %s" % (len(errs), json.dumps(list(errs)[:5])))
    return out


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.strip():
        raise ValueError("empty file")
    r = json.loads(raw.decode("utf-8"))
    if not isinstance(r, dict):
        raise ValueError("top level is %s, not an object" % type(r).__name__)
    return r


