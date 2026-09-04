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


