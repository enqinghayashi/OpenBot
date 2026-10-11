#!/usr/bin/env python3
"""Re-run the SONIC-lane mutation matrix AFTER the 2026-09-12 pin fix round.

Every row must (a) apply its mutation to a COPY of the tree, (b) run
test_writer_sonic_lane.py from that copy, (c) report rc + PASS/FAIL counts, and
(d) report the fate of the SPECIFIC pin the mutation targets.

Honesty rules enforced by this driver:
  * an anchor that does not match EXACTLY ONCE is a hard error, never a
    silent no-op mutation;
  * the pristine tree's md5 is recorded before and after, so a run that
    touched the real tree is visible;
  * a row whose targeted pin is not found in the output is reported as
    PIN_TEXT_ABSENT, not as a pass.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

REPO = "/home/leonlin/Unitree/video2contactdynamic/OpenBot"
# tree: this file's path, any repo root
# <repo>/skills/writer/tools/mutate_sonic_lane.py
SRC = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))                      # that's skills/
WORK = os.environ.get("SONIC_MUT_WORK", "/tmp/sonic_matrix")

WRITER = "writer/main_merged_writer.py"
CFG = "writer/config/merge_config.yaml"
TEST = "writer/tools/test_writer_sonic_lane.py"


def md5_tree(root):
    h = hashlib.md5()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            h.update(os.path.relpath(p, root).encode())
            with open(p, "rb") as fh:
                h.update(hashlib.md5(fh.read()).digest())
    return h.hexdigest()


def apply_mut(path, old, new):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"ANCHOR ERROR in {os.path.basename(path)}: "
                         f"expected 1 occurrence, found {n}\n--- anchor ---\n{old}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(old, new, 1))
