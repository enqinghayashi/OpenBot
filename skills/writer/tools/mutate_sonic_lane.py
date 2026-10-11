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


def run_row(name, mutations, pins):
    """mutations: list of (relpath, old, new).  pins: list of check texts that
    must appear with [FAIL] (or as an EXC) in the output."""
    dst = os.path.join(WORK, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    shutil.copytree(SRC, os.path.join(dst, "skills"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    for rel, old, new in mutations:
        apply_mut(os.path.join(dst, "skills", rel), old, new)
    # compile check on mutant
    sut = os.path.join(dst, "skills", WRITER)
    c = subprocess.run([sys.executable, "-m", "py_compile", sut],
                       capture_output=True, text=True)
    if c.returncode != 0:
        return {"name": name, "rc": None, "pass": 0, "fail": 0,
                "verdict": "MUTATION_DOES_NOT_COMPILE",
                "detail": c.stderr.strip().splitlines()[-1:] , "pins": []}
    p = subprocess.run([sys.executable, os.path.join(dst, "skills", TEST)],
                       capture_output=True, text=True, cwd=dst)
    out = p.stdout + p.stderr
    npass = len(re.findall(r"\[PASS\]", out))
    nfail = len(re.findall(r"\[FAIL\]", out))
    pin_fate = []
    for text in pins:
        if text not in out:
            pin_fate.append((text, "PIN_TEXT_ABSENT"))
        elif re.search(r"\[FAIL\][^\n]*" + re.escape(text), out) or \
                any(text in ln and "[FAIL]" in ln for ln in out.splitlines()):
            pin_fate.append((text, "RED"))
        else:
            pin_fate.append((text, "GREEN"))
    targeted = [f for _, f in pin_fate]
    if not targeted:
        verdict = "NO_PIN_SPECIFIED"
    elif all(f == "RED" for f in targeted):
        verdict = "PINS_HOLD" if p.returncode != 0 else "PIN_DEAD"
    else:
        verdict = "PIN_MISSED:" + ",".join(
            f"{t}={f}" for t, f in pin_fate if f != "RED")
    return {"name": name, "rc": p.returncode, "pass": npass, "fail": nfail,
            "verdict": verdict, "pins": pin_fate,
            "tail": out.strip().splitlines()[-1:][0] if out.strip() else ""}
