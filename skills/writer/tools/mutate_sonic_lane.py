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


# ----------------------------------------------------------------- the matrix
W = lambda: os.path.join(SRC, WRITER)          # noqa: E731  (anchor helper)
T = lambda: os.path.join(SRC, TEST)            # noqa: E731

ROWS = [
    # name, mutations, pins that must go red
    ("M1_probe_removed", [(WRITER,
        "            self._sonic_gate_probe()\n",
        "            pass  # M1: probe call removed\n")],
     ["the probe call precedes the"]),

    ("M2_probe_not_at_gate_head", [(WRITER,
        "            self._sonic_gate_probe()\n            if self.sonic_warm < self.sonic_warmup_frames:",
        "            if self.sonic_warm < self.sonic_warmup_frames:"),
        (WRITER,
         "            dev = self.sonic_gate_dev\n",
         "            self._sonic_gate_probe()  # M2: hoist reverted\n            dev = self.sonic_gate_dev\n")],
     ["the probe call precedes the"]),

    ("M3_warm_requires_armed", [(WRITER,
        "        if not self.sonic_lane_enabled:\n            return\n        prev = self.sonic_last_rx",
        "        if not self.sonic_lane_enabled:\n            return\n        if not self.sonic_armed:  # M3\n            return\n        prev = self.sonic_last_rx")],
     ["accepted frames on an UNARMED lane"]),

    ("M4_on_has_no_freshness_check", [(WRITER,
        "            age = time.monotonic() - self.sonic_last_rx\n            if age > self.sonic_stale_ms / 1000.0:",
        "            age = time.monotonic() - self.sonic_last_rx\n            if False:  # M4: freshness conjunct deleted")],
     ["a dead stream is refused even though warm == 100"]),

    ("M5_run_break_removed", [(WRITER,
        "        if prev > 0.0 and (now - prev) > self.sonic_stale_ms / 1000.0:",
        "        if False:  # M5: run break removed")],
     ["a gap just OVER sonic_stale_ms restarts the run at 1"]),

    ("M6_probe_allows_nan", [(WRITER,
        "                json.dumps(rec, allow_nan=False) + \"\\n\")",
        "                json.dumps(rec, allow_nan=True) + \"\\n\")  # M6")],
     ["the NaN record was NOT written"]),

    ("M7_reset_lock_removed", [(WRITER,
        "            with self._dispatch_lock:\n                self.sonic_warm = 0",
        "            if True:  # M7: reset lock removed\n                self.sonic_warm = 0")],
     ["the reset zeroing is DIRECTLY under"]),

    ("M8_probe_writes_lane_state", [(WRITER,
        "        if not self.sonic_gate_probe_dir:\n            return\n        try:",
        "        if not self.sonic_gate_probe_dir:\n            return\n        self.sonic_b = 0.5  # M8: probe mutates lane state\n        try:")],
     ["lane state `b` is identical"]),

    ("M9_probe_cap_off_by_one", [(WRITER,
        "            if self.sonic_gate_probe_n >= self.sonic_gate_probe_max:",
        "            if self.sonic_gate_probe_n > self.sonic_gate_probe_max:  # M9")],
     ["the counter stops at the cap"]),

    ("M10_accept_lock_removed", [(WRITER,
        "                with self._dispatch_lock:\n                    # WARMUP counts a CONSECUTIVE RUN",
        "                if True:  # M10: governing accept-path lock removed\n                    # WARMUP counts a CONSECUTIVE RUN")],
     ["the accept path ticks the counter DIRECTLY under the same lock"]),

    ("M11_scan_list_entry_dropped", [(TEST,
        '"_sonic_warm_tick", "_sonic_gate_probe")',
        '"_sonic_gate_probe")')],
     ["_SONIC_METHODS is EXACTLY the set of sonic-named"]),

    ("M12_stale_bound_strictness_flipped", [(WRITER,
        "        if prev > 0.0 and (now - prev) > self.sonic_stale_ms / 1000.0:",
        "        if prev > 0.0 and (now - prev) >= self.sonic_stale_ms / 1000.0:  # M12")],
     ["a gap of EXACTLY sonic_stale_ms does NOT break the run"]),

    # removes the freshness message literal, so the anchor for
    # "ERROR:sonic stream stale" cannot match and the check
    # fails ANCHOR MISSING instead of passing silently
    ("M13_gate_literal_deleted_anchor", [(WRITER,
        '                return (f"ERROR:sonic stream stale ({age:.3f}s > "\n'
        '                        f"{self.sonic_stale_ms / 1000.0:.3f}s)")',
        '                return "ERROR:sonic stream STALE literal-removed"')],
     ["ANCHOR MISSING"]),

    # text-only control for A1's guard pin: `or False` keeps runtime
    # behaviour identical, so only the exact-text pin can go red.
    ("M14_guard_literal_deleted_anchor", [(WRITER,
        "        if not self.sonic_gate_probe_dir:\n            return",
        "        if not self.sonic_gate_probe_dir or False:  # M14\n            return")],
     ["the probe's FIRST guard is"]),
]


