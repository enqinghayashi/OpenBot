#!/usr/bin/env python3
"""set_pc_switches.py - set the three batch-3b point-cloud switches in the robot's config.yaml.

OPERATOR TOOL.  Run it on the robot, from a box terminal.  Python 3.8, standard library
only, no shell-outs.  The robot's config.yaml is never pushed from the laptop (it carries
operator hand-edits), so this tool changes only the three switch lines, anchor-checked and
atomically, instead of a hand edit.

USAGE (environment variables; the tool takes no arguments):

    [CONFIG=/home/unitree/leon/config.yaml] \\
    PC_SHADOW=<true|false> PC_HOLD_CHECK=<true|false> PC_PLACE_RIM=<true|false> \\
    python3 set_pc_switches.py

All three variables are required and must be exactly the lowercase word true or false
(anything else, an empty value included, is refused), so no switch changes by accident.
The keys are pc_shadow, pc_hold_check and pc_place_rim, at 4-space indent inside
embodiment: -> envelope:.

WHAT IT DOES
  1. None of the three keys exists: ONE block is inserted immediately before the unique
     line that starts "    # -- low-workspace descent lane (plan par.8)" (drawn with the
     box characters the repo file uses): a comment line "batch 3b point-cloud consumer
     switches (all default false; docs/DEPLOY_yoloe_pointcloud.md)" and the three key lines.
  2. All three keys exist (each exactly once, 4-space indent, a plain true or false with an
     optional trailing comment): only those three lines are updated, trailing comments kept.
  3. Any other state is REFUSED: one or two keys present, a key twice, a value that is not
     a plain true/false, a key or the anchor outside the envelope block, an anchor that is
     missing or twice.
  4. Before anything is written the new text is loaded from a temp file in the config's
     directory with the REAL loader (brain.config_loader.load, boot profile "offline" as
     the gateway uses it; the directory of the config goes on sys.path) and the three
     values must read back exactly as requested.  Any loader error is a refusal (a loader
     that does not know the keys yet refuses here).  If the loader cannot be imported the
     tool prints "NOTE: real loader not importable, values NOT verified by it" and goes on.
  5. The first run that changes the file makes the backup <CONFIG>.pre_b6_pc (mode and
     mtime kept, like cp -p).  An existing backup is NEVER overwritten.  A run that
     changes nothing ("ALREADY SET") writes no backup.
  6. The write goes through a temp file in the same directory and os.replace; the file mode
     is kept and every other byte (every other line, the final newline state) is unchanged.

Success prints "PC_CFG APPLIED shadow=<v> hold_check=<v> place_rim=<v> md5=<md5 of the new
file>" (or "PC_CFG ALREADY SET ..." when nothing changed).  Every refusal prints
"REFUSED: <reason>; nothing written" and exits 1: no backup, no temp file, the file untouched.
"""
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile

DEFAULT_CONFIG = "/home/unitree/leon/config.yaml"
BACKUP_SUFFIX = ".pre_b6_pc"
PROFILE = "offline"
KEYS = ("pc_shadow", "pc_hold_check", "pc_place_rim")
ENV_NAMES = (("pc_shadow", "PC_SHADOW"), ("pc_hold_check", "PC_HOLD_CHECK"),
             ("pc_place_rim", "PC_PLACE_RIM"))
# in the repo file: U+2500 box light, U+00A7 section.
ANCHOR = "    # \u2500\u2500 low-workspace descent lane (plan \u00a78)".encode("utf-8")
HEADER = ("    # \u2500\u2500 batch 3b point-cloud consumer switches (all default false; "
          "docs/DEPLOY_yoloe_pointcloud.md) \u2500\u2500").encode("utf-8")


class Refusal(Exception):
    """A check failed before anything was written."""


class WriteError(Exception):
    """An OS error while writing; the config was not replaced."""


def read_switches(env):
    """{key: bool} from PC_SHADOW / PC_HOLD_CHECK / PC_PLACE_RIM, each exactly true or false."""
    want = {}
    for key, name in ENV_NAMES:
        value = env.get(name)
        if value is None:
            raise Refusal("%s is not set (it must be exactly true or false)" % name)
        if value not in ("true", "false"):
            raise Refusal("%s=%r is not exactly true or false" % (name, value))
        want[key] = value == "true"
    return want


def _blank_or_comment(line):
    stripped = line.strip()
    return stripped == b"" or stripped.startswith(b"#")
