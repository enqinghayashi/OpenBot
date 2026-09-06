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


def find_envelope(lines):
    """(start, end): index of "  envelope:" under the one top-level "embodiment:" and the
    index just past the envelope block; None when either header is absent or not unique."""
    tops = [i for i, line in enumerate(lines) if re.match(rb"embodiment:(?:[ \t]|$)", line)]
    if len(tops) != 1:
        return None
    end_top = len(lines)
    for j in range(tops[0] + 1, len(lines)):
        if not _blank_or_comment(lines[j]) and not lines[j][:1].isspace():
            end_top = j
            break
    envs = [j for j in range(tops[0] + 1, end_top)
            if re.match(rb"  envelope:(?:[ \t]|$)", lines[j])]
    if len(envs) != 1:
        return None
    end = end_top
    for j in range(envs[0] + 1, end_top):
        if not _blank_or_comment(lines[j]) and len(lines[j]) - len(lines[j].lstrip(b" ")) <= 2:
            end = j
            break
    return envs[0], end


def plan_text(data, want):
    """(new bytes, path) for the config bytes `data`; path is "insert" or "update"."""
    if b"\r" in data:
        raise Refusal("the config has CR characters (CRLF line ends); only LF files are edited")
    lines = data.split(b"\n")
    present = {}
    for key in KEYS:
        rx = re.compile(rb"[ \t]*" + key.encode("ascii") + rb"[ \t]*:")
        present[key] = [i for i, line in enumerate(lines) if rx.match(line)]
        if len(present[key]) > 1:
            raise Refusal("%s is present %d times (lines %s)" % (
                key, len(present[key]), ", ".join(str(i + 1) for i in present[key])))
    have = [key for key in KEYS if present[key]]
    if len(have) not in (0, len(KEYS)):
        raise Refusal("only %s of the three keys are present (%s); all or none are expected"
                      % (len(have), ", ".join(have)))
    block = find_envelope(lines)
    if block is None:
        raise Refusal("the embodiment: -> envelope: block was not found exactly once")
    start, end = block

    if not have:
        anchors = [i for i, line in enumerate(lines) if line.startswith(ANCHOR)]
        if len(anchors) != 1:
            raise Refusal("the insertion anchor (the low-workspace descent lane comment) was "
                          "found %d times (expected exactly 1)" % len(anchors))
        ai = anchors[0]
        if not start < ai < end:
            raise Refusal("the insertion anchor (line %d) is outside the envelope block" % (ai + 1))
        block_lines = [HEADER] + [
            ("    %s: %s" % (key, "true" if want[key] else "false")).encode("ascii")
            for key in KEYS]
        new_lines = lines[:ai] + block_lines + lines[ai:]
        return b"\n".join(new_lines), "insert"

    new_lines = list(lines)
    for key in KEYS:
        i = present[key][0]
        rx = re.compile(rb"(    " + key.encode("ascii") + rb":[ ]+)(true|false)((?:[ ]+#.*)?[ ]*)")
        m = rx.fullmatch(lines[i])
        if not start < i < end or m is None:
            raise Refusal("line %d (%s) is not a plain '    %s: true|false' line inside the "
                          "envelope block" % (i + 1, key, key))
        new_lines[i] = m.group(1) + (b"true" if want[key] else b"false") + m.group(3)
    return b"\n".join(new_lines), "update"


def verify_with_loader(path, want):
    """([NOTE lines], refusal reason or None): load `path` with the real loader."""
    cfg_dir = os.path.dirname(os.path.abspath(path))
    sys.dont_write_bytecode = True
    if cfg_dir not in sys.path:
        sys.path.insert(0, cfg_dir)
    try:
        from brain.config_loader import load
    except Exception:  # noqa: BLE001 - ImportError, SyntaxError, ...: means the loader can't run here
        return ["NOTE: real loader not importable, values NOT verified by it"], None
    notes = []
    profile = PROFILE
    if not os.path.isfile(os.path.join(cfg_dir, "config." + PROFILE + ".yaml")):
        profile = "laptop"
        notes.append("NOTE: %s not found; the real loader verified the values under profile "
                     "'laptop' (base file only)"
                     % os.path.join(cfg_dir, "config." + PROFILE + ".yaml"))
    try:
        env = load(path, profile=profile).embodiment.envelope
        got = dict((key, getattr(env, key)) for key in KEYS)
    except Exception as exc:  # noqa: BLE001 - anything the loader raises means "do not write"
        return notes, "the real loader rejects the new config (%s: %s)" % (type(exc).__name__, exc)
    for key in KEYS:
        if got[key] is not want[key]:
            return notes, "the real loader reads %s=%r, not the requested %r" % (
                key, got[key], want[key])
    return notes, None


def create_backup(config, backup, data):
    """Publish `data` as the backup atomically and without ever replacing a backup."""
    tmp_bk = None
    try:
        fd, tmp_bk = tempfile.mkstemp(prefix="." + os.path.basename(backup) + ".tmp.",
                                      dir=os.path.dirname(os.path.abspath(backup)))
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copystat(config, tmp_bk)
        try:
            os.link(tmp_bk, backup)
        except FileExistsError:
            pass
    finally:
        if tmp_bk is not None and os.path.lexists(tmp_bk):
            os.unlink(tmp_bk)


def apply(argv, env):
    """Run one edit.  Returns the lines to print; raises Refusal / WriteError."""
    if argv:
        raise Refusal("unexpected argument(s) %r (this tool reads its settings from the "
                      "environment; see --help)" % (list(argv),))
    want = read_switches(env)
    config = env.get("CONFIG", "") or DEFAULT_CONFIG
    if os.path.islink(config):
        raise Refusal("CONFIG %s is a symlink (it would be replaced by a plain file)" % config)
    if not os.path.isfile(config):
        raise Refusal("CONFIG %s not found" % config)
    try:
        with open(config, "rb") as fh:
            data = fh.read()
    except (OSError, IOError) as exc:
        raise Refusal("cannot read CONFIG %s (%s)" % (config, exc))
    backup = config + BACKUP_SUFFIX
    have_backup = os.path.lexists(backup)
    if have_backup and (os.path.islink(backup) or not os.path.isfile(backup)):
        raise Refusal("backup %s exists but is not a plain file" % backup)

    new, _path = plan_text(data, want)
    mode = stat.S_IMODE(os.stat(config).st_mode)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(config) + ".b6tmp.",
                                   dir=os.path.dirname(os.path.abspath(config)))
        with os.fdopen(fd, "wb") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        notes, bad = verify_with_loader(tmp, want)
        if bad:
            raise Refusal(bad)
        changed = new != data
        if changed:
            if not have_backup:
                create_backup(config, backup, data)
            os.replace(tmp, config)
            tmp = None
    except OSError as exc:
        raise WriteError("write failed (%s)" % exc)
    finally:
        if tmp is not None and os.path.lexists(tmp):
            os.unlink(tmp)

    flags = " ".join("%s=%s" % (label, "true" if want[key] else "false") for label, key in (
        ("shadow", "pc_shadow"), ("hold_check", "pc_hold_check"), ("place_rim", "pc_place_rim")))
    return notes + ["PC_CFG %s %s md5=%s" % ("APPLIED" if changed else "ALREADY SET", flags,
                                             hashlib.md5(new).hexdigest())]


def main(argv=None, environ=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if argv in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    try:
        out = apply(argv, env)
    except Refusal as exc:
        print("REFUSED: %s; nothing written" % " ".join(str(exc).split()))
        return 1
    except WriteError as exc:
        print("ERROR: %s; config unchanged" % " ".join(str(exc).split()))
        return 1
    for line in out:
        print(line)
    return 0
