#!/usr/bin/env python3
"""set_yoloe_switches.py - set the batch-6 YOLOE switches on the T2 line of the stack launcher.

OPERATOR TOOL.  Run it on the robot, from a box terminal.  Python 3.8, standard library
only, no shell-outs.  It rewrites ONE line of the launcher: the line that starts the yolo
node ("T2").  Every other byte of the launcher is left as the backup has it.

USAGE (environment variables; the tool takes no arguments):

    LAUNCHER=/home/unitree/leon/launch_merge_pick_v2b.sh STAGE=<0-4> \\
    [OPENBOT_YOLOE_WEIGHTS=<path>] [OPENBOT_YOLOE_THREADS=<1-8>] \\
    python3 set_yoloe_switches.py

  STAGE 0   every switch OFF: the T2 line exactly as it is in the backup (no overlay)
  STAGE 1   OPENBOT_YOLOE=1
  STAGE 2   OPENBOT_YOLOE=1 OPENBOT_YOLO_CLASSES=1
  STAGE 3   the same line as stage 2 (stage 3 is a config.voice.yaml change, not a node change)
  STAGE 4   stage 2 plus OPENBOT_INSTANCE_IDS=1

For STAGE 1-4 the ultralytics overlay directory (default /home/unitree/leon/yoloe_site)
goes FIRST on the node's PYTHONPATH, and OPENBOT_YOLOE_WEIGHTS / OPENBOT_YOLOE_THREADS
(when given) follow the switches.  STAGE 0 never puts the overlay on the path.

BACKUP AND REBUILD (read this)
  * The backup is <LAUNCHER>.pre_batch6.  If it does not exist, the first run creates it
    from the CURRENT launcher (mode and mtime kept, like cp -p), but only after checking
    that the T2 line occurs exactly once in its stock form.  An existing backup is NEVER
    overwritten, not even when the launcher was edited by hand afterwards.
  * EVERY run rebuilds the whole launcher from the backup.  So running the same stage
    twice gives the same bytes, STAGE 0 restores the backup byte for byte, and a
    value omitted on a later run is DROPPED: repeat OPENBOT_YOLOE_WEIGHTS and
    OPENBOT_YOLOE_THREADS on every run that should keep them.
  * Because of that, any change made to the launcher after the backup was taken is
    discarded on the next run (a NOTE on stderr says so when the launcher differs from the
    backup anywhere but the T2 line).  To start from a newer launcher, move the backup away
    first (the tool then makes a new one from the launcher it finds).

REFUSALS: every refusal prints "REFUSED: <reason>; nothing written" and exits 1 (no
backup, no temp file, the launcher untouched).  The tool refuses when STAGE is not one
of 0-4, LAUNCHER is missing or a symlink, the T2 line is not found exactly once in
stock form (in the backup when there is one, else in the launcher), the overlay
directory is missing (stages 1-4), a WEIGHTS / THREADS value could change the meaning of
a shell line (only [A-Za-z0-9_./+@%,=:-] is allowed; THREADS is a single digit 1-8),
or any argument is given.  OPENBOT_YOLOE_SITE overrides the overlay directory (for tests).

On success it prints "STAGE <n> form <jp|bare> line <ln>", the T2 line now in the
launcher, and the md5 of the launcher file.  The write goes through a temp file in the
launcher's directory and os.replace; the file mode is kept and the line count never changes.
"""
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile

BACKUP_SUFFIX = ".pre_batch6"
DEFAULT_SITE = "/home/unitree/leon/yoloe_site"
FOXY = "/opt/ros/foxy/lib/python3.8/site-packages:/usr/lib/python3.8/dist-packages"
JP = "/home/unitree/leon/jp_env/bin/python3"
TAIL = ' "$LEON_DIR/yolo_node.py" > "$LOG_DIR/yolo.log" 2>&1 &'

# the two stock T2 forms - jp_env (launch_pick_stack.sh) and bare
# (launch_pick_walk.sh). must match byte for byte to count.
STOCK = (
    ("jp", ("PYTHONPATH=" + FOXY + " " + JP + TAIL).encode("ascii")),
    ("bare", ("python3" + TAIL).encode("ascii")),
)

SWITCHES = {
    0: "",
    1: "OPENBOT_YOLOE=1 ",
    2: "OPENBOT_YOLOE=1 OPENBOT_YOLO_CLASSES=1 ",
    3: "OPENBOT_YOLOE=1 OPENBOT_YOLO_CLASSES=1 ",
    4: "OPENBOT_YOLOE=1 OPENBOT_YOLO_CLASSES=1 OPENBOT_INSTANCE_IDS=1 ",
}

# chars we may splice into the launcher's shell line
SAFE_VALUE = re.compile(r"[A-Za-z0-9_./+@%,=:-]+")
THREADS_VALUE = re.compile(r"[1-8]")


class Refusal(Exception):
    """A check failed before anything was written."""


class WriteError(Exception):
    """An OS error while writing; the launcher was not replaced."""


def build_line(form, stage, weights, threads, site):
    """The new T2 line (str) for a stage 1-4."""
    sw = SWITCHES[stage]
    if weights:
        sw += "OPENBOT_YOLOE_WEIGHTS=" + weights + " "
    if threads:
        sw += "OPENBOT_YOLOE_THREADS=" + threads + " "
    if form == "jp":
        return sw + "PYTHONPATH=" + site + ":" + FOXY + " " + JP + TAIL
    return sw + "PYTHONPATH=" + site + "${PYTHONPATH:+:$PYTHONPATH} python3" + TAIL


def find_t2(lines):
    """[(index, form)] for every line that is a stock T2 line."""
    return [(i, form) for i, line in enumerate(lines) for form, old in STOCK if line == old]


def _read(path, what):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except (OSError, IOError) as exc:
        raise Refusal("cannot read %s %s (%s)" % (what, path, exc))


def _checked(env, name, rx, what):
    value = env.get(name, "")
    if value != "" and rx.fullmatch(value) is None:
        raise Refusal("%s %r %s" % (name, value, what))
    return value


def create_backup(launcher, backup, data):
    """Publish `data` as the backup, atomically and without ever replacing a backup.

    The bytes go to a temp file first (mode and times copied from the launcher, as
    cp -p does), then os.link makes them appear under the backup name in one step; a
    backup that appeared meanwhile makes the link fail and is left alone.
    """
    tmp_bk = None
    try:
        fd, tmp_bk = tempfile.mkstemp(prefix="." + os.path.basename(backup) + ".tmp.",
                                      dir=os.path.dirname(os.path.abspath(backup)))
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        shutil.copystat(launcher, tmp_bk)
        try:
            os.link(tmp_bk, backup)
        except FileExistsError:
            pass
    finally:
        if tmp_bk is not None and os.path.lexists(tmp_bk):
            os.unlink(tmp_bk)


def apply(argv, env):
    """Run one edit.  Returns (lines to print, notes for stderr); raises Refusal / WriteError."""
    if argv:
        raise Refusal("unexpected argument(s) %r (this tool reads LAUNCHER and STAGE from "
                      "the environment; see --help)" % (list(argv),))
    launcher = env.get("LAUNCHER", "")
    if not launcher:
        raise Refusal("LAUNCHER is not set")
    stage_txt = env.get("STAGE")
    if stage_txt is None:
        raise Refusal("STAGE is not set")
    if re.fullmatch(r"[0-4]", stage_txt) is None:
        raise Refusal("STAGE %r is not one of 0, 1, 2, 3, 4" % (stage_txt,))
    stage = int(stage_txt)
    weights = _checked(env, "OPENBOT_YOLOE_WEIGHTS", SAFE_VALUE,
                       "has a character outside [A-Za-z0-9_./+@%,=:-] "
                       "(it would be spliced into a shell line)")
    threads = _checked(env, "OPENBOT_YOLOE_THREADS", THREADS_VALUE,
                       "is not a single digit 1-8")
    site = env.get("OPENBOT_YOLOE_SITE", "") or DEFAULT_SITE
    if SAFE_VALUE.fullmatch(site) is None:
        raise Refusal("overlay directory %r has a character outside [A-Za-z0-9_./+@%%,=:-]"
                      % (site,))

    if os.path.islink(launcher):
        raise Refusal("LAUNCHER %s is a symlink (it would be replaced by a plain file)"
                      % launcher)
    if not os.path.isfile(launcher):
        raise Refusal("LAUNCHER %s not found" % launcher)
    backup = launcher + BACKUP_SUFFIX
    have_backup = os.path.lexists(backup)
    if have_backup and (os.path.islink(backup) or not os.path.isfile(backup)):
        raise Refusal("backup %s exists but is not a plain file" % backup)

    cur = _read(launcher, "LAUNCHER")
    if have_backup:
        src = _read(backup, "backup")
    else:
        src = cur
    lines = src.split(b"\n")
    hits = find_t2(lines)
    if len(hits) != 1:
        if have_backup:
            raise Refusal("T2 line found %d times in the backup %s (expected exactly 1 stock "
                          "T2 line)" % (len(hits), backup))
        raise Refusal("no backup %s and T2 line found %d times in the launcher %s (expected "
                      "exactly 1 stock T2 line)" % (backup, len(hits), launcher))
    idx, form = hits[0]
    notes = []
    if have_backup:
        cur_lines = cur.split(b"\n")
        if len(cur_lines) != len(lines) or any(
                a != b for j, (a, b) in enumerate(zip(cur_lines, lines)) if j != idx):
            notes.append("NOTE: the launcher differs from the backup outside the T2 line; those "
                         "differences are discarded (the launcher is rebuilt from %s)" % backup)

    if stage == 0:
        new = src
    else:
        if not os.path.isdir(site):
            raise Refusal("overlay directory %s does not exist (stages 1-4 need it)" % site)
        new_lines = list(lines)
        new_lines[idx] = build_line(form, stage, weights, threads, site).encode("ascii")
        new = b"\n".join(new_lines)
    if new.count(b"\n") != src.count(b"\n"):
        raise Refusal("the line count would change")

    mode = stat.S_IMODE(os.stat(launcher).st_mode)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix="." + os.path.basename(launcher) + ".b6tmp.",
                                   dir=os.path.dirname(os.path.abspath(launcher)))
        with os.fdopen(fd, "wb") as fh:
            fh.write(new)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        if not have_backup:
            create_backup(launcher, backup, cur)
        os.replace(tmp, launcher)
        tmp = None
    except OSError as exc:
        raise WriteError("write failed (%s)" % exc)
    finally:
        if tmp is not None and os.path.lexists(tmp):
            os.unlink(tmp)

    return [
        "STAGE %d form %s line %d" % (stage, form, idx + 1),
        new.split(b"\n")[idx].decode("utf-8", "replace"),
        "%s (stage 0 = the %s md5)" % (hashlib.md5(new).hexdigest(), BACKUP_SUFFIX),
    ], notes


def main(argv=None, environ=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ if environ is None else environ
    if argv in (["-h"], ["--help"]):
        print(__doc__.strip())
        return 0
    try:
        out, notes = apply(argv, env)
    except Refusal as exc:
        print("REFUSED: %s; nothing written" % " ".join(str(exc).split()))
        return 1
    except WriteError as exc:
        print("ERROR: %s; launcher unchanged" % " ".join(str(exc).split()))
        return 1
    for note in notes:
        sys.stderr.write(note + "\n")
    for line in out:
        print(line)
    return 0
