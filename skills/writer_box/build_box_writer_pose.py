#!/usr/bin/env python3
"""build_box_writer_pose.py -- derive the box writer WITH the socket verb `pose` (hook H-WR-1).

The robot runs a writer that has no repo source revision (the BOX writer, md5
82a94be938b5ff6cfdf7fc443437a6fe, 2171 newline-terminated lines, saved read-only at
/home/leonlin/Unitree/.wf/box_writer_82a94be9.py).  This builder takes the box's OWN bytes and
applies three EXACT-ANCHOR insertions.  Every anchor must occur exactly once; otherwise it REFUSES
and writes nothing.  No existing line of the writer changes (each edit only inserts text):

  (a) a guarded `import pose_fk as PF` (try/except -> PF = None, the SONIC lane's idiom), placed
      immediately before the writer's `from unitree_sdk2py...` import block, i.e. right after the
      SONIC guarded import;
  (b) the verb branch `if verb == "pose":` immediately before `if verb == "status":`;
  (c) the small method `_pose_line`, immediately before `_apply_cmd_set`.

Outputs, in --out-dir (default /home/leonlin/Unitree/.wf/b6_box_derived/merged_writer):
  main_merged_writer.py   the derived writer
  pose_fk.py              a byte copy of pose_fk.py (the module the import in (a) finds)

    python3 build_box_writer_pose.py [--box-writer PATH] [--pose-fk PATH] [--out-dir DIR] [--check]

--check rebuilds in memory and exits 1 if either written file differs (or is missing); it writes
nothing.  Exit codes: 0 ok, 1 --check mismatch, 2 REFUSED.  stdlib only; parses under 3.8.
"""
import argparse
import ast
import hashlib
import os
import re
import sys

#: md5 of the writer binary this builder reads
BOX_MD5 = "82a94be938b5ff6cfdf7fc443437a6fe"
DEFAULT_BOX = "/home/leonlin/Unitree/.wf/box_writer_82a94be9.py"
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FK = os.path.join(HERE, "pose_fk.py")
DEFAULT_OUT = "/home/leonlin/Unitree/.wf/b6_box_derived/merged_writer"
OUT_WRITER = "main_merged_writer.py"
OUT_FK = "pose_fk.py"
#: old retired names - never written out; fragments on purpose,
#: so this file never spells them itself.
FORBIDDEN = re.compile("|".join(("came" "ron", "cam_" "ws", "setup_" "came" "ron")))

IMPORT_BLOCK = (
    "# ------------------------------------------------------------------ pose verb\n"
    "# pose_fk.py (installed beside this file) is the pure-numpy FK behind the socket\n"
    "# verb `pose` (point-cloud hook H-WR-1).  Guarded exactly like the SONIC import\n"
    "# above: on ANY failure PF stays None, nothing else in the writer changes, and\n"
    "# `pose` answers ERROR:pose unavailable (pose_fk import failed).  The 250 Hz loop\n"
    "# never touches it.\n"
    "try:\n"
    "    import pose_fk as PF\n"
    "except Exception as _pe:           # noqa: BLE001 -- degrade, never raise\n"
    "    PF = None\n"
    "    print(\"[pose] pose_fk import FAILED (%r) -- `pose` verb unavailable\" % (_pe,),\n"
    "          flush=True)\n"
    "\n"
)
VERB_BLOCK = (
    "        if verb == \"pose\":\n"
    "            # H-WR-1: ONE reply line, a MEASURED torso pose (see _pose_line).\n"
    "            return self._pose_line()\n"
)
METHOD_BLOCK = (
    "    def _pose_line(self):\n"
    "        \"\"\"Socket verb `pose` (point-cloud hook H-WR-1): ONE reply line, a JSON object\n"
    "        with a MEASURED torso pose computed by pose_fk from the lowstate this writer\n"
    "        already holds.  READ-ONLY: one pointer read of self.lowstate (the loop rebinds\n"
    "        it, never mutates it), no lock, no DDS, no log line, and the 250 Hz loop is\n"
    "        untouched.  Never raises: every failure is an `ERROR:` reply, which is not JSON\n"
    "        on purpose (writer_arm_client turns it into WriterError).\"\"\"\n"
    "        try:\n"
    "            t = time.time()\n"
    "            if PF is None:\n"
    "                return \"ERROR:pose unavailable (pose_fk import failed)\"\n"
    "            st = self.lowstate\n"
    "            if not isinstance(st, dict):\n"
    "                return \"ERROR:pose unavailable (no lowstate)\"\n"
    "            out = {\"src\": \"measured\", \"t\": round(t, 5)}\n"
    "            out.update(PF.torso_pose(st.get(\"q\"), st.get(\"quat\"),\n"
    "                                     st.get(\"mode_machine\"), st.get(\"motor_mode\")))\n"
    "            return json.dumps(out, separators=(\",\", \":\"), allow_nan=False)\n"
    "        except Exception as e:  # noqa: BLE001 -- the verb handler must never raise\n"
    "            return \"ERROR:pose failed (%s)\" % type(e).__name__\n"
    "\n"
)

#: (label, anchor, insertion) - insertion goes just before the anchor,
#: which is a whole line and must hit exactly once.
EDITS = (
    ("guarded pose_fk import (before the unitree_sdk2py import block)",
     b"\nfrom unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher\n",
     IMPORT_BLOCK),
    ("verb branch `pose` (immediately before the `status` branch)",
     b"\n        if verb == \"status\":\n",
     VERB_BLOCK),
    ("method _pose_line (before _apply_cmd_set)",
     b"\n    def _apply_cmd_set(self, parts):\n",
     METHOD_BLOCK),
)


class Refused(Exception):
    pass


def md5(b):
    return hashlib.md5(b).hexdigest()


def _lowest_level(src_text):
    """Lowest of (3,6),(3,7),(3,8) that `src_text` parses at, or None."""
    for fv in ((3, 6), (3, 7), (3, 8)):
        try:
            ast.parse(src_text, feature_version=fv)
            return fv
        except SyntaxError:
            continue
    return None


def derive(box_bytes, fk_bytes, expect_md5=None):
    """-> (writer_bytes, fk_bytes).  Raises Refused; writes nothing."""
    expect = BOX_MD5 if expect_md5 is None else expect_md5
    got = md5(box_bytes)
    if got != expect:
        raise Refused("the input writer md5 is %s, not the box writer %s" % (got, expect))
    out = box_bytes
    for label, anchor, insertion in EDITS:
        n = out.count(anchor)
        if n != 1:
            raise Refused("anchor for %s occurs %d times (want exactly 1): %r"
                          % (label, n, anchor))
        try:
            ins = insertion.encode("ascii")
        except UnicodeEncodeError:
            raise Refused("the insertion for %s is not ASCII" % label)
        out = out.replace(anchor, b"\n" + ins + anchor[1:])
    want_len = len(box_bytes) + sum(len(i.encode("ascii")) for _, _, i in EDITS)
    if len(out) != want_len:
        raise Refused("derived length %d != box length + insertions %d" % (len(out), want_len))
    try:
        box_text = box_bytes.decode("utf-8")
        out_text = out.decode("utf-8")
        fk_text = fk_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Refused("not valid UTF-8: %s" % e)
    level = _lowest_level(box_text)
    if level is None or _lowest_level(out_text) != level:
        raise Refused("the derived writer does not parse at the box writer's feature level %r"
                      % (level,))
    try:
        fk_tree = ast.parse(fk_text, feature_version=(3, 8))
    except SyntaxError as e:
        raise Refused("pose_fk.py does not parse under feature_version (3, 8): %s" % e)
    mods = set()
    names = set()
    for node in ast.walk(fk_tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    if not mods <= {"math", "numbers", "numpy"}:
        raise Refused("pose_fk.py imports %s; only math, numbers, numpy are allowed"
                      % sorted(mods))
    if "torso_pose" not in names:
        raise Refused("pose_fk.py defines no torso_pose")
    for what, text in (("the derived writer", out_text), ("pose_fk.py", fk_text)):
        m = FORBIDDEN.search(text)
        if m:
            raise Refused("%s contains the forbidden token %r" % (what, m.group(0)))
    return out, fk_bytes


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write_atomic(path, data):
    tmp = path + ".tmp_build"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--box-writer", default=DEFAULT_BOX)
    ap.add_argument("--pose-fk", default=DEFAULT_FK)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory; exit 1 if the written files differ; write nothing")
    args = ap.parse_args(argv)
    try:
        writer, fk = derive(_read(args.box_writer), _read(args.pose_fk), expect_md5=BOX_MD5)
    except Refused as e:
        sys.stderr.write("REFUSED: %s\n" % e)
        return 2
    except OSError as e:
        sys.stderr.write("REFUSED: cannot read an input: %s\n" % e)
        return 2
    out_w = os.path.join(args.out_dir, OUT_WRITER)
    out_f = os.path.join(args.out_dir, OUT_FK)
    if args.check:
        bad = []
        for path, want in ((out_w, writer), (out_f, fk)):
            try:
                have = _read(path)
            except OSError:
                bad.append("%s missing" % path)
                continue
            if have != want:
                bad.append("%s differs (on disk %s, rebuilt %s)" % (path, md5(have), md5(want)))
        if bad:
            for b in bad:
                print("CHECK FAILED: %s" % b)
            return 1
        print("CHECK OK %s %s" % (md5(writer), out_w))
        print("CHECK OK %s %s" % (md5(fk), out_f))
        return 0
    os.makedirs(args.out_dir, exist_ok=True)
    _write_atomic(out_w, writer)
    _write_atomic(out_f, fk)
    print("WROTE %s %s (%d lines)" % (md5(writer), out_w, writer.count(b"\n")))
    print("WROTE %s %s (%d lines)" % (md5(fk), out_f, fk.count(b"\n")))
    return 0
