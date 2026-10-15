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
