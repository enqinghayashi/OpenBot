#!/usr/bin/env python3
"""fsm_check.py — read the G1 loco FSM state (pre-flight, Task 7).

Prints fsm_id and fsm_mode via the SDK G1Loco interface (same mechanism as
the verified /tmp/fsm_check.py on Stuart — G1Loco.get_fsm_mode() reads the
loco debug endpoint LOCO_API_GET_FSM_MODE = 7002, mirroring g1_loco_api.py).

Used by launch_pick_walk.sh as a pre-flight check: fsm_mode == 0 means the
balance FSM is in a walking-capable state; fsm_mode != 0 (e.g. 3 = gait
blocked, seen 2026-08-16) means walking commands will be accepted but no
gait will run — the operator should run the MODE_RECOVERY.md ladder BEFORE
starting the pick+walk stack.

Run (Stuart):
    /home/unitree/miniconda3/envs/tv/bin/python3 fsm_check.py   (or any python
    with unitree_sdk2_python + g1_loco.py importable)

Exit code: 0 if fsm_mode == 0; 1 if fsm_mode != 0 (gait blocked);
2 if the read failed (loco not reachable — cannot verify).
"""

import sys

sys.path.insert(0, "/home/unitree/leon/openbot")
sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
from g1_loco import G1Loco

# connect to the loco service, read fsm_mode, exit with a verdict
loco = G1Loco()
loco.connect()
code_id, fsm_id = loco.get_fsm_id()
code_mode, fsm_mode = loco.get_fsm_mode()
print(f"fsm_id: code={code_id} data={fsm_id}")
print(f"fsm_mode: code={code_mode} data={fsm_mode}")

if code_mode != 0:
    print("fsm_check: ERROR reading fsm_mode (code={})".format(code_mode))
    sys.exit(2)
if fsm_mode != 0:
    print("fsm_check: GAIT BLOCKED (fsm_mode={}) — recovery ladder in "
          "MODE_RECOVERY.md".format(fsm_mode))
    sys.exit(1)
print("fsm_check: OK (fsm_mode=0, walking capable)")
sys.exit(0)
