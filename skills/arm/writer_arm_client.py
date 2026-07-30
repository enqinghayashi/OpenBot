"""writer_arm_client.py — socket client for the merged writer (v2b).

Single-connection UNIX-socket client (mirrors grip_client.py): retry connect
at __init__, ONE request line -> ONE reply line, WriterError on ERROR: or
empty replies.  Verbs implemented for the pick engine:

    slew_start / wait_slew / slew   raw 14-DOF cosine slews (`arm` verb;
                                    DONE = slew STARTED, so wait_slew polls
                                    `armstate` until active == 0)
    pose_start / pose               named writer poses (ready/lift/tuck/...)
    freeze                          halt the in-flight slew at its current
                                    interpolated position (contact-stop)
    q / get_current_dual_arm_q      lowstate arm q slice (motors 15..28)
    armstate                        (state_name, active, rem_T)
    hold_rc on|off                  RC-stick interlock (stationary guarantee)
    payload on|off                  writer payload gravity-compensation flag

The engine uses this client as the drop-in replacement for
G1_29_ArmController: the tau argument of the old ctrl_dual_arm is DROPPED
(the writer's validated regime writes tau=0), and there is no arm_sdk DDS
writer anywhere on this process.
"""

import socket
import time

import numpy as np

SOCK_PATH = "/tmp/merged_writer.sock"
_REPLY_TIMEOUT = 5.0        # per request; slews run async
_SLEW_POLL = 0.02           # armstate poll interval while slewing
_DEFAULT_CONNECT_RETRIES = 5
_DEFAULT_RETRY_DELAY = 0.5


