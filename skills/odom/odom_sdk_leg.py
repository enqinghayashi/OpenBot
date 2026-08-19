import sys
sys.path.insert(0, '/home/unitree/unitree_sdk2_python')   # xrkong wrong-copy bug again
"""odom_sdk_leg.py -- SDK-only lowstate source process for odom_node.py.

NEVER imports rclpy / nav_msgs / std_msgs (rclpy + raw DDS in one process is a
proven crash class on Stuart: walk_node.py:63-64).  Poll-only rt/lowstate
reader (state_reader.py:4 GIL rule): NO listener callback.  Read-only: it
opens no command channel of any kind.

Wire format on stdout (the ONLY thing printed there):
    SDK_LEG_READY                                   (once, after Init)
    {"t": <epoch>, "quat": [w,x,y,z], "q": [12 floats], "qw": [3 floats]}   (<= 50 Hz)
("qw" = waist yaw/roll/pitch, motor idx 12-14, B8k: the measured body pose.)
Everything else goes to stderr (no per-message printing: stdout-flood law).
"""

import json
import time

DEFAULT_IFACE = 'eth0'
EMIT_HZ = 50.0
READ_TIMEOUT_S = 0.1
READY_LINE = 'SDK_LEG_READY'


