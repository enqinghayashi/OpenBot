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


def sample_from_msg(msg, now):
    """Build the wire dict from an hg LowState_ message."""
    quat = [float(msg.imu_state.quaternion[i]) for i in range(4)]   # order is wxyz
    q = [float(msg.motor_state[i].q) for i in range(12)]            # motor idx 0-11
    qw = [float(msg.motor_state[i].q) for i in range(12, 15)]       # waist idx 12-14
    return {'t': now, 'quat': quat, 'q': q, 'qw': qw}


def stream(sub, out, clock=time.time, hz=EMIT_HZ, max_lines=None):
    """Poll-only loop: latest-wins decimation of the ~500 Hz stream to <= hz.
    Returns the number of lines written (only differs from forever when
    max_lines is given, which tests use)."""
    period = 1.0 / hz
    last_emit = 0.0
    n = 0
    while max_lines is None or n < max_lines:
        msg = sub.Read(READ_TIMEOUT_S)
        if msg is None:
            continue
        now = clock()
        if now - last_emit < period - 1e-6:
            continue
        last_emit = now
        out.write(json.dumps(sample_from_msg(msg, now)) + '\n')
        out.flush()
        n += 1
    return n


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    iface = argv[0] if argv else DEFAULT_IFACE
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    ChannelFactoryInitialize(0, iface)
    sub = ChannelSubscriber('rt/lowstate', LowState_)
    sub.Init()
    sys.stdout.write(READY_LINE + '\n')
    sys.stdout.flush()
    sys.stderr.write('odom_sdk_leg: subscribed rt/lowstate on %s\n' % iface)
    try:
        stream(sub, sys.stdout)
    except KeyboardInterrupt:
        pass
    return 0
