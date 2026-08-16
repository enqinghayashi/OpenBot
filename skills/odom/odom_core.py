"""odom_core.py -- pure-stdlib leg-FK + IMU odometry maths for Stuart (G1).

NO rclpy / nav_msgs / unitree_sdk2py / numpy / mujoco / scipy import here
(pinned by tests/test_odom_isolation.py).  Python-3.8-safe: no `X | None`,
no match, no walrus.  Nothing in this file touches a robot.

HONEST STATUS: yaw comes straight from the IMU quaternion (absolute each
sample, no gyro integration).  Position x,y comes from stance-foot forward
kinematics and is UNMEASURED on the robot (INFERRED flat-foot assumption).
Nothing here has run on the box: UNVERIFIABLE-FROM-HERE.
"""

import math
from collections import deque  # noqa: F401  (stdlib only; we keep it for callers)

# --------------------------------------------------------------------------
# constants.  Each is MEASURED / DERIVED / INFERRED
# --------------------------------------------------------------------------
# measured: the odom topic the walk node publishes
# ODOM_TOPIC = '/unitree/slam_mapping/odom'
ODOM_TOPIC = '/unitree/slam_mapping/odom'
FRAME_ID = 'odom'
CHILD_FRAME_ID = 'base_link'
# inferred: 50 Hz cap (leg decimates ~500 Hz lowstate)
PUBLISH_HZ = 50
# inferred: stricter than the walk node's 1.0 s staleness bound
#   `if odom_age < 0 or odom_age > 1.0:`
MAX_SAMPLE_AGE_S = 0.25
# inferred: 5 x 20 ms; never integrate a bigger gap
MAX_DT_S = 0.10
# inferred: switch when the other foot is > 1 cm lower
STANCE_HYSTERESIS_M = 0.01
# future-dating is_fresh tolerates (inferred;
# clock skew on one box is ~0, so 50 ms is generous)
FUTURE_TOL_S = 0.05
QUAT_NORM_TOL = 0.05

# --------------------------------------------------------------------------
# G1 leg chain, measured from
# skills/writer/sim/g1_gear_wbc.xml
# (line numbers = the XML lines holding the value;
#  pinned by tests/test_odom_fk.py, which re-parses)
# each link: (pos, fixed quat wxyz or None, joint axis)
# --------------------------------------------------------------------------
_Q_ROLL = (0.996179, 0.0, -0.0873386, 0.0)    # XML :73 / :112 (hip_roll)
_Q_KNEE = (0.996179, 0.0, 0.0873386, 0.0)     # XML :83 / :122 (knee)

LEFT_CHAIN = (
    # hip_pitch_link :68, axis Y (:70)
    ((0.0, 0.064452, -0.1027), None, (0, 1, 0)),
    # hip_roll_link :73, axis X (:75)
    ((0.0, 0.052, -0.030465), _Q_ROLL, (1, 0, 0)),
    # hip_yaw_link :78, axis Z (:80)
    ((0.025001, 0.0, -0.12412), None, (0, 0, 1)),
    # knee_link :83, axis Y (:85)
    ((-0.078273, 0.0021489, -0.17734), _Q_KNEE, (0, 1, 0)),
    # ankle_pitch_link :88, axis Y (:90)
    ((0.0, -9.4445e-05, -0.30001), None, (0, 1, 0)),
    # ankle_roll_link :93, axis X (:95)
    ((0.0, 0.0, -0.017558), None, (1, 0, 0)),
)
RIGHT_CHAIN = (
    # XML :107 / :109
    ((0.0, -0.064452, -0.1027), None, (0, 1, 0)),
    # XML :112 / :114
    ((0.0, -0.052, -0.030465), _Q_ROLL, (1, 0, 0)),
    # XML :117 / :119
    ((0.025001, 0.0, -0.12412), None, (0, 0, 1)),
    # XML :122 / :124
    ((-0.078273, -0.0021489, -0.17734), _Q_KNEE, (0, 1, 0)),
    # XML :127 / :129
    ((0.0, 9.4445e-05, -0.30001), None, (0, 1, 0)),
    # XML :132 / :134
    ((0.0, 0.0, -0.017558), None, (1, 0, 0)),
)

# lowstate motor order, measured in
# the stance probe's readout:
# 0-5 L: hip_pitch,hip_roll,hip_yaw,knee,ankle_pitch,ankle_roll
# 6-11 same for R
LEFT_IDX = (0, 1, 2, 3, 4, 5)
RIGHT_IDX = (6, 7, 8, 9, 10, 11)

# derived: midpoint of the four foot spheres
# ankle_roll frame, flat-foot assumption
# x centre 0.035: spheres "-0.05 0.025 -0.03" / "0.12 0.03 -0.03"
# z -0.03 minus 0.005 radius; XML :97-100, size 0.005
SOLE_PT_IN_ANKLE_ROLL = (0.035, 0.0, -0.035)

LEFT = 'left'
RIGHT = 'right'


# --------------------------------------------------------------------------
# little maths helpers
# --------------------------------------------------------------------------
def wrap_pi(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_to_yaw(w, x, y, z):
    """Yaw (rotation about +z) from a quaternion.  The SAME formula the
    consumer uses, skills/locomotion/walk_node.py:763-769:
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    """
    return math.atan2(2.0 * (w * z + x * y),
                      1.0 - 2.0 * (y * y + z * z))


def yaw_to_quat(yaw):
    """Pure-yaw quaternion (x, y, z, w) -- ROS xyzw order."""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _quat_to_mat(w, x, y, z):
    """Rotation matrix (3x3 nested tuples) of a unit quaternion wxyz."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def _axis_mat(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == (1, 0, 0):
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis == (0, 1, 0):
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _mm(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3))
                       for j in range(3)) for i in range(3))


def _mv(a, v):
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


_I3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def leg_fk(q12, side):
    """Return (p, R): the ankle_roll_link origin in the pelvis frame and its
    orientation matrix, for the lowstate 12-vector q12 and side 'left'/'right'."""
    chain = LEFT_CHAIN if side == LEFT else RIGHT_CHAIN
    idx = LEFT_IDX if side == LEFT else RIGHT_IDX
    p = (0.0, 0.0, 0.0)
    r = _I3
    for (pos, fq, axis), i in zip(chain, idx):
        off = _mv(r, pos)
        p = (p[0] + off[0], p[1] + off[1], p[2] + off[2])
        if fq is not None:
            r = _mm(r, _quat_to_mat(*fq))
        r = _mm(r, _axis_mat(axis, float(q12[i])))
    return p, r


def ankle_in_pelvis(q12, side):
    """ankle_roll_link origin in the pelvis frame (x, y, z)."""
    return leg_fk(q12, side)[0]


def foot_in_pelvis(q12, side):
    """Sole point in the pelvis frame (x, y, z): the ankle_roll origin plus the
    flat-foot sole point rotated by the ankle_roll orientation (INFERRED)."""
    p, r = leg_fk(q12, side)
    o = _mv(r, SOLE_PT_IN_ANKLE_ROLL)
    return (p[0] + o[0], p[1] + o[1], p[2] + o[2])


default_fk = foot_in_pelvis


# --------------------------------------------------------------------------
# freshness, startup, runtime
# --------------------------------------------------------------------------
def is_fresh(now, t_sample, max_age_s=MAX_SAMPLE_AGE_S):
    """True only for a finite sample time that is neither future-dated beyond
    FUTURE_TOL_S nor older than max_age_s.  None is never fresh."""
    if t_sample is None:
        return False
    try:
        age = float(now) - float(t_sample)
    except (TypeError, ValueError):
        return False
    if math.isnan(age) or math.isinf(age):
        return False
    if age < -FUTURE_TOL_S:
        return False
    return age <= max_age_s


def decide_startup(foreign_publisher_count):
    """'PROCEED' only when nobody else publishes the odom topic."""
    return 'PROCEED' if foreign_publisher_count == 0 else 'REFUSE'


def decide_runtime(total_publisher_count, ours=1):
    """'OK' while the only publisher is ours, 'YIELD' once another appears."""
    return 'OK' if total_publisher_count <= ours else 'YIELD'
