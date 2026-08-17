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


def odometry_fields(state):
    """Plain-dict form of the nav_msgs/Odometry the node publishes.  No twist,
    no covariance claims."""
    qx, qy, qz, qw = yaw_to_quat(state['yaw'])
    return {
        'frame_id': FRAME_ID,
        'child_frame_id': CHILD_FRAME_ID,
        'x': float(state['x']),
        'y': float(state['y']),
        'z': 0.0,
        'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
    }


# --------------------------------------------------------------------------
# the integrator
# --------------------------------------------------------------------------
def _finite(v):
    return not (math.isnan(v) or math.isinf(v))


class OdomIntegrator(object):
    """IMU-yaw + stance-foot-FK planar odometry.

    update(t, quat_wxyz, q12) -> {'x','y','yaw','t'} or None (rejected).
    Yaw is ABSOLUTE from the IMU quaternion each sample.  Position integrates
    -(R_k f_k - R_{k-1} f_{k-1}) for the stance foot chosen at k-1 on BOTH
    samples; increments across a gap > max_dt_s are skipped.  No gyro fallback.
    """

    def __init__(self, foot_fk=default_fk, max_dt_s=MAX_DT_S,
                 stance_hysteresis_m=STANCE_HYSTERESIS_M):
        self._fk = foot_fk
        self.max_dt_s = float(max_dt_s)
        self.hyst = float(stance_hysteresis_m)
        self.rejected = 0
        self.accepted = 0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self._t_prev = None
        self._yaw_origin = None
        self._fw_prev = None      # (left, right) world coords, last sample
        self._stance = None       # 0 means left, 1 means right

    def _reject(self):
        self.rejected += 1
        return None

    def update(self, t, quat_wxyz, q12):
        try:
            t = float(t)
            w, x, y, z = (float(v) for v in quat_wxyz)
            q = [float(v) for v in q12]
        except (TypeError, ValueError):
            return self._reject()
        if len(q) != 12 or not _finite(t):
            return self._reject()
        if not all(_finite(v) for v in (w, x, y, z)):
            return self._reject()
        if not all(_finite(v) for v in q):
            return self._reject()
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if abs(norm - 1.0) > QUAT_NORM_TOL:
            return self._reject()
        if self._t_prev is not None and t <= self._t_prev:
            return self._reject()

        imu_yaw = quat_to_yaw(w, x, y, z)
        first = self._yaw_origin is None
        if first:
            self._yaw_origin = imu_yaw
        r = _quat_to_mat(w, x, y, z)
        fw = (_mv(r, self._fk(q, LEFT)), _mv(r, self._fk(q, RIGHT)))

        if first:
            self._stance = 0 if fw[0][2] <= fw[1][2] else 1
        else:
            if t - self._t_prev <= self.max_dt_s:
                s = self._stance
                dx = -(fw[s][0] - self._fw_prev[s][0])
                dy = -(fw[s][1] - self._fw_prev[s][1])
                # IMU world -> odom, yaw origin = first sample
                c = math.cos(-self._yaw_origin)
                sn = math.sin(-self._yaw_origin)
                self.x += c * dx - sn * dy
                self.y += sn * dx + c * dy
                other = 1 - s
                if fw[other][2] < fw[s][2] - self.hyst:
                    self._stance = other
            else:
                # gap: hold position, re-set baseline + stance
                self._stance = 0 if fw[0][2] <= fw[1][2] else 1
        self._fw_prev = fw
        self._t_prev = t
        self.yaw = wrap_pi(imu_yaw - self._yaw_origin)
        self.accepted += 1
        return {'x': self.x, 'y': self.y, 'yaw': self.yaw, 't': t}


# --------------------------------------------------------------------------
# a measured body pose for the point-cloud lane
#
# gateway/pointcloud_client.py (pc_client.py on the robot)
# refuses point-cloud verdicts with UNKNOWN(pose_unmeasured)
# unless it can compare TWO measured poses (POSE_KEYS:
# torso_z / pitch / roll / waist_pitch, src "measured";
# equal = |dz| <= 1 cm, |d angle| <= 1 deg)
# with no writer `pose` verb available, this
# process reads
# rt/lowstate, so it hands the pose over read-only,
# no 250 Hz writer involved
#
#   torso_z      torso_link origin above the LOWER sole,
#                along gravity.  Waist origin offset
#                (pelvis -> waist_roll_link (-0.0039635, 0, 0.044),
#                XML) rotated by waist yaw
#   pitch, roll  IMU frame, ZYX Euler from the quaternion
#   waist_pitch  motor 14 (waist locked, constant still a key)
# for pose EQUALITY between two moments, not calibration
# --------------------------------------------------------------------------
WAIST_ROLL_IN_PELVIS = (-0.0039635, 0.0, 0.044)   # waist_roll_link pos, from the XML
WAIST_YAW_IDX, WAIST_ROLL_IDX, WAIST_PITCH_IDX = 12, 13, 14
BODY_POSE_WINDOW_S = 0.5      # rolling median window, sway/noise
BODY_POSE_MIN_SAMPLES = 5
BODY_POSE_HZ = 10.0
BODY_POSE_TOPIC = '/openbot/body_pose'   # std_msgs/String JSON, BodyPoseFilter.pose


def quat_to_pitch_roll(w, x, y, z):
    """(pitch, roll) in radians, ZYX Euler, of a unit quaternion wxyz."""
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def body_pose_sample(t, quat_wxyz, q12, waist3):
    """One instantaneous pose dict, or None for unusable input."""
    try:
        w, x, y, z = (float(v) for v in quat_wxyz)
        q = [float(v) for v in q12]
        wy, wr, wp = (float(v) for v in waist3)
        t = float(t)
    except (TypeError, ValueError):
        return None
    vals = [w, x, y, z, wy, wr, wp, t] + q
    if len(q) != 12 or not all(_finite(v) for v in vals):
        return None
    if abs(math.sqrt(w * w + x * x + y * y + z * z) - 1.0) > QUAT_NORM_TOL:
        return None
    r = _quat_to_mat(w, x, y, z)
    c, sn = math.cos(wy), math.sin(wy)
    ox, oy, oz = WAIST_ROLL_IN_PELVIS
    torso = (c * ox - sn * oy, sn * ox + c * oy, oz)
    soles = (_mv(r, foot_in_pelvis(q, LEFT)), _mv(r, foot_in_pelvis(q, RIGHT)))
    lower = soles[0] if soles[0][2] <= soles[1][2] else soles[1]
    tz = _mv(r, torso)[2] - lower[2]
    pitch, roll = quat_to_pitch_roll(w, x, y, z)
    return {"t": t, "torso_z": tz, "pitch": pitch, "roll": roll,
            "waist_pitch": wp}


def _median(vals):
    v = sorted(vals)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


class BodyPoseFilter(object):
    """Rolling median of body_pose_sample() over BODY_POSE_WINDOW_S.

    pose(now) -> {"src": "measured", "torso_z", "pitch", "roll",
    "waist_pitch", "t", "n", "via": "odom"} or None when fewer than
    BODY_POSE_MIN_SAMPLES fresh samples sit in the window (never a stale or
    fabricated pose)."""

    def __init__(self, window_s=BODY_POSE_WINDOW_S,
                 min_samples=BODY_POSE_MIN_SAMPLES):
        self.window_s = float(window_s)
        self.min_samples = int(min_samples)
        self._buf = []

