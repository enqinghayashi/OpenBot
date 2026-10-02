"""pose_fk.py -- MEASURED torso pose from the writer's own lowstate (point-cloud batch 3b, hook H-WR-1).

Used by the socket verb `pose` of the box writer (installed by build_box_writer_pose.py, which
copies this file beside main_merged_writer.py).  Pure functions: numpy + math only.  No unitree,
ROS, yaml or XML import.  The XML is NEVER parsed on the robot: every kinematic constant below is
embedded, with the XML element it was copied from named in a comment.  The test
skills/writer/tools/test_pose_verb.py parses the XML and proves each constant against it.

    torso_pose(q35, quat_wxyz, mode_machine, motor_mode) -> dict

The dict holds only the keys whose source is available, in this order, floats rounded to 5 decimals:

    torso_z       metres
    pitch, roll   radians
    waist_pitch   radians                  (waist_pitch and waist_src are present together,
    waist_src     "measured" | "locked"     or both absent)
`src` and `t` are added by the verb (the writer method), not here.  torso_pose never raises: any
input it cannot use simply leaves its keys out.  A key is NEVER a placeholder: no 0.0 and no NaN
stands in for a missing measurement.  The single 0.0 that is not a measurement is the documented
locked waist (waist_src "locked").

INPUTS (the writer's lowstate dict, state_reader._unpack)
    q35         35 float32, motor order = G1JointIndex: 0-5 left leg, 6-11 right leg
                (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll), 12-14 waist
                yaw/roll/pitch, 15+ arms.  Only q[0:12] and q[14] are read.
    quat_wxyz   4 float32, ORDER (w, x, y, z), the PELVIS IMU quaternion (body -> world).
    mode_machine int; 6 = lock-waist (g1_29dof_lock_waist_rev_1_0), 5 = 3-DoF waist.
    motor_mode  35 int; 0 = the motor reports nothing (dark / limp), nonzero = alive.

DECIDED DEFINITIONS
  frames       world: z up.  pelvis body frame: x forward, y left, z up (the XML's pelvis body).
               The IMU frame is ASSUMED identical to the pelvis frame, as the writer's own
               obs_builder assumes when it builds projected gravity from the same quaternion.
  pitch, roll  The ZYX Euler split of the pelvis rotation, R = Rz(yaw) . Ry(pitch) . Rx(roll), with
               the heading (yaw) discarded.  Pure yaw therefore gives pitch = roll = 0 and the
               heading never leaks into either angle.  With u = R^T . (0,0,1) = world-up in the
               pelvis frame = (2(xz - wy), 2(yz + wx), 1 - 2(x^2 + y^2)) for q = (w, x, y, z):
                   pitch = asin(-u_x)        POSITIVE = NOSE DOWN (a forward lean: the pelvis
                                             x axis points below the horizon)
                   roll  = atan2(u_y, u_z)   POSITIVE = LEFT SIDE UP (right side down: the pelvis
                                             y axis points above the horizon)
               Same formulas as skills/writer/tools/mj_stance_probe.py quat_roll_pitch.  pitch is
               in [-pi/2, pi/2], roll in (-pi, pi]; roll is ill-conditioned as |pitch| -> pi/2.
  torso_z      Height, in metres along WORLD-VERTICAL, of the ORIGIN of the XML body `torso_link`
               above the LOWEST SOLE POINT of the two feet.  Computed from the measured leg joints
               (motors 0-11), the waist treated as LOCKED AT ZERO, and the pelvis attitude from the
               IMU quaternion.  Only the world-up vector u enters (so pitch and roll, never yaw), and
               the pelvis' own height cancels, so nothing else is needed.
                 * torso_link is the body the XML names for the torso (XML line 154); the
                   projector's `torso` frame is that link; with the waist locked it is rigid in the
                   pelvis.  Its origin in the pelvis frame, waist at zero, is (-0.0039635, 0, 0.044)
                   (XML 146 waist_yaw_link at the pelvis origin, 150 waist_roll_link pos, 154
                   torso_link at its parent's origin).
                 * Sole points: the foot COLLISION geometry in the XML is exactly four spheres of
                   radius 0.005 on each `*_ankle_roll_link` body (XML 97-100 left, 136-139 right;
                   the ankle-roll mesh geom there is contype=0 conaffinity=0, i.e. visual only).
                   Centres, in that body's frame: (-0.05, +0.025, -0.03), (-0.05, -0.025, -0.03),
                   (0.12, +0.03, -0.03), (0.12, -0.03, -0.03).  The lowest point of a sphere along
                   world-vertical is its centre minus the radius, so the lowest sole point is
                   min over the 8 centres of z, minus 0.005.
               This is the quantity the MuJoCo reference in the test computes (mj_forward, torso
               body position minus the lowest sole-sphere point in the world frame).
  waist rule   mode_machine == 6 -> waist_pitch 0.0, waist_src "locked": a MECHANICAL CONSTANT, not
               a measurement.  (Stuart's waist is set to 1-DoF in the app and screwed: operator-
               reported.  The dark reading is MEASURED: motors 13/14 mode 0 with q exactly 0, and
               mode_machine 6 = g1_29dof_lock_waist_rev_1_0, 2026-10-04,
               .wf/probes/bodymode_probes_20261004_144116.txt lines 9, 17, 19.)  Otherwise,
               motor_mode[14] != 0 and q[14] finite -> waist_pitch = q[14], waist_src "measured".
               Any other case (mode_machine unknown, motor 14 dark, q[14] not finite, arrays too
               short) -> BOTH waist keys are omitted.
  omission     torso_z needs a usable quaternion AND 12 finite leg angles.  pitch and roll need a
               usable quaternion: shape (4,), all finite, norm >= 1e-6 (it is normalised before
               use).  A key whose source is unavailable is left out.

SOURCE OF THE CONSTANTS
  File: skills/writer/sim/g1_gear_wbc.xml (md5 8ffca967c299400100d80d2ce40a917f), the MuJoCo model of
  the G1 29-DoF used by the writer's own sim tools.  A chain row is
  (child <body pos> in the parent frame, child <body quat> as (w,x,y,z), <joint axis>); the joint
  angle turns the child about its axis in the CHILD frame:  R_child = R_parent . R(quat) . Rot(axis, q).
  Motor index -> joint is the G1JointIndex order, stated by POLICY_JOINTS in
  skills/writer/tools/mj_height_sweep.py:106-113 ("document order == G1JointIndex 0..14") and
  sim_joint_map.py:36-41, by the stance lane in the box writer (:897 "1 = L hip roll, 5 = L ankle
  roll, 7 = R hip roll, 11 = R ankle roll") and by the validated URDF leg table in
  skills/writer/tools/test_step_lane.py:119-134.

HONEST LIMITS (read before trusting a number)
  * NOT validated on the robot.  Everything here is checked against a MuJoCo model, never against a
    measurement of Stuart's body.
  * The sole geometry comes from the sim XML (four spheres of r = 5 mm per foot), not from a
    measurement of the real foot.  A real sole is a pad; the absolute torso_z carries that
    unmeasured offset.  The point-cloud consumer only COMPARES two poses, so a constant offset
    cancels, but it is still an offset.
  * The waist is a constant on this robot.  waist_pitch 0.0 with waist_src "locked" is the
    mechanical lock, not a reading.  Do not use it as evidence that the torso is upright.
  * If the waist is ever alive (mode_machine != 6), waist_pitch = q[14] is a measurement, but
    torso_z still treats the waist as zero.  The torso link ORIGIN lies on the waist roll and
    pitch axes, so torso_z does not depend on them (proved against MuJoCo in the test).  The
    torso link ORIENTATION does change with waist roll and pitch; that is what `waist_pitch`
    is for.
  * The waist YAW (motor 12) is alive even in lock-waist mode and is NOT read.  torso_z depends
    on it only through the 3.96 mm offset of the torso origin from the yaw axis: below 2 mm over
    the whole yaw range at 15 degrees of pelvis tilt (0.0039635 x sin 15 deg x 1.87).  pitch and
    roll are the PELVIS attitude (as decided), so with a non-zero waist yaw they are not the
    torso link's own pitch and roll (same tilt magnitude, split differently between the two),
    and a change of the waist yaw between two poses is invisible to this reply.
  * Ground contact is ASSUMED.  torso_z is the height above the lowest sole point of the two feet
    whether or not any foot carries load.  The lowstate keys used here say nothing about contact,
    so when both feet are airborne (gantry, fall, jump) the verb still reports the lowest-sole
    height, which is then NOT a height above the ground, and nothing in the reply tells a caller
    so.  No contact flag was added.  With one foot raised or on a step, the reference is the lower
    foot, not the support foot.
  * No lowstate age.  The verb's `t` is the wall time of the call, not of the lowstate: the
    lowstate dict carries no host timestamp.
"""

import math
import numbers

import numpy as np

# ------------------------------------------------------------------ the constants
# XML line numbers below: skills/writer/sim/g1_gear_wbc.xml
_AX_X, _AX_Y, _AX_Z = 0, 1, 2
_Q_ID = (1.0, 0.0, 0.0, 0.0)
# left_hip_roll_link quat="0.996179 0 -0.0873386 0" (73); right (112)
_Q_HIP_ROLL = (0.996179, 0.0, -0.0873386, 0.0)
# left_knee_link quat="0.996179 0 0.0873386 0" (83); right (122)
_Q_KNEE = (0.996179, 0.0, 0.0873386, 0.0)

# left leg, motors 0..5, root to tip
_LEG_L = (
    ((0.0, 0.064452, -0.1027), _Q_ID, _AX_Y),             # 68-70 left_hip_pitch_link, axis y
    ((0.0, 0.052, -0.030465), _Q_HIP_ROLL, _AX_X),        # 73-75 left_hip_roll_link, axis x
    ((0.025001, 0.0, -0.12412), _Q_ID, _AX_Z),            # 78-80 left_hip_yaw_link, axis z
    ((-0.078273, 0.0021489, -0.17734), _Q_KNEE, _AX_Y),   # 83-85 left_knee_link, axis y
    ((0.0, -9.4445e-05, -0.30001), _Q_ID, _AX_Y),         # 88-90 left_ankle_pitch_link, axis y
    ((0.0, 0.0, -0.017558), _Q_ID, _AX_X),                # 93-95 left_ankle_roll_link, axis x
)
# right leg, motors 6..11, root to tip
_LEG_R = (
    ((0.0, -0.064452, -0.1027), _Q_ID, _AX_Y),            # 107-109 right_hip_pitch_link, axis y
    ((0.0, -0.052, -0.030465), _Q_HIP_ROLL, _AX_X),       # 112-114 right_hip_roll_link, axis x
    ((0.025001, 0.0, -0.12412), _Q_ID, _AX_Z),            # 117-119 right_hip_yaw_link, axis z
    ((-0.078273, -0.0021489, -0.17734), _Q_KNEE, _AX_Y),  # 122-124 right_knee_link, axis y
    ((0.0, 9.4445e-05, -0.30001), _Q_ID, _AX_Y),          # 127-129 right_ankle_pitch_link, axis y
    ((0.0, 0.0, -0.017558), _Q_ID, _AX_X),                # 132-134 right_ankle_roll_link, axis x
)
# foot collision spheres, *_ankle_roll_link frame, same both feet
# <geom size="0.005" ...> 97-100 left, 136-139 right; no type given -> sphere
_FOOT_SPHERES = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)
_SOLE_RADIUS = 0.005          # the geoms' size="0.005"
# torso_link origin in the pelvis frame, waist at zero:
# 146 waist_yaw no pos -> 150 waist_roll pos="-0.0039635 0 0.044" -> 154 torso_link
_TORSO_IN_PELVIS = (-0.0039635, 0.0, 0.044)

_MODE_LOCK_WAIST = 6          # g1_29dof_lock_waist_rev_1_0; measured, see docstring
_MIN_QUAT_NORM = 1e-6


def _quat_to_mat(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _prep(chain):
    return tuple((np.array(off, dtype=np.float64), _quat_to_mat(*quat), ax)
                 for off, quat, ax in chain)


_CHAIN_L = _prep(_LEG_L)
_CHAIN_R = _prep(_LEG_R)
_S_FOOT = np.array(_FOOT_SPHERES, dtype=np.float64)
_P_TORSO = np.array(_TORSO_IN_PELVIS, dtype=np.float64)


def _rot(axis, a):
    """Rotation by angle a about the principal axis (0=x, 1=y, 2=z), right-handed."""
    c = math.cos(a)
    s = math.sin(a)
    if axis == _AX_X:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == _AX_Y:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _lowest_sole_z(chain, q6, u):
    """Lowest sphere-centre height of one foot, in the world frame up to the pelvis' own height.

    z of a pelvis-frame point p is u . p (u = world-up in the pelvis frame); the pelvis height is
    common to every point and cancels in torso_z.
    """
    rot = np.eye(3)
    pos = np.zeros(3)
    for k in range(6):
        off, rfix, ax = chain[k]
        pos = pos + rot.dot(off)
        rot = rot.dot(rfix).dot(_rot(ax, float(q6[k])))
    zc = float(u.dot(pos)) + _S_FOOT.dot(u.dot(rot))
    return float(zc.min())


def _up_in_pelvis(quat_wxyz):
    """World-up expressed in the pelvis frame, from the (w, x, y, z) quaternion, or None."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        return None
    n = math.sqrt(float(q.dot(q)))
    if not n >= _MIN_QUAT_NORM:
        return None
    w, x, y, z = (q / n).tolist()
    return (2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y))


def _leg_angles(q35):
    q = np.asarray(q35, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] < 12:
        return None
    legs = q[0:12]
    if not np.isfinite(legs).all():
        return None
    return legs


def _torso_height(legs, u):
    uv = np.array(u, dtype=np.float64)
    lowest = min(_lowest_sole_z(_CHAIN_L, legs[0:6], uv),
                 _lowest_sole_z(_CHAIN_R, legs[6:12], uv))
    return float(uv.dot(_P_TORSO)) - lowest + _SOLE_RADIUS


def _pitch_of(u):
    return math.asin(max(-1.0, min(1.0, -u[0])))


def _roll_of(u):
    return math.atan2(u[1], u[2])


def _as_int(x):
    if x is None or isinstance(x, bool):
        return None
    if not isinstance(x, numbers.Real):
        return None
    f = float(x)
    if not math.isfinite(f) or f != int(f):
        return None
    return int(f)


def _waist(q35, mode_machine, motor_mode):
    """(waist_pitch, waist_src) or None.  See the module docstring, `waist rule`."""
    mm = _as_int(mode_machine)
    if mm is None:
        return None
    if mm == _MODE_LOCK_WAIST:
        # mechanical constant, not a measurement
        return (0.0, "locked")
    mot = np.asarray(motor_mode)
    if mot.ndim != 1 or mot.shape[0] < 15 or int(mot[14]) == 0:
        return None
    q = np.asarray(q35, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] < 15:
        return None
    v = float(q[14])
    if not math.isfinite(v):
        return None
    return (v, "measured")


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 -- a bad input just leaves its keys out, it never raises
        return None


def _r5(x):
    """Round to 5 decimals; '+ 0.0' turns -0.0 into 0.0."""
    return round(float(x), 5) + 0.0


def _put(out, key, value):
    if value is None:
        return
    value = float(value)
    if math.isfinite(value):
        out[key] = _r5(value)


def torso_pose(q35, quat_wxyz, mode_machine, motor_mode):
    """Measured torso pose from one lowstate.  Returns a dict; see the module docstring."""
    u = _safe(_up_in_pelvis, quat_wxyz)
    legs = _safe(_leg_angles, q35)
    z = None
    if u is not None and legs is not None:
        z = _safe(_torso_height, legs, u)
    pitch = None
    roll = None
    if u is not None:
        pitch = _safe(_pitch_of, u)
        roll = _safe(_roll_of, u)
    waist = _safe(_waist, q35, mode_machine, motor_mode)
    out = {}
    _put(out, "torso_z", z)
    _put(out, "pitch", pitch)
    _put(out, "roll", roll)
    if waist is not None and math.isfinite(waist[0]):
        out["waist_pitch"] = _r5(waist[0])
        out["waist_src"] = waist[1]
    return out
