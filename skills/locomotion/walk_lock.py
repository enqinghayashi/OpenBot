"""walk_lock.py -- B8l pure laws for the walk FSM (stdlib only, py3.8).

Ships co-located with walk_node.py (robot: openbot/walk_lock.py; repo:
skills/locomotion/walk_lock.py).  No rclpy, no numpy: unit-testable.

F1  ApproachLock -- a BARE-label walk ('bin') keeps the ONE object it first
    saw.  MEASURED 2026-10-16 (walk.log 15:50:03-08, depth.log 14757-14781):
    the walk latched a bin at x -0.33 y 1.00 z -0.50, then followed rows
    labelled 'bin' at z -0.81 (y 1.36) and z +0.10 (y 0.42) -- other things --
    and its odometry-world target wandered 1.2 m while Stuart only turned.
    The walk instance lock (B8, '<label>#<iid>') only existed for targets the
    brain named by id.  The lock follows the object by tracker id ACROSS label
    changes (the tracker associates class-agnostically; on 10-16 the same bin
    track went bin -> backpack -> suitcase), and by geometry when the id dies
    (ids churned 246 -> 251 -> 253 in 10 s with the robot standing still,
    gateway.log [lock]).  A row of the target label ELSEWHERE is not the target.

F5  reacquire_turn -- after losing the target, turn back toward where it was
    (odometry when fresh, else the last body-frame bearing) instead of
    sweeping one fixed way.

F7  pose_command / pose_blocked -- walk to an odometry pose (the start point
    of a fetch) and face the requester.
"""
import math

# ── frame helpers we both use ────────────────────────────────────────────────
# detection body frame has x = RIGHT, y = FORWARD (walk_node, depth projector).
# odometry (/unitree/slam_mapping/odom, REP-103) is x = forward, y = left, yaw
# CCW (+omega = LEFT), so mind the swap.


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def body_to_world(pose, x_b, y_b):
    """Body-frame (right, forward) point -> odometry world (x, y)."""
    rx, ry, yaw = pose[0], pose[1], pose[2]
    fwd, left = y_b, -x_b
    return (rx + math.cos(yaw) * fwd - math.sin(yaw) * left,
            ry + math.sin(yaw) * fwd + math.cos(yaw) * left)


def world_to_body(pose, wx, wy):
    """Odometry world (x, y) -> body frame (right, forward)."""
    rx, ry, yaw = pose[0], pose[1], pose[2]
    dx, dy = wx - rx, wy - ry
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return -left, fwd


def bearing(x_b, y_b):
    """walk_node's yaw error: atan2(x, y); positive = target to the RIGHT."""
    return wrap_pi(math.atan2(x_b, y_b))


def _f(row, key):
    try:
        v = float(row.get(key))
    except (TypeError, ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def _xyz(row):
    if not isinstance(row, dict):
        return None
    x, y, z = _f(row, 'x'), _f(row, 'y'), _f(row, 'z')
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _iid(row):
    v = row.get('instance_id') if isinstance(row, dict) else None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ════════════════════════════════════════════════════════════════════════════
# F1 -- the lock that sticks with one object
# ════════════════════════════════════════════════════════════════════════════
#: how far (m) a row may sit from the PREDICTED spot. the bin's own rows spread
#: 0.20 m standing still (x -0.25..-0.49, y 0.92..1.04 within 0.6 s) while
#: the rejected rows sat 0.34 m (z -0.75), 0.66 m and 0.41 m (z -0.81) out.
LOCK_GATE_M = 0.25
#: a PICKABLE thing (small: bottle, cup, banana) gets a tighter gate and never
#: re-acquires a row of ANOTHER label by position, only by its tracker id:
#: a bottle 0.16 m from the target does not count as the target
#: banana (same rule as hunt_law.continuation allow_near).
LOCK_GATE_PICK_M = 0.15
#: height gate (m) against the median of the lock's recent z. the bin read
#: z -0.47..-0.55; the bogus rows sat at -0.75 / -0.81 / +0.10.
LOCK_DZ_M = 0.20
LOCK_Z_HIST_N = 7
#: with no fresh odometry we add the robot's own motion since the last lock row
#: to the gate (walk_node G18 law), but only up to this much.
LOCK_MOTION_CAP_M = 0.60
LOCK_VX_MAX = 0.30          # same number as walk_node VX_WALK
LOCK_YAW_RATE_MAX = 0.60    # walk_node YAW_RATE_MAX (K_THETA * OMEGA_MAX), copied over
#: odometry older than this doesn't get used for prediction (walk_node's latch
#: fallback refuses odom past 1.0 s; here we're stricter).
LOCK_ODOM_FRESH_S = 0.5


class ApproachLock(object):
    """The ONE object a bare-label walk follows.  Not thread-safe by itself:
    walk_node calls it under its _targets_lock."""

    def __init__(self):
        self.st = None
        self._jump_note = None

    def armed(self):
        return self.st is not None

    def arm(self, label, row, now, pose=None, large=True):
        """Lock onto ``row`` (the walk's first chosen row of ``label``).
        ``pose`` = fresh odometry (x, y, yaw) or None.  ``large`` = the
        target's stop tier is 'large' (bin, chair): False -> LOCK_GATE_PICK_M
        and no re-acquire through another label by position.  Returns the
        log event or None (a row without a usable x/y/z arms nothing)."""
        p = _xyz(row)
        if p is None:
            return None
        st = {'label': str(label), 'iid': _iid(row), 'x': p[0], 'y': p[1],
              'zs': [p[2]], 't': float(now), 'seen': str(row.get('label')),
              'w': None, 'reacq': 0, 'n': 1, 't_arm': float(now),
              'gate': LOCK_GATE_M if large else LOCK_GATE_PICK_M,
              'other': bool(large)}
        if pose is not None:
            st['w'] = body_to_world(pose, p[0], p[1])
        self.st = st
        self._jump_note = None
        return ('[alock] ARMED %s#%s x=%.3f y=%.3f z=%.3f world=%s gate=%.2f%s'
                % (label, st['iid'], p[0], p[1], p[2],
                   ('(%.3f, %.3f)' % st['w']) if st['w'] else '-', st['gate'],
                   '' if large else ' (pickable: other labels by id only)'))

    def clear(self, why):
        st, self.st = self.st, None
        if st is None:
            return None
        return ('[alock] CLEARED %s#%s (%s; %d rows, %d re-acquired)'
                % (st['label'], st['iid'], why, st['n'], st['reacq']))

    def z_ref(self):
        return _median(self.st['zs']) if self.st else None

