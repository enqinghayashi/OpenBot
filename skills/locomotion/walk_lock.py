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

    def predict(self, now, pose=None):
        """(x, y, gate_m, source) -- where the locked object should be NOW in
        the body frame, and how far a row may sit from it."""
        st = self.st
        if st is None:
            return None
        gate = st.get('gate', LOCK_GATE_M)
        if pose is not None and st['w'] is not None:
            x, y = world_to_body(pose, st['w'][0], st['w'][1])
            return x, y, gate, 'odom'
        dt = max(0.0, float(now) - st['t'])
        d = math.hypot(st['x'], st['y'])
        extra = min(LOCK_MOTION_CAP_M,
                    (LOCK_VX_MAX + LOCK_YAW_RATE_MAX * d) * dt)
        return st['x'], st['y'], gate + extra, 'body'

    def select(self, rows, now, pose=None, min_conf=0.0):
        """(row|None, how, events) for one /object_detections_body packet.

        how: 'iid' (same tracker id, consistent position), 'near' (a row of
        the target label at the predicted position), 'near_other' (a row of
        ANOTHER label there: the class flipped), or None.  A same-id row that
        jumped is NOT followed (event JUMP).  The caller passes rows already
        stripped of the robot's own body (walk_node SELF_ZONE_M)."""
        st = self.st
        events = []
        if st is None:
            return None, None, events
        px, py, gate, _src = self.predict(now, pose)
        zr = self.z_ref()
        cands = []
        for r in rows or ():
            if not isinstance(r, dict):
                continue
            try:
                if float(r.get('conf', 0) or 0) < min_conf:
                    continue
            except (TypeError, ValueError):
                continue
            p = _xyz(r)
            if p is None:
                continue
            cands.append((r, p, math.hypot(p[0] - px, p[1] - py),
                          abs(p[2] - zr)))
        pick, how = None, None
        if st['iid'] is not None:
            for r, p, dxy, dz in cands:
                if _iid(r) != st['iid']:
                    continue
                if dxy <= gate and dz <= LOCK_DZ_M:
                    pick, how = (r, p), 'iid'
                else:
                    note = (round(p[0], 1), round(p[1], 1))
                    if note != self._jump_note:
                        self._jump_note = note
                        events.append(
                            '[alock] JUMP %s#%s det=%s moved %.2f m dz=%.2f '
                            '(gate %.2f/%.2f): not followed'
                            % (st['label'], st['iid'], r.get('label'), dxy,
                               dz, gate, LOCK_DZ_M))
                break
        if pick is None:
            near = [(0 if str(r.get('label')) == st['label'] else 1, dxy, r, p)
                    for r, p, dxy, dz in cands
                    if dxy <= gate and dz <= LOCK_DZ_M
                    and (st['iid'] is None or _iid(r) != st['iid'])
                    and (st.get('other', True)
                         or str(r.get('label')) == st['label'])]
            if near:
                near.sort(key=lambda t: (t[0], t[1]))
                other, dxy, r, p = near[0]
                pick, how = (r, p), ('near_other' if other else 'near')
                rid = _iid(r)
                if rid is not None and rid != st['iid']:
                    events.append('[alock] REACQUIRED %s#%s (was #%s) det=%s '
                                  'dist=%.3f' % (st['label'], rid, st['iid'],
                                                 r.get('label'), dxy))
                    st['iid'] = rid
                    st['reacq'] += 1
        if pick is None:
            return None, None, events
        r, p = pick
        st['x'], st['y'] = p[0], p[1]
        st['zs'] = (st['zs'] + [p[2]])[-LOCK_Z_HIST_N:]
        st['t'] = float(now)
        st['n'] += 1
        if pose is not None:
            st['w'] = body_to_world(pose, p[0], p[1])
        seen = str(r.get('label'))
        if seen != st['seen']:
            events.append('[alock] FOLLOW %s#%s det %s->%s (%s)'
                          % (st['label'], st['iid'], st['seen'], seen, how))
            st['seen'] = seen
        return r, how, events


# ════════════════════════════════════════════════════════════════════════════
# F5 -- turn back to where we lost the target
# ════════════════════════════════════════════════════════════════════════════
REACQ_DWELL_S = 1.5         # stop and look first (a flicker usually comes back)
#: 0.30 rad/s does not turn the robot in place; 0.50 rad/s is the sweep
#: rate that does.
REACQ_OMEGA = 0.50          # rad/s, how fast we swing back
REACQ_TURN_MAX_S = 4.0      # caps the turn at ~69 deg
REACQ_LOOK_S = 1.5          # keep looking once the turn is done
REACQ_TOL_RAD = 0.09        # same as walk_node ALIGN_TOL_RAD


def reacquire_turn(err, omega=REACQ_OMEGA, max_s=REACQ_TURN_MAX_S,
                   tol=REACQ_TOL_RAD, omega_sign=-1.0):
    """(omega_cmd, duration_s) of the open-loop turn that brings a target at
    bearing ``err`` (walk_node yaw error, + = right) to the centre;
    (0.0, 0.0) when it is already within ``tol`` or ``err`` is unusable.
    omega_cmd follows walk_node's OMEGA_SIGN (+omega turns LEFT)."""
    try:
        e = float(err)
    except (TypeError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(e) or abs(e) <= tol:
        return 0.0, 0.0
    w = abs(float(omega))
    return (omega_sign * (w if e > 0 else -w)), min(float(max_s), abs(e) / w)
