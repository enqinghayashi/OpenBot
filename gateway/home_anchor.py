"""home_anchor.py -- B8l F7: where a fetch started and who asked (stdlib, py3.8).

Operator request 2026-10-17: "hey stuart, can you go get me a bottle of
water?" -> Stuart navigates, finds the bottle, walks to it, picks it up,
REMEMBERS where it first walked from, returns there and gives the bottle to
the person near that spot; "any person-detection that's away (far) from
Stuart's original location will not be regarded as the person who asked".

The anchor is an odometry pose (/unitree/slam_mapping/odom, REP-103: x
forward, y left, yaw CCW) plus, when a person was in view at that moment,
the requester's odometry-world position.  Detection rows are body frame
(x RIGHT, y FORWARD; walk_node / depth projector).

Honest limit: the odometry is leg-FK dead reckoning.  At rest it held 2 mm /
0.4 deg over 2 min (10-16), but its WALKING accuracy has never been measured
-- the return lands wherever the odometry thinks home is.
"""
import math
import threading

HOME_TTL_S = 900.0          # 15 min, then gone
#: the voice lane fires navigate BEFORE the words are transcribed
#: (17:43:50.650 navigate, 17:43:50.824 words), so every empty-handed
#: walk keeps a CANDIDATE start; the pick promotes it while fresh.
CANDIDATE_MAX_AGE_S = 180.0
HOME_STAY_M = 0.50          # 'auto' keeps it within this of home
REQ_RADIUS_M = 1.50         # near requester
REQ_RADIUS_NOREQ_M = 2.00   # ... or near home, no requester
REQ_MAX_RANGE_M = 3.00      # max person range
#: ... and never this close: 'person' rows inside SELF_ZONE_M (0.30)
#: are the robot's own idle arm (depth.log 1009b: 641 rows at
#: 0.234-0.244 m, labelled keyboard/person/backpack/remote).
REQ_MIN_RANGE_M = 0.35
REQ_MAX_BEARING_RAD = 1.0   # ~57 deg off centre
REQ_MIN_CONF = 0.50         # = TURN_PERSON_MIN_CONF
REQ_MIN_H_M = 0.30          # = TURN_PERSON_MIN_HEIGHT_M
ODOM_FRESH_S = 0.5


def body_to_world(pose, x_b, y_b):
    rx, ry, yaw = pose[0], pose[1], pose[2]
    fwd, left = y_b, -x_b
    return (rx + math.cos(yaw) * fwd - math.sin(yaw) * left,
            ry + math.sin(yaw) * fwd + math.cos(yaw) * left)


def world_to_body(pose, wx, wy):
    rx, ry, yaw = pose[0], pose[1], pose[2]
    dx, dy = wx - rx, wy - ry
    fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
    left = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return -left, fwd


def _f(row, key):
    try:
        v = float(row.get(key))
    except (TypeError, ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def is_plausible_person(row):
    """A 'person' row the requester rules may use (conf, height, x/y)."""
    if not isinstance(row, dict) or str(row.get('label', '')).lower() != 'person':
        return False
    conf = _f(row, 'conf')
    if conf is None or conf < REQ_MIN_CONF:
        return False
    h = _f(row, 'height_m')
    if h is not None and h < REQ_MIN_H_M:
        return False
    return _f(row, 'x') is not None and _f(row, 'y') is not None


def pick_requester(rows):
    """The person row that asked: the nearest plausible person in front
    (REQ_MAX_RANGE_M, REQ_MAX_BEARING_RAD), or None."""
    best = None
    for r in rows or ():
        if not is_plausible_person(r):
            continue
        x, y = _f(r, 'x'), _f(r, 'y')
        d = math.hypot(x, y)
        if (y <= 0.0 or d > REQ_MAX_RANGE_M or d < REQ_MIN_RANGE_M
                or abs(math.atan2(x, y)) > REQ_MAX_BEARING_RAD):
            continue
        if best is None or d < best[0]:
            best = (d, r)
    return None if best is None else best[1]


class HomeAnchor(object):
    """The start point of the current fetch.  Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self._st = None
        self._cand = None

    def active(self, now):
        with self._lock:
            return self._live(now) is not None

    def _live(self, now):
        st = self._st
        if st is not None and float(now) - st['t'] > HOME_TTL_S:
            self._st = None
            st = None
        return st

    def mark(self, pose, rows, now, src='mark'):
        """Record ``pose`` (x, y, yaw) as home, and the requester from
        ``rows`` (body frame) if a person is in view.  Returns the event
        string, or None when there is no pose."""
        if pose is None:
            return None
        req = pick_requester(rows)
        st = {'x': float(pose[0]), 'y': float(pose[1]), 'yaw': float(pose[2]),
              't': float(now), 'src': src, 'req': None, 'req_iid': None,
              'req_body': None, 'held': False}
        if req is not None:
            xb, yb = _f(req, 'x'), _f(req, 'y')
            st['req'] = body_to_world(pose, xb, yb)
            st['req_body'] = (round(xb, 3), round(yb, 3))
            iid = req.get('instance_id')
            st['req_iid'] = iid if isinstance(iid, int) and not isinstance(iid, bool) else None
        with self._lock:
            self._st = st
        return ('MARKED home (%.3f, %.3f, %.1f deg) src=%s requester=%s'
                % (st['x'], st['y'], math.degrees(st['yaw']), src,
                   ('person#%s at body (%.2f, %.2f) world (%.3f, %.3f)'
                    % (st['req_iid'], st['req_body'][0], st['req_body'][1],
                       st['req'][0], st['req'][1])) if req is not None
                   else 'none in view'))

    def candidate(self, pose, rows, now):
        """B8l-hf3: remember where an empty-handed walk started (pose + the
        person in front) WITHOUT making it the start point.  Returns the event."""
        if pose is None:
            return None
        req = pick_requester(rows)
        cand = {'pose': (float(pose[0]), float(pose[1]), float(pose[2])), 't': float(now),
                'req': None if req is None else dict(req)}
        with self._lock:
            self._cand = cand
        return ('CANDIDATE home (%.3f, %.3f, %.1f deg) requester=%s'
                % (cand['pose'][0], cand['pose'][1], math.degrees(cand['pose'][2]),
                   'person#%s' % req.get('instance_id') if req is not None else 'none in view'))

    def promote(self, pose, rows, now, max_age_s=CANDIDATE_MAX_AGE_S):
        """B8l-hf3: the fetch is confirmed (its first pick): a candidate at most
        ``max_age_s`` old becomes the start point (where the robot stood when
        the walk began); without one the robot never walked, so ``pose`` (here)
        is the start point.  Returns the event."""
        with self._lock:
            cand, self._cand = self._cand, None
        if cand is not None and float(now) - cand['t'] <= max_age_s:
            rows_then = [cand['req']] if cand['req'] is not None else []
            ev = self.mark(cand['pose'], rows_then, cand['t'], src='candidate')
            return None if ev is None else ev + ' (promoted %.0f s later)' % (float(now) - cand['t'])
        return self.mark(pose, rows, now, src='pick')

    def auto(self, pose, rows, now):
        """'auto' (navigate with an empty hand): keep a live anchor while the
        robot is still within HOME_STAY_M of it (a re-navigate from the start
        point), else mark here.  Returns (marked, event)."""
        if pose is None:
            return False, None
        with self._lock:
            st = self._live(now)
            if st is not None:
                d = math.hypot(pose[0] - st['x'], pose[1] - st['y'])
                if d <= HOME_STAY_M:
                    if st['req'] is None:
                        req = pick_requester(rows)
                        if req is not None:     # requester spotted
                            st['req'] = body_to_world(pose, _f(req, 'x'), _f(req, 'y'))
                            st['req_body'] = (round(_f(req, 'x'), 3),
                                              round(_f(req, 'y'), 3))
                            return False, ('KEPT home (%.2f m away); requester '
                                           'now seen at world (%.3f, %.3f)'
                                           % (d, st['req'][0], st['req'][1]))
                    return False, 'KEPT home (%.2f m away)' % d
        return True, self.mark(pose, rows, now, src='auto')

    def set_held(self, now):
        """The fetched object is in the hand (a pick succeeded during the fetch):
        from now on a stop only PAUSES the fetch (operator decision 2026-10-17).
        Returns the event string, or None (no anchor)."""
        with self._lock:
            st = self._live(now)
            if st is None:
                return None
            st['held'] = True
        return 'HELD home: the fetched object is in the hand (a stop now pauses)'

    def stop(self, why, now):
        """A stop / cancel / 'wait' / 'never mind' (operator decision 2026-10-17):
        before the pick the fetch is ABANDONED (cleared); with the object in the
        hand it is only paused (kept).  Returns the event string or None."""
        with self._lock:
            st = self._live(now)
            if st is None:
                return None
            if st.get('held'):
                return 'KEPT home (%s): the fetched object is in the hand, the fetch is paused' % why
            self._st = None
        return 'CLEARED home (%s): stopped before the pick, the fetch is abandoned' % why

    def clear(self, why):
        with self._lock:
            st, self._st = self._st, None
        if st is None:
            return None
        return 'CLEARED home (%s)' % why

    def goal(self, now):
        """The /walk_pose goal for the return, or None (no anchor)."""
        with self._lock:
            st = self._live(now)
            if st is None:
                return None
            g = {'x': round(st['x'], 4), 'y': round(st['y'], 4),
                 'yaw': round(st['yaw'], 4), 'label': 'home'}
            if st['req'] is not None:
                g['face_x'] = round(st['req'][0], 4)
                g['face_y'] = round(st['req'][1], 4)
            return g

    def snapshot(self, pose, now):
        """The anchor as a JSON-able dict (None = no anchor).  With a fresh
        ``pose`` it also carries where home and the requester are NOW in the
        body frame, and the radius a person must be within."""
        with self._lock:
            st = self._live(now)
            if st is None:
                return None
            st = dict(st)
        out = {'home': [round(st['x'], 3), round(st['y'], 3),
                        round(math.degrees(st['yaw']), 1)],
               'age_s': round(float(now) - st['t'], 1), 'src': st['src'],
               'held': bool(st.get('held')),
               'requester_seen': st['req'] is not None,
               'requester_world': (None if st['req'] is None else
                                   [round(st['req'][0], 3), round(st['req'][1], 3)]),
               'radius_m': REQ_RADIUS_M if st['req'] is not None else REQ_RADIUS_NOREQ_M}
        if pose is not None:
            hx, hy = world_to_body(pose, st['x'], st['y'])
            out['home_body'] = [round(hx, 3), round(hy, 3)]
            out['robot_to_home_m'] = round(math.hypot(hx, hy), 3)
            cx, cy = (world_to_body(pose, st['req'][0], st['req'][1])
                      if st['req'] is not None else (hx, hy))
            out['center_body'] = [round(cx, 3), round(cy, 3)]
        return out

    def filter_persons(self, rows, pose, now):
        """(kept, ignored): person rows within the radius of the requester's
        spot (or of home) -- the only people a handover may target.  Rows of
        other labels pass through in ``kept``.  No anchor or no pose: every
        row is kept (today's behaviour)."""
        snap = self.snapshot(pose, now) if pose is not None else None
        if snap is None or 'center_body' not in snap:
            return list(rows or ()), []
        cx, cy = snap['center_body']
        rad = snap['radius_m']
        kept, ignored = [], []
        for r in rows or ():
            if isinstance(r, dict) and str(r.get('label', '')).lower() == 'person':
                x, y = _f(r, 'x'), _f(r, 'y')
                if (x is None or y is None or math.hypot(x - cx, y - cy) > rad
                        or math.hypot(x, y) < REQ_MIN_RANGE_M):
                    ignored.append(r)
                    continue
            kept.append(r)
        return kept, ignored


# -- the handover creep-in (operator 10-17: "Centre + creep in") --
# Stuart 10-17 18:34, back at the start: the person stood 0.68 m out and
# off-centre; 'place into person' refused (reach residual 0.35 m) and
# navigate('person') counted REACHED right away (walk gate 0.55 m), so
# nothing closed the gap.  This turns until they're centred (the
# turn_around law), then short forward pulses until in reach.
HANDOVER_ENV = 'OPENBOT_HANDOVER_APPROACH'   # '0' = refused
#: "in reach": the person row is at most this far out.  Reach law (brain
#: tools.reach_residual, shoulder (0.10, 0.004, 0.248), arm 0.386 m):
#: centred at 0.46 m leaves residual 0.086 <= PLACE_ON_MAX_RESIDUAL 0.10.
HANDOVER_ARRIVE_M = 0.46
#: pulse only when its worst case (HANDOVER_P_HI_M) still leaves them
#: out here: above REQ_MIN_RANGE_M (0.35, own-arm rows).
HANDOVER_MIN_M = 0.38
HANDOVER_START_MAX_M = 0.80  # farther = navigate; 0.46 + 6 x 0.05 + margin
HANDOVER_VX = 0.20           # = CREEP_VX (gait floor)
HANDOVER_PULSE_S = 0.5       # = CREEP_PULSE_S
HANDOVER_SETTLE_S = 0.6      # = CREEP_SETTLE_S
HANDOVER_P_NOM_M = 0.05      # = CREEP_P_NOM_M
HANDOVER_P_HI_M = 0.08       # = CREEP_P_HI_M
HANDOVER_MAX_PULSES = 6      # <= ~0.30 m nominal
HANDOVER_TOL_RAD = 0.09      # "centred": ~5 deg
HANDOVER_TURN_RAD = 0.15     # else turn first
HANDOVER_K_THETA = 1.5       # per-pulse correction
HANDOVER_OMEGA_CORR_MAX = 0.15
HANDOVER_OMEGA_SIGN = -1.0   # +x = RIGHT, +omega = LEFT
HANDOVER_ASSOC_M = 0.35      # near the last row
HANDOVER_SEED_M = 0.60       # near caller's row
HANDOVER_LOST_S = 2.0
HANDOVER_BUDGET_S = 15.0
HANDOVER_FINAL_SETTLE_S = 1.0
HANDOVER_OBST_MIN_CONF = 0.35  # = CREEP_OBST_MIN_CONF
HANDOVER_OBST_CORRIDOR_M = 0.30
HANDOVER_OBST_Z_MAX = 0.05
HANDOVER_SELF_ZONE_M = 0.30
HANDOVER_RESULTS_OK = ('arrived', 'close_lost')
# first lock at 0.462 m, one pulse, then 'lost': the camera drops a person
# this close.  Lost AFTER a forward pulse planned off a centred sighting,
# dead-reckoned range at most this, counts as reached ('close_lost'):
# open the hand right there.
HANDOVER_CLOSE_LOST_M = 0.55


def handover_on(environ):
    return str(environ.get(HANDOVER_ENV, '1')).strip() != '0'


def handover_track(rows, last=None, seed=None):
    """The person to close in on: a plausible person row (REQ_MIN_RANGE_M or
    farther) nearest the last tracked row (within HANDOVER_ASSOC_M), else on
    the first lock nearest the caller's seed (within HANDOVER_SEED_M), else
    the nearest plausible person.  None when nobody qualifies."""
    cands = []
    for r in rows or ():
        if not is_plausible_person(r):
            continue
        x, y = _f(r, 'x'), _f(r, 'y')
        if math.hypot(x, y) < REQ_MIN_RANGE_M:
            continue
        cands.append((x, y, r))
    if not cands:
        return None
    if last is not None:
        lx, ly = last
        x, y, r = min(cands, key=lambda c: math.hypot(c[0] - lx, c[1] - ly))
        return r if math.hypot(x - lx, y - ly) <= HANDOVER_ASSOC_M else None
    if seed is not None:
        sx, sy = seed
        x, y, r = min(cands, key=lambda c: math.hypot(c[0] - sx, c[1] - sy))
        return r if math.hypot(x - sx, y - sy) <= HANDOVER_SEED_M else None
    return min(cands, key=lambda c: math.hypot(c[0], c[1]))[2]


def handover_step(rng, err, centred, pulses, p_hi=None):
    """'arrived' | 'turn' | 'forward' | 'max_travel' for a person at range
    ``rng`` (m) and bearing ``err`` (rad, + = right).  ``p_hi`` = the worst
    case one pulse may close (B8l-hf6: handover_p_hi of the pulses seen so
    far; never below HANDOVER_P_HI_M)."""
    hi = HANDOVER_P_HI_M if p_hi is None else max(HANDOVER_P_HI_M, p_hi)
    if rng <= HANDOVER_ARRIVE_M:
        return 'arrived' if abs(err) <= HANDOVER_TOL_RAD else 'turn'
    if abs(err) > (HANDOVER_TURN_RAD if centred else HANDOVER_TOL_RAD):
        return 'turn'
    if pulses >= HANDOVER_MAX_PULSES:
        return 'max_travel'
    if rng - hi < HANDOVER_MIN_M:
        return 'arrived'            # worst-case closest
    return 'forward'


def handover_corr_omega(err):
    """Heading correction during a forward pulse (0 inside the tolerance)."""
    if abs(err) <= HANDOVER_TOL_RAD:
        return 0.0
    w = HANDOVER_OMEGA_SIGN * HANDOVER_K_THETA * err
    return max(-HANDOVER_OMEGA_CORR_MAX, min(HANDOVER_OMEGA_CORR_MAX, w))
