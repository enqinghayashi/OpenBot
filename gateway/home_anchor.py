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

