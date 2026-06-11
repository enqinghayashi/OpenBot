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
