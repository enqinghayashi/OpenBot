"""strict_vocab.py -- B8i strict detection: the gateway's open HOLDS.

stdlib only, py3.8, rclpy-free (unit-testable).  A HOLD narrows the yolo
node's active vocabulary to a few labels for ONE bounded phase:

  navigate               opened by GatewayNode._do_navigate; closed when the
                         verb has returned AND the walk is no longer
                         SEARCHING / DETECTED / APPROACHING, on cancel/stop,
                         or at NAV_HOLD_MAX_S;
  look_for, place, pick  opened by the brain (verb yolo_strict op 'begin');
                         closed by op 'end', on cancel/stop, or at
                         BRAIN_HOLD_MAX_S (a brain that died never ends it).

The node runs the UNION of every open hold's labels plus KEEP_LABELS; with
no hold open it runs the operator's full vocabulary, which only the node
holds and strict detection never edits.  This module decides WHAT the
gateway publishes on /yolo/strict; the node owns the swap and the lease
(LEASE_S, re-sent every RENEW_S: a dead gateway restores the node by itself).
"""
import itertools
import math
import os
import threading
import time

KEEP_LABELS = ('person',)     # operator 2026-10-09: "keep person"
LEASE_S = 4.0                 # ~3 misses = drop
RENEW_S = 1.0                 # gateway renew rate
ACK_S = 1.0                   # node echo wait
NAV_HOLD_MAX_S = 135.0        # NAV_WAIT_S 115 + coordinator 120 s + margin
BRAIN_HOLD_MAX_S = 15.0       # one target resolve
MAX_LABELS = 8                # = vocab_state.STRICT_MAX_LABELS
BRAIN_OWNERS = ('look_for', 'place', 'pick')
#: keys off the gateway's OWN stamps, not grip refresh.  begin: a packet
#: landed SETTLE_S after the swap (re-mint needs CONFIRM_SECONDS 1.0 s +
#: one 0.1 s tick + projector p99 0.027 s + p95 gap 0.25 s).  end: a
#: packet landed REFILL_S after the restore.
SETTLE_S = 1.5
SETTLE_MAX_S = 3.0
REFILL_S = 1.5
REFILL_MAX_S = 2.5
POLL_S = 0.05


class StrictHolds(object):
    """The open holds.  Thread-safe; ``clock`` is injectable (monotonic)."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._holds = {}          # hid -> hold dict

    def open(self, owner, labels, max_s):
        """Open a hold for ``labels`` (order kept, duplicates dropped);
        returns its id (int >= 1)."""
        labs = []
        for lab in labels:
            if isinstance(lab, str) and lab and lab not in labs:
                labs.append(lab)
        with self._lock:
            hid = next(self._ids)
            self._holds[hid] = {'owner': str(owner), 'labels': tuple(labs),
                                'deadline': self._clock() + float(max_s),
                                'done': False}
        return hid

    def close(self, hid):
        """Close one hold; False when it was not open."""
        with self._lock:
            return self._holds.pop(hid, None) is not None

    def close_all(self):
        """Close every hold; returns how many were open."""
        with self._lock:
            n = len(self._holds)
            self._holds.clear()
        return n

    def mark_done(self, hid):
        """navigate: the verb returned; the hold now ends with the walk."""
        with self._lock:
            h = self._holds.get(hid)
            if h is None:
                return False
            h['done'] = True
            return True

    def expire(self, walk_active):
        """Close the holds past their cap, and the done navigate holds once
        the walk is inactive.  Returns [(hid, owner, why)]."""
        now = self._clock()
        out = []
        with self._lock:
            for hid in sorted(self._holds):
                h = self._holds[hid]
                if now >= h['deadline']:
                    out.append((hid, h['owner'], 'cap'))
                elif h['done'] and not walk_active:
                    out.append((hid, h['owner'], 'walk_inactive'))
            for hid, _o, _w in out:
                del self._holds[hid]
        return out

    def labels(self):
        """The union of the open holds' labels (oldest hold first) plus
        KEEP_LABELS, at most MAX_LABELS; () when no hold is open."""
        with self._lock:
            out = []
            for hid in sorted(self._holds):
                for lab in self._holds[hid]['labels']:
                    if lab not in out:
                        out.append(lab)
        if not out:
            return ()
        keep = [k for k in KEEP_LABELS if k not in out]
        out = out[:MAX_LABELS - len(keep)] + keep
        return tuple(out)

    def __len__(self):
        with self._lock:
            return len(self._holds)


# ════════════════════════════════════════════════════════════════════════════
# The CONTAINER LOCK plus the CREEP plan (pure, stdlib, py3.8).
#
# MEASURED (gateway.log:338-350, yolo.log:14710-14773): navigate('bin')
# hit REACHED at 12:35:44.66 on track iid 158; the full vocabulary came
# back (RESTORED 44.80) and the SAME track got re-labelled 'backpack'
# (conf .73) at 46.05; the place hold {bin, person} at 46.39 then dropped
# it (dropped=8: tracks drop when their label leaves the active set),
# nothing scored >= .35 (the new-track floor), so it sat silent 3.1 s and
# the place refused NOT_DETECTED -- while the tub never moved (x .026
# y .578 z -.427, depth.log:22434).  So: keep the NAVIGATED container by
# identity.  The walk's REACHED arms it on the row the walk stopped at;
# packets follow that row by instance id ACROSS label changes (the
# tracker associates class-agnostically), or re-acquire by geometry once
# the id dies; only the creep may move it.
# ════════════════════════════════════════════════════════════════════════════
LOCK_ENV = 'OPENBOT_CONTAINER_LOCK'   # '0' = no lock
CREEP_ENV = 'OPENBOT_CREEP'           # '0' = no creep
CREEP_EDGE_ENV = 'OPENBOT_CREEP_EDGE_M'
#: what we place INTO.  MIRROR of brain/agent/tools.py CONTAINER_CLASSES,
#: kept in sync by tests/integration/gateway/test_gateway_container_lock.py
CONTAINER_CLASSES = frozenset((
    'bowl', 'bin', 'box', 'basket', 'bucket', 'tray', 'pot', 'pan',
    'lunchbox', 'toolbox', 'backpack', 'handbag', 'suitcase'))
LOCK_MAX_AGE_S = 300.0       # 120 s ran out mid-talk; the scene check is the row itself
LOCK_FOLLOW_JUMP_M = 0.15    # planar jump = other object
LOCK_FOLLOW_DZ_M = 0.20      # ... or z
LOCK_REACQ_M = 0.10          # re-acquire once the id dies
LOCK_REACQ_DZ_M = 0.12       # ... z too
LOCK_ARM_EV_M = 0.08         # no-label-row gate
LOCK_HIST_N = 20             # ~2 s at 10 Hz
LOCK_HALF_MIN_M = 0.06
LOCK_HALF_MAX_M = 0.35
LOCK_HALF_DEFAULT_M = 0.10   # no extents
LOCK_CUT_FRAC = 0.6          # under this = cut
#: bottom image edge (MEASURED mount, W/lab/RECON_FULL_20261008.md:541): a
#: point at torso height z only shows up at planar distance >= vis_edge(z).
VIS_A, VIS_B, VIS_C = 0.3737, 0.46268, 0.0476
VIS_MARGIN_M = 0.02          # a near rim this far above the cut still counts
# ── creep (the gateway 'creep' verb; H constants unless tagged) ──
CREEP_VX = 0.20              # m/s: walk_node VX_MIN 0.20 (gait floor), G17a 0.22
CREEP_PULSE_S = 0.5          # writer expires it
CREEP_SETTLE_S = 0.6         # gait settles + fresh packet (p95 0.25 s)
CREEP_P_NOM_M = 0.05         # sim 0.047-0.053 m @ 0.20 x 0.5 s
CREEP_P_HI_M = 0.08          # coast included; planning uses this
CREEP_P_LO_M = 0.02          # under this = no progress
CREEP_EDGE_DEFAULT_M = 0.25  # operator 2026-10-10: "creep ... to near edge 0.25"
CREEP_EDGE_LO_M = 0.20
CREEP_EDGE_HI_M = 0.40
CREEP_EDGE_MIN_M = 0.20      # hard worst case: toe +0.186 m + 1.4 cm
CREEP_MAX_PULSES = 5         # per creep, even after a re-plan
CREEP_START_MAX_EDGE_M = 0.60  # farther = navigate first
CREEP_FLOOR_BOT_Z = -0.55    # below = on floor
CREEP_TABLE_MAX_M = 0.15     # container on a support: max travel, its
                             # front edge isn't observed (H)
CREEP_BUDGET_S = 9.0         # loop + FINAL_SETTLE + place <= 60 s bound
CREEP_FINAL_SETTLE_S = 1.0   # after 'cmd_set off', before hold_rc (zeroes walking)
CREEP_TICK_S = 0.1
CREEP_OMEGA_MAX = 0.15       # rad/s clamp
CREEP_TOL_RAD = 0.09         # = ALIGN_TOL_RAD
CREEP_K_THETA = 1.5          # = K_THETA
CREEP_OMEGA_SIGN = -1.0      # +x = right
CREEP_PERSON_STOP_M = 0.70   # person near = stop
CREEP_PERSON_MIN_CONF = 0.50 # = TURN_PERSON_MIN_CONF
CREEP_PERSON_MIN_H_M = 0.30  # = TURN_PERSON_MIN_HEIGHT_M
CREEP_OBST_MIN_CONF = 0.35
CREEP_OBST_CORRIDOR_M = 0.30 # inside this |x|
CREEP_OBST_Z_MAX = 0.05      # hands / held
CREEP_SELF_ZONE_M = 0.30     # own arm rows
CREEP_BUMP_M = 0.10          # moved away = bump
CREEP_RESULTS_OK = ('arrived', 'planned', 'in_place')


def env_on(name, environ=None):
    """Unset / anything but '0' = ON (B8j switches default ON)."""
    env = os.environ if environ is None else environ
    return str(env.get(name, '1')).strip() != '0'


def creep_edge_target(environ=None):
    """OPENBOT_CREEP_EDGE_M (gateway terminal), clamped; default 0.25."""
    env = os.environ if environ is None else environ
    raw = env.get(CREEP_EDGE_ENV)
    try:
        v = float(raw) if raw is not None else CREEP_EDGE_DEFAULT_M
    except (TypeError, ValueError):
        v = CREEP_EDGE_DEFAULT_M
    if not math.isfinite(v):
        v = CREEP_EDGE_DEFAULT_M
    return max(CREEP_EDGE_LO_M, min(CREEP_EDGE_HI_M, v))


def creep_edge_floor(edge_t):
    """The WORST-case floor of a creep toward ``edge_t`` (B8j repair 2, gate 2
    finding 1): CREEP_EDGE_MIN_M at the default target and below, raised 1:1
    with a target above it, so OPENBOT_CREEP_EDGE_M also moves the bound that
    binds.  With the floor fixed at 0.20 m a larger target removed no pulse
    from a walk end of 0.44 m or more (0.30 planned 3 pulses from 0.45, the
    default plan); 0.35 now removes at least one pulse from every start of
    0.30-0.60 m and keeps even 0.08 m per pulse at 0.30 m or more."""
    try:
        t = float(edge_t)
    except (TypeError, ValueError):
        return CREEP_EDGE_MIN_M
    if not math.isfinite(t):
        return CREEP_EDGE_MIN_M
    return round(CREEP_EDGE_MIN_M + max(0.0, t - CREEP_EDGE_DEFAULT_M), 6)


def vis_edge(z):
    """Planar distance (m) of the bottom image edge at torso height ``z``."""
    return VIS_A * (VIS_B - float(z)) + VIS_C


def creep_plan(e0, edge_t, p_nom=CREEP_P_NOM_M, p_hi=CREEP_P_HI_M,
               e_min=None, max_pulses=CREEP_MAX_PULSES):
    """Pulses for a creep from near edge ``e0`` toward ``edge_t``: the most
    pulses whose NOMINAL end stays at or beyond ``edge_t`` and whose WORST
    case (p_hi per pulse) stays at or beyond ``e_min`` (None = that target's
    floor, creep_edge_floor: 0.20 m at the default 0.25).  0 = no motion."""
    try:
        e0 = float(e0)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(e0) or e0 <= edge_t:
        return 0
    if e_min is None:
        e_min = creep_edge_floor(edge_t)
    n_nom = int(math.floor((e0 - edge_t) / p_nom + 1e-9))
    n_bound = int(math.floor((e0 - e_min) / p_hi + 1e-9))
    return max(0, min(n_nom, n_bound, int(max_pulses)))


def creep_table_pulses(p_nom=CREEP_P_NOM_M, cap=CREEP_TABLE_MAX_M):
    """Pulses a container ON a support may take at nominal travel: the real
    creep stops before a pulse that would pass ``cap`` (+5 mm), so the dry
    plan ends where the real creep ends (B8j repair, gate 1)."""
    return max(0, int(math.floor((cap + 0.005) / p_nom + 1e-9)))


def _f(row, key):
    try:
        v = float(row.get(key))
    except (TypeError, ValueError, AttributeError):
        return None
    return v if math.isfinite(v) else None


def _iid(row):
    v = row.get('instance_id') if isinstance(row, dict) else None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        return None
    return v


def _row_label(row):
    lab = row.get('label') if isinstance(row, dict) else None
    return lab.strip().lower() if isinstance(lab, str) else ''


def _xyz(row):
    x, y, z = _f(row, 'x'), _f(row, 'y'), _f(row, 'z')
    if x is None or y is None or z is None:
        return None
    return x, y, z


def _planar(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _row_geo(row, z):
    """(half, top, bottom, h) of one row; half None without extents."""
    w, dd, h = _f(row, 'width_m'), _f(row, 'depth_m'), _f(row, 'height_m')
    ext = [v for v in (w, dd) if v is not None and v > 0.0]
    half = max(ext) / 2.0 if ext else None
    if h is not None and h > 0.0:
        return half, z + h / 2.0, z - h / 2.0, h
    return half, z, z, 0.0
