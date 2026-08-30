"""pc_core: the pure-numpy geometry core of the point-cloud consumer.

stdlib + numpy ONLY (no rclpy, scipy, open3d, sklearn, grasp_geometry).
Python 3.8-safe.  Design: docs/POINTCLOUD_CONSUMER_DESIGN.md (batch 3, rev 2).

Five consumers, each returning a result dict
    {"status": OK|UNKNOWN|ERROR, "verdict": <member of VERDICTS>,
     "reason": <machine-readable, non-empty on UNKNOWN>, "evidence": {...}}
and wrapped by ``envelope()`` for the wire.

  P1 hold_baseline / hold_check : NEGATIVE-ONLY.  The consumer never asserts
     the object is held.  Decisive verdicts are STILL_ON_TABLE and MOVED, both
     negative.  ABSENT_UNPROVEN is NON-decisive ("no new evidence"); a caller
     must never read it as held.  P1 does NOT close run19 (bottle occluded ->
     no row -> no signal); that path is closed by the batch-2 CANCELLED fix.
  P2 fit_bowl      : rim height by a z-sweep circle fit (not z + h/2).
  P3 split_support : support split (support row, or a width LEDGE).
  P4 grasp_diag    : front surface / radius / corridor, DIAG only.
  P5 person_track  : robust lateral midpoint + range, SHADOW only.

Every geometric quantity is in the projector's torso frame:
  +x = RIGHT (projector :397 "'x': ... # lateral  (+x = right)"),
  +y = forward, +z = up.  The torso frame is NOT world-fixed; every task that
  compares or consumes geometry across time requires a MEASURED pose token.
"""
import json
import math

import numpy as np

try:  # package import
    from . import pc_config as _cfgmod
except ImportError:  # flat import
    # box layout / tests
    import pc_config as _cfgmod

PcConfig = _cfgmod.PcConfig
DEFAULT_CFG = _cfgmod.DEFAULT
cfg_hash = _cfgmod.cfg_hash

# -------------------------------------------------------------- the projector
# literals from
# skills/perception/depth_projector_node.py
# (rclpy/scipy, can't import)
# pinned by tests/unit/perception/
# test_pointcloud_synth.py (re-reads source)
FX, FY, CX, CY = 302.7, 302.8, 212.9, 121.1          # from projector :21
DEPTH_MIN = 0.3                                      # from projector :22
DEPTH_MAX = 6.0                                      # from projector :23
COLOR_W, COLOR_H = 424, 240                          # from projector :29
MAX_PTS_PER_OBJ = 2000                               # from projector :32
MAX_CLOUD_POINTS = 20000                             # from projector :37
TILT_DEG = 48.0                                      # from projector :54
_t = np.radians(TILT_DEG)
_R_base = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
_Rx_tilt = np.array([[1, 0, 0], [0, np.cos(_t), np.sin(_t)], [0, -np.sin(_t), np.cos(_t)]])
R_CAM_TO_TORSO = _R_base @ _Rx_tilt                  # from launch_pick_stack.sh:84
T_CAM_IN_TORSO = np.array([0.0, 0.04764, 0.46268])  # see yolo_node_seg.py:236

# sensor_msgs/PointField types
PF_UINT32 = 6
PF_FLOAT32 = 7
EXPECTED_FIELDS = (("x", 0, PF_FLOAT32, 1), ("y", 4, PF_FLOAT32, 1),
                   ("z", 8, PF_FLOAT32, 1), ("object_id", 12, PF_UINT32, 1))
CLOUD_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("oid", "<u4")])

# -------------------------------------------------------------- the verdicts
OK = "OK"
UNKNOWN = "UNKNOWN"
ERROR = "ERROR"
STATUSES = frozenset({OK, UNKNOWN, ERROR})

V_UNKNOWN = "UNKNOWN"
V_ERROR = "ERROR"
V_BASELINE_OK = "BASELINE_OK"
V_STILL = "STILL_ON_TABLE"
V_MOVED = "MOVED"
V_ABSENT = "ABSENT_UNPROVEN"
V_RIM = "RIM_FIT"
V_SPLIT = "SPLIT"
V_NOSPLIT = "NO_SPLIT_EVIDENCE"
V_GRASP = "GRASP_DIAG"
V_PERSON = "PERSON_TRACK"
VERDICTS = frozenset({V_UNKNOWN, V_ERROR, V_BASELINE_OK, V_STILL, V_MOVED, V_ABSENT,
                      V_RIM, V_SPLIT, V_NOSPLIT, V_GRASP, V_PERSON})
# P1: negative-only
# no HOLD verdict claims held
HOLD_VERDICTS = frozenset({V_STILL, V_MOVED, V_ABSENT, V_UNKNOWN})
HOLD_DECISIVE_NEGATIVE = frozenset({V_STILL, V_MOVED})
HOLD_NON_DECISIVE = frozenset({V_ABSENT, V_UNKNOWN})

PHASES = ("tucked", "cancelled", "error", "other")
TASKS = ("hold_baseline", "hold_check", "place_rim", "split", "grasp", "person")

POSE_KEYS = ("torso_z", "pitch", "roll", "waist_pitch")

# keep states apart
NO_PACKET = "NO_PACKET"
NO_ROW = "NO_ROW"
NO_POINTS = "NO_POINTS"
TRUNCATED = "TRUNCATED"
SEL_OK = "OK"


class CloudFormatError(ValueError):
    pass


def result(status, verdict, reason="", **evidence):
    if status == UNKNOWN and not reason:
        raise AssertionError("UNKNOWN must carry a reason")
    return {"status": status, "verdict": verdict, "reason": reason, "evidence": evidence}


def unknown(reason, **ev):
    return result(UNKNOWN, V_UNKNOWN, reason, **ev)


# ---------------------------------------------------------------- decoding
def _field_tuple(f):
    if isinstance(f, (tuple, list)):
        t = list(f) + [1]
        return (str(t[0]), int(t[1]), int(t[2]), int(t[3]))
    return (str(f.name), int(f.offset), int(f.datatype), int(getattr(f, "count", 1)))


def decode_cloud(data, width, point_step, fields, is_bigendian=False, height=1):
    """PointCloud2 payload -> (pts float32 (N,3), oid uint32 (N,), n_nonfinite).

    REFUSES (CloudFormatError) any layout other than the projector's:
    point_step 16, fields x/y/z FLOAT32 @0/4/8 + object_id UINT32 @12, LE."""
    if is_bigendian:
        raise CloudFormatError("big-endian cloud")
    if int(point_step) != 16:
        raise CloudFormatError("point_step %r != 16" % (point_step,))
    ft = tuple(_field_tuple(f) for f in fields)
    if ft != EXPECTED_FIELDS:
        raise CloudFormatError("fields %r != %r" % (ft, EXPECTED_FIELDS))
    n = int(width) * int(height)
    mv = memoryview(data)
    if mv.nbytes != n * 16:
        raise CloudFormatError("len(data)=%d != %d*16" % (mv.nbytes, n))
    rec = np.frombuffer(mv, dtype=CLOUD_DTYPE, count=n)
    pts = np.empty((n, 3), dtype=np.float32)
    pts[:, 0], pts[:, 1], pts[:, 2] = rec["x"], rec["y"], rec["z"]
    oid = rec["oid"].astype(np.uint32)
    fin = np.isfinite(pts).all(axis=1)
    return pts[fin], oid[fin], int(n - int(fin.sum()))


def parse_body(text):
    rows = json.loads(text)
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError("body is not a list of dicts")
    return rows


# ---------------------------------------------------------- label matching
def base_noun(label):
    w = str(label or "").lower().split()
    return w[-1] if w else ""


def label_matches(row_label, label):
    q = base_noun(label)
    return bool(q) and q in str(row_label or "").lower().split()


# ---------------------------------------------------------- frame handling
class Frame(object):
    """One PAIRED cloud: rows (body JSON list) + per-oid torso points."""
    __slots__ = ("t_wall", "t_mono", "rows", "segs", "truncated", "pair_mode", "stale")

    def __init__(self, t_wall, t_mono, rows, segs, truncated=(), pair_mode="heuristic",
                 stale=()):
        self.t_wall = t_wall
        self.t_mono = t_mono
        self.rows = rows
        self.segs = segs
        self.truncated = frozenset(truncated)
        self.pair_mode = pair_mode
        self.stale = frozenset(stale)


def split_segments(pts, oid):
    segs = {}
    if len(pts) == 0:
        return segs
    order = np.argsort(oid, kind="stable")
    o = oid[order]
    P = np.asarray(pts, dtype=np.float64)[order]
    u, start = np.unique(o, return_index=True)
    ends = list(start[1:]) + [len(o)]
    for k, s, e in zip(u, start, ends):
        segs[int(k)] = P[s:e]
    return segs


def _row_xyz(row):
    try:
        v = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
    except (KeyError, TypeError, ValueError):
        return None
    return v if np.all(np.isfinite(v)) else None


def _row_inside(row, seg, pad):
    v = _row_xyz(row)
    if v is None or len(seg) == 0:
        return False
    lo, hi = np.percentile(seg, [5, 95], axis=0)
    return bool(np.all(v >= lo - pad) and np.all(v <= hi + pad))


def check_body(rows, segs, trunc_oid, cfg):
    """Can this body packet be the one the cloud was built with?"""
    if not segs:
        return False, "empty_cloud"
    if max(segs) >= len(rows):
        return False, "oid_out_of_range"
    seg_rows = set(i for i, r in enumerate(rows) if "width_m" in r)
    present = set(segs)
    if present - seg_rows:
        return False, "points_on_fallback_row"
    missing = seg_rows - present
    if missing:
        if trunc_oid is None or not all(m > trunc_oid for m in missing):
            return False, "seg_row_without_points"
    for o, s in segs.items():
        if o == trunc_oid:
            continue
        if not _row_inside(rows[o], s, cfg.PAIR_BOX_PAD_M):
            return False, "row_xyz_outside_segment"
    return True, ""


def _bodies_equivalent(a, b, segs, cfg):
    if len(a) != len(b):
        return False
    for o in segs:
        if str(a[o].get("label")) != str(b[o].get("label")):
            return False
        va, vb = _row_xyz(a[o]), _row_xyz(b[o])
        if va is None or vb is None or np.max(np.abs(va - vb)) > cfg.PAIR_BOX_PAD_M:
            return False
    return True


def pair_cloud(bodies, pts, oid, t_cloud_rx, cfg=None, t_wall=None, n_raw=None,
               hdr_stamp_ns=None):
    """Pair a decoded cloud with one of the latest two body packets.

    bodies: [(t_rx_mono, rows), ...] NEWEST FIRST.
    Exact mode (after hook H-PROJ-1): at least one row carries 'cloud_stamp_ns' equal
    to the cloud header stamp; every row indexed by a segment carries 'cloud_stamp_ns'.
    Fallback rows (no 'width_m', not indexed by any segment) do not block exact mode.
    Heuristic mode (today): structural + xyz cross-check; if BOTH bodies pass
    and they are not equivalent for every segment -> (None, 'pair_ambiguous').
    Returns (Frame|None, reason)."""
    cfg = cfg or DEFAULT_CFG
    segs = split_segments(pts, oid)
    n_raw = len(pts) if n_raw is None else n_raw
    trunc_oid = max(segs) if (segs and n_raw >= cfg.MAX_CLOUD_POINTS) else None
    cands = []
    exact_seen = False
    last_why = "no_body"
    for (t_rx, rows) in list(bodies)[:2]:
        gap = t_cloud_rx - t_rx
        if gap < 0 or gap > cfg.PAIR_MAX_GAP_S:
            last_why = "body_gap"
            continue
        # exact mode
        # row stamps match cloud
        has_any_stamp = rows and any("cloud_stamp_ns" in r for r in rows)
        indexed_oids = set(segs.keys())
        all_indexed_stamped = (indexed_oids and
                               all("cloud_stamp_ns" in rows[o] for o in indexed_oids if o < len(rows)))
        if has_any_stamp and hdr_stamp_ns is not None and all_indexed_stamped:
            exact_seen = True
            # stamps must agree
            if any("cloud_stamp_ns" in r and int(r["cloud_stamp_ns"]) != int(hdr_stamp_ns) for r in rows):
                last_why = "stamp_mismatch"
                continue
            bad = False
            for o, s in segs.items():
                if o >= len(rows) or ("n_pts" in rows[o] and o != trunc_oid
                                      and int(rows[o]["n_pts"]) != len(s)):
                    bad = True
            if bad:
                last_why = "n_pts_mismatch"
                continue
        ok, why = check_body(rows, segs, trunc_oid, cfg)
        if ok:
            cands.append(rows)
        else:
            last_why = why
    if not cands:
        return None, last_why
    if exact_seen:
        mode = "exact"
    elif len(cands) == 1:
        mode = "heuristic"
    elif _bodies_equivalent(cands[0], cands[1], segs, cfg):
        mode = "heuristic_equiv"
    else:
        return None, "pair_ambiguous"
    rows = cands[0]
    stale = set()
    if t_wall is not None:
        for i, r in enumerate(rows):
            st = r.get("stamp")
            try:
                age = t_wall - float(st)
            except (TypeError, ValueError):
                stale.add(i)
                continue
            if age > cfg.ROW_MAX_AGE_S or age < -cfg.ROW_MAX_AGE_S:
                stale.add(i)
    return Frame(t_wall, t_cloud_rx, rows, segs,
                 truncated=() if trunc_oid is None else (trunc_oid,),
                 pair_mode=mode, stale=stale), ""


def select_target(frame, label, ref_xy, cfg, need_ref=False):
    """(state, idx, row, seg).  state in NO_ROW / NO_POINTS / TRUNCATED / OK."""
    if frame is None:
        return NO_PACKET, None, None, None
    best = None
    for i, r in enumerate(frame.rows):
        if i in frame.stale or not label_matches(r.get("label"), label):
            continue
        v = _row_xyz(r)
        if v is None:
            continue
        if ref_xy is not None:
            d = math.hypot(v[0] - ref_xy[0], v[1] - ref_xy[1])
            if d > cfg.ASSOC_GATE_M:
                continue
            key = d
        elif need_ref:
            continue
        else:
            try:
                key = -float(r.get("conf") or 0.0)
            except (TypeError, ValueError):
                key = 0.0
        if best is None or key < best[0]:
            best = (key, i, r)
    if best is None:
        return NO_ROW, None, None, None
    _, i, r = best
    if i in frame.truncated:
        return TRUNCATED, i, r, frame.segs.get(i)
    seg = frame.segs.get(i)
    if seg is None or len(seg) == 0:
        return NO_POINTS, i, r, None
    return SEL_OK, i, r, seg


# ---------------------------------------------------------- geometry helpers
def torso_to_pixel(P):
    """Torso points -> (u, v, z_cam) in COLOUR pixels with the projector intrinsics."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    pc = (P - T_CAM_IN_TORSO) @ R_CAM_TO_TORSO   # R^T (p - T)
    # R orthonormal
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * pc[:, 0] / z + CX
        v = FY * pc[:, 1] / z + CY
    return u, v, z


def in_view(P):
    u, v, z = torso_to_pixel(P)
    return (z > DEPTH_MIN) & (z < DEPTH_MAX) & (u >= 0) & (u < COLOR_W) & (v >= 0) & (v < COLOR_H)


def width_est(x, cfg):
    """Full width of a silhouette from a percentile span, de-biased for a
    uniform-in-x point set (p5-p95 spans 0.90 of the width: DERIVED)."""
    lo, hi = np.percentile(x, [cfg.WIDTH_Q_LO, cfg.WIDTH_Q_HI])
    return float(hi - lo) / ((cfg.WIDTH_Q_HI - cfg.WIDTH_Q_LO) / 100.0)


def descent_band(z, cfg):
    """Copy of brain/descent_law.classify_pick_z with the box floor (-0.54)."""
    if z is None or not np.isfinite(z):
        return "ABSENT"
    if z >= cfg.COMFORT_Z:
        return "NORMAL"
    if z >= cfg.DEEP_FLOOR_Z:
        return "DESCEND"
    return "DEEP_FLOOR"


# ----------------------------------------------------------- pose checks
def pose_check(pose, cfg):
    if not isinstance(pose, dict):
        return False, "no_pose"
    if cfg.POSE_REQUIRE_MEASURED and pose.get("src") != "measured":
        return False, "pose_unmeasured"
    for k in POSE_KEYS:
        try:
            if not math.isfinite(float(pose[k])):
                return False, "pose_incomplete"
        except (KeyError, TypeError, ValueError):
            return False, "pose_incomplete"
    return True, ""


def pose_equal(a, b, cfg):
    """Same torso frame?  Both poses must be valid MEASURED vectors."""
    for p in (a, b):
        ok, why = pose_check(p, cfg)
        if not ok:
            return False, why
    if abs(float(a["torso_z"]) - float(b["torso_z"])) > cfg.POSE_Z_TOL_M:
        return False, "pose_changed"
    for k in ("pitch", "roll", "waist_pitch"):
        if abs(float(a[k]) - float(b[k])) > cfg.POSE_ANG_TOL_RAD:
            return False, "pose_changed"
    return True, ""


# ================================================================ P1 - hold
def _ext(seg):
    lo, hi = np.percentile(seg, [5, 95], axis=0)
    return hi - lo


def hold_baseline(frames, label, ref_xy, pose, cfg=None):
    """Baseline B0 of the pick target, taken before the arm enters the view."""
    cfg = cfg or DEFAULT_CFG
    if ref_xy is None:
        return unknown("no_ref")
    ok, why = pose_check(pose, cfg)
    if not ok:
        return unknown(why)
    if not frames:
        return unknown("no_packet")
    segs, others, states = [], [], []
    for f in list(frames)[-cfg.HOLD_BASE_FRAMES:]:
        st, idx, row, seg = select_target(f, label, ref_xy, cfg, need_ref=True)
        states.append(st)
        if st != SEL_OK:
            continue
        segs.append(seg)
        for o, s in f.segs.items():
            if o == idx or o >= len(f.rows) or not label_matches(f.rows[o].get("label"), label):
                continue
            others.append(np.median(s[:, :2], axis=0).tolist())
    if len(segs) < cfg.HOLD_BASE_MIN_FRAMES:
        return unknown("baseline_short", n_ok=len(segs), states=states)
    counts = [len(s) for s in segs]
    N0 = float(np.median(counts))
    if N0 < cfg.HOLD_MIN_N0:
        return unknown("baseline_sparse", N0=N0)
    P = np.vstack(segs)
    c0 = np.median(P, axis=0)
    zb0, zt0 = [float(v) for v in np.percentile(P[:, 2], [5, 95])]
    h0 = zt0 - zb0
    if h0 < cfg.HOLD_MIN_H0_M or float(c0[2]) - zb0 < cfg.HOLD_MIN_MEDIAN_RISE * h0:
        # flat or swamped by its own base
        # (mask leak on a clear bottle)
        return unknown("baseline_flat", h0=h0, median_rise=float(c0[2]) - zb0)
    r0 = max(width_est(P[:, 0], cfg) / 2.0, cfg.HOLD_R_MIN_M)
    p5s = [float(np.percentile(s[:, 2], 5)) for s in segs]
    p95s = [float(np.percentile(s[:, 2], 95)) for s in segs]
    dz_tol = max(cfg.HOLD_BASE_DZ_M, 3.0 * float(np.std(p5s)))
    top_tol = max(cfg.HOLD_TOP_DZ_M, 3.0 * float(np.std(p95s)))
    others = [o for o in others if math.hypot(o[0] - c0[0], o[1] - c0[1]) > cfg.HOLD_OTHER_EXCL_M]
    base = {
        "label": label, "c0": [float(v) for v in c0], "zb0": zb0, "zt0": zt0, "h0": h0,
        "r0": r0, "N0": N0, "dz_tol": dz_tol, "top_tol": top_tol,
        "ext": [float(v) for v in _ext(P)], "others": others, "pose": dict(pose),
        "t_wall": frames[-1].t_wall, "ref": [float(ref_xy[0]), float(ref_xy[1])],
        "n_frames": len(segs),
    }
    return result(OK, V_BASELINE_OK, "", baseline=base)


def footprint_corners(base, cfg):
    c = base["c0"]
    R = base["r0"] + cfg.HOLD_FOOT_PAD_M
    zl = base["zb0"] - cfg.HOLD_FOOT_ZLO_M
    zh = base["zt0"] + cfg.HOLD_FOOT_ZHI_M
    return np.array([[c[0] + sx * R, c[1] + sy * R, z]
                     for sx in (-1, 1) for sy in (-1, 1) for z in (zl, zh)])


def _in_footprint(P, base, cfg):
    c = base["c0"]
    R = base["r0"] + cfg.HOLD_FOOT_PAD_M
    d = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
    return (d <= R) & (P[:, 2] >= base["zb0"] - cfg.HOLD_FOOT_ZLO_M) & \
        (P[:, 2] <= base["zt0"] + cfg.HOLD_FOOT_ZHI_M)


def _shape_match(seg, base, cfg):
    e = _ext(seg)
    b = np.asarray(base["ext"])
    return bool(np.all(np.abs(e - b) <= cfg.HOLD_SHAPE_TOL * np.maximum(b, 1e-3)))


def _classify_hold_frame(f, base, cfg):
    """-> (kind, info). kind in hit / ghost_flat / partial / empty / occluded.

    hit   : a label- (or shape-) matched segment with VERTICAL STRUCTURE
            (z-span >= HOLD_SPAN_FRAC*h0) whose centre is within HOLD_STILL_XY_M
            of c0, base within dz_tol of zb0, top within top_tol of zt0, and
            enough points inside the footprint.  A ghost-mask table patch is
            flat and can never be a hit.
    moved : (info) a same-label, shape-matched, structured cluster that sits on
            the table HOLD_MOVED_XY_M..HOLD_MOVED_MAX_XY_M from c0 and is not a
            baseline neighbour."""
    label = base["label"]
    c0 = base["c0"]
    in_pts, other_pts = [], []
    moved = None
    hit = False
    n_need = max(cfg.HOLD_MIN_ABS, cfg.HOLD_MIN_FRAC * base["N0"])
    for o, s in f.segs.items():
        if o >= len(f.rows) or o in f.stale:
            continue
        row = f.rows[o]
        same = label_matches(row.get("label"), label)
        shaped = _shape_match(s, base, cfg)
        if not (same or shaped):
            other_pts.append(s)
            continue
        m = _in_footprint(s, base, cfg)
        if m.any():
            in_pts.append(s[m])
        p5z, p95z = [float(v) for v in np.percentile(s[:, 2], [5, 95])]
        span = p95z - p5z
        if span < cfg.HOLD_SPAN_FRAC * base["h0"]:
            continue
        med = np.median(s[:, :2], axis=0)
        dxy = math.hypot(med[0] - c0[0], med[1] - c0[1])
        on_table = abs(p5z - base["zb0"]) <= base["dz_tol"]
        if (dxy <= cfg.HOLD_STILL_XY_M and on_table
                and abs(p95z - base["zt0"]) <= base["top_tol"] and int(m.sum()) >= n_need):
            hit = True
            continue
        excl = any(math.hypot(med[0] - q[0], med[1] - q[1]) <= cfg.HOLD_OTHER_EXCL_M
                   for q in base["others"])
        fell = p5z < base["zb0"] - cfg.HOLD_MOVED_DZ_M
        if (same and shaped and not excl and dxy <= cfg.HOLD_MOVED_MAX_XY_M
                and ((dxy > cfg.HOLD_MOVED_XY_M and on_table) or fell)):
            if moved is None or dxy < moved[0]:
                moved = (dxy, [float(med[0]), float(med[1])], p5z)
    info = {"moved": moved}
    if hit:
        return "hit", info
    # occluder in front
    if other_pts:
        fc = footprint_corners(base, cfg)
        u, v, zc = torso_to_pixel(fc)
        Q = np.vstack(other_pts)
        qu, qv, qz = torso_to_pixel(Q)
        occ = (qu >= u.min()) & (qu <= u.max()) & (qv >= v.min()) & (qv <= v.max()) & \
            (qz < zc.min())
        info["n_occl"] = int(occ.sum())
        if int(occ.sum()) >= cfg.HOLD_OCCL_MIN_PTS:
            return "occluded", info
    if in_pts:
        I = np.vstack(in_pts)
        n_in = len(I)
        p5z, p50z, p95z = [float(v) for v in np.percentile(I[:, 2], [5, 50, 95])]
        span = p95z - p5z
        info.update(n_in=n_in, p5z=p5z, p95z=p95z, span=span)
        if n_in >= cfg.HOLD_PARTIAL_MIN_PTS:
            flat = span < 0.25 * base["h0"] and abs(p50z - base["zb0"]) <= base["dz_tol"] + 0.01
            return ("ghost_flat" if flat else "partial"), info
    info.setdefault("n_in", 0)
    return "empty", info


def hold_check(baseline, frames, pose, phase, t_reply_wall, cfg=None):
    """NEGATIVE-ONLY hold check after the engine reply.

    phase: what the engine reply says the arm did - 'tucked' (DONE after the
    carry tuck), 'cancelled', 'error' or 'other'.  Only frames whose wall time
    is >= t_reply_wall + HOLD_CHK_DELAY_S are used (the 2 s stale-mask
    lifetime is ENFORCED here, not assumed)."""
    cfg = cfg or DEFAULT_CFG
    if not baseline:
        return unknown("no_baseline")
    if phase not in PHASES:
        return unknown("bad_phase")
    same, why = pose_equal(baseline.get("pose"), pose, cfg)
    if not same:
        return unknown(why)
    if t_reply_wall is None:
        return unknown("no_reply_time")
    if t_reply_wall - baseline["t_wall"] > cfg.HOLD_MAX_SPAN_S:
        return unknown("baseline_expired")
    frames = list(frames or [])
    if not frames:
        return unknown("no_packet")
    t_open = t_reply_wall + cfg.HOLD_CHK_DELAY_S
    use = [f for f in frames if f.t_wall is not None and f.t_wall >= t_open][:cfg.HOLD_CHK_FRAMES]
    n_early = len(frames) - len([f for f in frames if f.t_wall is not None and f.t_wall >= t_open])
    if len(use) < cfg.HOLD_CHK_MIN_FRAMES:
        return unknown("delay_not_elapsed" if not use and n_early else "few_frames",
                       n_used=len(use), n_early=n_early)
    if not bool(np.all(in_view(footprint_corners(baseline, cfg)))):
        return unknown("footprint_out_of_view")
    kinds, infos = [], []
    for f in use:
        k, inf = _classify_hold_frame(f, baseline, cfg)
        kinds.append(k)
        infos.append(inf)
    ev = {"kinds": kinds, "n_in": [i.get("n_in", 0) for i in infos], "n_early": n_early,
          "phase": phase}
    hits = kinds.count("hit")
    if hits >= cfg.HOLD_HIT_FRAMES:
        return result(OK, V_STILL, "", **ev)
    mv = [i["moved"] for i in infos if i.get("moved") is not None]
    if len(mv) >= cfg.HOLD_HIT_FRAMES and hits == 0:
        xy = np.array([m[1] for m in mv])
        med = np.median(xy, axis=0)
        if np.all(np.hypot(xy[:, 0] - med[0], xy[:, 1] - med[1]) <= cfg.HOLD_MOVED_CONSIST_M):
            ev["moved_to"] = [float(med[0]), float(med[1])]
            return result(OK, V_MOVED, "", **ev)
    if "occluded" in kinds:
        return unknown("footprint_occluded", **ev)
    if "partial" in kinds or hits:
        return unknown("partial_in_footprint", **ev)
    if mv:
        return unknown("moved_inconsistent", **ev)
    empties = sum(1 for k in kinds if k in ("empty", "ghost_flat"))
    if phase != "tucked":
        return unknown("arm_not_tucked", **ev)
    if empties >= cfg.HOLD_ABSENT_MIN_FRAMES:
        return result(OK, V_ABSENT, "", **ev)
    return unknown("insufficient", **ev)


# ============================================================ P2 - bowl fit
def _kasa(xy):
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def _arc_deg(xy, cx, cy):
    if len(xy) < 2:
        return 0.0
    a = np.sort(np.degrees(np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)))
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


def ransac_circle(xy, iters, tol, rng, r_lo, r_hi):
    """Vectorised RANSAC: `iters` 3-point circumcircles at once, best inlier
    count wins, then a Kasa refit on its inliers."""
    n = len(xy)
    if n < 3:
        return None
    idx = np.stack([rng.choice(n, 3, replace=False) for _ in range(iters)])
    A, B, Cc = xy[idx[:, 0]], xy[idx[:, 1]], xy[idx[:, 2]]
    d = 2.0 * (A[:, 0] * (B[:, 1] - Cc[:, 1]) + B[:, 0] * (Cc[:, 1] - A[:, 1])
               + Cc[:, 0] * (A[:, 1] - B[:, 1]))
    a2, b2, c2 = (A * A).sum(1), (B * B).sum(1), (Cc * Cc).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux = (a2 * (B[:, 1] - Cc[:, 1]) + b2 * (Cc[:, 1] - A[:, 1]) + c2 * (A[:, 1] - B[:, 1])) / d
        uy = (a2 * (Cc[:, 0] - B[:, 0]) + b2 * (A[:, 0] - Cc[:, 0]) + c2 * (B[:, 0] - A[:, 0])) / d
        r = np.hypot(A[:, 0] - ux, A[:, 1] - uy)
    good = np.isfinite(r) & (r >= r_lo) & (r <= r_hi)
    if not good.any():
        return None
    ux, uy, r = ux[good], uy[good], r[good]
    dist = np.hypot(xy[None, :, 0] - ux[:, None], xy[None, :, 1] - uy[:, None])
    cnt = (np.abs(dist - r[:, None]) <= tol).sum(1)
    k = int(np.argmax(cnt))
    if cnt[k] < 3:
        return None
    inl = np.abs(dist[k] - r[k]) <= tol
    fit = _kasa(xy[inl])
    if fit is None:
        return None
    cx, cy, rr = fit
    inl = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - rr) <= tol
    return cx, cy, rr, inl


def fit_bowl(frames, label, ref_xy, pose, cfg=None):
    """P2: rim height of a bowl by a top-down z-sweep of circle fits.

    The rim level is the HIGHEST z slice whose points fit a circle that passes
    every gate (radius window vs width_m, centre near the row median, inlier
    fraction, arc).  A contaminating blob above the rim (a hand) cannot pass
    the circle gates, so the sweep descends past it (no p95 anchor)."""
    cfg = cfg or DEFAULT_CFG
    ok, why = pose_check(pose, cfg)
    if not ok:
        return unknown(why)
    segs, row = [], None
    for f in list(frames or [])[::-1]:
        st, idx, r, seg = select_target(f, label, ref_xy, cfg)
        if st == SEL_OK:
            segs.append(seg)
            if row is None:
                row = r
        if len(segs) >= cfg.BOWL_FRAMES:
            break
    if row is None:
        return unknown("no_packet" if not frames else "no_segment")
    formula_z = _formula_z(row)
    ev0 = {"formula_z": formula_z}
    P = np.vstack(segs)
    if len(P) < cfg.BOWL_MIN_PTS:
        return unknown("sparse", n=len(P), **ev0)
    mx, my, rz = float(row["x"]), float(row["y"]), float(row["z"])
    w = row.get("width_m")
    r_nom = (float(w) / 2.0) / ((cfg.WIDTH_Q_HI - cfg.WIDTH_Q_LO) / 100.0) \
        if isinstance(w, (int, float)) and w > 0 else None
    trim = (r_nom * (1.0 + cfg.BOWL_R_WIDTH_TOL) if r_nom else cfg.BOWL_R_MAX_M) + 0.03
    P = P[np.hypot(P[:, 0] - mx, P[:, 1] - my) <= trim]
    if _border_pts(P, cfg) >= cfg.BOWL_EDGE_MIN_PTS:
        # border bowl: rim LOW
        # MEASURED -0.022 m
        return unknown("edge_clipped", **ev0)
    if len(P) < cfg.BOWL_MIN_PTS:
        return unknown("sparse_after_trim", n=len(P), **ev0)
    r_lo, r_hi = cfg.BOWL_R_MIN_M, cfg.BOWL_R_MAX_M
    if r_nom:
        r_lo = max(r_lo, r_nom * (1.0 - cfg.BOWL_R_WIDTH_TOL))
        r_hi = min(r_hi, r_nom * (1.0 + cfg.BOWL_R_WIDTH_TOL))
    if r_lo > r_hi:
        return unknown("radius_window_empty", r_nom=r_nom, **ev0)
    rng = np.random.default_rng(0)
    z = P[:, 2]
    levels = np.arange(np.percentile(z, 99.5), np.percentile(z, 40), -cfg.BOWL_SWEEP_STEP_M)
    best = None
    tried = 0
    for zl in levels:
        S = P[np.abs(z - zl) <= cfg.BOWL_SLICE_HALF_M]
        if len(S) < cfg.BOWL_RIM_MIN_PTS:
            continue
        tried += 1
        fit = ransac_circle(S[:, :2], cfg.BOWL_RANSAC_ITERS, cfg.BOWL_RIM_TOL_M, rng, r_lo, r_hi)
        if fit is None:
            continue
        cx, cy, r, inl = fit
        frac = float(inl.mean())
        arc = _arc_deg(S[inl, :2], cx, cy)
        if not (r_lo <= r <= r_hi):
            continue
        if math.hypot(cx - mx, cy - my) > cfg.BOWL_CENTER_GATE_M:
            continue
        if frac < cfg.BOWL_RIM_INLIER_FRAC or arc < cfg.BOWL_RIM_MIN_ARC_DEG:
            continue
        best = (float(zl), cx, cy, r, frac, arc)
        break
    if best is None:
        return unknown("rim_fit", levels_tried=tried, **ev0)
    zl, cx, cy, r, frac, arc = best
    d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    # rim-is-top gate
    # sparse rim -> sweep drops
    # to a WALL circle 5-6 cm low
    # bowl widens upward -> wall/rim
    # on/outside; inside ignored
    above = (d >= r - cfg.BOWL_RIM_TOL_M) & (z > zl + cfg.BOWL_TOP_GAP_M)
    n_above = int(above.sum())
    if n_above >= cfg.BOWL_TOP_ABOVE_MAX_PTS:
        return unknown("rim_not_top", zl=float(zl), r=float(r), n_above=n_above, **ev0)
    ring =np.abs(d - r) <= cfg.BOWL_RIM_TOL_M
    near = ring & (z >= zl - 0.03) & (z <= zl + 0.03)
    Z = z[near]
    if len(Z) < cfg.BOWL_RIM_MIN_PTS:
        return unknown("rim_sparse", **ev0)
    # rim = top of z spread
    # sigma from upper spread
    # (wall is one-sided)
    rim_z = float(np.percentile(Z, 90))
    upper = Z[Z >= np.median(Z)]
    sigma = float(1.4826 * np.median(np.abs(upper - np.median(upper)))) if len(upper) > 3 else 0.0
    if sigma > cfg.BOWL_NOISE_UNKNOWN_M:
        return unknown("too_noisy", sigma=sigma, **ev0)
    if not (rz - cfg.BOWL_RIM_BELOW_MEDIAN_MAX_M <= rim_z <= rz + cfg.BOWL_RIM_ABOVE_MEDIAN_MAX_M):
        return unknown("rim_vs_median_gross", rim_z=rim_z, row_z=rz, **ev0)
    margin = max(0.0, cfg.BOWL_MARGIN_K * sigma)
    drop = {"x": cx, "y": cy, "z": rim_z + cfg.PLACE_CLEARANCE_M + margin}
    fl = (d <= cfg.BOWL_FLOOR_R_FRAC * r) & (z < rim_z - cfg.BOWL_FLOOR_MIN_DEPTH_M)
    floor_z, floor_reason = None, "floor_not_seen"
    if int(fl.sum()) >= cfg.BOWL_FLOOR_MIN_PTS:
        floor_z = float(np.percentile(z[fl], 10))
        floor_reason = ""
    return result(OK, V_RIM, "", drop=drop, rim_z=rim_z, r=r, center=[cx, cy],
                  arc_deg=arc, inlier_frac=frac, sigma=sigma, margin=margin, n_above=n_above,
                  floor_z=floor_z, floor_reason=floor_reason, n_pts=int(len(P)),
                  formula_z=formula_z,
                  delta_z=(drop["z"] - formula_z) if formula_z is not None else None,
                  band=descent_band(drop["z"], cfg),
                  band_formula=descent_band(formula_z, cfg) if formula_z is not None else "ABSENT",
                  pose=dict(pose))


def _border_pts(P, cfg):
    u, v, z = torso_to_pixel(P)
    e = cfg.BOWL_EDGE_PX
    return int(((u <= e) | (u >= COLOR_W - 1 - e) | (v <= e) | (v >= COLOR_H - 1 - e)).sum())


def _formula_z(row):
    """Today's drop z: brain/agent/tools.py _drop_point 'z + lift + 0.02'."""
    try:
        z = float(row["z"])
    except (KeyError, TypeError, ValueError):
        return None
    h = row.get("height_m")
    lift = h / 2.0 if isinstance(h, (int, float)) and h > 0.0 else 0.0
    return z + lift + 0.02


# ================================================================ P3 - split
def _xy_box(seg):
    lo, hi = np.percentile(seg[:, :2], [5, 95], axis=0)
    return lo, hi


def _overlap_frac(a, b):
    (alo, ahi), (blo, bhi) = a, b
    w = np.clip(np.minimum(ahi, bhi) - np.maximum(alo, blo), 0, None)
    inter = float(w[0] * w[1])
    area = float(np.prod(np.maximum(ahi - alo, 1e-6)))
    return inter / area


def _width_step(T, cfg):
    z = T[:, 2]
    z0 = float(z.min())
    n = int(math.ceil((float(z.max()) - z0) / cfg.SPLIT_SLAB_M)) or 1
    k = np.minimum(((z - z0) / cfg.SPLIT_SLAB_M).astype(int), n - 1)
    w = np.full(n, np.nan)
    for i in range(n):
        xs = T[k == i, 0]
        if len(xs) >= cfg.SPLIT_MIN_SLAB_PTS:
            w[i] = float(np.percentile(xs, 95) - np.percentile(xs, 5))
    max_low = int(round(cfg.SUPPORT_MAX_H_M / cfg.SPLIT_SLAB_M))
    # largest lower run first
    # support top = wide top
    for kk in range(min(max_low, n - 1), 0, -1):
        lower = w[:kk]
        lower = lower[np.isfinite(lower)]
        if len(lower) == 0:
            continue
        Wl = float(np.median(lower))
        upper_all = w[kk:]
        # wide->narrow within
        # SPLIT_LEDGE_MAX_SLABS slabs
        upper = w[kk + cfg.SPLIT_LEDGE_MAX_SLABS:]
        upper = upper[np.isfinite(upper)]
        if len(upper) < 2:
            continue
        Wu = float(np.median(upper))
        if Wu <= 0 or Wl < cfg.SPLIT_WIDTH_RATIO * Wu:
            continue
        h_low = kk * cfg.SPLIT_SLAB_M
        h_up = (n - kk) * cfg.SPLIT_SLAB_M
        if h_up < max(cfg.SPLIT_MIN_UPPER_H_M, cfg.SPLIT_UPPER_TO_LOWER_H * h_low):
            continue
        if np.any(upper > 1.25 * Wu + 0.01):
            continue
        tr = upper_all[:cfg.SPLIT_LEDGE_MAX_SLABS]
        tr = tr[np.isfinite(tr)]
        if np.any(tr > Wl * 1.25):
            continue
        # lower slabs all wide
        if np.any(lower < 0.75 * Wl):
            continue
        # support top = p95 z of
        # wide-run points outside
        # the upper object
        zc = z0 + (kk + cfg.SPLIT_LEDGE_MAX_SLABS) * cfg.SPLIT_SLAB_M
        U = T[z >= zc]
        ulo, uhi = np.percentile(U[:, 0], [cfg.WIDTH_Q_LO, cfg.WIDTH_Q_HI])
        Wpts = T[(z < zc) & ((T[:, 0] < ulo - 0.01) | (T[:, 0] > uhi + 0.01))]
        sz = float(np.percentile(Wpts[:, 2], 95)) if len(Wpts) >= 10 else z0 + h_low
        return sz, {"W_lower": Wl, "W_upper": Wu, "h_lower": h_low, "n_support_pts": int(len(Wpts))}
    return None, {"widths": [None if not np.isfinite(v) else round(float(v), 4) for v in w]}


def split_support(frames, label, ref_xy, cfg=None):
    cfg = cfg or DEFAULT_CFG
    frame, idx, row, T = None, None, None, None
    for f in list(frames or [])[::-1]:
        st, i, r, seg = select_target(f, label, ref_xy, cfg)
        if st == SEL_OK:
            frame, idx, row, T = f, i, r, seg
            break
    if T is None:
        return unknown("no_packet" if not frames else "no_segment")
    if len(T) < cfg.SPLIT_MIN_PTS:
        return unknown("sparse", n=len(T))
    method, support_z, info, corroborated = None, None, {}, False
    tb = _xy_box(T)
    t_med_z = float(np.median(T[:, 2]))
    for o, s in frame.segs.items():
        if o == idx or o >= len(frame.rows) or o in frame.stale:
            continue
        if base_noun(frame.rows[o].get("label")) not in cfg.SUPPORT_LABELS:
            continue
        if len(s) < cfg.SPLIT_MIN_SLAB_PTS:
            continue
        span = float(np.percentile(s[:, 2], 95) - np.percentile(s[:, 2], 5))
        top = float(np.percentile(s[:, 2], 90))
        ov = _overlap_frac(tb, _xy_box(s))
        if span <= cfg.SUPPORT_MAX_H_M and ov >= cfg.SPLIT_OVERLAP_FRAC and top < t_med_z:
            method, support_z, corroborated = "support_row", top, True
            info = {"support_label": frame.rows[o].get("label"), "overlap": ov, "support_span": span}
            break
    if method is None:
        sz, info = _width_step(T, cfg)
        if sz is not None:
            method, support_z = "width_step", sz
    if method is None:
        return result(OK, V_NOSPLIT, "", centroid=[float(v) for v in np.median(T, axis=0)],
                      n_total=int(len(T)), **info)
    above = T[T[:, 2] > support_z + cfg.SPLIT_MARGIN_M]
    if len(above) < cfg.SPLIT_MIN_ABOVE_PTS:
        return unknown("above_sparse", method=method, support_z=support_z)
    return result(OK, V_SPLIT, "", method=method, support_z=support_z,
                  corroborated=corroborated,
                  centroid=[float(v) for v in np.median(above, axis=0)],
                  height_above=float(np.percentile(above[:, 2], 95) - support_z),
                  n_above=int(len(above)), n_total=int(len(T)), **info)
