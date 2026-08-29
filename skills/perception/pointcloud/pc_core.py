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
