"""pc_synth: a pure-numpy synthetic renderer that emulates the depth projector.

NOT a physics model.  It reproduces the GEOMETRY the projector sees:

  * the camera intrinsics, 48 deg tilt and torso extrinsics of
    skills/perception/depth_projector_node.py (literals are
    held in pc_core and pinned against the projector SOURCE by test_pc_synth);
  * ray-cast primitives in the torso frame (z up, +x RIGHT, +y forward):
    table plane, vertical cylinders (stacked = tapered/capped bottles),
    axis-aligned boxes (book, hip, arm, bag), a spherical-cap bowl shell;
  * the projector's SEG path for each detection: (mask>0) & depth gate,
    5-95 camera-depth band, colour-scaled deprojection, median-in-camera row
    xyz, p5-p95 dims, ceil-stride to MAX_PTS_PER_OBJ, oid = row index,
    the global MAX_CLOUD_POINTS truncation, and the struct.pack('<fffI') wire.

Depth noise (ALL INFERRED until capture_clouds.py has run on the box):
  sigma(z) = k * z**2 + s0  (Gaussian on camera depth), quantised to 1 mm,
  optional flying pixels (fraction, depth scaled by U(1.0, 1.25)),
  transparent objects: per-pixel depth holes and see-through (depth of what
  is behind the object).

Mask models:
  * leak_px  : dilate the visible-object mask so background depth leaks in
  * merge    : several objects reported as ONE detection (bottle on a book)
  * ghost    : a STALE mask (from an earlier scene) deprojected on CURRENT
               depth - the yolo 2 s candidate lifetime (yolo :382, :375, :407)
  * occluder : an unsegmented object (an arm) that hides what is behind it

Python 3.8-safe; numpy only.
"""
import math
import struct

import numpy as np

try:  # normal package import
    from . import pc_core as core
except ImportError:  # flat import, tests add path
    import pc_core as core


# ------------------------------------------------------------- depth noise
NOISE_NOMINAL = {"k": 0.004, "s0": 0.0, "fly": 0.0}          # just a guess
NOISE_D435 = {"k": 0.0075, "s0": 0.0, "fly": 0.0}            # z^2*subpx/(f*b)
NOISE_STRESS = {"k": 0.0, "s0": 0.025, "fly": 0.05}          # stress: 2-3 cm, 5 % fly


def camera_rays(dh, dw):
    """Unit-z camera rays for every depth pixel, the way the projector
    deprojects them (colour-scaled pixel coords, :270-273).  Returns the
    torso-frame direction (dh, dw, 3) such that p = T + t * d, t = openbot depth."""
    xs = np.arange(dw, dtype=np.float64)
    ys = np.arange(dh, dtype=np.float64)
    cols_c = xs * (core.COLOR_W / float(dw))
    rows_c = ys * (core.COLOR_H / float(dh))
    X, Y = np.meshgrid((cols_c - core.CX) / core.FX, (rows_c - core.CY) / core.FY)
    d_cam = np.stack([X, Y, np.ones_like(X)], axis=-1)
    return d_cam @ core.R_CAM_TO_TORSO.T


# ----------------------------------------------------------- the primitives
def _hit_plane(O, D, z, xlim=None, ylim=None):
    dz = D[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z - O[2]) / dz
    t = np.where((dz < 0) & (t > 0), t, np.inf)
    if xlim is not None or ylim is not None:
        P = O + t[..., None] * np.where(np.isfinite(t)[..., None], D, 0.0)
        ok = np.isfinite(t)
        if xlim is not None:
            ok &= (P[..., 0] >= xlim[0]) & (P[..., 0] <= xlim[1])
        if ylim is not None:
            ok &= (P[..., 1] >= ylim[0]) & (P[..., 1] <= ylim[1])
        t = np.where(ok, t, np.inf)
    return t


def _hit_cyl(O, D, cx, cy, r, z0, z1):
    ox, oy = O[0] - cx, O[1] - cy
    dx, dy, dz = D[..., 0], D[..., 1], D[..., 2]
    a = dx * dx + dy * dy
    b = 2.0 * (dx * ox + dy * oy)
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        t1 = (-b - sq) / (2 * a)
    z = O[2] + t1 * dz
    t_side = np.where((disc >= 0) & (t1 > 0) & (z >= z0) & (z <= z1), t1, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (z1 - O[2]) / dz
    px = O[0] + tt * dx - cx
    py = O[1] + tt * dy - cy
    t_top = np.where((dz < 0) & (tt > 0) & (px * px + py * py <= r * r), tt, np.inf)
    t = np.minimum(t_side, t_top)
    return np.where(np.isnan(t), np.inf, t)


def _hit_box(O, D, lo, hi):
    tmin = np.full(D.shape[:-1], -np.inf)
    tmax = np.full(D.shape[:-1], np.inf)
    for k in range(3):
        d = D[..., k]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[k] - O[k]) / d
            tb = (hi[k] - O[k]) / d
        t_lo = np.minimum(ta, tb)
        t_hi = np.maximum(ta, tb)
        par = d == 0
        inside = (O[k] >= lo[k]) & (O[k] <= hi[k])
        t_lo = np.where(par, np.where(inside, -np.inf, np.inf), t_lo)
        t_hi = np.where(par, np.where(inside, np.inf, -np.inf), t_hi)
        tmin = np.maximum(tmin, t_lo)
        tmax = np.minimum(tmax, t_hi)
    return np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)


def _sphere_iv(O, D, c, R):
    oc = O - np.asarray(c, dtype=float)
    a = (D * D).sum(-1)
    b = 2.0 * (D * oc).sum(-1)
    cc = (oc * oc).sum() - R * R
    disc = b * b - 4 * a * cc
    sq = np.sqrt(np.clip(disc, 0, None))
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    miss = disc < 0
    return np.where(miss, np.inf, t1), np.where(miss, -np.inf, t2)


def _hit_bowl(O, D, cx, cy, z_table, r_rim, depth, thick):
    """Spherical-cap shell, rim (outer) radius r_rim at z_table+depth."""
    Rs = (r_rim * r_rim + depth * depth) / (2.0 * depth)
    zc = z_table + Rs
    z_rim = z_table + depth
    a1, b1 = _sphere_iv(O, D, (cx, cy, zc), Rs)
    c1, d1 = _sphere_iv(O, D, (cx, cy, zc), Rs - thick)
    dz = D[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_rim = (z_rim - O[2]) / dz          # dz<0: t>=t_rim means z<=z_rim
    a = np.maximum(a1, np.where(dz < 0, t_rim, np.inf))
    b = b1
    # solid = [a,b]
    # minus [c1,d1]
    entry = np.where(a < c1, a, np.where((d1 < b) & (d1 >= a), d1, np.inf))
    entry = np.where((a <= b) & (entry <= b), entry, np.inf)
    return np.where(entry > 0, entry, np.inf)


class Obj(object):
    """A scene primitive.  kind in {plane, cyl, stack, box, bowl}."""

    def __init__(self, kind, label=None, seg=True, conf=0.9, transparent=None,
                 **geom):
        self.kind = kind
        self.label = label
        self.seg = seg
        self.conf = conf
        self.transparent = transparent       # (hole_frac, see_through_frac)
        self.geom = geom

    def hit(self, O, D):
        g = self.geom
        if self.kind == "plane":
            return _hit_plane(O, D, g["z"], g.get("xlim"), g.get("ylim"))
        if self.kind == "cyl":
            return _hit_cyl(O, D, g["cx"], g["cy"], g["r"], g["z0"], g["z0"] + g["h"])
        if self.kind == "stack":
            t = np.full(D.shape[:-1], np.inf)
            for (r, z0, z1) in g["profile"]:
                t = np.minimum(t, _hit_cyl(O, D, g["cx"], g["cy"], r, z0, z1))
            return t
        if self.kind == "box":
            return _hit_box(O, D, np.asarray(g["lo"], float), np.asarray(g["hi"], float))
        if self.kind == "bowl":
            return _hit_bowl(O, D, g["cx"], g["cy"], g["z_table"], g["r_rim"],
                             g["depth"], g.get("thick", 0.005))
        raise ValueError(self.kind)


# ----------------------------------------------------- little scene helpers
def table(z):
    return Obj("plane", label=None, seg=False, z=z)


def bottle(x, y, z_table, r=0.0425, h=0.20, label="bottle", **kw):
    return Obj("cyl", label=label, cx=x, cy=y, r=r, z0=z_table, h=h, **kw)


def tapered_bottle(x, y, z_table, r_body=0.0425, h_body=0.14, h_shoulder=0.03,
                   r_neck=0.015, h_neck=0.04, cap=False, label="bottle", **kw):
    prof = [(r_body, z_table, z_table + h_body)]
    n = 3
    for i in range(n):  # 3-step taper body -> neck
        rr = r_body + (r_neck - r_body) * (i + 1) / float(n + 1)
        za = z_table + h_body + h_shoulder * i / float(n)
        prof.append((rr, za, za + h_shoulder / float(n)))
    zn = z_table + h_body + h_shoulder
    prof.append((r_neck, zn, zn + h_neck))
    if cap:
        prof.append((r_neck + 0.003, zn + h_neck, zn + h_neck + 0.015))
    return Obj("stack", label=label, cx=x, cy=y, profile=prof, **kw)


def box(lo, hi, label=None, seg=True, **kw):
    return Obj("box", label=label, seg=seg, lo=lo, hi=hi, **kw)


