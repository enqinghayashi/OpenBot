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
