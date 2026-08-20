import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
import numpy as np
import json
import gc
import struct
from scipy.spatial.transform import Rotation

from grasp_geometry import fit_geometric_primitive, compute_x_off

try:
    import cv2
except Exception:  # noqa: BLE001 - cv2 optional
    cv2 = None

# ── the camera intrinsics (color-res 424x240) ────────────────────────────────
FX, FY, CX, CY = 302.7, 302.8, 212.9, 121.1
DEPTH_MIN = 0.3
DEPTH_MAX = 6.0
# contrast floor, m
# unused; guard off
PLANE_CONTRAST_MIN = 0.015

# stream res
COLOR_W, COLOR_H = 424, 240

# per-object cap
MAX_PTS_PER_OBJ = 2000

# fit cap
# same ceil-stride rule as the cloud
# SVD/dims: all pts
MAX_FIT_PTS = 2000

# global cap, truncated before publish
# 20000 keeps dense clouds whole
MAX_CLOUD_POINTS = 20000

# ── the camera extrinsics - physical measurements ────────────────────────────
# cam: X right, Y down, Z fwd
# torso: X right, Y fwd, Z up
# physical mount:
# height 462.68mm = 0.46268m
# forward 47.64mm = 0.04764m
# sideways 0
# tilt down 48 degrees from horizontal
# transform:
#   torso_x = x_cam (lateral passes)
#   torso_y = -0.74314483*y_cam + 0.66913061*z_cam + 0.04764
#   torso_z = -0.66913061*y_cam - 0.74314483*z_cam + 0.46268

_t = np.radians(48.0)
R_base   = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
Rx_tilt  = np.array([[1,0,0],[0,np.cos(_t),np.sin(_t)],[0,-np.sin(_t),np.cos(_t)]])
R_CAM_TO_TORSO = R_base @ Rx_tilt
T_CAM_IN_TORSO = np.array([0.0, 0.04764, 0.46268])

# world yaw: cam sideways
CAM_YAW_OFFSET = -1.5708


def foreground_depth(depth_values, num_bins=5, min_cluster_size=3):
    """5-bin histogram of depth values.

    Return the median of the closest (smallest depth) cluster that has at
    least ``min_cluster_size`` pixels, together with a mask of the indices
    that belong to that cluster.
    """
    if len(depth_values) == 0:
        raise ValueError("empty depth array")
    counts, bin_edges = np.histogram(depth_values, bins=num_bins)
    for i in range(len(counts)):
        if counts[i] >= min_cluster_size:
            lo, hi = bin_edges[i], bin_edges[i + 1]
            cluster_mask = (depth_values >= lo) & (depth_values < hi)
            cluster_vals = depth_values[cluster_mask]
            return float(np.median(cluster_vals)), cluster_mask
    return float(np.median(depth_values)), np.ones(len(depth_values), dtype=bool)


# fit the normal on the mask centre: silhouette pixels carry the
# table's depth, so a whole-mask fit picks width as the least-variance
# axis. centre rows, 10% edge cut, robust band; under 30 pts -> whole mask
NORMAL_CENTRAL_FRAC = 0.5
NORMAL_EDGE_FRAC = 0.1
NORMAL_MAD_K = 3.0
NORMAL_BAND_MIN_M = 0.01
NORMAL_MIN_PTS = 30


def surface_normal_cam(pts_cam, xs, ys):
    """(normal_cam, n_used, central) for one object's deprojected mask points.

    ``pts_cam`` (N, 3) camera-frame points, ``xs`` / ``ys`` their depth-image
    pixel columns / rows.  The normal is the least-variance axis of the
    central-region points (see the NORMAL_* constants), flipped to point
    toward the camera (camera z <= 0); ``central`` says whether the centre
    region was used (False = the whole-mask fit).  Pure numpy, never raises
    for N >= 3."""
    pts_cam = np.asarray(pts_cam, dtype=float)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    use = pts_cam
    central = False
    try:
        order = np.argsort(ys, kind="stable")
        ys_s = ys[order]
        xs_s = xs[order]
        rows, start = np.unique(ys_s, return_index=True)
        rmin = np.minimum.reduceat(xs_s, start).astype(float)
        rmax = np.maximum.reduceat(xs_s, start).astype(float)
        ri = np.searchsorted(rows, ys)
        c = 0.5 * (rmin[ri] + rmax[ri])
        half = 0.5 * (rmax[ri] - rmin[ri])
        keep = np.abs(xs - c) <= NORMAL_CENTRAL_FRAC * half
        ylo, yhi = float(ys.min()), float(ys.max())
        span = yhi - ylo
        keep &= (ys >= ylo + NORMAL_EDGE_FRAC * span) & \
                (ys <= yhi - NORMAL_EDGE_FRAC * span)
        cand = pts_cam[keep]
        if len(cand) >= NORMAL_MIN_PTS:
            z = cand[:, 2]
            med = float(np.median(z))
            mad = float(np.median(np.abs(z - med)))
            band = max(NORMAL_MAD_K * 1.4826 * mad, NORMAL_BAND_MIN_M)
            cand = cand[np.abs(z - med) <= band]
            if len(cand) >= NORMAL_MIN_PTS:
                use = cand
                central = True
    except Exception:  # noqa: BLE001 - fallback fit
        use = pts_cam
        central = False
    centered = use - np.median(use, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal_cam = vt[2]
    if normal_cam[2] > 0:
        normal_cam = -normal_cam
    return normal_cam, int(len(use)), central


def rasterize_mask(mask_poly, out_h, out_w, src_h=COLOR_H, src_w=COLOR_W):
    """Rasterize a color-space segmentation polygon into a binary mask.

    ``mask_poly`` is a COCO-style polygon (list of [x, y] pairs) in color-pixel
    coords (from yolo_node_SEG_version.py).  Returns an ``(out_h, out_w)``
    uint8 binary mask with the polygon filled, scaling coords from color to the
    output (depth) resolution.  Returns None if cv2 is unavailable or the
    polygon is degenerate — callers fall back to the 7x7 bbox patch.
    """
    if cv2 is None or not mask_poly or len(mask_poly) < 3:
        return None
    try:
        pts = np.asarray(mask_poly, dtype=np.float32).reshape(-1, 2)
        pts = pts * np.array([out_w / src_w, out_h / src_h], dtype=np.float32)
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        m = np.zeros((out_h, out_w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        return m
    except Exception:
        return None


def build_pointcloud(points, obj_ids, frame_id, stamp):
    """Pack a list of (x, y, z) torso-frame points + per-point object_id into a
    sensor_msgs/PointCloud2 with fields x/y/z (float32) and object_id (uint32).
    object_id indexes the co-published /object_detections_body array so labels
    join by index.  ``points`` is a list of (x, y, z) tuples or an (N, 3) array,
    ``obj_ids`` a matching sequence of non-negative ints (H-PROJ-3)."""
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='object_id', offset=12, datatype=PointField.UINT32, count=1),
    ]
    header = Header()
    header.frame_id = frame_id
    header.stamp = stamp
    cloud = PointCloud2()
    cloud.header = header
    cloud.height = 1
    cloud.width = len(points)
    cloud.is_bigendian = False
    cloud.point_step = 16
    cloud.row_step = cloud.point_step * cloud.width
    cloud.is_dense = True
    cloud.fields = fields
    # vectorised wire
    # byte-identical to the old per-point '<fffI'
    # loop (LE, 16 B: x y z f32 + oid u32)
    rec = np.zeros(len(points), dtype=np.dtype(
        [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('oid', '<u4')]))
    if len(points):
        P = np.asarray(points, dtype=np.float64)
        rec['x'], rec['y'], rec['z'] = P[:, 0], P[:, 1], P[:, 2]
        rec['oid'] = np.asarray(obj_ids, dtype=np.uint32)
    cloud.data = rec.tobytes()
    return cloud
