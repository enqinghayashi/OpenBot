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
