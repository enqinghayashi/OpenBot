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


class DepthProjectorNode(Node):
    def __init__(self):
        super().__init__('depth_projector_node')

        self._latest_depth = None
        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_z   = 0.0
        self._robot_yaw = 0.0

        self.create_subscription(
            String,   '/detections',                self._on_detections, 10)
        # depth QoS
        # KEEP_LAST 1, best-effort - only the newest
        # frame is used; a queue just holds RAM
        self.create_subscription(
            Image, '/camera/depth/image_raw', self._on_depth,
            qos_profile=rclpy.qos.QoSProfile(
                depth=1,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(
            Odometry, '/unitree/slam_mapping/odom', self._on_odom,       10)

        self.pub      = self.create_publisher(String, '/object_detections',      10)
        self.pub_body = self.create_publisher(String, '/object_detections_body', 10)
        self.pub_cloud = self.create_publisher(
            PointCloud2, '/object_pointcloud', 10)

        self.get_logger().info(
            f'Depth projector (SEG) ready. '
            f'Camera: z={T_CAM_IN_TORSO[2]:.4f}m above torso, '
            f'y={T_CAM_IN_TORSO[1]:.4f}m forward, tilt=48deg '
            f'cv2={"yes" if cv2 is not None else "NO (mask path disabled)"}')

        # periodic gc + rss logging keeps memory bounded
        self._cb_count = 0
        self._gc_every = 200
        self._log_rss_every = 300

    def _on_odom(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        self._robot_z = msg.pose.pose.position.z
        q = msg.pose.pose.orientation
        r = Rotation.from_quat([q.x, q.y, q.z, q.w])
        self._robot_yaw = r.as_euler('ZYX')[0]

    def _log_rss(self):
        """RSS telemetry from /proc/self/statm (stdlib) — the deploy gate
        measures <500 MB steady-state over 30 min."""
        try:
            with open('/proc/self/statm') as f:
                kb = int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') // 1024
        except Exception:
            return
        self.get_logger().info(f'[mem] rss={kb} kB')

    def _on_depth(self, msg):
        # uint16 -> f32 m, scaled in place
        # (no third full-frame temp per callback;
        # QoS keeps only the newest frame)
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape(
            msg.height, msg.width).astype(np.float32)
        arr *= 0.001
        self._latest_depth = arr

    def _cloud_wanted(self):
        """H-PROJ-3: True when the PointCloud2 must be built and published.

        Nothing consumes /object_pointcloud unless a node subscribes, so the
        per-callback build is skipped while the subscription count is 0.  Fails
        OPEN: any doubt returns True, which is the behaviour before this hook
        (build on every callback).  rclpy Foxy's Publisher.get_subscription_count
        is INFERRED, NOT verified on the box; if the attribute is missing the
        node-level count_subscribers() is asked about the cloud topic instead,
        and any other failure returns True.
        """
        try:
            try:
                return self.pub_cloud.get_subscription_count() > 0
            except AttributeError:
                return self.count_subscribers('/object_pointcloud') > 0
        except Exception:  # noqa: BLE001 -- fail open
            return True

    def _on_detections(self, msg):
        """HOTFIX 2026-10-09: one bad frame must never kill the node.

        An exception escaping a subscription callback ends rclpy.spin() and the
        node dies (robot, 16:04:29: IndexError in the fallback normal fit ->
        /object_detections_body silent, the brain's scene 'unavailable' for the
        rest of the session).  The frame is dropped and logged instead; the
        next frame is processed normally.
        """
        try:
            self._on_detections_impl(msg)
        except Exception as exc:  # noqa: BLE001 -- node survives
            n = getattr(self, '_cb_failures', 0) + 1
            self._cb_failures = n
            if n <= 5 or n % 100 == 0:
                tb, where = exc.__traceback__, ''
                while tb is not None:  # innermost frame, this file
                    if tb.tb_frame.f_code.co_filename == __file__:
                        where = ' at line %d' % tb.tb_lineno
                    tb = tb.tb_next
                self.get_logger().error(
                    f'detections callback failed ({n}x), frame dropped{where}: '
                    f'{type(exc).__name__}: {exc}')

    def _on_detections_impl(self, msg):
        if self._latest_depth is None:
            return
        # GC + RSS telemetry
        self._cb_count += 1
        if self._cb_count % self._gc_every == 0:
            gc.collect()
        if self._cb_count % self._log_rss_every == 0:
            self._log_rss()

        detections = json.loads(msg.data)
        depth      = self._latest_depth
        dh, dw     = depth.shape

        world_results = []
        body_results  = []
        cloud_pts     = []
        cloud_ids     = []

        for det in detections:
            label = det['label']
            conf  = det['conf']
            cx = int(det['cx'])
            cy = int(det['cy'])

            # ── working out the orientation ──────────────────────────────
            orientation = det.get('orientation', 'upright')
            if not orientation or orientation == 'unknown':
                orientation = "upright"
                if det.get('label') == 'bottle' and 'aspect_ratio' in det:
                    ar = det['aspect_ratio']
                    if ar > 1.5:
                        orientation = "horizontal"
                    elif ar < 0.7:
                        orientation = "upright"

            normal_x = normal_y = normal_z = None
            width_m = depth_m = height_m = None
            x_off = object_radius = quality = None
            primitive_type = None

            # ── SEG path: mask → just the object's pixels ────────────────
            # mask pixels only
            # the 7x7 bbox patch is gone, so table/
            # background can't leak into centroid, normal
            seg_ok = False
            n_pts_this = None  # dims or cloud
            mask_poly = det.get('mask')
            depth_mask = rasterize_mask(mask_poly, dh, dw) if mask_poly else None
            if depth_mask is not None and int((depth_mask > 0).sum()) >= 3:
                obj_pix = (depth_mask > 0) & (depth > DEPTH_MIN) & (depth < DEPTH_MAX)
                if int(obj_pix.sum()) >= 3:
                    ys, xs = np.where(obj_pix)
                    z_pts = depth[ys, xs]
                    # no foreground_depth
                    # the mask already isolates the object, and
                    # closest-cluster would keep only the nearest
                    # slice of a depth gradient (48deg bottle),
                    # shrinking height_m; use a robust band
                    # on boundary pixels
                    if len(z_pts) >= 3:
                        z_lo, z_hi = np.percentile(z_pts, [5, 95])
                        keep = (z_pts >= z_lo) & (z_pts <= z_hi)
                        if int(keep.sum()) >= 3:
                            ys = ys[keep]
                            xs = xs[keep]
                            z_pts = z_pts[keep]
                        # colour space
                        cols_c = xs * (COLOR_W / float(dw))
                        rows_c = ys * (COLOR_H / float(dh))
                        x_c = (cols_c - CX) * z_pts / FX
                        y_c = (rows_c - CY) * z_pts / FY
                        pts_cam = np.column_stack([x_c, y_c, z_pts])
                        # median centroid
                        centroid_cam = np.median(pts_cam, axis=0)

                        # centre SVD
                        # flipped to face camera
                        normal_cam, _n_nrm, _nrm_central = surface_normal_cam(
                            pts_cam, xs, ys)
                        normal_torso = R_CAM_TO_TORSO @ normal_cam
                        normal_x = round(float(normal_torso[0]), 4)
                        normal_y = round(float(normal_torso[1]), 4)
                        normal_z = round(float(normal_torso[2]), 4)

                        # percentile dims
                        # (outliers don't skew) + cloud points
                        pts_torso = (R_CAM_TO_TORSO @ pts_cam.T).T + T_CAM_IN_TORSO
                        p5, p95 = np.percentile(pts_torso, [5, 95], axis=0)
                        width_m = round(float(p95[0] - p5[0]), 4)
                        depth_m = round(float(p95[1] - p5[1]), 4)
                        height_m = round(float(p95[2] - p5[2]), 4)

                        # point = centroid
                        x_cam = float(centroid_cam[0])
                        y_cam = float(centroid_cam[1])
                        z_cam = float(centroid_cam[2])

                        # ── point-cloud grasp params (grasp_geometry) ─────
                        # fit a primitive
                        # x_off = -(radius + palm + insertion),
                        # replacing the daemon's -0.08 standoff
                        try:
                            # stride fit points
                            # to MAX_FIT_PTS (ceil rule, like the cloud);
                            # SVD normal + dims use every point
                            fit_stride = max(1, (len(pts_torso) + MAX_FIT_PTS - 1) // MAX_FIT_PTS)
                            pts_fit = pts_torso[::fit_stride]
                            primitive = fit_geometric_primitive(pts_fit)
                            object_radius = (float(primitive.radius)
                                             if primitive.radius is not None
                                             else 0.035)
                            _grasp_z = float(primitive.centre[2]) + 0.04
                            _x, _q = compute_x_off(
                                primitive.type,
                                radius=object_radius,
                                object_width=2.0 * object_radius,
                                object_centre_z=float(primitive.centre[2]),
                                grasp_z=_grasp_z)
                            x_off = round(float(_x), 4)
                            quality = round(float(_q), 4)
                            object_radius = round(object_radius, 4)
                            primitive_type = primitive.type
                        except Exception:
                            x_off = object_radius = quality = None
                            primitive_type = None

                        # cloud downsample
                        # ceil division so the cap applies: floor let
                        # N=3800 publish 3800, 2x the cap
                        stride = max(1, (len(pts_torso) + MAX_PTS_PER_OBJ - 1) // MAX_PTS_PER_OBJ)
                        oid = len(body_results)  # body_results index
                        # accumulate arrays, not points: one vstack
                        # before the cap; same points, same order
                        sub = pts_torso[::stride]
                        cloud_pts.append(sub)
                        cloud_ids.append(np.full(len(sub), oid, dtype=np.uint32))
                        # post-stride count
                        # pre-cap
                        n_pts_this = len(sub)

                        seg_ok = True

            if not seg_ok:
                # ── FALLBACK: fixed 7x7 bbox-center patch (same as before) ─────
                cx_d = int(cx * dw / COLOR_W)
                cy_d = int(cy * dh / COLOR_H)
                x1 = max(0, cx_d - 3); x2 = min(dw, cx_d + 4)
                y1 = max(0, cy_d - 3); y2 = min(dh, cy_d + 4)

                patch = depth[y1:y2, x1:x2]
                valid_mask = (patch > DEPTH_MIN) & (patch < DEPTH_MAX)
                valid = patch[valid_mask]
                if len(valid) == 0:
                    continue

                z_cam, cluster_mask_1d = foreground_depth(valid)
                # '< hi' drops ties
                # in the LAST bin, so ">= 3" can hold 0-2
                # points; 0 -> NaN depth, no row, never on
                # the wire
                if not np.isfinite(z_cam):
                    continue

                # ── the 7x7 patch normal (SVD plane fit) ────────────────
                # count CLUSTER points
                # 2 pts crashed Vt[2]
                if int(np.count_nonzero(cluster_mask_1d)) >= 3:
                    cluster_mask_2d = np.zeros_like(patch, dtype=bool)
                    valid_indices_2d = np.where(valid_mask)
                    selected_indices = np.where(cluster_mask_1d)[0]
                    cluster_mask_2d[valid_indices_2d[0][selected_indices],
                                    valid_indices_2d[1][selected_indices]] = True

                    py, px = np.where(cluster_mask_2d)
                    cx_full = x1 + px
                    cy_full = y1 + py
                    z_vals = patch[py, px]

                    x_cam_i = (cx_full - CX) * z_vals / FX
                    y_cam_i = (cy_full - CY) * z_vals / FY
                    z_cam_i = z_vals

                    points_3d = np.column_stack([x_cam_i, y_cam_i, z_cam_i])
                    centered = points_3d - points_3d.mean(axis=0)
                    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                    normal_cam = Vt[2]
                    if normal_cam[2] > 0:
                        normal_cam = -normal_cam
                    normal_torso = R_CAM_TO_TORSO @ normal_cam
                    normal_x = round(float(normal_torso[0]), 4)
                    normal_y = round(float(normal_torso[1]), 4)
                    normal_z = round(float(normal_torso[2]), 4)

                # pixel -> cam
                x_cam = (cx - CX) * z_cam / FX
                y_cam = (cy - CY) * z_cam / FY

                # 2D height signal: bbox height in px -> m via depth, tilt-corrected
                if 'bbox_h' in det:
                    bbox_h_d = det['bbox_h'] * dh / COLOR_H
                    height_m = round(float(bbox_h_d * z_cam / (FY * 0.66913)), 4)

            # ── shared: torso-frame + world-frame transforms ─────────────
            p_cam = np.array([x_cam, y_cam, z_cam])
            p_torso = R_CAM_TO_TORSO @ p_cam + T_CAM_IN_TORSO

            body = {
                'label': label,
                'conf':  conf,
                'x': round(float(p_torso[0]), 3),   # +x = right
                'y': round(float(p_torso[1]), 3),   # forward distance
                'z': round(float(p_torso[2]), 3),   # height above torso
                # capture stamp
                # 0.041-0.281 s latency (p50 0.061,
                # p90 0.209), publish gaps to 0.331 s
                # walk_node needs the CAPTURE gap, not the
                # receipt gap; additive + None-safe
                'stamp': det.get('stamp'),
                'nx': normal_x,
                'ny': normal_y,
                'nz': normal_z,
                'orientation': orientation,
                # colour link
                # forward hue/color_name; frame_side is
                # display-only, so it's dropped
                'hue': det.get('hue'),
                'color_name': det.get('color_name'),
                # torso only
                # never from the world transform (KNOWN BROKEN)
                'body_side': 'right' if p_torso[0] > 0 else 'left',
            }
            if width_m is not None:
                body['width_m'] = width_m
            if depth_m is not None:
                body['depth_m'] = depth_m
            if height_m is not None:
                body['height_m'] = height_m
            if x_off is not None:
                body['x_off'] = x_off
                body['object_radius'] = object_radius
                body['primitive_type'] = primitive_type
                body['quality'] = quality
            # instance block
            # only rows with OPENBOT_INSTANCE_IDS=1
            # carry it; class_name is never forwarded
            if 'instance_id' in det:
                body['instance_id'] = det['instance_id']
                body['track_age_s'] = det.get('track_age_s')
                body['class_name'] = det.get('class_name')
            # n_pts: SEG only
            # (they carry width_m); cloud_stamp_ns
            # goes on just before the body is published
            if n_pts_this is not None:
                body['n_pts'] = n_pts_this
            body_results.append(body)

            # world: KNOWN BROKEN
            # (double -pi/2 quirk); nothing subscribes
            pt_body = np.array([z_cam, -x_cam, -y_cam])
            yaw     = self._robot_yaw + CAM_YAW_OFFSET
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            world_x = self._robot_x + cos_y * pt_body[0] - sin_y * pt_body[1]
            world_y = self._robot_y + sin_y * pt_body[0] + cos_y * pt_body[1]
            world_z = self._robot_z + T_CAM_IN_TORSO[2] + pt_body[2]
            world_results.append({
                'label': label,
                'conf':  conf,
                'x': round(world_x, 3),
                'y': round(world_y, 3),
                'z': round(world_z, 3),
            })

        if world_results:
            out = String()
            out.data = json.dumps(world_results)
            self.pub.publish(out)

        # one stamp
        # taken before the body publish so SEG rows
        # carry cloud_stamp_ns (ns = sec*1e9 +
        # nanosec); the cloud reuses this object
        stamp = self.get_clock().now().to_msg()
        cloud_stamp_ns = int(stamp.sec) * 1000000000 + int(stamp.nanosec)
        for _r in body_results:
            if 'width_m' in _r:
                _r['cloud_stamp_ns'] = cloud_stamp_ns
        if body_results:
            bout = String()
            bout.data = json.dumps(body_results)
            self.pub_body.publish(bout)
            for r in body_results:
                self.get_logger().info(
                    f"{r['label']} | "
                    f"x={r['x']:+.3f}(lr) "
                    f"y={r['y']:+.3f}(fwd) "
                    f"z={r['z']:+.3f}(ht)"
                    + (f" w={r['width_m']:.3f}" if 'width_m' in r else "")
                    + (f" d={r['depth_m']:.3f}" if 'depth_m' in r else "")
                    + (f" h={r['height_m']:.3f}" if 'height_m' in r else "")
                    + (f" x_off={r['x_off']:.3f} r={r['object_radius']:.3f}"
                       f" q={r['quality']:.2f} {r['primitive_type']}"
                       if 'x_off' in r else "")
                )

        # publish only if
        # subscribed (_cloud_wanted fails OPEN);
        # body JSON goes out either way.  CPU <
        # 91.5 % with nothing subscribed is ROBOT
        # ACCEPTANCE, not measured here (one top
        # -b sample, no rate probe)
        if cloud_pts and self._cloud_wanted():
            # cap before publish
            # bounds peak
            cloud_pts = np.vstack(cloud_pts)
            cloud_ids = np.concatenate(cloud_ids)
            n_pts = len(cloud_pts)
            if n_pts > MAX_CLOUD_POINTS:
                cloud_pts = cloud_pts[:MAX_CLOUD_POINTS]
                cloud_ids = cloud_ids[:MAX_CLOUD_POINTS]
                self.get_logger().warn(
                    f'[mem] cloud truncated {n_pts} -> {MAX_CLOUD_POINTS} pts')
            self.pub_cloud.publish(
                build_pointcloud(cloud_pts, cloud_ids, 'torso', stamp))


def main(args=None):
    rclpy.init(args=args)
    node = DepthProjectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
