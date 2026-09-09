#!/usr/bin/env python3
"""
what_robot_sees.py — LIVE viewer for the pick-walk perception stack.

Run this on the LAPTOP while the pipeline runs on Stuart.  It subscribes to
the same four topics the pick pipeline uses and renders what the robot's
eyes + brain produce:

  LEFT  panel : the raw color frame (2x) with YOLO seg masks overlaid per
                label (semi-transparent fill + bbox + centroid cross +
                conf / orientation), exactly what yolo_node sees.
  RIGHT panel : the 3D /object_pointcloud in torso frame (x=right, y=forward,
                z=up), coloured per object_id, with the torso axes, the
                camera mount, and each /object_detections_body centroid with
                its surface-normal arrow (what depth_projector thinks the
                grasp surface is).
  bottom bar  : the latest body-frame numbers (x/y/z, dims, normal,
                x_off/radius/quality/primitive) — the exact inputs the
                daemon uses to plan the grasp.

Keys:
  s  = save snapshot (2D panel + 3D panel + raw color + detections JSON)
       into ./vision_snapshots/
  r  = auto-rotate the 3D panel
  q  = quit
Mouse on the RIGHT panel: drag to rotate.

Every /object_detections_body message is also appended (raw JSON) to
./vision_snapshots/detections.log for offline analysis.

Demo recording flags (2026-10-17, defence videos):
  --no-cloud        do NOT subscribe to /object_pointcloud and drop the 3D
                    panel (canvas = colour+masks panel + numbers bar, 848x576).
                    The projector builds the cloud ONLY while something
                    subscribes (depth_projector_node.py _cloud_wanted), so a
                    viewer with the cloud on costs the Orin CPU during a demo;
                    this flag keeps the robot's load exactly as without the
                    viewer.
  --record OUT.mp4  write every rendered canvas to OUT.mp4 (H.264 'avc1',
                    'mp4v' fallback) at --fps, plus OUT.mp4.times.csv (one
                    wall-clock stamp per frame) and OUT.mp4.meta.json (start,
                    end, frames, size) so demo_compose.py can align it with
                    the trace panel and the room camera.  A wall clock
                    (HH:MM:SS.d) is drawn in the footer for camera sync.
  --fps N           render/record rate (default 10, the historical 0.1 s timer)
  --no-window       headless: record without an X window
  --out-dir DIR     where snapshots and detections.log go (default: the
                    --record file's directory, else ./vision_snapshots here)
  --odom [--odom-topic T]
                    a second, square pane "odometry map": the odometry trail
                    (/unitree/slam_mapping/odom, the walk node's topic)
                    coloured by walk state (/walk_status), the robot with
                    its heading and camera wedge, the return goal of a
                    RETURNING status ("home"), markers at REACHED /
                    NOT_FOUND and at success verdicts (/skill_verdict), and
                    the current detections placed in the odometry frame
                    (walk_lock.body_to_world).  With --record it is written
                    beside the main file as map.mp4 (+ sidecars), the trail
                    to odom.csv, and the final map to odom_path.png.

Usage (laptop):
    sudo apt install -y ros-humble-rmw-cyclonedds-cpp   # once
    source /opt/ros/humble/setup.bash
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp        # match Stuart's RMW
    export CYCLONEDDS_URI=file://<repo>/skills/config/cyclonedds_laptop.xml
    ros2 daemon stop
    python3 what_robot_sees.py
    python3 what_robot_sees.py --no-cloud --record eyes.mp4   # demo take
    # ROS_DOMAIN_ID must match Stuart (0).  Stuart's pipeline must be launched
    # with CYCLONEDDS_URI=file:///home/unitree/cyclonedds_lan.xml so its ROS2
    # participants announce to this laptop (192.168.123.99).
"""
import argparse
import math
import signal
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
import numpy as np
import cv2
import json
import time
import os
import struct

COLOR_W, COLOR_H = 424, 240
T_CAM_IN_TORSO = np.array([0.0, 0.04764, 0.46268])  # camera position, torso frame
UPSCALE = 2                       # panel upscale
PANEL_W = COLOR_W * UPSCALE       # 848
PANEL_H = COLOR_H * UPSCALE       # 480
INFO_H = 96                       # bottom info-bar height
WIN_W, WIN_H = PANEL_W * 2, PANEL_H + INFO_H

PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (128, 0, 255), (255, 128, 0), (0, 128, 255), (255, 0, 128),
    (0, 255, 128), (128, 255, 0), (255, 128, 128), (128, 255, 255),
]
AXIS_COLORS = {'X': (0, 0, 255), 'Y': (0, 255, 0), 'Z': (255, 0, 0)}  # BGR

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vision_snapshots')

# --odom: extra top-down map pane
ODOM_TOPIC_DEFAULT = '/unitree/slam_mapping/odom'   # walk_node.ODOM_TOPIC
MAP_SIZE = 576                    # square pane, px
MAP_MARGIN = 44
MAP_MIN_SPAN_M = 3.0              # never zoom in closer
CAM_FOV_DEG = 70.0                # hfov (Table E.1)
CAM_RANGE_M = 0.9                 # table objects visible to here
WALK_COLORS = {                   # trail colour per walk state (BGR)
    'SEARCHING': (60, 160, 255), 'APPROACHING': (60, 220, 255),
    'REACHED': (90, 220, 90), 'RETURNING': (255, 170, 120),
    'NOT_FOUND': (80, 80, 230), 'IDLE': (150, 150, 150),
}


def label_color(label):
    return PALETTE[abs(hash(label)) % len(PALETTE)]


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _body_to_world(pose, x_b, y_b):
    """Body frame (x right, y forward) -> odometry world; the walk node's
    walk_lock.body_to_world, copied so this laptop tool has no robot import."""
    rx, ry, yaw = pose
    fwd, left = y_b, -x_b
    return (rx + math.cos(yaw) * fwd - math.sin(yaw) * left,
            ry + math.sin(yaw) * fwd + math.cos(yaw) * left)


def _clock_text(t):
    """HH:MM:SS.d of wall time ``t`` — the sync mark for a camera that films
    the laptop screen for a second at the start of a take."""
    return time.strftime('%H:%M:%S', time.localtime(t)) + '.%d' % int((t - int(t)) * 10)


class WhatRobotSees(Node):
    def __init__(self, opts=None):
        super().__init__('what_robot_sees')
        if opts is None:
            opts = _parse_args([])[0]
        self._no_cloud = bool(opts.no_cloud)
        self._no_window = bool(opts.no_window)
        self._fps = max(1.0, min(30.0, float(opts.fps)))
        self._win_w = PANEL_W if self._no_cloud else PANEL_W * 2
        self._win_h = PANEL_H + INFO_H
        self._snap_dir = (opts.out_dir or
                          (os.path.dirname(os.path.abspath(opts.record))
                           if opts.record else SNAP_DIR))
        os.makedirs(self._snap_dir, exist_ok=True)
        # --record: writer, per-frame stamp, meta sidecar
        self._writer = None
        self._times = None
        self._frames = 0
        self._meta = None
        if opts.record:
            self._open_record(opts.record)
        self._frame = None          # newest frame
        self._detections = []       # newest detections
        self._det_time = 0.0
        self._body = []             # body detections
        self._body_time = 0.0
        self._cloud_pts = np.zeros((0, 3))
        self._cloud_ids = np.zeros(0, dtype=np.int64)
        self._cloud_time = 0.0

        # best-effort read, but binds reliable writers too:
        # no QoS mismatch can silence a topic
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(Image, '/camera/color/image_raw',
                                 self._on_frame, qos)
        self.create_subscription(String, '/detections',
                                 self._on_detections, qos)
        self.create_subscription(String, '/object_detections_body',
                                 self._on_body, qos)
        if not self._no_cloud:
            self.create_subscription(PointCloud2, '/object_pointcloud',
                                     self._on_cloud, qos)
        # --odom: pane state
        self._odom_on = bool(getattr(opts, 'odom', False))
        self._odom_topic = getattr(opts, 'odom_topic', ODOM_TOPIC_DEFAULT)
        self._pose = None             # (x, y, yaw, wall)
        self._trail = []              # (wall, x, y, yaw, state)
        self._walk_state, self._walk_label = 'IDLE', None
        self._home_goal = None        # (x, y) world, from RETURNING
        self._markers = []            # x, y, text, colour
        self._odom_csv = None
        self._mwriter = self._mtimes = self._mmeta = None
        self._mframes = 0
        if self._odom_on:
            from nav_msgs.msg import Odometry   # only used here
            self.create_subscription(Odometry, self._odom_topic, self._on_odom, qos)
            self.create_subscription(String, '/walk_status', self._on_walk_status, qos)
            self.create_subscription(String, '/skill_verdict', self._on_verdict, qos)
            self._odom_csv = open(os.path.join(self._snap_dir, 'odom.csv'), 'w')
            self._odom_csv.write('wall_s,x,y,yaw,state\n')
            if opts.record:
                self._open_map_record(os.path.join(
                    os.path.dirname(os.path.abspath(opts.record)), 'map.mp4'))
        self.create_timer(1.0 / self._fps, self._render)

        # self-check 5 s in: DDS env + topics we can see
        # _diag_done keeps it to one report
        self.create_timer(5.0, self._diagnose)
        self._diag_done = False

        # 3D view state
        self._yaw = 0.7
        self._pitch = 0.35
        self._auto_rotate = False
        self._drag = None
        self._last_saved = 0.0

        self.get_logger().info(
            'what_robot_sees live viewer ready — waiting for pipeline topics. '
            'Keys: s=save  r=auto-rotate  q=quit'
            + ('  [no-cloud]' if self._no_cloud else '')
            + ('  [recording %s @ %g fps]' % (opts.record, self._fps)
               if opts.record else ''))

    # ── recording it ─────────────────────────────────────────────────────
    def _make_writer(self, path, size):
        for codec in ('avc1', 'mp4v'):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec),
                                self._fps, size)
            if w.isOpened():
                return w, codec
            w.release()
        raise RuntimeError('cannot open a video writer for %s' % path)

    @staticmethod
    def _write_meta_dict(meta):
        with open(meta['record'] + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=1)

    def _open_record(self, path):
        size = (self._win_w, self._win_h)
        self._writer, codec = self._make_writer(path, size)
        self._times = open(path + '.times.csv', 'w')
        self._times.write('frame,wall_s\n')
        self._meta = {'record': os.path.abspath(path), 'codec': codec,
                      'fps': self._fps, 'size': list(size),
                      'no_cloud': self._no_cloud, 'kind': 'eyes',
                      'start_wall': time.time(), 'frames': 0}
        self._write_meta()

    def _open_map_record(self, path):
        size = (MAP_SIZE, MAP_SIZE)
        self._mwriter, codec = self._make_writer(path, size)
        self._mtimes = open(path + '.times.csv', 'w')
        self._mtimes.write('frame,wall_s\n')
        self._mmeta = {'record': os.path.abspath(path), 'codec': codec,
                       'fps': self._fps, 'size': list(size), 'kind': 'map',
                       'odom_topic': self._odom_topic,
                       'start_wall': time.time(), 'frames': 0}
        self._write_meta_dict(self._mmeta)

    def _write_meta(self):
        if self._meta is not None:
            self._write_meta_dict(self._meta)

    def close_record(self):
        """Flush and close the recordings (idempotent)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._times is not None:
            self._times.close()
            self._times = None
        if self._meta is not None:
            self._meta['end_wall'] = time.time()
            self._meta['frames'] = self._frames
            self._write_meta()
            # print, not ROS log: rclpy is gone by SIGINT
            print('recording closed: %d frames -> %s' %
                  (self._frames, self._meta['record']), file=sys.stderr)
            self._meta = None
        if self._mwriter is not None:
            self._mwriter.release()
            self._mwriter = None
        if self._mtimes is not None:
            self._mtimes.close()
            self._mtimes = None
        if self._mmeta is not None:
            self._mmeta['end_wall'] = time.time()
            self._mmeta['frames'] = self._mframes
            self._write_meta_dict(self._mmeta)
            print('map recording closed: %d frames -> %s' %
                  (self._mframes, self._mmeta['record']), file=sys.stderr)
            self._mmeta = None
        if self._odom_csv is not None:
            self._odom_csv.close()
            self._odom_csv = None
            if self._pose is not None:
                png = os.path.join(self._snap_dir, 'odom_path.png')
                cv2.imwrite(png, self._render_map())
                print('odometry path -> %s (%d trail points)' %
                      (png, len(self._trail)), file=sys.stderr)

    # ── the odometry map ─────────────────────────────────────────────────
    def _on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # yaw from quat, same as walk_node._on_odom
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        now = time.time()
        self._pose = (p.x, p.y, yaw, now)
        last = self._trail[-1] if self._trail else None
        if (last is None or now - last[0] >= 0.5
                or math.hypot(p.x - last[1], p.y - last[2]) >= 0.01
                or abs(_wrap_angle(yaw - last[3])) >= math.radians(1.0)):
            self._trail.append((now, p.x, p.y, yaw, self._walk_state))
            if len(self._trail) > 30000:
                del self._trail[:5000]
            if self._odom_csv is not None:
                self._odom_csv.write('%.3f,%.4f,%.4f,%.4f,%s\n' %
                                     (now, p.x, p.y, yaw, self._walk_state))

    def _on_walk_status(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        state = str(d.get('state', 'IDLE'))
        label = d.get('label')
        detail = d.get('detail') if isinstance(d.get('detail'), dict) else {}
        changed = state != self._walk_state
        self._walk_state, self._walk_label = state, label
        goal = detail.get('goal')
        if state == 'RETURNING' and isinstance(goal, list) and len(goal) == 2:
            try:
                self._home_goal = (float(goal[0]), float(goal[1]))
            except (TypeError, ValueError):
                pass
        if changed and self._pose is not None and state in ('REACHED', 'NOT_FOUND'):
            self._add_marker(state.lower() + (' ' + str(label) if label else ''),
                             WALK_COLORS.get(state, (200, 200, 200)))

    def _add_marker(self, text, col):
        """A marker at the current pose; one that lands within 5 cm of the
        previous marker joins its label (REACHED and the verdict that follows
        it are at the same spot)."""
        x, y = self._pose[0], self._pose[1]
        if self._markers and math.hypot(x - self._markers[-1][0], y - self._markers[-1][1]) < 0.05:
            px, py, ptext, pcol = self._markers[-1]
            self._markers[-1] = (px, py, ptext + ', ' + text, pcol)
            return
        self._markers.append((x, y, text, col))

    def _on_verdict(self, msg):
        text = msg.data
        if not text.startswith('VERDICT:success') or self._pose is None:
            return
        low = text.lower()
        what = ('released' if 'release' in low else
                'picked' if ('lift' in low or 'held' in low or 'grasp' in low) else 'verdict ok')
        self._add_marker(what, (90, 220, 90))

    def _render_map(self):
        """Top-down odometry map: forward (odom x) is UP, left (odom y) is LEFT."""
        panel = np.full((MAP_SIZE, MAP_SIZE, 3), (28, 28, 32), dtype=np.uint8)
        now = time.time()
        if self._pose is None:
            cv2.putText(panel, 'waiting for odometry ...', (20, MAP_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(panel, self._odom_topic, (20, MAP_SIZE // 2 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
            return panel
        px, py, pyaw, pt = self._pose
        pose = (px, py, pyaw)
        pts = [(x, y) for _, x, y, _, _ in self._trail] + [(px, py)]
        if self._home_goal is not None:
            pts.append(self._home_goal)
        pts.extend((m[0], m[1]) for m in self._markers)
        dets = []
        for det in self._body:
            try:
                bx, by = float(det.get('x')), float(det.get('y'))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(bx) and math.isfinite(by)):
                continue
            wx, wy = _body_to_world(pose, bx, by)
            dets.append((wx, wy, str(det.get('label', '?'))))
            pts.append((wx, wy))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        span = max(MAP_MIN_SPAN_M, max(xs) - min(xs) + 1.0, max(ys) - min(ys) + 1.0)
        scale = (MAP_SIZE - 2 * MAP_MARGIN) / span

        def S(wx, wy):
            return (int(round(MAP_SIZE / 2.0 - (wy - cy) * scale)),
                    int(round(MAP_SIZE / 2.0 - (wx - cx) * scale)))

        # grid: 0.5 m cells, brighter per metre
        step = 0.5 if span <= 6.0 else 1.0
        g0 = math.floor((cx - span / 2.0) / step) * step
        g = g0
        while g <= cx + span / 2.0 + step:
            col = (62, 62, 70) if abs(g - round(g)) < 1e-6 else (44, 44, 50)
            cv2.line(panel, S(g, cy - span), S(g, cy + span), col, 1)
            g += step
        g = math.floor((cy - span / 2.0) / step) * step
        while g <= cy + span / 2.0 + step:
            col = (62, 62, 70) if abs(g - round(g)) < 1e-6 else (44, 44, 50)
            cv2.line(panel, S(cx - span, g), S(cx + span, g), col, 1)
            g += step
        # camera wedge
        half = math.radians(CAM_FOV_DEG / 2.0)
        wedge = [S(px, py)] + [S(px + CAM_RANGE_M * math.cos(pyaw + a),
                                 py + CAM_RANGE_M * math.sin(pyaw + a))
                               for a in np.linspace(-half, half, 12)]
        overlay = panel.copy()
        cv2.fillPoly(overlay, [np.array(wedge, dtype=np.int32)], (70, 60, 40))
        cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)
        # trail by state
        prev = None
        for _, x, y, _, state in self._trail:
            cur = S(x, y)
            if prev is not None:
                cv2.line(panel, prev, cur, WALK_COLORS.get(state, (150, 150, 150)), 2, cv2.LINE_AA)
            prev = cur
        # home, markers, detections
        if self._home_goal is not None:
            hx, hy = S(*self._home_goal)
            cv2.rectangle(panel, (hx - 6, hy - 6), (hx + 6, hy + 6), (255, 170, 120), 2)
            cv2.putText(panel, 'home', (hx + 9, hy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 170, 120), 1, cv2.LINE_AA)
        for mx, my, text, col in self._markers[-40:]:
            sx, sy = S(mx, my)
            cv2.circle(panel, (sx, sy), 4, col, -1)
            cv2.putText(panel, text, (sx + 7, sy - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, col, 1, cv2.LINE_AA)
        for wx, wy, label in dets:
            sx, sy = S(wx, wy)
            col = label_color(label)
            cv2.circle(panel, (sx, sy), 5, col, 2)
            cv2.putText(panel, label, (sx + 8, sy + 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, col, 1, cv2.LINE_AA)
        # robot, aimed by yaw
        tip = S(px + 0.14 * math.cos(pyaw), py + 0.14 * math.sin(pyaw))
        bl = S(px + 0.09 * math.cos(pyaw + 2.4), py + 0.09 * math.sin(pyaw + 2.4))
        br = S(px + 0.09 * math.cos(pyaw - 2.4), py + 0.09 * math.sin(pyaw - 2.4))
        cv2.fillPoly(panel, [np.array([tip, bl, br], dtype=np.int32)], (255, 255, 255))
        # header, scale bar, clock
        age = now - pt
        cv2.rectangle(panel, (0, 0), (MAP_SIZE, 40), (36, 36, 44), -1)
        cv2.putText(panel, 'odometry  x=%+.2f y=%+.2f yaw=%+.0f deg  age %.1fs' %
                    (px, py, math.degrees(pyaw), age), (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, 'walk: %s%s   trail %d pts' %
                    (self._walk_state, (' ' + str(self._walk_label)) if self._walk_label else '',
                     len(self._trail)), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    WALK_COLORS.get(self._walk_state, (200, 200, 200)), 1, cv2.LINE_AA)
        bar = int(round(1.0 * scale))
        cv2.line(panel, (12, MAP_SIZE - 14), (12 + bar, MAP_SIZE - 14), (220, 220, 220), 2)
        cv2.putText(panel, '1 m', (12, MAP_SIZE - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, 'forward = up', (MAP_SIZE - 118, MAP_SIZE - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        cv2.putText(panel, _clock_text(now), (MAP_SIZE - 112, MAP_SIZE - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        return panel

    # ── incoming messages ────────────────────────────────────────────────
    def _diagnose(self):
        if self._diag_done:
            return
        self._diag_done = True
        self.get_logger().info(
            'DIAG env: '
            f'ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "(unset=0)")} '
            f'ROS_LOCALHOST_ONLY={os.environ.get("ROS_LOCALHOST_ONLY", "(unset)")} '
            f'CYCLONEDDS_URI={os.environ.get("CYCLONEDDS_URI", "(unset)")}')
        try:
            names = set(n for n, _ in self.get_topic_names_and_types())
        except Exception as e:
            self.get_logger().error(f'DIAG topic scan failed: {e}')
            return
        for t in ('/camera/color/image_raw', '/detections',
                  '/object_detections_body', '/object_pointcloud'):
            self.get_logger().info(
                f'DIAG topic {"VISIBLE" if t in names else "MISSING  "}: {t}')
        self.get_logger().info(
            f'DIAG total topics visible: {len(names)}')
    def _on_frame(self, msg):
        self._frame = msg

    def _on_detections(self, msg):
        try:
            d = json.loads(msg.data)
            if isinstance(d, list):
                self._detections = d
                self._det_time = time.time()
        except Exception:
            pass

    def _on_body(self, msg):
        try:
            d = json.loads(msg.data)
            if isinstance(d, list):
                self._body = d
                self._body_time = time.time()
                with open(os.path.join(self._snap_dir, 'detections.log'), 'a') as f:
                    f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                         msg.data))
        except Exception:
            pass

    def _on_cloud(self, msg):
        try:
            pts = []
            ids = []
            off = 0
            step = msg.point_step
            for _ in range(msg.width):
                x, y, z, oid = struct.unpack_from('<fffI', bytes(msg.data), off)
                pts.append((x, y, z))
                ids.append(oid)
                off += step
            self._cloud_pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            self._cloud_ids = np.asarray(ids, dtype=np.int64)
            self._cloud_time = time.time()
        except Exception:
            pass

    # ── drawing ──────────────────────────────────────────────────────────
