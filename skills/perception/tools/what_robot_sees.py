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

