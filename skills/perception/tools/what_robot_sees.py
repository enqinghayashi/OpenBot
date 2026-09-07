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


