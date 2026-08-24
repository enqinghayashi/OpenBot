#!/usr/bin/env python3
"""openbot_camera_node.py -- OpenBot's RealSense publisher (colour + raw depth).

Publishes the two image streams the OpenBot perception nodes subscribe to:

  /camera/color/image_raw   sensor_msgs/Image  bgr8   424x240  every frameset   (30 Hz)
  /camera/depth/image_raw   sensor_msgs/Image  16UC1  480x270  every 3rd frameset (10 Hz)

Both messages of one frameset carry the same stamp (the node clock when the
frameset arrives) and frame_id 'camera'; publisher queue depth 10.  Depth is
the sensor's NATIVE stream, not registered to the colour grid:
depth_projector_node.py maps between the two resolutions itself, so this
stream contract must not change without changing the projector.

A missing or busy camera makes the start fail loudly (an exception at start,
in camera.log); a frame timeout or a dropped frameset is logged and the loop
keeps going.

Run directly (no ROS package to build):
    python3 openbot_camera_node.py
Needs rclpy + sensor_msgs (ROS 2), numpy, pyrealsense2.
"""

import threading

import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

NODE_NAME = 'openbot_camera'
COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/depth/image_raw'
FRAME_ID = 'camera'
QUEUE_DEPTH = 10

COLOR_SIZE = (424, 240)        # (w, h), bgr8
DEPTH_SIZE = (480, 270)        # w,h; z16 -> 16UC1
FPS = 30
DEPTH_EVERY_N = 3              # publish depth every Nth frameset
FRAME_TIMEOUT_MS = 5000


def to_image_msg(pixels, stamp, encoding, bytes_per_pixel):
    """A sensor_msgs/Image for one HxW (or HxWx3) uint array."""
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = FRAME_ID
    msg.height = int(pixels.shape[0])
    msg.width = int(pixels.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = int(pixels.shape[1]) * bytes_per_pixel
    msg.data = pixels.tobytes()
    return msg
