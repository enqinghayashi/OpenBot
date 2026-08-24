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


class OpenBotCamera(Node):
    """Owns the RealSense pipeline; a worker thread publishes framesets."""

    def __init__(self, pipeline_factory=None, config_factory=None):
        super().__init__(NODE_NAME)
        self.color_pub = self.create_publisher(Image, COLOR_TOPIC, QUEUE_DEPTH)
        self.depth_pub = self.create_publisher(Image, DEPTH_TOPIC, QUEUE_DEPTH)
        self.framesets = 0
        self._stop = threading.Event()

        self.pipeline = (pipeline_factory or rs.pipeline)()
        cfg = (config_factory or rs.config)()
        cfg.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1],
                          rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1],
                          rs.format.z16, FPS)
        self.pipeline.start(cfg)
        self.get_logger().info(
            'OpenBot camera started: colour %dx%d bgr8 + depth %dx%d z16 @ %d fps '
            '(depth published every %d framesets)'
            % (COLOR_SIZE + DEPTH_SIZE + (FPS, DEPTH_EVERY_N)))

        self._worker = threading.Thread(target=self._publish_loop, daemon=True)
        self._worker.start()
        self.get_logger().info('OpenBot camera publishing on %s and %s'
                               % (COLOR_TOPIC, DEPTH_TOPIC))

    def publish_frameset(self, frames):
        """Publish one frameset; False when it lacks a colour or depth frame."""
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            return False
        stamp = self.get_clock().now().to_msg()
        self.framesets += 1
        self.color_pub.publish(to_image_msg(
            np.asanyarray(color.get_data()), stamp, 'bgr8', 3))
        if self.framesets % DEPTH_EVERY_N == 0:
            self.depth_pub.publish(to_image_msg(
                np.asanyarray(depth.get_data()), stamp, '16UC1', 2))
        return True

    def _publish_loop(self):
        while not self._stop.is_set():
            try:
                self.publish_frameset(
                    self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS))
            except Exception as exc:     # timeout / USB hiccup: keep going
                if not self._stop.is_set():
                    self.get_logger().error('OpenBot camera frame error: %s' % exc)

    def destroy_node(self):
        self._stop.set()
        try:
            self.pipeline.stop()
        except Exception as exc:
            self.get_logger().error('OpenBot camera stop error: %s' % exc)
        super().destroy_node()
