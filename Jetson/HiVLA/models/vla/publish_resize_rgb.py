#!/usr/bin/env python3
"""
publish_resize_rgb.py

Subscribes to ZED RGB, center-crops to square, resizes to 384x384,
and publishes to /hivla/navila_view at 10Hz.

Always running — allows web UI and navila_bridge.py to both consume
the pre-processed image without duplicating crop logic.

Usage (from HiVLA root):
  python3 models/vla/publish_resize_rgb.py
"""

import sys
import os
import time

_HIVLA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _HIVLA_ROOT not in sys.path:
    sys.path.insert(0, _HIVLA_ROOT)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import numpy as np
import cv2


class ResizeRGBNode(Node):
    def __init__(self):
        super().__init__('publish_resize_rgb')
        self.cv_bridge = CvBridge()
        self._last_pub_time = 0.0  # rate-limit to 10Hz

        zed_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.sub_zed = self.create_subscription(
            RosImage,
            '/zed/zed_node/rgb/color/rect/image',
            self._zed_callback,
            zed_qos,
        )

        self.pub_view = self.create_publisher(RosImage, '/hivla/navila_view', 1)

        self.get_logger().info('publish_resize_rgb ready: /zed → /hivla/navila_view 384x384 @10Hz')

    def _zed_callback(self, msg):
        now = time.monotonic()
        if now - self._last_pub_time < 0.1:
            return
        self._last_pub_time = now

        try:
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            h, w = cv_img.shape[:2]
            side = min(h, w)
            top = (h - side) // 2
            left = (w - side) // 2
            cropped = cv_img[top:top + side, left:left + side]
            resized = cv2.resize(cropped, (384, 384), interpolation=cv2.INTER_LINEAR)

            out_msg = self.cv_bridge.cv2_to_imgmsg(resized, encoding='rgb8')
            out_msg.header = msg.header
            self.pub_view.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'resize_rgb error: {e}', throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = ResizeRGBNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
