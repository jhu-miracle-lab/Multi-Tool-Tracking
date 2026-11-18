"""
ROS2 dvrk_display stereo unix-socket capture + publisher.

Reads the FULL stereo side-by-side frame from dvrk_display's GStreamer
unix socket (tracking/dvrk_stereo_reader.py), splits left/right, resizes
each eye down to DLC_W x DLC_H (640x480 -- the pixel space every other
node/tool in this package assumes images arrive in, see
tracking/stereo.py's SimpleProjector), and publishes both with matching
header stamps so downstream nodes (tracking_node.py) get a genuinely
synchronized stereo pair.

Publishes:
  /camera/left/image_raw   sensor_msgs/Image  (bgr8, DLC_W x DLC_H)
  /camera/right/image_raw  sensor_msgs/Image  (bgr8, DLC_W x DLC_H)

Usage:
  ros2 run surgical_tool_tracking_ros2 dvrk_stereo_publisher

  or via the launch file:
  ros2 launch surgical_tool_tracking_ros2 live_dvrk.launch.py
"""

import os
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def _resolve_src_dir() -> str:
    if "STT_SRC_DIR" in os.environ:
        return os.path.normpath(os.environ["STT_SRC_DIR"])
    candidate = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(candidate, "tracking")):
        return candidate
    raise RuntimeError(
        "Cannot locate source directory. "
        "Set the STT_SRC_DIR environment variable to the package root."
    )


SRC_DIR = _resolve_src_dir()
sys.path.insert(0, SRC_DIR)

from tracking.common.constants import DLC_W, DLC_H
from tracking.messages.dvrk_stereo_reader import default_socket_path, open_threaded_stereo_reader


class DvrkStereoPublisher(Node):

    def __init__(self):
        super().__init__("dvrk_stereo_publisher")

        self.declare_parameter("socket_path", "")
        self.declare_parameter("left_topic", "/camera/left/image_raw")
        self.declare_parameter("right_topic", "/camera/right/image_raw")

        socket_path = self.get_parameter("socket_path").value or default_socket_path()
        left_topic = self.get_parameter("left_topic").value
        right_topic = self.get_parameter("right_topic").value

        self._bridge = CvBridge()
        self._left_pub = self.create_publisher(Image, left_topic, 1)
        self._right_pub = self.create_publisher(Image, right_topic, 1)

        self.get_logger().info(f"Opening dvrk stereo socket: {socket_path}")
        self._reader, self._full_w, self._full_h = open_threaded_stereo_reader(socket_path, clock=lambda: self.get_clock().now().to_msg())
        self.get_logger().info(
            f"Publishing {DLC_W}x{DLC_H} per eye on {left_topic} / {right_topic}"
        )

    def spin_capture_loop(self):
        """Blocking capture -> publish loop. ThreadedReader.read() already
        blocks efficiently on the next frame, so a manual loop (matching
        offline_node.py's style) is used instead of a timer callback."""
        half = self._full_w // 2
        while rclpy.ok():
            ok, frame, stamp = self._reader.read()
            if not ok:
                self.get_logger().error("Stereo socket read failed, stopping.")
                break

            left = cv2.resize(frame[:, :half], (DLC_W, DLC_H))
            right = cv2.resize(frame[:, half:], (DLC_W, DLC_H))

            # Same stamp on both -- they were split from one captured
            # frame, so this is exact, not approximate.
            # Use reader receipt time, before queueing; not delayed publish time.

            left_msg = self._bridge.cv2_to_imgmsg(left, encoding="bgr8")
            left_msg.header.stamp = stamp
            left_msg.header.frame_id = "left_camera_optical_frame"
            self._left_pub.publish(left_msg)

            right_msg = self._bridge.cv2_to_imgmsg(right, encoding="bgr8")
            right_msg.header.stamp = stamp
            right_msg.header.frame_id = "right_camera_optical_frame"
            self._right_pub.publish(right_msg)

            rclpy.spin_once(self, timeout_sec=0.0)

    def destroy_node(self):
        self._reader.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = DvrkStereoPublisher()
        node.spin_capture_loop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
