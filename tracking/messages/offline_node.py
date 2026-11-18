"""
ROS2 offline tracking node (reads data from files).

Same logic as tracking/offline_tracking.py, plus publishing the
tracking results as ROS2 topics.

Publishes:
  /tracking/psm1/pose        geometry_msgs/PoseStamped
  /tracking/psm2/pose        geometry_msgs/PoseStamped
  /tracking/visualization    sensor_msgs/Image

Usage:
  ros2 run surgical_tool_tracking_ros2 offline_node \
      --ros-args -p data:=/path/to/dataset

Tracking uses one shared 12-state PF. Primary selection and mode transitions
are handled internally by CombinedArmTracker using tracker_config.
"""

import os
import sys
import time

import cv2
import rclpy
from tracking.common.resources import resolve_asset_root
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

SHARE_DIR = str(resolve_asset_root())


def _resolve_src_dir() -> str:
    if "STT_SRC_DIR" in os.environ:
        return os.path.normpath(os.environ["STT_SRC_DIR"])

    candidate = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.isdir(os.path.join(candidate, "tracking")):
        return candidate
    raise RuntimeError(
        "Cannot locate source directory. "
        "Set the STT_SRC_DIR environment variable to the package root "
        "(the folder containing tracking/, dexined/, etc.)."
    )


SRC_DIR = _resolve_src_dir()
sys.path.insert(0, SRC_DIR)

from tracking.common.visualization import draw_legend, draw_main_tool_hud
from tracking.pf.main_tool import wrist_overlap
from tracking.common.constants import DLC_W, DLC_H
from tracking.messages.offline_session import OfflineSession
from tracking.messages.tracking_output import TrackingOutput
from tracking.messages.pose_messages import pose_to_stamped, publish_rectification_tf


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


class OfflineTrackingNode(Node):

    def __init__(self, *, allow_primary_switch=False):
        super().__init__("offline_tracking")

        self.declare_parameter("data", "")
        self.declare_parameter("out_dir", "")
        self.declare_parameter("shaft_backend", "elsed")
        self.declare_parameter("no_display", False)
        self.declare_parameter("save_states", "")
        self.declare_parameter("camera_frame", "left_camera_optical_frame")
        self.declare_parameter("rectified_camera_frame", "left_camera_rectified_optical_frame")
        self.declare_parameter("allow_primary_switch", allow_primary_switch)
        self.declare_parameter("n_particles", 500)
        self.declare_parameter("handeye", "")
        self.declare_parameter("overlap_trigger_px", 80.0)

    def run(self):
        """Process the dataset and publish poses/images after node initialization."""
        data_dir = self.get_parameter("data").value
        no_display = self.get_parameter("no_display").value
        save_states = self.get_parameter("save_states").value
        cam_frame = self.get_parameter("camera_frame").value
        rectified_frame = self.get_parameter("rectified_camera_frame").value
        n_particles = self.get_parameter("n_particles").value
        handeye = self.get_parameter("handeye").value
        overlap_trigger_px = self.get_parameter("overlap_trigger_px").value

        if not data_dir:
            self.get_logger().error('Parameter "data" is required.')
            raise RuntimeError("data parameter not set")

        session = OfflineSession(
            SHARE_DIR, data_dir, n_particles, handeye, log=self.get_logger().info,
            allow_primary_switch=self.get_parameter("allow_primary_switch").value, rectify=False,
            shaft_backend=self.get_parameter("shaft_backend").value)
        self._rectification_tf = None
        if session.projector.rectified:
            self._rectification_tf = publish_rectification_tf(
                self, session.projector, cam_frame, rectified_frame)
        else:
            rectified_frame = cam_frame
        trackers = session.trackers
        combined = session.combined

        # --- Publishers ---
        self._bridge = CvBridge()
        self._pose_pubs = {
            arm: self.create_publisher(PoseStamped, f"/tracking/{arm}/pose", 10) for arm in trackers
        }
        self._vis_pub = self.create_publisher(Image, "/tracking/visualization", 1)

        session.warmup()

        # Output directory
        method = "APRT" if self.get_parameter("allow_primary_switch").value else "FPRT"
        with TrackingOutput(data_dir, method, self.get_parameter("out_dir").value,
                            ros=True, display=not no_display, size=(DLC_W, DLC_H)) as output:
            for frame_idx in session.frame_indices:
                if not rclpy.ok():
                    break

                input_frame = session.read_frame(frame_idx)
                left_img = input_frame.left_image
                if left_img is None:
                    continue

                vis_l = left_img.copy()
                t0 = time.time()

                # ROS2 timestamp
                now = self.get_clock().now().to_msg()

                results = input_frame.update(combined)
                statuses = session.draw_and_record(
                    vis_l, results, frame_idx, save_states=bool(save_states))
                for name, tracker in trackers.items():
                    if results[name].available:
                        self._pose_pubs[name].publish(
                            pose_to_stamped(session.projector.to_raw_left_pose(tracker.cam_T_b), now, cam_frame))

                elapsed = time.time() - t0
                self.get_logger().info(f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}')

                overlap_px = None
                if len(trackers) == 2:
                    name_a, name_b = list(trackers.keys())
                    overlap = wrist_overlap(trackers[name_a], trackers[name_b], session.K, session.dist)
                    overlap_px = overlap[0] if overlap is not None else None

                draw_legend(vis_l)
                draw_main_tool_hud(vis_l, combined.last_relative_main, overlap_px, trigger_px=overlap_trigger_px)
                output.write(frame_idx, vis_l)

                # Publish visualization image
                try:
                    vis_msg = self._bridge.cv2_to_imgmsg(vis_l, encoding="bgr8")
                    vis_msg.header.stamp = now
                    vis_msg.header.frame_id = rectified_frame
                    self._vis_pub.publish(vis_msg)
                except Exception:
                    pass

                if not no_display:
                    cv2.imshow("Offline tracking (q=quit)", vis_l)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key != 255:
                        cv2.waitKey(0)

        self.get_logger().info(f"Video saved: {output.video_path}")

        if save_states:
            session.save_states(save_states)


def main(args=None, *, allow_primary_switch=False):
    rclpy.init(args=args)
    node = None
    try:
        node = OfflineTrackingNode(allow_primary_switch=allow_primary_switch)
        node.run()
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
