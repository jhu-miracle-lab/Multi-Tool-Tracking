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
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

SHARE_DIR = get_package_share_directory("surgical_tool_tracking_ros2")


def _resolve_src_dir() -> str:
    if "STT_SRC_DIR" in os.environ:
        return os.path.normpath(os.environ["STT_SRC_DIR"])

    candidate = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(candidate, "tracking")):
        return candidate
    raise RuntimeError(
        "Cannot locate source directory. "
        "Set the STT_SRC_DIR environment variable to the package root "
        "(the folder containing tracking/, dexined/, etc.)."
    )


SRC_DIR = _resolve_src_dir()
sys.path.insert(0, SRC_DIR)

from tracking.visualization import draw_legend, draw_main_tool_hud
from tracking.main_tool import wrist_overlap
from tracking.constants import DLC_W, DLC_H
from tracking.offline_session import OfflineSession
from ros2_node.pose_messages import pose_to_stamped


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


class OfflineTrackingNode(Node):

    def __init__(self, *, allow_primary_switch=False):
        super().__init__("offline_tracking")

        self.declare_parameter("data", "")
        self.declare_parameter("no_display", False)
        self.declare_parameter("save_states", "")
        self.declare_parameter("camera_frame", "left_camera_optical_frame")
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
        n_particles = self.get_parameter("n_particles").value
        handeye = self.get_parameter("handeye").value
        overlap_trigger_px = self.get_parameter("overlap_trigger_px").value

        if not data_dir:
            self.get_logger().error('Parameter "data" is required.')
            raise RuntimeError("data parameter not set")

        session = OfflineSession(
            SHARE_DIR, data_dir, n_particles, handeye, log=self.get_logger().info,
            allow_primary_switch=self.get_parameter("allow_primary_switch").value)
        trackers = session.trackers
        combined = session.combined
        left_img_dir = session.left_img_dir

        # --- Publishers ---
        self._bridge = CvBridge()
        self._pose_pubs = {
            arm: self.create_publisher(PoseStamped, f"/tracking/{arm}/pose", 10) for arm in trackers
        }
        self._vis_pub = self.create_publisher(Image, "/tracking/visualization", 1)

        session.warmup()

        # Output directory
        method = "APRT" if self.get_parameter("allow_primary_switch").value else "FPRT"
        out_base = os.path.join(SRC_DIR, method, "tracking_output_offline_ros2")
        os.makedirs(out_base, exist_ok=True)
        out_img_dir = os.path.join(out_base, "frames")
        os.makedirs(out_img_dir, exist_ok=True)
        out_video_name = f"{os.path.basename(os.path.normpath(data_dir))}_tracking.mp4"
        out_video_path = os.path.join(out_base, out_video_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (DLC_W, DLC_H))

        # --- Main loop ---
        for frame_idx in session.frame_indices:
            if not rclpy.ok():
                break

            left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
            if left_img is None:
                continue

            vis_l = left_img.copy()
            t0 = time.time()

            # ROS2 timestamp
            now = self.get_clock().now().to_msg()

            results = combined.update(frame_idx, left_img=left_img)
            statuses = session.draw_and_record(
                vis_l, results, frame_idx, save_states=bool(save_states))
            for name, tracker in trackers.items():
                if results[name].available:
                    self._pose_pubs[name].publish(
                        pose_to_stamped(tracker.cam_T_b, now, cam_frame))

            elapsed = time.time() - t0
            self.get_logger().info(f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}')

            overlap_px = None
            if len(trackers) == 2:
                name_a, name_b = list(trackers.keys())
                overlap = wrist_overlap(trackers[name_a], trackers[name_b], session.K, session.dist)
                overlap_px = overlap[0] if overlap is not None else None

            draw_legend(vis_l)
            draw_main_tool_hud(vis_l, combined.last_relative_main, overlap_px, trigger_px=overlap_trigger_px)
            out_video.write(vis_l)
            cv2.imwrite(os.path.join(out_img_dir, f"{frame_idx:06d}.png"), vis_l)

            # Publish visualization image
            try:
                vis_msg = self._bridge.cv2_to_imgmsg(vis_l, encoding="bgr8")
                vis_msg.header.stamp = now
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

        out_video.release()
        if not no_display:
            cv2.destroyAllWindows()
        self.get_logger().info(f"Video saved: {out_video_path}")

        if save_states:
            session.save_states(save_states)


def main(args=None, *, allow_primary_switch=False):
    rclpy.init(args=args)
    node = OfflineTrackingNode(allow_primary_switch=allow_primary_switch)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
