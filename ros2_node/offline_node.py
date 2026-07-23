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

"""

import json
import os
import sys
import time

import cv2
import numpy as np
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

from tracking import ArmTracker, SimpleProjector, draw_legend
from tracking.arm_configs import build_arm_configs
from tracking.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.dlc_csv import load_dlc_csv


def _pose_to_stamped(cam_T_b, stamp, frame_id):
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    R = cam_T_b[:3, :3]
    t = cam_T_b[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    axis = rvec.flatten() / angle if angle > 1e-9 else np.array([0.0, 0.0, 1.0])
    s = np.sin(angle / 2)
    msg.pose.orientation.x = float(axis[0] * s)
    msg.pose.orientation.y = float(axis[1] * s)
    msg.pose.orientation.z = float(axis[2] * s)
    msg.pose.orientation.w = float(np.cos(angle / 2))
    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])
    return msg


# ---------------------------------------------------------------------------
# Main node
# ---------------------------------------------------------------------------


class OfflineTrackingNode(Node):

    def __init__(self):
        super().__init__("offline_tracking")

        self.declare_parameter("data", "")
        self.declare_parameter("no_display", False)
        self.declare_parameter("save_states", "")
        self.declare_parameter("camera_frame", "left_camera_optical_frame")
        self.declare_parameter("n_particles", 500)
        self.declare_parameter("handeye", "")

        data_dir = self.get_parameter("data").value
        no_display = self.get_parameter("no_display").value
        save_states = self.get_parameter("save_states").value
        cam_frame = self.get_parameter("camera_frame").value
        n_particles = self.get_parameter("n_particles").value
        handeye = self.get_parameter("handeye").value

        if not data_dir:
            self.get_logger().error('Parameter "data" is required.')
            raise RuntimeError("data parameter not set")

        # Auto-detect this dataset's own PnP hand-eye result when not given
        # explicitly (same convention as tracking/offline_tracking.py).
        if not handeye:
            cand = os.path.join(data_dir, "pnp_handeye_manual.yaml")
            if os.path.exists(cand):
                handeye = cand

        # --- Paths ---
        robot_file = os.path.join(SHARE_DIR, "Parameter", "LND.json")
        for up in range(4):
            cand = os.path.normpath(os.path.join(data_dir, *([".."] * up), "LND.json"))
            if os.path.exists(cand):
                robot_file = cand
                break
        intrinsics_npz = os.path.join(SHARE_DIR, "Parameter", "intrinsics.npz")
        left_img_dir = os.path.join(data_dir, "regular/image/left_resized")
        right_img_dir = os.path.join(data_dir, "regular/image/right_resized")

        projector = SimpleProjector(intrinsics_npz, DLC_W, DLC_H, CAL_W, CAL_H)
        K_dlc = projector.K1
        dist_l = np.load(intrinsics_npz)["dist_left"].flatten().astype(np.float64)

        arm_cfgs = build_arm_configs(SHARE_DIR, data_dir, handeye_file=handeye)
        trackers = {}
        for arm_name, cfg in arm_cfgs.items():
            if not os.path.isdir(cfg["kin_dir"]):
                self.get_logger().warn(f"[{arm_name}] kin_dir not found, skipping")
                continue
            csv_l = os.path.join(left_img_dir, cfg["csv_name"])
            if not os.path.exists(csv_l):
                self.get_logger().warn(f"[{arm_name}] CSV not found: {csv_l}, skipping")
                continue
            dlc_left = load_dlc_csv(csv_l, cfg["label_remap"])
            dlc_right = None
            csv_r = os.path.join(right_img_dir, cfg["csv_name"])
            if os.path.exists(csv_r):
                dlc_right = load_dlc_csv(csv_r, cfg["label_remap"])

            # Per-dataset tuned shaft-edge params (from test_canny_hough.py's
            # interactive tuner), if this dataset has one -- falls back to
            # ArmTracker's built-in defaults otherwise (same as
            # tracking/offline_tracking.py).
            shaft_params = None
            shaft_params_path = os.path.join(data_dir, f"canny_params_{arm_name}.json")
            if os.path.exists(shaft_params_path):
                with open(shaft_params_path) as f:
                    shaft_params = json.load(f)
                shaft_params.pop("arm", None)

            trackers[arm_name] = ArmTracker(
                arm_name,
                cfg,
                dlc_left,
                dlc_right,
                K_dlc,
                dist_l,
                projector,
                robot_file,
                n_particles=n_particles,
                shaft_params=shaft_params,
            )
            self.get_logger().info(f"[{arm_name}] ready — {len(dlc_left)} frames")

        if not trackers:
            raise RuntimeError("No trackers initialised.")

        # --- Publishers ---
        self._bridge = CvBridge()
        self._pose_pubs = {
            arm: self.create_publisher(PoseStamped, f"/tracking/{arm}/pose", 10) for arm in trackers
        }
        self._vis_pub = self.create_publisher(Image, "/tracking/visualization", 1)

        # --- Frame indices ---
        all_kin = set()
        for tr in trackers.values():
            all_kin |= {
                int(os.path.splitext(f)[0]) for f in os.listdir(tr.kin_dir) if f.endswith(".json")
            }
        frame_indices = sorted(
            i for i in all_kin if os.path.exists(os.path.join(left_img_dir, f"{i}.png"))
        )
        self.get_logger().info(f"{len(frame_indices)} frames, arms={list(trackers.keys())}")

        # --- Warm-up ---
        warmup_n = min(50, len(frame_indices))
        if warmup_n > 0:
            self.get_logger().info(f"Warm-up: {warmup_n} frames...")
            for tr in trackers.values():
                tr.warmup(frame_indices[:warmup_n], lkl_thresh=0.05, left_img_dir=left_img_dir)
            self.get_logger().info("Warm-up done.")

        # Output directory
        out_base = os.path.join(SRC_DIR, "tracking_output_offline_ros2")
        os.makedirs(out_base, exist_ok=True)
        out_img_dir = os.path.join(out_base, "frames")
        os.makedirs(out_img_dir, exist_ok=True)
        out_video_name = f"{os.path.basename(os.path.normpath(data_dir))}_tracking.mp4"
        out_video_path = os.path.join(out_base, out_video_name)
        state_log = {arm: {"frames": [], "ema_states": [], "fk_tips": []} for arm in trackers}

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (DLC_W, DLC_H))

        # --- Main loop ---
        for frame_idx in frame_indices:
            if not rclpy.ok():
                break

            left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
            if left_img is None:
                continue

            vis_l = left_img.copy()
            t0 = time.time()
            statuses = []

            # ROS2 timestamp
            now = self.get_clock().now().to_msg()

            for arm_name, tr in trackers.items():
                corrected, n_pts, _ = tr.update(frame_idx, left_img=left_img)
                if corrected is None:
                    continue
                tr.draw(vis_l, corrected)
                label = f" [{tr._last_state_label}]" if tr._last_state_label else ""
                statuses.append(f"{arm_name}: pts={n_pts}{label}")

                # Publish pose
                self._pose_pubs[arm_name].publish(_pose_to_stamped(tr.cam_T_b, now, cam_frame))

                if save_states:
                    tip_base = tr.robot_arm.baseToJointT[-3][:3, 3]
                    tip_cam = (tr.cam_T_b_kin @ np.append(tip_base, 1.0))[:3]
                    state_log[arm_name]["frames"].append(frame_idx)
                    # key kept as 'ema_states' for backward-compat with downstream
                    # readers; the filter itself no longer does EMA smoothing —
                    # this is the raw per-frame PF weighted-mean (tr.pose_state).
                    state_log[arm_name]["ema_states"].append(tr.pose_state.copy())
                    state_log[arm_name]["fk_tips"].append(tip_cam.copy())

            elapsed = time.time() - t0
            self.get_logger().info(f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}')

            draw_legend(vis_l)
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
            save_data = {}
            for arm, log in state_log.items():
                if log["frames"]:
                    save_data[f"{arm}_frames"] = np.array(log["frames"])
                    save_data[f"{arm}_ema_states"] = np.array(log["ema_states"])
                    save_data[f"{arm}_fk_tips"] = np.array(log["fk_tips"])
            np.savez(save_states, **save_data)
            self.get_logger().info(f"States saved: {save_states}")


def main(args=None):
    rclpy.init(args=args)
    node = OfflineTrackingNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
