"""Read and organize synchronized ROS images/joints, then run extraction and PF.

Images are paired by header timestamps. Each arm's joints and jaw are taken
from bounded histories within joint_sync_slop of the left image. Missing or
stale joint data skips that arm. DEKR still requires a stereo image pair.
Outputs remain in the left camera optical frame. Warmup remains left PnP.
"""

import copy
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from tracking.common.resources import resolve_asset_root
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tracking.messages.synchronization import JointSynchronizer, stamp_seconds

# SHARE_DIR: installed resource directory (Parameter/, pnp_*.yaml, etc.)
SHARE_DIR = str(resolve_asset_root())


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

from tracking import (
    ArmTracker,
    CombinedArmTracker,
    SimpleProjector,
    draw_legend,
    draw_hud,
    draw_main_tool_hud,
    wrist_overlap,
)
from tracking.messages.arm_configs import build_arm_configs
from tracking.common.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.features.keypoint_sources import DekrKeypointProvider
from tracking.features.feature_detection import ArmFeatureDetector
from tracking.common.frame import TrackingInput

try:
    from tracking.features.dekr_live_runner import DekrLiveRunner

    _DEKR_AVAILABLE = True
except ImportError:
    DekrLiveRunner = None
    _DEKR_AVAILABLE = False

# QoS: use best-effort for sensor data to avoid queue buildup
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


from tracking.messages.pose_messages import pose_to_stamped, publish_rectification_tf


class DualArmTrackingNode(Node):

    def __init__(self, *, allow_primary_switch=False, keypoint_provider=None,
                 feature_factory=ArmFeatureDetector, joint_synchronizer=None):
        super().__init__("dual_arm_tracking")

        # --- Parameters ---
        self.declare_parameter("handeye", "")
        self.declare_parameter("arms", ["psm1", "psm2"])
        self.declare_parameter("allow_primary_switch", allow_primary_switch)
        self.declare_parameter("n_particles", 300)
        self.declare_parameter("warmup_frames", 0)
        # Primary selection is passed explicitly by the entry or ROS parameter.
        self.declare_parameter("overlap_trigger_px", 80.0)
        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/image_raw")
        self.declare_parameter("stereo_sync_slop", 0.015)
        self.declare_parameter("joint_sync_slop", 0.030)
        self.declare_parameter("camera_frame", "left_camera_optical_frame")
        self.declare_parameter("rectified_camera_frame", "left_camera_rectified_optical_frame")
        self.declare_parameter("dekr_model_path", "")
        self.declare_parameter("dekr_fp16", True)
        self.declare_parameter("dekr_compile", True)
        # Model's own identity head -- the ONLY identity mechanism in this
        # pipeline now (see dekr_live_runner.py's module docstring). False
        # means no arm identity at all: every detection stays unassigned.
        self.declare_parameter("dekr_use_identity_head", True)
        # Local OpenCV window showing /tracking/visualization, in addition
        # to publishing it -- convenient for a workstation with a display;
        # turn off (true) on a headless robot host with no X server.
        self.declare_parameter("shaft_backend", "elsed")
        self.declare_parameter("no_display", False)
        # The published image is DLC_W x DLC_H (640x480) -- small on a
        # modern monitor, so the preview window is upscaled by this factor.
        self.declare_parameter("display_scale", 2.0)

        arms = self.get_parameter("arms").value
        self._arm_cfgs = build_arm_configs(
            SHARE_DIR, handeye_file=self.get_parameter("handeye").value)
        n_particles = self.get_parameter("n_particles").value
        overlap_trigger_px = self.get_parameter("overlap_trigger_px").value
        self._warmup_frames_needed = self.get_parameter("warmup_frames").value
        image_topic = self.get_parameter("image_topic").value
        right_image_topic = self.get_parameter("right_image_topic").value
        self._cam_frame = self.get_parameter("camera_frame").value
        self._rectified_cam_frame = self.get_parameter("rectified_camera_frame").value
        dekr_model_path = self.get_parameter("dekr_model_path").value
        dekr_fp16 = self.get_parameter("dekr_fp16").value
        dekr_compile = self.get_parameter("dekr_compile").value
        dekr_use_identity_head = self.get_parameter("dekr_use_identity_head").value
        self._display = not self.get_parameter("no_display").value
        self._display_scale = self.get_parameter("display_scale").value
        self._window_name = "Dual-Arm Tracking (q=close window)"
        if self._display:
            try:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            except cv2.error as e:
                self.get_logger().warn(f"Cannot open a display window ({e}) — disabling preview")
                self._display = False

        # --- Perf HUD state (Hz / DEKR inference ms, smoothed) ---
        self._t_prev = None
        self._fps_smooth = 0.0
        self._infer_ms_smooth = 0.0

        # Use the configured provider, or construct the default DEKR backend.
        self._dekr = None
        if keypoint_provider is None:
            if _DEKR_AVAILABLE:
                self._dekr = DekrLiveRunner(
                    model_path=dekr_model_path or None,
                    use_fp16=dekr_fp16,
                    use_compile=dekr_compile,
                    use_identity_head=dekr_use_identity_head,
                )
            else:
                self.get_logger().error(
                    "tracking.features.dekr_live_runner unavailable (dlclive not installed?) — "
                    "tracking is disabled, only camera passthrough will be published."
                )

        # --- Initialize trackers ---
        calibration_file = os.path.join(SHARE_DIR, "Parameter", "stereo_calibration.yaml")
        robot_file = os.path.join(SHARE_DIR, "Parameter", "LND.json")
        projector = SimpleProjector(calibration_file, DLC_W, DLC_H, CAL_W, CAL_H, rectify=False, swap_cameras=True)
        self._projector = projector
        self._rectification_tf = None
        if projector.rectified:
            self._rectification_tf = publish_rectification_tf(
                self, projector, self._cam_frame, self._rectified_cam_frame)
        else:
            self._rectified_cam_frame = self._cam_frame
        K_dlc = projector.K1
        dist_l = projector.dist1

        self._keypoints = keypoint_provider or (DekrKeypointProvider(self._dekr, self._arm_cfgs) if self._dekr is not None else None)

        self._trackers: dict[str, ArmTracker] = {}
        for arm_name in arms:
            if arm_name not in self._arm_cfgs:
                self.get_logger().warn(f"Unknown arm: {arm_name}, skipping")
                continue
            cfg = self._arm_cfgs[arm_name]
            self._trackers[arm_name] = ArmTracker(
                arm_name,
                cfg,
                None,  # dlc_left -- detections are supplied per-call via update(raw_l=...)
                None,  # dlc_right
                K_dlc,
                dist_l,
                projector,
                robot_file,
                shaft_params={"backend": self.get_parameter("shaft_backend").value},
                feature_factory=feature_factory,
            )
            self.get_logger().info(f"[{arm_name}] tracker ready")

        self._combined = CombinedArmTracker(
            self._trackers, n_particles,
            allow_primary_switch=self.get_parameter("allow_primary_switch").value)

        self._K_dlc = K_dlc
        self._dist_l = dist_l
        self._overlap_trigger_px = overlap_trigger_px

        # --- Joint angle cache (thread-safe) ---
        # {arm_name: (js_msg, jaw_msg)}
        self._joint_sync_slop = float(self.get_parameter("joint_sync_slop").value)
        self._joints = joint_synchronizer or JointSynchronizer(self._trackers, self._joint_sync_slop)
        stereo_slop = float(self.get_parameter("stereo_sync_slop").value)
        if not np.isfinite(stereo_slop) or self._joint_sync_slop <= 0 or stereo_slop <= 0:
            raise ValueError("Synchronization tolerances must be positive")

        # --- Warm-up buffering (Algorithm 1 needs a well-spread initial prior;
        # ArmTracker.warmup_live() estimates it via PnP over these buffered
        # frames instead of the narrow default DELTA_PROCESS_STD prior).
        # ONE shared buffer across all not-yet-warmed arms, so one DEKR call
        # per buffered frame can serve every pending arm -- see
        # _run_warmup(). Only frames where the right image was already
        # available get buffered (DekrLiveRunner.infer() is stereo-only). ---
        self._warmup_buffer: list = []  # [(left_img, right_img, {arm_name: joint_angles})]
        self._warmup_done: dict[str, bool] = {
            arm: self._warmup_frames_needed <= 0 for arm in self._trackers
        }

        # --- Publishers ---
        self._pose_pubs: dict[str, Any] = {}
        for arm_name in self._trackers:
            self._pose_pubs[arm_name] = self.create_publisher(
                PoseStamped, f"/tracking/{arm_name}/pose", 10
            )
        self._vis_pub = self.create_publisher(Image, "/tracking/visualization", 1)

        # --- Subscribers ---
        self._bridge = CvBridge()

        # Joint angles (separate callback per arm)
        for arm_name in self._trackers:
            arm_upper = arm_name.upper()  # PSM1 / PSM2
            self.create_subscription(
                JointState,
                f"/{arm_upper}/measured_js",
                lambda msg, a=arm_name: self._js_cb(msg, a, 0),
                _SENSOR_QOS,
            )
            self.create_subscription(
                JointState,
                f"/{arm_upper}/jaw/measured_js",
                lambda msg, a=arm_name: self._js_cb(msg, a, 1),
                _SENSOR_QOS,
            )

        self._left_sub = Subscriber(self, Image, image_topic, qos_profile=_SENSOR_QOS)
        self._right_sub = Subscriber(self, Image, right_image_topic, qos_profile=_SENSOR_QOS)
        self._image_sync = ApproximateTimeSynchronizer(
            [self._left_sub, self._right_sub], queue_size=10, slop=stereo_slop)
        self._image_sync.registerCallback(self._image_cb)
        self.get_logger().info(f"Synchronized stereo: {image_topic} / {right_image_topic}")

    def _js_cb(self, msg: JointState, arm_name: str, slot: int):
        self._joints.add(arm_name, slot, stamp_seconds(msg), msg.position)

    def _publish_vis(self, vis: np.ndarray, header, show_hud: bool = False, overlap_px=None):
        """Draw the perf HUD (when show_hud) + legend + main-tool/overlap
        status, publish to /tracking/visualization, and (if enabled) show
        it in a local OpenCV preview window, upscaled by display_scale.

        The published topic always stays at native DLC_W x DLC_H -- only
        the local preview window is upscaled (cv2.resizeWindow() alone
        doesn't reliably resize an empty WINDOW_NORMAL window on every
        platform/window-manager combo; resizing the image itself is not
        subject to that).
        """
        if show_hud:
            draw_hud(vis, self._fps_smooth, self._infer_ms_smooth)
            draw_main_tool_hud(
                vis, self._combined.last_relative_main, overlap_px, trigger_px=self._overlap_trigger_px
            )
        draw_legend(vis)
        try:
            vis_msg = self._bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            vis_msg.header = copy.deepcopy(header)
            vis_msg.header.frame_id = self._rectified_cam_frame
            self._vis_pub.publish(vis_msg)
        except Exception as e:
            self.get_logger().error(f"vis publish error: {e}")
        if self._display:
            preview = vis
            if self._display_scale != 1.0:
                preview = cv2.resize(vis, None, fx=self._display_scale, fy=self._display_scale)
            cv2.imshow(self._window_name, preview)
            cv2.waitKey(1)

    def _run_warmup(self, ready: list):
        """Infer each buffered stereo pair once for all ready arms present in it."""
        self.get_logger().info(
            f"warm-up: running DEKR inference over {len(self._warmup_buffer)} buffered frames "
            f"for {ready} -- first call includes model load"
            + (" + torch.compile (~30-60s)" if getattr(self._dekr, "use_compile", False) else "")
            + "..."
        )
        per_arm_frames: dict[str, list] = {a: [] for a in ready}
        for i, (left_img, right_img, ja_by_arm) in enumerate(self._warmup_buffer):
            present = [a for a in ready if a in ja_by_arm]
            if not present:
                continue
            detections = self._keypoints.infer(left_img, right_img)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(self._warmup_buffer):
                self.get_logger().info(
                    f"warm-up: {i + 1}/{len(self._warmup_buffer)} frames inferred"
                )
            for arm_name in present:
                pair = detections.get(arm_name)
                raw_l = pair.left if pair is not None else {}
                per_arm_frames[arm_name].append((left_img, ja_by_arm[arm_name], raw_l))

        for arm_name in ready:
            self.get_logger().info(
                f"[{arm_name}] warm-up: {len(per_arm_frames[arm_name])} frames, initializing..."
            )
        self._combined.warmup_live({a: per_arm_frames[a] for a in ready})
        for arm_name in ready:
            self._warmup_done[arm_name] = True

    def _prepare_warmup(self, left_img, right_img, joint_angles_by_arm):
        """Buffer pending arms and remove those not yet ready from this update."""
        pending = [a for a in self._trackers if not self._warmup_done[a]]
        if pending:
            present = {a: joint_angles_by_arm[a] for a in pending if a in joint_angles_by_arm}
            if present:
                self._warmup_buffer.append((left_img.copy(), right_img.copy(), present))

            ready = [
                a
                for a in pending
                if sum(1 for _, _, ja in self._warmup_buffer if a in ja)
                >= self._warmup_frames_needed
            ]
            if ready:
                self._run_warmup(ready)
                if all(self._warmup_done[a] for a in self._trackers):
                    self._warmup_buffer = []

            # An arm becomes eligible for tracking on the tick it finishes warmup.
            for a in pending:
                if a not in ready:
                    joint_angles_by_arm.pop(a, None)


    def _image_cb(self, img_msg: Image, right_msg: Image):
        try:
            left_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            right_img = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="bgr8")
            left_img, right_img = self._projector.process_images(left_img, right_img)
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        vis = left_img.copy()

        if self._keypoints is None:
            self._publish_vis(vis, img_msg.header)
            return

        # Match each arm independently: a missing arm must not block the other.
        timestamp = stamp_seconds(img_msg)
        joint_angles_by_arm = self._joints.snapshot(timestamp)
        for arm_name in self._trackers:
            if arm_name not in joint_angles_by_arm:
                self.get_logger().warn(
                    f"[{arm_name}] missing, stale or invalid joints/jaw, skipping arm",
                    throttle_duration_sec=2.0)

        self._prepare_warmup(left_img, right_img, joint_angles_by_arm)

        # Share one provider result across arms, then update the joint PF.
        if joint_angles_by_arm:
            t0 = time.perf_counter()
            detections = self._keypoints.infer(left_img, right_img)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            self._infer_ms_smooth = (
                infer_ms if self._t_prev is None else 0.9 * self._infer_ms_smooth + 0.1 * infer_ms
            )
            t_now = time.perf_counter()
            if self._t_prev is not None:
                dt = t_now - self._t_prev
                self._fps_smooth = 0.9 * self._fps_smooth + 0.1 / max(dt, 1e-6)
            self._t_prev = t_now

            per_arm = {}
            for arm_name, joints in joint_angles_by_arm.items():
                pair = detections.get(arm_name)
                per_arm[arm_name] = dict(joint_angles=joints,
                                        raw_l=pair.left if pair is not None else {},
                                        raw_r=pair.right if pair is not None else {})
            results = TrackingInput(left_image=left_img, right_image=right_img,
                                    per_arm=per_arm).update(self._combined)

            for arm_name, frame in results.items():
                if not frame.available:
                    continue
                tr = self._trackers[arm_name]
                tr.draw(vis, frame.corrected_keypoints)

                # Publish pose
                pose_msg = pose_to_stamped(self._projector.to_raw_left_pose(tr.cam_T_b), img_msg.header.stamp, self._cam_frame)
                self._pose_pubs[arm_name].publish(pose_msg)

                self.get_logger().debug(f"[{arm_name}] pts={frame.observation_count}")

        overlap_px = None
        if len(self._trackers) == 2 and self._t_prev is not None:
            name_a, name_b = list(self._trackers.keys())
            overlap = wrist_overlap(
                self._trackers[name_a], self._trackers[name_b], self._K_dlc, self._dist_l
            )
            overlap_px = overlap[0] if overlap is not None else None

        # show_hud only once real inference has run at least once (self._t_prev
        # is set there) -- otherwise fps/infer_ms are stale/zero and misleading.
        self._publish_vis(
            vis, img_msg.header, show_hud=self._t_prev is not None, overlap_px=overlap_px
        )

    def destroy_node(self):
        if self._display:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None, *, allow_primary_switch=False):
    rclpy.init(args=args)
    node = None
    try:
        node = DualArmTrackingNode(allow_primary_switch=allow_primary_switch)
        rclpy.spin(node)
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
