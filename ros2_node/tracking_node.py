"""
ROS2 dual-arm surgical tool tracking node.

Subscribes:
  /camera/left/image_raw          sensor_msgs/Image
  /camera/right/image_raw         sensor_msgs/Image   (REQUIRED -- the DEKR
                                                         live model is stereo-
                                                         only, see
                                                         dekr_live_runner.py;
                                                         frames are skipped
                                                         entirely until both
                                                         views are available)
  /PSM1/measured_js               sensor_msgs/JointState
  /PSM1/jaw/measured_js           sensor_msgs/JointState
  /PSM2/measured_js               sensor_msgs/JointState
  /PSM2/jaw/measured_js           sensor_msgs/JointState

Publishes:
  /tracking/psm1/pose             geometry_msgs/PoseStamped
  /tracking/psm2/pose             geometry_msgs/PoseStamped
  /tracking/visualization         sensor_msgs/Image  (annotated left camera image)

Parameters:
  ~arms            [psm1, psm2]   arms to track
  ~n_particles     300            number of particles (all arms share ONE
                                  6*n_arms-state filter, see
                                  tracking/combined_tracker.py)
  ~warmup_frames   0              frames to collect for PF warm-up at startup (0=skip)
  ~overlap_trigger_px 80.0        display-only overlap warning threshold
  ~image_topic     /camera/left/image_raw
  ~right_image_topic  /camera/right/image_raw
  ~dekr_model_path ""             DEKR live model path override (default: DekrLiveRunner's
                                  own DEKR_LIVE_MODEL_PATH env var / built-in default)
  ~dekr_fp16       True           enable FP16 autocast for the DEKR live model
  ~dekr_compile    True           enable torch.compile for the DEKR live model
  ~dekr_use_identity_head True    resolve psm1/psm2 identity via the model's identity head
  ~no_display      False          skip the local OpenCV preview window (headless hosts)
  ~display_scale   2.0            preview window upscale factor (published image is 640x480)

Live inference: one shared DEKR multi-animal model (tracking/dekr_live_runner.py)
runs ONCE per frame, always batched across both eyes -- DekrLiveRunner has
no single-eye inference path, so frames arriving before the right image
topic has published anything are skipped entirely (no tracking, no warm-up
buffering; only the raw left-camera passthrough is republished to
/tracking/visualization). Instance-to-arm identity (which detection is
psm1 vs psm2) is resolved by the model's own identity head (DekrLiveRunner's
assign_tool_identities()), which reorders each eye's detections onto the
fixed [tool1/psm1, tool2/psm2] slots given by tracking/constants.py's
DEKR_ARM_ORDER -- there is no FK-based fallback for this (see
dekr_live_runner.py's module docstring for the trade-off).

Warm-up buffering is a single buffer shared across all not-yet-warmed arms
purely so one DEKR call per buffered frame can serve every pending arm
(cheaper than one call per arm) -- see _run_warmup().
"""

import os
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState

# SHARE_DIR: installed resource directory (Parameter/, pnp_*.yaml, etc.)
SHARE_DIR = get_package_share_directory("surgical_tool_tracking_ros2")


def _resolve_src_dir() -> str:
    if "STT_SRC_DIR" in os.environ:
        return os.path.normpath(os.environ["STT_SRC_DIR"])
    candidate = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from tracking.arm_configs import build_arm_configs
from tracking.constants import DLC_W, DLC_H, CAL_W, CAL_H, DEKR_BODYPARTS, DEKR_ARM_ORDER
from tracking.label_assignment import instance_to_dets, remap_labels

try:
    from tracking.dekr_live_runner import DekrLiveRunner

    _DEKR_AVAILABLE = True
except ImportError:
    DekrLiveRunner = None
    _DEKR_AVAILABLE = False
from tracking.kinematics import joint_angles_from_msgs

# QoS: use best-effort for sensor data to avoid queue buildup
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


from ros2_node.pose_messages import pose_to_stamped


class DualArmTrackingNode(Node):

    def __init__(self, *, allow_primary_switch=False):
        super().__init__("dual_arm_tracking")

        # --- Parameters ---
        self.declare_parameter("arms", ["psm1", "psm2"])
        self.declare_parameter("allow_primary_switch", allow_primary_switch)
        self.declare_parameter("n_particles", 300)
        self.declare_parameter("warmup_frames", 0)
        # Primary selection is passed explicitly by the entry or ROS parameter.
        self.declare_parameter("overlap_trigger_px", 80.0)
        self.declare_parameter("image_topic", "/camera/left/image_raw")
        self.declare_parameter("right_image_topic", "/camera/right/image_raw")
        self.declare_parameter("camera_frame", "left_camera_optical_frame")
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
        self.declare_parameter("no_display", False)
        # The published image is DLC_W x DLC_H (640x480) -- small on a
        # modern monitor, so the preview window is upscaled by this factor.
        self.declare_parameter("display_scale", 2.0)

        arms = self.get_parameter("arms").value
        n_particles = self.get_parameter("n_particles").value
        overlap_trigger_px = self.get_parameter("overlap_trigger_px").value
        self._warmup_frames_needed = self.get_parameter("warmup_frames").value
        image_topic = self.get_parameter("image_topic").value
        right_image_topic = self.get_parameter("right_image_topic").value
        self._cam_frame = self.get_parameter("camera_frame").value
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

        # --- Shared DEKR multi-animal live model (one forward pass/frame,
        # both arms) -- replaces the old one-model-per-arm DLCRunner setup.
        self._dekr = None
        if _DEKR_AVAILABLE:
            self._dekr = DekrLiveRunner(
                model_path=dekr_model_path or None,
                use_fp16=dekr_fp16,
                use_compile=dekr_compile,
                use_identity_head=dekr_use_identity_head,
            )
        else:
            self.get_logger().error(
                "tracking.dekr_live_runner unavailable (dlclive not installed?) — "
                "tracking is disabled, only camera passthrough will be published."
            )

        # --- Initialize trackers ---
        intrinsics_npz = os.path.join(SHARE_DIR, "Parameter", "intrinsics.npz")
        robot_file = os.path.join(SHARE_DIR, "Parameter", "LND.json")
        projector = SimpleProjector(intrinsics_npz, DLC_W, DLC_H, CAL_W, CAL_H)
        K_dlc = projector.K1
        dist_l = np.load(intrinsics_npz)["dist_left"].flatten().astype(np.float64)
        self._arm_cfgs = build_arm_configs(SHARE_DIR)  # ROS2 live mode, data_dir=''

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
        self._js_lock = threading.Lock()
        self._js_cache: dict[str, list] = {arm: [None, None] for arm in self._trackers}

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

        # Right camera image (best-effort cache, no hard sync with left --
        # see module docstring; adequate for the triangulation vote's
        # MIN_TRI_DIST_MM tolerance).
        self._latest_right_img = None
        self.create_subscription(Image, right_image_topic, self._right_image_cb, _SENSOR_QOS)

        # Left camera image (subscribed last, triggers the main processing loop)
        self.create_subscription(Image, image_topic, self._image_cb, _SENSOR_QOS)
        self.get_logger().info(f"Subscribed to {image_topic} / {right_image_topic}")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _js_cb(self, msg: JointState, arm_name: str, slot: int):
        """slot=0: arm joints,  slot=1: jaw joint."""
        with self._js_lock:
            self._js_cache[arm_name][slot] = msg

    def _right_image_cb(self, img_msg: Image):
        try:
            self._latest_right_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error (right): {e}")

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
            vis_msg.header = header
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
        """Run warmup_live() for every arm in `ready`, using the shared
        buffer. One DEKR call per buffered frame, shared across whichever
        `ready` arms have a joint-angle entry in that frame's snapshot --
        the model's identity head already gives each arm's detections in
        DEKR_ARM_ORDER's fixed slots, so no per-arm inference needed.
        """
        self.get_logger().info(
            f"warm-up: running DEKR inference over {len(self._warmup_buffer)} buffered frames "
            f"for {ready} -- first call includes model load"
            + (" + torch.compile (~30-60s)" if self._dekr.use_compile else "")
            + "..."
        )
        per_arm_frames: dict[str, list] = {a: [] for a in ready}
        for i, (left_img, right_img, ja_by_arm) in enumerate(self._warmup_buffer):
            present = [a for a in ready if a in ja_by_arm]
            if not present:
                continue
            instances, _instances_r = self._dekr.infer(left_img, right_img)
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(self._warmup_buffer):
                self.get_logger().info(
                    f"warm-up: {i + 1}/{len(self._warmup_buffer)} frames inferred"
                )
            for idx, arm_name in enumerate(DEKR_ARM_ORDER):
                if arm_name not in present or idx >= len(instances):
                    continue
                remap = self._arm_cfgs[arm_name]["label_remap"]
                raw_l = remap_labels(instance_to_dets(instances[idx], DEKR_BODYPARTS), remap)
                per_arm_frames[arm_name].append((left_img, ja_by_arm[arm_name], raw_l))

        for arm_name in ready:
            self.get_logger().info(
                f"[{arm_name}] warm-up: {len(per_arm_frames[arm_name])} frames, initializing..."
            )
        self._combined.warmup_live({a: per_arm_frames[a] for a in ready})
        for arm_name in ready:
            self._warmup_done[arm_name] = True

    def _image_cb(self, img_msg: Image):
        try:
            left_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        vis = left_img.copy()

        if self._dekr is None:
            self._publish_vis(vis, img_msg.header)
            return

        right_img = self._latest_right_img

        # DekrLiveRunner is stereo-only -- nothing to infer, and nothing to
        # buffer for warm-up, until the right image topic has published at
        # least one frame. Still republish the raw left camera passthrough
        # so /tracking/visualization isn't silent while waiting.
        if right_img is None:
            self._publish_vis(vis, img_msg.header)
            return

        # --- Joint angles available this tick, per arm ---
        joint_angles_by_arm: dict[str, np.ndarray] = {}
        for arm_name in self._trackers:
            with self._js_lock:
                js_msg, jaw_msg = self._js_cache[arm_name]
            if js_msg is None:
                self.get_logger().warn(
                    f"[{arm_name}] no joint state yet, skipping frame", throttle_duration_sec=2.0
                )
                continue
            joint_angles_by_arm[arm_name] = joint_angles_from_msgs(js_msg, jaw_msg)

        # --- Warm-up buffering (shared -- see class/module docstring) ---
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

            # Arms still pending (not ready this tick) don't get a regular
            # update below -- matches per-arm behaviour before this frame
            # would otherwise track against an unwarmed prior.
            for a in pending:
                if a not in ready:
                    joint_angles_by_arm.pop(a, None)

        # --- Main per-frame tracking: one shared DEKR forward pass (its
        # identity head resolves psm1 vs psm2, see class docstring), then
        # one joint update of the shared 12-state (6 per arm) particle
        # filter -- see tracking/combined_tracker.py. ---
        if joint_angles_by_arm:
            t0 = time.perf_counter()
            instances_l, instances_r = self._dekr.infer(left_img, right_img)
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
            for idx, arm_name in enumerate(DEKR_ARM_ORDER):
                if arm_name not in joint_angles_by_arm or idx >= len(instances_l):
                    continue
                remap = self._arm_cfgs[arm_name]["label_remap"]
                raw_l = remap_labels(instance_to_dets(instances_l[idx], DEKR_BODYPARTS), remap)
                raw_r = (
                    remap_labels(instance_to_dets(instances_r[idx], DEKR_BODYPARTS), remap)
                    if idx < len(instances_r)
                    else {}
                )
                per_arm[arm_name] = dict(
                    joint_angles=joint_angles_by_arm[arm_name], raw_l=raw_l, raw_r=raw_r
                )

            results = self._combined.update(left_img=left_img, right_img=right_img, per_arm=per_arm)

            for arm_name, frame in results.items():
                if not frame.available:
                    continue
                tr = self._trackers[arm_name]
                tr.draw(vis, frame.corrected_keypoints)

                # Publish pose
                pose_msg = pose_to_stamped(tr.cam_T_b, img_msg.header.stamp, self._cam_frame)
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
    node = DualArmTrackingNode(allow_primary_switch=allow_primary_switch)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
