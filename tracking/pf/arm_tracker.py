"""Per-arm geometry, observations and poses for the shared particle filter."""

import os

import cv2
import numpy as np
import yaml

from tracking.pf.robot_model import RobotLink
from tracking.common.transforms import axisAngleToRotationMatrix

from tracking.common.frame import ArmFrame, ArmObservation, CameraObservation
from tracking.common.constants import DELTA_PROCESS_STD
from tracking.pf.tracker_config import PRIOR_SHAFT_ROI
from tracking.pf.observation import create_combined_obs_model_vec
from tracking.common.visualization import project_skeleton, draw_keypoints
from tracking.features.feature_detection import ArmFeatureDetector
from tracking.features.camera_features import extract_camera_features

class ArmTracker:
    """Maintains one arm's geometry and six-state pose correction.

    PF state = [dtx, dty, dtz, drx, dry, drz] — SE(3) perturbation around the
    current anchor: cam_T_b = anchor @ delta_T.
    Zero state = identity = no correction relative to the current anchor.
    """

    def __init__(
        self,
        name: str,
        cfg: dict,
        dlc_left,
        dlc_right,
        K: np.ndarray,
        dist: np.ndarray,
        projector,
        robot_file: str,
        shaft_params: dict = None,
        *, feature_factory=ArmFeatureDetector, joint_source=None,
    ):
        """Load per-arm geometry and observation sources.

        CombinedArmTracker owns the particles. This class prepares observations,
        estimates a warmup prior and applies the shared filter's posterior pose.
        shaft_params optionally overrides the detector's per-arm defaults.
        """
        self.name = name
        self.kin_dir = cfg["kin_dir"]
        self.joint_source = joint_source
        self._j6_offset = cfg.get("joint6_offset", 0.0)
        self.K = K
        self.dist = dist
        self.projector = projector
        self.dlc_left = dlc_left
        self.dlc_right = dlc_right
        self.skel_color = cfg["skeleton_color"]
        self.kp_color = cfg["kp_color"]

        self.name2pt = {}

        # Stereo projection matrices for triangulation
        P1 = projector.K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        R = projector.T_stereo[:3, :3]
        t = projector.T_stereo[:3, 3].reshape(3, 1)
        P2 = projector.K2 @ np.hstack([R, t])
        self.P1 = P1.astype(np.float64)
        self.P2 = P2.astype(np.float64)

        self.robot_arm = RobotLink(robot_file)

        # Load hand-eye calibration
        print(f"[{name}] hand-eye file: {cfg['he_file']}")
        with open(cfg["he_file"]) as f:
            he = yaml.load(f, Loader=yaml.FullLoader)
        key = cfg["he_key"]
        self.cam_T_b = np.eye(4)
        self.cam_T_b[:3, 3] = np.array(he[f"{key}_tvec"]) / 1000.0
        rvec = np.array(he[f"{key}_rvec"], dtype=np.float64)
        if cfg["he_rot_loader"] == "axis_angle":
            self.cam_T_b[:3, :3] = axisAngleToRotationMatrix(rvec)
        else:
            self.cam_T_b[:3, :3], _ = cv2.Rodrigues(rvec)

        self.cam_T_b = projector.to_tracking_pose(self.cam_T_b)

        # Fixed kinematic extrinsic — used for label assignment and as delta_T base.
        self.cam_T_b_kin = self.cam_T_b.copy()
        self.b_T_cam_kin = np.linalg.inv(self.cam_T_b_kin)

        self.feature_detector = feature_factory(
            name, dlc_left, dlc_right, shaft_params
        )

        self.right_feature_detector = feature_factory(name, dlc_right, None, shaft_params)

        # The combined tracker supplies this anchor while in relative mode.
        # The hand-eye calibration remains fixed for lump-error calculations.
        self._relative_anchor = None
        self._pending_std = DELTA_PROCESS_STD.copy()
        # Warmup diagnostics: success counts and raw (unfloored) spread.
        self._pending_n_ok = 0
        self._pending_n_total = 0
        self._pending_iqr_raw = DELTA_PROCESS_STD.copy()

        # Shaft geometry is consumed by the PF; image detection lives in ArmFeatureDetector.
        shaft_pos_list, shaft_dir_list, _, shaft_radius_list = self.robot_arm.getShaftFeatures()
        self._has_shaft = self.feature_detector.shaft_params is not None and len(shaft_pos_list) > 0
        self._shaft_pos_b = shaft_pos_list[0] if self._has_shaft else None
        self._shaft_dir_b = shaft_dir_list[0] if self._has_shaft else None
        self._shaft_radius = shaft_radius_list[0] if self._has_shaft else None

        # PF state = [dtx, dty, dtz, drx, dry, drz], zero = identity
        state_init = np.zeros(6)
        self._vec_obs_func = create_combined_obs_model_vec(lambda: self.anchor, self.shaft_radius)
        self._right_obs_func = create_combined_obs_model_vec(
            lambda: self.projector.T_stereo @ self.anchor, self.shaft_radius)
        self.observation_variance_scale = 1.0

        self.pose_state = state_init.copy()

    @property
    def anchor(self):
        """Current transform used as the origin of the six-state correction."""
        return self.cam_T_b_kin if self._relative_anchor is None else self._relative_anchor

    @property
    def has_relative_anchor(self):
        return self._relative_anchor is not None

    def set_anchor(self, anchor):
        """Set/reset the geometric reference; the combined tracker reanchors particles."""
        self._relative_anchor = None if anchor is None else anchor.copy()

    @property
    def warmup_std(self):
        return self._pending_std

    @property
    def shaft_radius(self):
        return self._shaft_radius

    def _update_joint_geometry(self, joint_angles):
        """Apply the configured joint offset and refresh base-frame keypoints."""
        if self._j6_offset:
            joint_angles = joint_angles.copy()
            joint_angles[5] += self._j6_offset
        self.robot_arm.updateJointAngles(joint_angles)
        pts_3d, names = self.robot_arm.getPointFeatures()
        self.name2pt = dict(zip(names, pts_3d))
        return joint_angles

    def prepare_frame(
        self,
        frame_idx=None,
        left_img=None,
        right_img=None,
        joint_angles=None,
        raw_l=None,
        raw_r=None,
    ):
        """Refresh FK and detect features; return an unavailable ArmFrame when joint data is missing."""
        if joint_angles is None:
            joint_angles = self.joint_source(frame_idx) if self.joint_source is not None else None
            if joint_angles is None:
                return ArmFrame()

        joint_angles = self._update_joint_geometry(joint_angles)
        if self._has_shaft:
            # Shaft position/direction are joint-angle-dependent (they move
            # with the wrist), so — like name2pt above — they must be
            # refreshed every frame from the just-updated joint angles, not
            # just once at __init__ time (which only ever saw the all-zero
            # default pose RobotLink starts at).
            shaft_pos_list, shaft_dir_list, _, _ = self.robot_arm.getShaftFeatures()
            self._shaft_pos_b = shaft_pos_list[0]
            self._shaft_dir_b = shaft_dir_list[0]

        feature_context = dict(name2pt=self.name2pt, robot_arm=self.robot_arm,
                               shaft_dir_b=self._shaft_dir_b, projector=self.projector,
                               P1=self.P1, P2=self.P2, prior_roi=PRIOR_SHAFT_ROI)
        features = extract_camera_features(
            self.feature_detector, frame_idx, left_img, raw_l, raw_r,
            self.cam_T_b_kin, self.cam_T_b, self.K, self.dist,
            right_img=right_img, stereo_vote=True, **feature_context)
        right_pose = self.projector.T_stereo @ self.cam_T_b
        right_kin = self.projector.T_stereo @ self.cam_T_b_kin
        right_features = extract_camera_features(
            self.right_feature_detector, frame_idx, right_img, raw_r, None,
            right_kin, right_pose, self.projector.K2, self.projector.dist2, **feature_context)
        observation = ArmObservation(
            keypoints=features.base_dets, points_3d=self.name2pt,
            shaft_position=self._shaft_pos_b, shaft_direction=self._shaft_dir_b,
            shaft_edges=features.detected_edges_hough,
            width=features.width, height=features.height,
            right=CameraObservation(right_features.base_dets,
                                    right_features.detected_edges_hough,
                                    right_features.width, right_features.height))
        return ArmFrame(
            observation=observation, corrected_keypoints=features.corrected,
            joint_angles=joint_angles, observation_count=len(features.base_dets))

    def finalize_pose(self, pose_state_6):
        """Apply the shared PF's six-state mean under the current anchor."""
        self.pose_state = pose_state_6

        anchor = self.anchor
        R_kin = anchor[:3, :3]
        t_kin = anchor[:3, 3]

        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        t_delta = self.pose_state[:3]
        self.cam_T_b = self.cam_T_b.copy()
        self.cam_T_b[:3, :3] = R_kin @ R_delta
        self.cam_T_b[:3, 3] = R_kin @ t_delta + t_kin

    def set_observation_variance_scale(self, scale):
        """Set R_t/R_0 for overlap-adaptive likelihood tempering."""
        self.observation_variance_scale = max(float(scale), 1.0)

    def log_likelihood(self, particles, observation: ArmObservation):
        """Evaluate one arm's visual evidence, including overlap scaling."""
        total = np.zeros(len(particles))
        for camera, view in observation.camera_views():
            if not view.keypoints and (view.shaft_edges is None or len(view.shaft_edges) == 0):
                continue
            if camera == "left":
                model, K, dist = self._vec_obs_func, self.K, self.dist
            else:
                model, K, dist = self._right_obs_func, self.projector.K2, self.projector.dist2
            total += model(
                particles, base_dets=view.keypoints,
                name2pt=observation.points_3d, robot_arm=self.robot_arm,
                K=K, dist=dist, shaft_pos_b=observation.shaft_position,
                shaft_dir_b=observation.shaft_direction,
                detected_edges_hough=view.shaft_edges, W=view.width, H=view.height)
        return total / self.observation_variance_scale

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def warmup(
        self,
        frame_indices,
        min_pts: int = 3,
        pnp_error: float = 12.0,
        lkl_thresh: float = 0.4,
        left_img_dir: str = None,
    ):
        """Estimate the prior from cached points and file-backed joint input.

        min_pts and pnp_error control PnP; lkl_thresh filters detections.
        left_img_dir optionally supplies raw images for shaft-angle refinement.
        Missing joint frames remain part of the reported total frame count.
        """
        frame_list = list(frame_indices)
        delta_estimates = []

        for frame_idx in frame_list:
            raw_l = self.feature_detector.get_keypoints(self.dlc_left, frame_idx, None)
            joint_angles = self.joint_source(frame_idx) if self.joint_source is not None else None
            if joint_angles is None:
                continue
            est = self._pnp_delta_estimate(raw_l, joint_angles, min_pts, pnp_error, lkl_thresh)
            if est is not None:
                delta_estimates.append(est)

        init_std = self._apply_warmup_estimates(delta_estimates, len(frame_list))
        if init_std is None:
            return

        if left_img_dir is not None and self.feature_detector.shaft_params is not None:

            def _offline_frames():
                for frame_idx in frame_list:
                    left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
                    if left_img is None:
                        continue
                    raw_l = self.feature_detector.get_keypoints(self.dlc_left, frame_idx, None)
                    joint_angles = self.joint_source(frame_idx) if self.joint_source is not None else None
                    if joint_angles is None:
                        continue
                    yield self.projector.process_image(left_img, "left"), joint_angles, raw_l

            self._correct_shaft_angle(_offline_frames(), init_std)

    def warmup_live(
        self,
        frames,
        min_pts: int = 3,
        pnp_error: float = 12.0,
        lkl_thresh: float = 0.4,
    ):
        """Estimate the prior from (tracking-image, joints, keypoints) triples.

        Images and keypoints must already use the active camera calibration.
        PnP settings and the optional shaft-angle refinement match warmup().
        """
        frame_list = list(frames)
        delta_estimates = []

        for left_img, joint_angles, raw_l in frame_list:
            est = self._pnp_delta_estimate(raw_l, joint_angles, min_pts, pnp_error, lkl_thresh)
            if est is not None:
                delta_estimates.append(est)

        init_std = self._apply_warmup_estimates(delta_estimates, len(frame_list))
        if init_std is None:
            return

        if self.feature_detector.shaft_params is not None:
            self._correct_shaft_angle(frame_list, init_std)

    def _assign_warmup_labels(self, raw_l):
        """Use the same face/yaw assignment for PnP and shaft-angle warmup."""
        corrected, _, _, _ = self.feature_detector.assign_labels(
            raw_l,
            None,
            self.name2pt,
            self.P1,
            self.P2,
            self.b_T_cam_kin,
            self.K,
            self.dist,
            self.robot_arm,
            self.cam_T_b_kin,
            self.feature_detector.curr_lr,
            self.feature_detector.curr_fb,
            cam_T_b=self.cam_T_b,
        )
        return corrected

    def _pnp_delta_estimate(self, raw_l, joint_angles, min_pts, pnp_error, lkl_thresh):
        """Estimate delta_T for a single frame via PnP (shared by warmup() and
        warmup_live()). Mutates self.name2pt as a side effect (refreshed to
        match this frame's joint angles).

        Returns a (6,) [t, rvec] delta_T array, or None if this frame doesn't
        yield a usable estimate (too few correspondences, or PnP failure).
        """
        joint_angles = self._update_joint_geometry(joint_angles)

        corrected = self._assign_warmup_labels(raw_l)

        obj_pts, img_pts = [], []
        for bp, (x, y, lkl) in corrected.items():
            if lkl >= lkl_thresh and bp in self.name2pt:
                obj_pts.append(self.name2pt[bp])
                img_pts.append([x, y])

        if len(obj_pts) < min_pts:
            return None

        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                np.array(obj_pts, dtype=np.float64),
                np.array(img_pts, dtype=np.float64),
                self.K,
                self.dist,
                iterationsCount=200,
                reprojectionError=pnp_error,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return None

        if not ok or rvec is None:
            return None

        # delta_T = inv(cam_T_b_kin) @ cam_T_b_pnp
        # cam_T_b = cam_T_b_kin @ delta_T  =>  R_delta = R_kin^T @ R_pnp
        R_pnp, _ = cv2.Rodrigues(rvec)
        R_kin = self.cam_T_b_kin[:3, :3]
        t_kin = self.cam_T_b_kin[:3, 3]
        R_delta = R_kin.T @ R_pnp
        t_delta = R_kin.T @ (tvec.flatten() - t_kin)
        rvec_delta, _ = cv2.Rodrigues(R_delta)
        return np.concatenate([t_delta, rvec_delta.flatten()])

    def _apply_warmup_estimates(self, delta_estimates, n_frames_total):
        """Turn a list of per-frame PnP delta_T estimates into a new PF prior
        (median location, IQR-based spread) and re-initialize the filter.

        Returns the init_std used, or None if no estimates were usable (the
        filter is left at its previous prior in that case).
        """
        if not delta_estimates:
            print(f"[{self.name}] warmup: no valid PnP estimates — keeping zero prior")
            self._pending_n_ok = 0
            self._pending_n_total = n_frames_total
            return None

        delta_arr = np.array(delta_estimates)
        # drop any rows that contain NaN / Inf (degenerate PnP solutions)
        delta_arr = delta_arr[np.all(np.isfinite(delta_arr), axis=1)]
        if not delta_arr.size:
            print(f"[{self.name}] warmup: all estimates degenerate — keeping zero prior")
            self._pending_n_ok = 0
            self._pending_n_total = n_frames_total
            return None

        mean_state = np.median(delta_arr, axis=0)
        iqr_spread = np.percentile(delta_arr, 75, axis=0) - np.percentile(delta_arr, 25, axis=0)
        # fmax ignores NaN; floor ensures scale is always strictly positive
        init_std = np.fmax(iqr_spread, DELTA_PROCESS_STD * 3)

        self.pose_state = mean_state.copy()
        # The combined tracker uses this spread to initialize/reset its PF.
        self._pending_std = init_std.copy()
        # Retain warmup diagnostics independently of the floored PF spread.
        self._pending_n_ok = len(delta_arr)
        self._pending_n_total = n_frames_total
        self._pending_iqr_raw = iqr_spread.copy()

        t_mm = mean_state[:3] * 1000
        r_deg = np.rad2deg(mean_state[3:])
        s_mm = init_std[:3] * 1000
        sr_deg = np.rad2deg(init_std[3:])
        print(f"[{self.name}] warmup: {len(delta_estimates)}/{n_frames_total} frames OK")
        print(f"  delta_t = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}] mm")
        print(f"  delta_r = [{r_deg[0]:.2f}, {r_deg[1]:.2f}, {r_deg[2]:.2f}] deg")
        print(
            f"  particle spread: t=[{s_mm[0]:.1f},{s_mm[1]:.1f},{s_mm[2]:.1f}] mm  "
            f"r=[{sr_deg[0]:.2f},{sr_deg[1]:.2f},{sr_deg[2]:.2f}] deg"
        )
        return init_std

    def _correct_shaft_angle(self, frame_source, init_std):
        """Refine the PnP rotation estimate's in-image shaft angle via Canny+Hough
        edge detection (same detector used during tracking), averaged
        over the given frames. Shared by warmup() (reads frames from disk) and
        warmup_live() (frames already buffered in memory).

        Args:
            frame_source: Iterable of (left_img, joint_angles, raw_l) triples.
            init_std:     Particle spread to re-initialize the filter with
                          after the angle correction (from _apply_warmup_estimates).

        Only the rotation about the camera's optical axis is corrected —
        translation and the other two rotation DOF from the PnP median are
        left untouched.
        """
        R_kin = self.cam_T_b_kin[:3, :3]
        t_kin = self.cam_T_b_kin[:3, 3]

        # Use the PnP-refined pose (not the raw hand-eye value) for label
        # assignment / ROI centring while building up the edge-detection EMA.
        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        self.cam_T_b = self.cam_T_b_kin.copy()
        self.cam_T_b[:3, :3] = R_kin @ R_delta
        self.cam_T_b[:3, 3] = R_kin @ self.pose_state[:3] + t_kin

        self.feature_detector.ema_lines = None
        for left_img, joint_angles, raw_l in frame_source:
            if left_img is None:
                continue

            self._update_joint_geometry(joint_angles)

            corrected = self._assign_warmup_labels(raw_l)
            _, directions, _, _ = self.robot_arm.getShaftFeatures()
            self.feature_detector.shaft.detect(
                left_img, corrected, K=self.K, dist=self.dist,
                shaft_dir_b=directions[0] if len(directions) else None,
                cam_T_b_kin=self.cam_T_b_kin)

        if self.feature_detector.ema_lines is None:
            print(f"[{self.name}] warmup: no shaft edges detected — angle uncorrected")
            return

        shaft_pos_list, shaft_dir_list, _, _ = self.robot_arm.getShaftFeatures()
        if len(shaft_pos_list) == 0:
            return
        dir_b = shaft_dir_list[0] / np.linalg.norm(shaft_dir_list[0])

        # EMA midline angle (Hough NORMAL-angle convention, from cv2.HoughLines).
        thetas = [th for _, th in self.feature_detector.ema_lines]
        theta_mid = np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas))) % np.pi

        d_cam = self.cam_T_b[:3, :3] @ dir_b
        if d_cam[2] < 0:
            d_cam = -d_cam
        if d_cam[2] <= 1e-6:
            print(f"[{self.name}] warmup: shaft nearly edge-on to camera — angle uncorrected")
            return
        # atan2(fy*dy/z, fx*dx/z) approximates the projected axis's DIRECTION
        # angle, not its normal angle — rotate by 90 deg to match theta_mid's
        # Hough normal-angle convention before differencing.
        theta_pred_dir = np.arctan2(
            self.K[1, 1] * d_cam[1] / d_cam[2], self.K[0, 0] * d_cam[0] / d_cam[2]
        )
        theta_pred = (theta_pred_dir + np.pi / 2) % np.pi
        dtheta = theta_mid - theta_pred
        if dtheta > np.pi / 2:
            dtheta -= np.pi
        elif dtheta < -np.pi / 2:
            dtheta += np.pi

        if abs(dtheta) > np.deg2rad(30):
            print(
                f"[{self.name}] warmup: shaft-angle correction {np.rad2deg(dtheta):.1f} "
                f"deg exceeds limit, skipped"
            )
            return

        c, s = np.cos(dtheta), np.sin(dtheta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R_new = Rz @ self.cam_T_b[:3, :3]
        R_delta_new = R_kin.T @ R_new
        rvec_new, _ = cv2.Rodrigues(R_delta_new)
        self.pose_state[3:] = rvec_new.flatten()  # rotation only — translation untouched

        self._pending_std = init_std
        print(f"[{self.name}] warmup: shaft-angle correction {np.rad2deg(dtheta):+.2f} deg")

    def draw(self, vis_img: np.ndarray, corrected: dict):
        """Draw skeleton, keypoints, and the detected shaft edges.

        The FK skeleton's shaft segment is replaced with a single merged
        line from the detected edge midline when available (smoothed,
        FK-anchored — shows the filter's belief). The two raw Canny+Hough
        detected edges (self.feature_detector.last_detected_segs, EMA-smoothed but otherwise
        exactly what the detector found) are drawn separately in cyan, so
        detection quality/failures are visible directly instead of being
        hidden behind the FK-anchored reconstruction.
        """
        use_ema_shaft = self.feature_detector.shaft_params is not None and self.feature_detector.ema_lines is not None

        # Draw wrist/jaw segments (skip shaft index 0 when using EMA line)
        project_skeleton(
            self.robot_arm,
            self.cam_T_b,
            self.projector,
            [vis_img],
            color=self.skel_color,
            skip_indices=(0,) if use_ema_shaft else (),
        )

        # Replace shaft segment with the merged EMA midline, truncated at
        # the link4 projection — a single line, not the two raw detected
        # edges. Purely cosmetic: does not feed back into pose_state/cam_T_b
        # (the PF likelihood uses the raw per-frame detection instead, see
        # self.feature_detector.last_raw_hough).
        if use_ema_shaft:
            H, W = vis_img.shape[:2]
            thetas = [th for _, th in self.feature_detector.ema_lines]
            theta_mid = np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas))) % np.pi
            cos_t, sin_t = np.cos(theta_mid), np.sin(theta_mid)

            # Project link4 (shaft end / wrist root) with corrected cam_T_b
            _, link4_b = self.robot_arm.getSkeletonPoints()[0]
            link4_b = np.array(link4_b)
            p_cam = self.cam_T_b[:3, :3] @ link4_b + self.cam_T_b[:3, 3]
            if p_cam[2] > 0.01:
                u4 = int(round(self.K[0, 0] * p_cam[0] / p_cam[2] + self.K[0, 2]))
                v4 = int(round(self.K[1, 1] * p_cam[1] / p_cam[2] + self.K[1, 2]))

                # theta_mid is a LINE angle (mod pi) with no inherent sign —
                # resolve which of the two opposite directions actually
                # points from link4 toward the shaft's base/entry-port side.

                dx, dy = -sin_t, cos_t
                want_positive_dx = self.name == "psm1"
                if (dx > 0) != want_positive_dx:
                    dx, dy = -dx, -dy
                ts = []
                for num, denom in [(-u4, dx), (W - 1 - u4, dx), (-v4, dy), (H - 1 - v4, dy)]:
                    if abs(denom) > 1e-6:
                        t = num / denom
                        if t > 0:
                            ts.append(t)
                if ts:
                    t_far = min(ts)
                    x_far = int(round(u4 + t_far * dx))
                    y_far = int(round(v4 + t_far * dy))
                    cv2.line(vis_img, (u4, v4), (x_far, y_far), self.skel_color, 2)

        # Draw the two raw detected shaft edges (Canny+Hough, EMA-smoothed
        # but not FK-anchored) directly, so the actual detection is visible
        # regardless of whether the FK-anchored midline above looks right.
        for x1, y1, x2, y2 in self.feature_detector.last_detected_segs:
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

        draw_keypoints(vis_img, corrected, self.kp_color)
