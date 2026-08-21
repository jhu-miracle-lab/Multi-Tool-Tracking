"""ArmTracker: per-arm particle-filter tracking, shared by offline and realtime pipelines."""

import os

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from core.RobotLink import RobotLink
from core.ParticleFilter import ParticleFilter
from core.utils import axisAngleToRotationMatrix

from .constants import (
        DELTA_PROCESS_STD,
)
from .kinematics import load_kin_data
from .observation import pf_init, pf_motion, create_combined_obs_model_vec, _project_batch
from .label_assignment import LR_BASES, FB_BASES
from .visualization import project_skeleton, draw_keypoints
from .feature_detection import ArmFeatureDetector

class ArmTracker:
    """Tracks a single robot arm with a delta_T particle filter.

    PF state = [dtx, dty, dtz, drx, dry, drz] — SE(3) perturbation around the
    fixed FK extrinsic:  cam_T_b = cam_T_b_kin @ delta_T.
    Zero state = identity = FK is perfectly accurate.
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
        n_particles: int = 300,
        shaft_params: dict = None,
    ):
        """
        Args:
            name:        Arm identifier, e.g. 'psm1'.
            cfg:         Per-arm configuration dict (kin_dir, he_file, …).
            dlc_left:    DLC source for the left camera (dict or DLCRunner).
            dlc_right:   DLC source for the right camera, or None.
            K:           Left camera intrinsic matrix (3×3).
            dist:        Left camera distortion coefficients.
            projector:   SimpleProjector instance.
            robot_file:  Path to the LND robot JSON description.
            n_particles: Number of particles in the filter.
            shaft_params: Tuned Canny+HoughLines shaft-edge params (schema:
                          tracking/canny_hough.py's process() + roi_pad/
                          edge_margin), e.g. loaded from a dataset's
                          canny_params_{arm}.json. Falls back to
                          _SHAFT_CANNY_PARAMS.get(name) when None.
        """
        self.name = name
        self.kin_dir = cfg["kin_dir"]
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

        # Fixed kinematic extrinsic — used for label assignment and as delta_T base.
        self.cam_T_b_kin = self.cam_T_b.copy()
        self.b_T_cam_kin = np.linalg.inv(self.cam_T_b_kin)

        self.feature_detector = ArmFeatureDetector(
            name, dlc_left, dlc_right, shaft_params
        )

        # Relative-tracking mode: when primary_tracker is set (via
        # set_relative_to()), this arm's pose is anchored to the primary
        # arm's live cam_T_b instead of this arm's own fixed cam_T_b_kin.
        # See _anchor_T() / set_relative_to().
        self.primary_tracker = None
        self.T_rel_bar = np.eye(4)
        self._explicit_anchor = None
        self._last_init_std = DELTA_PROCESS_STD.copy()

        # Shaft geometry is consumed by the PF; image detection lives in ArmFeatureDetector.
        shaft_pos_list, shaft_dir_list, _, shaft_radius_list = self.robot_arm.getShaftFeatures()
        self._has_shaft = self.feature_detector.shaft_params is not None and len(shaft_pos_list) > 0
        self._shaft_pos_b = shaft_pos_list[0] if self._has_shaft else None
        self._shaft_dir_b = shaft_dir_list[0] if self._has_shaft else None
        self._shaft_radius = shaft_radius_list[0] if self._has_shaft else None

        # PF state = [dtx, dty, dtz, drx, dry, drz], zero = identity
        state_init = np.zeros(6)
        self._vec_obs_func = create_combined_obs_model_vec(self._anchor_T, self._shaft_radius)
        self.observation_variance_scale = 1.0

        self.pf = ParticleFilter(
            num_states=6,
            initialDistributionFunc=pf_init,
            motionModelFunc=pf_motion,
            obsModelFunc=None,
            num_particles=n_particles,
            min_num_effective_particles=n_particles * 0.5,
        )
        self.pf.initializeFilter(mean_state=state_init, std=DELTA_PROCESS_STD)
        self.pose_state = state_init.copy()
        self.prediction_unreliability = 0.0
        self.prediction_reliability_details = {}
        self.last_log_evidence = float("nan")

    def _anchor_T(self):
        """The fixed pose that pose_state's delta_T is applied on top of.

        Normally this arm's own fixed hand-eye extrinsic. After
        set_relative_to() is called, it instead tracks the primary arm's
        live pose composed with the (once-computed) relative offset —
        every other part of the PF/observation pipeline is unaware of the
        difference, since this is the only thing that changes.
        """
        if self._explicit_anchor is not None:
            return self._explicit_anchor
        if self.primary_tracker is None:
            return self.cam_T_b_kin
        return self.primary_tracker.cam_T_b @ self.T_rel_bar

    def set_relative_to(self, primary_tracker):
        """Switch this tracker to track relative to primary_tracker's live
        pose instead of this arm's own fixed hand-eye anchor.

        Call once, after both this tracker's warmup() and primary_tracker's
        warmup() have completed. Converts the current (warmup-derived)
        absolute pose estimate into a relative offset from the primary's
        current pose, so tracked pose is unchanged at the moment of the
        switch — only the anchor that future frames' delta_T is estimated
        around changes, from this arm's own calibration to the primary
        arm's continuously-updated pose.
        """
        if self.primary_tracker is primary_tracker:
            return
        old_anchor = self._anchor_T().copy()
        abs_pose = self.cam_T_b.copy()

        self._explicit_anchor = None
        self.primary_tracker = primary_tracker
        self.T_rel_bar = np.linalg.inv(primary_tracker.cam_T_b) @ abs_pose

        new_anchor = self._anchor_T().copy()
        self._reanchor_particles(old_anchor, new_anchor)

        t_mm = self.T_rel_bar[:3, 3] * 1000
        r_deg = np.rad2deg(cv2.Rodrigues(self.T_rel_bar[:3, :3])[0].flatten())
        print(
            f"[{self.name}] now tracked relative to [{primary_tracker.name}]: "
            f"T_rel = t=[{t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f}]mm "
            f"r=[{r_deg[0]:.1f},{r_deg[1]:.1f},{r_deg[2]:.1f}]deg"
        )

    def refresh_relative_to(self, primary_tracker):
        """Use Method 3's frame-local relative transform.

        Immediately before the primary is updated at frame t, both trackers
        still contain their estimates from t-1.  Refreshing

            T_rel,t-1 = inv(T_primary,t-1) @ T_secondary,t-1

        makes the secondary anchor become

            T_primary,t @ T_rel,t-1
          = Delta_T_primary,t @ T_secondary,t-1

        after the primary update.  Thus only the primary's latest temporal
        increment is transferred; no fixed inter-tool transform is retained.
        Re-anchoring preserves every particle's absolute pose at refresh time.
        """
        old_anchor = self._anchor_T().copy()
        abs_pose = self.cam_T_b.copy()
        self.primary_tracker = primary_tracker
        self.T_rel_bar = np.linalg.inv(primary_tracker.cam_T_b) @ abs_pose
        self._reanchor_particles(old_anchor, self._anchor_T().copy())

    def base_to_tool_pose(self):
        """Current FK transform from the arm base to the tool frame."""
        return self.robot_arm.baseToJointT[-3].copy()

    def camera_to_tool_pose(self):
        """Current estimated tool pose T^C_E = T^C_B T^B_E(q)."""
        return self.cam_T_b @ self.base_to_tool_pose()

    def apply_method3_tool_increment(self, primary_tracker, primary_increment,
                                     previous_base_to_tool):
        """Apply Method 3's camera-frame tool motion to every PF particle."""
        old_anchor = self._anchor_T().copy()
        particles = self.pf._particles
        delta_R = Rotation.from_rotvec(particles[:, 3:]).as_matrix()
        absolute_R = np.einsum("ij,njk->nik", old_anchor[:3, :3], delta_R)
        absolute_t = (np.einsum("ij,nj->ni", old_anchor[:3, :3], particles[:, :3])
                      + old_anchor[:3, 3])

        current_base_to_tool = self.base_to_tool_pose()
        fk_change = previous_base_to_tool @ np.linalg.inv(current_base_to_tool)
        left_R, left_t = primary_increment[:3, :3], primary_increment[:3, 3]
        right_R, right_t = fk_change[:3, :3], fk_change[:3, 3]
        left_absolute_R = np.einsum("ij,njk->nik", left_R, absolute_R)
        moved_R = np.einsum("nij,jk->nik", left_absolute_R, right_R)
        moved_t = (np.einsum("ij,nj->ni", left_R, absolute_t) + left_t
                   + np.einsum("nij,j->ni", left_absolute_R, right_t))

        predicted_anchor = primary_increment @ self.cam_T_b @ fk_change
        self.primary_tracker = primary_tracker
        self.T_rel_bar = np.linalg.inv(primary_tracker.cam_T_b) @ predicted_anchor

        anchor_R = predicted_anchor[:3, :3]
        particles[:, :3] = np.einsum(
            "ij,nj->ni", anchor_R.T, moved_t - predicted_anchor[:3, 3])
        relative_R = np.einsum("ij,njk->nik", anchor_R.T, moved_R)
        particles[:, 3:] = Rotation.from_matrix(relative_R).as_rotvec()
        self.pose_state = self.pf.getMeanParticle()

    def set_independent(self):
        """Return to the tool's CTB baseline without changing absolute poses."""
        if self.primary_tracker is None and self._explicit_anchor is None:
            return
        old_anchor = self._anchor_T().copy()
        self.primary_tracker = None
        self._explicit_anchor = None
        self.T_rel_bar = np.eye(4)
        self._reanchor_particles(old_anchor, self._anchor_T().copy())

    def set_advisor_method3_anchor(self, predicted_cam_T_b):
        """Update Method 3's frame-local prior while retaining PSM2 lump state.

        The particle states and weights are intentionally left untouched, so
        T_lump_2,t-1 and its posterior uncertainty propagate through the
        ordinary PF motion model exactly as in the PDF formulation.
        """
        self.primary_tracker = None
        self.T_rel_bar = np.eye(4)
        self._explicit_anchor = np.asarray(predicted_cam_T_b, dtype=np.float64).copy()
        self.pose_state = self.pf.getMeanParticle()

    def initialize_advisor_method3_anchor(self, predicted_cam_T_b):
        """Enter Method 3 once without changing any particle's world pose.

        Warm-up particles are parameterized relative to cam_T_b_kin.  Simply
        replacing that anchor while retaining their coordinates applies the
        warm-up lump correction twice.  This one-time coordinate conversion
        expresses the same weighted posterior relative to the Method-3 anchor;
        later frames use set_advisor_method3_anchor() to propagate it normally.
        """
        old_anchor = self._anchor_T().copy()
        new_anchor = np.asarray(predicted_cam_T_b, dtype=np.float64).copy()
        self.primary_tracker = None
        self.T_rel_bar = np.eye(4)
        self._explicit_anchor = new_anchor
        self._reanchor_particles(old_anchor, new_anchor)

    def _reanchor_particles(self, old_anchor, new_anchor):
        particles = self.pf._particles
        delta_R = Rotation.from_rotvec(particles[:, 3:]).as_matrix()
        absolute_R = np.einsum("ij,njk->nik", old_anchor[:3, :3], delta_R)
        absolute_t = (np.einsum("ij,nj->ni", old_anchor[:3, :3], particles[:, :3])
                      + old_anchor[:3, 3])
        new_R = np.einsum("ij,njk->nik", new_anchor[:3, :3].T, absolute_R)
        new_t = np.einsum(
            "ij,nj->ni", new_anchor[:3, :3].T,
            absolute_t - new_anchor[:3, 3])
        particles[:, :3] = new_t
        particles[:, 3:] = Rotation.from_matrix(new_R).as_rotvec()
        self.pose_state = self.pf.getMeanParticle()

    def prepare_frame(self, frame_idx=None, left_img=None, right_img=None,
                      joint_angles=None, raw_l=None, raw_r=None):
        return self._prepare_frame(
            frame_idx, left_img, right_img, joint_angles, raw_l, raw_r)

    def compute_prediction_unreliability(self, prep):
        """Continuous prediction--detection inconsistency in [0,1]."""
        if prep is None:
            return 1.0
        particles = self.pf._particles
        weights = np.asarray(self.pf._weights, dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-300)
        anchor = self._anchor_T()
        delta_R = Rotation.from_rotvec(particles[:, 3:]).as_matrix()
        R_eff = np.einsum("ij,njk->nik", anchor[:3, :3], delta_R)
        t_eff = (np.einsum("ij,nj->ni", anchor[:3, :3], particles[:, :3])
                 + anchor[:3, 3])
        scores = []
        details = {}
        expected = sorted(LR_BASES | FB_BASES) + ["yaw_1", "yaw_2"]
        for label in expected:
            detection = prep["base_dets"].get(label)
            if label in LR_BASES:
                point_name = f"{label}:{self.feature_detector.curr_lr}"
            elif label in FB_BASES:
                point_name = f"{label}:{self.feature_detector.curr_fb}"
            else:
                point_name = label
            point = prep["name2pt"].get(point_name)
            if detection is None or point is None:
                score = 1.0
                details[label] = {"missing": True, "score": score}
            else:
                projected = _project_batch(point, R_eff, t_eff, self.K, self.dist)
                mean = np.sum(weights[:, None] * projected, axis=0)
                centered = projected - mean
                covariance = np.einsum(
                    "n,ni,nj->ij", weights, centered, centered) + np.eye(2) * 16.0
                residual = np.asarray(detection[:2], dtype=np.float64) - mean
                distance = float(np.sqrt(max(
                    residual @ np.linalg.pinv(covariance) @ residual, 0.0)))
                sigmoid = 1.0 / (1.0 + np.exp(-np.clip((distance - 3.0) / 0.5, -60, 60)))
                confidence = float(np.clip(detection[2], 0.0, 1.0))
                score = 1.0 - confidence * (1.0 - sigmoid)
                details[label] = {"missing": False, "distance": distance,
                                  "confidence": confidence, "score": score}
            scores.append(score)
        self.prediction_unreliability = float(np.mean(scores))
        self.prediction_reliability_details = details
        self.set_observation_variance_scale(1.0 + 3.0 * self.prediction_unreliability)
        return self.prediction_unreliability

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _prepare_frame(
        self,
        frame_idx=None,
        left_img=None,
        right_img=None,
        joint_angles=None,
        raw_l=None,
        raw_r=None,
    ):
        """Per-frame data prep shared by update() and joint-mode tracking
        (tracking/joint_tracker.py): FK refresh, label assignment, the
        obs-model's base_dets, and Canny+Hough shaft edge detection.

        Args/Returns: see update() -- same signature, minus the PF step.
        Returns a dict of everything the obs model / drawing code needs:
            base_dets, name2pt, shaft_pos_b, shaft_dir_b,
            detected_edges_hough, W, H, corrected, joint_angles
        or None if kinematic data is unavailable for this frame.
        """
        if joint_angles is None:
            try:
                joint_angles, _ = load_kin_data(self.kin_dir, frame_idx)
            except Exception:
                return None

        if self._j6_offset:
            joint_angles = joint_angles.copy()
            joint_angles[5] += self._j6_offset
        self.robot_arm.updateJointAngles(joint_angles)
        pts_3d, names = self.robot_arm.getPointFeatures()
        self.name2pt = dict(zip(names, pts_3d))
        if self._has_shaft:
            # Shaft position/direction are joint-angle-dependent (they move
            # with the wrist), so — like name2pt above — they must be
            # refreshed every frame from the just-updated joint angles, not
            # just once at __init__ time (which only ever saw the all-zero
            # default pose RobotLink starts at).
            shaft_pos_list, shaft_dir_list, _, _ = self.robot_arm.getShaftFeatures()
            self._shaft_pos_b = shaft_pos_list[0]
            self._shaft_dir_b = shaft_dir_list[0]

        features = self.feature_detector.detect(
            frame_idx, left_img, right_img, raw_l, raw_r,
            name2pt=self.name2pt, P1=self.P1, P2=self.P2,
            b_T_cam_kin=self.b_T_cam_kin, K=self.K, dist=self.dist,
            robot_arm=self.robot_arm, cam_T_b_kin=self.cam_T_b_kin,
            cam_T_b=self.cam_T_b, shaft_dir_b=self._shaft_dir_b, detection_noise=getattr(self, "detection_noise", None),
            noise_peers=getattr(self, "noise_peers", {}),
        )
        return {
            "base_dets": features.base_dets,
            "name2pt": self.name2pt,
            "shaft_pos_b": self._shaft_pos_b,
            "shaft_dir_b": self._shaft_dir_b,
            "detected_edges_hough": features.detected_edges_hough,
            "W": features.width,
            "H": features.height,
            "corrected": features.corrected,
            "joint_angles": joint_angles,
        }

    def update(
        self,
        frame_idx=None,
        left_img=None,
        right_img=None,
        joint_angles=None,
        raw_l=None,
        raw_r=None,
    ):
        """Run one tracking step for the given frame.

        Args:
            frame_idx:     Dataset frame index (offline mode). None in ROS2 mode.
            left_img:      Left camera BGR image.
            right_img:     Right camera BGR image or None.
            joint_angles:  (N,) float64 array provided directly (ROS2 mode).
                           When None, loaded from file using frame_idx (offline mode).
            raw_l, raw_r:  Pre-computed {bodypart: (x, y, likelihood)} detections,
                           already split to this arm (via the DEKR model's own
                           identity head, tracking/dekr_live_runner.py's
                           assign_tool_identities()) and already through
                           label_assignment.remap_labels(). When given, these are
                           used directly instead of pulling from self.dlc_left/
                           dlc_right -- for callers that run one shared multi-
                           animal model per frame ahead of calling update() for
                           each arm, rather than a per-arm DLCRunner/dict.

        Returns:
            corrected:    {fk_label: (x, y, likelihood)} — remapped detections.
            n_obs_pts:    Number of 3-D/2-D correspondences fed to the PF.
            joint_angles: Raw joint angle array.
            Returns (None, 0, {}) if kinematic data is unavailable.
        """
        prep = self._prepare_frame(frame_idx, left_img, right_img, joint_angles, raw_l, raw_r)
        return self.update_prepared(prep)

    def update_prepared(self, prep):
        if prep is None:
            return None, 0, {}

        self.last_keypoint_detections = prep["corrected"]

        self.pf.predictionStep_vec(std=DELTA_PROCESS_STD)
        obs_kwargs = dict(
            base_dets=prep["base_dets"],
            name2pt=prep["name2pt"],
            robot_arm=self.robot_arm,
            K=self.K,
            dist=self.dist,
            shaft_pos_b=prep["shaft_pos_b"],
            shaft_dir_b=prep["shaft_dir_b"],
            detected_edges_hough=prep["detected_edges_hough"],
            W=prep["W"],
            H=prep["H"],
        )
        log_obs = self._overlap_scaled_obs(self.pf._particles, **obs_kwargs)
        finite = np.isfinite(log_obs)
        if np.any(finite):
            maximum = float(np.max(log_obs[finite]))
            weighted = np.sum(
                self.pf._weights[finite] * np.exp(log_obs[finite] - maximum))
            self.last_log_evidence = maximum + np.log(max(float(weighted), 1e-300))
        else:
            self.last_log_evidence = -np.inf
        self.pf.updateStep_vec(
            self._overlap_scaled_obs,
            **obs_kwargs,
        )

        self.pose_state = self.pf.getMeanParticle()

        anchor = self._anchor_T()
        R_kin = anchor[:3, :3]
        t_kin = anchor[:3, 3]

        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        t_delta = self.pose_state[:3]
        self.cam_T_b = self.cam_T_b.copy()
        self.cam_T_b[:3, :3] = R_kin @ R_delta
        self.cam_T_b[:3, 3] = R_kin @ t_delta + t_kin

        return prep["corrected"], len(prep["base_dets"]), prep["joint_angles"]

    def set_observation_variance_scale(self, scale):
        """Set R_t/R_0 for overlap-adaptive likelihood tempering."""
        self.observation_variance_scale = max(float(scale), 1.0)

    def _overlap_scaled_obs(self, particles, **kwargs):
        return self._vec_obs_func(
            particles, point_variance_scale=self.observation_variance_scale,
            **kwargs)

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
        """Estimate delta_T via PnP over multiple dataset frames, then re-initialize PF.

        Offline-mode counterpart of warmup_live(): reads detections/joint angles
        for each frame index from disk instead of taking already-collected data.

        Call this once before the main tracking loop when PnP hand-eye calibration
        may be inaccurate. Collects per-frame PnP estimates of delta_T, takes the
        median as the new PF mean, and uses the interquartile spread (floored at
        3× process std) as particle scatter.

        Args:
            frame_indices: Iterable of frame indices to use (e.g. range(0, 30)).
            min_pts:       Min 3D-2D correspondences required per frame (default 3).
            pnp_error:     RANSAC reprojection error threshold in pixels (default 12).
            lkl_thresh:    Likelihood threshold for keypoints during warmup (default 0.4,
                           lower than LIKELIHOOD_THRESH_OPT to handle noisy detections).
            left_img_dir:  Left-image directory. When given (and this arm has
                           Canny+Hough shaft params), the PnP rotation estimate's
                           in-image shaft angle is additionally refined via edge
                           detection — only that one rotation DOF is touched,
                           translation and the other two rotation DOF from PnP
                           are left as-is.
        """
        frame_list = list(frame_indices)
        delta_estimates = []

        for frame_idx in frame_list:
            raw_l = self.feature_detector.get_keypoints(self.dlc_left, frame_idx, None)
            try:
                joint_angles, _ = load_kin_data(self.kin_dir, frame_idx)
            except Exception:
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
                    try:
                        joint_angles, _ = load_kin_data(self.kin_dir, frame_idx)
                    except Exception:
                        continue
                    yield left_img, joint_angles, raw_l

            self._correct_shaft_angle(_offline_frames(), init_std)

    def warmup_live(
        self,
        frames,
        min_pts: int = 3,
        pnp_error: float = 12.0,
        lkl_thresh: float = 0.4,
    ):
        """Live-mode counterpart of warmup(): estimates delta_T via PnP over a
        set of already-collected (image, joint_angles, raw_l) triples instead
        of reading a dataset from disk, then re-initializes the PF the same way.

        Intended for ROS2 nodes: buffer a few seconds of camera/joint-state
        callbacks at startup, then call this once before entering the main
        per-frame update() loop.

        Args:
            frames: Sequence of (left_img, joint_angles, raw_l) triples --
                    raw_l is this arm's already-split/remapped detections for
                    that buffered frame (computed by the caller, same as
                    update()'s raw_l/raw_r override -- see that method's
                    docstring for why this can't be computed internally here
                    for a shared multi-animal model).
            min_pts, pnp_error, lkl_thresh: see warmup().
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

            def _live_frames():
                for left_img, joint_angles, raw_l in frame_list:
                    if left_img is None:
                        continue
                    yield left_img, joint_angles, raw_l

            self._correct_shaft_angle(_live_frames(), init_std)

    def _frame_correspondences(self, raw_l, joint_angles, lkl_thresh):
        """Compute this frame's 3-D (this arm's own base frame) <-> 2-D PnP
        correspondences. Mutates self.name2pt as a side effect (refreshed to
        match this frame's joint angles). Shared by _pnp_delta_estimate()
        (single-arm PnP, used by warmup()/warmup_live()) and
        tracking/joint_tracker.py's pooled two-arm PnP warmup.

        Returns (obj_pts, img_pts) lists (possibly empty).
        """
        if self._j6_offset:
            joint_angles = joint_angles.copy()
            joint_angles[5] += self._j6_offset
        self.robot_arm.updateJointAngles(joint_angles)
        pts_3d, names = self.robot_arm.getPointFeatures()
        self.name2pt = dict(zip(names, pts_3d))

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

        obj_pts, img_pts = [], []
        for bp, (x, y, lkl) in corrected.items():
            if lkl >= lkl_thresh and bp in self.name2pt:
                obj_pts.append(self.name2pt[bp])
                img_pts.append([x, y])
        return obj_pts, img_pts

    def _pnp_delta_estimate(self, raw_l, joint_angles, min_pts, pnp_error, lkl_thresh):
        """Estimate delta_T for a single frame via PnP (shared by warmup() and
        warmup_live()).

        Returns a (6,) [t, rvec] delta_T array, or None if this frame doesn't
        yield a usable estimate (too few correspondences, or PnP failure).
        """
        obj_pts, img_pts = self._frame_correspondences(raw_l, joint_angles, lkl_thresh)
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
            return None

        delta_arr = np.array(delta_estimates)
        # drop any rows that contain NaN / Inf (degenerate PnP solutions)
        delta_arr = delta_arr[np.all(np.isfinite(delta_arr), axis=1)]
        if not delta_arr.size:
            print(f"[{self.name}] warmup: all estimates degenerate — keeping zero prior")
            return None

        mean_state = np.median(delta_arr, axis=0)
        iqr_spread = np.percentile(delta_arr, 75, axis=0) - np.percentile(delta_arr, 25, axis=0)
        # fmax ignores NaN; floor ensures scale is always strictly positive
        init_std = np.fmax(iqr_spread, DELTA_PROCESS_STD * 3)

        self.pf.initializeFilter(mean_state=mean_state, std=init_std)
        self.pose_state = mean_state.copy()
        # Remembered so set_relative_to() can re-seed with a comparable
        # spread instead of falling back to the generic constructor default.
        self._last_init_std = init_std.copy()

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
        edge detection (same detector/formula as _run_canny_hough), averaged
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

            if self._j6_offset:
                joint_angles = joint_angles.copy()
                joint_angles[5] += self._j6_offset
            self.robot_arm.updateJointAngles(joint_angles)
            pts_3d, names = self.robot_arm.getPointFeatures()
            self.name2pt = dict(zip(names, pts_3d))

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
            self.feature_detector.detect_shaft(
                left_img, corrected,
                frame_idx=getattr(self, "_current_frame_idx", None),
                noise=getattr(self, "detection_noise", None),
                noise_peers=getattr(self, "noise_peers", {}),
                shaft_dir_b=self._shaft_dir_b,
                cam_T_b_kin=self.cam_T_b_kin,
                K=self.K,
            )

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

        self.pf.initializeFilter(mean_state=self.pose_state, std=init_std)
        print(f"[{self.name}] warmup: shaft-angle correction {np.rad2deg(dtheta):+.2f} deg")

    @staticmethod



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
