"""JointTracker: shared 6-DOF delta_T particle filter across two arms,
using a known fixed base-to-base transform (b1_T_b2) instead of two
independent per-arm delta_T estimates.

Assumes both arms' pose error comes from a shared upstream miscalibration
source (camera extrinsics / ECM kinematics) rather than independent
per-arm error -- valid when both PSM bases are rigidly, stationarily
mounted relative to each other.
"""

import cv2
import numpy as np

from core.ParticleFilter import ParticleFilter

from .arm_tracker import ArmTracker
from .feature_detection import ArmFeatureDetector
from .constants import JOINT_DELTA_PROCESS_STD
from .kinematics import load_kin_data
from .observation import create_combined_obs_model_vec, pf_init, pf_motion
from .overlap import bidirectional_keypoint_coverage, observation_variance_scale


class JointTracker:
    """Tracks two arms with ONE shared delta_T:

        cam_T_b1 = cam_T_b1_kin @ delta_T
        cam_T_b2 = cam_T_b1_kin @ delta_T @ b1_T_b2

    tr1's own fixed kinematic extrinsic (cam_T_b1_kin) anchors the shared
    state; tr2's own hand-eye calibration is only used as an initial guess
    during independent warmup, not during joint tracking.
    """

    def __init__(self, tr1: ArmTracker, tr2: ArmTracker, b1_T_b2: np.ndarray, n_particles: int = 300):
        self.tr1 = tr1
        self.tr2 = tr2
        self.b1_T_b2 = b1_T_b2

        self._obs_fn_1 = create_combined_obs_model_vec(
            self._shared_anchor, tr1._shaft_radius, extra_T=None
        )
        self._obs_fn_2 = create_combined_obs_model_vec(
            self._shared_anchor, tr2._shaft_radius, extra_T=b1_T_b2
        )

        self.pf = ParticleFilter(
            num_states=6,
            initialDistributionFunc=pf_init,
            motionModelFunc=pf_motion,
            obsModelFunc=None,
            num_particles=n_particles,
            min_num_effective_particles=n_particles * 0.5,
        )
        self.delta_T_state = np.zeros(6)
        self.pf.initializeFilter(mean_state=self.delta_T_state, std=JOINT_DELTA_PROCESS_STD)
        self.last_n_dets = {}

    def _shared_anchor(self):
        """Fixed anchor both arms' delta_T is expressed against: tr1's own
        kinematic extrinsic. Bound-method reference (like ArmTracker's own
        _anchor_T), re-evaluated each PF step but constant in practice.
        """
        return self.tr1.cam_T_b_kin

    def warmup(
        self,
        frame_indices,
        min_pts: int = 3,
        pnp_error: float = 25.0,
        lkl_thresh: float = 0.4,
        left_img_dir: str = None,
    ):
        """Pooled two-arm PnP warmup: for each frame, gather BOTH arms' 3-D/2-D
        correspondences (tr2's 3-D points transformed into tr1's base frame
        via b1_T_b2), then solve ONE cv2.solvePnPRansac per frame for the
        shared delta_T -- instead of two independent per-arm PnP solves.

        This is only meaningful when both arms' hand-eye ultimately comes
        from a single joint/shared calibration (e.g. one PnP over a combined
        point set at calibration time). If tr1/tr2 were calibrated
        independently, their errors are NOT a shared source and pooling
        won't help (see tracking/offline_tracking.py's --joint docs).

        pnp_error default (25.0, was 12.0): with only ~14 candidate
        correspondences per frame (two arms' worth of keypoint slots), the
        RANSAC inlier-count vs. threshold curve is U-shaped, not monotonic --
        swept 3-50 against dataset 20260718_115614 (pf_simulation's
        pooled_pnp_pose(), same mechanics as this method): 6-12 sits in the
        WORST stability zone (frame-to-frame pose std ~8.6-12.5mm / 16-23deg,
        because RANSAC picks a different inlier subset almost every frame),
        while >=20 lets nearly all correspondences in every frame and is far
        more stable (~0.8-3mm / 0.5-4deg). Below ~5 loses too many frames to
        min_pts entirely. This does NOT close the gap to an independently-
        computed hand-eye calibration (stays ~25-40mm/7-40deg regardless of
        pnp_error) -- that gap is a real cross-check discrepancy, not RANSAC
        noise; see JOINT_DELTA_PROCESS_STD's docstring in constants.py.
        """
        frame_list = list(frame_indices)
        delta_estimates = []
        R_b1b2 = self.b1_T_b2[:3, :3]
        t_b1b2 = self.b1_T_b2[:3, 3]

        for frame_idx in frame_list:
            raw_l1 = ArmFeatureDetector.get_keypoints(self.tr1.dlc_left, frame_idx, None)
            raw_l2 = ArmFeatureDetector.get_keypoints(self.tr2.dlc_left, frame_idx, None)
            try:
                ja1, _ = load_kin_data(self.tr1.kin_dir, frame_idx)
                ja2, _ = load_kin_data(self.tr2.kin_dir, frame_idx)
            except Exception:
                continue

            obj1, img1 = self.tr1._frame_correspondences(raw_l1, ja1, lkl_thresh)
            obj2, img2 = self.tr2._frame_correspondences(raw_l2, ja2, lkl_thresh)
            obj2_in_b1 = [R_b1b2 @ p + t_b1b2 for p in obj2]

            obj_pts = obj1 + obj2_in_b1
            img_pts = img1 + img2
            if len(obj_pts) < min_pts:
                continue

            try:
                ok, rvec, tvec, _ = cv2.solvePnPRansac(
                    np.array(obj_pts, dtype=np.float64),
                    np.array(img_pts, dtype=np.float64),
                    self.tr1.K,
                    self.tr1.dist,
                    iterationsCount=200,
                    reprojectionError=pnp_error,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
            except cv2.error:
                continue
            if not ok or rvec is None:
                continue

            R_pnp, _ = cv2.Rodrigues(rvec)
            R_kin = self.tr1.cam_T_b_kin[:3, :3]
            t_kin = self.tr1.cam_T_b_kin[:3, 3]
            R_delta = R_kin.T @ R_pnp
            t_delta = R_kin.T @ (tvec.flatten() - t_kin)
            rvec_delta, _ = cv2.Rodrigues(R_delta)
            delta_estimates.append(np.concatenate([t_delta, rvec_delta.flatten()]))

        if not delta_estimates:
            print("[joint] warmup: no valid pooled PnP estimates -- keeping zero prior")
            return

        delta_arr = np.array(delta_estimates)
        delta_arr = delta_arr[np.all(np.isfinite(delta_arr), axis=1)]
        if not delta_arr.size:
            print("[joint] warmup: all pooled estimates degenerate -- keeping zero prior")
            return

        mean_state = np.median(delta_arr, axis=0)
        iqr_spread = np.percentile(delta_arr, 75, axis=0) - np.percentile(delta_arr, 25, axis=0)
        init_std = np.fmax(iqr_spread, JOINT_DELTA_PROCESS_STD * 3)

        self.delta_T_state = mean_state.copy()
        self.pf.initializeFilter(mean_state=mean_state, std=init_std)
        self.tr1.pose_state = mean_state.copy()
        self.tr2.pose_state = mean_state.copy()

        t_mm = mean_state[:3] * 1000
        r_deg = np.rad2deg(mean_state[3:])
        s_mm = init_std[:3] * 1000
        sr_deg = np.rad2deg(init_std[3:])
        print(f"[joint] warmup: {len(delta_estimates)}/{len(frame_list)} frames OK (pooled PnP)")
        print(f"  delta_t = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}] mm")
        print(f"  delta_r = [{r_deg[0]:.2f}, {r_deg[1]:.2f}, {r_deg[2]:.2f}] deg")
        print(
            f"  particle spread: t=[{s_mm[0]:.1f},{s_mm[1]:.1f},{s_mm[2]:.1f}] mm  "
            f"r=[{sr_deg[0]:.2f},{sr_deg[1]:.2f},{sr_deg[2]:.2f}] deg"
        )

    def update(self, frame_idx=None, left_img=None):
        """Run one joint tracking step. Both arms observe the same camera
        image (single endoscope, both tools in view).

        Returns:
            {tr1.name: corrected_dict, tr2.name: corrected_dict}, or None if
            either arm's kinematic data is unavailable this frame.
        """
        prep1 = self.tr1._prepare_frame(frame_idx=frame_idx, left_img=left_img)
        prep2 = self.tr2._prepare_frame(frame_idx=frame_idx, left_img=left_img)
        if prep1 is None or prep2 is None:
            return None

        self.overlap_coverage = bidirectional_keypoint_coverage(
            prep1["base_dets"], prep2["base_dets"], 30.0
        )
        variance_scale = observation_variance_scale(self.overlap_coverage, 3.0)

        self.pf.predictionStep_vec(std=JOINT_DELTA_PROCESS_STD)

        def _joint_obs(particles, **_):
            log1 = self._obs_fn_1(
                particles,
                base_dets=prep1["base_dets"],
                name2pt=prep1["name2pt"],
                robot_arm=self.tr1.robot_arm,
                K=self.tr1.K,
                dist=self.tr1.dist,
                shaft_pos_b=prep1["shaft_pos_b"],
                shaft_dir_b=prep1["shaft_dir_b"],
                detected_edges_hough=prep1["detected_edges_hough"],
                W=prep1["W"],
                H=prep1["H"],
            )
            log2 = self._obs_fn_2(
                particles,
                base_dets=prep2["base_dets"],
                name2pt=prep2["name2pt"],
                robot_arm=self.tr2.robot_arm,
                K=self.tr2.K,
                dist=self.tr2.dist,
                shaft_pos_b=prep2["shaft_pos_b"],
                shaft_dir_b=prep2["shaft_dir_b"],
                detected_edges_hough=prep2["detected_edges_hough"],
                W=prep2["W"],
                H=prep2["H"],
            )
            return (log1 + log2) / variance_scale

        self.pf.updateStep_vec(_joint_obs)
        self.delta_T_state = self.pf.getMeanParticle()

        R_kin1 = self.tr1.cam_T_b_kin[:3, :3]
        t_kin1 = self.tr1.cam_T_b_kin[:3, 3]
        R_delta, _ = cv2.Rodrigues(self.delta_T_state[3:].reshape(3, 1))
        t_delta = self.delta_T_state[:3]

        self.tr1.cam_T_b = self.tr1.cam_T_b.copy()
        self.tr1.cam_T_b[:3, :3] = R_kin1 @ R_delta
        self.tr1.cam_T_b[:3, 3] = R_kin1 @ t_delta + t_kin1
        self.tr2.cam_T_b = self.tr1.cam_T_b @ self.b1_T_b2

        self.tr1.pose_state = self.delta_T_state
        self.tr2.pose_state = self.delta_T_state

        self.last_n_dets = {
            self.tr1.name: len(prep1["base_dets"]),
            self.tr2.name: len(prep2["base_dets"]),
        }
        return {self.tr1.name: prep1["corrected"], self.tr2.name: prep2["corrected"]}

    def draw(self, vis_img, corrected_by_arm):
        self.tr1.draw(vis_img, corrected_by_arm[self.tr1.name])
        self.tr2.draw(vis_img, corrected_by_arm[self.tr2.name])
