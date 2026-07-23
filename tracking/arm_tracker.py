"""ArmTracker: per-arm particle-filter tracking, shared by offline and realtime pipelines."""

import os

import cv2
import numpy as np
import yaml

from core.RobotLink import RobotLink
from core.ParticleFilter import ParticleFilter
from core.utils import axisAngleToRotationMatrix

from .constants import (
    LIKELIHOOD_THRESH_OPT,
    DELTA_PROCESS_STD,
    SHAFT_EDGE_EMA_ALPHA,
    SHAFT_USE_HOUGH,
)
from .kinematics import load_kin_data
from .label_assignment import assign_labels, LR_BASES, FB_BASES
from .observation import pf_init, pf_motion, create_combined_obs_model_vec
from .visualization import project_skeleton, draw_keypoints
from .canny_hough import process as _hough_process, get_roi as _hough_get_roi

# Fallback Canny+HoughLines shaft-edge params, per arm -- used only when a
# dataset doesn't have its own tuned canny_params_{arm}.json (see
# offline_tracking.py). Schema matches tracking/canny_hough.py's process(),
# the same detector test_canny_hough.py's interactive tuner uses, so any
# dataset-specific tuned params drop in directly via the `shaft_params`
# constructor arg.
_SHAFT_CANNY_PARAMS = {
    "psm1": {
        "blur": 2,
        "canny_lo": 49,
        "canny_hi": 92,
        "do_skel": False,
        "hough_thr": 57,
        "min_w": 38,
        "max_w": 60,
        "angle_tol": 18,
        "temporal_tol": 29,
        "par_tol": 4,
        "roi_pad": 140,
        "edge_margin": 0,
    },
    "psm2": {
        "blur": 2,
        "canny_lo": 25,
        "canny_hi": 103,
        "do_skel": False,
        "hough_thr": 34,
        "min_w": 43,
        "max_w": 63,
        "angle_tol": 36,
        "temporal_tol": 15,
        "par_tol": 5,
        "roi_pad": 79,
        "edge_margin": 30,
    },
}


def _get_detections(source, frame_idx: int, img):
    """Retrieve keypoint detections from either a pre-loaded dict or a DLCRunner.

    Supports two source types:
      - dict  {frame_idx: {label: (x, y, likelihood)}} — offline CSV data.
      - DLCRunner with an .infer(img) method — realtime inference.
    """
    if source is None:
        return {}
    if hasattr(source, "infer"):
        return source.infer(img) if img is not None else {}
    return source.get(frame_idx, {})


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
        self._last_state_label = ""

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

        self.curr_lr = "left"
        self.curr_fb = "back"

        # Shaft (cylinder) feature for the edge-likelihood term, if this arm has one.
        self._shaft_params = (
            shaft_params if shaft_params is not None else _SHAFT_CANNY_PARAMS.get(name)
        )
        shaft_pos_list, shaft_dir_list, _, shaft_radius_list = self.robot_arm.getShaftFeatures()
        self._has_shaft = self._shaft_params is not None and len(shaft_pos_list) > 0
        self._shaft_pos_b = shaft_pos_list[0] if self._has_shaft else None
        self._shaft_dir_b = shaft_dir_list[0] if self._has_shaft else None
        self._shaft_radius = shaft_radius_list[0] if self._has_shaft else None

        # PF state = [dtx, dty, dtz, drx, dry, drz], zero = identity
        state_init = np.zeros(6)
        self._vec_obs_func = create_combined_obs_model_vec(
            lambda: self.cam_T_b_kin, self._shaft_radius
        )

        # Detected shaft edge segments in full-image coords (for drawing)
        self._last_detected_segs: list = []
        # EMA of detected Hough lines [(rho1,th1),(rho2,th2)] sorted by rho, full-image coords
        # (drawing / detection-dropout fallback only — NOT fed to the PF likelihood)
        self._ema_lines = None
        # Raw (theta, rho_c) pair from THIS frame's detection, image-centre-origin Hough
        # convention matching shaft_edge.py — fed directly into the PF likelihood, with no
        # temporal smoothing (that's the filter's job, not the detector's).
        self._last_raw_hough = None
        # Last successful pair selection, full-image (rho, theta) -- used as
        # _select_two_polar's temporal prior for the NEXT frame (position +
        # width matching), same mechanism as test_canny_hough.py's tuner.
        # Separate from _ema_lines (which is smoothed and used for drawing/
        # dropout-fallback) so a bad EMA doesn't bias which pair gets picked.
        self._prev_final_global = None

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

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        if raw_l is None:
            raw_l = _get_detections(self.dlc_left, frame_idx, left_img)
        if raw_r is None:
            raw_r = _get_detections(self.dlc_right, frame_idx, right_img)

        if joint_angles is None:
            try:
                joint_angles, _ = load_kin_data(self.kin_dir, frame_idx)
            except Exception:
                return None, 0, {}

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

        corrected, self.curr_lr, self.curr_fb, self._last_state_label = assign_labels(
            raw_l,
            raw_r,
            self.name2pt,
            self.P1,
            self.P2,
            self.b_T_cam_kin,
            self.K,
            self.dist,
            self.robot_arm,
            self.cam_T_b_kin,
            self.curr_lr,
            self.curr_fb,
            cam_T_b=self.cam_T_b,
        )

        # Raw per-joint detections (LIKELIHOOD_THRESH_OPT applied), keyed by the
        # generic joint name — left/right, front/back, and yaw-tip identity is
        # NOT resolved here. That's left to the observation model, which
        # resolves it per particle from that particle's own hypothesised pose
        # rather than committing to one assignment
        # per frame the way `corrected` (from assign_labels above) does.
        base_dets = {}
        for base in LR_BASES | FB_BASES:
            best = None
            for cand_key in (
                f"{base}:left",
                f"{base}:right",
                f"{base}:front",
                f"{base}:back",
                base,
            ):
                det = raw_l.get(cand_key)
                if det is not None and det[2] >= LIKELIHOOD_THRESH_OPT:
                    if best is None or det[2] > best[2]:
                        best = det
            if best is not None:
                base_dets[base] = best
        for yaw in ("yaw_1", "yaw_2"):
            det = raw_l.get(yaw)
            if det is not None and det[2] >= LIKELIHOOD_THRESH_OPT:
                base_dets[yaw] = det

        # Canny+Hough shaft edge detection
        if self._shaft_params is not None and left_img is not None and SHAFT_USE_HOUGH:
            self._run_canny_hough(left_img, corrected)
        else:
            self._last_detected_segs = []
            self._last_raw_hough = None

        self.pf.predictionStep_vec(std=DELTA_PROCESS_STD)
        H_img, W_img = left_img.shape[:2] if left_img is not None else (None, None)
        self.pf.updateStep_vec(
            self._vec_obs_func,
            base_dets=base_dets,
            name2pt=self.name2pt,
            robot_arm=self.robot_arm,
            K=self.K,
            dist=self.dist,
            shaft_pos_b=self._shaft_pos_b,
            shaft_dir_b=self._shaft_dir_b,
            detected_edges_hough=self._last_raw_hough,
            W=W_img,
            H=H_img,
        )

        self.pose_state = self.pf.getMeanParticle()

        R_kin = self.cam_T_b_kin[:3, :3]
        t_kin = self.cam_T_b_kin[:3, 3]

        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        t_delta = self.pose_state[:3]
        self.cam_T_b = self.cam_T_b.copy()
        self.cam_T_b[:3, :3] = R_kin @ R_delta
        self.cam_T_b[:3, 3] = R_kin @ t_delta + t_kin

        return corrected, len(base_dets), joint_angles

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
            raw_l = _get_detections(self.dlc_left, frame_idx, None)
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

        if left_img_dir is not None and self._shaft_params is not None:

            def _offline_frames():
                for frame_idx in frame_list:
                    left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
                    if left_img is None:
                        continue
                    raw_l = _get_detections(self.dlc_left, frame_idx, None)
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

        if self._shaft_params is not None:

            def _live_frames():
                for left_img, joint_angles, raw_l in frame_list:
                    if left_img is None:
                        continue
                    yield left_img, joint_angles, raw_l

            self._correct_shaft_angle(_live_frames(), init_std)

    def _pnp_delta_estimate(self, raw_l, joint_angles, min_pts, pnp_error, lkl_thresh):
        """Estimate delta_T for a single frame via PnP (shared by warmup() and
        warmup_live()). Mutates self.name2pt as a side effect (refreshed to
        match this frame's joint angles).

        Returns a (6,) [t, rvec] delta_T array, or None if this frame doesn't
        yield a usable estimate (too few correspondences, or PnP failure).
        """
        if self._j6_offset:
            joint_angles = joint_angles.copy()
            joint_angles[5] += self._j6_offset
        self.robot_arm.updateJointAngles(joint_angles)
        pts_3d, names = self.robot_arm.getPointFeatures()
        self.name2pt = dict(zip(names, pts_3d))

        corrected, _, _, _ = assign_labels(
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
            self.curr_lr,
            self.curr_fb,
            cam_T_b=self.cam_T_b,
        )

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

        self._ema_lines = None
        for left_img, joint_angles, raw_l in frame_source:
            if left_img is None:
                continue

            if self._j6_offset:
                joint_angles = joint_angles.copy()
                joint_angles[5] += self._j6_offset
            self.robot_arm.updateJointAngles(joint_angles)
            pts_3d, names = self.robot_arm.getPointFeatures()
            self.name2pt = dict(zip(names, pts_3d))

            corrected, _, _, _ = assign_labels(
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
                self.curr_lr,
                self.curr_fb,
                cam_T_b=self.cam_T_b,
            )
            self._run_canny_hough(left_img, corrected)

        if self._ema_lines is None:
            print(f"[{self.name}] warmup: no shaft edges detected — angle uncorrected")
            return

        shaft_pos_list, shaft_dir_list, _, _ = self.robot_arm.getShaftFeatures()
        if len(shaft_pos_list) == 0:
            return
        dir_b = shaft_dir_list[0] / np.linalg.norm(shaft_dir_list[0])

        # EMA midline angle (Hough NORMAL-angle convention, from cv2.HoughLines).
        thetas = [th for _, th in self._ema_lines]
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
    def _hough_to_seg(rho, theta, h, w):
        """Clip HoughLines (rho, theta) infinite line to ROI rectangle."""
        a, b = np.cos(theta), np.sin(theta)
        pts = []
        for xb in (0, w - 1):
            if abs(b) > 1e-6:
                yb = (rho - xb * a) / b
                if 0 <= yb <= h - 1:
                    pts.append((xb, int(round(yb))))
        for yb in (0, h - 1):
            if abs(a) > 1e-6:
                xb = (rho - yb * b) / a
                if 0 <= xb <= w - 1:
                    pts.append((int(round(xb)), yb))
        if len(pts) < 2:
            return None
        return (pts[0][0], pts[0][1], pts[-1][0], pts[-1][1])

    def _ema_segs_from_lines(self, off_x, off_y, h_roi, w_roi):
        """Return segments from EMA Hough lines, or [] if EMA not yet initialised."""
        if self._ema_lines is None:
            return []
        H_full = h_roi + off_y
        W_full = w_roi + off_x
        segs = []
        for rho, theta in self._ema_lines:
            seg = self._hough_to_seg(rho, theta, H_full, W_full)
            if seg is not None:
                segs.append(seg)
        return segs if len(segs) == 2 else []

    def _run_canny_hough(self, left_img: np.ndarray, corrected: dict):
        """Detect shaft edges with the shared Canny+HoughLines pipeline
        (tracking/canny_hough.py) and store the result in
        _last_detected_segs / _last_raw_hough.

        Uses the exact same process()/get_roi() as test_canny_hough.py's
        interactive tuner, so self._shaft_params (loaded from a dataset's
        canny_params_{arm}.json, or _SHAFT_CANNY_PARAMS as a fallback)
        actually takes effect here instead of a separately hand-ported
        approximation drifting out of sync with the tuner.
        """
        p = self._shaft_params
        H, W = left_img.shape[:2]

        # ROI centre: highest-likelihood keypoint
        if corrected:
            best = max(corrected.values(), key=lambda v: v[2])
            tx, ty = int(round(best[0])), int(round(best[1]))
        else:
            self._last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
            self._last_raw_hough = None
            return
        if not (0 <= tx < W and 0 <= ty < H):
            self._last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
            self._last_raw_hough = None
            return

        roi, off_x, off_y = _hough_get_roi(
            left_img, (tx, ty), p["roi_pad"], self.name, H, W, p.get("edge_margin", 0)
        )
        h_roi, w_roi = roi.shape[:2]

        if roi.size == 0:
            self._last_detected_segs = self._ema_segs_from_lines(off_x, off_y, h_roi, w_roi)
            self._last_raw_hough = None
            return

        # FK shaft-angle prior (fixed kinematic extrinsic -- same convention
        # as test_canny_hough.py's main()), used only to restrict Hough's
        # angle search band, not to draw or correct the tracked pose.
        expected_angle = None
        if self._shaft_dir_b is not None:
            dir_b = self._shaft_dir_b / np.linalg.norm(self._shaft_dir_b)
            d_cam = self.cam_T_b_kin[:3, :3] @ dir_b
            if d_cam[2] < 0:
                d_cam = -d_cam
            if d_cam[2] > 1e-6:
                theta_pred_dir = np.arctan2(
                    self.K[1, 1] * d_cam[1] / d_cam[2], self.K[0, 0] * d_cam[0] / d_cam[2]
                )
                expected_angle = np.degrees(theta_pred_dir) % 180

        params = dict(p)
        params["expected_angle"] = expected_angle
        if self._prev_final_global is not None:
            params["prior_rhos"] = [
                rho_g - (off_x * np.cos(theta_g) + off_y * np.sin(theta_g))
                for rho_g, theta_g in self._prev_final_global
            ]
        else:
            params["prior_rhos"] = None

        _, _, _, final = _hough_process(roi, params)

        if len(final) < 2:
            # No detection: fall back to last EMA value if available. Keep
            # _prev_final_global as-is (don't reset the temporal prior on a
            # single missed frame).
            self._last_detected_segs = self._ema_segs_from_lines(off_x, off_y, h_roi, w_roi)
            self._last_raw_hough = None
            return

        full_lines = [
            (rho_l + off_x * np.cos(theta_l) + off_y * np.sin(theta_l), theta_l)
            for rho_l, theta_l in final
        ]
        full_lines.sort(key=lambda x: x[0])
        self._prev_final_global = full_lines

        # This frame's raw (theta, rho_c) detection for the PF likelihood — image-centre-origin
        # convention (matches shaft_edge.py's line_to_hough), NOT the EMA-smoothed line below.
        # Fed straight into the filter with no temporal smoothing of its own.
        self._last_raw_hough = [
            (theta, rho - (W / 2) * np.cos(theta) - (H / 2) * np.sin(theta))
            for rho, theta in full_lines
        ]

        # EMA update (drawing / dropout-fallback only, not fed to the PF likelihood)
        a = SHAFT_EDGE_EMA_ALPHA
        if self._ema_lines is None:
            self._ema_lines = full_lines
        else:
            smoothed = []
            for (re, te), (rn, tn) in zip(self._ema_lines, full_lines):
                # Wrap theta to avoid discontinuity near 0/pi boundary
                if abs(tn - te) > np.pi / 2:
                    tn = tn - np.pi if tn > te else tn + np.pi
                smoothed.append(((1 - a) * re + a * rn, (1 - a) * te + a * tn))
            self._ema_lines = smoothed

        # Convert smoothed Hough lines back to segments for obs model
        segs = []
        H_full, W_full = left_img.shape[:2]
        for rho, theta in self._ema_lines:
            seg = self._hough_to_seg(rho, theta, H_full, W_full)
            if seg is not None:
                segs.append(seg)
        self._last_detected_segs = segs if len(segs) == 2 else []

    def draw(self, vis_img: np.ndarray, corrected: dict):
        """Draw skeleton, keypoints, and the detected shaft edges.

        The FK skeleton's shaft segment is replaced with a single merged
        line from the detected edge midline when available (smoothed,
        FK-anchored — shows the filter's belief). The two raw Canny+Hough
        detected edges (self._last_detected_segs, EMA-smoothed but otherwise
        exactly what the detector found) are drawn separately in cyan, so
        detection quality/failures are visible directly instead of being
        hidden behind the FK-anchored reconstruction.
        """
        use_ema_shaft = self._shaft_params is not None and self._ema_lines is not None

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
        # self._last_raw_hough).
        if use_ema_shaft:
            H, W = vis_img.shape[:2]
            thetas = [th for _, th in self._ema_lines]
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
        for x1, y1, x2, y2 in self._last_detected_segs:
            cv2.line(vis_img, (x1, y1), (x2, y2), (255, 255, 0), 1)

        draw_keypoints(vis_img, corrected, self.kp_color)
