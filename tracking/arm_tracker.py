"""ArmTracker: per-arm particle-filter tracking, shared by offline and realtime pipelines."""

import os

import cv2
import numpy as np
import yaml

from core.RobotLink import RobotLink
from core.ParticleFilter import ParticleFilter
from core.utils import axisAngleToRotationMatrix

from .constants import (

        DELTA_PROCESS_STD,
        SYNTHETIC_OBS_LIKELIHOOD,
        SYNTHETIC_OBS_DECAY_TAU,
        SYNTHETIC_OBS_EDGE_WEIGHT,
        SHAFT_TRANSLATION_DAMPING,
        SHAFT_CORRECTION_MAX_COND,
        SHAFT_CORRECTION_MAX_MM,
        BLIND_PROCESS_STD_GROWTH_RATE,
        BLIND_PROCESS_STD_MAX_MULTIPLIER,
)
from .kinematics import load_kin_data
from .observation import pf_init, pf_motion, create_combined_obs_model_vec
from .visualization import project_skeleton, draw_keypoints
from .feature_detection import ArmFeatureDetector
from .shaft_edge import predict_shaft_edges_hough_batch
from .combined_tracker import _reanchor_particle_block

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
        own_pf: bool = True,
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
            own_pf:      When True (default), this tracker owns and drives its
                         own 6-state ParticleFilter (used standalone). When
                         False, no PF is created here -- an external combined
                         tracker drives a shared, larger PF and calls
                         prepare_frame()/finalize_pose() directly (see
                         tracking/combined_tracker.py).
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
        # Explicit frame-local anchor used by advisor Method 3.  It is kept
        # separate from cam_T_b_kin because the latter defines the lump error
        # and must remain the fixed hand-eye calibration.
        self._method3_anchor = None
        # Hybrid mode (see set_hybrid_primary()/update()): the tracker to
        # couple to ONLY while this arm's own detector is blind. Distinct
        # from primary_tracker, which reflects whether the coupling is
        # CURRENTLY active -- this holds the candidate to (re)couple to the
        # moment blindness starts, so update() can toggle set_relative_to()/
        # set_independent() automatically frame to frame. None = hybrid mode
        # off (this arm's primary_tracker, if any, is fixed for the whole
        # run via the plain set_relative_to() call site instead).
        self._primary_candidate = None
        # Soft-coupling mode (see set_soft_coupling()): unlike
        # _primary_candidate/primary_tracker, this arm's own anchor is
        # NEVER switched -- _soft_coupling_primary just names the tracker
        # to filter a relative-offset estimate against, and
        # _T_rel_filtered holds that slowly-EMA'd estimate (None until the
        # filter has seen its first good sample).
        self._soft_coupling_primary = None
        self._T_rel_filtered = None
        self._rel_filter_alpha = 0.02
        # Consecutive frames this arm's OWN detector has found nothing (see
        # update()'s synthetic-observation fallback / _blind_synthetic_likelihood()).
        # Reset to 0 the moment a real detection succeeds again.
        self._blind_frames = 0
        # pose_state as of the most recent frame this arm's OWN detector
        # found something -- what _synthetic_base_dets_from_primary() holds
        # onto instead of assuming delta_T=0 while blind (see its docstring).
        # Updated alongside _blind_frames in update().
        self._last_good_pose_state = np.zeros(6)
        self._pending_std = DELTA_PROCESS_STD.copy()
        # Warmup confidence bookkeeping (see _apply_warmup_estimates()) --
        # read by CombinedArmTracker's relative_to="auto" to pick whichever
        # arm's warmup was more reliable as the initial primary. Deliberately
        # separate from _pending_std: that one is floored at 3x
        # DELTA_PROCESS_STD (a clean warmup on both arms then floors BOTH to
        # the same value, destroying exactly the signal needed here), so
        # _pending_iqr_raw keeps the un-floored spread for scoring only.
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
        self._vec_obs_func = create_combined_obs_model_vec(self._anchor_T, self._shaft_radius)
        self.observation_variance_scale = 1.0

        self._own_pf = own_pf
        if own_pf:
            self.pf = ParticleFilter(
                num_states=6,
                initialDistributionFunc=pf_init,
                motionModelFunc=pf_motion,
                obsModelFunc=None,
                num_particles=n_particles,
                min_num_effective_particles=n_particles * 0.5,
            )
            self.pf.initializeFilter(mean_state=state_init, std=DELTA_PROCESS_STD)
        else:
            self.pf = None
        self.pose_state = state_init.copy()

    def _anchor_T(self):
        """The fixed pose that pose_state's delta_T is applied on top of.

        Normally this arm's own fixed hand-eye extrinsic. After
        set_relative_to() is called, it instead tracks the primary arm's
        live pose composed with the (once-computed) relative offset —
        every other part of the PF/observation pipeline is unaware of the
        difference, since this is the only thing that changes.
        """
        if self._method3_anchor is not None:
            return self._method3_anchor
        if self.primary_tracker is None:
            return self.cam_T_b_kin
        return self.primary_tracker.cam_T_b @ self.T_rel_bar

    def _abs_pose(self):
        """This arm's current absolute (camera-frame) pose, i.e.
        _anchor_T() @ delta_T(pose_state) -- computed from whatever anchor
        is CURRENTLY in effect (own fixed extrinsic, or a primary's live
        pose), not assumed to be cam_T_b_kin. Used by set_relative_to()/
        set_independent() so re-anchoring is correct even when this arm is
        already in relative mode (e.g. switching to a different primary).
        """
        anchor = self._anchor_T()
        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        abs_pose = anchor.copy()
        abs_pose[:3, :3] = anchor[:3, :3] @ R_delta
        abs_pose[:3, 3] = anchor[:3, :3] @ self.pose_state[:3] + anchor[:3, 3]
        return abs_pose

    def set_relative_to(self, primary_tracker):
        """Switch this tracker to track relative to primary_tracker's live
        pose instead of this arm's own fixed hand-eye anchor (or a
        different primary it was previously relative to).

        Call after both this tracker's warmup() and primary_tracker's
        warmup() have completed (or, for a later dynamic switch, at any
        point during tracking -- see CombinedArmTracker.set_primary(),
        which also re-anchors this arm's particle block in the shared PF
        to match). Converts the current absolute pose estimate (from
        whatever anchor is in effect right now) into a relative offset
        from the primary's current pose, so tracked pose is unchanged at
        the moment of the switch — only the anchor that future frames'
        delta_T is estimated around changes.

        When this tracker owns its own PF (own_pf=True), the existing
        particle cloud is re-expressed under the new anchor via an exact
        SE(3) relabelling (_reanchor_particle_block(), the same technique
        combined_tracker.py's set_primary() uses for the shared-PF
        architecture) instead of being reseeded from scratch around
        _pending_std. A full initializeFilter() discards whatever spread/
        shape the filter has actually converged to and replaces it with a
        generic warmup-time prior -- fine for a one-time switch, but costly
        if this is called repeatedly (see set_hybrid_primary()), since every
        switch would otherwise pay a full reconvergence cost even for a
        filter that was already tracking confidently.
        """
        old_anchor = self._anchor_T()
        abs_pose = self._abs_pose()

        self.primary_tracker = primary_tracker
        self.T_rel_bar = np.linalg.inv(primary_tracker.cam_T_b) @ abs_pose
        new_anchor = self._anchor_T()

        self.pose_state = np.zeros(6)
        if self._own_pf:
            self.pf._particles = _reanchor_particle_block(self.pf._particles, old_anchor, new_anchor)

        t_mm = self.T_rel_bar[:3, 3] * 1000
        r_deg = np.rad2deg(cv2.Rodrigues(self.T_rel_bar[:3, :3])[0].flatten())
        print(
            f"[{self.name}] now tracked relative to [{primary_tracker.name}]: "
            f"T_rel = t=[{t_mm[0]:.1f},{t_mm[1]:.1f},{t_mm[2]:.1f}]mm "
            f"r=[{r_deg[0]:.1f},{r_deg[1]:.1f},{r_deg[2]:.1f}]deg"
        )

    def set_independent(self):
        """Revert this tracker to its own fixed hand-eye anchor (inverse of
        set_relative_to()). No-op if it's already independent. Preserves
        the current absolute pose exactly (no jump) -- only the anchor
        pose_state is expressed relative to changes. Reanchors the particle
        cloud in place (own_pf=True) rather than reseeding it -- see
        set_relative_to()'s docstring.
        """
        if self.primary_tracker is None:
            return
        old_anchor = self._anchor_T()
        abs_pose = self._abs_pose()

        self.primary_tracker = None
        self.T_rel_bar = np.eye(4)

        R_kin = self.cam_T_b_kin[:3, :3]
        t_kin = self.cam_T_b_kin[:3, 3]
        R_new_delta = R_kin.T @ abs_pose[:3, :3]
        t_new_delta = R_kin.T @ (abs_pose[:3, 3] - t_kin)
        rvec_new, _ = cv2.Rodrigues(R_new_delta)
        self.pose_state = np.concatenate([t_new_delta, rvec_new.flatten()])
        if self._own_pf:
            new_anchor = self._anchor_T()
            self.pf._particles = _reanchor_particle_block(self.pf._particles, old_anchor, new_anchor)

        print(f"[{self.name}] now tracked independently (own hand-eye anchor)")

    def set_hybrid_primary(self, primary_tracker):
        """Enable hybrid mode: this arm tracks fully independently (own
        fixed hand-eye anchor, own PF, unaffected by primary_tracker) for
        as long as its own detector keeps finding real detections, and only
        couples to primary_tracker's live pose while blind -- see update(),
        which calls set_relative_to()/set_independent() automatically each
        frame as had_real_obs flips.

        Unlike the plain set_relative_to() call site (which couples once,
        for the whole run), the relative offset here is recomputed FRESH
        every time a blind stretch begins, from whatever this arm's own
        independent tracking had most recently converged to -- not a
        possibly long-stale value from run start. This is deliberately an
        alternative to (not on top of) a call to set_relative_to(): call
        this once after both arms' warmup() instead, and leave this arm
        independent (primary_tracker stays None) until it actually goes
        blind.
        """
        self._primary_candidate = primary_tracker

    def _synthetic_base_dets_from_primary(self, likelihood=SYNTHETIC_OBS_LIKELIHOOD):
        """When this (secondary) arm's OWN detector found nothing this
        frame but primary_tracker DID just get corrected, synthesize
        base_dets from this arm's own FK points (self.name2pt, already
        refreshed this frame from the current joint angles) projected
        through primary_tracker.cam_T_b @ self.T_rel_bar @
        delta_T(self._last_good_pose_state) -- i.e. the rigid-offset
        anchor PLUS this arm's own last real correction (as of the most
        recent frame its own detector actually found something), NOT a
        bare delta_T=0 assumption. Using delta_T=0 would silently discard
        whatever real residual correction (translation especially -- see
        update()'s _last_good_pose_state bookkeeping) this arm's own PF
        had legitimately converged to before going blind, snapping the
        prediction back toward the raw uncorrected rigid offset instead of
        holding the best available estimate steady.

        Feeding this into this arm's own updateStep_vec (in place of an
        empty base_dets) lets its own PF still do a real Bayesian
        update/ESS-triggered resample pulling toward that assumption,
        instead of contributing zero observation weight -- which leaves
        weights uniform, resampling never triggers, and pose_state just
        random-walks unconstrained under process noise alone (see
        core/ParticleFilter.py's updateStep_vec).

        likelihood is deliberately well below a typical real detection's
        (often 0.7-0.9+): a real detection should always outweigh this
        fallback whenever one becomes available again.

        Returns {} if this arm isn't in relative mode (primary_tracker is
        None), matching base_dets' empty-dict convention.
        """
        if self.primary_tracker is None:
            return {}
        anchor = self._anchor_T()
        R_delta, _ = cv2.Rodrigues(self._last_good_pose_state[3:].reshape(3, 1))
        t_delta = self._last_good_pose_state[:3]
        predicted_pose = anchor.copy()
        predicted_pose[:3, :3] = anchor[:3, :3] @ R_delta
        predicted_pose[:3, 3] = anchor[:3, :3] @ t_delta + anchor[:3, 3]
        return self._project_base_dets(predicted_pose, likelihood)

    def _project_base_dets(self, predicted_pose, likelihood):
        """Project this arm's own FK points (self.name2pt) through a given
        4x4 camera-frame pose, producing a base_dets dict in the same
        {label: (x, y, likelihood)} shape as a real detection. Shared by
        _synthetic_base_dets_from_primary() (hybrid mode) and
        _synthetic_base_dets_from_filtered_primary() (soft-coupling mode) --
        the only difference between them is which pose gets passed in.
        """
        rvec, _ = cv2.Rodrigues(predicted_pose[:3, :3])
        tvec = predicted_pose[:3, 3].reshape(3, 1)

        base_dets = {}
        for base in LR_BASES | FB_BASES:
            face = self.feature_detector.curr_lr if base in LR_BASES else self.feature_detector.curr_fb
            pt3d = self.name2pt.get(f"{base}:{face}")
            if pt3d is None:
                continue
            pix, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, tvec, self.K, self.dist)
            base_dets[base] = (float(pix[0, 0, 0]), float(pix[0, 0, 1]), likelihood)
        for yaw in ("yaw_1", "yaw_2"):
            pt3d = self.name2pt.get(yaw)
            if pt3d is None:
                continue
            pix, _ = cv2.projectPoints(pt3d.reshape(1, 1, 3), rvec, tvec, self.K, self.dist)
            base_dets[yaw] = (float(pix[0, 0, 0]), float(pix[0, 0, 1]), likelihood)
        return base_dets

    def set_soft_coupling(self, primary_tracker, alpha=0.02):
        """Alternative to set_hybrid_primary(): this arm's own anchor
        (cam_T_b_kin) and own PF are NEVER touched -- no set_relative_to()/
        set_independent() call, ever, so no anchor switching and no
        particle reinit/reanchor cost regardless of how often blindness
        starts and stops (the cost that remained even after reanchoring
        set_relative_to()/set_independent() in place -- see those methods'
        docstrings -- because *switching itself*, not just the reinit, was
        injecting noise on short/frequent natural dropouts).

        Instead, while both this arm and primary_tracker have a real
        detection this frame, a slowly-filtered estimate of the relative
        offset between them (self._T_rel_filtered, EMA'd over many good
        frames via _update_rel_filter(), NOT a live or single-frame
        snapshot) is maintained. While this arm is blind, that filtered
        offset projects a synthetic pseudo-observation (see
        _synthetic_base_dets_from_filtered_primary()), added into this
        arm's own independent PF as a low-weight extra observation term --
        the same synthetic-observation/decay/edge-weight machinery
        set_hybrid_primary() uses, just sourced from a filtered cross-arm
        estimate instead of a re-anchored PF.
        """
        self._soft_coupling_primary = primary_tracker
        self._rel_filter_alpha = alpha

    def _update_rel_filter(self):
        """EMA-update self._T_rel_filtered from this frame's (trusted --
        both arms had a real detection, see update()'s call site) absolute
        poses. Rotation is blended in rotation-vector space, which is only
        strictly valid for slowly-varying/near-constant targets, not
        arbitrary large rotations -- accepted here since the whole premise
        of this filter (see set_soft_coupling()) is that the true relative
        offset between two arms mounted on the same cart is close to
        constant, so consecutive good samples should already be close.
        """
        primary = self._soft_coupling_primary
        sample = np.linalg.inv(primary.cam_T_b) @ self.cam_T_b
        if self._T_rel_filtered is None:
            self._T_rel_filtered = sample
            return
        a = self._rel_filter_alpha
        r_old = cv2.Rodrigues(self._T_rel_filtered[:3, :3])[0].flatten()
        r_new = cv2.Rodrigues(sample[:3, :3])[0].flatten()
        r_blend = (1 - a) * r_old + a * r_new
        t_blend = (1 - a) * self._T_rel_filtered[:3, 3] + a * sample[:3, 3]
        R_blend, _ = cv2.Rodrigues(r_blend.reshape(3, 1))
        T = np.eye(4)
        T[:3, :3] = R_blend
        T[:3, 3] = t_blend
        self._T_rel_filtered = T

    def _synthetic_base_dets_from_filtered_primary(self, likelihood):
        """Soft-coupling counterpart of _synthetic_base_dets_from_primary():
        projects through primary.cam_T_b @ self._T_rel_filtered (the slowly
        -filtered cross-arm estimate) instead of this arm's own live
        anchor/T_rel_bar/_last_good_pose_state -- because in soft-coupling
        mode this arm never switches anchor, so there is no "relative to
        primary" delta_T state to fall back on. Returns {} if soft coupling
        isn't enabled or the filter hasn't seen a single good sample yet.
        """
        if self._soft_coupling_primary is None or self._T_rel_filtered is None:
            return {}
        predicted_pose = self._soft_coupling_primary.cam_T_b @ self._T_rel_filtered
        return self._project_base_dets(predicted_pose, likelihood)

    def _shaft_translation_correction(self, W, H):
        """Estimate a 3-D translation correction (perpendicular to the
        shaft axis only -- translation ALONG the axis or roll about it
        don't change a cylinder's silhouette, the classic aperture
        problem) from this frame's fresh shaft-edge detection
        (self.feature_detector.last_raw_hough), via a local linearisation (finite-
        difference Jacobian) of predict_shaft_edges_hough_batch() around
        the currently held pose -- then applies only
        SHAFT_TRANSLATION_DAMPING of it, and to the WHOLE particle cloud
        (re-centering it, not collapsing its spread) rather than a hard
        overwrite of the held mean alone. This lets the PF's own
        point+edge likelihood keep validating/refining the correction on
        subsequent frames -- an undamped single-step version tried
        earlier committed fully to each frame's estimate, which measured
        worse run-to-run variance than not correcting at all whenever a
        single frame's edge match or the local linearisation was off.

        Mutates self.pf._particles' translation columns and
        self._last_good_pose_state[:3] in place (both shifted by the same
        damped amount, so they stay consistent with each other). No-op if
        there's no shaft model, no primary_tracker (not a coupled
        secondary arm), or no fresh 2-line detection this frame.
        """
        if not self._has_shaft or self.primary_tracker is None or self.feature_detector.last_raw_hough is None:
            return
        if len(self.feature_detector.last_raw_hough) != 2:
            return

        anchor = self._anchor_T()
        R_delta, _ = cv2.Rodrigues(self._last_good_pose_state[3:].reshape(3, 1))
        t_delta = self._last_good_pose_state[:3]
        R_eff = anchor[:3, :3] @ R_delta
        t_eff = anchor[:3, :3] @ t_delta + anchor[:3, 3]

        p_c_0 = R_eff @ self._shaft_pos_b + t_eff
        d_c = R_eff @ self._shaft_dir_b
        d_c = d_c / np.linalg.norm(d_c)

        pred_edges, valid = predict_shaft_edges_hough_batch(
            p_c_0[None, :], d_c[None, :], self._shaft_radius, W, H, self.K
        )
        if not valid[0]:
            return
        p1_r, p2_r = pred_edges[0, 0, 1], pred_edges[0, 1, 1]

        det_t1, det_r1 = self.feature_detector.last_raw_hough[0]
        det_t2, det_r2 = self.feature_detector.last_raw_hough[1]

        # Straight/swap edge-label matching, same criterion as
        # shaft_edge.py's lumped_edge_log_likelihood -- which detected line
        # is "edge 1" vs "edge 2" isn't fixed, so match whichever pairing
        # is closer in rho before taking the residual.
        cost_straight = abs(p1_r - det_r1) + abs(p2_r - det_r2)
        cost_swapped = abs(p1_r - det_r2) + abs(p2_r - det_r1)
        target_r1, target_r2 = (det_r1, det_r2) if cost_straight <= cost_swapped else (det_r2, det_r1)
        d_rho = np.array([target_r1 - p1_r, target_r2 - p2_r])

        # Two directions perpendicular to d_c (Gram-Schmidt from an
        # arbitrary vector not parallel to d_c) -- span the plane
        # translation perpendicular to the shaft can move in; translation
        # along d_c itself is excluded (see docstring).
        ref = np.array([1.0, 0.0, 0.0]) if abs(d_c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_hat = ref - np.dot(ref, d_c) * d_c
        u_hat /= np.linalg.norm(u_hat)
        v_hat = np.cross(d_c, u_hat)

        eps = 0.001  # 1 mm finite-difference step
        J = np.zeros((2, 2))
        for i, dir_hat in enumerate((u_hat, v_hat)):
            pred_pert, valid_pert = predict_shaft_edges_hough_batch(
                (p_c_0 + eps * dir_hat)[None, :], d_c[None, :], self._shaft_radius, W, H, self.K
            )
            if not valid_pert[0]:
                return
            J[0, i] = (pred_pert[0, 0, 1] - p1_r) / eps
            J[1, i] = (pred_pert[0, 1, 1] - p2_r) / eps

        # Reject an ill-conditioned Jacobian outright -- the two edges'
        # directions are too close to parallel here to reliably separate
        # the two perpendicular translation components, so solving would
        # amplify any small detection/linearisation error into a huge,
        # essentially arbitrary correction.
        if np.linalg.cond(J) > SHAFT_CORRECTION_MAX_COND:
            return

        try:
            duv = np.linalg.solve(J, d_rho)
        except np.linalg.LinAlgError:
            return

        delta_t_cam = duv[0] * u_hat + duv[1] * v_hat
        # Reject an implausibly large single-frame correction regardless --
        # a legitimate one should be small relative to how far the target
        # could plausibly have drifted in one frame (catches bad edge
        # matches / detections that passed the conditioning check but are
        # still wrong).
        if np.linalg.norm(delta_t_cam) > SHAFT_CORRECTION_MAX_MM:
            return

        delta_t_base = anchor[:3, :3].T @ delta_t_cam
        damped = SHAFT_TRANSLATION_DAMPING * delta_t_base

        self.pf._particles[:, :3] += damped
        self._last_good_pose_state = self._last_good_pose_state.copy()
        self._last_good_pose_state[:3] += damped

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
        """Per-frame data prep shared by update() and prepare_frame() (the
        latter used externally by CombinedArmTracker): FK refresh, label
        assignment, the obs-model's base_dets, and Canny+Hough shaft edge
        detection.

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

        fallback_keypoints = None
        if (not self._own_pf
                and os.environ.get("MTT_ONE_PF_PRIOR_SHAFT_ROI", "0") == "1"):
            # Previous-posterior projection used only to centre the shaft ROI
            # when detector keypoints are absent.  These points are not added
            # to base_dets and therefore never become PF observations.
            rvec = cv2.Rodrigues(self.cam_T_b[:3, :3])[0]
            projected = {}
            for name, point in self.name2pt.items():
                uv, _ = cv2.projectPoints(
                    np.asarray(point, dtype=np.float64)[None], rvec,
                    self.cam_T_b[:3, 3], self.K, self.dist)
                x, y = uv.reshape(2)
                if np.isfinite(x) and np.isfinite(y):
                    projected[name] = (float(x), float(y), 0.01)
            fallback_keypoints = projected or None
        elif self._own_pf and self.primary_tracker is not None:
            fallback_keypoints = self._synthetic_base_dets_from_primary()

        features = self.feature_detector.detect(
            frame_idx, left_img, right_img, raw_l, raw_r,
            name2pt=self.name2pt, P1=self.P1, P2=self.P2,
            b_T_cam_kin=self.b_T_cam_kin, K=self.K, dist=self.dist,
            robot_arm=self.robot_arm, cam_T_b_kin=self.cam_T_b_kin,
            cam_T_b=self.cam_T_b, shaft_dir_b=self._shaft_dir_b, detection_noise=getattr(self, "detection_noise", None),
            noise_peers=getattr(self, "noise_peers", {}),
            fallback_keypoints=fallback_keypoints,
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

    def prepare_frame(
        self,
        frame_idx=None,
        left_img=None,
        right_img=None,
        joint_angles=None,
        raw_l=None,
        raw_r=None,
    ):
        """Public counterpart of _prepare_frame(), for external callers that
        drive their own PF (tracking/combined_tracker.py's CombinedArmTracker)
        instead of this tracker's own_pf.

        Returns:
            corrected:   {fk_label: (x, y, likelihood)}, or None if kinematic
                         data is unavailable.
            obs_kwargs:  kwargs dict ready to call
                         self._vec_obs_func(particles, **obs_kwargs), or {}.
            n_obs_pts:   Number of 3-D/2-D correspondences fed to the PF.
            joint_angles: Raw joint angle array, or {} if unavailable.
        """
        prep = self._prepare_frame(frame_idx, left_img, right_img, joint_angles, raw_l, raw_r)
        if prep is None:
            return None, {}, 0, {}

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
        return prep["corrected"], obs_kwargs, len(prep["base_dets"]), prep["joint_angles"]

    def finalize_pose(self, pose_state_6):
        """Apply an externally-computed 6-state delta_T mean to self.cam_T_b.

        Uses _anchor_T() (this arm's own fixed hand-eye extrinsic, or the
        primary arm's live pose after set_relative_to() -- see _anchor_T())
        rather than a fixed extrinsic directly, so relative-tracking mode
        composes correctly here regardless of whether pose_state_6 came from
        this tracker's own PF or an external CombinedArmTracker's shared one.
        """
        self.pose_state = pose_state_6

        anchor = self._anchor_T()
        R_kin = anchor[:3, :3]
        t_kin = anchor[:3, 3]

        R_delta, _ = cv2.Rodrigues(self.pose_state[3:].reshape(3, 1))
        t_delta = self.pose_state[:3]
        self.cam_T_b = self.cam_T_b.copy()
        self.cam_T_b[:3, :3] = R_kin @ R_delta
        self.cam_T_b[:3, 3] = R_kin @ t_delta + t_kin

    def update(
        self,
        frame_idx=None,
        left_img=None,
        right_img=None,
        joint_angles=None,
        raw_l=None,
        raw_r=None,
    ):
        """Run one tracking step for the given frame, driving this tracker's
        own PF. Only valid when own_pf=True (see __init__) -- otherwise use
        prepare_frame()/finalize_pose() directly (see CombinedArmTracker).

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
        assert (
            self._own_pf
        ), "update() requires own_pf=True; use prepare_frame()/finalize_pose() otherwise"

        prep = self._prepare_frame(frame_idx, left_img, right_img, joint_angles, raw_l, raw_r)
        if prep is None:
            return None, 0, {}

        self.last_keypoint_detections = prep["corrected"]

        base_dets = prep["base_dets"]
        had_real_obs = bool(base_dets)
        if had_real_obs:
            self._blind_frames = 0
            # Hybrid mode: drop the coupling to primary the instant this
            # arm's own detector is working again, so real detections drive
            # a fully independent PF (own fixed anchor) exactly like
            # baseline -- not diluted by any residual pull toward primary's
            # live pose / a possibly-stale T_rel_bar.
            if self._primary_candidate is not None and self.primary_tracker is not None:
                self.set_independent()
        else:
            self._blind_frames += 1
            # Hybrid mode: couple to primary only now, at the moment
            # blindness starts -- set_relative_to() computes T_rel_bar
            # fresh from THIS arm's current (independently-tracked, still
            # accurate) absolute pose, not a value that's been sitting
            # since run start. Reset _last_good_pose_state to match the
            # zeroed pose_state set_relative_to() leaves behind, since the
            # old value was expressed relative to the previous (now
            # inactive) anchor and no longer means anything under the new one.
            if self._primary_candidate is not None and self.primary_tracker is None:
                self.set_relative_to(self._primary_candidate)
                self._last_good_pose_state = self.pose_state.copy()
            # Damped shaft-edge-derived translation correction, applied to
            # the whole particle cloud (see _shaft_translation_correction())
            # BEFORE computing the synthetic point observation below, so it
            # and next frame's ROI placement both benefit immediately.
            self._shaft_translation_correction(prep["W"], prep["H"])
            # Own detector found nothing this frame -- if this is a secondary
            # arm (coupled to a primary that DID get corrected), fall back to
            # a synthetic pseudo-observation (see
            # _synthetic_base_dets_from_primary()) instead of contributing
            # zero observation weight. Its confidence decays the longer the
            # blind stretch runs (SYNTHETIC_OBS_DECAY_TAU) -- "trust the last
            # relative offset" is only justified briefly; joint-configuration
            # -dependent encoder error isn't captured by this assumption and
            # can't be verified without vision, so don't stay confidently
            # anchored to it indefinitely.
            decayed_likelihood = SYNTHETIC_OBS_LIKELIHOOD * np.exp(
                -self._blind_frames / SYNTHETIC_OBS_DECAY_TAU
            )
            base_dets = self._synthetic_base_dets_from_primary(likelihood=decayed_likelihood)
            # Soft-coupling mode: only tried if hybrid mode's synthetic
            # source above didn't apply (primary_tracker is None here,
            # since soft coupling never switches anchor) -- projects
            # through the slowly-filtered cross-arm estimate instead.
            if not base_dets and self._soft_coupling_primary is not None:
                base_dets = self._synthetic_base_dets_from_filtered_primary(
                    likelihood=decayed_likelihood
                )

        # While running on a synthetic point observation, trust the shaft
        # edge term more: it's still a real Canny+Hough detection against
        # this frame's actual image (see _run_canny_hough()'s synthetic ROI
        # fallback), whereas the point term above is just a reprojected
        # assumption with no independent information of its own.
        edge_weight = SYNTHETIC_OBS_EDGE_WEIGHT if not had_real_obs else 1.0

        # Grow the process noise while blind so the particle cloud's spread
        # keeps pace with how far the true position could plausibly have
        # drifted -- otherwise, once a real detection resumes, resampling
        # can only pick among whatever the (too-narrow) existing cloud
        # already covers, and snapping back to full accuracy takes several
        # more frames instead of the very first re-acquired detection.
        if had_real_obs:
            std = DELTA_PROCESS_STD
        else:
            growth = min(
                1.0 + BLIND_PROCESS_STD_GROWTH_RATE * np.sqrt(self._blind_frames),
                BLIND_PROCESS_STD_MAX_MULTIPLIER,
            )
            std = DELTA_PROCESS_STD * growth

        self.pf.predictionStep_vec(std=std)
        self.pf.updateStep_vec(
            self._overlap_scaled_obs,
            base_dets=base_dets,
            name2pt=prep["name2pt"],
            robot_arm=self.robot_arm,
            K=self.K,
            dist=self.dist,
            shaft_pos_b=prep["shaft_pos_b"],
            shaft_dir_b=prep["shaft_dir_b"],
            detected_edges_hough=prep["detected_edges_hough"],
            W=prep["W"],
            H=prep["H"],
            edge_weight=edge_weight,
        )
        self.finalize_pose(self.pf.getMeanParticle())
        if had_real_obs:
            self._last_good_pose_state = self.pose_state.copy()
            # Soft-coupling mode: only trust this frame's sample for the
            # relative-offset filter if the OTHER arm also had a real
            # detection this frame (primary._blind_frames==0) -- otherwise
            # primary's own pose is itself just process-noise random walk
            # this frame, and filtering that in would corrupt the estimate
            # instead of refining it.
            if (
                self._soft_coupling_primary is not None
                and self._soft_coupling_primary._blind_frames == 0
            ):
                self._update_rel_filter()

        return prep["corrected"], len(prep["base_dets"]), prep["joint_angles"]

    def set_observation_variance_scale(self, scale):
        """Set R_t/R_0 for overlap-adaptive likelihood tempering."""
        self.observation_variance_scale = max(float(scale), 1.0)

    def _overlap_scaled_obs(self, particles, **kwargs):
        return self._vec_obs_func(particles, **kwargs) / self.observation_variance_scale

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

        if self._own_pf:
            self.pf.initializeFilter(mean_state=mean_state, std=init_std)
        self.pose_state = mean_state.copy()
        # Remembered so set_relative_to() can re-seed with a comparable
        # spread instead of falling back to the generic constructor default,
        # and so CombinedArmTracker can seed the shared PF (own_pf=False)
        # from every arm's own warmup.
        self._pending_std = init_std.copy()
        # Remembered so CombinedArmTracker's relative_to="auto" can pick
        # whichever arm's warmup was more reliable (see _warmup_confidence()
        # in combined_tracker.py) as the initial primary.
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

        if self._own_pf:
            self.pf.initializeFilter(mean_state=self.pose_state, std=init_std)
        self._pending_std = init_std
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
