"""Joint particle filtering for multiple ArmTrackers.

Each arm owns a six-dimensional error block; particles share weights.
The ordinary path updates arm likelihoods together. FPRT/APRT instead
update the primary first, transfer its cumulative lump-error change, then apply
secondary and relative-shaft evidence before the resampling decision.
Primary selection is delegated to PrimarySelector."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from core.ParticleFilter import ParticleFilter

from .constants import DELTA_PROCESS_STD
from .tracker_config import (
    SWITCH_RESET_COVARIANCE_SCALE, OVERLAP_BETA,
)
from .primary_selector import PrimarySelector
from .observation import pf_init, pf_motion
from .overlap import bidirectional_keypoint_coverage, observation_variance_scale
from .relative_shaft_observation import relative_shaft_log_likelihood


def _reanchor_particle_block(particles_block, old_anchor, new_anchor):
    """Change an (N, 6) error block reference while preserving absolute poses.
    
    old_anchor @ delta_old == new_anchor @ delta_new; no resampling occurs."""
    M = np.linalg.inv(new_anchor) @ old_anchor
    M_R, M_t = M[:3, :3], M[:3, 3]
    R_p = Rotation.from_rotvec(particles_block[:, 3:]).as_matrix()  # (N,3,3)
    new_R = np.einsum("ij,njk->nik", M_R, R_p)
    new_t = np.einsum("ij,nj->ni", M_R, particles_block[:, :3]) + M_t
    new_rotvec = Rotation.from_matrix(new_R).as_rotvec()
    return np.concatenate([new_t, new_rotvec], axis=1)


class CombinedArmTracker:
    switch_reset_covariance_scale = SWITCH_RESET_COVARIANCE_SCALE

    def __init__(self, arm_trackers: dict, n_particles: int, *, allow_primary_switch=False):
        """Bind the arms, initialize runtime state, and create their shared PF.

        Dictionary order determines each arm's six-state particle block.
        allow_primary_switch explicitly selects FPRT (False) or APRT (True).
        """
        self.arm_trackers = arm_trackers
        self.arm_names = list(arm_trackers)
        self._n_arms = len(self.arm_names)
        self._particle_slices = {
            name: slice(i * 6, (i + 1) * 6) for i, name in enumerate(self.arm_names)}
        self.primary_selector = PrimarySelector(self.arm_names, allow_primary_switch)

        self._initialize_tracking_state()
        self.pf = self._create_particle_filter(n_particles)

    def _initialize_tracking_state(self):
        """Initialize per-instance history; these fields change during tracking."""
        # Warmup poses define the initial relation between the arms.
        self._warmup_poses = None

        # Current relative mode and the fixed reference for cumulative error transfer.
        self._primary_arm_name = None
        self._secondary_anchor_initialized = False
        self._reference_primary_lump = None

        # Diagnostics exposed to callers.
        self.mode_switch_reset_count = 0
        self.last_assignment_reliability = {name: 0.0 for name in self.arm_names}
        self.last_relative_main = None

    def _create_particle_filter(self, n_particles):
        """Create zero-centered six-state error blocks with shared particle weights."""
        state_size = 6 * self._n_arms
        process_std = np.concatenate([DELTA_PROCESS_STD] * self._n_arms)
        particle_filter = ParticleFilter(
            num_states=state_size,
            initialDistributionFunc=pf_init,
            motionModelFunc=pf_motion,
            obsModelFunc=None,
            num_particles=n_particles,
            min_num_effective_particles=n_particles * 0.5,
        )
        particle_filter.initializeFilter(
            mean_state=np.zeros(state_size), std=process_std)
        return particle_filter

    def _distance_reassign(self, observations, sigma_px=20.0, gate_px=60.0):
        """Reassign same-label detections to the closest predicted arm.

        Returns a separate keypoint reliability for each arm.  Reliability
        combines admitted-keypoint coverage and prediction distance, so a
        swapped or missing PSM2 observation does not reduce PSM1's score.
        """
        if len(self.arm_names) != 2:
            return {name: 0.0 for name in self.arm_names}
        from .label_assignment import LR_BASES, FB_BASES

        labels = sorted(LR_BASES | FB_BASES) + ["yaw_1", "yaw_2"]
        predictions = {}
        for i, name in enumerate(self.arm_names):
            tr = self.arm_trackers[name]
            observation = observations.get(name)
            predictions[name] = {}
            rvec = cv2.Rodrigues(tr.cam_T_b[:3, :3])[0]
            for label in labels:
                if label in LR_BASES:
                    point_name = f"{label}:{tr.feature_detector.curr_lr}"
                elif label in FB_BASES:
                    point_name = f"{label}:{tr.feature_detector.curr_fb}"
                else:
                    point_name = label
                point = observation.points_3d.get(point_name) if observation is not None else None
                if point is None:
                    continue
                uv, _ = cv2.projectPoints(
                    np.asarray(point, dtype=np.float64)[None], rvec,
                    tr.cam_T_b[:3, 3], tr.K, tr.dist)
                predictions[name][label] = uv.reshape(2)
        assigned = {name: {} for name in self.arm_names}
        accepted_distances = {name: [] for name in self.arm_names}
        expected = {name: len(predictions[name]) for name in self.arm_names}
        for label in labels:
            candidates = []
            for i, name in enumerate(self.arm_names):
                det = observations[name].keypoints.get(label) if name in observations else None
                if det is not None:
                    candidates.append((i, det))
            if not candidates:
                continue
            costs = np.full((2, len(candidates)), np.inf)
            for ai, name in enumerate(self.arm_names):
                pred = predictions[name].get(label)
                if pred is not None:
                    for di, (_, det) in enumerate(candidates):
                        costs[ai, di] = np.linalg.norm(
                            np.asarray(det[:2], dtype=np.float64) - pred)
            if len(candidates) >= 2:
                pairs = (((0, 0), (1, 1))
                         if costs[0, 0] + costs[1, 1]
                         <= costs[0, 1] + costs[1, 0]
                         else ((0, 1), (1, 0)))
            else:
                # A single-arm candidate is not evidence of a swap. Keep it
                # with its detector identity so complete PSM2 dropout cannot
                # steal PSM1's only observation. Cross-arm reassignment is
                # reserved for the paired-candidate case above.
                source_arm = candidates[0][0]
                pairs = ((source_arm, 0),)
            for ai, di in pairs:
                distance = costs[ai, di]
                if np.isfinite(distance) and distance <= gate_px:
                    name = self.arm_names[ai]
                    assigned[name][label] = candidates[di][1]
                    accepted_distances[name].append(float(distance))

        reliability = {}
        for i, name in enumerate(self.arm_names):
            if name in observations:
                observations[name].keypoints = assigned[name]
            coverage = len(assigned[name]) / max(expected[name], 1)
            geometry = (float(np.exp(-0.5 * (
                np.median(accepted_distances[name]) / sigma_px) ** 2))
                        if accepted_distances[name] else 0.0)
            reliability[name] = float(coverage * geometry)
        return reliability

    def _transition_relative_mode(self, main_name):
        """Re-anchor particles for a mode change, then optionally rejuvenate them."""
        if main_name == self._primary_arm_name:
            return False
        previous_main = self._primary_arm_name
        # Return every currently relative block to its independent anchor.
        for name in self.arm_names:
            tracker = self.arm_trackers[name]
            if not tracker.has_relative_anchor:
                continue
            old_anchor = tracker.anchor.copy()
            tracker.set_anchor(None)
            new_anchor = tracker.anchor.copy()
            block = self._slice(name)
            self.pf._particles[:, block] = _reanchor_particle_block(
                self.pf._particles[:, block], old_anchor, new_anchor)
        self._primary_arm_name = main_name
        self._secondary_anchor_initialized = False
        # A new relative episode gets its own fixed error reference. Never
        # carry a reference from the old primary into the new direction.
        self._reference_primary_lump = None
        if main_name is not None:
            main = self.arm_trackers[main_name]
            self._reference_primary_lump = (
                np.linalg.inv(main.cam_T_b_kin) @ main.cam_T_b)
        # A baseline/relative transition changes the distribution that is
        # useful to the observation model.  Preserve the current weighted
        # lump-error estimate, but restore the initial search covariance.
        # Re-anchoring above is deliberately done first, so the reset mean
        # is expressed in the parameterisation used after the switch.
        affected = set()
        if previous_main is not None:
            affected.update(n for n in self.arm_names if n != previous_main)
        if main_name is not None:
            affected.update(n for n in self.arm_names if n != main_name)
        for name in affected:
            block = self._slice(name)
            mean = self.pf.getMeanParticle()[block]
            std = np.asarray(
                self.arm_trackers[name].warmup_std, dtype=np.float64)
            std = std * np.sqrt(self.switch_reset_covariance_scale)
            self.pf._particles[:, block] = np.random.normal(
                loc=mean, scale=std,
                size=(self.pf._particles.shape[0], 6))
        if affected:
            self.pf._weights.fill(1.0 / len(self.pf._weights))
            self.mode_switch_reset_count += 1
        return True

    def _slice(self, name):
        return self._particle_slices[name]

    def _pool_from_arms(self):
        mean_state = np.concatenate([self.arm_trackers[name].pose_state for name in self.arm_names])
        std = np.concatenate([self.arm_trackers[name].warmup_std for name in self.arm_names])
        self.pf.initializeFilter(mean_state=mean_state, std=std)

        if len(self.arm_names) == 2:
            self._warmup_poses = {
                name: self.arm_trackers[name].cam_T_b.copy()
                for name in self.arm_names}

    def warmup(self, frame_indices, lkl_thresh: float = 0.4, left_img_dir: str = None):
        """Warm up each arm, then pool its mean and spread into the shared PF."""
        for name in self.arm_names:
            self.arm_trackers[name].warmup(
                frame_indices, lkl_thresh=lkl_thresh, left_img_dir=left_img_dir
            )
        self._pool_from_arms()

    def warmup_live(self, frames_by_arm: dict):
        """Warm up the supplied arms, then re-pool all current means and spreads."""
        for name, frames in frames_by_arm.items():
            self.arm_trackers[name].warmup_live(frames)
        self._pool_from_arms()

    def update(self, frame_idx=None, left_img=None, right_img=None, per_arm=None):
        """Prepare observations, select the mode, then update the shared PF.

        per_arm optionally supplies joint_angles/raw_l/raw_r per arm.
        Returns {arm_name: ArmFrame}. Arms with
        unavailable frame data keep their pose, but still receive process noise.
        """
        results = self._prepare_frames(frame_idx, left_img, right_img, per_arm or {})
        observations = {name: frame.observation for name, frame in results.items()
                        if frame.available}
        reliability = self._distance_reassign(observations)
        self.last_assignment_reliability = dict(reliability)

        self._update_overlap_scale(observations)
        self._select_relative_mode(reliability)
        # Re-anchor/rejuvenate before adding this frame's process noise.
        std = np.concatenate([DELTA_PROCESS_STD] * self._n_arms)
        self.pf.predictionStep_vec(std=std)

        if self._primary_arm_name is not None:
            return self._update_relative_tracking(
                results, observations, self._primary_arm_name)
        return self._update_joint(results, observations)

    def _prepare_frames(self, frame_idx, left_img, right_img, per_arm):
        """Keep each arm's named frame data intact throughout the update."""
        return {name: tracker.prepare_frame(
                    frame_idx, left_img=left_img, right_img=right_img,
                    **per_arm.get(name, {}))
                for name, tracker in self.arm_trackers.items()}

    def _update_overlap_scale(self, observations):
        """Apply the same overlap-based observation variance scale to both arms."""
        if self._n_arms != 2:
            return
        first, second = self.arm_names
        coverage = bidirectional_keypoint_coverage(
            observations[first].keypoints if first in observations else {},
            observations[second].keypoints if second in observations else {}, 30.0)
        scale = observation_variance_scale(coverage, OVERLAP_BETA)
        for tracker in self.arm_trackers.values():
            tracker.set_observation_variance_scale(scale)

    def _select_relative_mode(self, reliability):
        """Choose the reliability-gated primary for FPRT or APRT."""
        if self._n_arms != 2:
            self.last_relative_main = None
            return
        main = self.primary_selector.update(reliability)
        self._transition_relative_mode(main)
        self.last_relative_main = main

    def _observation_log_likelihood(self, particles, observations, arm_names,
                                    include_relative_shaft=True):
        """Sum selected arms' log-likelihoods and optional cross-tool evidence.

        Keep all observations available to the relative-shaft model, even
        when only the secondary arm's ordinary likelihood remains to apply.
        """
        total = np.zeros(len(particles))
        for name in arm_names:
            if name in observations:
                tracker = self.arm_trackers[name]
                total += tracker.log_likelihood(
                    particles[:, self._slice(name)], observations[name])
        if include_relative_shaft:
            total += relative_shaft_log_likelihood(
                particles, self.arm_trackers, observations, self._particle_slices)
        return total

    def _update_joint(self, results, observations):
        """Update all available arm blocks together, then publish their poses."""
        if observations:
            def observation_model(particles):
                return self._observation_log_likelihood(
                    particles, observations, observations,
                    include_relative_shaft=False)

            # Preserve the existing block policy: membership comes from
            # prepare_frame success, not from the number of detected points.
            blocks = [(self._slice(name).start, self._slice(name).stop) for name in observations]
            self.pf.updateStep_vec(observation_model, resample_col_blocks=blocks)
        return self._finalize_poses(results)

    def _finalize_poses(self, results):
        """Distribute the joint posterior mean to arms with available frame data."""
        full_state = self.pf.getMeanParticle()
        for name in self.arm_names:
            if results[name].available:
                self.arm_trackers[name].finalize_pose(full_state[self._slice(name)])
        return results

    def _update_weights_without_resampling(self, log_likelihood):
        """Bayesian weight update used for the primary-arm stage."""
        values = np.asarray(log_likelihood, dtype=np.float64)
        finite = np.isfinite(values)
        if not np.any(finite):
            return
        maximum = np.max(values[finite])
        factors = np.zeros_like(values)
        factors[finite] = np.exp(values[finite] - maximum)
        self.pf._weights *= factors
        total = float(np.sum(self.pf._weights))
        if total <= 0.0 or not np.isfinite(total):
            self.pf._weights.fill(1.0 / len(self.pf._weights))
        else:
            self.pf._weights /= total

    def _update_relative_tracking(self, results, observations, main_name):
        """Update the primary, transfer its cumulative error change, then the secondary.
        
        FPRT fixes the primary; APRT may select either arm. There is no resampling
        between primary and secondary likelihoods."""
        main_index = self.arm_names.index(main_name)
        secondary_index = 1 - main_index
        secondary_name = self.arm_names[secondary_index]
        main = self.arm_trackers[main_name]
        secondary = self.arm_trackers[secondary_name]

        if main_name in observations:
            log_main = main.log_likelihood(
                self.pf._particles[:, self._slice(main_name)], observations[main_name])
            self._update_weights_without_resampling(log_main)

        intermediate = self.pf.getMeanParticle()
        main.finalize_pose(intermediate[self._slice(main_name)])
        current_main_lump = np.linalg.inv(main.cam_T_b_kin) @ main.cam_T_b
        # G_total = L_current @ inverse(L_reference). The reference stays
        # fixed until leaving relative mode or selecting a different primary.
        # This equals G_t @ ... @ G_start without repeated matrix products:
        # a primary error that changes once and holds keeps its correction.
        cumulative_main_change = (
            current_main_lump @ np.linalg.inv(self._reference_primary_lump))
        pnp_main = self._warmup_poses[main_name]
        pnp_relation = (
            np.linalg.inv(pnp_main) @ self._warmup_poses[secondary_name])
        predicted_secondary_anchor = (
            pnp_main @ cumulative_main_change @ pnp_relation)

        secondary_slice = self._slice(secondary_name)
        if not self._secondary_anchor_initialized:
            old_anchor = secondary.anchor.copy()
            self.pf._particles[:, secondary_slice] = _reanchor_particle_block(
                self.pf._particles[:, secondary_slice], old_anchor,
                predicted_secondary_anchor)
            self._secondary_anchor_initialized = True
        secondary.set_anchor(predicted_secondary_anchor)

        def remaining_observation(particles):
            return self._observation_log_likelihood(
                particles, observations, (secondary_name,))

        self.pf.updateStep_vec(remaining_observation)

        self._finalize_poses(results)
        return results
