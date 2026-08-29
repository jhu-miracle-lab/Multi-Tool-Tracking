"""CombinedArmTracker: single shared ParticleFilter across multiple arms.

Instead of each ArmTracker owning its own 6-state delta_T filter, this class
drives ONE ParticleFilter whose state is the concatenation of every arm's
6-state delta_T block (state = [dtx,dty,dtz,drx,dry,drz] per arm, zero =
identity = FK is perfectly accurate). All arms share the same particle set,
weights, and resampling step.

Each ArmTracker passed in must have been constructed with own_pf=False.

Optionally couples arms via relative_to (see __init__): a secondary arm's
delta_T is still an independent block of the shared 12-state vector, but its
_anchor_T() (see arm_tracker.py) composes off the primary arm's live cam_T_b
instead of its own fixed hand-eye extrinsic. Because the primary's cam_T_b is
only updated once per frame -- after the WHOLE shared particle set's single
joint weight update/resample -- a secondary arm's likelihood this frame is
evaluated against the primary's pose as of the END of the PREVIOUS frame, not
this frame's not-yet-finalized value. This is an unavoidable consequence of
folding both arms into one simultaneous update (there is no longer a
"primary updates first, then secondary" sequential order to exploit) and
matches how this same anchor was already only ever a single scalar transform
(the primary's weighted-mean pose), never a per-particle quantity, even
before this file existed.
"""

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from core.ParticleFilter import ParticleFilter

from .constants import DELTA_PROCESS_STD
from .observation import pf_init, pf_motion
from .overlap import bidirectional_keypoint_coverage, observation_variance_scale
from .relative_shaft_observation import relative_shaft_log_likelihood


def _warmup_confidence(tr):
    """Higher = more confident this arm's warmup-derived pose estimate is
    (see ArmTracker._apply_warmup_estimates(), which sets _pending_n_ok/
    _pending_n_total/_pending_iqr_raw). Primarily the PnP success ratio
    (n_ok/n_total warmup frames); ties (common -- warmup frame counts are
    small, so the ratio is coarsely quantized) are broken by whichever
    arm's raw IQR-based translation spread ended up tighter.

    Uses _pending_iqr_raw, NOT _pending_std -- the latter is floored at
    3x DELTA_PROCESS_STD (so that the PF always gets a sane minimum
    particle scatter), which means a clean warmup on BOTH arms floors
    them to the exact same value and destroys the very signal this
    tie-break needs.
    """
    if tr._pending_n_total == 0:
        return (-1.0, -np.inf)
    success_ratio = tr._pending_n_ok / tr._pending_n_total
    t_std = np.linalg.norm(tr._pending_iqr_raw[:3])
    return (success_ratio, -t_std)


def _reanchor_particle_block(particles_block, old_anchor, new_anchor):
    """Re-express an (N,6) [t,rotvec] delta_T particle block -- currently
    relative to old_anchor -- as an equivalent block relative to new_anchor,
    preserving every particle's absolute pose EXACTLY:
    old_anchor @ delta_old == new_anchor @ delta_new.

    Since old_anchor/new_anchor are both single (non-per-particle) 4x4
    transforms (see arm_tracker.py's _anchor_T()), delta_new = M @ delta_old
    for a SINGLE fixed M = inv(new_anchor) @ old_anchor -- this is an exact
    relabelling, not a resample/reseed, so it costs no filter uncertainty.
    """
    M = np.linalg.inv(new_anchor) @ old_anchor
    M_R, M_t = M[:3, :3], M[:3, 3]
    R_p = Rotation.from_rotvec(particles_block[:, 3:]).as_matrix()  # (N,3,3)
    new_R = np.einsum("ij,njk->nik", M_R, R_p)
    new_t = np.einsum("ij,nj->ni", M_R, particles_block[:, :3]) + M_t
    new_rotvec = Rotation.from_matrix(new_R).as_rotvec()
    return np.concatenate([new_t, new_rotvec], axis=1)


class CombinedArmTracker:

    def __init__(self, arm_trackers: dict, n_particles: int, relative_to: str = None):
        """
        Args:
            arm_trackers: {arm_name: ArmTracker} — each built with own_pf=False.
                          Iteration order fixes each arm's slice in the shared
                          state vector: arm i occupies state[6*i : 6*i+6].
            n_particles:  Number of particles in the single shared filter.
            relative_to:  Optional arm name (e.g. 'psm1'), or the literal
                          string "auto". When an arm name, every other arm
                          is switched (via ArmTracker.set_relative_to()) to
                          track relative to this arm's live pose instead of
                          its own independent hand-eye anchor. When "auto",
                          the primary is instead PICKED from warmup results
                          (see _warmup_confidence()) -- whichever arm's PnP
                          warmup had a higher frame success ratio (tie-
                          broken by a tighter IQR spread) becomes the
                          primary. Either way, applied once, inside
                          warmup()/warmup_live(), right after every arm's
                          own PnP warmup has completed -- same ordering
                          requirement as ArmTracker.set_relative_to() itself.
        """
        self.arm_trackers = arm_trackers
        self.arm_names = list(arm_trackers.keys())
        self._n_arms = len(self.arm_names)
        self.relative_to = relative_to
        self._relative_applied = False
        self._warmed_arms = set()
        self.method3 = os.environ.get("MTT_ONE_PF_METHOD3", "0") == "1"
        self.distance_reassignment = (
            os.environ.get("MTT_ONE_PF_DISTANCE_REASSIGN", "0") == "1")
        self.reliability_switching = (
            os.environ.get("MTT_ONE_PF_RELIABILITY_SWITCHING", "0") == "1")
        self.auto_main_switching = (
            os.environ.get("MTT_ONE_PF_AUTO_MAIN_SWITCH", "0") == "1")
        self._method3_active = False
        self._method3_main_name = None
        self._gate_enter_streak = 0
        self._gate_exit_streak = 0
        self._gate_dwell = 0
        self._gate_candidate = None
        self._method3_initialized = False
        self._method3_pnp_main = None
        self._method3_pnp_relation = None
        self._method3_previous_main_lump = None
        self._method3_pnp_poses = None
        self.last_assignment_reliability = {
            name: 0.0 for name in self.arm_names}
        self.last_relative_main = None
        if self.method3 and relative_to is not None:
            raise ValueError("one-PF Method 3 starts from independent anchors")
        if relative_to is not None and relative_to != "auto" and relative_to not in arm_trackers:
            raise ValueError(f"relative_to arm '{relative_to}' not in {self.arm_names}")

        std0 = np.concatenate([DELTA_PROCESS_STD] * self._n_arms)
        self.pf = ParticleFilter(
            num_states=6 * self._n_arms,
            initialDistributionFunc=pf_init,
            motionModelFunc=pf_motion,
            obsModelFunc=None,
            num_particles=n_particles,
            min_num_effective_particles=n_particles * 0.5,
        )
        self.pf.initializeFilter(mean_state=np.zeros(6 * self._n_arms), std=std0)

    def _distance_reassign(self, obs_kwargs, sigma_px=20.0, gate_px=60.0):
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
            kwargs = obs_kwargs.get(i, {})
            predictions[name] = {}
            for label in labels:
                if label in LR_BASES:
                    point_name = f"{label}:{tr.feature_detector.curr_lr}"
                elif label in FB_BASES:
                    point_name = f"{label}:{tr.feature_detector.curr_fb}"
                else:
                    point_name = label
                point = kwargs.get("name2pt", {}).get(point_name)
                if point is None:
                    continue
                rvec = cv2.Rodrigues(tr.cam_T_b[:3, :3])[0]
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
                det = obs_kwargs.get(i, {}).get("base_dets", {}).get(label)
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
            if i in obs_kwargs:
                obs_kwargs[i]["base_dets"] = assigned[name]
            coverage = len(assigned[name]) / max(expected[name], 1)
            geometry = (float(np.exp(-0.5 * (
                np.median(accepted_distances[name]) / sigma_px) ** 2))
                        if accepted_distances[name] else 0.0)
            reliability[name] = float(coverage * geometry)
        return reliability

    def _reliability_gate(self, reliability):
        """Hysteretic Method-4 gate with PSM1 fixed as the leading arm."""
        main, secondary = self.arm_names[0], self.arm_names[1]
        enter = reliability[main] >= 0.55 and reliability[secondary] <= 0.35
        exit_condition = reliability[main] < 0.40 or reliability[secondary] >= 0.55
        if not self._method3_active:
            self._gate_enter_streak = self._gate_enter_streak + 1 if enter else 0
            if self._gate_enter_streak >= 2:
                self._method3_active = True
                self._gate_enter_streak = 0
                self._gate_exit_streak = 0
                self._gate_dwell = 0
        else:
            self._gate_dwell += 1
            if self._gate_dwell >= 10:
                self._gate_exit_streak = (
                    self._gate_exit_streak + 1 if exit_condition else 0)
                if self._gate_exit_streak >= 3:
                    self._method3_active = False
                    self._gate_exit_streak = 0
        return self._method3_active

    def _automatic_main_gate(self, reliability):
        """Select either arm as Method-3 leader with hysteresis.

        A direction is requested only when one arm is visually reliable and
        the other is unreliable.  Two-frame confirmation, a ten-frame dwell,
        and three-frame exit confirmation suppress threshold chatter.
        """
        first, second = self.arm_names
        requested = None
        if reliability[first] >= 0.55 and reliability[second] <= 0.35:
            requested = first
        elif reliability[second] >= 0.55 and reliability[first] <= 0.35:
            requested = second

        current = self._method3_main_name
        if current is None:
            if requested is None:
                self._gate_candidate = None
                self._gate_enter_streak = 0
            elif self._gate_candidate == requested:
                self._gate_enter_streak += 1
            else:
                self._gate_candidate = requested
                self._gate_enter_streak = 1
            if self._gate_enter_streak >= 2:
                current = requested
                self._gate_enter_streak = 0
                self._gate_exit_streak = 0
                self._gate_dwell = 0
        else:
            self._gate_dwell += 1
            other = second if current == first else first
            if requested is not None and requested != current:
                if self._gate_candidate != requested:
                    self._gate_candidate = requested
                    self._gate_enter_streak = 1
                else:
                    self._gate_enter_streak += 1
                if self._gate_dwell >= 10 and self._gate_enter_streak >= 2:
                    current = requested
                    self._gate_enter_streak = 0
                    self._gate_exit_streak = 0
                    self._gate_dwell = 0
            else:
                self._gate_candidate = None
                self._gate_enter_streak = 0
                exit_condition = (
                    reliability[current] < 0.40
                    or reliability[other] >= 0.55)
                if self._gate_dwell >= 10:
                    self._gate_exit_streak = (
                        self._gate_exit_streak + 1 if exit_condition else 0)
                    if self._gate_exit_streak >= 3:
                        current = None
                        self._gate_exit_streak = 0
        self._method3_active = current is not None
        return current

    def _transition_method3(self, main_name):
        """Change Method-3 direction without changing any particle pose."""
        if main_name == self._method3_main_name:
            return
        # Return every currently relative block to its independent anchor.
        for index, name in enumerate(self.arm_names):
            tracker = self.arm_trackers[name]
            if tracker._method3_anchor is None:
                continue
            old_anchor = tracker._anchor_T().copy()
            tracker.primary_tracker = None
            tracker.T_rel_bar = np.eye(4)
            tracker._method3_anchor = None
            new_anchor = tracker._anchor_T().copy()
            block = self._slice(index)
            self.pf._particles[:, block] = _reanchor_particle_block(
                self.pf._particles[:, block], old_anchor, new_anchor)
        self._method3_main_name = main_name
        self._method3_initialized = False
        if main_name is not None:
            main = self.arm_trackers[main_name]
            self._method3_previous_main_lump = (
                np.linalg.inv(main.cam_T_b_kin) @ main.cam_T_b)

    def _slice(self, i):
        return slice(i * 6, i * 6 + 6)

    def _apply_relative_to(self):
        """Switch every non-primary arm to relative tracking, once every arm
        involved has completed at least one warmup pass.

        Safe to call from both warmup() and warmup_live() even though ROS2
        live mode's staged warmup (see ros2_node/tracking_node.py's
        _run_warmup()) may call warmup_live() several times as different
        arms become ready independently: set_relative_to() is only valid to
        call once per secondary arm (a second call would re-anchor off an
        already-relative pose_state, double-composing T_rel_bar), and it
        also requires the primary's own warmup to have already produced a
        refined absolute pose estimate -- so this deliberately waits for
        self._warmed_arms to cover every arm before firing, rather than
        applying it against whichever arms merely happen to be warmed on
        the first call. This same gate also postpones resolving
        relative_to="auto" until every arm's warmup confidence is available.
        """
        if self.relative_to is None or self._relative_applied:
            return
        if not set(self.arm_names).issubset(self._warmed_arms):
            return
        if self.relative_to == "auto":
            best = max(self.arm_names, key=lambda n: _warmup_confidence(self.arm_trackers[n]))
            print(f"[CombinedArmTracker] auto-selected primary from warmup confidence: {best}")
            self.relative_to = best
        primary = self.arm_trackers[self.relative_to]
        for name, tr in self.arm_trackers.items():
            if name != self.relative_to:
                tr.set_relative_to(primary)
        self._relative_applied = True

    def _pool_from_arms(self):
        mean_state = np.concatenate([self.arm_trackers[name].pose_state for name in self.arm_names])
        std = np.concatenate([self.arm_trackers[name]._pending_std for name in self.arm_names])
        self.pf.initializeFilter(mean_state=mean_state, std=std)

        if self.method3 and len(self.arm_names) == 2:
            self._method3_pnp_poses = {
                name: self.arm_trackers[name].cam_T_b.copy()
                for name in self.arm_names}
            main = self.arm_trackers[self.arm_names[0]]
            secondary = self.arm_trackers[self.arm_names[1]]
            self._method3_pnp_main = self._method3_pnp_poses[self.arm_names[0]]
            self._method3_pnp_relation = (
                np.linalg.inv(self._method3_pnp_main)
                @ self._method3_pnp_poses[self.arm_names[1]])
            self._method3_previous_main_lump = (
                np.linalg.inv(main.cam_T_b_kin) @ main.cam_T_b)

    def warmup(self, frame_indices, lkl_thresh: float = 0.4, left_img_dir: str = None):
        """Run each arm's independent PnP-based warmup, optionally apply
        relative_to coupling, then initialise the shared filter once from
        the concatenated per-arm means/stds.
        """
        for name in self.arm_names:
            self.arm_trackers[name].warmup(
                frame_indices, lkl_thresh=lkl_thresh, left_img_dir=left_img_dir
            )
        self._warmed_arms.update(self.arm_names)
        self._apply_relative_to()
        self._pool_from_arms()

    def warmup_live(self, frames_by_arm: dict):
        """Live-mode counterpart of warmup(): frames_by_arm = {arm_name:
        [(left_img, joint_angles, raw_l), ...]} for whichever arms have
        enough buffered frames this call (see ros2_node/tracking_node.py's
        _run_warmup()). Each named arm runs its own warmup_live(); the
        shared filter is then re-seeded from ALL arms' current
        pose_state/_pending_std, so calling this again later (once more
        arms become ready) safely re-pools without disturbing arms already
        warmed up. relative_to (if set) is only actually applied once every
        involved arm has been warmed up at least once -- see
        _apply_relative_to().
        """
        for name, frames in frames_by_arm.items():
            self.arm_trackers[name].warmup_live(frames)
        self._warmed_arms.update(frames_by_arm.keys())
        self._apply_relative_to()
        self._pool_from_arms()

    def update(self, frame_idx=None, left_img=None, right_img=None, per_arm=None):
        """Run one combined tracking step for every arm.

        Args:
            per_arm: optional {arm_name: {"joint_angles":..., "raw_l":...,
                     "raw_r":...}} overrides for callers that supply
                     per-arm data directly (e.g. ROS2 realtime mode with
                     one shared DEKR model per frame) instead of relying on
                     each ArmTracker's own kin_dir/dlc_left/dlc_right.

        Returns:
            {arm_name: (corrected, n_obs_pts, joint_angles)} — same per-arm
            values ArmTracker.update() would have returned. corrected is None
            for arms whose kinematic data was unavailable this frame; those
            arms are excluded from the observation update (contribute zero
            log-likelihood) and their cam_T_b/pose_state are left unchanged,
            but they still receive this frame's prediction-step process noise
            since the whole particle set is predicted together.
        """
        per_arm = per_arm or {}
        results = {}
        obs_kwargs = {}
        for i, name in enumerate(self.arm_names):
            tr = self.arm_trackers[name]
            corrected, kwargs, n_obs, joint_angles = tr.prepare_frame(
                frame_idx, left_img=left_img, right_img=right_img, **per_arm.get(name, {})
            )
            results[name] = (corrected, n_obs, joint_angles)
            if corrected is not None:
                obs_kwargs[i] = kwargs

        reliability = ({name: 0.0 for name in self.arm_names})
        if self.distance_reassignment:
            reliability = self._distance_reassign(obs_kwargs)
        self.last_assignment_reliability = dict(reliability)

        # Both trackers now contain the current frame's kinematic keypoints.
        # Convert their symmetric projected coverage into R_t/R_0 and use the
        # same image-quality factor for both blocks of the shared posterior.
        if len(self.arm_names) == 2:
            tr_a = self.arm_trackers[self.arm_names[0]]
            tr_b = self.arm_trackers[self.arm_names[1]]
            coverage = bidirectional_keypoint_coverage(
                obs_kwargs.get(0, {}).get("base_dets", {}),
                obs_kwargs.get(1, {}).get("base_dets", {}),
                30.0,
            )
            beta = float(os.environ.get("MTT_OVERLAP_BETA", "3.0"))
            scale = observation_variance_scale(coverage, beta)
            tr_a.set_observation_variance_scale(scale)
            tr_b.set_observation_variance_scale(scale)

        std = np.concatenate([DELTA_PROCESS_STD] * self._n_arms)
        self.pf.predictionStep_vec(std=std)

        if self.method3 and len(self.arm_names) == 2:
            if self.reliability_switching:
                if self.auto_main_switching:
                    main_name = self._automatic_main_gate(reliability)
                else:
                    active = self._reliability_gate(reliability)
                    main_name = self.arm_names[0] if active else None
            else:
                main_name = self.arm_names[0]
            self._transition_method3(main_name)
            self.last_relative_main = main_name
            if main_name is not None:
                return self._update_method3(results, obs_kwargs, main_name)
        else:
            self.last_relative_main = None

        def _combined_obs(particles):
            total = np.zeros(len(particles))
            for i, kwargs in obs_kwargs.items():
                name = self.arm_names[i]
                total += self.arm_trackers[name]._overlap_scaled_obs(
                    particles[:, self._slice(i)], **kwargs
                )
            shaft_model = os.environ.get("MTT_RELATIVE_SHAFT_MODEL", "2d").lower()
            if (os.environ.get("MTT_ENABLE_RELATIVE_SHAFT_OBS", "0") == "1"
                    and (not self.reliability_switching or self._method3_active)
                    and shaft_model in {"2d", "3d"}):
                total += relative_shaft_log_likelihood(
                    particles, self.arm_trackers, obs_kwargs, shaft_model)
            return total

        if obs_kwargs:
            # Only resample the column-blocks of arms that actually have an
            # observation this frame -- an arm with none would otherwise be
            # resampled purely off the other arm(s)' likelihood, a criterion
            # irrelevant to its own state (see ParticleFilter.updateStep_vec's
            # resample_col_blocks docstring). Degenerates to a full-row
            # resample (previous behavior) whenever every arm has data.
            resample_col_blocks = [(i * 6, i * 6 + 6) for i in obs_kwargs]
            self.pf.updateStep_vec(_combined_obs, resample_col_blocks=resample_col_blocks)

        full_state = self.pf.getMeanParticle()
        for i, name in enumerate(self.arm_names):
            if results[name][0] is not None:
                self.arm_trackers[name].finalize_pose(full_state[self._slice(i)])

        return results

    def _update_weights_without_resampling(self, log_likelihood):
        """Bayesian weight update used for Method 3's leading-arm stage."""
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

    def _update_method3(self, results, obs_kwargs, main_name):
        """Advisor Method 3 for one shared 12-state particle filter.

        PSM1 is updated first without an intermediate resample.  Its current
        lump increment defines PSM2's frame-local anchor.  PSM2 and the
        optional relative-shaft observation then complete the same joint
        Bayesian update before a single ESS/resampling decision.
        """
        main_index = self.arm_names.index(main_name)
        secondary_index = 1 - main_index
        secondary_name = self.arm_names[secondary_index]
        main = self.arm_trackers[main_name]
        secondary = self.arm_trackers[secondary_name]

        if main_index in obs_kwargs:
            log_main = main._overlap_scaled_obs(
                self.pf._particles[:, self._slice(main_index)],
                **obs_kwargs[main_index])
            self._update_weights_without_resampling(log_main)

        intermediate = self.pf.getMeanParticle()
        main.finalize_pose(intermediate[self._slice(main_index)])
        current_main_lump = np.linalg.inv(main.cam_T_b_kin) @ main.cam_T_b
        main_increment = (
            current_main_lump @ np.linalg.inv(self._method3_previous_main_lump))
        pnp_main = self._method3_pnp_poses[main_name]
        pnp_relation = (
            np.linalg.inv(pnp_main) @ self._method3_pnp_poses[secondary_name])
        predicted_secondary_anchor = (
            pnp_main @ main_increment @ pnp_relation)

        secondary_slice = self._slice(secondary_index)
        if not self._method3_initialized:
            old_anchor = secondary._anchor_T().copy()
            self.pf._particles[:, secondary_slice] = _reanchor_particle_block(
                self.pf._particles[:, secondary_slice], old_anchor,
                predicted_secondary_anchor)
            self._method3_initialized = True
        secondary.primary_tracker = None
        secondary.T_rel_bar = np.eye(4)
        secondary._method3_anchor = predicted_secondary_anchor.copy()

        def remaining_observation(particles):
            total = np.zeros(len(particles))
            if secondary_index in obs_kwargs:
                total += secondary._overlap_scaled_obs(
                    particles[:, secondary_slice], **obs_kwargs[secondary_index])
            shaft_model = os.environ.get("MTT_RELATIVE_SHAFT_MODEL", "2d").lower()
            if (os.environ.get("MTT_ENABLE_RELATIVE_SHAFT_OBS", "0") == "1"
                    and shaft_model in {"2d", "3d"}):
                total += relative_shaft_log_likelihood(
                    particles, self.arm_trackers, obs_kwargs, shaft_model)
            return total

        if (secondary_index in obs_kwargs
                or os.environ.get("MTT_ENABLE_RELATIVE_SHAFT_OBS", "0") == "1"):
            self.pf.updateStep_vec(remaining_observation)

        full_state = self.pf.getMeanParticle()
        for i, name in enumerate(self.arm_names):
            if results[name][0] is not None:
                self.arm_trackers[name].finalize_pose(full_state[self._slice(i)])
        self._method3_previous_main_lump = current_main_lump.copy()
        return results

    def set_primary(self, name):
        """Dynamically switch which arm every other arm is anchored to,
        mid-tracking (not just once at warmup) -- e.g. driven by
        tracking/main_tool.py's MainToolSelector as observation quality
        shifts between arms.

        Unlike warmup()/warmup_live()'s _apply_relative_to() (a one-time,
        full initializeFilter() reseed), this re-anchors each affected
        arm's particle BLOCK in place via _reanchor_particle_block() --
        an exact SE(3) relabelling, not a resample -- so the filter's
        current uncertainty/shape is preserved across the switch instead
        of being discarded back to a generic prior.

        Args:
            name: arm name to make the new coupling anchor, or None to
                  make every arm independent again (own hand-eye anchor).

        Returns:
            True if a switch was actually made, False if `name` already
            was the current primary (no-op).
        """
        if name == self.relative_to:
            return False
        if name is not None and name not in self.arm_trackers:
            raise ValueError(f"'{name}' not in {self.arm_names}")

        old_anchors = {n: tr._anchor_T() for n, tr in self.arm_trackers.items()}

        self.relative_to = name
        self._relative_applied = True  # this call supersedes the warmup-time auto-apply
        if name is None:
            for tr in self.arm_trackers.values():
                tr.set_independent()
        else:
            primary = self.arm_trackers[name]
            primary.set_independent()
            for n, tr in self.arm_trackers.items():
                if n != name:
                    tr.set_relative_to(primary)

        for i, n in enumerate(self.arm_names):
            new_anchor = self.arm_trackers[n]._anchor_T()
            sl = self._slice(i)
            self.pf._particles[:, sl] = _reanchor_particle_block(
                self.pf._particles[:, sl], old_anchors[n], new_anchor
            )
        return True
