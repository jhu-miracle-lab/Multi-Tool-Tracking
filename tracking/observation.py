"""Particle filter observation and motion models."""


import numpy as np
from scipy.spatial.transform import Rotation


from .constants import (
    LUMPED_GAMMA_M,
    LUMPED_GAMMA_RHO,
    LUMPED_GAMMA_PHI,
    LUMPED_CMAX_PT,
    LUMPED_CMAX_EDGE,
    LUMPED_N_POINTS,
)
from .label_assignment import LR_BASES, FB_BASES

# ---------------------------------------------------------------------------
# Motion model
# ---------------------------------------------------------------------------


def pf_init(mean_state: np.ndarray, std: np.ndarray):
    from core.probability_functions import sampleNormalDistribution

    sample, prob = sampleNormalDistribution(std)
    return mean_state + sample, prob


def pf_motion(state: np.ndarray, std: np.ndarray):
    from core.probability_functions import additiveGaussianNoise

    return additiveGaussianNoise(state, std)


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


def _project_batch(pts3d_b, R_eff, t_eff, K, dist):
    """Project a single base-frame 3-D point through N particle poses.

    Args:
        pts3d_b: (3,) base-frame point.
        R_eff:   (N,3,3) per-particle effective rotation.
        t_eff:   (N,3)   per-particle effective translation.
        K, dist: camera intrinsics / distortion.

    Returns:
        (N,2) pixel coordinates per particle.
    """
    pc = np.einsum("nij,j->ni", R_eff, pts3d_b) + t_eff  # (N,3)
    z = pc[:, 2]
    xn = pc[:, 0] / z
    yn = pc[:, 1] / z
    d = np.asarray(dist, dtype=np.float64).flatten()
    k1 = d[0] if len(d) > 0 else 0.0
    k2 = d[1] if len(d) > 1 else 0.0
    p1 = d[2] if len(d) > 2 else 0.0
    p2 = d[3] if len(d) > 3 else 0.0
    k3 = d[4] if len(d) > 4 else 0.0
    r2 = xn**2 + yn**2
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn**2)
    yd = yn * radial + p1 * (r2 + 2.0 * yn**2) + 2.0 * p2 * xn * yn
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=1)  # (N,2)


def create_delta_obs_model_vec(cam_T_b_kin_getter):
    """Vectorised point observation model for delta_T state

    State = [dtx, dty, dtz, drx, dry, drz]; zero = identity = FK is perfect.
    Effective pose: cam_T_b = cam_T_b_kin @ delta_T

    Args:
        cam_T_b_kin_getter: callable () -> (4,4) fixed FK extrinsic.

    Returns:
        Callable (particles, base_dets, name2pt, robot_arm, K, dist) -> (N,) log-weights.
        base_dets: {base_name: (x, y, likelihood)} raw per-joint detections,
                   already likelihood-thresholded by the caller, keyed by
                   the generic joint name (e.g. 'roll_2', not 'roll_2:left').
    """

    def _vec_obs(particles, base_dets, name2pt, robot_arm, K, dist):
        if not base_dets:
            return np.zeros(len(particles))

        cam_T_b_kin = cam_T_b_kin_getter()
        R_kin = cam_T_b_kin[:3, :3]
        t_kin = cam_T_b_kin[:3, 3]

        R_delta = Rotation.from_rotvec(particles[:, 3:]).as_matrix()  # (N,3,3)
        R_eff = np.einsum("ij,njk->nik", R_kin, R_delta)  # (N,3,3)
        t_eff = np.einsum("ij,nj->ni", R_kin, particles[:, :3]) + t_kin  # (N,3)

        N = len(particles)
        sum_prob = np.zeros(N)
        n_matched = 0

        def add_term(det_xy, proj_xy, lkl=1.0):
            """lkl in [0,1] scales the cost, not just a leading factor on the
            probability -- at lkl=0 cost is exactly 0 for every particle
            (exp(0)=1 uniformly), meaning this point contributes zero
            discriminating information between particles, same as if it
            were simply absent. At lkl=1 (a normal, fully-trusted real
            detection) this reduces exactly to the previous unweighted
            formula. Previously a detection's likelihood value was
            completely unused here -- only its (x,y) position mattered,
            so e.g. a low-confidence detection pulled particles just as
            hard as a highly-confident one."""
            resid2 = np.sum((proj_xy - np.asarray(det_xy)) ** 2, axis=1)  # (N,)
            cost = LUMPED_GAMMA_M * resid2 * lkl
            return np.exp(-np.minimum(cost, 60.0))  # exp(-60) underflows harmlessly

        # Per-particle face-normal sign: link4 = left/right axis, link5 = front/back axis
        # (same geometric convention as label_assignment.compute_face_norms,
        # just evaluated per particle instead of for a single best pose).
        R_l4 = robot_arm.baseToJointT[3][:3, :3]
        R_l5 = robot_arm.baseToJointT[4][:3, :3]
        left_z = np.einsum("nij,jk,k->ni", R_eff, R_l4, np.array([0.0, -1.0, 0.0]))[:, 2]
        front_z = np.einsum("nij,jk,k->ni", R_eff, R_l5, np.array([0.0, -1.0, 0.0]))[:, 2]

        for base in LR_BASES | FB_BASES:
            if base not in base_dets:
                continue
            axis_z = left_z if base in LR_BASES else front_z
            face_a, face_b = ("left", "right") if base in LR_BASES else ("front", "back")
            fk_a, fk_b = name2pt.get(f"{base}:{face_a}"), name2pt.get(f"{base}:{face_b}")
            if fk_a is None or fk_b is None:
                continue
            proj_a = _project_batch(fk_a, R_eff, t_eff, K, dist)
            proj_b = _project_batch(fk_b, R_eff, t_eff, K, dist)
            proj = np.where((axis_z < 0)[:, None], proj_a, proj_b)  # (N,2) per-particle pick
            sum_prob += add_term(base_dets[base][:2], proj, base_dets[base][2])
            n_matched += 1

        if (
            "yaw_1" in base_dets
            and "yaw_2" in base_dets
            and "yaw_1" in name2pt
            and "yaw_2" in name2pt
        ):
            proj1 = _project_batch(name2pt["yaw_1"], R_eff, t_eff, K, dist)
            proj2 = _project_batch(name2pt["yaw_2"], R_eff, t_eff, K, dist)
            d1, d2 = np.asarray(base_dets["yaw_1"][:2]), np.asarray(base_dets["yaw_2"][:2])
            lkl1, lkl2 = base_dets["yaw_1"][2], base_dets["yaw_2"][2]
            cost_straight = np.sum((proj1 - d1) ** 2, axis=1) + np.sum((proj2 - d2) ** 2, axis=1)
            cost_swapped = np.sum((proj2 - d1) ** 2, axis=1) + np.sum((proj1 - d2) ** 2, axis=1)
            swap = (cost_swapped < cost_straight)[:, None]
            proj1_final = np.where(swap, proj2, proj1)
            proj2_final = np.where(swap, proj1, proj2)
            sum_prob += add_term(d1, proj1_final, lkl1)
            sum_prob += add_term(d2, proj2_final, lkl2)
            n_matched += 2

        missed = max(LUMPED_N_POINTS - n_matched, 0) * np.exp(-LUMPED_CMAX_PT)
        return np.log(sum_prob + missed + 1e-300)

    return _vec_obs


def create_combined_obs_model_vec(cam_T_b_kin_getter, shaft_radius):
    """Joint point + shaft-edge log-likelihood for a single PF weight update
    (Algorithm 1, lines 20-23: point and edge observations feed the same
    per-particle weight rather than being applied separately).

    Args:
        cam_T_b_kin_getter: callable () -> (4,4) fixed FK extrinsic.
        shaft_radius:       shaft radius [m], or None if this arm has no
                            shaft edge model (falls back to point-only).

    Returns:
        Callable (particles, base_dets, name2pt, robot_arm, K, dist,
                  shaft_pos_b=None, shaft_dir_b=None,
                  detected_edges_hough=None, W=None, H=None,
                  ) -> (N,) log-weights.
    """
    from .shaft_edge import lumped_edge_log_likelihood

    point_fn = create_delta_obs_model_vec(cam_T_b_kin_getter)

    def _combined(
        particles,
        base_dets,
        name2pt,
        robot_arm,
        K,
        dist,
        shaft_pos_b=None,
        shaft_dir_b=None,
        detected_edges_hough=None,
        W=None,
        H=None,
    ):
        log_pts = point_fn(particles, base_dets, name2pt, robot_arm, K, dist)

        if shaft_radius is None or shaft_pos_b is None or shaft_dir_b is None:
            return log_pts

        cam_T_b_kin = cam_T_b_kin_getter()
        R_kin = cam_T_b_kin[:3, :3]
        t_kin = cam_T_b_kin[:3, 3]

        R_delta = Rotation.from_rotvec(particles[:, 3:]).as_matrix()  # (N,3,3)
        R_eff = np.einsum("ij,njk->nik", R_kin, R_delta)  # (N,3,3)
        t_eff = np.einsum("ij,nj->ni", R_kin, particles[:, :3]) + t_kin  # (N,3)

        p_c_0_all = np.einsum("nij,j->ni", R_eff, shaft_pos_b) + t_eff  # (N,3)
        d_c_all = np.einsum("nij,j->ni", R_eff, shaft_dir_b)  # (N,3)
        d_c_all = d_c_all / np.maximum(np.linalg.norm(d_c_all, axis=1, keepdims=True), 1e-9)

        log_edge = lumped_edge_log_likelihood(
            p_c_0_all,
            d_c_all,
            detected_edges_hough,
            shaft_radius,
            W,
            H,
            K,
            gamma_rho=LUMPED_GAMMA_RHO,
            gamma_phi=LUMPED_GAMMA_PHI,
            c_max=LUMPED_CMAX_EDGE,
        )
        return log_pts + log_edge

    return _combined
