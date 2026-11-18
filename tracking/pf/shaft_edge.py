"""Shaft cylinder-projection observation model.

Live shaft edge detection runs through tracking/canny_hough.py (the same
detector used by test_code/feature_tools/edge_detection/test_canny_hough.py's interactive tuner).
This module only covers the particle-filter side: projecting the FK shaft
cylinder into Hough space per particle and scoring it against the detected
edges (predict_shaft_edges_hough_batch + lumped_edge_log_likelihood),
consumed by tracking/observation.py's create_combined_obs_model_vec().
"""

import numpy as np

# ---------------------------------------------------------------------------
# Cylinder projection (Richter et al. 2022, "Robotic Tool Tracking Under
# Partially Visible Kinematic Chain", Appendix C / Doignon & Mathelin 2007)
# ---------------------------------------------------------------------------


def predict_shaft_edges_hough_batch(p_c_0_all, d_c_all, r, W, H, K):
    """Vectorised per-particle shaft edge prediction via cylinder projection.

    Args:
        p_c_0_all: (N, 3) shaft centreline points in camera frame.
        d_c_all:   (N, 3) unit shaft directions in camera frame.
        r:         shaft radius (same units as p_c_0_all, metres).
        W, H:      image width and height.
        K:         (3, 3) camera intrinsic matrix.

    Returns:
        edges: (N, 2, 2) — axes: [particle, edge_index, (θ, ρ_c)].
        valid: (N,) bool — False where the camera is inside or on the cylinder.
    """
    fx_cam, fy_cam = K[0, 0], K[1, 1]
    cx_cam, cy_cam = K[0, 2], K[1, 2]

    x0, y0, z0 = p_c_0_all[:, 0], p_c_0_all[:, 1], p_c_0_all[:, 2]
    a, b, c = d_c_all[:, 0], d_c_all[:, 1], d_c_all[:, 2]

    p_dot_p = x0**2 + y0**2 + z0**2
    p_dot_d = x0 * a + y0 * b + z0 * c
    disc = p_dot_p - p_dot_d**2 - r**2
    valid = disc > 0
    inv_sqrt = np.where(valid, 1.0 / np.sqrt(np.maximum(disc, 1e-12)), 0.0)

    fx = r * (x0 - a * p_dot_d) * inv_sqrt
    fy = r * (y0 - b * p_dot_d) * inv_sqrt
    fz = r * (z0 - c * p_dot_d) * inv_sqrt

    cx = c * y0 - b * z0
    cy = a * z0 - c * x0
    cz = b * x0 - a * y0

    edge_list = []
    for sign in (+1, -1):
        A = fx + sign * cx
        B = fy + sign * cy
        C = fz + sign * cz

        # (A, B, C) is the tangent plane's normal, giving the shaft edge as
        # a line A*X + B*Y + C = 0 in NORMALIZED camera-ray coordinates
        # (X, Y) = (x/z, y/z) -- i.e. a "unit camera" with f=1, no
        # principal-point offset. Converting to a real pixel-space Hough
        # line requires substituting X=(u-c_x)/f_x, Y=(v-c_y)/f_y (Richter
        # et al. 2022 Eq. 50) BEFORE normalizing to (theta, rho) -- omitting
        # this (as an earlier version of this function did) silently
        # computes theta/rho in the normalized-ray domain instead of pixels
        # (implicitly f_x=f_y=1, c_x=c_y=0), which is off from true pixel
        # rho by roughly a factor of f_x (~1000x for this project's camera)
        # and isn't a fixed rescale of LUMPED_GAMMA_RHO/LUMPED_CMAX_EDGE
        # away since rho is only part of the tuned cost -- see constants.py.
        A, B, C = A / fx_cam, B / fy_cam, C - A * cx_cam / fx_cam - B * cy_cam / fy_cam

        safe_norm = np.maximum(np.sqrt(A**2 + B**2), 1e-12)
        A, B, C = A / safe_norm, B / safe_norm, C / safe_norm

        rho = -C
        theta = np.arctan2(B, A)
        neg = theta < 0
        theta = np.where(neg, theta + np.pi, theta)
        rho = np.where(neg, -rho, rho)
        rho_c = rho - (W / 2) * np.cos(theta) - (H / 2) * np.sin(theta)
        edge_list.append(np.stack([theta, rho_c], axis=1))

    return np.stack(edge_list, axis=1), valid  # (N, 2, 2), (N,)


def lumped_edge_log_likelihood(
    p_c_0_all, d_c_all, detected_edges_hough, r, W, H, K, gamma_rho, gamma_phi, c_max
):
    """Vectorised per-particle shaft edge log-likelihood: L1 association cost,
    summed exponential probability over the two matched edges plus a missed-detection term.

    Edge-label ambiguity (which detected line is which projected edge) is
    resolved per particle: both pairings are tried and the lower-total-cost
    one is kept -- Algorithm 1's per-particle edge association, specialised
    to a closed-form 2-edge swap test instead of a generic greedy matcher.

    Args:
        p_c_0_all:            (N, 3) shaft centreline in camera frame.
        d_c_all:              (N, 3) unit shaft directions.
        detected_edges_hough: [(θ_1, ρ_1), (θ_2, ρ_2)] from the Canny+Hough
                              detector (tracking/canny_hough.py), or
                              None/not length 2 if no detection this frame.
        r, W, H:              shaft radius, image dimensions.
        K:                    (3, 3) camera intrinsic matrix.
        gamma_rho, gamma_phi: L1 cost weights for rho [1/px] and theta [1/rad].
        c_max:                per-edge cost cap used for the missed-detection term.

    Returns:
        (N,) log-likelihood per particle.
    """
    n_particles = len(p_c_0_all)
    if detected_edges_hough is None or len(detected_edges_hough) != 2:
        return np.full(n_particles, np.log(2.0 * np.exp(-c_max) + 1e-300))

    pred_edges, valid = predict_shaft_edges_hough_batch(p_c_0_all, d_c_all, r, W, H, K)

    det_t1, det_r1 = detected_edges_hough[0]
    det_t2, det_r2 = detected_edges_hough[1]

    p1_t, p1_r = pred_edges[:, 0, 0], pred_edges[:, 0, 1]
    p2_t, p2_r = pred_edges[:, 1, 0], pred_edges[:, 1, 1]

    def _adiff(a, b):
        d = np.abs(a - b) % np.pi
        return np.minimum(d, np.pi - d)

    def _l1_cost(pt_t, pt_r, dt, dr):
        return gamma_rho * np.abs(pt_r - dr) + gamma_phi * _adiff(pt_t, dt)

    # Straight pairing: edge1<->det1, edge2<->det2. Swapped: edge1<->det2, edge2<->det1.
    c1_straight = _l1_cost(p1_t, p1_r, det_t1, det_r1)
    c2_straight = _l1_cost(p2_t, p2_r, det_t2, det_r2)
    c1_swapped = _l1_cost(p1_t, p1_r, det_t2, det_r2)
    c2_swapped = _l1_cost(p2_t, p2_r, det_t1, det_r1)

    use_straight = (c1_straight + c2_straight) <= (c1_swapped + c2_swapped)
    c1 = np.where(use_straight, c1_straight, c1_swapped)
    c2 = np.where(use_straight, c2_straight, c2_swapped)

    sum_prob = np.exp(-np.minimum(c1, 60.0)) + np.exp(-np.minimum(c2, 60.0))
    log_edge = np.log(sum_prob + 1e-300)
    return np.where(valid, log_edge, np.log(2.0 * np.exp(-c_max) + 1e-300))
