"""Cross-tool shaft observations for the one-PF ablation.

The functions in this module operate on the shared 12-state particle set.
They intentionally contain no gate, reliability scaling, or adaptive noise:
the experiment changes only the representation of the cross-tool shaft
observation (2-D Hough geometry versus reconstructed 3-D ray direction).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .shaft_edge import predict_shaft_edges_hough_batch


def _signed_line_angle(value):
    return (value + np.pi / 2.0) % np.pi - np.pi / 2.0


def _edge_centre(edges):
    edges = np.asarray(edges, dtype=np.float64)
    theta = 0.5 * np.arctan2(
        np.mean(np.sin(2.0 * edges[:, 0])),
        np.mean(np.cos(2.0 * edges[:, 0])),
    )
    return theta % np.pi, float(np.mean(edges[:, 1]))


def axis_from_image_edges(edges, K, width, height):
    """Reconstruct the unsigned 3-D cylinder-axis direction from two edges.

    Each Hough edge is back-projected to a plane through the camera centre.
    Both tangent planes contain the cylinder axis, so their intersection is
    the observed axis ray.  Its sign is intrinsically ambiguous and is
    removed later by an absolute dot product.
    """
    normals = []
    for theta, rho_c in np.asarray(edges, dtype=np.float64):
        rho = rho_c + 0.5 * width * np.cos(theta) + 0.5 * height * np.sin(theta)
        line = np.array([np.cos(theta), np.sin(theta), -rho], dtype=np.float64)
        normal = K.T @ line
        normal /= max(np.linalg.norm(normal), 1e-12)
        normals.append(normal)
    axis = np.cross(normals[0], normals[1])
    norm = np.linalg.norm(axis)
    return None if norm < 1e-8 else axis / norm


def _particle_shaft_geometry(block, tracker, prep):
    anchor = tracker._anchor_T()
    delta_R = Rotation.from_rotvec(block[:, 3:]).as_matrix()
    R_eff = np.einsum("ij,njk->nik", anchor[:3, :3], delta_R)
    t_eff = np.einsum("ij,nj->ni", anchor[:3, :3], block[:, :3]) + anchor[:3, 3]
    positions = np.einsum("nij,j->ni", R_eff, prep["shaft_pos_b"]) + t_eff
    directions = np.einsum("nij,j->ni", R_eff, prep["shaft_dir_b"])
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return positions, directions


def relative_shaft_log_likelihood(particles, trackers, preps, model):
    """Return one cross-tool log-likelihood value for every joint particle."""
    if model not in {"2d", "3d"}:
        raise ValueError(f"unknown relative shaft model: {model}")
    if len(trackers) != 2 or any(i not in preps for i in (0, 1)):
        return np.zeros(len(particles))
    names = list(trackers)
    tr1, tr2 = trackers[names[0]], trackers[names[1]]
    p1, p2 = preps[0], preps[1]
    obs1, obs2 = p1.get("detected_edges_hough"), p2.get("detected_edges_hough")
    if obs1 is None or obs2 is None or len(obs1) != 2 or len(obs2) != 2:
        return np.zeros(len(particles))

    pos1, direction1 = _particle_shaft_geometry(particles[:, :6], tr1, p1)
    pos2, direction2 = _particle_shaft_geometry(particles[:, 6:12], tr2, p2)

    if model == "3d":
        observed1 = axis_from_image_edges(obs1, tr1.K, p1["W"], p1["H"])
        observed2 = axis_from_image_edges(obs2, tr2.K, p2["W"], p2["H"])
        if observed1 is None or observed2 is None:
            return np.zeros(len(particles))
        observed_cosine = abs(float(np.dot(observed1, observed2)))
        predicted_cosine = np.abs(np.sum(direction1 * direction2, axis=1))
        # A broad sigma is deliberate: this is reconstructed from the same
        # edge pixels as the native shaft term and must not dominate it.
        return -0.5 * ((predicted_cosine - observed_cosine) / 0.10) ** 2

    pred1, valid1 = predict_shaft_edges_hough_batch(
        pos1, direction1, tr1._shaft_radius, p1["W"], p1["H"], tr1.K)
    pred2, valid2 = predict_shaft_edges_hough_batch(
        pos2, direction2, tr2._shaft_radius, p2["W"], p2["H"], tr2.K)
    obs_t1, _ = _edge_centre(obs1)
    obs_t2, _ = _edge_centre(obs2)
    pred_t1 = 0.5 * np.arctan2(np.mean(np.sin(2.0 * pred1[:, :, 0]), axis=1),
                               np.mean(np.cos(2.0 * pred1[:, :, 0]), axis=1))
    pred_t2 = 0.5 * np.arctan2(np.mean(np.sin(2.0 * pred2[:, :, 0]), axis=1),
                               np.mean(np.cos(2.0 * pred2[:, :, 0]), axis=1))
    angle_residual = _signed_line_angle(
        _signed_line_angle(obs_t2 - obs_t1) - _signed_line_angle(pred_t2 - pred_t1))
    # Relative-direction-only model: absolute line offsets can be corrupted
    # by shadows/background edges, so rho deliberately contributes no weight.
    value = -0.5 * (angle_residual / np.deg2rad(8.0)) ** 2
    return np.where(valid1 & valid2, value, 0.0)
