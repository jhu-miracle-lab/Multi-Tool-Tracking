"""2-D relative shaft-angle likelihood for the shared particle filter."""

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


def _particle_shaft_geometry(block, tracker, prep):
    anchor = tracker.anchor
    delta_R = Rotation.from_rotvec(block[:, 3:]).as_matrix()
    R_eff = np.einsum("ij,njk->nik", anchor[:3, :3], delta_R)
    t_eff = np.einsum("ij,nj->ni", anchor[:3, :3], block[:, :3]) + anchor[:3, 3]
    positions = np.einsum("nij,j->ni", R_eff, prep.shaft_position) + t_eff
    directions = np.einsum("nij,j->ni", R_eff, prep.shaft_direction)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return positions, directions


def relative_shaft_log_likelihood(particles, trackers, observations, particle_slices):
    """Return one cross-tool log-likelihood value for every joint particle."""
    if len(trackers) != 2 or any(name not in observations for name in trackers):
        return np.zeros(len(particles))
    names = list(trackers)
    tr1, tr2 = trackers[names[0]], trackers[names[1]]
    p1, p2 = observations[names[0]], observations[names[1]]
    obs1, obs2 = p1.shaft_edges, p2.shaft_edges
    if obs1 is None or obs2 is None or len(obs1) != 2 or len(obs2) != 2:
        return np.zeros(len(particles))

    pos1, direction1 = _particle_shaft_geometry(particles[:, particle_slices[names[0]]], tr1, p1)
    pos2, direction2 = _particle_shaft_geometry(particles[:, particle_slices[names[1]]], tr2, p2)

    pred1, valid1 = predict_shaft_edges_hough_batch(
        pos1, direction1, tr1.shaft_radius, p1.width, p1.height, tr1.K)
    pred2, valid2 = predict_shaft_edges_hough_batch(
        pos2, direction2, tr2.shaft_radius, p2.width, p2.height, tr2.K)
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
