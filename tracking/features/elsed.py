"""ELSED candidates with ROI-local shaft-pair selection.

Candidate geometry and length scoring follow test_code/elsed_roi/benchmark.py.
The shared shaft extractor supplies ROI, angle prior, and display handling.
"""
import numpy as np
import cv2
from tracking.features.shaft_config import ELSED_PARAMS, ELSED_NATIVE_PARAMS


def detect_segments(roi, params=None):
    """Return fresh ELSED segments; never silently substitute another backend."""
    try:
        from tracking.features import _elsed_native
    except ImportError as exc:
        raise RuntimeError(
            "ELSED extension is unavailable for this Python interpreter. "
            "Build the project with OpenCV development libraries and pybind11, "
            "or select shaft_backend=canny_hough."
        ) from exc
    if min(roi.shape[:2]) < 3:
        return np.empty((0, 4), dtype=np.float32)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    _elsed_native.set_num_threads(1)
    settings = {**ELSED_NATIVE_PARAMS, **{k: v for k, v in (params or {}).items() if k in ELSED_NATIVE_PARAMS}}
    segments, _ = _elsed_native.detect(gray, **settings)
    return segments


def select_pair(segments, shape, params):
    """Return two (rho, theta) lines in ROI top-left pixel coordinates."""
    params = {**ELSED_PARAMS, **params}
    segments = np.asarray(segments, dtype=float).reshape(-1, 4)
    segments = segments[np.isfinite(segments).all(axis=1)]
    delta = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(delta, axis=1)
    keep = lengths > 1e-6
    segments, delta, lengths = segments[keep], delta[keep], lengths[keep]
    theta = np.arctan2(delta[:, 0], -delta[:, 1]) % np.pi
    expected = params.get("expected_angle")
    if expected is not None:
        normal_angle = np.deg2rad(expected + 90)
        error = np.abs((theta - normal_angle + np.pi / 2) % np.pi - np.pi / 2)
        keep = error <= np.deg2rad(params["angle_tol"])
        segments, theta, lengths = segments[keep], theta[keep], lengths[keep]
    ids = np.argsort(-lengths, kind="stable")[:int(params["max_candidates"])]
    segments, theta, lengths = segments[ids], theta[ids], lengths[ids]
    if len(theta) < 2:
        return []
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    rho = np.sum(normals * segments[:, :2], axis=1)
    center = np.array([shape[1] / 2, shape[0] / 2])
    distances = rho - normals @ center
    i, j = np.triu_indices(len(theta), 1)
    dot = np.sum(normals[i] * normals[j], axis=1)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    width = np.abs(distances[i] - distances[j] * np.where(dot >= 0, 1, -1))
    keep = ((angle <= np.deg2rad(params["par_tol"]))
            & (width >= params["min_w"]) & (width <= params["max_w"]))
    i, j = i[keep], j[keep]
    if not len(i):
        return []
    score = np.minimum(lengths[i], lengths[j]) + params["long_edge_weight"] * np.maximum(lengths[i], lengths[j])
    prior = params.get("prior_lines")
    if prior is not None:
        prior = np.asarray(prior)
        prior_normals = np.column_stack((np.cos(prior[:, 1]), np.sin(prior[:, 1])))
        dots = normals @ prior_normals.T
        prior_distance = prior[:, 0] - prior_normals @ center
        position = np.abs(distances[:, None] - np.where(dots >= 0, 1, -1) * prior_distance)
        angular = np.arccos(np.clip(np.abs(dots), 0, 1))
        straight = (position[i, 0] + position[j, 1]) / 2
        swapped = (position[i, 1] + position[j, 0]) / 2
        direct = straight + params["association_angle_weight"] * (angular[i, 0] + angular[j, 1]) <= swapped + params["association_angle_weight"] * (angular[i, 1] + angular[j, 0])
        score -= params["temporal_weight"] * np.minimum(np.where(direct, straight, swapped), params["temporal_cap_px"])
    best = np.argmax(score)
    return [(float(rho[k]), float(theta[k])) for k in (i[best], j[best])]


def process(roi, params):
    """Match the shared extractor's process interface without running Canny."""
    segments = detect_segments(roi, params)
    return None, None, segments, select_pair(segments, roi.shape, params)
