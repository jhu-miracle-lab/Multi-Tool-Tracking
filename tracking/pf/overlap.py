"""Confidence-filtered detected-keypoint overlap and noise adaptation."""

import numpy as np


def _confident_points(detections, confidence_threshold):
    """Convert ``{label: (x, y, confidence)}`` to an ``(N, 2)`` array."""
    if not detections:
        return np.empty((0, 2), dtype=np.float64)
    points = [
        value[:2] for value in detections.values()
        if len(value) >= 3 and value[2] >= confidence_threshold
    ]
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def bidirectional_keypoint_coverage(
    detections_a,
    detections_b,
    threshold_px=30.0,
    confidence_threshold=0.05,
):
    """Return symmetric coverage of two confidence-filtered detection sets.

    A detection is covered when its nearest detection from the other tool is
    closer than ``threshold_px``.  ``None`` is returned if either accepted
    detection set is empty, because overlap is then not observable.
    """
    if threshold_px <= 0:
        raise ValueError("threshold_px must be positive")
    points_a = _confident_points(detections_a, confidence_threshold)
    points_b = _confident_points(detections_b, confidence_threshold)
    if len(points_a) == 0 or len(points_b) == 0:
        return None
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    coverage_a = np.mean(np.min(distances, axis=1) < threshold_px)
    coverage_b = np.mean(np.min(distances, axis=0) < threshold_px)
    return float((coverage_a + coverage_b) / 2.0)


def observation_variance_scale(coverage, beta=3.0):
    """Return ``R_t/R_0 = 1 + beta*coverage``."""
    if coverage is None or not np.isfinite(coverage):
        return 1.0
    return 1.0 + float(beta) * float(np.clip(coverage, 0.0, 1.0))
