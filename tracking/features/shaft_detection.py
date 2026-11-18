"""Replaceable per-camera shaft extractor; coordinates are input-image pixels.

Each arm/view owns its own instance. Raw lines are fresh evidence, EMA lines
are display-only. A replacement implements detect() and the state below.
"""
from typing import Protocol
import numpy as np
from tracking.common.constants import SHAFT_EDGE_EMA_ALPHA
from tracking.features.canny_hough import process as _hough_process
from tracking.features.shaft_roi import get_roi
from tracking.features.shaft_config import resolve_shaft_params



class ShaftExtractor(Protocol):
    shaft_params: object
    last_detected_segs: list
    last_raw_hough: object
    ema_lines: object
    prev_final_global: object

    def detect(self, image, keypoints, *, K, dist, shaft_dir_b=None,
               cam_T_b_kin=None, fallback_keypoints=None):
        """Update fresh centre-origin (theta, rho) lines plus display state.

        No detection must set last_raw_hough=None, even if display EMA survives.
        """
        ...


class ShaftEdgeExtractor:
    def __init__(self, name, params=None):
        self.name = name
        self.shaft_params = resolve_shaft_params(name, self.backend, params)
        self.last_detected_segs = []
        self.last_raw_hough = None
        self.ema_lines = None
        self.smooth_display = False
        self.prev_final_global = None

    def detect(self, image, keypoints, *, K, dist, shaft_dir_b=None,
               cam_T_b_kin=None, fallback_keypoints=None):
        self.last_raw_hough = None
        if image is None or self.shaft_params is None:
            self.last_detected_segs = []
            return
        # Preserve the input image and coordinates, including lens distortion.
        self.detect_shaft(image, keypoints, shaft_dir_b, cam_T_b_kin, K, fallback_keypoints)

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
        if self.ema_lines is None:
            return []
        H_full = h_roi + off_y
        W_full = w_roi + off_x
        segs = []
        for rho, theta in self.ema_lines:
            seg = self._hough_to_seg(rho, theta, H_full, W_full)
            if seg is not None:
                segs.append(seg)
        return segs if len(segs) == 2 else []

    def detect_shaft(self, left_img: np.ndarray, corrected: dict,
                     shaft_dir_b=None, cam_T_b_kin=None, K=None,
                     fallback_keypoints=None):
        """Detect fresh lines using the selected backend and shared ROI geometry.

        Dataset shaft_params files separate common geometry and backend settings.
        Backend-specific Canny thresholds apply only to Canny/Hough.
        """
        p = self.shaft_params
        H, W = left_img.shape[:2]

        # Use the strongest detection, or the prior pose projection when
        # detections are missing. Projected points only position the ROI.
        roi_source = corrected or fallback_keypoints
        if roi_source:
            best = max(roi_source.values(), key=lambda v: v[2])
            tx, ty = int(round(best[0])), int(round(best[1]))
        else:
            self.last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
            self.last_raw_hough = None
            return
        if not (0 <= tx < W and 0 <= ty < H):
            self.last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
            self.last_raw_hough = None
            return

        roi, off_x, off_y = get_roi(
            left_img, (tx, ty), p["roi_pad"], self.name, H, W, p.get("edge_margin", 0)
        )
        h_roi, w_roi = roi.shape[:2]

        if roi.size == 0:
            self.last_detected_segs = self._ema_segs_from_lines(off_x, off_y, h_roi, w_roi)
            self.last_raw_hough = None
            return

        # FK shaft-angle prior (fixed kinematic extrinsic -- same convention
        # as test_canny_hough.py's main()), used only to restrict Hough's
        # angle search band, not to draw or correct the tracked pose.
        expected_angle = None
        if shaft_dir_b is not None:
            dir_b = shaft_dir_b / np.linalg.norm(shaft_dir_b)
            d_cam = cam_T_b_kin[:3, :3] @ dir_b
            if d_cam[2] < 0:
                d_cam = -d_cam
            if d_cam[2] > 1e-6:
                theta_pred_dir = np.arctan2(
                    K[1, 1] * d_cam[1] / d_cam[2], K[0, 0] * d_cam[0] / d_cam[2]
                )
                expected_angle = np.degrees(theta_pred_dir) % 180

        params = dict(p)
        params["expected_angle"] = expected_angle
        if self.prev_final_global is not None:
            params["prior_rhos"] = [
                rho_g - (off_x * np.cos(theta_g) + off_y * np.sin(theta_g))
                for rho_g, theta_g in self.prev_final_global
            ]
        else:
            params["prior_rhos"] = None

        params["prior_lines"] = (
            [(rho - off_x * np.cos(theta) - off_y * np.sin(theta), theta)
             for rho, theta in self.prev_final_global]
            if self.prev_final_global is not None else None)
        _, _, _, final = self._process(roi, params)

        if len(final) < 2:
            # No detection: fall back to last EMA value if available. Keep
            # _prev_final_global as-is (don't reset the temporal prior on a
            # single missed frame).
            self.last_detected_segs = self._ema_segs_from_lines(off_x, off_y, h_roi, w_roi)
            self.last_raw_hough = None
            return

        full_lines = [
            (rho_l + off_x * np.cos(theta_l) + off_y * np.sin(theta_l), theta_l)
            for rho_l, theta_l in final
        ]
        full_lines.sort(key=lambda x: x[0])
        self.prev_final_global = full_lines

        # This frame's raw (theta, rho_c) detection for the PF likelihood — image-centre-origin
        # convention (matches shaft_edge.py's line_to_hough), NOT the EMA-smoothed line below.
        # Fed straight into the filter with no temporal smoothing of its own.
        self.last_raw_hough = [
            (theta, rho - (W / 2) * np.cos(theta) - (H / 2) * np.sin(theta))
            for rho, theta in full_lines
        ]
        # Smooth display lines with EMA; PF observations remain unsmoothed.
        a = SHAFT_EDGE_EMA_ALPHA
        if not self.smooth_display or self.ema_lines is None:
            self.ema_lines = full_lines
        else:
            smoothed = []
            for (re, te), (rn, tn) in zip(self.ema_lines, full_lines):
                # Wrap theta to avoid discontinuity near 0/pi boundary
                if abs(tn - te) > np.pi / 2:
                    tn = tn - np.pi if tn > te else tn + np.pi
                smoothed.append(((1 - a) * re + a * rn, (1 - a) * te + a * tn))
            self.ema_lines = smoothed

        # Display state is separate from the raw PF observations.
        self.last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)


class CannyHoughShaftExtractor(ShaftEdgeExtractor):
    """Original Canny/Hough backend."""
    backend = "canny_hough"
    @staticmethod
    def _process(roi, params):
        return _hough_process(roi, params)


class ElsedShaftExtractor(ShaftEdgeExtractor):
    """ELSED backend with shared pinhole coordinates and display state."""
    backend = "elsed"
    @staticmethod
    def _process(roi, params):
        from tracking.features.elsed import process
        return process(roi, params)


def create_shaft_extractor(name, params=None):
    backend = (params or {}).get("backend", "elsed")
    factories = {"elsed": ElsedShaftExtractor, "canny_hough": CannyHoughShaftExtractor}
    if backend not in factories:
        raise ValueError(f"Unknown shaft backend: {backend}")
    return factories[backend](name, params)
