"""Image feature detection for arm tracking.

This module owns both keypoint detection/assignment (DEKR/DLC input) and
Canny+Hough shaft-edge detection.  It deliberately has no particle-filter
dependency: its output is a plain ``FrameFeatures`` value consumed by the
tracking layer.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .constants import LIKELIHOOD_THRESH_OPT, SHAFT_EDGE_EMA_ALPHA, SHAFT_USE_HOUGH
from .label_assignment import assign_labels, LR_BASES, FB_BASES
from .canny_hough import process as _hough_process, get_roi as _hough_get_roi


DEFAULT_SHAFT_PARAMS = {
    "psm1": {"blur": 2, "canny_lo": 49, "canny_hi": 92, "do_skel": False,
             "hough_thr": 57, "min_w": 38, "max_w": 60, "angle_tol": 18,
             "temporal_tol": 29, "par_tol": 4, "roi_pad": 140, "edge_margin": 0},
    "psm2": {"blur": 2, "canny_lo": 25, "canny_hi": 103, "do_skel": False,
             "hough_thr": 34, "min_w": 43, "max_w": 63, "angle_tol": 36,
             "temporal_tol": 15, "par_tol": 5, "roi_pad": 79, "edge_margin": 30},
}


@dataclass(frozen=True)
class FrameFeatures:
    corrected: dict
    base_dets: dict
    detected_edges_hough: object
    width: object
    height: object


class ArmFeatureDetector:
    """Stateful detector with no knowledge of particles or PF updates."""

    def __init__(self, name, left_source, right_source, shaft_params=None):
        self.name = name
        self.left_source = left_source
        self.right_source = right_source
        self.shaft_params = shaft_params if shaft_params is not None else DEFAULT_SHAFT_PARAMS.get(name)
        self.curr_lr = "left"
        self.curr_fb = "back"
        self.last_state_label = ""
        self.last_detected_segs = []
        self.last_raw_hough = None
        self.ema_lines = None
        self.prev_final_global = None
        self.last_roi_bounds = None
        self.last_roi_source = "none"

    assign_labels = staticmethod(assign_labels)

    @staticmethod
    def get_keypoints(source, frame_idx, image):
        if source is None:
            return {}
        if hasattr(source, "infer"):
            return source.infer(image) if image is not None else {}
        return source.get(frame_idx, {})

    def detect_keypoints(self, frame_idx, left_img, right_img, raw_l, raw_r,
                         name2pt, P1, P2, b_T_cam_kin, K, dist, robot_arm,
                         cam_T_b_kin, cam_T_b):
        if raw_l is None:
            raw_l = self.get_keypoints(self.left_source, frame_idx, left_img)
        if raw_r is None:
            raw_r = self.get_keypoints(self.right_source, frame_idx, right_img)
        corrected, self.curr_lr, self.curr_fb, self.last_state_label = assign_labels(
            raw_l, raw_r, name2pt, P1, P2, b_T_cam_kin, K, dist, robot_arm,
            cam_T_b_kin, self.curr_lr, self.curr_fb, cam_T_b=cam_T_b)
        base_dets = {}
        for base in LR_BASES | FB_BASES:
            candidates = [raw_l.get(k) for k in (f"{base}:left", f"{base}:right",
                          f"{base}:front", f"{base}:back", base)]
            candidates = [d for d in candidates if d is not None and d[2] >= LIKELIHOOD_THRESH_OPT]
            if candidates:
                base_dets[base] = max(candidates, key=lambda d: d[2])
        for yaw in ("yaw_1", "yaw_2"):
            det = raw_l.get(yaw)
            if det is not None and det[2] >= LIKELIHOOD_THRESH_OPT:
                base_dets[yaw] = det
        return corrected, base_dets

    def detect(self, frame_idx, left_img, right_img, raw_l, raw_r, **context):
        detector_context = {k: v for k, v in context.items()
                            if k not in ("detection_noise", "noise_peers", "shaft_dir_b",
                                         "prior_points_px", "fallback_keypoints")}
        corrected, base_dets = self.detect_keypoints(
            frame_idx, left_img, right_img, raw_l, raw_r, **detector_context)
        if self.shaft_params is not None and left_img is not None and SHAFT_USE_HOUGH:
            self.detect_shaft(left_img, corrected, frame_idx=frame_idx,
                              noise=context.get("detection_noise"),
                              noise_peers=context.get("noise_peers"),
                              shaft_dir_b=context.get("shaft_dir_b"),
                              cam_T_b_kin=context.get("cam_T_b_kin"), K=context.get("K"),
                              prior_points_px=(context.get("prior_points_px")
                                  if context.get("prior_points_px") is not None else
                                  [v[:2] for v in (context.get("fallback_keypoints") or {}).values()]))
        else:
            self.last_detected_segs = []
            self.last_raw_hough = None
        height, width = left_img.shape[:2] if left_img is not None else (None, None)
        return FrameFeatures(corrected, base_dets, self.last_raw_hough, width, height)

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

    def detect_shaft(self, left_img: np.ndarray, corrected: dict, frame_idx=None, noise=None,
                     noise_peers=None, shaft_dir_b=None, cam_T_b_kin=None, K=None,
                     prior_points_px=None):
        """Detect shaft edges with the shared Canny+HoughLines pipeline
        (tracking/canny_hough.py) and store the result in
        _last_detected_segs / _last_raw_hough.

        Uses the exact same process()/get_roi() as test_canny_hough.py's
        interactive tuner, so self.shaft_params (loaded from a dataset's
        canny_params_{arm}.json, or _SHAFT_CANNY_PARAMS as a fallback)
        actually takes effect here instead of a separately hand-ported
        approximation drifting out of sync with the tuner.
        """
        p = dict(self.shaft_params)
        H, W = left_img.shape[:2]

        # The horizontal ROI remains anchored at the image edge toward which
        # this arm's shaft extends. A keypoint, or the prior-particle
        # projection when all keypoints are absent, determines its y centre.
        if corrected:
            best = max(corrected.values(), key=lambda v: v[2])
            tx, ty = int(round(best[0])), int(round(best[1]))
            self.last_roi_source = "keypoint"
        else:
            points = np.asarray(prior_points_px, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                points = np.empty((0, 2), dtype=np.float64)
            points = points[np.all(np.isfinite(points), axis=1)]
            points = points[(points[:, 1] >= -H / 2) & (points[:, 1] <= 1.5 * H)]
            if len(points) == 0:
                self.last_roi_source = "none"
                self.last_roi_bounds = None
                self.last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
                self.last_raw_hough = None
                return
            centre = np.median(points, axis=0)
            tx = int(np.clip(round(centre[0]), 0, W - 1))
            ty = int(round(centre[1]))
            y_lo, y_hi = np.percentile(points[:, 1], [5.0, 95.0])
            required_half_height = max(ty - y_lo, y_hi - ty) + 20.0
            adaptive_pad = int(np.ceil(required_half_height / 1.5))
            p["roi_pad"] = min(max(p["roi_pad"], adaptive_pad),
                               max(p["roi_pad"], int(np.ceil(H / 3))))
            self.last_roi_source = "prior_particles"
        if not (0 <= tx < W and 0 <= ty < H):
            self.last_roi_bounds = None
            self.last_detected_segs = self._ema_segs_from_lines(0, 0, H, W)
            self.last_raw_hough = None
            return

        roi, off_x, off_y = _hough_get_roi(
            left_img, (tx, ty), p["roi_pad"], self.name, H, W, p.get("edge_margin", 0)
        )
        h_roi, w_roi = roi.shape[:2]
        self.last_roi_bounds = (off_x, off_y, off_x + w_roi, off_y + h_roi)

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

        _, _, _, final = _hough_process(roi, params)

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
        noise = noise
        frame_idx = frame_idx
        if (self.name == "psm2" and noise is not None
                and noise.get("kind") == "shaft_swap"
                and frame_idx in noise.get("frames", set())):
            primary = (noise_peers or {}).get("psm1")
            if primary is not None and primary.feature_detector.last_raw_hough:
                self.last_raw_hough = list(primary.feature_detector.last_raw_hough)
        if (self.name == "psm2" and noise is not None
                and noise.get("kind") in ("compound_blackout", "compound_drop")
                and frame_idx in noise.get("frames", set())):
            self.last_raw_hough = None
            self.last_detected_segs = []

        # EMA update (drawing / dropout-fallback only, not fed to the PF likelihood)
        a = SHAFT_EDGE_EMA_ALPHA
        if self.ema_lines is None:
            self.ema_lines = full_lines
        else:
            smoothed = []
            for (re, te), (rn, tn) in zip(self.ema_lines, full_lines):
                # Wrap theta to avoid discontinuity near 0/pi boundary
                if abs(tn - te) > np.pi / 2:
                    tn = tn - np.pi if tn > te else tn + np.pi
                smoothed.append(((1 - a) * re + a * rn, (1 - a) * te + a * tn))
            self.ema_lines = smoothed

        # Convert smoothed Hough lines back to segments for obs model
        segs = []
        H_full, W_full = left_img.shape[:2]
        for rho, theta in self.ema_lines:
            seg = self._hough_to_seg(rho, theta, H_full, W_full)
            if seg is not None:
                segs.append(seg)
        self.last_detected_segs = segs if len(segs) == 2 else []
