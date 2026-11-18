"""Image feature detection for arm tracking.

This module owns both keypoint detection/assignment (DEKR/DLC input) and
replaceable shaft-edge detection.  It deliberately has no particle-filter
dependency: its output is a plain ``FrameFeatures`` value consumed by the
tracking layer.
"""

from dataclasses import dataclass

import numpy as np

from tracking.common.constants import LIKELIHOOD_THRESH_OPT, SHAFT_USE_HOUGH
from tracking.features.label_assignment import assign_labels, LR_BASES, FB_BASES
from tracking.features.shaft_detection import create_shaft_extractor
from tracking.features.keypoint_sources import read_keypoints, sanitize_keypoints


@dataclass(frozen=True)
class FrameFeatures:
    corrected: dict
    base_dets: dict
    detected_edges_hough: object
    width: object
    height: object


class ArmFeatureDetector:
    """Stateful detector with no knowledge of particles or PF updates."""

    def __init__(self, name, left_source, right_source, shaft_params=None, *,
                 shaft_factory=create_shaft_extractor, label_assigner=assign_labels):
        self.name = name
        self.left_source = left_source
        self.right_source = right_source
        self.shaft = shaft_factory(name, shaft_params)
        self.assign_labels = label_assigner
        self.curr_lr = "left"
        self.curr_fb = "back"
        self.last_state_label = ""

    @staticmethod
    def get_keypoints(source, frame_idx, image):
        return read_keypoints(source, frame_idx, image)

    def detect_keypoints(self, frame_idx, left_img, right_img, raw_l, raw_r,
                         name2pt, P1, P2, b_T_cam_kin, K, dist, robot_arm,
                         cam_T_b_kin, cam_T_b, right_K=None, right_dist=None):
        if raw_l is None:
            raw_l = self.get_keypoints(self.left_source, frame_idx, left_img)
        if raw_r is None:
            raw_r = self.get_keypoints(self.right_source, frame_idx, right_img)
        raw_l, raw_r = sanitize_keypoints(raw_l), sanitize_keypoints(raw_r)
        corrected, self.curr_lr, self.curr_fb, self.last_state_label = self.assign_labels(
            raw_l, raw_r, name2pt, P1, P2, b_T_cam_kin, K, dist, robot_arm,
            cam_T_b_kin, self.curr_lr, self.curr_fb, cam_T_b=cam_T_b,
            right_K=right_K, right_dist=right_dist)
        base_dets = {}
        for base in LR_BASES | FB_BASES:
            candidates = [raw_l.get(k) for k in (f"{base}:left", f"{base}:right",
                          f"{base}:front", f"{base}:back", base)]
            candidates = [d for d in candidates if d is not None and np.all(np.isfinite(d[:3])) and d[2] >= LIKELIHOOD_THRESH_OPT]
            if candidates:
                base_dets[base] = max(candidates, key=lambda d: d[2])
        for yaw in ("yaw_1", "yaw_2"):
            det = raw_l.get(yaw)
            if det is not None and np.all(np.isfinite(det[:3])) and det[2] >= LIKELIHOOD_THRESH_OPT:
                base_dets[yaw] = det
        return corrected, base_dets

    def detect(self, frame_idx, left_img, right_img, raw_l, raw_r, **context):
        detector_context = {k: v for k, v in context.items()
                            if k not in ("shaft_dir_b", "fallback_keypoints")}
        corrected, base_dets = self.detect_keypoints(
            frame_idx, left_img, right_img, raw_l, raw_r, **detector_context)
        if self.shaft_params is not None and left_img is not None and SHAFT_USE_HOUGH:
            self.shaft.detect(left_img, corrected, K=context["K"], dist=context["dist"],
                              shaft_dir_b=context.get("shaft_dir_b"),
                              cam_T_b_kin=context.get("cam_T_b_kin"),
                              fallback_keypoints=context.get("fallback_keypoints"))
        else:
            self.last_detected_segs = []
            self.last_raw_hough = None
        height, width = left_img.shape[:2] if left_img is not None else (None, None)
        return FrameFeatures(corrected, base_dets, self.last_raw_hough, width, height)


    @property
    def shaft_params(self):
        return self.shaft.shaft_params

    @shaft_params.setter
    def shaft_params(self, value):
        self.shaft.shaft_params = value

    @property
    def last_detected_segs(self):
        return self.shaft.last_detected_segs

    @last_detected_segs.setter
    def last_detected_segs(self, value):
        self.shaft.last_detected_segs = value

    @property
    def last_raw_hough(self):
        return self.shaft.last_raw_hough

    @last_raw_hough.setter
    def last_raw_hough(self, value):
        self.shaft.last_raw_hough = value

    @property
    def ema_lines(self):
        return self.shaft.ema_lines

    @ema_lines.setter
    def ema_lines(self, value):
        self.shaft.ema_lines = value

    @property
    def prev_final_global(self):
        return self.shaft.prev_final_global

    @prev_final_global.setter
    def prev_final_global(self, value):
        self.shaft.prev_final_global = value

    def detect_shaft(self, *args, **kwargs):
        return self.shaft.detect_shaft(*args, **kwargs)

