"""Projected wrist/keypoint overlap for visualization.

Primary selection is handled by CombinedArmTracker reliability gating."""

import cv2
import numpy as np


def _project_pt(cam_T_b, pt_b, K, dist):
    """3-D base-frame point -> (2-D pixel, camera-frame depth), or
    (None, z) if the point is behind the camera (z <= 0)."""
    p_cam = cam_T_b[:3, :3] @ np.asarray(pt_b) + cam_T_b[:3, 3]
    if p_cam[2] <= 1e-6:
        return None, p_cam[2]
    pix, _ = cv2.projectPoints(p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    return pix.reshape(2), p_cam[2]


def wrist_overlap(tr_a, tr_b, K, dist):
    """Project both arms' wrist/jaw keypoints (tr.name2pt -- refreshed
    every frame by ArmTracker._prepare_frame()/update(), one 3-D
    base-frame point per DEKR bodypart) into the image using each arm's
    current cam_T_b, and return:

        (overlap_px, z_a, z_b)

    overlap_px: minimum 2-D pixel distance between ANY keypoint of arm_a
                and ANY keypoint of arm_b (all-pairs, typically 7x7=49
                pairs -- cheap).
    z_a, z_b:   camera-frame depth of the specific two keypoints that set
                overlap_px (not a whole-arm average), so depth ordering
                reflects who's actually nearer AT the near-contact point.

    Returns None if either arm has no keypoints that currently project in
    front of the camera (e.g. name2pt not yet populated this frame).
    """

    def _pts_2d(tr):
        pts = []
        for pt_b in tr.name2pt.values():
            pix, z = _project_pt(tr.cam_T_b, pt_b, K, dist)
            if pix is not None:
                pts.append((pix, z))
        return pts

    pts_a = _pts_2d(tr_a)
    pts_b = _pts_2d(tr_b)
    if not pts_a or not pts_b:
        return None

    best = None
    for pa, za in pts_a:
        for pb, zb in pts_b:
            d = np.linalg.norm(pa - pb)
            if best is None or d < best[0]:
                best = (d, za, zb)
    return best
