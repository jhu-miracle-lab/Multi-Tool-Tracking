"""Visualization helpers: skeleton projection, keypoint drawing, legend."""

import cv2
import numpy as np

from .constants import LIKELIHOOD_THRESHOLD

_FACE_COLORS = {
    "left": (0, 0, 255),  # red
    "right": (255, 0, 0),  # blue
    "front": (0, 255, 255),  # yellow
    "back": (255, 0, 255),  # magenta
}

_LEGEND_ITEMS = [
    ((0, 255, 0), "line", "Tracked tool (PSM1)"),
    ((0, 165, 255), "line", "Tracked tool (PSM2)"),
    ((255, 255, 0), "line", "Detected shaft edge (Canny+Hough)"),
    ((0, 0, 255), "dot", "Keypoint: left face"),
    ((255, 0, 0), "dot", "Keypoint: right face"),
    ((0, 255, 255), "dot", "Keypoint: front face"),
    ((255, 0, 255), "dot", "Keypoint: back face"),
]


def project_skeleton(robot_arm, cam_T_b, projector, imgs, color=(0, 255, 0), skip_indices=()):
    """Draw the robot skeleton onto each image in imgs.

    skip_indices: segment indices to skip (0-based, matching skeleton_structure).
    """
    for i, seg in enumerate(robot_arm.getSkeletonPoints()):
        if i in skip_indices:
            continue
        pts = np.array(seg)
        pts_h = np.hstack([pts, np.ones((2, 1))])
        pts_cam = (cam_T_b @ pts_h.T).T[:, :3]
        if np.any(pts_cam[:, 2] <= 0):
            continue
        projs = projector.projectPoints(pts_cam)
        for img, proj in zip(imgs, projs):
            try:
                cv2.line(
                    img,
                    (int(proj[0, 0]), int(proj[0, 1])),
                    (int(proj[1, 0]), int(proj[1, 1])),
                    color,
                    2,
                )
            except Exception:
                pass


def draw_keypoints(img, kps_dict: dict, default_color):
    """Draw detected keypoints on img, colour-coded by face label."""
    for bp, (x, y, lkl) in kps_dict.items():
        if lkl < LIKELIHOOD_THRESHOLD:
            continue
        color = next(
            (c for face, c in _FACE_COLORS.items() if face in bp),
            default_color,
        )
        cv2.circle(img, (int(x), int(y)), 5, color, -1)


def draw_legend(img):
    """Overlay the colour legend in the top-left corner of img."""
    lx, ly = 10, 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.40, 1
    for i, (color, kind, label) in enumerate(_LEGEND_ITEMS):
        y_i = ly + i * 18
        if kind == "line":
            cv2.line(img, (lx, y_i), (lx + 24, y_i), color, 2)
        else:
            cv2.circle(img, (lx + 12, y_i), 5, color, -1)
        cv2.putText(img, label, (lx + 30, y_i + 4), font, fs, (255, 255, 255), ft + 1, cv2.LINE_AA)
        cv2.putText(img, label, (lx + 30, y_i + 4), font, fs, (0, 0, 0), ft, cv2.LINE_AA)


def draw_hud(img, fps: float, infer_ms: float):
    """Overlay a "X.X Hz | infer Y.Y ms" status line in the top-right
    corner of img -- same convention as Multi_tool_tracking's
    dlc_live_rt.py HUD."""
    text = f"{fps:.1f} Hz  |  infer {infer_ms:.1f} ms"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
    x = img.shape[1] - tw - 10
    y = th + 12
    cv2.putText(img, text, (x, y), font, fs, (0, 0, 0), ft + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, fs, (255, 255, 255), ft, cv2.LINE_AA)


def draw_main_tool_hud(img, primary_name, overlap_px, trigger_px=None, coverage=None):
    """Overlay "main tool: <name/independent>  |  overlap: Npx" as a second
    status line in the top-right corner, just below draw_hud()'s fps line.

    primary_name: CombinedArmTracker.last_relative_main, or None in joint mode.
    overlap_px: minimum projected keypoint separation, or None if unavailable.
    trigger_px: optional display-only warning threshold. It does not control
                the tracker's reliability gate or primary selection.
    """
    primary_label = primary_name if primary_name else "independent"
    if overlap_px is None:
        overlap_label = "overlap: n/a"
        color = (255, 255, 255)
    else:
        overlap_label = f"overlap: {overlap_px:.0f}px"
        at_risk = trigger_px is not None and overlap_px < trigger_px
        color = (0, 0, 255) if at_risk else (255, 255, 255)  # BGR red vs white

    coverage_label = "coverage: n/a" if coverage is None else f"coverage: {coverage * 100:.0f}%"
    text = f"main tool: {primary_label}  |  {overlap_label}  |  {coverage_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, ft = 0.55, 2
    (tw, th), _ = cv2.getTextSize(text, font, fs, ft)
    x = img.shape[1] - tw - 10
    y = th + 12 + 28  # one line below draw_hud()'s fps/infer-ms line
    cv2.putText(img, text, (x, y), font, fs, (0, 0, 0), ft + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, fs, color, ft, cv2.LINE_AA)
