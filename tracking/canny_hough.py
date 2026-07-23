"""Shared Canny+HoughLines shaft-edge detection core.

Extracted from edge_detection/test_canny_hough.py (the interactive tuner)
so both the tuner and ArmTracker's live shaft-edge observation use the
exact same detector -- previously ArmTracker had its own hand-ported
HoughLinesP approximation that drifted out of sync with the tuner.
"""

import cv2
import numpy as np


def _polar_to_seg(rho, theta, w, h):
    """Convert a HoughLines (rho, theta) infinite line into a segment long
    enough to span the ROI, for drawing purposes only."""
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    L = w + h
    x1 = int(round(x0 + L * (-b)))
    y1 = int(round(y0 + L * a))
    x2 = int(round(x0 - L * (-b)))
    y2 = int(round(y0 - L * a))
    return x1, y1, x2, y2


def _select_two_polar(lines, min_w=5, max_w=60, prior_rhos=None, temporal_tol=15.0, par_tol=5.0):
    """lines: list of (rho, theta) already restricted to a narrow theta
    band by HoughLines' min_theta/max_theta. Since every candidate shares
    ~the same theta, the perpendicular gap between two of them (the
    shaft's apparent width) is just their rho difference."""
    if len(lines) < 2:
        return []
    mid_w = (min_w + max_w) / 2.0
    par_tol_rad = np.radians(par_tol)
    width_valid = [
        (i, j, abs(lines[i][0] - lines[j][0]))
        for i in range(len(lines))
        for j in range(i + 1, len(lines))
        if min_w <= abs(lines[i][0] - lines[j][0]) <= max_w
        and abs(lines[i][1] - lines[j][1]) <= par_tol_rad
    ]

    if prior_rhos is not None:
        p0, p1 = prior_rhos
        prior_width = abs(p0 - p1)
        best, best_cost = None, None
        for i, j, d in width_valid:
            r_i, r_j = lines[i][0], lines[j][0]
            pos_cost = min(abs(r_i - p0) + abs(r_j - p1), abs(r_i - p1) + abs(r_j - p0))
            width_cost = abs(d - prior_width)
            cost = pos_cost + width_cost
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best = [lines[i], lines[j]]
        return best if best else []

    best, best_dev = None, None
    for i, j, d in width_valid:
        dev = abs(d - mid_w)
        if best_dev is None or dev < best_dev:
            best_dev = dev
            best = [lines[i], lines[j]]
    return best if best else []


def get_roi(img, tip_px, roi_pad, arm_name, H, W, edge_margin=0):
    """ROI window anchored on the frame edge the shaft runs toward
    (right edge for psm1, left edge for psm2), not on the tip position --
    the shaft reliably reaches that edge, so this avoids the near-tip
    jaw/clutter region entirely instead of just nudging away from it.
    `edge_margin` pulls the window in from that edge by a few px if the
    very edge itself has artifacts (vignetting, cable clutter, etc.)."""
    tx, ty = tip_px
    # Extra headroom perpendicular to the shaft (height): the ROI is now
    # anchored on the frame edge, and when the shaft is tilted, stretching
    # this far horizontally means it can drift well away from the tip's y
    # coordinate vertically -- too narrow a window would clip the shaft out.
    y1 = max(0, ty - roi_pad * 3 // 2)
    y2 = min(H - 1, ty + roi_pad * 3 // 2)
    if arm_name == "psm1":
        # Shaft extends to the frame's right edge: take a 1.5*pad-wide
        # window leftward from the right edge (minus edge_margin).
        x2 = max(0, (W - 1) - edge_margin)
        x1 = max(0, x2 - roi_pad * 3 // 2)
    else:
        # Shaft extends to the frame's left edge: take a 1.5*pad-wide
        # window rightward from the left edge (plus edge_margin).
        x1 = min(W - 1, edge_margin)
        x2 = min(W - 1, x1 + roi_pad * 3 // 2)
    if x2 < x1:
        return img[0:0, 0:0], x1, y1
    return img[y1 : y2 + 1, x1 : x2 + 1], x1, y1


def process(roi, params):
    """Run Canny + angle-restricted HoughLines on roi, return
    (canny, skel, lines, final) where `lines` and `final` are (rho, theta)
    tuples in OpenCV's HoughLines convention."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blur = cv2.GaussianBlur(gray, (params["blur"] * 2 + 1, params["blur"] * 2 + 1), 0)

    # --- Canny ---
    canny = cv2.Canny(blur, params["canny_lo"], params["canny_hi"])

    # --- Optional skeletonize ---
    # Lazy import: scikit-image is only needed when do_skel=True, which no
    # tuned config enables by default -- keeps it an optional dependency.
    if params.get("do_skel", False):
        from skimage.morphology import skeletonize

        skel = (skeletonize(canny > 0) * 255).astype(np.uint8)
    else:
        skel = canny

    # --- Angle-restricted HoughLines ---
    # expected_angle/angle_tol are in the DIRECTION-angle convention
    # (arctan2(dy,dx), matching how the FK shaft angle is computed);
    # HoughLines' theta is the NORMAL angle (perpendicular to the line),
    # so shift by 90 deg before building the search window. The window
    # restricts the Hough vote itself instead of filtering results after
    # the fact, so it's immune to Canny fragmenting the edge into pieces
    # too short for HoughLinesP's minLineLength.
    expected_angle = params.get("expected_angle")
    angle_tol = params.get("angle_tol", 15.0)
    if expected_angle is not None:
        normal_deg = (expected_angle + 90.0) % 180.0
        # OpenCV requires 0 <= min_theta <= max_theta <= pi with no
        # wraparound; clamp instead of wrapping (rare edge case when the
        # shaft angle sits right at the 0/180 seam).
        min_theta = max(0.0, np.radians(normal_deg - angle_tol))
        max_theta = min(np.pi, np.radians(normal_deg + angle_tol))
    else:
        min_theta, max_theta = 0.0, np.pi

    raw = cv2.HoughLines(
        skel, 1, np.pi / 180, params["hough_thr"], min_theta=min_theta, max_theta=max_theta
    )
    lines = [(float(l[0][0]), float(l[0][1])) for l in raw] if raw is not None else []

    final = _select_two_polar(
        lines,
        min_w=params["min_w"],
        max_w=params["max_w"],
        prior_rhos=params.get("prior_rhos"),
        temporal_tol=params.get("temporal_tol", 15.0),
        par_tol=params.get("par_tol", 5.0),
    )

    return canny, skel, lines, final
