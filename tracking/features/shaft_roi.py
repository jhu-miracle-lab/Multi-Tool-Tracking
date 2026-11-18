"""Shared image ROI geometry for shaft detectors."""


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

