"""
Interactive Canny + HoughLines parameter tuner.

Usage:
    python test_canny_hough.py --data /path/to/dataset --arm psm1
    python test_canny_hough.py --data /path/to/dataset --arm psm2 --frame 100

Sliders control all parameters in real time.
Keys:
    a / left arrow  : previous frame
    d / right arrow : next frame
    p               : save current parameters to {data}/canny_params_{arm}.json
    q               : quit
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tracking import SimpleProjector, build_arm_configs
from tracking.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.dlc_csv import load_dlc_csv
from tracking.kinematics import load_kin_data
from tracking.canny_hough import _polar_to_seg, get_roi, process
from core.RobotLink import RobotLink
from core.utils import axisAngleToRotationMatrix


def draw_result(roi, lines, final):
    vis = roi.copy()
    h, w = roi.shape[:2]
    for rho, theta in lines:
        x1, y1, x2, y2 = _polar_to_seg(rho, theta, w, h)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for rho, theta in final:
        x1, y1, x2, y2 = _polar_to_seg(rho, theta, w, h)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 80, 255), 2)

    # Diagnose failure reason
    if len(lines) == 0:
        reason = "NO HOUGH LINES (in angle band)"
        color = (0, 0, 255)
    elif len(final) < 2:
        reason = f"NO WIDTH-MATCHED PAIR (candidates={len(lines)})"
        color = (0, 180, 255)
    else:
        reason = "OK (2 lines)"
        color = (0, 255, 0)

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        vis,
        f"hough:{len(lines)} final:{len(final)}",
        (2, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    cv2.putText(vis, reason, (2, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis, reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--arm", default="psm1", choices=["psm1", "psm2"])
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument(
        "--handeye", default="", help="Hand-eye YAML override (default: auto-detect)"
    )
    args = parser.parse_args()

    # Auto-detect this dataset's own PnP hand-eye result when --handeye isn't
    # given explicitly (same convention as tracking/offline_tracking.py).
    if not args.handeye:
        cand = os.path.join(args.data, "pnp_handeye_manual.yaml")
        if os.path.exists(cand):
            args.handeye = cand

    left_img_dir = os.path.join(args.data, "regular/image/left_resized")
    arm_cfgs = build_arm_configs(ROOT, args.data, handeye_file=args.handeye)
    cfg = arm_cfgs[args.arm]

    dlc = load_dlc_csv(os.path.join(left_img_dir, cfg["csv_name"]), cfg["label_remap"])
    frame_list = sorted(
        int(os.path.splitext(f)[0])
        for f in os.listdir(cfg["kin_dir"])
        if f.endswith(".json")
        if os.path.exists(os.path.join(left_img_dir, f"{os.path.splitext(f)[0]}.png"))
    )

    intrinsics_npz = os.path.join(ROOT, "Parameter/intrinsics.npz")
    proj = SimpleProjector(intrinsics_npz, DLC_W, DLC_H, CAL_W, CAL_H)
    K = proj.K1

    # FK-based shaft-angle prior: project the robot's actual 3D shaft axis
    # into the image via hand-eye + intrinsics, instead of eyeballing a line
    # through two DLC wrist/jaw keypoints (which mostly reflects the wrist
    # articulation, not the shaft's real orientation).
    robot_file = os.path.join(ROOT, "Parameter", "LND.json")
    robot_arm = RobotLink(robot_file)
    with open(cfg["he_file"]) as f:
        he = yaml.load(f, Loader=yaml.FullLoader)
    he_key = cfg["he_key"]
    cam_T_b_kin = np.eye(4)
    cam_T_b_kin[:3, 3] = np.array(he[f"{he_key}_tvec"]) / 1000.0
    rvec = np.array(he[f"{he_key}_rvec"], dtype=np.float64)
    if cfg["he_rot_loader"] == "axis_angle":
        cam_T_b_kin[:3, :3] = axisAngleToRotationMatrix(rvec)
    else:
        cam_T_b_kin[:3, :3], _ = cv2.Rodrigues(rvec)

    idx = frame_list.index(args.frame) if args.frame and args.frame in frame_list else 0

    WIN = "Canny+Hough tuner"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1400, 600)

    # Trackbars
    cv2.createTrackbar("blur", WIN, 1, 10, lambda x: None)
    cv2.createTrackbar("canny_lo", WIN, 30, 300, lambda x: None)
    cv2.createTrackbar("canny_hi", WIN, 100, 600, lambda x: None)
    # HoughLines' threshold is a raw vote count (~pixels supporting the
    # line), a different scale than HoughLinesP's -- retune by eye.
    cv2.createTrackbar("hough_thr", WIN, 60, 400, lambda x: None)
    cv2.createTrackbar("min_w", WIN, 5, 150, lambda x: None)
    cv2.createTrackbar("max_w", WIN, 60, 150, lambda x: None)
    cv2.createTrackbar("roi_pad", WIN, 140, 300, lambda x: None)
    cv2.createTrackbar("edge_margin", WIN, 30, 150, lambda x: None)
    cv2.createTrackbar("angle_tol", WIN, 15, 90, lambda x: None)
    cv2.createTrackbar("temporal_tol", WIN, 15, 60, lambda x: None)
    cv2.createTrackbar("par_tol", WIN, 5, 30, lambda x: None)

    prev_final_global = (
        None  # [(rho_global, theta), (rho_global, theta)] from the last successful frame
    )

    while True:
        frame_idx = frame_list[idx]
        img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
        if img is None:
            idx = (idx + 1) % len(frame_list)
            continue

        H, W = img.shape[:2]

        # FK tip position from kinematics
        try:
            joint_angles, _ = load_kin_data(cfg["kin_dir"], frame_idx)
        except Exception:
            idx = (idx + 1) % len(frame_list)
            continue

        # Read trackbar values
        params = {
            "blur": max(0, cv2.getTrackbarPos("blur", WIN)),
            "canny_lo": max(1, cv2.getTrackbarPos("canny_lo", WIN)),
            "canny_hi": max(2, cv2.getTrackbarPos("canny_hi", WIN)),
            "hough_thr": max(1, cv2.getTrackbarPos("hough_thr", WIN)),
            "min_w": max(0, cv2.getTrackbarPos("min_w", WIN)),
            "max_w": max(10, cv2.getTrackbarPos("max_w", WIN)),
            "angle_tol": max(0, cv2.getTrackbarPos("angle_tol", WIN)),
            "temporal_tol": max(1, cv2.getTrackbarPos("temporal_tol", WIN)),
            "par_tol": max(0, cv2.getTrackbarPos("par_tol", WIN)),
        }
        roi_pad = max(20, cv2.getTrackbarPos("roi_pad", WIN))
        edge_margin = max(0, cv2.getTrackbarPos("edge_margin", WIN))

        # Use DLC tip position as ROI center (highest-likelihood keypoint)
        frame_dlc = dlc.get(frame_idx, {})
        if not frame_dlc:
            idx = (idx + 1) % len(frame_list)
            continue
        best_kp = max(frame_dlc.values(), key=lambda v: v[2])
        tx, ty = int(round(best_kp[0])), int(round(best_kp[1]))
        if not (0 <= tx < W and 0 <= ty < H):
            idx = (idx + 1) % len(frame_list)
            continue

        # FK-based shaft angle: project the robot's true 3D shaft axis
        # (from joint_angles, already loaded above) into the image.
        robot_arm.updateJointAngles(joint_angles)
        shaft_pos_list, shaft_dir_list, _, _ = robot_arm.getShaftFeatures()
        angle_known = len(shaft_dir_list) > 0
        shaft_angle_2d = 90.0
        if angle_known:
            dir_b = shaft_dir_list[0] / np.linalg.norm(shaft_dir_list[0])
            d_cam = cam_T_b_kin[:3, :3] @ dir_b
            if d_cam[2] < 0:
                d_cam = -d_cam
            if d_cam[2] <= 1e-6:
                # Shaft nearly edge-on to the camera -- projection is
                # numerically unstable, don't trust it this frame.
                angle_known = False
            else:
                theta_pred_dir = np.arctan2(
                    K[1, 1] * d_cam[1] / d_cam[2], K[0, 0] * d_cam[0] / d_cam[2]
                )
                shaft_angle_2d = np.degrees(theta_pred_dir) % 180
        # Only enforce the angle-prior filter when the angle is actually
        # known -- an unreliable fallback value would wrongly reject the
        # real shaft edges.
        params["expected_angle"] = shaft_angle_2d if angle_known else None

        roi, off_x, off_y = get_roi(img, (tx, ty), roi_pad, args.arm, H, W, edge_margin)
        if roi.size == 0:
            idx = (idx + 1) % len(frame_list)
            continue

        # Reproject the previous frame's winning pair into THIS frame's
        # ROI-local coordinates before using it as a temporal prior -- the
        # ROI's vertical origin (off_y) tracks the tip and shifts frame to
        # frame, so a raw rho comparison would be off by that shift.
        if prev_final_global is not None:
            params["prior_rhos"] = [
                rho_g - (off_x * np.cos(theta_g) + off_y * np.sin(theta_g))
                for rho_g, theta_g in prev_final_global
            ]
        else:
            params["prior_rhos"] = None

        canny, _, lines, final = process(roi, params)

        if len(final) == 2:
            prev_final_global = [
                (rho_l + off_x * np.cos(theta_l) + off_y * np.sin(theta_l), theta_l)
                for rho_l, theta_l in final
            ]

        vis_roi, reason = draw_result(roi, lines, final)
        canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

        # Resize all panels to same height for display
        ph = 320

        def resize_h(im):
            h, w = im.shape[:2]
            return cv2.resize(im, (int(w * ph / h), ph))

        img_panels = np.hstack([resize_h(roi), resize_h(canny_bgr), resize_h(vis_roi)])

        # Parameter sidebar
        sidebar_w = 200
        sidebar = np.zeros((ph, sidebar_w, 3), dtype=np.uint8)
        sidebar_lines = [
            ("--- params ---", (180, 180, 180)),
            (f"blur:      {params['blur']}", (200, 200, 200)),
            (f"canny_lo:  {params['canny_lo']}", (200, 200, 200)),
            (f"canny_hi:  {params['canny_hi']}", (200, 200, 200)),
            (f"hough_thr: {params['hough_thr']}", (200, 200, 200)),
            (f"min_w:     {params['min_w']}", (200, 200, 200)),
            (f"max_w:     {params['max_w']}", (200, 200, 200)),
            (f"roi_pad:   {roi_pad}", (200, 200, 200)),
            (f"edge_margin:{edge_margin}", (200, 200, 200)),
            (
                f"angle_tol: {params['angle_tol']}"
                + ("" if angle_known else " (off, angle unknown)"),
                (200, 200, 200),
            ),
            (
                f"temporal_tol:{params['temporal_tol']}"
                + ("" if params.get("prior_rhos") is not None else " (no prior yet)"),
                (200, 200, 200),
            ),
            (f"par_tol:  {params['par_tol']}", (200, 200, 200)),
            ("", (0, 0, 0)),
            ("a/d: prev/next", (160, 200, 160)),
            ("p:   save", (160, 200, 160)),
            ("q:   quit", (160, 200, 160)),
        ]
        for i, (txt, col) in enumerate(sidebar_lines):
            cv2.putText(sidebar, txt, (6, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        panel = np.hstack([sidebar, img_panels])

        info = (
            f"frame {frame_idx}  [{idx+1}/{len(frame_list)}]  ({args.arm})"
            f"  shaft_angle={shaft_angle_2d:.1f}deg  -  {reason}"
        )
        (tw, th), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bar_y = panel.shape[0] - 8
        cv2.rectangle(panel, (0, bar_y - th - 4), (tw + 6, bar_y + 4), (0, 0, 0), -1)
        cv2.putText(panel, info, (3, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.imshow(WIN, panel)

        key = cv2.waitKeyEx(30)
        if key == ord("q"):
            # Auto-save on quit
            save_path = os.path.join(args.data, f"canny_params_{args.arm}.json")
            save_params = {
                k: (
                    int(v)
                    if isinstance(v, (int, np.integer))
                    else bool(v) if isinstance(v, bool) else v
                )
                for k, v in params.items()
                if k not in ("expected_angle", "prior_rhos")
            }
            save_params["roi_pad"] = roi_pad
            save_params["edge_margin"] = edge_margin
            save_params["arm"] = args.arm
            with open(save_path, "w") as f:
                json.dump(save_params, f, indent=2)
            print(f"Parameters saved to {save_path}")
            print(json.dumps(save_params, indent=2))
            break
        elif key in (ord("d"), 65363, 0xFF53):  # right arrow
            idx = min(idx + 1, len(frame_list) - 1)
        elif key in (ord("a"), 65361, 0xFF51):  # left arrow
            idx = max(idx - 1, 0)
        elif key == ord("p"):
            save_path = os.path.join(args.data, f"canny_params_{args.arm}.json")
            save_params = {
                k: int(v) if isinstance(v, (int, np.integer)) else v
                for k, v in params.items()
                if k not in ("expected_angle", "prior_rhos")
            }
            save_params["roi_pad"] = roi_pad
            save_params["edge_margin"] = edge_margin
            save_params["arm"] = args.arm
            with open(save_path, "w") as f:
                json.dump(save_params, f, indent=2)
            print(f"Parameters saved to {save_path}")
            print(json.dumps(save_params, indent=2))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
