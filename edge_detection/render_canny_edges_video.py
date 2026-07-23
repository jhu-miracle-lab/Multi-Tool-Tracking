"""
Render a Canny+HoughLines edge-detection result video from saved tuner params.

Usage (single arm):
    python render_canny_edges_video.py --data /path/to/session --arm psm1 \\
        --params canny_params.json --out out_psm1.mp4

Usage (both arms side by side in one video):
    python render_canny_edges_video.py --data /path/to/session --arm both \\
        --params-psm1 canny_params_psm1.json --params-psm2 canny_params_psm2.json \\
        --out out_both.mp4
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
from tracking.canny_hough import process, get_roi, _polar_to_seg
from core.RobotLink import RobotLink
from core.utils import axisAngleToRotationMatrix

ARM_COLOR = {"psm1": (0, 140, 255), "psm2": (255, 0, 255)}

DEFAULT_PARAMS = {
    "psm1": {
        "blur": 2,
        "canny_lo": 49,
        "canny_hi": 92,
        "hough_thr": 57,
        "min_w": 38,
        "max_w": 60,
        "angle_tol": 18,
        "temporal_tol": 29,
        "par_tol": 4,
        "roi_pad": 140,
        "edge_margin": 0,
        "arm": "psm1",
    },
    "psm2": {
        "blur": 2,
        "canny_lo": 25,
        "canny_hi": 103,
        "hough_thr": 34,
        "min_w": 43,
        "max_w": 63,
        "angle_tol": 36,
        "temporal_tol": 15,
        "par_tol": 5,
        "roi_pad": 79,
        "edge_margin": 30,
        "arm": "psm2",
    },
}


class ArmState:
    """Per-arm loaded config, DLC predictions, FK/hand-eye setup, and
    running temporal-prior state -- everything needed to process one arm
    frame by frame, kept separate from the other arm's state."""

    def __init__(self, data_dir, arm, params_path, handeye_file=""):
        if params_path is not None:
            with open(params_path) as f:
                saved = json.load(f)
        else:
            saved = DEFAULT_PARAMS[arm]
        self.roi_pad = saved["roi_pad"]
        self.edge_margin = saved["edge_margin"]
        self.base_params = {
            k: v for k, v in saved.items() if k not in ("roi_pad", "edge_margin", "arm")
        }

        self.arm = arm
        self.left_img_dir = os.path.join(data_dir, "regular/image/left_resized")
        # Auto-detect this dataset's own PnP hand-eye result when not given
        # explicitly (same convention as tracking/offline_tracking.py).
        if not handeye_file:
            cand = os.path.join(data_dir, "pnp_handeye_manual.yaml")
            if os.path.exists(cand):
                handeye_file = cand
        self.cfg = build_arm_configs(ROOT, data_dir, handeye_file=handeye_file)[arm]
        self.dlc = load_dlc_csv(
            os.path.join(self.left_img_dir, self.cfg["csv_name"]), self.cfg["label_remap"]
        )
        self.frame_set = set(
            int(os.path.splitext(f)[0])
            for f in os.listdir(self.cfg["kin_dir"])
            if f.endswith(".json")
            if os.path.exists(os.path.join(self.left_img_dir, f"{os.path.splitext(f)[0]}.png"))
        )

        with open(self.cfg["he_file"]) as f:
            he = yaml.load(f, Loader=yaml.FullLoader)
        he_key = self.cfg["he_key"]
        self.cam_T_b_kin = np.eye(4)
        self.cam_T_b_kin[:3, 3] = np.array(he[f"{he_key}_tvec"]) / 1000.0
        rvec = np.array(he[f"{he_key}_rvec"], dtype=np.float64)
        if self.cfg["he_rot_loader"] == "axis_angle":
            self.cam_T_b_kin[:3, :3] = axisAngleToRotationMatrix(rvec)
        else:
            self.cam_T_b_kin[:3, :3], _ = cv2.Rodrigues(rvec)

        self.prev_final_global = None
        self.n_ok = 0
        self.n_total = 0

    def process_frame(self, img, canvas, frame_idx, robot_arm, K):
        """Detect edges for this arm and draw the result directly onto
        `canvas` (the shared full-size output frame, mutated in place).
        Returns a (reason, shaft_angle_2d, tip_xy) status tuple, or None if
        this arm has no data for this frame (missing kinematics/DLC/ROI)."""
        H, W = img.shape[:2]
        if frame_idx not in self.frame_set:
            return None
        try:
            joint_angles, _ = load_kin_data(self.cfg["kin_dir"], frame_idx)
        except Exception:
            return None

        frame_dlc = self.dlc.get(frame_idx, {})
        if not frame_dlc:
            return None
        best_kp = max(frame_dlc.values(), key=lambda v: v[2])
        tx, ty = int(round(best_kp[0])), int(round(best_kp[1]))
        if not (0 <= tx < W and 0 <= ty < H):
            return None

        robot_arm.updateJointAngles(joint_angles)
        shaft_pos_list, shaft_dir_list, _, _ = robot_arm.getShaftFeatures()
        angle_known = len(shaft_dir_list) > 0
        shaft_angle_2d = 90.0
        if angle_known:
            dir_b = shaft_dir_list[0] / np.linalg.norm(shaft_dir_list[0])
            d_cam = self.cam_T_b_kin[:3, :3] @ dir_b
            if d_cam[2] < 0:
                d_cam = -d_cam
            if d_cam[2] <= 1e-6:
                angle_known = False
            else:
                theta_pred_dir = np.arctan2(
                    K[1, 1] * d_cam[1] / d_cam[2], K[0, 0] * d_cam[0] / d_cam[2]
                )
                shaft_angle_2d = np.degrees(theta_pred_dir) % 180

        params = dict(self.base_params)
        params["expected_angle"] = shaft_angle_2d if angle_known else None

        roi, off_x, off_y = get_roi(img, (tx, ty), self.roi_pad, self.arm, H, W, self.edge_margin)
        if roi.size == 0:
            return None

        if self.prev_final_global is not None:
            params["prior_rhos"] = [
                rho_g - (off_x * np.cos(theta_g) + off_y * np.sin(theta_g))
                for rho_g, theta_g in self.prev_final_global
            ]
        else:
            params["prior_rhos"] = None

        _, _, lines, final = process(roi, params)

        if len(final) == 2:
            self.prev_final_global = [
                (rho_l + off_x * np.cos(theta_l) + off_y * np.sin(theta_l), theta_l)
                for rho_l, theta_l in final
            ]

        if len(lines) == 0:
            reason = "NO HOUGH LINES"
        elif len(final) < 2:
            reason = f"NO PAIR (cand={len(lines)})"
        else:
            reason = "OK"

        self.n_total += 1
        if reason == "OK":
            self.n_ok += 1

        color = ARM_COLOR[self.arm]

        for rho, theta in final:
            rho_full = rho + off_x * np.cos(theta) + off_y * np.sin(theta)
            x1, y1, x2, y2 = _polar_to_seg(rho_full, theta, W, H)
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2)

        return reason, shaft_angle_2d, (tx, ty)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--arm", default="psm1", choices=["psm1", "psm2", "both"])
    parser.add_argument(
        "--params",
        default=os.path.join(ROOT, "canny_params.json"),
        help="Params json for --arm psm1/psm2",
    )
    parser.add_argument("--params-psm1", default=None)
    parser.add_argument("--params-psm2", default=None)
    parser.add_argument(
        "--handeye", default="", help="Hand-eye YAML override (default: auto-detect)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    if args.arm == "both":
        arms = {
            "psm1": ArmState(args.data, "psm1", args.params_psm1, args.handeye),
            "psm2": ArmState(args.data, "psm2", args.params_psm2, args.handeye),
        }
    else:
        arms = {args.arm: ArmState(args.data, args.arm, args.params, args.handeye)}

    if args.out is None:
        session = os.path.basename(os.path.normpath(args.data))
        args.out = os.path.join(ROOT, f"canny_edges_{session}_{args.arm}.mp4")

    frame_list = sorted(set().union(*(a.frame_set for a in arms.values())))
    if args.start is not None:
        frame_list = [f for f in frame_list if f >= args.start]
    if args.end is not None:
        frame_list = [f for f in frame_list if f <= args.end]

    intrinsics_npz = os.path.join(ROOT, "Parameter/intrinsics.npz")
    proj = SimpleProjector(intrinsics_npz, DLC_W, DLC_H, CAL_W, CAL_H)
    K = proj.K1

    robot_file = os.path.join(ROOT, "Parameter", "LND.json")
    robot_arm = RobotLink(robot_file)

    left_img_dirs = {name: a.left_img_dir for name, a in arms.items()}
    writer = None

    for frame_idx in frame_list:
        # All arms read from the same left_resized image directory.
        img = None
        for d in left_img_dirs.values():
            candidate = cv2.imread(os.path.join(d, f"{frame_idx}.png"))
            if candidate is not None:
                img = candidate
                break
        if img is None:
            continue

        canvas = img.copy()
        status_lines = [f"frame {frame_idx}"]
        for name in ("psm1", "psm2"):
            if name not in arms:
                continue
            result = arms[name].process_frame(img, canvas, frame_idx, robot_arm, K)
            if result is None:
                status_lines.append(f"{name}: no data")
                continue
            reason, shaft_angle_2d, (tx, ty) = result
            color = ARM_COLOR[name]
            cv2.putText(
                canvas,
                f"{name} {reason}",
                (tx - 90 if name == "psm1" else tx + 6, ty - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            status_lines.append(f"{name}: angle={shaft_angle_2d:.1f}deg {reason}")

        info = "  |  ".join(status_lines)
        cv2.rectangle(
            canvas, (0, canvas.shape[0] - 22), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1
        )
        cv2.putText(
            canvas,
            info,
            (3, canvas.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )

        if writer is None:
            h, w = canvas.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
        writer.write(canvas)

    if writer is not None:
        writer.release()
    for name, a in arms.items():
        print(f"{name}: {a.n_ok}/{a.n_total} frames OK")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
