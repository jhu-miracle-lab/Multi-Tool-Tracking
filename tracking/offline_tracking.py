"""
Standalone offline tracker — no ROS2 required.

Usage:
    python3 offline_tracking.py \
        --data /path/to/dataset \
        [--handeye /path/to/pnp_icp_pickup1.yaml] \
        [--no_display] \
        [--save_states states.npz] \
        [--n_particles 500]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tracking import ArmTracker, SimpleProjector, draw_legend, build_arm_configs
from tracking.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.dlc_csv import load_dlc_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument(
        "--handeye", default="", help="Hand-eye YAML override (default: auto-detect)"
    )
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save_states", default="", help="Save EMA states to this .npz path")
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument(
        "--out_dir", default="", help="Output directory (default: ROOT/tracking_output_offline)"
    )
    args = parser.parse_args()

    data_dir = args.data
    left_img_dir = os.path.join(data_dir, "regular", "image", "left_resized")
    right_img_dir = os.path.join(data_dir, "regular", "image", "right_resized")
    robot_file = os.path.join(ROOT, "Parameter", "LND.json")
    intrinsics_npz = os.path.join(ROOT, "Parameter", "intrinsics.npz")

    # Auto-detect this dataset's own PnP hand-eye result when --handeye isn't
    # given explicitly -- each session calibrates its own hand-eye, so the
    # project-root fallback files (if any still exist) shouldn't be assumed.
    if not args.handeye:
        cand = os.path.join(data_dir, "pnp_handeye_manual.yaml")
        if os.path.exists(cand):
            args.handeye = cand

    # Fall back to LND.json found up the data-dir tree
    for up in range(4):
        cand = os.path.normpath(os.path.join(data_dir, *([".."] * up), "LND.json"))
        if os.path.exists(cand):
            robot_file = cand
            break

    projector = SimpleProjector(intrinsics_npz, DLC_W, DLC_H, CAL_W, CAL_H)
    K_dlc = projector.K1
    dist_l = np.load(intrinsics_npz)["dist_left"].flatten().astype(np.float64)

    arm_cfgs = build_arm_configs(ROOT, data_dir, handeye_file=args.handeye)
    trackers = {}
    for arm_name, cfg in arm_cfgs.items():
        if not os.path.isdir(cfg["kin_dir"]):
            print(f'[{arm_name}] kin_dir not found, skipping: {cfg["kin_dir"]}')
            continue
        csv_l = os.path.join(left_img_dir, cfg["csv_name"])
        if not os.path.exists(csv_l):
            print(f"[{arm_name}] CSV not found, skipping: {csv_l}")
            continue
        dlc_left = load_dlc_csv(csv_l, cfg["label_remap"])
        dlc_right = None
        csv_r = os.path.join(right_img_dir, cfg["csv_name"])
        if os.path.exists(csv_r):
            dlc_right = load_dlc_csv(csv_r, cfg["label_remap"])

        # Per-dataset tuned shaft-edge params (from test_canny_hough.py's
        # interactive tuner), if this dataset has one -- falls back to
        # ArmTracker's built-in defaults otherwise.
        shaft_params = None
        shaft_params_path = os.path.join(data_dir, f"canny_params_{arm_name}.json")
        if os.path.exists(shaft_params_path):
            with open(shaft_params_path) as f:
                shaft_params = json.load(f)
            shaft_params.pop("arm", None)

        trackers[arm_name] = ArmTracker(
            arm_name,
            cfg,
            dlc_left,
            dlc_right,
            K_dlc,
            dist_l,
            projector,
            robot_file,
            n_particles=args.n_particles,
            shaft_params=shaft_params,
        )
        print(
            f'[{arm_name}] ready — {len(dlc_left)} frames  he_file={cfg["he_file"]}'
            f'  shaft_params={"dataset" if shaft_params else "default"}'
        )

    if not trackers:
        sys.exit("No trackers initialised.")

    # Frame index: union of all kin dirs, intersected with existing left images
    all_kin = set()
    for tr in trackers.values():
        all_kin |= {
            int(os.path.splitext(f)[0]) for f in os.listdir(tr.kin_dir) if f.endswith(".json")
        }
    frame_indices = sorted(
        i for i in all_kin if os.path.exists(os.path.join(left_img_dir, f"{i}.png"))
    )
    print(f"{len(frame_indices)} frames, arms={list(trackers.keys())}")

    # Warm-up
    warmup_n = min(50, len(frame_indices))
    if warmup_n > 0:
        print(f"Warm-up: {warmup_n} frames...")
        for tr in trackers.values():
            tr.warmup(frame_indices[:warmup_n], lkl_thresh=0.05, left_img_dir=left_img_dir)
        print("Warm-up done.")

    out_base = args.out_dir if args.out_dir else os.path.join(ROOT, "tracking_output_offline")
    out_img_dir = os.path.join(out_base, "frames")
    os.makedirs(out_img_dir, exist_ok=True)
    out_video_name = f"{os.path.basename(os.path.normpath(data_dir))}_tracking.mp4"
    out_video_path = os.path.join(out_base, out_video_name)

    state_log = {arm: {"frames": [], "ema_states": [], "fk_tips": []} for arm in trackers}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (DLC_W, DLC_H))

    for frame_idx in frame_indices:
        left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
        if left_img is None:
            continue

        vis_l = left_img.copy()
        t0 = time.time()
        statuses = []

        for arm_name, tr in trackers.items():
            corrected, n_pts, _ = tr.update(frame_idx, left_img=left_img)
            if corrected is None:
                continue
            tr.draw(vis_l, corrected)
            label = f" [{tr._last_state_label}]" if tr._last_state_label else ""
            statuses.append(f"{arm_name}: pts={n_pts}{label}")

            if args.save_states:
                tip_base = tr.robot_arm.baseToJointT[-3][:3, 3]
                tip_cam = (tr.cam_T_b_kin @ np.append(tip_base, 1.0))[:3]
                state_log[arm_name]["frames"].append(frame_idx)
                state_log[arm_name]["ema_states"].append(tr.pose_state.copy())
                state_log[arm_name]["fk_tips"].append(tip_cam.copy())

        elapsed = time.time() - t0
        print(f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}')

        draw_legend(vis_l)
        out_video.write(vis_l)
        cv2.imwrite(os.path.join(out_img_dir, f"{frame_idx:06d}.png"), vis_l)

        if not args.no_display:
            cv2.imshow("Offline tracking (q=quit)", vis_l)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key != 255:
                cv2.waitKey(0)

    out_video.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Video saved: {out_video_path}")

    if args.save_states:
        save_data = {}
        for arm, log in state_log.items():
            if log["frames"]:
                save_data[f"{arm}_frames"] = np.array(log["frames"])
                save_data[f"{arm}_ema_states"] = np.array(log["ema_states"])
                save_data[f"{arm}_fk_tips"] = np.array(log["fk_tips"])
        np.savez(args.save_states, **save_data)
        print(f"States saved: {args.save_states}")


if __name__ == "__main__":
    main()
