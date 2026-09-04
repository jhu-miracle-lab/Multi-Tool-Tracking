"""
Standalone offline tracker — no ROS2 required.

Usage:
    python3 offline_tracking.py \
        --data /path/to/dataset \
        [--handeye /path/to/pnp_icp_pickup1.yaml] \
        [--no_display] \
        [--save_states states.npz] \
        [--n_particles 500] \
        [--relative_to psm1]
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

from tracking import (
    ArmTracker,
    CombinedArmTracker,
    JointTracker,
    SimpleProjector,
    bidirectional_keypoint_coverage,
    observation_variance_scale,
    build_arm_configs,
    draw_legend,
    draw_overlap_hud,
)
from tracking.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.dlc_csv import load_dlc_csv
from tracking.detection_noise import load_noise_level, corrupt_keypoints
from tracking.joint_config import load_b1_T_b2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument(
        "--handeye", default="", help="Hand-eye YAML override (default: auto-detect)"
    )
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save_states", default="", help="Save EMA states to this .npz path")
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument("--overlap_neighborhood_px", type=float, default=30.0)
    parser.add_argument("--noise_manifest", default="")
    parser.add_argument("--noise_level", type=int, default=0)
    parser.add_argument("--metrics_only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--out_dir", default="", help="Output directory (default: ROOT/tracking_output_offline)"
    )
    parser.add_argument(
        "--relative_to",
        default="",
        help="Arm name (e.g. 'psm1') to track every other arm relative to, via "
        "ArmTracker.set_relative_to(), instead of each arm's own independent "
        "hand-eye anchor. Default: disabled (every arm tracked independently). "
        "Mutually exclusive with --joint.",
    )
    parser.add_argument(
        "--joint",
        action="store_true",
        help="Joint-optimization mode: track psm1 and psm2 with ONE shared "
        "6-DOF delta_T filter (tracking.joint_tracker.JointTracker), using a "
        "known fixed b1_T_b2 base-to-base transform, instead of two "
        "independent per-arm filters. Mutually exclusive with --relative_to.",
    )
    parser.add_argument(
        "--b1_t_b2",
        default="",
        help="b1_T_b2 YAML path for --joint mode (default: auto-detect "
        "<data>/b1_t_b2.yaml or <ROOT>/b1_t_b2.yaml).",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    use_combined = os.environ.get("MTT_ONE_PF_METHOD3", "0") == "1"
    if args.joint and args.relative_to:
        sys.exit("--joint and --relative_to are mutually exclusive")
    if use_combined and (args.joint or args.relative_to):
        sys.exit(
            "MTT_ONE_PF_METHOD3=1 is mutually exclusive with --joint and "
            "--relative_to; Method 3/4 manages its relative anchor inside "
            "CombinedArmTracker"
        )

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
    noise = load_noise_level(args.noise_manifest, args.noise_level)
    loaded = {}
    for arm_name, cfg in arm_cfgs.items():
        csv_l = os.path.join(left_img_dir, cfg["csv_name"])
        if os.path.exists(csv_l):
            loaded[arm_name] = load_dlc_csv(csv_l, cfg["label_remap"])
    for arm_name, cfg in arm_cfgs.items():
        if not os.path.isdir(cfg["kin_dir"]):
            print(f'[{arm_name}] kin_dir not found, skipping: {cfg["kin_dir"]}')
            continue
        csv_l = os.path.join(left_img_dir, cfg["csv_name"])
        if not os.path.exists(csv_l):
            print(f"[{arm_name}] CSV not found, skipping: {csv_l}")
            continue
        dlc_left = loaded[arm_name]
        if noise is not None:
            other_name = "psm2" if arm_name == "psm1" else "psm1"
            dlc_left = corrupt_keypoints(dlc_left, arm_name, loaded.get(other_name, {}), noise)
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
            own_pf=not use_combined,
        )
        trackers[arm_name].detection_noise = noise
        trackers[arm_name].noise_peers = trackers
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

    # Joint-optimization mode: one shared delta_T filter for psm1+psm2, via a
    # known fixed b1_T_b2 base-to-base transform, instead of independent
    # per-arm filters.
    joint_tracker = None
    combined_tracker = None
    if args.joint:
        if "psm1" not in trackers or "psm2" not in trackers:
            sys.exit(f"--joint requires both 'psm1' and 'psm2' trackers, got {list(trackers)}")
        b1_t_b2_path = args.b1_t_b2
        if not b1_t_b2_path:
            for cand in (os.path.join(data_dir, "b1_t_b2.yaml"), os.path.join(ROOT, "b1_t_b2.yaml")):
                if os.path.exists(cand):
                    b1_t_b2_path = cand
                    break
        if not b1_t_b2_path or not os.path.exists(b1_t_b2_path):
            sys.exit(
                "--joint requires a b1_T_b2 YAML (pass --b1_t_b2, or place b1_t_b2.yaml "
                f"in {data_dir} or {ROOT})"
            )
        b1_T_b2 = load_b1_T_b2(b1_t_b2_path)
        joint_tracker = JointTracker(
            trackers["psm1"], trackers["psm2"], b1_T_b2, n_particles=args.n_particles
        )
    elif use_combined:
        if "psm1" not in trackers or "psm2" not in trackers:
            sys.exit(
                "MTT_ONE_PF_METHOD3=1 requires both psm1 and psm2, got "
                f"{list(trackers)}"
            )
        combined_tracker = CombinedArmTracker(
            trackers, n_particles=args.n_particles
        )

    # Warm-up
    warmup_n = min(50, len(frame_indices))
    if warmup_n > 0:
        print(f"Warm-up: {warmup_n} frames...")
        if joint_tracker is not None:
            joint_tracker.warmup(frame_indices[:warmup_n], lkl_thresh=0.05, left_img_dir=left_img_dir)
        elif combined_tracker is not None:
            combined_tracker.warmup(
                frame_indices[:warmup_n], lkl_thresh=0.05,
                left_img_dir=left_img_dir
            )
        else:
            for tr in trackers.values():
                tr.warmup(frame_indices[:warmup_n], lkl_thresh=0.05, left_img_dir=left_img_dir)
        print("Warm-up done.")

    # Relative tracking: switch every other arm to track relative to the
    # primary's live pose (ArmTracker.set_relative_to()) instead of its own
    # independent hand-eye anchor. Must run after every tracker's warmup()
    # above, since set_relative_to() converts the just-computed absolute
    # warmup estimate into a relative offset from the primary's pose.
    # (Mutually exclusive with --joint, enforced above.)
    if args.relative_to:
        if args.relative_to not in trackers:
            sys.exit(f"--relative_to arm '{args.relative_to}' is not one of {list(trackers)}")
        primary = trackers[args.relative_to]
        for arm_name, tr in trackers.items():
            if arm_name != args.relative_to:
                tr.set_relative_to(primary)

    out_base = args.out_dir if args.out_dir else os.path.join(ROOT, "tracking_output_offline")
    out_img_dir = os.path.join(out_base, "frames")
    if not args.metrics_only:
        os.makedirs(out_img_dir, exist_ok=True)
    out_video_name = f"{os.path.basename(os.path.normpath(data_dir))}_tracking.mp4"
    out_video_path = os.path.join(out_base, out_video_name)

    state_log = {arm: {"frames": [], "ema_states": [], "fk_tips": []} for arm in trackers}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = None if args.metrics_only else cv2.VideoWriter(
        out_video_path, fourcc, 30.0, (DLC_W, DLC_H))

    for frame_idx in frame_indices:
        left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
        if left_img is None:
            continue

        vis_l = left_img.copy()
        t0 = time.time()
        statuses = []

        if joint_tracker is not None:
            corrected_by_arm = joint_tracker.update(frame_idx, left_img=left_img)
            if corrected_by_arm is None:
                continue
            joint_tracker.draw(vis_l, corrected_by_arm)
            n_pts_by_arm = joint_tracker.last_n_dets
        elif combined_tracker is not None:
            corrected_by_arm = combined_tracker.update(
                frame_idx, left_img=left_img
            )
            n_pts_by_arm = {}
            for arm_name, (corrected, n_pts, _) in corrected_by_arm.items():
                if corrected is None:
                    continue
                trackers[arm_name].draw(vis_l, corrected)
                n_pts_by_arm[arm_name] = n_pts
        else:
            n_pts_by_arm = {}
            for arm_name, tr in trackers.items():
                corrected, n_pts, _ = tr.update(frame_idx, left_img=left_img)
                if corrected is None:
                    continue
                tr.draw(vis_l, corrected)
                n_pts_by_arm[arm_name] = n_pts

        for arm_name, tr in trackers.items():
            if arm_name not in n_pts_by_arm:
                continue
            label = f" [{tr._last_state_label}]" if tr._last_state_label else ""
            statuses.append(f"{arm_name}: pts={n_pts_by_arm[arm_name]}{label}")

            if args.save_states:
                tip_base = tr.robot_arm.baseToJointT[-3][:3, 3]
                tip_cam = (tr.cam_T_b_kin @ np.append(tip_base, 1.0))[:3]
                state_log[arm_name]["frames"].append(frame_idx)
                state_log[arm_name]["ema_states"].append(tr.pose_state.copy())
                state_log[arm_name]["fk_tips"].append(tip_cam.copy())

        coverage = None
        if len(trackers) == 2:
            tracker_a, tracker_b = trackers.values()
            coverage = bidirectional_keypoint_coverage(
                getattr(tracker_a, "last_keypoint_detections", {}),
                getattr(tracker_b, "last_keypoint_detections", {}),
                args.overlap_neighborhood_px,
            )
            scale = observation_variance_scale(coverage, 3.0)
            tracker_a.set_observation_variance_scale(scale)
            tracker_b.set_observation_variance_scale(scale)

        elapsed = time.time() - t0
        overlap_text = "n/a" if coverage is None else f"{coverage * 100:.0f}%"
        print(
            f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}'
            f" | overlap={overlap_text}"
        )

        if not args.metrics_only:
            draw_overlap_hud(vis_l, coverage)
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

    if out_video is not None:
        out_video.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print(f"Video saved: {out_video_path}")

    if args.save_states:
        save_parent = os.path.dirname(os.path.abspath(args.save_states))
        os.makedirs(save_parent, exist_ok=True)
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
