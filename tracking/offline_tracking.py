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
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tracking.visualization import draw_legend
from tracking.constants import DLC_W, DLC_H
from tracking.offline_session import OfflineSession


def main(*, allow_primary_switch=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument(
        "--handeye", default="", help="Hand-eye YAML override (default: auto-detect)"
    )
    parser.add_argument("--no_display", action="store_true")
    parser.add_argument("--save_states", default="", help="Save pose states to this .npz path")
    parser.add_argument("--n_particles", type=int, default=500)
    parser.add_argument(
        "--out_dir", default="", help="Output directory (default: METHOD/tracking_output_offline)"
    )
    args = parser.parse_args()

    data_dir = args.data
    session = OfflineSession(
        ROOT, data_dir, args.n_particles, args.handeye,
        allow_primary_switch=allow_primary_switch)
    combined = session.combined
    left_img_dir = session.left_img_dir
    session.warmup()

    method = "APRT" if allow_primary_switch else "FPRT"
    out_base = args.out_dir or os.path.join(ROOT, method, "tracking_output_offline")
    out_img_dir = os.path.join(out_base, "frames")
    os.makedirs(out_img_dir, exist_ok=True)
    out_video_name = f"{os.path.basename(os.path.normpath(data_dir))}_tracking.mp4"
    out_video_path = os.path.join(out_base, out_video_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(
        out_video_path, fourcc, 30.0, (DLC_W, DLC_H))

    for frame_idx in session.frame_indices:
        left_img = cv2.imread(os.path.join(left_img_dir, f"{frame_idx}.png"))
        if left_img is None:
            continue

        vis_l = left_img.copy()
        t0 = time.time()

        results = combined.update(frame_idx, left_img=left_img)

        statuses = session.draw_and_record(
            vis_l, results, frame_idx, save_states=bool(args.save_states))

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
        session.save_states(args.save_states)


if __name__ == "__main__":
    main()
