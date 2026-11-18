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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from tracking.common.visualization import draw_legend
from tracking.common.constants import DLC_W, DLC_H
from tracking.messages.offline_session import OfflineSession
from tracking.common.resources import resolve_asset_root
from tracking.messages.tracking_output import TrackingOutput


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
        "--out_dir", default="", help="Output directory (default: ./METHOD/tracking_output_offline)"
    )
    parser.add_argument("--shaft_backend", choices=("elsed", "canny_hough"), default="elsed")
    args = parser.parse_args()

    data_dir = args.data
    session = OfflineSession(
        resolve_asset_root(), data_dir, args.n_particles, args.handeye,
        allow_primary_switch=allow_primary_switch, rectify=False, shaft_backend=args.shaft_backend)
    combined = session.combined
    session.warmup()

    method = "APRT" if allow_primary_switch else "FPRT"
    with TrackingOutput(data_dir, method, args.out_dir,
                        display=not args.no_display, size=(DLC_W, DLC_H)) as output:
        for frame_idx in session.frame_indices:
            input_frame = session.read_frame(frame_idx)
            left_img = input_frame.left_image
            if left_img is None:
                continue

            vis_l = left_img.copy()
            t0 = time.time()

            results = input_frame.update(combined)

            statuses = session.draw_and_record(
                vis_l, results, frame_idx, save_states=bool(args.save_states))

            elapsed = time.time() - t0
            print(f'[{frame_idx:4d}] {elapsed:.3f}s | {" | ".join(statuses)}')

            draw_legend(vis_l)
            output.write(frame_idx, vis_l)

            if not args.no_display:
                cv2.imshow("Offline tracking (q=quit)", vis_l)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key != 255:
                    cv2.waitKey(0)

    print(f"Video saved: {output.video_path}")

    if args.save_states:
        session.save_states(args.save_states)


if __name__ == "__main__":
    main()
