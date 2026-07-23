#!/usr/bin/env python3
"""Render a DEKR shuffle10 keypoint visualization video for one dataset.

Auto-locates the shuffle10 prediction CSVs (as written by
generate_dekr10_predictions.py) under the dataset's left_resized image
directory, so only the dataset root needs to be given.

Usage:
    python render_dekr10_video.py --data /path/to/dataset [--fps 30] [--thr 0.3]

Output:
    {out_dir or dataset root}/{dataset_basename}_output.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from visualize_dekr import BODYPARTS, TOOL_COLOR, load_csv, draw_predictions

CSV_NAME_TOOL1 = "image_predictions_DEKR_shuffle10_best140_tool1.csv"
CSV_NAME_TOOL2 = "image_predictions_DEKR_shuffle10_best140_tool2.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Dataset root directory")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: dataset root)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--thr", type=float, default=0.3, help="likelihood threshold")
    args = parser.parse_args()

    img_dir = os.path.join(args.data, "regular", "image", "left_resized")
    csv_tool1 = os.path.join(img_dir, CSV_NAME_TOOL1)
    csv_tool2 = os.path.join(img_dir, CSV_NAME_TOOL2)
    for path in (csv_tool1, csv_tool2):
        if not os.path.exists(path):
            sys.exit(
                f"[ERROR] Prediction CSV not found: {path}\n"
                "Run generate_dekr10_predictions.py on this dataset first."
            )

    basename = os.path.basename(os.path.normpath(args.data))
    out_dir = args.out_dir if args.out_dir else args.data
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{basename}_output.mp4")

    print("Loading CSVs...")
    pred1 = load_csv(csv_tool1, "tool1")
    pred2 = load_csv(csv_tool2, "tool2")

    def sort_key(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    all_paths = sorted(pred1.keys(), key=sort_key)
    total = len(all_paths)
    print(f"{total} frames total, thr={args.thr}")

    test = cv2.imread(all_paths[0])
    if test is None:
        test = cv2.imread(os.path.join(img_dir, os.path.basename(all_paths[0])))
    H, W = test.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (W, H))
    if not writer.isOpened():
        sys.exit(f"[ERROR] Could not create video: {out_path}")

    for i, img_path in enumerate(all_paths):
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(os.path.join(img_dir, os.path.basename(img_path)))
        if img is None:
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
            continue

        kps1 = pred1.get(img_path, {bp: (np.nan, np.nan, 0.0) for bp in BODYPARTS})
        kps2 = pred2.get(img_path, {bp: (np.nan, np.nan, 0.0) for bp in BODYPARTS})

        frame = draw_predictions(img, kps1, kps2, args.thr)

        cv2.putText(
            frame,
            f"{i+1}/{total}",
            (W - 90, H - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        cv2.circle(frame, (10, H - 30), 5, TOOL_COLOR["tool1"], -1)
        cv2.putText(
            frame,
            "tool1(PSM1)",
            (18, H - 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            TOOL_COLOR["tool1"],
            1,
        )
        cv2.circle(frame, (10, H - 12), 5, TOOL_COLOR["tool2"], -1)
        cv2.putText(
            frame, "tool2(PSM2)", (18, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TOOL_COLOR["tool2"], 1
        )

        writer.write(frame)

        if (i + 1) % 200 == 0 or i + 1 == total:
            print(f"  {i+1}/{total}", flush=True)

    writer.release()
    print(f"\nVideo saved: {out_path}")


if __name__ == "__main__":
    main()
