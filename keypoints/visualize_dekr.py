"""
Overlay DEKR-predicted keypoints on the source images and render a
visualization video.

Usage:
    python visualize_dekr.py \
        --img_dir /path/to/images \
        --csv_tool1 /path/to/tool1.csv \
        --csv_tool2 /path/to/tool2.csv \
        --out output.mp4 \
        [--fps 30] [--thr 0.3]
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

BODYPARTS = ["yaw_1", "yaw_2", "roll_1", "roll_2", "pitch_1", "pitch_2", "pitch_3"]

# BGR colors
TOOL_COLOR = {
    "tool1": (60, 180, 255),  # orange
    "tool2": (255, 160, 40),  # blue
}
BP_COLOR = {
    "yaw_1": (0, 255, 255),
    "yaw_2": (0, 200, 200),
    "roll_1": (0, 255, 80),
    "roll_2": (0, 180, 50),
    "pitch_1": (255, 80, 200),
    "pitch_2": (200, 50, 160),
    "pitch_3": (160, 30, 120),
}
# Skeleton connections between keypoints (within the same tool)
SKELETON = [
    ("yaw_1", "yaw_2"),
    ("roll_1", "roll_2"),
    ("pitch_1", "pitch_2"),
    ("pitch_2", "pitch_3"),
    ("yaw_1", "roll_1"),
    ("yaw_2", "roll_2"),
    ("roll_1", "pitch_1"),
]


def load_csv(path, individual):
    df = pd.read_csv(path, header=[0, 1, 2, 3], index_col=0)
    # Returns {img_path: {bp: (x, y, lik)}}
    result = {}
    for img_path, row in df.iterrows():
        kps = {}
        for bp in BODYPARTS:
            try:
                x = float(
                    row[
                        (row.index.get_level_values(2) == bp)
                        & (row.index.get_level_values(3) == "x")
                    ].iloc[0]
                )
                y = float(
                    row[
                        (row.index.get_level_values(2) == bp)
                        & (row.index.get_level_values(3) == "y")
                    ].iloc[0]
                )
                lik = float(
                    row[
                        (row.index.get_level_values(2) == bp)
                        & (row.index.get_level_values(3) == "likelihood")
                    ].iloc[0]
                )
            except Exception:
                x, y, lik = np.nan, np.nan, 0.0
            kps[bp] = (x, y, lik)
        result[img_path] = kps
    return result


def draw_predictions(img, kps1, kps2, thr):
    vis = img.copy()

    for kps, tool in [(kps1, "tool1"), (kps2, "tool2")]:
        tc = TOOL_COLOR[tool]
        valid = {}

        # Keypoints
        for bp, (x, y, lik) in kps.items():
            if lik < thr or np.isnan(x):
                continue
            px, py = int(round(x)), int(round(y))
            if not (0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]):
                continue
            valid[bp] = (px, py)
            c = BP_COLOR[bp]
            r = max(3, int(6 * lik))
            cv2.circle(vis, (px, py), r, c, -1)
            cv2.circle(vis, (px, py), r + 1, tc, 1)
            cv2.putText(vis, bp, (px + 5, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1)

        # Skeleton lines
        for a, b in SKELETON:
            if a in valid and b in valid:
                cv2.line(vis, valid[a], valid[b], tc, 1, cv2.LINE_AA)

    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--csv_tool1", required=True)
    parser.add_argument("--csv_tool2", required=True)
    parser.add_argument("--out", default="dekr_vis.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--thr", type=float, default=0.3, help="likelihood threshold")
    args = parser.parse_args()

    print("Loading CSVs...")
    pred1 = load_csv(args.csv_tool1, "tool1")
    pred2 = load_csv(args.csv_tool2, "tool2")

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
        test = cv2.imread(os.path.join(args.img_dir, os.path.basename(all_paths[0])))
    H, W = test.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
    if not writer.isOpened():
        print(f"[ERROR] Could not create video: {args.out}")
        sys.exit(1)

    for i, img_path in enumerate(all_paths):
        img = cv2.imread(img_path)
        if img is None:
            img = cv2.imread(os.path.join(args.img_dir, os.path.basename(img_path)))
        if img is None:
            writer.write(np.zeros((H, W, 3), dtype=np.uint8))
            continue

        kps1 = pred1.get(img_path, {bp: (np.nan, np.nan, 0.0) for bp in BODYPARTS})
        kps2 = pred2.get(img_path, {bp: (np.nan, np.nan, 0.0) for bp in BODYPARTS})

        frame = draw_predictions(img, kps1, kps2, args.thr)

        # Frame number + legend
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
    print(f"\nVideo saved: {args.out}")


if __name__ == "__main__":
    main()
