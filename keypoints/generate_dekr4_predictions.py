#!/usr/bin/env python3
"""Generate DEKR shuffle4 predictions for offline tracking.

Runs the DEKR dual-tool shuffle4 model on all PNG frames in a dataset's
regular/image/{left,right}_resized directories and writes per-arm CSVs.

Usage:
    python generate_dekr4_predictions.py /path/to/dataset [/path/to/dataset2 ...]

Output (per image directory):
    image_predictions_DEKR_shuffle4_tool1.csv  — PSM1 (right arm)
    image_predictions_DEKR_shuffle4_tool2.csv  — PSM2 (left arm)
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.core.numeric as _ncn

sys.modules["numpy._core.numeric"] = _ncn

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis import get_pose_inference_runner

DEKR4_CFG = (
    "/media/joec/data/Multi-Tool-Tracking-Project/01-code/Deeplabcut/DualTool-DEKR-Mackenzie-2026-05-27"
    "/dlc-models-pytorch/iteration-0/DualTool-DEKRMay27-trainset80shuffle4"
    "/train/pytorch_config.yaml"
)
DEKR4_SNAP = (
    "/media/joec/data/Multi-Tool-Tracking-Project/01-code/Deeplabcut/DualTool-DEKR-Mackenzie-2026-05-27"
    "/dlc-models-pytorch/iteration-0/DualTool-DEKRMay27-trainset80shuffle4"
    "/train/snapshot-best-120.pt"
)
SCORER = "DLC_DekrW32_DualTool-DEKRMay27shuffle4_snapshot_best-120"
BODYPARTS = ["yaw_1", "yaw_2", "roll_1", "roll_2", "pitch_1", "pitch_2", "pitch_3"]
INDIVIDUALS = ["tool1", "tool2"]
BATCH_SIZE = 8


def _build_columns(individual):
    return pd.MultiIndex.from_tuples(
        [(SCORER, individual, bp, coord) for bp in BODYPARTS for coord in ("x", "y", "likelihood")],
        names=["scorer", "individuals", "bodyparts", "coords"],
    )


def process_image_dir(img_dir: str, runner) -> None:
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_paths:
        print(f"  No PNG files found in {img_dir}, skipping.")
        return

    print(f"  {len(img_paths)} frames — running inference...")
    results = runner.inference(tqdm([Path(p) for p in img_paths], leave=False))

    n_bp = len(BODYPARTS)
    nan_row = [np.nan] * (n_bp * 3)

    for ind_idx, individual in enumerate(INDIVIDUALS):
        cols = _build_columns(individual)
        rows = []
        for result in results:
            poses = result["bodyparts"]  # (n_detected, n_bp, 3)
            if ind_idx < len(poses):
                rows.append(poses[ind_idx].flatten().tolist())
            else:
                rows.append(nan_row)

        df = pd.DataFrame(rows, index=img_paths, columns=cols)
        df.index.name = None

        out_name = f"image_predictions_DEKR_shuffle4_{individual}.csv"
        out_path = os.path.join(img_dir, out_name)
        df.to_csv(out_path)
        print(f"  Saved: {out_path}")


def main():
    data_dirs = sys.argv[1:]
    if not data_dirs:
        print("Usage: generate_dekr4_predictions.py /path/to/dataset [...]")
        sys.exit(1)

    print("Loading DEKR shuffle4 model...")
    model_cfg = read_config_as_dict(DEKR4_CFG)
    runner = get_pose_inference_runner(model_cfg, DEKR4_SNAP, batch_size=BATCH_SIZE, device="cuda")
    print("Model loaded.")

    for data_dir in data_dirs:
        print(f"\nDataset: {data_dir}")
        for side in ("left", "right"):
            img_dir = os.path.join(data_dir, "regular", "image", f"{side}_resized")
            if os.path.isdir(img_dir):
                print(f"  [{side}]")
                process_image_dir(img_dir, runner)
            else:
                print(f"  [{side}] not found, skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
