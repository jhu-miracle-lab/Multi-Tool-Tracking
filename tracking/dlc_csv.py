"""Load pre-computed DLC predictions from CSV files (offline pipeline)."""

import csv
import os

import numpy as np


def load_dlc_csv(csv_path: str, label_remap: dict) -> dict:
    """Parse a DLC predictions CSV into a frame-indexed keypoint dict.

    Supports both single-animal (3-row header) and multi-animal
    (4-row header) CSV formats.

    Args:
        csv_path:    Path to the DLC output CSV.
        label_remap: Mapping from raw body-part names to canonical names,
                     e.g. {'roll_1': 'roll_1:back'}.

    Returns:
        {frame_idx: {label: (x, y, likelihood)}}
    """
    try:
        import pandas as pd

        _pandas_available = True
    except ImportError:
        _pandas_available = False

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    is_multi = rows[1][1].strip() not in ("x", "y", "likelihood", "")

    if is_multi:
        if not _pandas_available:
            raise ImportError("pandas is required to parse multi-animal DLC CSVs")
        df = pd.read_csv(csv_path, header=[0, 1, 2, 3], index_col=0)
        scorer = df.columns.get_level_values(0)[0]
        df = df[scorer]
        individuals = df.columns.get_level_values(0).unique().tolist()
        bodyparts = df.columns.get_level_values(1).unique().tolist()

        result = {}
        for idx, row in df.iterrows():
            idx_str = os.path.splitext(os.path.basename(str(idx).replace("\\", "/")))[0]
            idx_str = idx_str.replace("img", "").lstrip("0") or "0"
            try:
                frame_idx = int(idx_str)
            except ValueError:
                continue

            merged = {}
            for bp in bodyparts:
                best = (np.nan, np.nan, 0.0)
                for ind in individuals:
                    try:
                        x = float(row[(ind, bp, "x")])
                        y = float(row[(ind, bp, "y")])
                        p = float(row[(ind, bp, "likelihood")])
                        if p > best[2]:
                            best = (x, y, p)
                    except (KeyError, ValueError):
                        pass
                mapped_bp = label_remap.get(bp, bp)
                if not np.isnan(best[0]):
                    merged[mapped_bp] = best
            result[frame_idx] = merged
        return result

    # Single-animal format
    bodyparts = rows[2][1:]
    coords = rows[3][1:]
    col_meta = list(zip(bodyparts, coords))
    result = {}
    for row in rows[4:]:
        fname = os.path.basename(row[0].replace("\\", "/"))
        try:
            frame_idx = int(os.path.splitext(fname)[0])
        except ValueError:
            continue
        values = row[1:]
        kps = {}
        i = 0
        while i + 2 < len(values):
            bp_raw = col_meta[i][0]
            mapped_bp = label_remap.get(bp_raw, bp_raw)
            try:
                kps[mapped_bp] = (float(values[i]), float(values[i + 1]), float(values[i + 2]))
            except ValueError:
                pass
            i += 3
        result[frame_idx] = kps
    return result
