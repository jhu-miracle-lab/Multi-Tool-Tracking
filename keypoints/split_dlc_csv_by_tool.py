#!/usr/bin/env python3
"""Split a DLC multi-animal CSV into separate per-tool files.

Usage:
    python3 split_dlc_csv_by_tool.py \\
        --input  /path/to/left4_resized...best-90_el.csv \\
        --outdir /media/.../left_resized \\
        --suffix shuffle3_best90

Writes:
    {outdir}/image_predictions_DEKR_{suffix}_tool1.csv
    {outdir}/image_predictions_DEKR_{suffix}_tool2.csv
"""

import argparse
import csv
import os


def split_csv(src: str, outdir: str, suffix: str) -> None:
    with open(src, newline="") as f:
        rows = list(csv.reader(f))

    # Identify header rows: scorer / individuals / bodyparts / coords
    scorer_row = rows[0]
    individuals_row = rows[1]
    bodyparts_row = rows[2]
    coords_row = rows[3]
    data_rows = rows[4:]

    # Discover column indices for each individual
    individuals = {}
    for col_idx, ind in enumerate(individuals_row[1:], start=1):
        ind = ind.strip()
        if ind not in individuals:
            individuals[ind] = []
        individuals[ind].append(col_idx)

    for ind_name, col_idxs in individuals.items():
        header = [
            ["scorer"] + [scorer_row[c] for c in col_idxs],
            ["individuals"] + [individuals_row[c] for c in col_idxs],
            ["bodyparts"] + [bodyparts_row[c] for c in col_idxs],
            ["coords"] + [coords_row[c] for c in col_idxs],
        ]
        out_path = os.path.join(outdir, f"image_predictions_DEKR_{suffix}_{ind_name}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(header)
            for row in data_rows:
                writer.writerow([row[0]] + [row[c] for c in col_idxs])
        print(f"Written: {out_path}  ({len(data_rows)} frames)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source multi-animal DLC CSV")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument(
        "--suffix", default="shuffle3_best90", help="CSV name suffix (default: shuffle3_best90)"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    split_csv(args.input, args.outdir, args.suffix)


if __name__ == "__main__":
    main()
