"""DLC CSV adapter: standard three/four-row headers, no pandas dependency."""
import csv
from pathlib import PurePosixPath
import re
import numpy as np


def load_dlc_csv(csv_path: str, label_remap: dict, *, individual=None) -> dict:
    """Read one arm's predictions. Multi-tool files require explicit individual.

    Numeric frame IDs and img000123.png paths (Windows or POSIX) are accepted.
    Column coordinates are matched by name, never by an assumed x/y/p order.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    header = {}
    data_start = None
    for index, row in enumerate(rows[:4]):
        if row:
            header[row[0].strip().lower()] = row
            if row[0].strip().lower() == "coords":
                data_start = index + 1
                break
    # Some exported prediction caches omit all index-column header names.
    if data_start is None:
        for index in (2, 3):
            if len(rows) > index and len(rows[index]) > 1 and all(
                    value.strip() in ("x", "y", "likelihood") for value in rows[index][1:]):
                header = {"coords": rows[index], "bodyparts": rows[index - 1]}
                if index == 3:
                    header["individuals"] = rows[1]
                data_start = index + 1
                break
    if data_start is None or "bodyparts" not in header:
        raise ValueError(f"Invalid DLC CSV header: {csv_path}")
    coords, parts = header["coords"], header["bodyparts"]
    ids = header.get("individuals", [""] * len(coords))
    if not len(coords) == len(parts) == len(ids):
        raise ValueError(f"Inconsistent DLC CSV header widths: {csv_path}")
    individuals = list(dict.fromkeys(ids[1:]))
    if individual is None:
        if len(individuals) != 1:
            raise ValueError("Multi-tool CSV requires individual=...; tools must not be merged")
        individual = individuals[0]
    if individual not in individuals:
        raise ValueError(f"Unknown individual {individual!r} in {csv_path}")
    columns = {}
    for col in range(1, len(coords)):
        if ids[col] == individual:
            columns.setdefault(parts[col], {})[coords[col].strip()] = col
    if any(not {"x", "y", "likelihood"}.issubset(c) for c in columns.values()):
        raise ValueError(f"Incomplete x/y/likelihood columns: {csv_path}")
    result = {}
    for row in rows[data_start:]:
        if not row:
            continue
        stem = PurePosixPath(row[0].replace("\\", "/")).stem
        match = re.fullmatch(r"(?:img)?(\d+)", stem)
        if match is None:
            continue
        frame_idx = int(match.group(1))
        if frame_idx in result:
            raise ValueError(f"Duplicate frame {frame_idx} in {csv_path}")
        points = {}
        for part, cols in columns.items():
            try:
                value = tuple(float(row[cols[c]]) for c in ("x", "y", "likelihood"))
            except (IndexError, ValueError):
                continue
            if np.isfinite(value).all():
                points[label_remap.get(part, part)] = value
        result[frame_idx] = points
    return result
