"""Reproducible offline detection corruption loaded from a JSON manifest."""

import json
import numpy as np


def load_noise_level(path, level):
    if not path or not level:
        return None
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    spec = manifest["levels"][str(level)]
    return {"level": int(level), "kind": spec["kind"],
            "frames": set(map(int, spec["active_frames"])),
            "events": {int(e["frame"]): e.get("keypoints", []) for e in spec["events"]}}


def corrupt_keypoints(detections, arm_name, other_detections, noise):
    if noise is None or noise["kind"] == "shaft_swap":
        return detections
    output = {f: dict(points) for f, points in detections.items()}
    def matching_key(points, base):
        return next((key for key in points if key.split(":", 1)[0] == base), None)
    for frame, labels in noise["events"].items():
        if frame not in output:
            continue
        if noise["kind"] == "swap":
            for label in labels:
                own_key = matching_key(output[frame], label)
                other_key = matching_key(other_detections.get(frame, {}), label)
                if own_key is not None and other_key is not None:
                    output[frame][own_key] = other_detections[frame][other_key]
        elif noise["kind"] == "drop" and arm_name == "psm2":
            for label in labels:
                own_key = matching_key(output[frame], label)
                if own_key is not None:
                    x, y, _ = output[frame][own_key]
                    output[frame][own_key] = (x, y, 0.0)
    return output


def apply_shaft_swap(trackers, frame_idx, noise):
    """Replace secondary raw Hough observation with primary observation."""
    if noise is None or noise["kind"] != "shaft_swap" or frame_idx not in noise["frames"]:
        return
    primary, secondary = trackers.get("psm1"), trackers.get("psm2")
    if primary is not None and secondary is not None and primary._last_raw_hough:
        secondary._last_raw_hough = list(primary._last_raw_hough)
