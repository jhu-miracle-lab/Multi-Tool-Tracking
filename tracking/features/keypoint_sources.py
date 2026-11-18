"""Keypoint input adapters, independent of ROS, files and particle filters."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from tracking.common.constants import DEKR_ARM_ORDER, DEKR_BODYPARTS
from tracking.features.label_assignment import instance_to_dets, remap_labels


def sanitize_keypoints(detections):
    """Drop malformed/non-finite observations before geometry or ROI selection."""
    clean = {}
    for label, value in (detections or {}).items():
        try:
            point = np.asarray(value, dtype=float)
            if point.shape == (3,) and np.isfinite(point).all():
                clean[label] = tuple(point)
        except (TypeError, ValueError):
            continue
    return clean


class KeypointSource(Protocol):
    def read(self, frame_idx, image):
        """Return canonical {label: (x, y, confidence)} in input-image pixels."""
        ...


class CachedKeypoints:
    def __init__(self, frames):
        self.frames = frames

    def read(self, frame_idx, image=None):
        return self.frames.get(frame_idx, {})


class ImageKeypoints:
    def __init__(self, infer):
        self.infer = infer

    def read(self, frame_idx, image):
        return {} if image is None else self.infer(image)


def read_keypoints(source, frame_idx, image):
    if source is None:
        return {}
    if hasattr(source, "read"):
        value = source.read(frame_idx, image)
    elif hasattr(source, "infer"):
        value = source.infer(image) if image is not None else {}
    else:
        value = source.get(frame_idx, {})  # legacy cached dictionaries
    return sanitize_keypoints(value)


@dataclass
class StereoKeypoints:
    left: dict
    right: dict


class StereoKeypointProvider(Protocol):
    def infer(self, left_image, right_image):
        """Return {arm_name: StereoKeypoints}; missing detections are empty dicts."""
        ...


class DekrKeypointProvider:
    """Only this adapter knows DEKR instance slots/bodypart ordering."""
    def __init__(self, runner, arm_configs, arm_order=DEKR_ARM_ORDER, bodyparts=DEKR_BODYPARTS):
        self.runner = runner
        self.arm_configs = arm_configs
        self.arm_order = tuple(arm_order)
        self.bodyparts = tuple(bodyparts)

    def infer(self, left_image, right_image):
        left, right = self.runner.infer(left_image, right_image)
        def convert(instances, index, remap):
            if instances is None or index >= len(instances) or instances[index] is None:
                return {}
            return sanitize_keypoints(remap_labels(instance_to_dets(instances[index], self.bodyparts), remap))
        detections = {}
        for index, name in enumerate(self.arm_order):
            if name not in self.arm_configs:
                continue
            remap = self.arm_configs[name]["label_remap"]
            detections[name] = StereoKeypoints(
                convert(left, index, remap), convert(right, index, remap))
        return detections
