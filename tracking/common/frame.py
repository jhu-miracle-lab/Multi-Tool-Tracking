"""Named per-arm data exchanged by detection, tracking and output layers."""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

Keypoints = Dict[str, Tuple[float, float, float]]

@dataclass
class CameraObservation:
    """Keypoints and shaft lines use the active input image coordinates.

    Raw images retain distortion; shaft extraction performs no internal remapping."""
    keypoints: Keypoints
    shaft_edges: Optional[Sequence[Tuple[float, float]]] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ArmObservation:
    keypoints: Keypoints
    points_3d: Dict[str, np.ndarray]
    shaft_position: Optional[np.ndarray] = None
    shaft_direction: Optional[np.ndarray] = None
    shaft_edges: Optional[Sequence[Tuple[float, float]]] = None
    width: Optional[int] = None
    height: Optional[int] = None
    right: Optional[CameraObservation] = None

    def camera_views(self):
        """Left fields retain the public API used by selection and diagnostics."""
        yield "left", CameraObservation(self.keypoints, self.shaft_edges, self.width, self.height)
        if self.right is not None:
            yield "right", self.right

@dataclass
class ArmFrame:
    observation: Optional[ArmObservation] = None
    corrected_keypoints: Optional[Keypoints] = None
    joint_angles: Optional[np.ndarray] = None
    # Keep the original detector count even after cross-arm reassignment.
    observation_count: int = 0

    @property
    def available(self):
        return self.observation is not None


@dataclass
class TrackingInput:
    """Transport-neutral, already preprocessed input. Missing arms stay absent.

    per_arm values supply joint_angles, raw_l, raw_r. Empty detection dicts mean
    no evidence, while None allows a configured keypoint source to read data.
    Images and keypoints must use the same active camera calibration.
    """
    frame_idx: Optional[int] = None
    left_image: Optional[np.ndarray] = None
    right_image: Optional[np.ndarray] = None
    per_arm: Optional[dict] = None

    def update(self, tracker):
        return tracker.update(self.frame_idx, left_img=self.left_image,
                              right_img=self.right_image, per_arm=self.per_arm)
