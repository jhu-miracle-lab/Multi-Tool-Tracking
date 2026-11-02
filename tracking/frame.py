"""Named per-arm data exchanged by detection, tracking and output layers."""
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np

Keypoints = Dict[str, Tuple[float, float, float]]

@dataclass
class ArmObservation:
    keypoints: Keypoints
    points_3d: Dict[str, np.ndarray]
    shaft_position: Optional[np.ndarray] = None
    shaft_direction: Optional[np.ndarray] = None
    shaft_edges: Optional[Sequence[Tuple[float, float]]] = None
    width: Optional[int] = None
    height: Optional[int] = None

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
