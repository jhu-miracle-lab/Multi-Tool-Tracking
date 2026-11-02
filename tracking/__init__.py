"""Surgical tool tracking package."""

from .frame import ArmFrame, ArmObservation
from .primary_selector import PrimarySelector

from .arm_tracker import ArmTracker
from .combined_tracker import CombinedArmTracker
from .main_tool import wrist_overlap
from .overlap import bidirectional_keypoint_coverage, observation_variance_scale
from .arm_configs import build_arm_configs
from .stereo import SimpleProjector
from .kinematics import load_kin_data
from .visualization import draw_legend, draw_hud, draw_main_tool_hud
from .constants import DLC_W, DLC_H, CAL_W, CAL_H

__all__ = [
    "ArmFrame", "ArmObservation", "PrimarySelector",
    "ArmTracker",
    "CombinedArmTracker",
    "wrist_overlap",
    "bidirectional_keypoint_coverage",
    "observation_variance_scale",
    "build_arm_configs",
    "SimpleProjector",
    "load_kin_data",
    "draw_legend",
    "draw_hud",
    "draw_main_tool_hud",
    "DLC_W",
    "DLC_H",
    "CAL_W",
    "CAL_H",
]
