"""Tracking package with lazy public exports.

Importing messages/features alone does not import PF, SciPy, ROS or a model.
Submodules use explicit messages, features, pf and common package paths.
"""
from importlib import import_module as _import_module

_EXPORTS = {'ArmFrame': 'common.frame', 'ArmObservation': 'common.frame', 'PrimarySelector': 'pf.primary_selector', 'ArmTracker': 'pf.arm_tracker', 'CombinedArmTracker': 'pf.combined_tracker', 'wrist_overlap': 'pf.main_tool', 'bidirectional_keypoint_coverage': 'pf.overlap', 'observation_variance_scale': 'pf.overlap', 'build_arm_configs': 'messages.arm_configs', 'SimpleProjector': 'common.stereo', 'load_kin_data': 'messages.kinematics', 'draw_legend': 'common.visualization', 'draw_hud': 'common.visualization', 'draw_main_tool_hud': 'common.visualization', 'DLC_W': 'common.constants', 'DLC_H': 'common.constants', 'CAL_W': 'common.constants', 'CAL_H': 'common.constants'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(_import_module("tracking." + _EXPORTS[name]), name)
    globals()[name] = value
    return value
