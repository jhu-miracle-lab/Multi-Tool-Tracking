"""Factory for per-arm runtime configuration dicts."""

import os

import numpy as np


def build_arm_configs(root: str, data_dir: str = "", handeye_file: str = "") -> dict:
    """Build the ARM_CONFIGS dict.

    Args:
        root:         Package root used to search for default hand-eye calibration.
        data_dir:     Dataset root for offline mode (contains regular/kinematic/).
                      Leave empty '' for ROS2 live mode (no file-based kin loading).
        handeye_file: Optional path to a hand-eye YAML to override the defaults
                      for both arms (e.g. a freshly computed ICP result).

    Returns:
        {arm_name: cfg_dict} with all path fields resolved.
    """

    def _handeye_file(arm, candidates):
        paths = [handeye_file] if handeye_file else candidates
        for path in paths:
            if os.path.isfile(path):
                return path
        checked = "\n  ".join(os.fspath(path) for path in paths)
        raise FileNotFoundError(
            f"{arm}: hand-eye calibration file not found. Checked:\n  {checked}\n"
            "Provide an existing hand-eye calibration file.")

    psm1_handeye = _handeye_file("PSM1", [
        os.path.join(root, name)
        for name in ("pnp_icp_ipp1.yaml", "pnp_handeye_icp.yaml", "pnp_handeye.yaml")
    ])
    psm2_handeye = _handeye_file("PSM2", [
        os.path.join(root, "pnp_icp_ipp2.yaml"),
        os.path.join(root, "pnp_handeye_icp.yaml"),
        os.path.join(data_dir, "regular", "pnp_handeye_batch.yaml"),
        os.path.join(data_dir, "regular", "pnp_handeye.yaml"),
        os.path.join(root, "pnp_handeye.yaml"),
    ])

    return {
        "psm1": {
            "kin_dir": os.path.join(data_dir, "regular/kinematic/PSM1"),
            "he_file": psm1_handeye,
            "he_key": "PSM1",
            "he_rot_loader": "rodrigues",
            "process_std": np.array([0.002, 0.002, 0.002, 0.004, 0.004, 0.004]),
            "obs_strategy": "delta_t",
            "label_remap": {
                "roll_1": "roll_1:back",
                "roll_2": "roll_2:right",
                "pitch_1": "pitch_1:right",
                "pitch_2": "pitch_2:back",
                "pitch_3": "pitch_3:back",
            },
            "joint6_offset": 0.1,
            "skeleton_color": (0, 255, 0),
            "kp_color": (0, 255, 0),
            "csv_name": "image_predictions_DEKR_shuffle10_best140_tool1.csv",
        },
        "psm2": {
            "kin_dir": os.path.join(data_dir, "regular/kinematic/PSM2"),
            "he_file": psm2_handeye,
            "he_key": "PSM2",
            "he_rot_loader": "rodrigues",
            "process_std": np.array([0.001, 0.001, 0.001, 0.002, 0.002, 0.002]),
            "obs_strategy": "delta_t",
            "label_remap": {
                "roll_1": "roll_1:back",
                "roll_2": "roll_2:left",
                "pitch_1": "pitch_1:left",
                "pitch_2": "pitch_2:back",
                "pitch_3": "pitch_3:back",
            },
            "joint6_offset": -0.1,
            "dlc_face_vote": False,
            "skeleton_color": (0, 165, 255),
            "kp_color": (0, 165, 255),
            "csv_name": "image_predictions_DEKR_shuffle10_best140_tool2.csv",
        },
    }
