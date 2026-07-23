"""Kinematic data loading and pose utilities.

Supports two modes:
  - Offline (file-based): load_kin_data(kin_dir, frame_idx)
  - ROS2 (live):          joint_angles_from_msgs(js_msg, jaw_msg)
"""

import json
import os

import numpy as np


def _quat_to_R(q):
    qx, qy, qz, qw = q
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def _mcp_to_T(pos, quat):
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(quat)
    T[:3, 3] = pos
    return T


def load_kin_data(kin_dir: str, frame_idx: int):
    """Load joint angles and base-to-tip transform from a kinematic JSON file.

    Returns:
        joint_angles: (N,) float64 array — measured joint positions including jaw.
        T_base_tip:   (4, 4) homogeneous transform from robot base to tooltip.
    """
    with open(os.path.join(kin_dir, f"{frame_idx}.json")) as f:
        data = json.load(f)
    arm = data[0]["arm"]
    joints = arm["measured_data"]["measured_js"]["position"]
    jaw = [max(0.0, arm["jaw"]["measured_data"]["position"][0])]
    mcp = arm["measured_data"]["measured_cp"]
    return (
        np.array(joints + jaw, dtype=np.float64),
        _mcp_to_T(mcp["position"], mcp["orientation"]),
    )


def joint_angles_from_msgs(js_msg, jaw_msg) -> np.ndarray:
    """Convert dVRK ROS2 JointState messages to joint angle array.

    Args:
        js_msg:  sensor_msgs/JointState from /PSMx/measured_js  (6 joints)
        jaw_msg: sensor_msgs/JointState from /PSMx/jaw/measured_js  (1 joint)

    Returns:
        (7,) float64 array: [j1, j2, j3, j4, j5, j6, jaw]
    """
    joints = list(js_msg.position)
    jaw = list(jaw_msg.position) if jaw_msg is not None else [0.0]
    return np.array(joints + jaw, dtype=np.float64)
