"""Kinematic data loading and pose utilities.

Supports two modes:
  - Offline (file-based): load_kin_data(kin_dir, frame_idx)
  - ROS2 (live):          joint_angles_from_msgs(js_msg, jaw_msg)
"""

import json
import os

import numpy as np


def _quat_to_R(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Expected a finite nonzero quaternion [x, y, z, w]")
    qx, qy, qz, qw = q / np.linalg.norm(q)
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
    jaw = arm["jaw"]["measured_data"]["position"]
    mcp = arm["measured_data"]["measured_cp"]
    return (
        validated_joint_angles(joints, jaw),
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
    return validated_joint_angles(js_msg.position, None if jaw_msg is None else jaw_msg.position)


def validated_joint_angles(joints, jaw):
    """One boundary policy for file/ROS input: six joints and one finite jaw."""
    joints = np.asarray(joints, dtype=np.float64)
    jaw = np.asarray(jaw, dtype=np.float64)
    if joints.shape != (6,) or jaw.shape != (1,):
        raise ValueError("Expected six arm joints and one jaw value")
    if not np.isfinite(joints).all() or not np.isfinite(jaw).all():
        raise ValueError("Joint positions must be finite")
    return np.concatenate([joints, np.maximum(jaw, 0.0)])


class FileJointSource:
    """Callable adapter: frame ID -> joint vector or None when file is absent.

    Malformed files raise instead of being silently confused with a missing frame.
    A bag/database source can implement the same call signature.
    """
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, frame_idx):
        if frame_idx is None:
            return None
        try:
            return load_kin_data(self.directory, frame_idx)[0]
        except FileNotFoundError:
            return None
