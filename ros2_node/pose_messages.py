"""Shared conversion of camera-frame poses to ROS messages."""

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped


def pose_to_stamped(cam_T_b: np.ndarray, stamp, frame_id: str) -> PoseStamped:
    """Convert a 4x4 cam_T_b matrix into a PoseStamped (robot base pose in the camera frame)."""
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    R = cam_T_b[:3, :3]
    t = cam_T_b[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle > 1e-9:
        axis = rvec.flatten() / angle
    else:
        axis = np.array([0.0, 0.0, 1.0])
    # axis-angle → quaternion
    s = np.sin(angle / 2)
    msg.pose.orientation.x = float(axis[0] * s)
    msg.pose.orientation.y = float(axis[1] * s)
    msg.pose.orientation.z = float(axis[2] * s)
    msg.pose.orientation.w = float(np.cos(angle / 2))
    msg.pose.position.x = float(t[0])
    msg.pose.position.y = float(t[1])
    msg.pose.position.z = float(t[2])
    return msg

