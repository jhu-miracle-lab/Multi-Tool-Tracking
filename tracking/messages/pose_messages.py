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



def publish_rectification_tf(node, projector, raw_frame, rectified_frame):
    """Publish raw-left parent -> virtual rectified-left child as a fixed TF."""
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import StaticTransformBroadcaster
    if raw_frame == rectified_frame:
        raise ValueError("Raw and rectified camera frame IDs must be different")
    stamp = node.get_clock().now().to_msg()
    pose = pose_to_stamped(projector.left_rectification.T, stamp, raw_frame).pose
    transform = TransformStamped()
    transform.header.stamp = stamp
    transform.header.frame_id = raw_frame
    transform.child_frame_id = rectified_frame
    transform.transform.translation.x = pose.position.x
    transform.transform.translation.y = pose.position.y
    transform.transform.translation.z = pose.position.z
    transform.transform.rotation = pose.orientation
    broadcaster = StaticTransformBroadcaster(node)
    broadcaster.sendTransform(transform)
    return broadcaster
