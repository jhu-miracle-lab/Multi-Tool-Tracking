"""Rigid-transform operations shared by kinematics and calibration loaders."""
import numpy as np


def axisAngleToRotationMatrix(axis_angle):
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-5:
        return np.eye(3)

    axis = axis_angle / angle
    cross_product_mat_axis = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross_product_mat_axis
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
    )


def invertTransformMatrix(T):
    out = np.eye(4)
    out[:-1, :-1] = np.transpose(T[:-1, :-1])
    out[:-1, -1] = -np.dot(out[:-1, :-1], T[:-1, -1])
    return out
