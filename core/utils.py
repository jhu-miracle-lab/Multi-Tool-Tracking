import numpy as np
import cv2
import imutils


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


def rotationMatrixToAxisAngle(rotation_matrix):
    angle = np.arccos(
        (rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2] - 1.0) / 2.0
    )
    axis = np.array(
        [
            rotation_matrix[2, 1] - rotation_matrix[1, 2],
            rotation_matrix[0, 2] - rotation_matrix[2, 0],
            rotation_matrix[1, 0] - rotation_matrix[0, 1],
        ]
    )
    axis = axis / np.linalg.norm(axis)
    return axis * angle


def batchPoseToMatrix(poses):
    """
    poses: (P, 6) — [tx, ty, tz, omega_x, omega_y, omega_z] for P particles.
    Returns (P, 4, 4) homogeneous transforms. Vectorised version of poseToMatrix.
    """
    P = poses.shape[0]
    T = np.zeros((P, 4, 4))
    T[:, 3, 3] = 1.0
    T[:, :3, 3] = poses[:, :3]

    aa = poses[:, 3:]  # (P, 3) axis-angle
    angles = np.linalg.norm(aa, axis=1)  # (P,)
    R = np.tile(np.eye(3), (P, 1, 1))  # default identity

    nz = angles > 1e-10
    if np.any(nz):
        ax = aa[nz] / angles[nz, np.newaxis]  # (M, 3) unit axes
        c = np.cos(angles[nz])  # (M,)
        s = np.sin(angles[nz])
        t = 1.0 - c
        x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]
        R[nz] = np.stack(
            [
                np.stack([t * x * x + c, t * x * y - s * z, t * x * z + s * y], axis=1),
                np.stack([t * x * y + s * z, t * y * y + c, t * y * z - s * x], axis=1),
                np.stack([t * x * z - s * y, t * y * z + s * x, t * z * z + c], axis=1),
            ],
            axis=1,
        )

    T[:, :3, :3] = R
    return T


def poseToMatrix(pose):
    # Pose is [position, ori]
    T = np.eye(4)
    T[:-1, -1] = np.array(pose[:3])
    T[:-1, :-1] = axisAngleToRotationMatrix(pose[3:])
    return T


def matrixToPose(T):
    pose = np.zeros((6,))
    pose[:3] = T[:-1, -1]
    pose[3:] = rotationMatrixToAxisAngle(T[:-1, :-1])
    return pose


def invertTransformMatrix(T):
    out = np.eye(4)
    out[:-1, :-1] = np.transpose(T[:-1, :-1])
    out[:-1, -1] = -np.dot(out[:-1, :-1], T[:-1, -1])
    return out


def segmentColorAndGetKeyPoints(
    img, hsv_min=(90, 40, 40), hsv_max=(120, 255, 255), draw_contours=False
):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)

    centroids = []
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = M["m10"]
            cY = M["m01"]
        else:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
        centroids.append(np.array([cX, cY]))

    if draw_contours:
        cv2.drawContours(img, cnts, -1, (255, 0, 0), thickness=3)

    return np.array(centroids), img
