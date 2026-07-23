"""Stereo camera projection utilities."""

import cv2
import numpy as np


class SimpleProjector:
    """Projects 3-D camera-frame points into left and right image planes.

    Loads intrinsics and stereo extrinsics from a .npz calibration file.
    Focal lengths and principal points are rescaled from the calibration
    resolution (cal_w × cal_h) down to the inference resolution (dlc_w × dlc_h).
    """

    def __init__(
        self,
        npz_path: str,
        dlc_w: int = 640,
        dlc_h: int = 480,
        cal_w: int = 1920,
        cal_h: int = 1080,
    ):
        d = np.load(npz_path)
        sx, sy = dlc_w / cal_w, dlc_h / cal_h
        K_l = d["K_left"].copy()
        K_r = d["K_right"].copy()
        self.K1 = np.array(
            [
                [K_l[0, 0] * sx, 0, K_l[0, 2] * sx],
                [0, K_l[1, 1] * sy, K_l[1, 2] * sy],
                [0, 0, 1.0],
            ]
        )
        self.K2 = np.array(
            [
                [K_r[0, 0] * sx, 0, K_r[0, 2] * sx],
                [0, K_r[1, 1] * sy, K_r[1, 2] * sy],
                [0, 0, 1.0],
            ]
        )
        self.T_stereo = np.eye(4)
        self.T_stereo[:3, :3] = d["R_stereo"]
        T_st = d["T_stereo"].flatten()
        if np.linalg.norm(T_st) > 1.0:
            T_st = T_st / 1000.0  # mm → m if needed
        self.T_stereo[:3, 3] = T_st

        self.dist1 = (
            d["dist_left"].flatten().astype(np.float64) if "dist_left" in d else np.zeros(5)
        )
        self.dist2 = (
            d["dist_right"].flatten().astype(np.float64) if "dist_right" in d else np.zeros(5)
        )

    def project_points(self, points_cam: np.ndarray):
        """Project (N, 3) camera-frame points to both image planes.

        Returns:
            [proj_left, proj_right] — each (N, 2) in pixel coordinates.
        """
        pts = points_cam.T  # (3, N)
        proj_l = (self.K1 @ (pts / pts[2:3, :]))[:2, :].T
        pts_r = (self.T_stereo @ np.vstack([pts, np.ones((1, pts.shape[1]))]))[:3, :]
        proj_r = (self.K2 @ (pts_r / pts_r[2:3, :]))[:2, :].T
        return [proj_l, proj_r]

    def project_points_distorted(self, points_cam: np.ndarray):
        """Project (N, 3) camera-frame points with full lens distortion.

        Uses cv2.projectPoints (radial + tangential) for both cameras.
        Returns:
            [proj_left, proj_right] — each (N, 2) in pixel coordinates.
        """
        pts = points_cam.reshape(-1, 1, 3).astype(np.float64)
        zero = np.zeros(3, dtype=np.float64)
        proj_l, _ = cv2.projectPoints(pts, zero, zero, self.K1, self.dist1)
        pts_r = (self.T_stereo @ np.hstack([points_cam, np.ones((len(points_cam), 1))]).T).T[:, :3]
        proj_r, _ = cv2.projectPoints(pts_r.reshape(-1, 1, 3), zero, zero, self.K2, self.dist2)
        return [proj_l[:, 0, :], proj_r[:, 0, :]]

    # Keep camelCase alias so existing call sites continue to work.
    def projectPoints(self, points_cam: np.ndarray):
        return self.project_points(points_cam)
