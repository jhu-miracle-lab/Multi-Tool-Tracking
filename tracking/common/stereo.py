"""Stereo camera projection utilities."""

import cv2
import numpy as np
from pathlib import Path


def _load_opencv_calibration(path):
    """Read dVRK OpenCV stereo calibration; YAML translation is in millimetres."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        if not storage.isOpened():
            raise ValueError(f"Cannot open stereo calibration: {path}")
        values = {}
        for target, source in (
            ("K_left", "K1"), ("K_right", "K2"),
            ("dist_left", "D1"), ("dist_right", "D2"),
            ("R_stereo", "R"), ("T_stereo", "T"),
        ):
            value = storage.getNode(source).mat()
            if value is None or not np.isfinite(value).all():
                raise ValueError(f"Missing or invalid {source} in {path}")
            values[target] = value
        for key in ("K_left", "K_right", "R_stereo"):
            if values[key].shape != (3, 3):
                raise ValueError(f"{key} must have shape (3, 3) in {path}")
        if values["T_stereo"].size != 3:
            raise ValueError(f"T must contain three values in {path}")
        width = storage.getNode("image_width").real()
        height = storage.getNode("image_height").real()
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError(f"Invalid calibration image dimensions in {path}")
        values["T_stereo"] = values["T_stereo"].reshape(3) / 1000.0
        return values, int(width), int(height)
    finally:
        storage.release()


class SimpleProjector:
    """Projects 3-D camera-frame points into left and right image planes.

    Loads OpenCV stereo YAML or legacy NPZ calibration files.
    Focal lengths and principal points are rescaled from the calibration
    resolution (cal_w × cal_h) down to the inference resolution (dlc_w × dlc_h).
    """

    def __init__(
        self,
        calibration_path: str,
        dlc_w: int = 640,
        dlc_h: int = 480,
        cal_w: int = 1920,
        cal_h: int = 1080,
        *,
        rectify: bool = False,
        swap_cameras: bool = False,
    ):
        path = Path(calibration_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stereo calibration file not found: {path}")
        if path.suffix.lower() in (".yaml", ".yml"):
            d, cal_w, cal_h = _load_opencv_calibration(path)
        else:
            with np.load(path) as calibration:
                d = {key: calibration[key] for key in calibration.files}
            translation = d["T_stereo"].flatten()
            if np.linalg.norm(translation) > 1.0:
                translation = translation / 1000.0
            d["T_stereo"] = translation
        if swap_cameras:
            d["K_left"], d["K_right"] = d["K_right"], d["K_left"]
            d["dist_left"], d["dist_right"] = d.get("dist_right", np.zeros(5)), d.get("dist_left", np.zeros(5))
            rotation = d["R_stereo"].T
            d["T_stereo"] = -rotation @ d["T_stereo"].reshape(3)
            d["R_stereo"] = rotation
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
        self.T_stereo[:3, 3] = T_st

        self.dist1 = (
            d["dist_left"].flatten().astype(np.float64) if "dist_left" in d else np.zeros(5)
        )
        self.dist2 = (
            d["dist_right"].flatten().astype(np.float64) if "dist_right" in d else np.zeros(5)
        )

        # Keep original calibration: cached detections and incoming images use it.
        self.image_size = (dlc_w, dlc_h)
        self.raw_K1, self.raw_K2 = self.K1.copy(), self.K2.copy()
        self.raw_dist1, self.raw_dist2 = self.dist1.copy(), self.dist2.copy()
        self.raw_T_stereo = self.T_stereo.copy()
        self.rectified = bool(rectify)
        self.left_rectification = np.eye(4)
        self.right_rectification = np.eye(4)
        self._rectification_maps = {}
        if self.rectified:
            self._initialize_rectification()

    def _initialize_rectification(self):
        """Define virtual rectified cameras and cache image remapping tables."""
        if not np.isfinite(self.raw_T_stereo).all() or np.linalg.norm(self.raw_T_stereo[:3, 3]) <= 1e-12:
            raise ValueError("Stereo rectification requires finite, nonzero camera baseline")
        R1, R2, P1, P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(
            self.raw_K1, self.raw_dist1, self.raw_K2, self.raw_dist2,
            self.image_size, self.raw_T_stereo[:3, :3], self.raw_T_stereo[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0, newImageSize=self.image_size)
        self.left_rectification[:3, :3] = R1
        self.right_rectification[:3, :3] = R2
        self.K1, self.K2 = P1[:, :3].copy(), P2[:, :3].copy()
        # right_rect <- right_raw <- left_raw <- left_rect
        self.T_stereo = (self.right_rectification @ self.raw_T_stereo
                         @ self.left_rectification.T)
        self.dist1, self.dist2 = np.zeros(5), np.zeros(5)
        for side, K, D, R, new_K in (
            ("left", self.raw_K1, self.raw_dist1, R1, self.K1),
            ("right", self.raw_K2, self.raw_dist2, R2, self.K2)):
            self._rectification_maps[side] = cv2.initUndistortRectifyMap(
                K, D, R, new_K, self.image_size, cv2.CV_32FC1)

    def process_image(self, image, side):
        """Resize and remap one RAW image, once at the input boundary."""
        if side not in ("left", "right"):
            raise ValueError(f"Unknown camera side: {side}")
        if image is None:
            return None
        if (image.shape[1], image.shape[0]) != self.image_size:
            image = cv2.resize(image, self.image_size)
        if not self.rectified:
            return image
        return cv2.remap(image, *self._rectification_maps[side],
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def process_images(self, left, right):
        return self.process_image(left, "left"), self.process_image(right, "right")

    def rectify_keypoints(self, detections, side):
        """Map cached RAW 640x480 keypoints to the tracking image coordinates.

        New detections inferred on rectified images must NOT be passed here.
        No source dictionary/CSV is modified. Out-of-view or invalid points
        cannot support the rectified image observation and are omitted.
        """
        if side not in ("left", "right"):
            raise ValueError(f"Unknown camera side: {side}")
        if not self.rectified:
            return dict(detections)
        K, D, H, new_K = ((self.raw_K1, self.raw_dist1, self.left_rectification, self.K1)
                         if side == "left" else
                         (self.raw_K2, self.raw_dist2, self.right_rectification, self.K2))
        result = {}
        for name, value in detections.items():
            if not np.isfinite(value[:3]).all():
                continue
            xy = cv2.undistortPoints(np.asarray(value[:2], dtype=float).reshape(1, 1, 2),
                                     K, D, R=H[:3, :3], P=new_K).reshape(2)
            if np.isfinite(xy).all() and 0 <= xy[0] < self.image_size[0] and 0 <= xy[1] < self.image_size[1]:
                result[name] = (float(xy[0]), float(xy[1]), float(value[2]))
        return result

    def keypoint_frames_for_tracking(self, frames, side):
        if frames is None:
            return None
        return {index: self.rectify_keypoints(points, side) for index, points in frames.items()}

    def to_tracking_pose(self, raw_left_T_base):
        """Rotate hand-eye/pose from the original to the tracking left camera."""
        return self.left_rectification @ raw_left_T_base

    def to_raw_left_pose(self, tracking_left_T_base):
        """Keep external poses expressed in the original left optical frame."""
        return self.left_rectification.T @ tracking_left_T_base

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
