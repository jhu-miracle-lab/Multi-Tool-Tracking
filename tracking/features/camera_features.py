"""Camera-specific geometry and feature extraction; no PF dependency."""
import cv2
import numpy as np


def extract_camera_features(detector, frame_idx, image, raw, other_raw,
                  kinematic_pose, pose, K, dist, right_img=None, stereo_vote=False, *,
                            name2pt, robot_arm, shaft_dir_b, projector, P1, P2, prior_roi):
    """Prepare camera geometry; extraction owns separate per-view histories."""
    fallback = {}
    if prior_roi:
        rvec = cv2.Rodrigues(pose[:3, :3])[0]
        visible_points = [(name, point) for name, point in name2pt.items()
                          if not (pose[:3, :3] @ point + pose[:3, 3])[2] <= 0]
        if visible_points:
            names, points = zip(*visible_points)
            projected, _ = cv2.projectPoints(
                np.asarray(points, dtype=np.float64), rvec, pose[:3, 3], K, dist)
            for name, (x, y) in zip(names, projected.reshape(-1, 2)):
                if np.isfinite(x) and np.isfinite(y):
                    fallback[name] = (float(x), float(y), 0.01)
    return detector.detect(
        frame_idx, image, right_img, raw, other_raw,
        name2pt=name2pt, P1=P1, P2=P2,
        b_T_cam_kin=np.linalg.inv(kinematic_pose), K=K, dist=dist,
        robot_arm=robot_arm, cam_T_b_kin=kinematic_pose, cam_T_b=pose,
        shaft_dir_b=shaft_dir_b, fallback_keypoints=fallback,
        right_K=projector.K2 if stereo_vote else None,
        right_dist=projector.dist2 if stereo_vote else None)

