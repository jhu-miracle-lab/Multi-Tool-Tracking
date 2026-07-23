"""Face-label assignment: stereo triangulation and yaw matching.

Per-arm keypoint identity (tool1/tool2) is provided directly by the DLC/DEKR
model's own identity head, so no separate instance-to-arm assignment step
is needed here.

Instance-to-arm (tool1/psm1 vs tool2/psm2) identity used to be resolved
here too (assign_instances_to_arms(), FK-proximity-based) -- removed in
favor of tracking/dekr_live_runner.py's assign_tool_identities(), which
uses the DEKR model's own identity head instead. See that module's
docstring for the trade-off.
"""

import cv2
import numpy as np

from .constants import LIKELIHOOD_THRESHOLD, MIN_TRI_DIST_MM

# Keypoint bases grouped by the face axis they disambiguate.
LR_BASES = {"roll_2", "pitch_1"}
FB_BASES = {"roll_1", "pitch_2", "pitch_3"}


def instance_to_dets(instance: np.ndarray, bodyparts: list) -> dict:
    """Convert one raw multi-animal-model instance (J, 3) into the
    {bodypart: (x, y, likelihood)} dict format assign_labels() expects.

    Bridges DLCLive's per-frame output into the same shape the offline CSV
    loader (tracking/dlc_csv.py) already produces.
    """
    return {
        bp: (float(x), float(y), float(score)) for bp, (x, y, score) in zip(bodyparts, instance)
    }


def remap_labels(dets: dict, label_remap: dict) -> dict:
    """Apply a per-arm label_remap (e.g. {'roll_1': 'roll_1:back'}) to a
    plain-bodypart detections dict, canonicalizing face-less DEKR bodypart
    names the same way tracking/dlc_csv.py's load_dlc_csv() already does
    for offline DEKR CSV predictions.
    """
    return {label_remap.get(bp, bp): v for bp, v in dets.items()}


def compute_face_norms(robot_arm, cam_T_b):
    """Compute camera-frame Z components of the left and front face normals.

    Negative Z means the face points toward the camera (visible).
    Positive Z means the face points away (hidden).

    Returns:
        left_z:  Z of the left-face normal (link 4, axis [0, -1, 0]).
        front_z: Z of the front-face normal (link 5, axis [0, -1, 0]).
    """
    R_cam_4 = cam_T_b[:3, :3] @ robot_arm.baseToJointT[3][:3, :3]
    left_z = float((R_cam_4 @ np.array([0.0, -1.0, 0.0]))[2])

    R_cam_5 = cam_T_b[:3, :3] @ robot_arm.baseToJointT[4][:3, :3]
    front_z = float((R_cam_5 @ np.array([0.0, -1.0, 0.0]))[2])

    return left_z, front_z


def assign_labels(
    raw_l: dict,
    raw_r: dict,
    name2pt: dict,
    P1: np.ndarray,
    P2: np.ndarray,
    b_T_cam_kin: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    robot_arm,
    cam_T_b_kin: np.ndarray,
    curr_lr: str,
    curr_fb: str,
    cam_T_b: np.ndarray = None,
):
    """Assign FK-consistent labels to left-camera detections.

    Uses stereo triangulation (preferred) or face normals (fallback) to
    determine which face is visible, then resolves the yaw-tip ambiguity
    with a 2×2 optimal matching.

    Args:
        raw_l:       Left-camera detections {label: (x, y, likelihood)}.
        raw_r:       Right-camera detections (may be empty).
        name2pt:     FK point name → 3-D base-frame position.
        P1, P2:      Left and right projection matrices.
        b_T_cam_kin: Camera-to-base transform (from hand-eye, fixed).
        K, dist:     Left camera intrinsics for yaw projection.
        robot_arm:   RobotLink instance (used for face normal fallback).
        cam_T_b_kin: Camera extrinsic (fixed), for face normal computation.
        curr_lr:     Previous left/right face state ('left' or 'right').
        curr_fb:     Previous front/back face state ('front' or 'back').

    Returns:
        corrected: {fk_point_name: (x, y, likelihood)} remapped detections.
        curr_lr:   Updated left/right face state.
        curr_fb:   Updated front/back face state.
        state_label: Human-readable "FRONT|LEFT" style string.
    """
    BLIND_EPS = 0.1

    def _best_det(raw, base, faces):
        best = None
        for cand in [f"{base}:{f}" for f in faces] + [base]:
            if cand in raw and raw[cand][2] >= LIKELIHOOD_THRESHOLD:
                if best is None or raw[cand][2] > best[2]:
                    best = raw[cand]
        return best

    def _tri_vote(bases, face_pair):
        """Accumulate 3-D distances for each face over all bases.

        Returns (d_face0, d_face1, n_valid_points).
        """
        d0 = d1 = 0.0
        n = 0
        for base in bases:
            bl = _best_det(raw_l, base, face_pair)
            br = _best_det(raw_r, base, face_pair) if raw_r else None
            if bl is None or br is None:
                continue
            fk0 = name2pt.get(f"{base}:{face_pair[0]}")
            fk1 = name2pt.get(f"{base}:{face_pair[1]}")
            if fk0 is None or fk1 is None:
                continue
            X = cv2.triangulatePoints(
                P1,
                P2,
                np.array(bl[:2], dtype=np.float64).reshape(2, 1),
                np.array(br[:2], dtype=np.float64).reshape(2, 1),
            )
            p_cam = (X[:3] / X[3]).flatten()
            p_base = (b_T_cam_kin @ np.append(p_cam, 1.0))[:3]
            d0 += np.linalg.norm(p_base - fk0)
            d1 += np.linalg.norm(p_base - fk1)
            n += 1
        return d0, d1, n

    # Left / right face disambiguation
    d_left, d_right, n_lr = _tri_vote(LR_BASES, ["left", "right"])
    if n_lr > 0:
        if abs(d_left - d_right) / n_lr * 1000.0 >= MIN_TRI_DIST_MM:
            curr_lr = "left" if d_left < d_right else "right"
    else:
        left_z, _ = compute_face_norms(robot_arm, cam_T_b_kin)
        if abs(left_z) > BLIND_EPS:
            curr_lr = "left" if left_z < 0 else "right"

    # Front / back face disambiguation
    d_front, d_back, n_fb = _tri_vote(FB_BASES, ["front", "back"])
    if n_fb > 0:
        if abs(d_front - d_back) / n_fb * 1000.0 >= MIN_TRI_DIST_MM:
            curr_fb = "front" if d_front < d_back else "back"
    else:
        _, front_z = compute_face_norms(robot_arm, cam_T_b_kin)
        if abs(front_z) > BLIND_EPS:
            curr_fb = "front" if front_z < 0 else "back"

    state_label = f"{curr_fb.upper()}|{curr_lr.upper()}"

    # Yaw-tip 2×2 optimal matching
    # Use PF-corrected pose when available; falls back to fixed FK hand-eye.
    # old: always used cam_T_b_kin (FK-only), wrong when hand-eye has systematic offset.
    _yaw_pose = cam_T_b if cam_T_b is not None else cam_T_b_kin
    rvec_kin, _ = cv2.Rodrigues(_yaw_pose[:3, :3])
    tvec_kin = _yaw_pose[:3, 3].reshape(3, 1)

    def _proj2d(pt3d):
        p, _ = cv2.projectPoints(pt3d.reshape(1, 3).astype(np.float64), rvec_kin, tvec_kin, K, dist)
        return p.reshape(2)

    det1 = raw_l.get("yaw_1")
    det2 = raw_l.get("yaw_2")
    if (
        det1 is not None
        and det1[2] >= LIKELIHOOD_THRESHOLD
        and det2 is not None
        and det2[2] >= LIKELIHOOD_THRESHOLD
        and "yaw_1" in name2pt
        and "yaw_2" in name2pt
    ):
        fk1 = _proj2d(name2pt["yaw_1"])
        fk2 = _proj2d(name2pt["yaw_2"])
        d1 = np.array(det1[:2])
        d2 = np.array(det2[:2])
        straight = np.hypot(*(d1 - fk1)) + np.hypot(*(d2 - fk2))
        swapped = np.hypot(*(d1 - fk2)) + np.hypot(*(d2 - fk1))
        yaw_map = (
            {"yaw_1": "yaw_1", "yaw_2": "yaw_2"}
            if straight <= swapped
            else {"yaw_1": "yaw_2", "yaw_2": "yaw_1"}
        )
    else:
        yaw_map = {"yaw_1": "yaw_1", "yaw_2": "yaw_2"}

    # Build full label → FK-name mapping
    label_to_fk = dict(yaw_map)
    for base in LR_BASES:
        for face in ("left", "right"):
            label_to_fk[f"{base}:{face}"] = f"{base}:{curr_lr}"
    for base in FB_BASES:
        for face in ("front", "back"):
            label_to_fk[f"{base}:{face}"] = f"{base}:{curr_fb}"

    corrected = {}
    for bp, fk_bp in label_to_fk.items():
        if bp in raw_l and raw_l[bp][2] >= 0.1:
            lkl = raw_l[bp][2]
            if fk_bp not in corrected or lkl > corrected[fk_bp][2]:
                corrected[fk_bp] = raw_l[bp]

    return corrected, curr_lr, curr_fb, state_label
