"""Loading/computing the fixed PSM1-base-to-PSM2-base transform (b1_T_b2)
used by tracking/joint_tracker.py's shared delta_T filter.

Both PSM bases are rigidly mounted (stationary relative to each other), so
b1_T_b2 is a one-time-computed constant, not something re-estimated per
dataset/session.
"""

import argparse
import json

import cv2
import numpy as np
import yaml


def _pose_to_T(entry) -> np.ndarray:
    """Convert one pose entry to a (4,4) homogeneous transform.

    Accepts either:
      - a flat/nested 4x4 matrix (list of 4 lists of 4 floats), or
      - a {"position": ..., "orientation": ...} dict (the common JSON dump
        shape for a ROS geometry_msgs Pose/PoseStamped, e.g. from a dVRK
        /SUJ/<arm>/local/measured_cp echo, or this dataset's
        regular/kinematic/base_frame.json). "position"/"orientation" may
        each be either a {x,y,z[,w]} dict or a flat [x,y,z[,w]] list.
        Quaternion order is ROS convention (x, y, z, w).
    """
    if isinstance(entry, dict) and "position" in entry and "orientation" in entry:
        p = entry["position"]
        q = entry["orientation"]
        T = np.eye(4)
        if isinstance(q, dict):
            qx, qy, qz, qw = q["x"], q["y"], q["z"], q["w"]
        else:
            qx, qy, qz, qw = q  # ROS geometry_msgs/Quaternion field order: x, y, z, w
        T[:3, :3] = _quat_xyzw_to_matrix(qx, qy, qz, qw)
        T[:3, 3] = [p["x"], p["y"], p["z"]] if isinstance(p, dict) else list(p)
        return T
    arr = np.asarray(entry, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix or position/orientation dict, got shape {arr.shape}")
    return arr


def _quat_xyzw_to_matrix(x, y, z, w) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat([x, y, z, w]).as_matrix()


def compute_b1_t_b2_from_matrices(ecm_T_psm1, ecm_T_psm2) -> np.ndarray:
    """b1_T_b2 = inv(ecm_T_psm1) @ ecm_T_psm2.

    Both inputs are each PSM base's pose expressed in a common reference
    frame (e.g. the ECM/cart frame from dVRK SUJ kinematics). Since both
    bases are stationary, the common frame cancels out and the result is a
    fixed constant independent of the ECM's live joint state.
    """
    ecm_T_psm1 = _pose_to_T(ecm_T_psm1)
    ecm_T_psm2 = _pose_to_T(ecm_T_psm2)
    return np.linalg.inv(ecm_T_psm1) @ ecm_T_psm2


def compute_b1_t_b2_from_suj(suj_json_path: str) -> np.ndarray:
    """Compute b1_T_b2 from a dVRK SUJ JSON dump.

    Expects a JSON file with top-level "ECM_T_PSM1" / "ECM_T_PSM2" (or
    "PSM1" / "PSM2") entries, each either a 4x4 matrix or a
    {"position": ..., "orientation": ...} pose dict (see _pose_to_T).
    Adjust the key names below if your SUJ dump uses different ones.
    """
    with open(suj_json_path) as f:
        data = json.load(f)

    def _get(*keys):
        for k in keys:
            if k in data:
                return data[k]
        raise KeyError(
            f"None of {keys} found in {suj_json_path}. "
            "Rename your JSON's keys, or call compute_b1_t_b2_from_matrices() directly."
        )

    ecm_T_psm1 = _get("ECM_T_PSM1", "PSM1")
    ecm_T_psm2 = _get("ECM_T_PSM2", "PSM2")
    return compute_b1_t_b2_from_matrices(ecm_T_psm1, ecm_T_psm2)


def load_b1_T_b2(path: str) -> np.ndarray:
    """Load a fixed b1_T_b2 from a YAML file.

    Same convention as the per-arm hand-eye YAMLs (tracking/arm_tracker.py):
    translation in mm under "b1_T_b2_tvec", rotation as a Rodrigues vector
    under "b1_T_b2_rvec".
    """
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    T = np.eye(4)
    T[:3, 3] = np.array(data["b1_T_b2_tvec"], dtype=np.float64) / 1000.0
    rvec = np.array(data["b1_T_b2_rvec"], dtype=np.float64)
    T[:3, :3], _ = cv2.Rodrigues(rvec)
    return T


def save_b1_T_b2(T: np.ndarray, path: str):
    """Save a (4,4) transform to the YAML convention load_b1_T_b2() expects."""
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    data = {
        "b1_T_b2_tvec": (T[:3, 3] * 1000.0).tolist(),
        "b1_T_b2_rvec": rvec.flatten().tolist(),
    }
    with open(path, "w") as f:
        yaml.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute b1_T_b2 (PSM1-base-to-PSM2-base) from a dVRK SUJ JSON dump."
    )
    parser.add_argument("--suj", required=True, help="Path to the SUJ JSON dump")
    parser.add_argument("--out", required=True, help="Output YAML path")
    args = parser.parse_args()

    T = compute_b1_t_b2_from_suj(args.suj)
    save_b1_T_b2(T, args.out)

    t_mm = T[:3, 3] * 1000
    r_deg = np.rad2deg(cv2.Rodrigues(T[:3, :3])[0].flatten())
    print(f"b1_T_b2 written to {args.out}")
    print(f"  t = [{t_mm[0]:.1f}, {t_mm[1]:.1f}, {t_mm[2]:.1f}] mm")
    print(f"  r = [{r_deg[0]:.1f}, {r_deg[1]:.1f}, {r_deg[2]:.1f}] deg")


if __name__ == "__main__":
    main()
