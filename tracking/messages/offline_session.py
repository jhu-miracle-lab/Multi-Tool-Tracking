"""Dataset loading and frame results shared by CLI and ROS offline tracking."""

import json
from pathlib import Path

import numpy as np

from tracking.pf.arm_tracker import ArmTracker
from tracking.pf.combined_tracker import CombinedArmTracker
from tracking.messages.arm_configs import build_arm_configs
from tracking.common.constants import DLC_W, DLC_H, CAL_W, CAL_H
from tracking.messages.dlc_csv import load_dlc_csv
from tracking.common.stereo import SimpleProjector
from tracking.features.feature_detection import ArmFeatureDetector
from tracking.messages.kinematics import FileJointSource
from tracking.messages.image_sources import DirectoryStereoImages
from tracking.common.frame import TrackingInput


class OfflineSession:
    def __init__(self, asset_root, data_dir, n_particles=500, handeye='', log=print, *, allow_primary_switch=False, rectify=False, shaft_backend="elsed",
                 feature_factory=ArmFeatureDetector, image_source=None,
                 keypoint_loader=load_dlc_csv, joint_source_factory=FileJointSource):
        root, data = Path(asset_root), Path(data_dir)
        self.log = log
        self.left_img_dir = str(data / 'regular/image/left_resized')
        right_img_dir = data / 'regular/image/right_resized'
        self.right_img_dir = str(right_img_dir)
        self.image_source = image_source or DirectoryStereoImages(self.left_img_dir, self.right_img_dir)
        candidate = data / 'pnp_handeye_manual.yaml'
        if not handeye and candidate.exists():
            handeye = str(candidate)
        robot_file = root / 'Parameter/LND.json'
        for parent in [data, *list(data.parents)[:3]]:
            if (parent / 'LND.json').exists():
                robot_file = parent / 'LND.json'
                break
        intrinsics = root / 'Parameter/stereo_calibration.yaml'
        projector = SimpleProjector(str(intrinsics), DLC_W, DLC_H, CAL_W, CAL_H, rectify=rectify, swap_cameras=True)
        self.projector = projector
        self.K = projector.K1
        self.dist = projector.dist1
        self.trackers = {}
        self.raw_keypoints = {}
        for name, cfg in build_arm_configs(str(root), str(data), handeye_file=handeye).items():
            csv_left = Path(self.left_img_dir) / cfg['csv_name']
            if not Path(cfg['kin_dir']).is_dir() or not csv_left.exists():
                log(f'[{name}] missing kinematics directory or keypoint CSV, skipping')
                continue
            left = keypoint_loader(str(csv_left), cfg['label_remap'])
            csv_right = right_img_dir / cfg['csv_name']
            right = keypoint_loader(str(csv_right), cfg['label_remap']) if csv_right.exists() else None
            if right is None:
                log(f'[{name}] right keypoint CSV missing: {csv_right}; right shaft only when images exist')
            self.raw_keypoints[name] = {'left': left, 'right': right}
            left = projector.keypoint_frames_for_tracking(left, 'left')
            right = projector.keypoint_frames_for_tracking(right, 'right')
            params = None
            params_file = data / f'shaft_params_{name}.json'
            if not params_file.exists():
                params_file = data / f'canny_params_{name}.json'
            if params_file.exists():
                params = json.loads(params_file.read_text())
                params.pop('arm', None)
            params = {**(params or {}), "backend": shaft_backend}
            self.trackers[name] = ArmTracker(
                name, cfg, left, right, self.K, self.dist, projector,
                str(robot_file), shaft_params=params, feature_factory=feature_factory,
                joint_source=joint_source_factory(cfg["kin_dir"]))
            log(f'[{name}] ready — {len(left)} frames')
        if not self.trackers:
            raise RuntimeError('No trackers initialised.')
        all_kin = {int(f.stem) for tr in self.trackers.values()
                   for f in Path(tr.kin_dir).glob('*.json') if f.stem.isdigit()}
        self.frame_indices = sorted(i for i in all_kin
                                    if (Path(self.left_img_dir) / f'{i}.png').exists())
        log(f'{len(self.frame_indices)} frames, arms={list(self.trackers)}')
        self.combined = CombinedArmTracker(
            self.trackers, n_particles, allow_primary_switch=allow_primary_switch)
        self.state_log = {name: {'frames': [], 'ema_states': [], 'fk_tips': []}
                          for name in self.trackers}

    def set_cached_keypoints(self, arm_name, side, frames, *, coordinate_space="raw"):
        """Install extra CSV predictions without mixing raw and rectified pixels."""
        if side not in ("left", "right") or coordinate_space not in ("raw", "tracking"):
            raise ValueError("Expected side left/right and coordinate_space raw/tracking")
        converted = (self.projector.keypoint_frames_for_tracking(frames, side)
                     if coordinate_space == "raw" else frames)
        arm = self.trackers[arm_name]
        self.raw_keypoints[arm_name][side] = frames if coordinate_space == "raw" else None
        if side == "left":
            arm.dlc_left = arm.feature_detector.left_source = converted
        else:
            arm.dlc_right = arm.feature_detector.right_source = converted
            arm.right_feature_detector.left_source = converted

    def read_images(self, frame_idx):
        """Read the same recorded frame ID in both eyes; never reuse an old right image.

        Missing right images still allow same-frame right CSV keypoints, but
        yield no right shaft observation. Recording frame IDs must be paired.
        """
        return self.projector.process_images(*self.image_source.read(frame_idx))

    def read_frame(self, frame_idx):
        left, right = self.read_images(frame_idx)
        per_arm = {}
        for name, arm in self.trackers.items():
            joints = arm.joint_source(frame_idx)
            if joints is not None:
                per_arm[name] = dict(joint_angles=joints)
        return TrackingInput(frame_idx, left, right, per_arm)

    def warmup(self):
        frames = self.frame_indices[:50]
        if frames:
            self.log(f'Warm-up: {len(frames)} frames...')
            per_arm = {name: [] for name in self.trackers}
            for index in frames:
                left, _ = self.read_images(index)
                for name, arm in self.trackers.items():
                    joints = arm.joint_source(index)
                    if joints is not None:
                        points = arm.feature_detector.get_keypoints(arm.dlc_left, index, left)
                        per_arm[name].append((left, joints, points))
            self.combined.warmup_live(per_arm, lkl_thresh=0.05)
            self.log('Warm-up done.')

    def draw_and_record(self, image, results, frame_idx, save_states=False):
        statuses = []
        for name, tr in self.trackers.items():
            frame = results[name]
            if not frame.available:
                continue
            tr.draw(image, frame.corrected_keypoints)
            state_label = tr.feature_detector.last_state_label
            label = f' [{state_label}]' if state_label else ''
            statuses.append(f'{name}: pts={frame.observation_count}{label}')
            if save_states:
                tip_base = tr.robot_arm.baseToJointT[-3][:3, 3]
                tip_cam = (self.projector.to_raw_left_pose(tr.cam_T_b_kin) @ np.append(tip_base, 1.0))[:3]
                record = self.state_log[name]
                record['frames'].append(frame_idx)
                # Preserve the historical NPZ key; values are PF means, not EMA.
                record['ema_states'].append(tr.pose_state.copy())
                record['fk_tips'].append(tip_cam.copy())
        return statuses

    def save_states(self, path):
        Path(path).absolute().parent.mkdir(parents=True, exist_ok=True)
        data = {f'{name}_{key}': np.array(values)
                for name, record in self.state_log.items() if record['frames']
                for key, values in record.items()}
        np.savez(path, **data)
        self.log(f'States saved: {path}')
