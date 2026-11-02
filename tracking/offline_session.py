"""Dataset loading and frame results shared by CLI and ROS offline tracking."""

import json
from pathlib import Path

import numpy as np

from .arm_tracker import ArmTracker
from .combined_tracker import CombinedArmTracker
from .arm_configs import build_arm_configs
from .constants import DLC_W, DLC_H, CAL_W, CAL_H
from .dlc_csv import load_dlc_csv
from .stereo import SimpleProjector


class OfflineSession:
    def __init__(self, asset_root, data_dir, n_particles=500, handeye='', log=print, *, allow_primary_switch=False):
        root, data = Path(asset_root), Path(data_dir)
        self.log = log
        self.left_img_dir = str(data / 'regular/image/left_resized')
        right_img_dir = data / 'regular/image/right_resized'
        candidate = data / 'pnp_handeye_manual.yaml'
        if not handeye and candidate.exists():
            handeye = str(candidate)
        robot_file = root / 'Parameter/LND.json'
        for parent in [data, *list(data.parents)[:3]]:
            if (parent / 'LND.json').exists():
                robot_file = parent / 'LND.json'
                break
        intrinsics = root / 'Parameter/intrinsics.npz'
        projector = SimpleProjector(str(intrinsics), DLC_W, DLC_H, CAL_W, CAL_H)
        self.K = projector.K1
        with np.load(intrinsics) as calibration:
            self.dist = calibration['dist_left'].flatten().astype(np.float64)
        self.trackers = {}
        for name, cfg in build_arm_configs(str(root), str(data), handeye_file=handeye).items():
            csv_left = Path(self.left_img_dir) / cfg['csv_name']
            if not Path(cfg['kin_dir']).is_dir() or not csv_left.exists():
                log(f'[{name}] missing kinematics directory or keypoint CSV, skipping')
                continue
            left = load_dlc_csv(str(csv_left), cfg['label_remap'])
            csv_right = right_img_dir / cfg['csv_name']
            right = load_dlc_csv(str(csv_right), cfg['label_remap']) if csv_right.exists() else None
            params = None
            params_file = data / f'canny_params_{name}.json'
            if params_file.exists():
                params = json.loads(params_file.read_text())
                params.pop('arm', None)
            self.trackers[name] = ArmTracker(
                name, cfg, left, right, self.K, self.dist, projector,
                str(robot_file), shaft_params=params)
            log(f'[{name}] ready — {len(left)} frames')
        if not self.trackers:
            raise RuntimeError('No trackers initialised.')
        all_kin = {int(f.stem) for tr in self.trackers.values()
                   for f in Path(tr.kin_dir).glob('*.json')}
        self.frame_indices = sorted(i for i in all_kin
                                    if (Path(self.left_img_dir) / f'{i}.png').exists())
        log(f'{len(self.frame_indices)} frames, arms={list(self.trackers)}')
        self.combined = CombinedArmTracker(
            self.trackers, n_particles, allow_primary_switch=allow_primary_switch)
        self.state_log = {name: {'frames': [], 'ema_states': [], 'fk_tips': []}
                          for name in self.trackers}

    def warmup(self):
        frames = self.frame_indices[:50]
        if frames:
            self.log(f'Warm-up: {len(frames)} frames...')
            self.combined.warmup(frames, lkl_thresh=0.05, left_img_dir=self.left_img_dir)
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
                tip_cam = (tr.cam_T_b_kin @ np.append(tip_base, 1.0))[:3]
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
