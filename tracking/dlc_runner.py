"""Real-time DLC inference runner and model configurations."""

import os

from deeplabcut.core.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis import get_pose_inference_runner

_DLC_ROOT_PSM1 = os.environ.get(
    "DLC_ROOT_PSM1",
    "/media/joec/data/Deeplabcut/Trained/Reaching-Mackenzie-2026-02-27",
)
_DLC_ROOT_PSM2 = os.environ.get(
    "DLC_ROOT_PSM2",
    "/media/joec/data/Deeplabcut/Reaching-Mackenzie-2026-04-01-tool2only",
)

ARM_MODEL_CONFIGS = {
    "psm1": {
        "pytorch_cfg": os.path.join(
            _DLC_ROOT_PSM1,
            "dlc-models-pytorch/iteration-0/ReachingFeb27-trainset80shuffle5/train/pytorch_config.yaml",
        ),
        "snapshot": os.path.join(
            _DLC_ROOT_PSM1,
            "dlc-models-pytorch/iteration-0/ReachingFeb27-trainset80shuffle5/train/snapshot-075.pt",
        ),
        "dlc_bodyparts": ["top1", "top4", "top5", "top6", "top7", "side2", "side3"],
        "dlc_bodypart_map": {
            "top1": "roll_1:front",
            "top4": "pitch_2:front",
            "top5": "pitch_3:front",
            "top6": "yaw_1",
            "top7": "yaw_2",
            "side2": "roll_2:right",
            "side3": "pitch_1:right",
        },
    },
    "psm2": {
        "pytorch_cfg": os.path.join(
            _DLC_ROOT_PSM2,
            "dlc-models-pytorch/iteration-0/ReachingApr1-trainset80shuffle1/train/pytorch_config.yaml",
        ),
        "snapshot": os.path.join(
            _DLC_ROOT_PSM2,
            "dlc-models-pytorch/iteration-0/ReachingApr1-trainset80shuffle1/train/snapshot-130.pt",
        ),
        "dlc_bodyparts": ["yaw_1", "yaw_2", "roll_1", "roll_2", "pitch_1", "pitch_2", "pitch_3"],
        "dlc_bodypart_map": {},
    },
}


class DLCRunner:
    """Wraps a DeepLabCut PoseInferenceRunner for single-frame inference."""

    def __init__(self, arm_name: str, label_remap: dict, warmup_frame=None):
        """
        Args:
            arm_name:     Key into ARM_MODEL_CONFIGS ('psm1' or 'psm2').
            label_remap:  Optional face-label remap applied after body-part
                          renaming, e.g. {'roll_1': 'roll_1:back'}.
            warmup_frame: BGR image used to pre-warm the GPU on construction.
        """
        mc = ARM_MODEL_CONFIGS[arm_name]
        print(f"[{arm_name}] loading DLC model: {os.path.basename(mc['snapshot'])}")
        model_cfg = read_config_as_dict(mc["pytorch_cfg"])
        self._runner = get_pose_inference_runner(
            model_cfg, mc["snapshot"], batch_size=1, device="cuda"
        )
        self._bodyparts = mc["dlc_bodyparts"]
        self._bp_map = mc["dlc_bodypart_map"]
        self._label_remap = label_remap

        if warmup_frame is not None:
            print(f"[{arm_name}] warming up GPU...")
            for _ in range(3):
                self._runner.inference([warmup_frame])

    def infer(self, frame) -> dict:
        """Run pose inference on a single BGR frame.

        Returns:
            {bodypart_name: (x, y, likelihood)} with model rename and
            face-label remap applied.
        """
        results = self._runner.inference([frame])
        poses = results[0]["bodyparts"][0]  # (n_bodyparts, 3)
        kps = {}
        for i, bp_raw in enumerate(self._bodyparts):
            x, y, lkl = float(poses[i, 0]), float(poses[i, 1]), float(poses[i, 2])
            bp = self._bp_map.get(bp_raw, bp_raw)
            bp = self._label_remap.get(bp, bp)
            kps[bp] = (x, y, lkl)
        return kps
