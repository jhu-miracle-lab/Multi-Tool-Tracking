"""Real-time DEKR multi-animal DLC-Live inference runner.

Ported from dlc_live_rt.py (Multi_tool_tracking) -- a standalone stereo
capture + inference prototype. This module keeps the model-loading and
inference half of that script (RT-1/RT-3/RT-4/RT-8/RT-9 in its comments).

Identity (tool1/psm1 vs tool2/psm2) is resolved here, via the model's own
identity head (assign_tool_identities()) -- infer() bypasses DLCLive's
public get_pose()/init_inference() API entirely and reaches into the
PyTorch runner directly, because that public API only ever exposes the
bodypart-pose head, never the identity head. This is the ONLY identity
mechanism in the live pipeline: there is no FK-based fallback
(tracking/label_assignment.py's now-removed assign_instances_to_arms()
used to fill that role; dropped in favor of the model's own identity head
once its identity loss converged -- see _DEFAULT_MODEL_PATH's comment).
Consequence: with use_identity_head=False, or if the model's identity
head is unreliable for a given frame, there is currently no other signal
to fall back on.

Stereo-only: infer() requires both eyes and always runs them as one
batched forward pass -- there is no single-eye inference path in this
runner. Callers (ros2_node/tracking_node.py) must wait until both camera
views are available before calling it at all, including during warm-up.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import dlclive.pose_estimation_pytorch.models.predictors.identity_predictor  # noqa: F401 -- triggers registration
from dlclive import DLCLive

from .constants import DEKR_PCUTOFF

# Free perf flags -- must be set before the model runs. benchmark autotunes
# conv kernels per input shape; callers always hand in a fixed DLC_W x DLC_H
# image (see tracking/constants.py), which is the ideal case for this.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Default snapshot path from dlc_live_rt.py's current version -- identity
# head RESTORED (not the older _noident.pt export from model_strip.py;
# identity loss converged during training, eval.identity_heatmap ~0.0045).
# Override via the DEKR_LIVE_MODEL_PATH env var or the model_path
# constructor arg, same pattern as tracking/dlc_runner.py's DLC_ROOT_PSM1/PSM2.
_DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "DualTool-DEKR-Mackenzie-2026-05-27"
    / "exported-models-pytorch"
    / "DLC_DualTool-DEKR_dekr_w32_iteration-6_shuffle-10"
    / "DLC_DualTool-DEKR_dekr_w32_iteration-6_shuffle-10_snapshot-best-140_ident.pt"
)


def _resolve_model_path(model_path: str = None) -> str:
    if model_path:
        return model_path
    env = os.environ.get("DEKR_LIVE_MODEL_PATH")
    if env:
        return env
    return str(_DEFAULT_MODEL_PATH)


def try_compile_dlc_model(dlc, verbose: bool = True) -> bool:
    """Best-effort torch.compile of DLCLive's underlying nn.Module.

    dlclive's internal attribute layout varies between versions, so search
    the DLCLive object (two levels deep) for an nn.Module and swap it for a
    compiled version in place. Returns True on success.
    """
    import torch.nn as nn

    def search(parent, depth):
        for name, val in list(vars(parent).items()):
            if isinstance(val, nn.Module):
                return parent, name, val
            if depth > 0 and hasattr(val, "__dict__") and not isinstance(val, type):
                found = search(val, depth - 1)
                if found:
                    return found
        return None

    try:
        found = search(dlc, depth=2)
        if not found:
            if verbose:
                print("[DekrLiveRunner] torch.compile: no nn.Module found — skipping", flush=True)
            return False
        parent, name, module = found
        compiled = torch.compile(module, mode="default")
        setattr(parent, name, compiled)
        if verbose:
            print(
                f"[DekrLiveRunner] torch.compile: compiled '{type(parent).__name__}.{name}' "
                "(first inference will take ~30-60 s to compile)",
                flush=True,
            )
        return True
    except Exception as e:  # noqa: BLE001 -- best-effort, never fatal
        if verbose:
            print(
                f"[DekrLiveRunner] torch.compile failed ({e}) — continuing uncompiled", flush=True
            )
        return False


def assign_tool_identities(
    pose: np.ndarray, identity_heatmap: np.ndarray, pcutoff: float
) -> np.ndarray:
    """Reorders one image's raw per-frame detections onto fixed identity
    slots (index 0 = tool1/psm1, index 1 = tool2/psm2, matching this
    project's tool<->psm index convention) using the model's identity head.

    pose: (num_detections, num_keypoints, 3) in the padded input-image pixel
    space DEKRPredictor rescales to. identity_heatmap: (H, W, num_individuals)
    in that same pixel space.

    Score for (detection, individual) = mean of the identity heatmap over
    that detection's visible (score >= pcutoff) keypoints; assignment is the
    Hungarian max-score matching (scipy linear_sum_assignment on -scores) so
    two detections can't collide on one identity. Slots with no assigned
    detection (fewer detections than individuals) come back as all-zero rows
    -- same convention as "not detected" elsewhere in this pipeline.
    """
    num_det = pose.shape[0]
    num_kp = pose.shape[1]
    num_ident = identity_heatmap.shape[-1]
    h, w = identity_heatmap.shape[:2]

    out = np.zeros((num_ident, num_kp, 3), dtype=pose.dtype)
    if num_det == 0:
        return out

    scores = np.zeros((num_det, num_ident), dtype=np.float32)
    for d in range(num_det):
        kp = pose[d]
        visible = kp[:, 2] >= pcutoff
        if not visible.any():
            continue
        xs = np.clip(np.round(kp[visible, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(kp[visible, 1]).astype(int), 0, h - 1)
        scores[d] = identity_heatmap[ys, xs].mean(axis=0)

    det_idx, ident_idx = linear_sum_assignment(-scores)
    for d, i in zip(det_idx, ident_idx):
        out[i] = pose[d]
    return out


class DekrLiveRunner:
    """Wraps a DLCLive DEKR multi-animal model for live stereo inference.

    infer(bgr_left, bgr_right) reaches directly into the PyTorch runner
    (dlc.runner), batching both eyes into ONE forward pass -- bypasses
    DLCLive's public get_pose()/init_inference() API (single-image only,
    and never exposes the identity head). When use_identity_head is True
    (default), each eye's returned instances are reordered onto fixed
    [tool1/psm1, tool2/psm2] slots via assign_tool_identities(); always
    exactly 2 entries per eye then, some possibly all-zero ("not
    detected"). When False, order/count is whatever the model produced for
    that frame (not stable across frames, and this runner then has no
    opinion on identity at all -- see module docstring). Bodypart order
    matches tracking/constants.py's DEKR_BODYPARTS.

    Callers must already hand in DLC_W x DLC_H (640x480) images -- this
    runner does not resize; resizing happens once, up front, wherever the
    frame is captured (see ros2_node/dvrk_stereo_publisher.py).
    """

    def __init__(
        self,
        model_path: str = None,
        use_fp16: bool = True,
        use_compile: bool = True,
        use_identity_head: bool = True,
        identity_pcutoff: float = DEKR_PCUTOFF,
    ):
        self.model_path = _resolve_model_path(model_path)
        print(f"[DekrLiveRunner] Model: {self.model_path}", flush=True)
        self.dlc = DLCLive(
            self.model_path,
            model_type="pytorch",
            single_animal=False,
            device="cuda",
        )
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        self.use_identity_head = use_identity_head
        self.identity_pcutoff = identity_pcutoff
        self._model_loaded = False
        self._compiled = False

    # ------------------------------------------------------------------
    # Shared model-load / torch.compile bookkeeping
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._model_loaded:
            self.dlc.runner.load_model()
            self._model_loaded = True

    def _maybe_compile(self, forward_fn):
        """Compile once (best-effort) and immediately re-run forward_fn to
        force the lazy compilation to happen on THIS call instead of
        stalling the next one. Returns forward_fn()'s result if compile
        just happened, else None (nothing to reassign)."""
        if self.use_compile and not self._compiled:
            try_compile_dlc_model(self.dlc)
            self._compiled = True
            return forward_fn()
        return None

    # ------------------------------------------------------------------
    # Direct-model forward pass (always both eyes, batch=2)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _forward(self, bgr_left: np.ndarray, bgr_right: np.ndarray):
        """Runs the model on both already-DLC_W-x-DLC_H BGR eyes in ONE
        batch=2 forward pass. Bypasses DLCLive's public get_pose()/
        init_inference() API (which only ever extracts the bodypart-pose
        head, and is single-image only) and reaches into dlc.runner
        directly, replicating what DLCLive.process_frame()/
        PyTorchRunner.get_pose() do for our config (no detector, no dynamic
        cropping, convert2rgb=True, resize=None) -- see dlc_live_rt.py's
        infer_stereo_batch() for the original derivation of this.

        Returns (poses_np, ident_np_or_None): poses_np is
        (2, num_det, J, 3); ident_np is (2, H, W, num_individuals) when
        use_identity_head, else None (skips extracting/transferring it
        from the model output).
        """
        self._ensure_loaded()

        rgb_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in (bgr_left, bgr_right)]
        batch = torch.from_numpy(np.stack(rgb_imgs)).permute(0, 3, 1, 2)
        batch = self.dlc.runner.pose_transform(batch).to(self.dlc.runner.device)

        def forward():
            outputs = self.dlc.runner.model(batch)
            predictions = self.dlc.runner.model.get_predictions(outputs)
            poses = predictions["bodypart"]["poses"]
            ident = predictions["identity"]["heatmap"] if self.use_identity_head else None
            return poses, ident

        def forward_fp16():
            if self.use_fp16:
                try:
                    with torch.autocast("cuda", dtype=torch.float16):
                        return forward()
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[DekrLiveRunner] FP16 autocast failed ({e}) — falling back to FP32",
                        flush=True,
                    )
                    self.use_fp16 = False
            return forward()

        poses, ident = forward_fp16()
        recompiled = self._maybe_compile(forward_fp16)
        if recompiled is not None:
            poses, ident = recompiled

        poses_np = poses.cpu().numpy()
        ident_np = ident.cpu().numpy() if ident is not None else None
        return poses_np, ident_np

    def _resolve_identity(self, pose: np.ndarray, ident_heatmap) -> np.ndarray:
        if self.use_identity_head:
            return assign_tool_identities(pose, ident_heatmap, self.identity_pcutoff)
        return pose

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def infer(self, bgr_left: np.ndarray, bgr_right: np.ndarray):
        """Run inference on both eyes in ONE batch=2 forward pass.

        Returns (instances_left, instances_right), each a list of (J, 3)
        arrays -- see class docstring for the use_identity_head ordering/
        count contract. No single-eye fallback: a failure here propagates
        to the caller (ros2_node/tracking_node.py just skips that frame).
        """
        poses_np, ident_np = self._forward(bgr_left, bgr_right)
        pose_l = self._resolve_identity(poses_np[0], ident_np[0] if ident_np is not None else None)
        pose_r = self._resolve_identity(poses_np[1], ident_np[1] if ident_np is not None else None)
        return (
            [pose_l[i] for i in range(len(pose_l))],
            [pose_r[i] for i in range(len(pose_r))],
        )

    def close(self):
        self.dlc.close()
