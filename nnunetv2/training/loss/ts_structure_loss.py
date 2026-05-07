"""
TotalSegmentator-based structure + feature loss for CBCT→CT regression.

Idea
----
On each step:
  1. Denormalize the predicted sCT (training-space → HU), apply TS's CT
     normalization, and forward through a frozen TS encoder+decoder.
  2. The forward pass is hooked on selected encoder stages, so the hooks
     capture sCT features for free.
  3. Compare the TS logits against cached CT argmax labels using DC + CE
     (the same loss TS was trained with), and compare the hooked features
     against cached CT features using L1.

The CT-side targets (labels + features) are produced offline by
``scripts/cache_ts_targets.py`` at the preprocessed training spacing so that
patch-level cropping is voxel-aligned.

This module is deliberately agnostic to where the targets come from: it
receives them already cropped to match the current patch. The trainer is
responsible for the bbox crop.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import load_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _build_coarse_lut(ts_dataset_json_path: str) -> np.ndarray:
    """Same mapping as scripts/cache_ts_targets.py: {bg, lung, bone, muscle, other}."""
    labels = load_json(ts_dataset_json_path)["labels"]
    name_to_id = {n: int(i) for n, i in labels.items()}
    n = max(name_to_id.values()) + 1
    lut = np.full(n, 4, dtype=np.int64)
    lut[0] = 0
    def assign(names, cid):
        for nm in names:
            if nm in name_to_id:
                lut[name_to_id[nm]] = cid
    assign([f"lung_{x}" for x in ("upper_lobe_left","lower_lobe_left","upper_lobe_right","middle_lobe_right","lower_lobe_right")] + ["trachea"], 1)
    bones = (["sacrum","skull","sternum","costal_cartilages"]
             + [f"vertebrae_{v}" for v in ("S1","L1","L2","L3","L4","L5","T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","C1","C2","C3","C4","C5","C6","C7")]
             + [f"rib_{s}_{i}" for s in ("left","right") for i in range(1,13)]
             + [f"{b}_{s}" for b in ("humerus","scapula","clavicula","femur","hip") for s in ("left","right")])
    assign(bones, 2)
    assign([f"gluteus_{m}_{s}" for m in ("maximus","medius","minimus") for s in ("left","right")]
           + [f"autochthon_{s}" for s in ("left","right")]
           + [f"iliopsoas_{s}" for s in ("left","right")], 3)
    return lut


NUM_COARSE_CLASSES = 5


def _resolve_ts_trainer_folder(ts_results_dir: str, ts_config: str) -> str:
    from pathlib import Path
    base = Path(ts_results_dir)
    matches = [p for p in base.iterdir() if p.is_dir() and p.name.endswith(f"__{ts_config}")]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one trainer subfolder ending with '__{ts_config}' "
            f"in {ts_results_dir}, found: {[m.name for m in matches]}"
        )
    return str(matches[0])


def load_ts_network(
    ts_results_dir: str, ts_config: str, ts_fold: int, device: torch.device
) -> nn.Module:
    """Load the TS nnUNet model as a frozen eval-mode torch Module."""
    predictor = nnUNetPredictor(device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        _resolve_ts_trainer_folder(ts_results_dir, ts_config),
        use_folds=(ts_fold,),
        checkpoint_name="checkpoint_final.pth",
    )
    net = predictor.network.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    # Disable deep supervision so forward returns a single tensor
    if hasattr(net, "decoder") and hasattr(net.decoder, "deep_supervision"):
        net.decoder.deep_supervision = False
    return net


def pick_encoder_stages(network: nn.Module, names: List[str]) -> Dict[str, nn.Module]:
    stages = network.encoder.stages
    n = len(stages)
    picks: Dict[str, nn.Module] = {}
    for name in names:
        if name == "bottleneck":
            picks[name] = stages[-1]
        elif name == "mid":
            picks[name] = stages[n // 2]
        elif name.startswith("stage"):
            picks[name] = stages[int(name[len("stage"):])]
        else:
            raise ValueError(f"Unknown feature stage '{name}'.")
    return picks


class TSStructureLoss(nn.Module):
    """
    DC+CE on TS(sCT) vs cached TS(CT) labels + L1 on hooked encoder features.

    Args:
        ts_results_dir / ts_config / ts_fold: locate the frozen TS checkpoint.
        feature_stages: encoder stage names to hook ("bottleneck", "mid",
            or "stageN").
        target_mean / target_std: CT-channel normalization stats from nnUNet
            plans. Used to undo training-space normalization before applying
            TS normalization.
        w_seg / w_feat: weights for the two terms.
        num_classes: override the TS output class count (inferred if None).
        ts_clip: HU clip range applied before TS z-score.
        ts_mean / ts_std: TS CT normalization stats (conservative defaults).
    """

    def __init__(
        self,
        ts_results_dir: str,
        ts_config: str = "3d_fullres",
        ts_fold: int = 0,
        feature_stages: Optional[List[str]] = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        w_seg: float = 1.0,
        w_feat: float = 0.1,
        ts_clip: tuple = (-1024.0, 1024.0),
        ts_mean: float = 100.0,
        ts_std: float = 200.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.target_mean = target_mean
        self.target_std = target_std
        self.w_seg = w_seg
        self.w_feat = w_feat
        self.ts_clip = ts_clip
        self.ts_mean = ts_mean
        self.ts_std = ts_std

        if feature_stages is None:
            feature_stages = ["bottleneck", "mid"]
        self.feature_stages = feature_stages

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net = load_ts_network(ts_results_dir, ts_config, ts_fold, device)
        self._stages = pick_encoder_stages(self._net, feature_stages)

        ts_trainer_folder = _resolve_ts_trainer_folder(ts_results_dir, ts_config)
        from pathlib import Path as _P
        lut_np = _build_coarse_lut(str(_P(ts_trainer_folder) / "dataset.json"))
        self.num_coarse = NUM_COARSE_CLASSES
        # Aggregation matrix [C_ts, num_coarse]: one-hot over coarse groups.
        agg = np.zeros((len(lut_np), NUM_COARSE_CLASSES), dtype=np.float32)
        for c, g in enumerate(lut_np):
            agg[c, int(g)] = 1.0
        self.register_buffer("_coarse_agg", torch.from_numpy(agg).to(device))

        # Register hooks that cache the most recent features during forward
        self._features: Dict[str, torch.Tensor] = {}
        for name, module in self._stages.items():
            module.register_forward_hook(self._make_hook(name))

        self._dice_smooth = 1e-5

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._features[name] = out
        return hook

    def _to_ts_input(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Training-normalized sCT → HU → TS-normalized input."""
        hu = pred_norm.float() * self.target_std + self.target_mean
        hu = hu.clamp(*self.ts_clip)
        return (hu - self.ts_mean) / self.ts_std

    def forward(
        self,
        pred_norm: torch.Tensor,
        cached_labels: torch.Tensor,
        cached_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_norm: sCT prediction in training-normalized space,
                shape [B, 1, D, H, W].
            cached_labels: TS argmax labels for the CT patch,
                shape [B, 1, D, H, W] (int64 for CE).
            cached_features: dict name → [B, C_i, d_i, h_i, w_i] (float).

        Returns:
            dict with scalar tensors: 'loss', 'seg_loss', 'feat_loss'.
        """
        self._features.clear()
        ts_in = self._to_ts_input(pred_norm)

        # Run TS in fp16 autocast for speed (frozen weights; softmax+aggregation cast back to fp32).
        with torch.amp.autocast(device_type=ts_in.device.type, enabled=ts_in.device.type == "cuda",
                                dtype=torch.float16):
            logits = self._net(ts_in)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

        probs = torch.softmax(logits.float(), dim=1)  # [B, C_ts, D, H, W]
        # Aggregate TS classes → coarse tissues via einsum (single kernel).
        coarse_probs = torch.einsum("bcdhw,cn->bndhw", probs, self._coarse_agg)

        target = cached_labels.long()
        if target.ndim == 5:
            target = target.squeeze(1)

        log_probs = torch.log(coarse_probs.clamp_min(1e-7))
        ce_loss = F.nll_loss(log_probs, target)

        target_1h = F.one_hot(target, self.num_coarse).permute(0, 4, 1, 2, 3).float()
        dims = (0, 2, 3, 4)
        inter = (coarse_probs * target_1h).sum(dim=dims)
        denom = coarse_probs.sum(dim=dims) + target_1h.sum(dim=dims)
        dice = (2 * inter + self._dice_smooth) / (denom + self._dice_smooth)
        dice_loss = 1.0 - dice[1:].mean()

        seg_loss = ce_loss + dice_loss

        feat_losses = []
        for name in self.feature_stages:
            pred_feat = self._features[name].float()
            tgt_feat = cached_features[name].to(pred_feat.dtype).to(pred_feat.device)
            if pred_feat.shape != tgt_feat.shape:
                raise ValueError(
                    f"TS feature shape mismatch at stage '{name}': "
                    f"pred {tuple(pred_feat.shape)} vs cached {tuple(tgt_feat.shape)}"
                )
            feat_losses.append(F.l1_loss(pred_feat, tgt_feat))
        feat_loss = torch.stack(feat_losses).mean() if feat_losses else torch.zeros((), device=ts_in.device)

        total = self.w_seg * seg_loss + self.w_feat * feat_loss
        return {"loss": total, "seg_loss": seg_loss.detach(), "feat_loss": feat_loss.detach()}
