"""
Regression trainers with an input/prediction alignment regularizer.

loss = MAE(prediction, target) with deep supervision + ALIGN_WEIGHT * A(prediction, input)

A compares the full-resolution prediction with the network input (channel 0), which is in the correct
geometry even when the target (channel 1) is misaligned. All variants share one training step, so the
L1 baseline (`_align_none`) differs from the others only by the missing term.

Environment overrides:
    ALIGN_WEIGHT        weight of the alignment term (default per class)
    ALIGN_NMI_BINS      histogram bins for NMI (default 32)
    ALIGN_NMI_SAMPLES   voxels sampled per patch for NMI (default 50000)
    ALIGN_NUM_EPOCHS    number of epochs (default: the base trainer's 1000), e.g. for the λ search
    ALIGN_SIGLIP_CKPT   SigLIP checkpoint (simcbct-siglip best.pt) for the `_siglip` trainer; simcbct_siglip must
                        be on PYTHONPATH and <preprocessed dataset>/siglip_stats.json must exist
                        (controlled-deformation-benchmark scripts/siglip_norm_stats.py)
    ALIGN_SIGLIP_MODE   "windows" (default: patch covered by overlapping SigLIP training-size windows), "full"
                        (whole patch in one pass) or "crop" (one random training-size window)
    ALIGN_SIGLIP_MAX_WINDOWS  "windows" mode: random subset of this many windows per step (default 0 = all)
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import autocast

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.training.loss.alignment_losses import MINDSSCLoss, NMILoss, SigLIPFeatureLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_mae_deep import (
    nnUNetTrainerRegression_mae_deep,
)
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerRegression_align_none(nnUNetTrainerRegression_mae_deep):
    """MAE with deep supervision, no alignment term (baseline sharing the alignment training step)."""

    align_weight: float = 0.0

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.align_weight = float(os.environ.get("ALIGN_WEIGHT", str(self.align_weight)))
        self.num_epochs = int(os.environ.get("ALIGN_NUM_EPOCHS", str(self.num_epochs)))
        self.alignment_loss = self._build_alignment_loss()
        if self.alignment_loss is not None:
            self.alignment_loss.to(self.device)
        self.print_to_log_file(
            f"alignment term: {type(self.alignment_loss).__name__ if self.alignment_loss else 'none'} "
            f"weight={self.align_weight} epochs={self.num_epochs}"
        )

    def _build_alignment_loss(self):
        return None

    def _alignment_term(self, pred: torch.Tensor, source: torch.Tensor, keys) -> torch.Tensor:
        return self.alignment_loss(pred, source)

    def _compute_losses(self, data: torch.Tensor, keys=None):
        input_data = data[:, 0:1]
        target_data = data[:, 1:2]

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            output = self.network(input_data)
            if self.enable_deep_supervision:
                loss_reg = self.loss(output, self._downsample_target_for_ds(target_data))
                pred_hr = output[0] if isinstance(output, (list, tuple)) else output
            else:
                loss_reg = self.loss(output, target_data)
                pred_hr = output

        if self.alignment_loss is None or self.align_weight == 0:
            loss_align = torch.zeros((), device=data.device)
        else:
            loss_align = self._alignment_term(pred_hr.float(), input_data.float(), keys)
        loss = loss_reg + self.align_weight * loss_align
        return loss, loss_reg, loss_align

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss, loss_reg, loss_align = self._compute_losses(data, batch.get("keys"))

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_reg": loss_reg.detach().cpu().numpy(),
            "loss_align": loss_align.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        with torch.no_grad():
            loss, loss_reg, loss_align = self._compute_losses(data, batch.get("keys"))
        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_reg": loss_reg.detach().cpu().numpy(),
            "loss_align": loss_align.detach().cpu().numpy(),
            "tp_hard": 0,
            "fp_hard": 0,
            "fn_hard": 0,
        }

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        reg = np.mean([o["loss_reg"] for o in train_outputs])
        align = np.mean([o["loss_align"] for o in train_outputs])
        self.print_to_log_file(f"train loss_reg {reg:.4f} loss_align {align:.4f}")


class nnUNetTrainerRegression_align_nmi(nnUNetTrainerRegression_align_none):
    """MAE + ALIGN_WEIGHT * (2 - NMI(prediction, input))."""

    align_weight: float = 0.1

    def _build_alignment_loss(self):
        return NMILoss(num_bins=int(os.environ.get("ALIGN_NMI_BINS", 32)),
                       num_samples=int(os.environ.get("ALIGN_NMI_SAMPLES", 50_000)))


class nnUNetTrainerRegression_align_mind(nnUNetTrainerRegression_align_none):
    """MAE + ALIGN_WEIGHT * MIND-SSC distance(prediction, input)."""

    align_weight: float = 0.1

    def _build_alignment_loss(self):
        return MINDSSCLoss(radius=2, dilation=2)


class nnUNetTrainerRegression_align_siglip(nnUNetTrainerRegression_align_none):
    """MAE + ALIGN_WEIGHT * (1 - cos) of frozen SigLIP backbone tokens of prediction and input.

    The network sees normalized data, SigLIP needs HU: the prediction is de-normalized with the plans' global
    CT statistics (channel 1), the input CBCT (per-image z-score, channel 0) with its per-case statistics from
    siglip_stats.json. SigLIP's own per-volume statistics come from the same file (CT stats for the prediction).
    """

    align_weight: float = 1.0

    def _build_alignment_loss(self):
        stats_file = join(self.preprocessed_dataset_folder_base, "siglip_stats.json")
        self.siglip_stats = load_json(stats_file)
        ct_props = self.plans_manager.foreground_intensity_properties_per_channel["1"]
        self.ct_norm = (float(ct_props["mean"]), float(ct_props["std"]))
        return SigLIPFeatureLoss(os.environ["ALIGN_SIGLIP_CKPT"], self.configuration_manager.spacing,
                                 mode=os.environ.get("ALIGN_SIGLIP_MODE", "windows"),
                                 max_windows=int(os.environ.get("ALIGN_SIGLIP_MAX_WINDOWS", 0)))

    def _alignment_term(self, pred, source, keys):
        if keys is None:
            raise RuntimeError("the SigLIP alignment term needs the case keys of the batch")
        stats = [self.siglip_stats[k] for k in keys]
        col = lambda name: torch.tensor([s[name] for s in stats], device=pred.device, dtype=torch.float32)
        view = (-1, 1, 1, 1, 1)
        pred_hu = pred * self.ct_norm[1] + self.ct_norm[0]
        source_hu = source * col("zscore_std").view(view) + col("zscore_mean").view(view)
        return self.alignment_loss(pred_hu, source_hu, col("ct_mean"), col("ct_std"),
                                   col("cbct_mean"), col("cbct_std"))
