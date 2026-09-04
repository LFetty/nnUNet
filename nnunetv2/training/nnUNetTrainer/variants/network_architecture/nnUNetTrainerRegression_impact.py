"""
Regression trainer with IMPACT feature regularization.

The base objective remains the normal nnU-Net regression MAE with deep
supervision. A full-resolution IMPACT term is added between the source input
image (channel 0, for example MR) and the generated target image (network
output, for example sCT). This makes the term a source/output anatomical
regularizer rather than a target/output registration-sensitive loss.
"""
from __future__ import annotations

import os
from typing import Sequence

import torch
from torch import autocast

from nnunetv2.training.loss.impact_loss import IMPACTFeatureLoss
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_mae_deep import (
    nnUNetTrainerRegression_mae_deep,
)
from nnunetv2.utilities.helpers import dummy_context


def _parse_shape(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


class nnUNetTrainerRegression_impact(nnUNetTrainerRegression_mae_deep):
    """
    MAE deep supervision + IMPACT(source, prediction) regularization.

    Environment overrides:
        IMPACT_MODEL_NAME   TorchScript model name. Default: TS/M730.pt
        IMPACT_SHAPE        Comma-separated spatial size. Default: 0,0,0
        IMPACT_WEIGHTS      Comma-separated per-layer weights. Default: 1,1
        IMPACT_IN_CHANNELS  Feature-model input channels. Default: 1
        IMPACT_WEIGHT       Outer regularization weight. Default: 0.1
        IMPACT_REPO_ID      HF repo id. Default: VBoussot/impact-torchscript-models
        IMPACT_CACHE_DIR    Optional Hugging Face cache directory.
    """

    impact_model_name: str = "TS/M730.pt"
    impact_shape: Sequence[int] = (0, 0, 0)
    impact_weights: Sequence[float] = (1.0, 1.0)
    impact_in_channels: int = 1
    impact_weight: float = 0.1
    impact_repo_id: str = "VBoussot/impact-torchscript-models"

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.impact_model_name = os.environ.get("IMPACT_MODEL_NAME", self.impact_model_name)
        self.impact_shape = _parse_shape(os.environ.get("IMPACT_SHAPE", ",".join(map(str, self.impact_shape))))
        self.impact_weights = [
            float(v.strip())
            for v in os.environ.get("IMPACT_WEIGHTS", ",".join(map(str, self.impact_weights))).split(",")
            if v.strip()
        ]
        self.impact_in_channels = int(os.environ.get("IMPACT_IN_CHANNELS", str(self.impact_in_channels)))
        self.impact_weight = float(os.environ.get("IMPACT_WEIGHT", str(self.impact_weight)))
        self.impact_repo_id = os.environ.get("IMPACT_REPO_ID", self.impact_repo_id)

        intensity_props = self.plans_manager.foreground_intensity_properties_per_channel
        source_props = intensity_props.get("0", {})
        target_channel_idx = 1 if len(self.configuration_manager.normalization_schemes) > 1 else 0
        target_props = intensity_props.get(str(target_channel_idx), {})

        self.impact_loss = IMPACTFeatureLoss(
            model_name=self.impact_model_name,
            shape=self.impact_shape,
            in_channels=self.impact_in_channels,
            weights=self.impact_weights,
            source_mean=float(source_props.get("mean", 0.0)),
            source_std=float(source_props.get("std", 1.0)),
            target_mean=float(target_props.get("mean", 0.0)),
            target_std=float(target_props.get("std", 1.0)),
            repo_id=self.impact_repo_id,
            cache_dir=os.environ.get("IMPACT_CACHE_DIR"),
        )
        self.impact_loss.to(self.device)

        self.print_to_log_file(
            "IMPACT regularization active | "
            f"model={self.impact_model_name} shape={tuple(self.impact_shape)} "
            f"weights={self.impact_weights} w={self.impact_weight}"
        )

    def _compute_losses(self, data: torch.Tensor):
        input_data = data[:, 0:1]
        target_data = data[:, 1:2]

        ctx = autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context()
        with ctx:
            output = self.network(input_data)
            if self.enable_deep_supervision:
                target_scales = self._downsample_target_for_ds(target_data)
                loss_reg = self.loss(output, target_scales)
                pred_hr = output[0] if isinstance(output, (list, tuple)) else output
            else:
                loss_reg = self.loss(output, target_data)
                pred_hr = output

        loss_impact = self.impact_loss(pred_hr, input_data)
        loss = loss_reg + self.impact_weight * loss_impact
        return loss, loss_reg, loss_impact

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        loss, loss_reg, loss_impact = self._compute_losses(data)

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
            "loss_impact": loss_impact.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        loss, loss_reg, loss_impact = self._compute_losses(data)
        return {
            "loss": loss.detach().cpu().numpy(),
            "loss_reg": loss_reg.detach().cpu().numpy(),
            "loss_impact": loss_impact.detach().cpu().numpy(),
            "tp_hard": 0,
            "fp_hard": 0,
            "fn_hard": 0,
        }


class nnUNetTrainerRegression_impact_low(nnUNetTrainerRegression_impact):
    """Lower IMPACT regularization weight for quick ablations."""

    impact_weight: float = 0.01


class nnUNetTrainerRegression_impact_strong(nnUNetTrainerRegression_impact):
    """Stronger IMPACT regularization weight for quick ablations."""

    impact_weight: float = 0.5
