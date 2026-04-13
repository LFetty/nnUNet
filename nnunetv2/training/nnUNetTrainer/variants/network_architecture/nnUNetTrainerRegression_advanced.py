"""
Advanced regression trainer with compound loss (MAE + gradient sharpness term).

Inherits all data-loading, deep supervision, validation, and post-processing
logic from ``nnUNetTrainerRegression_mae_deep`` and replaces the loss with
``RegressionCompoundLoss`` wrapped in ``PerceptualHighestResolutionLossWrapper``
so that:
  - MAE + gradient loss are applied at every deep-supervision scale.
  - The perceptual term is applied only to the highest-resolution output.

To use with attention-gate networks, point the plans file at:
    "network_arch_class_name":
        "nnunetv2.network_architecture.attention_unet.PlainConvUNetRegression"
and add to arch_kwargs:
    "use_attention_gates": true,
    "use_bottleneck_sa": false
"""

from pathlib import Path
from typing import Optional  # used in local type annotation inside _build_loss

import numpy as np

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.regression_losses import (
    GradientLoss3D,
    MedicalPerceptualLoss,
    PerceptualHighestResolutionLossWrapper,
    RegressionCompoundLoss,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerRegression_mae_deep import (
    nnUNetTrainerRegression_mae_deep,
)


class nnUNetTrainerRegression_advanced(nnUNetTrainerRegression_mae_deep):
    """
    Trainer with compound loss: MAE + gradient sharpness (+ optional perceptual).

    MAE and gradient loss follow the normal nnUNet deep supervision behavior.
    The perceptual loss is kept outside deep supervision and applied only to
    the highest-resolution output, which avoids the expense of running the
    2D feature extractor on every downsampled scale.

    Class attributes (override in subclasses to change behaviour):
        w_mae:               MAE weight (default 1.0).
        w_grad:              Gradient loss weight (default 0.1).
        w_perc:              Perceptual loss weight (default 0.0 — disabled).
        perceptual_model_path: Absolute path to the JIT-traced ``.pt`` feature
                             model.  Required only when w_perc > 0.
        perceptual_n_slices: Axial slices sampled per volume for the perceptual
                             loss (default 4).
    """

    w_mae: float = 1.0
    w_grad: float = 0.1
    w_perc: float = 0.0
    perceptual_model_path: str = ""
    perceptual_n_slices: int = 4

    def _build_loss(self):
        """
        Build ``PerceptualHighestResolutionLossWrapper`` around a MAE-only
        ``DeepSupervisionWrapper``.

        Gradient loss is intentionally kept outside deep supervision: lower-DS
        targets are blurry downsampled versions where Sobel gradients carry no
        useful sharpness signal.  Both gradient and perceptual terms are applied
        only to the highest-resolution output.
        """
        normalization_schemes = self.configuration_manager.normalization_schemes
        intensity_props = self.plans_manager.foreground_intensity_properties_per_channel

        target_channel_idx = 1 if len(normalization_schemes) > 1 else 0
        target_props = intensity_props[str(target_channel_idx)]
        target_mean = float(target_props.get('mean', 0.0))
        target_std = float(target_props.get('std', 1.0))

        # Per-scale base loss: MAE only — gradient applied at HR only
        base_loss = RegressionCompoundLoss(
            w_mae=self.w_mae,
            w_grad=0.0,
            w_perc=0.0,
        )

        # Gradient loss (full-resolution only)
        grad_loss: Optional["GradientLoss3D"] = GradientLoss3D() if self.w_grad > 0 else None

        # Perceptual loss (full-resolution only)
        perc_loss: Optional["MedicalPerceptualLoss"] = None
        if self.w_perc > 0:
            if not self.perceptual_model_path:
                raise ValueError("perceptual_model_path must be set when w_perc > 0.")
            perc_loss = MedicalPerceptualLoss(
                self.perceptual_model_path,
                n_slices=self.perceptual_n_slices,
                target_mean=target_mean,
                target_std=target_std,
            )

        ds_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(ds_scales))], dtype=np.float32)
        weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0.0
        ds_loss = DeepSupervisionWrapper(base_loss, (weights / weights.sum()).tolist())

        return PerceptualHighestResolutionLossWrapper(
            base_loss=ds_loss,
            gradient_loss=grad_loss,
            gradient_weight=self.w_grad,
            perceptual_loss=perc_loss,
            perceptual_weight=self.w_perc,
        )


class nnUNetTrainerRegression_advanced_strongGrad(nnUNetTrainerRegression_advanced):
    """Stronger gradient sharpening: w_grad = 0.5."""
    w_grad: float = 0.5


class nnUNetTrainerRegression_advanced_noGrad(nnUNetTrainerRegression_advanced):
    """MAE-only loss (gradient term disabled)."""
    w_grad: float = 0.0


class nnUNetTrainerRegression_advanced_perceptual(nnUNetTrainerRegression_advanced):
    """
    MAE + gradient + perceptual loss.

    The perceptual model is loaded from ``perceptual_model.pt`` at the
    repository root (resolved relative to this file at import time).
    """
    w_perc: float = 0.01
    perceptual_model_path = str(Path(__file__).parents[5] / "perceptual_model.pt")
    perceptual_n_slices: int = 4


class nnUNetTrainerRegression_advanced_perceptual_strongGrad(nnUNetTrainerRegression_advanced_perceptual):
    """Perceptual + stronger gradient sharpening: w_grad = 0.5."""
    w_grad: float = 0.5


class nnUNetTrainerRegression_advanced_perceptual_spatial(nnUNetTrainerRegression_advanced):
    """
    MAE (DS) + gradient loss (HR only) + spatial multi-layer perceptual loss (HR only).

    Improvements over ``nnUNetTrainerRegression_advanced_perceptual``:
      - Gradient loss moved out of deep supervision — applied only at full resolution
        to avoid penalising sharpness on blurry downsampled DS targets.
      - Perceptual model uses spatial patch tokens (no mean pooling) from three
        intermediate ViT layers (4, 8, 12), giving richer structural comparison
        instead of a single global embedding.
      - Higher perceptual weight (0.05) and more slices (16) for a stronger,
        lower-variance gradient signal per training step.
    """
    w_perc: float = 0.05
    w_grad: float = 0.1
    perceptual_model_path: str = str(Path(__file__).parents[5] / "perceptual_model_spatial_multilayer.pt")
    perceptual_n_slices: int = 16
