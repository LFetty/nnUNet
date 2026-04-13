"""
Regression losses for 3D image-to-image translation (e.g., CBCT→CT).

- GradientLoss3D: L1 loss on 3D Sobel gradient magnitudes. Sharpens edges and
  fine structures without requiring a pretrained feature extractor.
- MedicalPerceptualLoss: Feature-space L1 loss using a frozen JIT-traced teacher
  network operating on sampled axial slices. Supports single-feature tensors as
  well as multi-layer traced outputs (tuple/list/dict of features) produced by
  scripts/trace_perceptual_model.py.
- RegressionCompoundLoss: Weighted sum of MAE + GradientLoss3D + MedicalPerceptualLoss.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GradientLoss3D(nn.Module):
    """
    L1 loss on 3D gradient magnitudes computed with fixed Sobel kernels.

    Penalises differences in edge structure between prediction and target,
    complementing the pixel-wise MAE loss with a sharpness term.

    Each Sobel kernel is the outer product of the centred difference kernel
    [-1, 0, 1] along the axis of interest and the smoothing kernel [1, 2, 1]
    along the two orthogonal axes, normalised by 32 so values lie in [-1, 1].
    All three kernels are stacked into a single (3, 1, 3, 3, 3) weight so the
    three gradient directions are computed in one F.conv3d call.
    """

    def __init__(self):
        super().__init__()
        smooth = torch.tensor([1.0, 2.0, 1.0])
        diff = torch.tensor([-1.0, 0.0, 1.0])

        # (3, 3, 3) Sobel kernels; normalisation factor = 4 * 4 * 2 = 32.
        kx = torch.einsum("i,j,k->ijk", smooth, smooth, diff) / 32.0  # ∂/∂W
        ky = torch.einsum("i,j,k->ijk", smooth, diff, smooth) / 32.0  # ∂/∂H
        kz = torch.einsum("i,j,k->ijk", diff, smooth, smooth) / 32.0  # ∂/∂D

        # Stack into (3, 1, 3, 3, 3): one conv3d call produces all three gradients.
        self.register_buffer("kernels", torch.stack([kx, ky, kz]).unsqueeze(1))

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute |∇x| via 3D Sobel; input shape (B, 1, D, H, W)."""
        # Output: (B, 3, D, H, W) — channels are gx, gy, gz
        grads = F.conv3d(x, self.kernels.to(x), padding=1)
        return (grads.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   Network output [B, 1, D, H, W].
            target: Ground-truth CT [B, 1, D, H, W].

        Returns:
            Scalar gradient L1 loss.
        """
        return F.l1_loss(self._gradient_magnitude(pred), self._gradient_magnitude(target))


class MedicalPerceptualLoss(nn.Module):
    """
    Perceptual loss using a frozen JIT-traced 2D feature extractor applied to
    axial slices sampled from 3D volumes.

    Workflow
    --------
    1. Run ``scripts/trace_perceptual_model.py`` to produce a ``.pt`` file
       containing the traced feature extractor (for example a MedDiNOv3 wrapper
       that returns one or more intermediate feature maps).
    2. Pass the path to this loss at construction time.
    3. The loss samples ``n_slices`` axial planes per volume, repeats the
       single grayscale channel to 3-channel RGB, normalises with ImageNet
       statistics, and returns L1 distance in feature space.
    4. The traced model may return:
       - a single tensor
       - a tuple/list of tensors
       - a dict of tensors
       All returned features are compared and averaged.

    Args:
        model_path: Path to the JIT-traced ``.pt`` feature model.
        n_slices: Number of uniformly-spaced axial slices to sample per volume.
        weight: Optional scalar multiplier (convenient when used outside
                RegressionCompoundLoss).
    """

    # MedDiNOv3-guided intensity normalization for CT-like inputs.
    # Inputs to this loss are in training-normalized space, so we first
    # denormalize them back to original intensities using the target-channel
    # normalization statistics from nnUNet plans before applying the MedDiNOv3
    # normalization.
    _MEAN_DINO = (65.0, 65.0, 65.0)
    _STD_DINO = (178.0, 178.0, 178.0)
    _INPUT_SIZE = (224, 224)

    def __init__(
        self,
        model_path: str,
        n_slices: int = 4,
        weight: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Perceptual model not found at '{model_path}'. "
                "Run scripts/trace_perceptual_model.py to generate it."
            )
        self.feature_net = torch.jit.load(model_path)
        self.feature_net.eval()
        for p in self.feature_net.parameters():
            p.requires_grad_(False)
        self.feature_net = torch.compile(self.feature_net)

        self.n_slices = n_slices
        self.weight = weight
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

        mean_dino = torch.tensor(self._MEAN_DINO).view(1, 3, 1, 1)
        std_dino = torch.tensor(self._STD_DINO).view(1, 3, 1, 1)
        self.register_buffer("mean_dino", mean_dino)
        self.register_buffer("std_dino", std_dino)

    def _extract_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sample axial slices and prepare them for the 2-D feature network.

        Args:
            x: 3D volume [B, 1, D, H, W] — values assumed in [0, 1] or
               normalised range coming from the regression trainer.

        Returns:
            Slices [B * n_slices, 3, 224, 224] denormalized from training space
            and then normalized with MedDiNOv3-style intensity statistics.
        """
        B, _, D, H, W = x.shape
        indices = torch.linspace(0, D - 1, self.n_slices, dtype=torch.long, device=x.device)
        # (B, 1, n_slices, H, W) → (B*n_slices, 1, H, W)
        # Cast to fp32 to avoid fp16 overflow when multiplying by target_std (~478)
        slices = x.float()[:, :, indices, :, :].permute(0, 2, 1, 3, 4).reshape(B * self.n_slices, 1, H, W)

        # Convert from training-normalized target space back to original intensity space.
        slices = slices * self.target_std + self.target_mean

        # Clamp to CT HU range to prevent fp16 overflow in bicubic interpolation
        # and the feature extractor (untrained networks can produce extreme values).
        slices = slices.clamp(-1024, 3072)

        # Duplicate grayscale to 3 channels for the 2D visual encoder.
        slices = slices.expand(-1, 3, -1, -1)

        if tuple(slices.shape[-2:]) != self._INPUT_SIZE:
            slices = F.interpolate(
                slices,
                size=self._INPUT_SIZE,
                mode="bicubic",
                align_corners=False,
            )

        mean_dino = self.mean_dino.to(slices)
        std_dino = self.std_dino.to(slices)
        slices = (slices - mean_dino) / std_dino
        return slices

    def _flatten_feature_outputs(self, outputs) -> list[torch.Tensor]:
        """
        Normalise traced feature extractor outputs into a flat list of tensors.

        Supports:
        - Tensor
        - tuple/list of tensors
        - dict[str, Tensor]

        Nested containers are flattened recursively. Non-tensor values raise.
        """
        if torch.is_tensor(outputs):
            return [outputs]

        if isinstance(outputs, dict):
            flattened = []
            for key in sorted(outputs.keys()):
                flattened.extend(self._flatten_feature_outputs(outputs[key]))
            return flattened

        if isinstance(outputs, (tuple, list)):
            flattened = []
            for item in outputs:
                flattened.extend(self._flatten_feature_outputs(item))
            return flattened

        raise TypeError(
            "Unsupported perceptual feature output type "
            f"{type(outputs)}. Expected Tensor, tuple/list, or dict."
        )

    def _feature_distance(self, pred_features, target_features) -> torch.Tensor:
        pred_list = self._flatten_feature_outputs(pred_features)
        target_list = self._flatten_feature_outputs(target_features)

        if len(pred_list) != len(target_list):
            raise ValueError(
                "Perceptual feature extractor returned a different number of "
                f"features for prediction ({len(pred_list)}) and target ({len(target_list)})."
            )

        if len(pred_list) == 0:
            raise ValueError("Perceptual feature extractor returned no features.")

        losses = []
        for pred_feat, target_feat in zip(pred_list, target_list):
            if pred_feat.shape != target_feat.shape:
                raise ValueError(
                    "Perceptual feature shape mismatch: "
                    f"pred {tuple(pred_feat.shape)} vs target {tuple(target_feat.shape)}."
                )
            losses.append(F.l1_loss(pred_feat, target_feat))

        return torch.stack(losses).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_slices = self._extract_slices(pred)
        target_slices = self._extract_slices(target)
        with torch.no_grad():
            target_feat = self.feature_net(target_slices)
        pred_feat = self.feature_net(pred_slices)
        return self._feature_distance(pred_feat, target_feat) * self.weight


class RegressionCompoundLoss(nn.Module):
    """
    Weighted compound loss for image-to-image regression:

        L = w_mae * MAE + w_grad * GradientLoss3D [+ w_perc * MedicalPerceptualLoss]

    Designed to be used inside ``DeepSupervisionWrapper``: its ``forward``
    signature matches ``loss(pred, target)`` where both tensors are [B, 1, D, H, W].

    Args:
        w_mae:  Weight for the pixel-wise MAE term. Default 1.0.
        w_grad: Weight for the 3D gradient loss term. Default 0.1.
            Set to 0 to disable gradient loss (saves one conv pass per step).
        w_perc: Weight for the perceptual loss term. Default 0.0 (disabled).
        perceptual_model_path: Path to the JIT-traced feature model. Required
            only if w_perc > 0.
        perceptual_n_slices: Axial slices to sample for the perceptual loss.
    """

    def __init__(
        self,
        w_mae: float = 1.0,
        w_grad: float = 0.1,
        w_perc: float = 0.0,
        perceptual_model_path: Optional[str] = None,
        perceptual_n_slices: int = 4,
        perceptual_target_mean: float = 0.0,
        perceptual_target_std: float = 1.0,
    ):
        super().__init__()
        self.w_mae = w_mae
        self.w_grad = w_grad
        self.w_perc = w_perc

        self.mae = nn.L1Loss()

        if w_grad > 0:
            self.grad_loss: Optional[GradientLoss3D] = GradientLoss3D()
        else:
            self.grad_loss = None

        if w_perc > 0:
            if perceptual_model_path is None:
                raise ValueError(
                    "perceptual_model_path must be provided when w_perc > 0."
                )
            self.perc_loss: Optional[MedicalPerceptualLoss] = MedicalPerceptualLoss(
                perceptual_model_path,
                n_slices=perceptual_n_slices,
                target_mean=perceptual_target_mean,
                target_std=perceptual_target_std,
            )
        else:
            self.perc_loss = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.w_mae * self.mae(pred, target)

        if self.grad_loss is not None:
            loss = loss + self.w_grad * self.grad_loss(pred, target)

        if self.perc_loss is not None:
            loss = loss + self.w_perc * self.perc_loss(pred, target)

        return loss


class PerceptualHighestResolutionLossWrapper(nn.Module):
    """
    Adds full-resolution-only terms (gradient + perceptual) on top of any base loss.

    ``base_loss`` handles all deep-supervision scales.  This wrapper applies the
    gradient and perceptual losses only to the highest-resolution output (index 0
    when pred is a list, or pred itself when it is a plain tensor).

    Keeping gradient loss here (rather than inside DeepSupervisionWrapper) avoids
    penalising sharpness on blurry downsampled targets at lower DS scales.

    Args:
        base_loss:         Loss applied to all outputs (e.g. DeepSupervisionWrapper).
        perceptual_loss:   Frozen feature-space loss, full resolution only.
        perceptual_weight: Scalar multiplier for the perceptual term.
        gradient_loss:     Optional GradientLoss3D, full resolution only.
        gradient_weight:   Scalar multiplier for the gradient term.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        perceptual_loss: Optional[MedicalPerceptualLoss] = None,
        perceptual_weight: float = 0.0,
        gradient_loss: Optional[GradientLoss3D] = None,
        gradient_weight: float = 0.0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.perceptual_loss = perceptual_loss
        self.perceptual_weight = perceptual_weight
        self.gradient_loss = gradient_loss
        self.gradient_weight = gradient_weight

    def forward(self, pred, target) -> torch.Tensor:
        loss = self.base_loss(pred, target)

        pred_hr = pred[0] if isinstance(pred, (tuple, list)) else pred
        target_hr = target[0] if isinstance(target, (tuple, list)) else target

        if self.gradient_loss is not None and self.gradient_weight > 0:
            loss = loss + self.gradient_weight * self.gradient_loss(pred_hr, target_hr)

        if self.perceptual_loss is not None and self.perceptual_weight > 0:
            loss = loss + self.perceptual_weight * self.perceptual_loss(pred_hr, target_hr)

        return loss
