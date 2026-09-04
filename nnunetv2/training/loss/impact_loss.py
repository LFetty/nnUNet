"""
IMPACT feature regularization for image-to-image regression.

This module follows the public PyTorch API from
https://github.com/vboussot/ImpactLoss while keeping the nnU-Net integration
self-contained. IMPACT compares frozen TorchScript feature maps and expects
raw modality intensities, so ``IMPACTFeatureLoss`` denormalizes nnU-Net
training-space tensors before computing the feature distance.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _as_float_list(values: Sequence[float] | str) -> list[float]:
    if isinstance(values, str):
        return [float(v.strip()) for v in values.split(",") if v.strip()]
    return [float(v) for v in values]


def _flatten_feature_outputs(outputs) -> list[torch.Tensor]:
    if torch.is_tensor(outputs):
        return [outputs]
    if isinstance(outputs, dict):
        result: list[torch.Tensor] = []
        for key in sorted(outputs.keys()):
            result.extend(_flatten_feature_outputs(outputs[key]))
        return result
    if isinstance(outputs, (tuple, list)):
        result = []
        for item in outputs:
            result.extend(_flatten_feature_outputs(item))
        return result
    raise TypeError(f"Unsupported IMPACT model output type: {type(outputs)}")


class ImpactRegLoss(nn.Module):
    """
    Frozen IMPACT TorchScript feature loss.

    Parameters mirror upstream ``IMPACTReg``:
        model_name: TorchScript model filename on Hugging Face, for example
            ``TS/M730.pt``.
        shape: Spatial size passed to the feature extractor. Use all zeros to
            keep the incoming patch size.
        in_channels: Number of channels expected by the feature model.
        weights: Per-layer feature weights.
    """

    def __init__(
        self,
        model_name: str = "TS/M730.pt",
        shape: Sequence[int] = (0, 0, 0),
        in_channels: int = 1,
        weights: Sequence[float] | str = (1.0, 1.0),
        repo_id: str = "VBoussot/impact-torchscript-models",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "IMPACT loss requires huggingface_hub. Install it or provide "
                "an environment with the ImpactLoss dependencies."
            ) from e

        self.in_channels = int(in_channels)
        self.shape = tuple(int(s) for s in shape)
        self.resample_shape = self.shape if all(s > 0 for s in self.shape) else None
        self.dim = len(self.shape)
        self.weights = _as_float_list(weights)
        self.nb_layer = len(self.weights)
        self.loss = nn.L1Loss()

        hf_cache_dir = cache_dir or os.environ.get("IMPACT_CACHE_DIR")
        try:
            self.model_path = hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                repo_type="model",
                cache_dir=hf_cache_dir,
            )
        except Exception:
            if model_name.endswith(".pt"):
                raise
            self.model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{model_name}.pt",
                repo_type="model",
                cache_dir=hf_cache_dir,
            )
        self.model: Optional[torch.nn.Module] = None
        self._call_mode: Optional[str] = None

    def _load_model(self, device: torch.device) -> torch.nn.Module:
        if self.model is None:
            self.model = torch.jit.load(self.model_path, map_location=device)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.model.to(device)
        return self.model

    def _preprocess(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        tensor = tensor.float()
        if self.resample_shape is not None and tuple(tensor.shape[-self.dim:]) != self.resample_shape:
            mode = "trilinear" if self.dim == 3 else "bilinear" if self.dim == 2 else "linear"
            tensor = F.interpolate(tensor, size=self.resample_shape, mode=mode, align_corners=False)

        if tensor.shape[1] != self.in_channels:
            if tensor.shape[1] != 1:
                raise ValueError(
                    f"Cannot adapt IMPACT input channels from {tensor.shape[1]} to {self.in_channels}."
                )
            reps = [1, self.in_channels] + [1] * self.dim
            tensor = tensor.repeat(*reps)

        nb_layer = torch.tensor([self.nb_layer], device=tensor.device)
        stats = torch.stack(
            [tensor.min(), tensor.max(), tensor.mean(), tensor.std().clamp_min(1e-6)]
        ).to(device=tensor.device, dtype=tensor.dtype)
        return [tensor, nb_layer, stats]

    def _run_model(self, tensor: torch.Tensor):
        model = self._load_model(tensor.device)
        tensor, nb_layer, stats = self._preprocess(tensor)

        if self._call_mode == "three_arg":
            return model(tensor, nb_layer, stats)
        if self._call_mode == "two_arg":
            return model(tensor, nb_layer)
        if self._call_mode == "one_arg":
            return model(tensor)

        try:
            out = model(tensor, nb_layer, stats)
            self._call_mode = "three_arg"
            return out
        except Exception as first_error:
            try:
                out = model(tensor, nb_layer)
                self._call_mode = "two_arg"
                return out
            except Exception:
                try:
                    out = model(tensor)
                    self._call_mode = "one_arg"
                    return out
                except Exception as final_error:
                    raise RuntimeError(
                        "Could not call IMPACT TorchScript model with the supported "
                        "signatures: (image, nb_layers, stats), (image, nb_layers), or (image)."
                    ) from final_error

    def _feature_distance(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_features = _flatten_feature_outputs(self._run_model(output))
        with torch.no_grad():
            target_features = _flatten_feature_outputs(self._run_model(target))

        if len(output_features) != len(target_features):
            raise ValueError(
                "IMPACT model returned a different number of feature maps for "
                f"output ({len(output_features)}) and target ({len(target_features)})."
            )

        if len(self.weights) != len(output_features):
            raise ValueError(
                f"IMPACT weights ({len(self.weights)}) do not match model outputs ({len(output_features)})."
            )

        losses = []
        for weight, output_feature, target_feature in zip(self.weights, output_features, target_features):
            if weight == 0:
                continue
            losses.append(float(weight) * self.loss(output_feature, target_feature))
        if not losses:
            return output.new_zeros(())
        return torch.stack(losses).sum()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if output.ndim == 5 and self.dim == 2:
            losses = [self._feature_distance(output[:, :, i], target[:, :, i]) for i in range(output.shape[2])]
            return torch.stack(losses).mean()
        return self._feature_distance(output, target)


class IMPACTFeatureLoss(nn.Module):
    """
    nnU-Net training-space wrapper for IMPACT.

    ``output`` is the model prediction in target-channel normalized space.
    ``source`` is the input modality in source-channel normalized space. Both
    are denormalized before the frozen IMPACT model sees them.
    """

    def __init__(
        self,
        model_name: str = "TS/M730.pt",
        shape: Sequence[int] = (0, 0, 0),
        in_channels: int = 1,
        weights: Sequence[float] | str = (1.0, 1.0),
        source_mean: float = 0.0,
        source_std: float = 1.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        repo_id: str = "VBoussot/impact-torchscript-models",
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.impact = ImpactRegLoss(
            model_name=model_name,
            shape=shape,
            in_channels=in_channels,
            weights=weights,
            repo_id=repo_id,
            cache_dir=cache_dir,
        )
        self.source_mean = float(source_mean)
        self.source_std = float(source_std)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def forward(self, output: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        output_raw = output.float() * self.target_std + self.target_mean
        source_raw = source.float() * self.source_std + self.source_mean
        return self.impact(output_raw, source_raw)
