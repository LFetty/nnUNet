"""
Alignment losses between the network input (e.g. CBCT) and the prediction (e.g. sCT).

Both terms are insensitive to the intensity mapping between the two modalities, so they measure geometry
rather than HU agreement. They are meant as regularizers next to a voxel-wise loss against the target:
when the target is misaligned with the input, they reward keeping the input's geometry.

- NMILoss: 2 - NMI with a differentiable Parzen-window joint histogram on a random voxel subset.
- MINDSSCLoss: mean squared difference of MIND-SSC descriptors (Heinrich et al., MICCAI 2013).
- SigLIPFeatureLoss: 1 - cosine of frozen simcbct-siglip backbone tokens (takes HU volumes).

All return a scalar in float32 and should be evaluated outside autocast.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _minmax_per_sample(x: torch.Tensor, lo_q: float = 0.005, hi_q: float = 0.995) -> torch.Tensor:
    """Scale every sample to [0, 1] with robust bounds; the bounds are treated as constants."""
    flat = x.detach().flatten(1)
    k = min(flat.shape[1], 200_000)
    sub = flat[:, torch.randperm(flat.shape[1], device=x.device)[:k]]
    lo = torch.quantile(sub.float(), lo_q, dim=1).view(-1, *([1] * (x.ndim - 1)))
    hi = torch.quantile(sub.float(), hi_q, dim=1).view(-1, *([1] * (x.ndim - 1)))
    return ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


class NMILoss(nn.Module):
    """2 - NMI(a, b), NMI = (H(a) + H(b)) / H(a, b) in [1, 2]; 0 means perfectly dependent intensities."""

    def __init__(self, num_bins: int = 32, num_samples: int = 50_000, sigma_ratio: float = 0.5):
        super().__init__()
        self.num_bins = num_bins
        self.num_samples = num_samples
        self.register_buffer("centers", torch.linspace(0.0, 1.0, num_bins), persistent=False)
        self.sigma = sigma_ratio / (num_bins - 1)

    def _weights(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, N) -> (B, N, bins), rows normalized to one
        w = torch.exp(-0.5 * ((v[..., None] - self.centers) / self.sigma) ** 2)
        return w / w.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = _minmax_per_sample(a.float()), _minmax_per_sample(b.float())
        a, b = a.flatten(1), b.flatten(1)
        n = min(self.num_samples, a.shape[1])
        idx = torch.randint(0, a.shape[1], (a.shape[0], n), device=a.device)
        wa = self._weights(torch.gather(a, 1, idx))
        wb = self._weights(torch.gather(b, 1, idx))
        p_ab = torch.bmm(wa.transpose(1, 2), wb) / n  # (B, bins, bins)
        p_a, p_b = p_ab.sum(2), p_ab.sum(1)
        eps = 1e-10
        h_a = -(p_a * (p_a + eps).log()).sum(1)
        h_b = -(p_b * (p_b + eps).log()).sum(1)
        h_ab = -(p_ab * (p_ab + eps).log()).sum((1, 2))
        nmi = (h_a + h_b) / h_ab.clamp_min(eps)
        return (2.0 - nmi).mean()


def _pdist_squared(x: torch.Tensor) -> torch.Tensor:
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    return dist.clamp_min(0.0)


def mind_ssc(img: torch.Tensor, radius: int = 2, dilation: int = 2) -> torch.Tensor:
    """MIND-SSC descriptor, (B, 1, D, H, W) -> (B, 12, D, H, W). Standard formulation from Heinrich et al."""
    kernel_size = radius * 2 + 1
    six_neighbourhood = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                                      [1, 1, 2], [2, 1, 1], [1, 2, 1]], dtype=torch.long, device=img.device)
    dist = _pdist_squared(six_neighbourhood.t().unsqueeze(0).float()).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6, device=img.device), torch.arange(6, device=img.device), indexing="ij")
    mask = (x > y).view(-1) & (dist == 2).view(-1)
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3, device=img.device, dtype=img.dtype)
    mshift1.view(-1)[torch.arange(12, device=img.device) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3, device=img.device, dtype=img.dtype)
    mshift2.view(-1)[torch.arange(12, device=img.device) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation)
                              - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    lo, hi = mind_var.mean() * 0.001, mind_var.mean() * 1000
    mind_var = torch.clamp(mind_var, lo.item(), hi.item())
    mind = mind / mind_var
    mind = torch.exp(-mind)
    # permute to the conventional channel order
    return mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=img.device), :, :, :]


class MINDSSCLoss(nn.Module):
    """Mean squared difference of MIND-SSC descriptors of a and b (computed on robustly [0, 1]-scaled images)."""

    def __init__(self, radius: int = 2, dilation: int = 2):
        super().__init__()
        self.radius = radius
        self.dilation = dilation

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a, b = _minmax_per_sample(a.float()), _minmax_per_sample(b.float())
        return F.mse_loss(mind_ssc(a, self.radius, self.dilation), mind_ssc(b, self.radius, self.dilation))


class SigLIPFeatureLoss(nn.Module):
    """1 - cosine similarity of frozen SigLIP (simcbct-siglip) backbone tokens of a and b, over body tokens.

    Inputs are HU volumes [B, 1, D, H, W] at `spacing_mm` (zyx). They are resampled to the model's voxel size,
    randomly cropped to the token grid used in SigLIP training, clamped to the HU window and z-scored with the
    given per-volume statistics (as in SigLIP training). Only `a` receives gradients; the model is frozen.
    Needs `simcbct_siglip` on PYTHONPATH.
    """

    def __init__(self, checkpoint: str, spacing_mm, tokens=(8, 12, 12), min_body: float = 0.5,
                 body_threshold_hu: float = -500.0):
        super().__init__()
        from simcbct_siglip.model.aligner import load_aligner

        model, config = load_aligner(checkpoint)
        model.requires_grad_(False)
        self.encoder = model.encoder
        pairs = config["pairs"]
        self.encoder.set_patch(tuple(pairs["patch"]))
        self.encoder.grad_checkpointing = True
        self.patch = tuple(int(p) for p in pairs["patch"])
        self.voxel_mm = tuple(float(t) / p for t, p in zip(pairs["token_mm"], self.patch))
        self.window = tuple(float(w) for w in pairs["window"])
        self.spacing_mm = tuple(float(s) for s in spacing_mm)
        self.tokens = tuple(int(t) for t in tokens)
        self.min_body = min_body
        self.body_threshold_hu = body_threshold_hu

    def train(self, mode: bool = True):
        # stay in train mode so that the encoder uses gradient checkpointing; the model has no dropout
        return super().train(True)

    def _resample_crop(self, x: torch.Tensor, starts) -> torch.Tensor:
        size = [max(round(n * s / v), t * p) for n, s, v, t, p in
                zip(x.shape[2:], self.spacing_mm, self.voxel_mm, self.tokens, self.patch)]
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        crop = [t * p for t, p in zip(self.tokens, self.patch)]
        return x[:, :, starts[0]:starts[0] + crop[0], starts[1]:starts[1] + crop[1], starts[2]:starts[2] + crop[2]]

    def _starts(self, shape) -> list[int]:
        size = [max(round(n * s / v), t * p) for n, s, v, t, p in
                zip(shape, self.spacing_mm, self.voxel_mm, self.tokens, self.patch)]
        crop = [t * p for t, p in zip(self.tokens, self.patch)]
        return [int(torch.randint(0, n - c + 1, (1,))) for n, c in zip(size, crop)]

    def _normalize(self, hu: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        view = (-1,) + (1,) * (hu.ndim - 1)
        return (hu.clamp(*self.window) - mean.view(view)) / std.view(view)

    def forward(self, a_hu, b_hu, a_mean, a_std, b_mean, b_std) -> torch.Tensor:
        starts = self._starts(a_hu.shape[2:])
        a = self._resample_crop(a_hu.float(), starts)
        with torch.no_grad():
            b = self._resample_crop(b_hu.float(), starts)
            body = F.avg_pool3d((b > self.body_threshold_hu).float(), self.patch).flatten(1) >= self.min_body
        if not body.any():
            return torch.zeros((), device=a_hu.device)
        amp = torch.autocast(a.device.type, dtype=torch.bfloat16, enabled=a.device.type == "cuda")
        with amp:
            za, _ = self.encoder(self._normalize(a, a_mean, a_std))
            with torch.no_grad():
                zb, _ = self.encoder(self._normalize(b, b_mean, b_std))
        za, zb = F.normalize(za.float(), dim=-1), F.normalize(zb.float(), dim=-1)
        dist = 1.0 - (za * zb).sum(-1)
        return (dist * body).sum() / body.sum()
