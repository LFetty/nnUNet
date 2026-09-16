"""
Alignment losses between the network input (e.g. CBCT) and the prediction (e.g. sCT).

Both terms are insensitive to the intensity mapping between the two modalities, so they measure geometry
rather than HU agreement. They are meant as regularizers next to a voxel-wise loss against the target:
when the target is misaligned with the input, they reward keeping the input's geometry.

- NMILoss: 2 - NMI with a differentiable Parzen-window joint histogram on a random voxel subset.
- MINDSSCLoss: mean squared difference of MIND-SSC descriptors (Heinrich et al., MICCAI 2013).

Both return a scalar in float32 and should be evaluated outside autocast.
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
