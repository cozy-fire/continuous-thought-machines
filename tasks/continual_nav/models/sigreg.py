"""Sketched ECF regularizer following LeJEPA's public MINIMAL.md algorithm.

Source: https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md
The predictive/action-conditioned use and loss weighting belong to this project.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import SIGRegConfig


class SIGReg(nn.Module):
    def __init__(self, config: SIGRegConfig):
        super().__init__()
        self.directions = config.directions
        t = torch.linspace(0, config.t_max, config.knots, dtype=torch.float32)
        phi = (-t.square()/2).exp()
        # Positive half-axis integral doubled: endpoint dt, interior 2*dt.
        weights = torch.full_like(t, 2*config.t_max/(config.knots-1))
        weights[[0, -1]] /= 2
        self.register_buffer("knots", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights*phi)

    def sample_directions(self, dimensions: int, generator: torch.Generator, device: torch.device) -> Tensor:
        # An explicit CPU generator keeps this RNG stream separate from policy sampling.
        if generator.device.type != "cpu":
            raise ValueError("SIGReg requires a CPU generator")
        a = torch.randn(dimensions, self.directions, generator=generator, dtype=torch.float32)
        return (a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)).to(device)

    def forward(self, z: Tensor, directions: Tensor) -> Tensor:
        if z.ndim != 2 or z.shape[0] == 0 or z.dtype != torch.float32:
            raise ValueError("SIGReg expects nonempty float32 [B,Z]")
        if directions.shape != (z.shape[1], self.directions):
            raise ValueError("invalid SIGReg directions")
        projected = (z @ directions).unsqueeze(-1) * self.knots
        error = (projected.cos().mean(0)-self.phi).square() + projected.sin().mean(0).square()
        return (error*self.weights).sum(-1).mean()*z.shape[0]
