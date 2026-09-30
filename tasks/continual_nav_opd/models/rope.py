"""Axial spatial RoPE applied after the actual Q/K projections."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def axial_angles(coords: Tensor, head_dim: int, theta: float) -> Tensor:
    """coords: [N,2] in (x,y) order; result: [N,head_dim/2] pair angles."""
    if head_dim % 4 or head_dim <= 0 or coords.ndim != 2 or coords.shape[1] != 2 or theta <= 0:
        raise ValueError("axial RoPE requires dh%4=0, positive theta and [N,2] coordinates")
    axis_dim = head_dim // 2
    frequency = theta ** (-torch.arange(0, axis_dim, 2, device=coords.device, dtype=torch.float32) / axis_dim)
    return torch.cat((coords[:, :1] * frequency, coords[:, 1:] * frequency), dim=-1)


def rotate_pairs(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate adjacent pairs, preserving the x-axis half then the y-axis half."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    u, v = pairs.unbind(-1)
    return torch.stack((u*cos-v*sin, u*sin+v*cos), dim=-1).flatten(-2)


class SpatialRoPE(nn.Module):
    def __init__(self, head_dim: int = 32, height: int = 21, width: int = 21,
                 theta: float = 10000.0, query_position: tuple[int, int] = (10, 10)):
        super().__init__()
        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        coords = torch.stack((x.flatten(), y.flatten()), dim=-1).float()
        query = torch.tensor([query_position], dtype=torch.float32)
        self.register_buffer("key_coords", coords)
        self.register_buffer("query_coords", query)
        self.register_buffer("theta", torch.tensor(theta))
        self.register_buffer("cache_version", torch.tensor(1, dtype=torch.int64))
        for name, positions in (("key", coords), ("query", query)):
            angles = axial_angles(positions, head_dim, theta)
            self.register_buffer(f"{name}_cos", angles.cos())
            self.register_buffer(f"{name}_sin", angles.sin())

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        return (rotate_pairs(q, self.query_cos, self.query_sin),
                rotate_pairs(k, self.key_cos, self.key_sin))


class SpatialAttention(nn.Module):
    def __init__(self, sync_dim: int, width: int = 128, heads: int = 4,
                 theta: float = 10000.0, query_position: tuple[int, int] = (10, 10)):
        super().__init__()
        if width % heads or (width // heads) % 4:
            raise ValueError("attention width must support axial RoPE in every head")
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.query_input = nn.Linear(sync_dim, width)
        self.token_input = nn.Sequential(nn.Linear(128, width), nn.LayerNorm(width))
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.rope = SpatialRoPE(self.head_dim, theta=theta, query_position=query_position)

    def forward(self, fmap: Tensor, action_sync: Tensor, *, return_weights: bool = False):
        batch = fmap.shape[0]
        tokens = self.token_input(fmap.flatten(2).transpose(1, 2))
        query = self.query_input(action_sync).unsqueeze(1)

        def heads(x: Tensor) -> Tensor:
            return x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.q(query)), heads(self.k(tokens)), heads(self.v(tokens))
        # Rotate projected Q/K, never V or the inputs to a later unrotated projection.
        q, k = self.rope(q, k)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, self.width)
        output = self.output(attended)
        return (output, weights) if return_weights else output
