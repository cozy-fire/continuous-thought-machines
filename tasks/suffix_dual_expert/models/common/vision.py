"""Identical RGB encoder, spatial RoPE attention and action-head form for both models."""
import math
from functools import partial
import torch
from torch import Tensor, nn
from models.resnet import prepare_resnet_backbone


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    u, v = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    return torch.stack((u*cos-v*sin, u*sin+v*cos), -1).flatten(-2)


class SpatialRoPE(nn.Module):
    def __init__(self):
        super().__init__()
        y, x = torch.meshgrid(torch.arange(21), torch.arange(21), indexing='ij')
        coords = torch.stack((x.flatten(), y.flatten()), -1).float()
        query = torch.tensor([[10., 10.]])
        self.register_buffer('key_coords', coords)
        self.register_buffer('query_coords', query)
        self.register_buffer('theta', torch.tensor(10000.))
        self.register_buffer('cache_version', torch.tensor(1, dtype=torch.int64))
        frequency = 10000. ** (-torch.arange(0, 16, 2, dtype=torch.float32) / 16)
        for name, positions in (('key', coords), ('query', query)):
            angles = torch.cat((positions[:, :1] * frequency, positions[:, 1:] * frequency), -1)
            self.register_buffer(name + '_cos', angles.cos())
            self.register_buffer(name + '_sin', angles.sin())


class SpatialAttention(nn.Module):
    def __init__(self, query_dim: int = 528):
        super().__init__()
        self.query_input = nn.Linear(query_dim, 128)
        self.token_input = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128))
        self.q, self.k, self.v, self.output = (nn.Linear(128, 128) for _ in range(4))
        self.rope = SpatialRoPE()

    def prepare(self, fmap: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.token_input(fmap.flatten(2).transpose(1, 2))
        batch = fmap.shape[0]
        k = self.k(tokens).reshape(batch, -1, 4, 32).transpose(1, 2)
        v = self.v(tokens).reshape(batch, -1, 4, 32).transpose(1, 2)
        return rotate(k, self.rope.key_cos, self.rope.key_sin), v

    def forward_prepared(self, prepared, sync: Tensor) -> Tensor:
        k, v = prepared
        q = self.q(self.query_input(sync).unsqueeze(1)).reshape(k.shape[0], 1, 4, 32).transpose(1, 2)
        q = rotate(q, self.rope.query_cos, self.rope.query_sin)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(32)).softmax(-1)
        return self.output((weights @ v).transpose(1, 2).reshape(k.shape[0], 128))


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = prepare_resnet_backbone('resnet34-2', norm_layer=partial(nn.GroupNorm, 32))

    def forward(self, rgb: Tensor) -> Tensor:
        return self.backbone(rgb.float() / 255.)


def actor(input_dim: int = 528) -> nn.Sequential:
    result = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)
    return result


def validate_rgb(rgb: Tensor, rank: int) -> None:
    if rgb.dtype != torch.uint8 or rgb.ndim != rank or tuple(rgb.shape[-3:]) != (3, 84, 84) or min(rgb.shape) < 1:
        raise ValueError('policy expects nonempty uint8 RGB with trailing shape [3,84,84]')
