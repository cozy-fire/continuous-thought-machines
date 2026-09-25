"""Shared ResNet with GroupNorm; visual gradients belong only to TA X."""
from __future__ import annotations

from functools import partial
import torch
from torch import Tensor, nn

from models.resnet import prepare_resnet_backbone
from ..config import Config, validate_config


class VisionEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        validate_config(config)
        self.backbone = prepare_resnet_backbone(config.vision.backbone,
                                                norm_layer=partial(nn.GroupNorm, 32))
        self.register_buffer("permanently_frozen", torch.tensor(False))
        self.register_buffer("encoder_version", torch.tensor(0, dtype=torch.int64))
        self._x_training = False
        self.set_x_training(False)

    def set_x_training(self, enabled: bool) -> None:
        if enabled and self.permanently_frozen.item():
            raise RuntimeError("visual learning cannot resume after permanent freezing")
        self._x_training = enabled
        self.requires_grad_(enabled)
        if not enabled:
            # Clear stale X gradients before C/F/P assert frozen ownership.
            for parameter in self.parameters():
                parameter.grad = None
        self.train(enabled)

    def freeze(self, *, permanent: bool = False) -> None:
        if permanent:
            self.permanently_frozen.fill_(True)
        self.set_x_training(False)

    def train(self, mode: bool = True) -> VisionEncoder:
        # A parent policy cannot reactivate the encoder outside TA X.
        super().train(mode and self._x_training and not self.permanently_frozen.item())
        return self

    def forward(self, normalized_rgb: Tensor) -> Tensor:
        if normalized_rgb.ndim != 4 or tuple(normalized_rgb.shape[1:]) != (3, 84, 84):
            raise ValueError("vision input must have shape [B,3,84,84]")
        if normalized_rgb.dtype != torch.float32:
            raise ValueError("vision input must be normalized float32")
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._x_training):
            return self.backbone(normalized_rgb)


def encode_obs(obs_u8: Tensor, encoder: VisionEncoder) -> Tensor:
    """The sole image preprocessing path; no ImageNet normalization or frame stacking."""
    if obs_u8.dtype != torch.uint8:
        raise ValueError("encode_obs requires uint8 RGB; do not normalize twice")
    return encoder(obs_u8.to(dtype=torch.float32) / 255.0)
