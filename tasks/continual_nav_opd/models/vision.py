"""Each column owns a trainable GroupNorm ResNet; freezing belongs to its policy."""
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

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.dtype != torch.uint8 or rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 84, 84) or rgb.shape[0] == 0:
            raise ValueError("encoder expects nonempty uint8 [B,3,84,84]")
        # There is no TA/X switch: the learning policy controls grad mode and ownership.
        return self.backbone(rgb.to(dtype=torch.float32) / 255.0)
