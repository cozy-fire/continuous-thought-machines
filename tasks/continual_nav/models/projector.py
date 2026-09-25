"""Visual projection used only by the X-stage SIGReg objective."""
from __future__ import annotations

from torch import Tensor, nn

from ..config import Config


class SigregProjector(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        x = config.exploration
        self.layers = nn.Sequential(nn.Linear(128, x.projector_hidden), nn.ReLU(),
                                    nn.Linear(x.projector_hidden, x.projection_dim))

    def forward(self, fmap: Tensor) -> Tensor:
        if fmap.ndim != 4 or tuple(fmap.shape[1:]) != (128, 21, 21):
            raise ValueError("projector expects [B,128,21,21] visual features")
        return self.layers(fmap.mean(dim=(-2, -1)))
