"""Shared visual batching and environment-step sequence replay."""
from copy import deepcopy
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from ...config import TICKS
from .vision import VisionEncoder, validate_rgb


class VisualPolicy(nn.Module):
    def __init__(self, encoder_chunk: int = 32, checkpoint_encoder: bool = True):
        super().__init__()
        self.encoder = VisionEncoder()
        self.encoder_chunk, self._frozen = encoder_chunk, False
        self.checkpoint_encoder = checkpoint_encoder

    def train(self, mode: bool = True):
        return super().train(False if self._frozen else mode)

    def encode(self, rgb: Tensor, chunk: int | None = None) -> Tensor:
        use_checkpoint = self.checkpoint_encoder and torch.is_grad_enabled() and not self._frozen
        return torch.cat([checkpoint(self.encoder,part,use_reentrant=False) if use_checkpoint else self.encoder(part)
                          for part in rgb.split(chunk or self.encoder_chunk)])

    def step(self, rgb, state, starts, *, task: str):
        validate_rgb(rgb, 4)
        return self.advance(self.encode(rgb),state,starts,TICKS[task],self.prepare_recurrence())

    def prepare_recurrence(self):
        return None

    def sequence(self, rgb, state, starts, *, task: str, encoder_chunk: int | None = None):
        validate_rgb(rgb, 5)
        length, batch = rgb.shape[:2]
        if not 1 <= length <= 50 or starts.shape != (length, batch) or starts.dtype != torch.bool:
            raise ValueError('invalid sequence length or reset mask')
        features = self.encode(rgb.flatten(0, 1), encoder_chunk).reshape(length, batch, 128, 21, 21)
        prepared = self.prepare_recurrence()
        outputs = []
        for t in range(length):
            logits,state = self.advance(features[t],state,starts[t],TICKS[task],prepared)
            outputs.append(logits)
        return torch.stack(outputs),state


def frozen_copy(policy):
    result = deepcopy(policy)
    # deepcopy breaks cuDNN's packed LSTM storage. Repack the independent copy
    # once, rather than allocating a temporary packed buffer at every tick.
    for module in result.modules():
        if isinstance(module,nn.LSTM):
            module.flatten_parameters()
    result._frozen = True
    return result.requires_grad_(False).eval()
