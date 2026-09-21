"""Single-frame action-conditioned predictor over a shared trainable visual encoder."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import Config, validate_config
from ..contracts import WorldPrediction
from .vision import VisionEncoder, encode_obs


class WorldModel(nn.Module):
    def __init__(self, config: Config, encoder: VisionEncoder):
        super().__init__()
        validate_config(config)
        self.encoder = encoder  # Exactly the same E used by the controllers.
        w = config.world
        self.projector = nn.Sequential(nn.Linear(128, w.projector_hidden), nn.ReLU(),
                                       nn.Linear(w.projector_hidden, w.projection_dim))
        self.predictor = nn.Sequential(nn.Linear(w.projection_dim+5, w.predictor_hidden), nn.ReLU(),
                                       nn.Linear(w.predictor_hidden, w.projection_dim))
        self.register_buffer("world_model_version", torch.tensor(0, dtype=torch.int64))
        self._fit_mode = False
        self._fit_complete = False
        self.freeze()

    def set_fit_mode(self) -> None:
        # Check permanent visual freezing before changing any other module's mode.
        self.encoder.set_world_training(True)
        self.projector.requires_grad_(True)
        self.predictor.requires_grad_(True)
        self._fit_mode = True
        self._fit_complete = False
        self.train(True)

    def freeze(self) -> None:
        self._fit_mode = False
        self.encoder.freeze()
        for module in (self.projector, self.predictor):
            module.requires_grad_(False)
            for parameter in module.parameters():
                parameter.grad = None
        self.train(False)

    def train(self, mode: bool = True) -> WorldModel:
        super().train(mode and self._fit_mode)
        return self

    def transition(self, obs: Tensor, next_obs: Tensor, action: Tensor) -> WorldPrediction:
        if obs.shape != next_obs.shape or obs.ndim != 4 or obs.shape[0] == 0:
            raise ValueError("world observation pairs must have matching nonempty [B,3,84,84] shapes")
        if action.dtype != torch.int64 or action.shape != (obs.shape[0],):
            raise ValueError("world actions must be int64 [B]")
        if bool(((action < 0) | (action >= 5)).any()):
            raise ValueError("world actions must keep policy indices 0..4, never sentinel 9")
        with torch.set_grad_enabled(torch.is_grad_enabled() and self._fit_mode):
            # One concatenated forward gives both time slices the same batch statistics.
            fmap = encode_obs(torch.cat((obs, next_obs), dim=0), self.encoder)
            projected = self.projector(fmap.mean(dim=(-2, -1)))
            z, z_next = projected.chunk(2, dim=0)
            z_pred = self.predictor(torch.cat((z, F.one_hot(action, 5).float()), dim=-1))
            error = torch.linalg.vector_norm(z_pred-z_next, dim=-1)
        return WorldPrediction(z, z_next, z_pred, error)

    def curiosity(self, obs: Tensor, next_obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        if self._fit_mode or self.training or self.encoder.training:
            raise RuntimeError("curiosity rewards require a frozen world model")
        prediction = self.transition(obs, next_obs, action)
        error = prediction.error.detach()
        return error.log1p(), error

    def commit_fit(self) -> tuple[int, int]:
        """Stage versions in memory after a complete fit; delivery 5 persists atomically.

        A caller must not enter X until its phase checkpoint commits successfully.
        No version is advanced by individual optimizer updates or by a failed fit.
        """
        if not self._fit_complete or self._fit_mode:
            raise RuntimeError("only a successfully completed, frozen fit can be published")
        self.encoder.encoder_version.add_(1)
        self.world_model_version.add_(1)
        self._fit_complete = False
        return int(self.encoder.encoder_version), int(self.world_model_version)
