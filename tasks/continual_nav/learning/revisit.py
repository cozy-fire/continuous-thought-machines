"""Per-episode raw-pixel revisit penalties with exact temporal distances."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..config import Config


def _identity(info: dict) -> tuple[str, object, int]:
    task = info["task_key"]
    map_id = info["map_sha256"] if task == "maze_medium" else info["episode_seed"]
    return task, map_id, int(info["episode_id"])


class VisualRevisit:
    def __init__(self, config: Config, device: torch.device):
        self.length = config.exploration.history_length
        self.threshold = config.exploration.similarity_threshold
        self.floor = config.exploration.min_gap_weight
        self.device = device
        self.history: Tensor | None = None
        self.times: Tensor | None = None
        self.steps: Tensor | None = None
        self.identities: list[tuple[str, object, int]] = []

    def _unit(self, images: np.ndarray) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 84, 84) or images.dtype != np.uint8:
            raise ValueError("revisit images must be uint8 [B,3,84,84]")
        flat = torch.from_numpy(images.copy()).to(self.device, dtype=torch.float32).flatten(1)
        return flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def reset(self, images: np.ndarray, infos: list[dict]) -> None:
        unit = self._unit(images)
        if len(infos) != len(unit):
            raise ValueError("one initial identity is required per environment slot")
        self.history = unit[:, None].expand(-1, self.length, -1).clone()
        self.times = torch.arange(1-self.length, 1, device=self.device).expand(len(unit), -1).clone()
        self.steps = torch.zeros(len(unit), device=self.device, dtype=torch.int64)
        self.identities = [_identity(info) for info in infos]

    @torch.no_grad()
    def score(self, final_images: np.ndarray, infos: list[dict], done: np.ndarray,
              reset_images: np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.history is None or self.times is None or self.steps is None:
            raise RuntimeError("revisit history must be initialized before scoring")
        b = len(self.steps)
        if len(infos) != b or done.shape != (b,) or reset_images.shape != final_images.shape:
            raise ValueError("revisit step batch mismatch")
        if any(_identity(info) != self.identities[slot] for slot, info in enumerate(infos)):
            raise ValueError("revisit history crossed an episode or map boundary")
        query = self._unit(final_images)
        similarity = torch.einsum("bkd,bd->bk", self.history, query).clamp(-1., 1.)
        # Identical nonzero raw frames must yield exactly one despite float32
        # reduction error; zero-norm frames retain the specified zero similarity.
        identical = (self.history == query[:, None]).all(dim=-1) & (query.square().sum(dim=-1)[:, None] > 0)
        similarity = torch.where(identical, 1., similarity)
        now = self.steps + 1
        gap = now[:, None] - self.times
        if bool(((gap < 1) | (gap > self.length)).any()):
            raise RuntimeError("revisit history has an invalid temporal gap")
        strength = ((similarity-self.threshold)/(1-self.threshold)).clamp(0., 1.)
        strength = torch.where(similarity >= 1., 1., strength)
        weight = self.floor + (1-self.floor)*(self.length-gap)/(self.length-1)
        penalty, index = (strength*weight).max(dim=1)
        max_similarity = similarity.max(dim=1).values
        matched_similarity = similarity.gather(1, index[:, None]).squeeze(1)
        matched_gap = gap.gather(1, index[:, None]).squeeze(1)
        reward = -penalty
        # Query before insertion: an unchanged post-action frame matches age one,
        # while the autoreset frame never enters the preceding episode's history.
        rows = torch.arange(b, device=self.device)
        column = (now-1) % self.length
        self.history[rows, column] = query
        self.times[rows, column] = now
        self.steps = now
        for slot, ended in enumerate(done):
            if ended:
                reset = self._unit(reset_images[slot:slot+1])[0]
                self.history[slot] = reset[None].expand(self.length, -1)
                self.times[slot] = torch.arange(1-self.length, 1, device=self.device)
                self.steps[slot] = 0
                self.identities[slot] = _identity(infos[slot]["reset_info"])
        return reward.cpu(), max_similarity.cpu(), matched_similarity.cpu(), matched_gap.cpu()
