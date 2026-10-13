"""CTM recurrence; checkpoint keys and numerical operations retain the original layout."""
import math
import torch
from torch import Tensor, nn
from models.modules import SuperLinear, Squeeze
from ..common.state import CTMState as State, reset
from ..common.vision import SpatialAttention, actor
from ..common.policy import VisualPolicy


class WindowSynchrony(nn.Module):
    def __init__(self, start: int):
        super().__init__()
        pairs = torch.triu_indices(32, 32)
        self.register_buffer('left', pairs[0] + start)
        self.register_buffer('right', pairs[1] + start)
        self.register_buffer('ages', torch.arange(39, -1, -1, dtype=torch.float32))
        self.decay = nn.Parameter(torch.zeros(528))

    def prepare(self) -> tuple[Tensor, Tensor]:
        decay = torch.where(self.decay < 0, torch.zeros_like(self.decay),
                            torch.where(self.decay > 4, torch.full_like(self.decay, 4), self.decay))
        weights = torch.exp(-decay[:, None] * self.ages[None, :])
        return weights, weights.sum(-1).sqrt()

    def forward(self, post: Tensor, prepared=None) -> Tensor:
        weights, denominator = self.prepare() if prepared is None else prepared
        return (post[:, self.left] * post[:, self.right] * weights).sum(-1) / denominator


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        scale = math.sqrt(1 / 552)
        self.start_pre = nn.Parameter(torch.empty(512, 40).uniform_(-scale, scale))
        self.start_post = nn.Parameter(torch.empty(512, 40).uniform_(-scale, scale))
        self.out_sync, self.action_sync = WindowSynchrony(0), WindowSynchrony(480)
        self.attention = SpatialAttention()
        self.synapse = nn.Sequential(nn.Linear(640, 1024), nn.GLU(), nn.LayerNorm(512),
                                     nn.Linear(512, 1024), nn.GLU(), nn.LayerNorm(512))
        self.nlm = nn.Sequential(SuperLinear(40, 32, 512), nn.GLU(),
                                 SuperLinear(16, 2, 512), nn.GLU(), Squeeze(-1))

    def initial_state(self, batch: int) -> State:
        return State(self.start_pre.unsqueeze(0).expand(batch, -1, -1).clone(),
                     self.start_post.unsqueeze(0).expand(batch, -1, -1).clone())

    def prepare_sync(self):
        return self.action_sync.prepare(), self.out_sync.prepare()

    def tick(self, prepared, state: State, sync, lateral: Tensor | None = None) -> tuple[State, Tensor]:
        attended = self.attention.forward_prepared(prepared, self.action_sync(state.post, sync))
        previous = state.post[:, :, -1]
        if lateral is not None:
            previous = previous + lateral
        new_pre = self.synapse(torch.cat((attended, previous), -1))
        pre = torch.cat((state.pre[:, :, 1:], new_pre.unsqueeze(-1)), -1)
        new_post = self.nlm(pre)
        post = torch.cat((state.post[:, :, 1:], new_post.unsqueeze(-1)), -1)
        return State(pre, post), new_post


class Policy(VisualPolicy):
    def __init__(self, encoder_chunk: int = 32, checkpoint_encoder: bool = True):
        super().__init__(encoder_chunk, checkpoint_encoder)
        self.controller = Controller()
        self.actor = actor()

    def prepare_recurrence(self):
        return self.controller.prepare_sync()

    def initial_state(self, batch: int) -> State:
        return self.controller.initial_state(batch)

    def advance(self, features: Tensor, state: State, starts: Tensor, ticks: int, sync) -> tuple[Tensor, State]:
        state = reset(state, starts, self.initial_state(len(features)))
        prepared = self.controller.attention.prepare(features)
        for _ in range(ticks):
            state, _ = self.controller.tick(prepared, state, sync[0])
        return self.actor(self.controller.out_sync(state.post, sync[1])), state
