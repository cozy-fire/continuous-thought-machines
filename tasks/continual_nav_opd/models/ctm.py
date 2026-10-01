"""v3 CTM recurrence: the v2 finite-window mechanism, with v3 configuration."""
from __future__ import annotations
import math
from functools import lru_cache
import torch
from torch import Tensor, nn
from models.modules import SuperLinear, Squeeze
from ..config import Config, validate_config
from ..contracts import CTMState
from .rope import SpatialAttention

class WindowSynchrony(nn.Module):
    def __init__(self, neurons: int, memory: int, start: int):
        super().__init__()
        pairs = torch.triu_indices(neurons, neurons)
        self.register_buffer("left", pairs[0] + start)
        self.register_buffer("right", pairs[1] + start)
        self.register_buffer("ages", torch.arange(memory-1, -1, -1, dtype=torch.float32))
        self.decay = nn.Parameter(torch.zeros(pairs.shape[1]))
        self.output_dim = pairs.shape[1]

    def prepare(self) -> tuple[Tensor, Tensor]:
        # Keep boundary gradients explicit: torch.clamp has version-dependent
        # subgradients at 0 and 4, and decay is initialized exactly at 0.
        decay = torch.where(self.decay < 0, torch.zeros_like(self.decay),
                            torch.where(self.decay > 4, torch.full_like(self.decay, 4), self.decay))
        weights = torch.exp(-decay[:, None] * self.ages[None, :])
        return weights, weights.sum(-1).sqrt()

    def forward(self, post: Tensor, prepared: tuple[Tensor, Tensor] | None = None) -> Tensor:
        # Prepared weights retain gradients and must never survive an optimizer update.
        weights, denominator = self.prepare() if prepared is None else prepared
        products = post[:, self.left, :] * post[:, self.right, :]
        return (products * weights).sum(-1) / denominator


class Controller(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        validate_config(config)
        c = config.ctm
        self.d_model, self.memory, self.ticks = c.d_model, c.memory_length, c.ticks
        self.compile_mode = config.optimization.ctm_compile
        scale = math.sqrt(1 / (c.d_model + c.memory_length))
        self.start_pre = nn.Parameter(torch.empty(c.d_model, c.memory_length).uniform_(-scale, scale))
        self.start_post = nn.Parameter(torch.empty(c.d_model, c.memory_length).uniform_(-scale, scale))
        self.out_sync = WindowSynchrony(c.n_out, c.memory_length, 0)
        self.action_sync = WindowSynchrony(c.n_action, c.memory_length, c.d_model-c.n_action)
        self.attention = SpatialAttention(self.action_sync.output_dim, c.d_input,
                                          config.attention.heads, config.attention.rope_theta,
                                          config.attention.query_position)
        self.synapse = nn.Sequential(nn.Linear(c.d_input+c.d_model, 2*c.d_model), nn.GLU(),
                                     nn.LayerNorm(c.d_model), nn.Linear(c.d_model, 2*c.d_model),
                                     nn.GLU(), nn.LayerNorm(c.d_model))
        self.nlm = nn.Sequential(SuperLinear(c.memory_length, 2*c.nlm_hidden, c.d_model), nn.GLU(),
                                SuperLinear(c.nlm_hidden, 2, c.d_model), nn.GLU(), Squeeze(-1))

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> CTMState:
        if batch < 1:
            raise ValueError("batch must be positive")
        if device is not None and torch.device(device) != self.start_pre.device:
            raise ValueError("move the controller to the requested device before creating state")
        # Clone the expanded view so callers cannot alias trainable initial storage.
        return CTMState(self.start_pre.unsqueeze(0).expand(batch, -1, -1).clone(),
                        self.start_post.unsqueeze(0).expand(batch, -1, -1).clone())

    def prepare_sync(self):
        return self.action_sync.prepare(), self.out_sync.prepare()

    def tick(self, fmap: Tensor, state: CTMState, lateral: Tensor | None = None, *,
             attention_prepared=None, sync_prepared=None) -> tuple[CTMState, Tensor]:
        if fmap.ndim != 4 or tuple(fmap.shape[1:]) != (128, 21, 21):
            raise ValueError("fmap must be [B,128,21,21]")
        expected = (fmap.shape[0], self.d_model, self.memory)
        if state.pre.shape != expected or state.post.shape != expected:
            raise ValueError("invalid CTM state shape")
        if fmap.dtype != torch.float32 or state.pre.dtype != torch.float32 or state.post.dtype != torch.float32:
            raise ValueError("v3 CTM uses float32 features and state")
        if lateral is not None and lateral.shape != (fmap.shape[0], self.d_model):
            raise ValueError("lateral input must be [B,D]")
        k, v = self.attention.prepare(fmap) if attention_prepared is None else attention_prepared
        weights, denominator = self.action_sync.prepare() if sync_prepared is None else sync_prepared
        operation = Controller._tick_math if self.compile_mode == 'disabled' else compiled_tick(self.compile_mode)
        pre, post, new_post = operation(self, k, v, state.pre, state.post, lateral, weights, denominator)
        if self.compile_mode == 'reduce-overhead':
            # Recurrence outlives a CUDA graph invocation. Own its outputs outside the
            # compiled region so the next replay cannot overwrite a previous tick.
            pre, post, new_post = pre.clone(), post.clone(), new_post.clone()
        return CTMState(pre, post), new_post

    def _tick_math(self, k: Tensor, v: Tensor, pre: Tensor, post: Tensor,
                   lateral: Tensor | None, weights: Tensor, denominator: Tensor):
        # Pure tensor region: validation, state containers and timers stay outside compilation.
        attended = self.attention.forward_prepared((k,v), self.action_sync(post, (weights,denominator)))
        previous = post[:, :, -1]
        if lateral is not None:
            previous = previous + lateral
        new_pre = self.synapse(torch.cat((attended, previous), dim=-1))
        pre = torch.cat((pre[:, :, 1:], new_pre.unsqueeze(-1)), dim=-1)
        new_post = self.nlm(pre)
        post = torch.cat((post[:, :, 1:], new_post.unsqueeze(-1)), dim=-1)
        return pre, post, new_post

    def readout(self, state: CTMState, prepared=None) -> Tensor:
        return self.out_sync(state.post, prepared)


@lru_cache(maxsize=2)
def compiled_tick(mode: str):
    if mode not in ('default','reduce-overhead'):
        raise ValueError('unsupported CTM compile mode')
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError('CTM compilation requires suppress_errors=False; no eager fallback is allowed')
    # Compile an UNBOUND function, not a model-owned closure. Deep copies and snapshots
    # therefore keep independent parameter storage and the original state_dict namespace.
    return torch.compile(Controller._tick_math, backend='inductor', mode=mode, fullgraph=True, dynamic=False)
