"""Finite-window RGB CTM and the frozen, checkpoint-compatible Dual teacher."""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math
import json

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from models.modules import SuperLinear, Squeeze
from models.resnet import prepare_resnet_backbone
from .config import MODEL_SPEC, TICKS


@dataclass
class State:
    pre: Tensor
    post: Tensor


@dataclass
class DualState:
    kb: State
    active: State


def state_map(state: State | DualState, operation) -> State | DualState:
    if isinstance(state, DualState):
        return DualState(state_map(state.kb, operation), state_map(state.active, operation))
    return State(operation(state.pre), operation(state.post))


def detached(state: State | DualState) -> State | DualState:
    return state_map(state, lambda t: t.detach().clone())


def select(state: State | DualState, indices: Tensor) -> State | DualState:
    return state_map(state, lambda t: t.index_select(0, indices.to(t.device)))


def pack_state(state: State | DualState) -> dict:
    if isinstance(state, DualState):
        return {'kb': pack_state(state.kb), 'active': pack_state(state.active)}
    return {'pre': state.pre.detach().cpu().clone(), 'post': state.post.detach().cpu().clone()}


def unpack_state(value: dict, device: str | torch.device) -> State | DualState:
    if 'kb' in value:
        return DualState(unpack_state(value['kb'], device), unpack_state(value['active'], device))
    return State(value['pre'].clone().to(device), value['post'].clone().to(device))


def reset(state: State, starts: Tensor, initial: State) -> State:
    if starts.shape != (state.pre.shape[0],) or starts.dtype != torch.bool:
        raise ValueError('episode_start must be bool [B]')
    mask = starts[:, None, None]
    return State(torch.where(mask, initial.pre, state.pre), torch.where(mask, initial.post, state.post))


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
    def __init__(self):
        super().__init__()
        self.query_input = nn.Linear(528, 128)
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


class VisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = prepare_resnet_backbone('resnet34-2', norm_layer=partial(nn.GroupNorm, 32))

    def forward(self, rgb: Tensor) -> Tensor:
        return self.backbone(rgb.float() / 255.)


def actor() -> nn.Sequential:
    result = nn.Sequential(nn.Linear(528, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.)
            nn.init.zeros_(layer.bias)
    return result


def validate_rgb(rgb: Tensor, rank: int) -> None:
    if rgb.dtype != torch.uint8 or rgb.ndim != rank or tuple(rgb.shape[-3:]) != (3, 84, 84) or min(rgb.shape) < 1:
        raise ValueError('policy expects nonempty uint8 RGB with trailing shape [3,84,84]')


class Policy(nn.Module):
    def __init__(self, encoder_chunk: int = 32, checkpoint_encoder: bool = True):
        super().__init__()
        self.encoder, self.controller = VisionEncoder(), Controller()
        self.actor = actor()
        self.encoder_chunk, self._frozen = encoder_chunk, False
        self.checkpoint_encoder = checkpoint_encoder

    def train(self, mode: bool = True):
        return super().train(False if self._frozen else mode)

    def initial_state(self, batch: int) -> State:
        return self.controller.initial_state(batch)

    def encode(self, rgb: Tensor, chunk: int | None = None) -> Tensor:
        use_checkpoint = self.checkpoint_encoder and torch.is_grad_enabled() and not self._frozen
        return torch.cat([checkpoint(self.encoder,part,use_reentrant=False) if use_checkpoint else self.encoder(part)
                          for part in rgb.split(chunk or self.encoder_chunk)])

    def advance(self, features: Tensor, state: State, starts: Tensor, ticks: int, sync) -> tuple[Tensor, State]:
        state = reset(state, starts, self.initial_state(len(features)))
        prepared = self.controller.attention.prepare(features)
        for _ in range(ticks):
            state, _ = self.controller.tick(prepared, state, sync[0])
        return self.actor(self.controller.out_sync(state.post, sync[1])), state

    def step(self, rgb: Tensor, state: State, starts: Tensor, *, task: str) -> tuple[Tensor, State]:
        validate_rgb(rgb, 4)
        return self.advance(self.encode(rgb), state, starts, TICKS[task], self.controller.prepare_sync())

    def sequence(self, rgb: Tensor, state: State, starts: Tensor, *, task: str, encoder_chunk: int | None = None) -> tuple[Tensor, State]:
        validate_rgb(rgb, 5)
        length, batch = rgb.shape[:2]
        if not 1 <= length <= 50 or starts.shape != (length, batch) or starts.dtype != torch.bool:
            raise ValueError('invalid sequence length or reset mask')
        features = self.encode(rgb.flatten(0, 1), encoder_chunk).reshape(length, batch, 128, 21, 21)
        sync = self.controller.prepare_sync()  # Differentiable, local to this forward.
        outputs = []
        for t in range(length):
            logits, state = self.advance(features[t], state, starts[t], TICKS[task], sync)
            outputs.append(logits)
        return torch.stack(outputs), state


def frozen_copy(policy: Policy) -> Policy:
    result = deepcopy(policy)
    result._frozen = True
    return result.requires_grad_(False).eval()


class LateralAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.projection = nn.Linear(512, 512, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, activation: Tensor) -> Tensor:
        return self.gate.tanh() * self.projection(self.norm(activation.detach()))


class DualTeacher(nn.Module):
    """Inference only: the loaded old KB is never replaced by the new student."""
    def __init__(self, encoder_chunk: int = 32):
        super().__init__()
        self.kb, self.active = Policy(encoder_chunk), Policy(encoder_chunk)
        self.adapter = LateralAdapter()
        self.register_buffer('kb_ready', torch.tensor(True, dtype=torch.bool))
        self.requires_grad_(False).eval()

    def initial_state(self, batch: int) -> DualState:
        return DualState(self.kb.initial_state(batch), self.active.initial_state(batch))

    @torch.no_grad()
    def step(self, rgb: Tensor, state: DualState, starts: Tensor, *, task: str) -> tuple[Tensor, DualState]:
        validate_rgb(rgb, 4)
        kb_state = reset(state.kb, starts, self.kb.initial_state(len(rgb)))
        active_state = reset(state.active, starts, self.active.initial_state(len(rgb)))
        kb_features, active_features = self.kb.encode(rgb), self.active.encode(rgb)
        kb_attention = self.kb.controller.attention.prepare(kb_features)
        active_attention = self.active.controller.attention.prepare(active_features)
        kb_sync, active_sync = self.kb.controller.prepare_sync(), self.active.controller.prepare_sync()
        for _ in range(TICKS[task]):
            kb_state, post = self.kb.controller.tick(kb_attention, kb_state, kb_sync[0])
            active_state, _ = self.active.controller.tick(active_attention, active_state, active_sync[0], self.adapter(post))
        logits = self.active.actor(self.active.controller.out_sync(active_state.post, active_sync[1]))
        return logits, DualState(kb_state, active_state)


def load_teacher(checkpoint, device: str, encoder_chunk: int = 32) -> DualTeacher:
    artifact = torch.load(checkpoint, map_location='cpu', weights_only=True)
    config = json.loads(json.dumps(artifact['config']))
    if artifact.get('policy_type') != 'dual' or any(config.get(k) != v for k, v in MODEL_SPEC.items()):
        raise ValueError('teacher model/input protocol mismatch')
    if not bool(artifact['state_dict'].get('kb_ready', False)):
        raise ValueError('teacher must retain an enabled old KB')
    teacher = DualTeacher(encoder_chunk)
    teacher.load_state_dict(artifact['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in teacher.parameters()):
        raise ValueError('nonfinite teacher weights')
    return teacher.to(device).requires_grad_(False).eval()
