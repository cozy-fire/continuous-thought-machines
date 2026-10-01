"""Complete RGB actor policies with independent vision and same-tick laterals."""
from __future__ import annotations
from copy import deepcopy
from typing import TypeVar
import torch
from torch import Tensor, nn
from ..config import Config, validate_config
from ..contracts import CTMState, DualState, PolicyState, PolicySequenceOutput
from .ctm import Controller
from .vision import VisionEncoder
from .state import reset_state, select_state, replace_slots


def _head(input_dim: int) -> nn.Sequential:
    result = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64),
                           nn.ReLU(), nn.Linear(64, 5))
    for layer in result:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.zeros_(layer.bias)
    return result


class StandalonePolicy(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        validate_config(config)
        self.config = config
        self._frozen = False
        self.encoder = VisionEncoder(config)
        self.controller = Controller(config)
        self.actor = _head(self.controller.out_sync.output_dim)

    def train(self, mode: bool = True) -> StandalonePolicy:
        super().train(False if self._frozen else mode)
        return self

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> CTMState:
        return self.controller.initial_state(batch, device)

    def _encode(self, rgb: Tensor, chunk_images: int | None = None) -> tuple[Tensor, ...]:
        microbatch = self.config.optimization.encoder_microbatch_images if chunk_images is None else chunk_images
        if not isinstance(microbatch, int) or microbatch < 1 or microbatch > self.config.optimization.encoder_microbatch_images:
            raise ValueError("encoder chunk must be within the configured image microbatch")
        return (torch.cat([self.encoder(part) for part in rgb.split(microbatch)], dim=0),)

    def _step_features(self, features: tuple[Tensor, ...], state: CTMState,
                       episode_start: Tensor, prepared_sync=None) -> tuple[Tensor, CTMState]:
        if not isinstance(state, CTMState):
            raise TypeError("StandalonePolicy requires CTMState")
        state = reset_state(state, episode_start, self.initial_state(features[0].shape[0]))
        prepared_attention = None
        if self.config.optimization.e3_cache:
            prepared_attention = self.controller.attention.prepare(features[0])
            if prepared_sync is None:
                prepared_sync = self._prepare_sequence()
        for _ in range(self.controller.ticks):
            state, _ = self.controller.tick(features[0], state, attention_prepared=prepared_attention,
                                            sync_prepared=prepared_sync[0] if prepared_sync else None)
        return self.actor(self.controller.readout(state, prepared_sync[1] if prepared_sync else None)), state

    def _prepare_sequence(self):
        return self.controller.prepare_sync() if self.config.optimization.e3_cache else None

    def step(self, rgb: Tensor, state: CTMState, episode_start: Tensor) -> tuple[Tensor, CTMState]:
        _validate_rgb(rgb, 4)
        return self._step_features(self._encode(rgb), state, episode_start)

    def sequence(self, rgb: Tensor, initial_state: CTMState, episode_start: Tensor,
                 valid_mask: Tensor | None = None, *, encoder_chunk_images: int | None = None) -> PolicySequenceOutput:
        return _sequence(self, rgb, initial_state, episode_start, valid_mask, encoder_chunk_images=encoder_chunk_images)


class LateralAdapter(nn.Module):
    def __init__(self, width: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, activation: Tensor) -> Tensor:
        # Detach only the KB activation. Gate, norm and projection remain trainable.
        return self.gate.tanh() * self.projection(self.norm(activation.detach()))


class DualPolicy(nn.Module):
    def __init__(self, config: Config, kb: StandalonePolicy, *, kb_ready: bool = False):
        super().__init__()
        validate_config(config)
        if type(kb) is not StandalonePolicy or kb.config != config:
            raise ValueError("KB must be a complete v3 StandalonePolicy with matching configuration")
        self.config, self._frozen = config, False
        # P owns an isolated old KB; never freeze or reuse the live Compress student in place.
        self.kb = frozen_copy(kb)
        self.active = StandalonePolicy(config).to(next(kb.parameters()).device)
        self.active.encoder.load_state_dict(self.kb.encoder.state_dict(), strict=True)
        self.adapter = LateralAdapter(config.ctm.d_model).to(next(kb.parameters()).device)
        self._kb_ready_enabled = bool(kb_ready)
        self.register_buffer("kb_ready", torch.tensor(kb_ready, dtype=torch.bool,
                                                    device=next(kb.parameters()).device))

    def set_kb_ready(self, ready: bool) -> None:
        if self._frozen:
            raise ValueError("cannot change a frozen inference snapshot")
        # Keep the persistent tensor for artifact compatibility; use the setter for changes.
        self.kb_ready.fill_(bool(ready))
        self._kb_ready_enabled = bool(ready)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Loading is a lifecycle boundary, never an observation-time device scalar read.
        self._kb_ready_enabled = bool(self.kb_ready.item())

    def train(self, mode: bool = True) -> DualPolicy:
        super().train(False if self._frozen else mode)
        self.kb.requires_grad_(False).eval()
        return self

    def initial_state(self, batch: int, device: torch.device | str | None = None) -> DualState:
        with torch.no_grad():
            kb = self.kb.initial_state(batch, device)
        return DualState(kb, self.active.initial_state(batch, device))

    def _encode(self, rgb: Tensor, chunk_images: int | None = None) -> tuple[Tensor, ...]:
        with torch.no_grad():
            kb_features = self.kb._encode(rgb, chunk_images)[0]
        active_features = self.active._encode(rgb, chunk_images)[0]
        return kb_features, active_features

    def _step_features(self, features: tuple[Tensor, ...], state: DualState,
                       episode_start: Tensor, prepared_sync=None) -> tuple[Tensor, DualState]:
        if not isinstance(state, DualState):
            raise TypeError("DualPolicy requires DualState")
        state = reset_state(state, episode_start, self.initial_state(features[0].shape[0]))
        kb_state, active_state = state.kb, state.active
        ready = self._kb_ready_enabled
        kb_attention = active_attention = None
        if self.config.optimization.e3_cache:
            with torch.no_grad():
                kb_attention = self.kb.controller.attention.prepare(features[0])
            active_attention = self.active.controller.attention.prepare(features[1])
            if prepared_sync is None:
                prepared_sync = self._prepare_sequence()
        for _ in range(self.active.controller.ticks):
            # KB advances FIRST: lateral uses its new post activation from this same tick.
            # Its own encoder is frozen; Active features never enter the KB path.
            with torch.no_grad():
                kb_state, kb_post = self.kb.controller.tick(features[0], kb_state, attention_prepared=kb_attention,
                    sync_prepared=prepared_sync[0][0] if prepared_sync else None)
            lateral = self.adapter(kb_post) if ready else None
            active_state, _ = self.active.controller.tick(features[1], active_state, lateral, attention_prepared=active_attention,
                sync_prepared=prepared_sync[1][0] if prepared_sync else None)
        logits = self.active.actor(self.active.controller.readout(active_state, prepared_sync[1][1] if prepared_sync else None))
        return logits, DualState(kb_state, active_state)

    def _prepare_sequence(self):
        if not self.config.optimization.e3_cache:
            return None
        with torch.no_grad():
            kb_sync = self.kb.controller.prepare_sync()
        return kb_sync, self.active.controller.prepare_sync()

    def step(self, rgb: Tensor, state: DualState, episode_start: Tensor) -> tuple[Tensor, DualState]:
        _validate_rgb(rgb, 4)
        return self._step_features(self._encode(rgb), state, episode_start)

    def sequence(self, rgb: Tensor, initial_state: DualState, episode_start: Tensor,
                 valid_mask: Tensor | None = None, *, encoder_chunk_images: int | None = None) -> PolicySequenceOutput:
        return _sequence(self, rgb, initial_state, episode_start, valid_mask, encoder_chunk_images=encoder_chunk_images)


def _validate_rgb(rgb: Tensor, rank: int) -> None:
    if rgb.dtype != torch.uint8 or rgb.ndim != rank or tuple(rgb.shape[-3:]) != (3, 84, 84) or any(size == 0 for size in rgb.shape):
        raise ValueError("policy input must be nonempty uint8 RGB with trailing [3,84,84]")


def _sequence(policy: StandalonePolicy | DualPolicy, rgb: Tensor, state: PolicyState,
              episode_start: Tensor, valid_mask: Tensor | None, *, force_generic: bool = False, encoder_chunk_images: int | None = None) -> PolicySequenceOutput:
    _validate_rgb(rgb, 5)
    length, batch = rgb.shape[:2]
    if not 1 <= length <= 50:
        raise ValueError("sequence must contain 1..50 new observations")
    dense = valid_mask is None
    if valid_mask is None:
        valid_mask = torch.ones_like(episode_start)
    for mask in (episode_start, valid_mask):
        if mask.shape != (length, batch) or mask.dtype != torch.bool or mask.device != rgb.device:
            raise ValueError("sequence masks must be bool [L,B] on the image device")
    if isinstance(policy, DualPolicy) != isinstance(state, DualState):
        raise TypeError("sequence initial_state must belong to the learning policy")
    # This cache belongs to THIS forward only. Its differentiable decay weights
    # accumulate all tick gradients; neither models nor collectors retain the graph.
    prepared_sync = policy._prepare_sequence()
    if not force_generic and (dense or bool(valid_mask.all())):
        # Full windows have no padding: avoid per-tick slot selection and state scatter.
        encoded = policy._encode(rgb.flatten(0, 1), encoder_chunk_images) if encoder_chunk_images is not None else policy._encode(rgb.flatten(0, 1))
        features = tuple(f.reshape(length, batch, *f.shape[1:]) for f in encoded)
        outputs = []
        for t in range(length):
            logits, state = policy._step_features(tuple(f[t] for f in features), state, episode_start[t], prepared_sync)
            outputs.append(logits)
        return PolicySequenceOutput(torch.stack(outputs), state)
    # Encode ONLY valid new observations. Microbatching changes image batching, not CTM time.
    positions = valid_mask.flatten().nonzero().flatten()
    if positions.numel():
        images = rgb.flatten(0, 1).index_select(0, positions)
        encoded = policy._encode(images, encoder_chunk_images) if encoder_chunk_images is not None else policy._encode(images)
        features = tuple(f.new_zeros(length * batch, *f.shape[1:]).index_copy(0, positions, f)
                         .reshape(length, batch, *f.shape[1:]) for f in encoded)
    else:
        features = ()
    outputs = []
    for t in range(length):
        indices = valid_mask[t].nonzero().flatten()
        logits = torch.zeros(batch, 5, dtype=torch.float32, device=rgb.device)
        if indices.numel():
            output, proposed = policy._step_features(tuple(f[t].index_select(0, indices) for f in features),
                                                     select_state(state, indices),
                                                     episode_start[t].index_select(0, indices), prepared_sync)
            state = replace_slots(state, indices, proposed)
            logits = logits.index_copy(0, indices, output)
        # Padding neither advances nor resets either column. No detach/burn-in occurs here.
        outputs.append(logits)
    return PolicySequenceOutput(torch.stack(outputs), state)


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def frozen_copy(module: ModuleT) -> ModuleT:
    snapshot = deepcopy(module)
    for child in snapshot.modules():
        if isinstance(child, (StandalonePolicy, DualPolicy)):
            child._frozen = True
    snapshot.requires_grad_(False).eval()
    for parameter in snapshot.parameters():
        parameter.grad = None
    return snapshot
