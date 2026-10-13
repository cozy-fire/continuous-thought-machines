"""Frozen complete CTM Dual teacher, independent of the student's architecture."""
import json
import torch
from torch import Tensor, nn
from ...config import CTM_MODEL_SPEC, TICKS
from ..common.state import DualState, reset
from ..common.vision import validate_rgb
from .policy import Policy


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
    if artifact.get('policy_type') != 'dual' or any(config.get(k) != v for k, v in CTM_MODEL_SPEC.items()):
        raise ValueError('teacher model/input protocol mismatch')
    if not bool(artifact['state_dict'].get('kb_ready', False)):
        raise ValueError('teacher must retain an enabled old KB')
    teacher = DualTeacher(encoder_chunk)
    teacher.load_state_dict(artifact['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in teacher.parameters()):
        raise ValueError('nonfinite teacher weights')
    return teacher.to(device).requires_grad_(False).eval()
