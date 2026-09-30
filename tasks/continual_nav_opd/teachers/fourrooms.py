"""Strict original CTM teacher restoration; learned state, no prefix masking."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from models.ctm_rl import ContinuousThoughtMachineRL
from ..contracts import CTMState


def map_probabilities(native):
    if native.ndim != 2 or native.shape[1] != 7:
        raise ValueError("native probabilities must be [B,7]")
    if not torch.isfinite(native).all() or (native < 0).any() or not torch.allclose(native.sum(-1), torch.ones_like(native[:, 0]), atol=1e-5, rtol=1e-4):
        raise ValueError("invalid native probability distribution")
    wait = native[:, 3:].sum(-1, keepdim=True) / 2
    return torch.cat((native[:, :3], wait, wait), dim=-1)


class NativeAgent(nn.Module):
    """The original Agent module layout, including its checkpoint-only critic."""
    def __init__(self, args):
        super().__init__()
        self.recurrent_model = ContinuousThoughtMachineRL(
            iterations=args.iterations, d_model=args.d_model, d_input=args.d_input,
            n_synch_out=args.n_synch_out, synapse_depth=args.synapse_depth,
            memory_length=args.memory_length, deep_nlms=args.deep_memory,
            memory_hidden_dims=args.memory_hidden_dims, do_layernorm_nlm=args.do_normalisation,
            backbone_type="navigation-backbone", prediction_reshaper=[-1],
            dropout=args.dropout, neuron_select_type=args.neuron_select_type)
        size = self.recurrent_model.synch_representation_size_out
        def head(out):
            return nn.Sequential(nn.Linear(size, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, out))
        self.actor, self.critic = head(7), head(1)


@dataclass(frozen=True)
class FourRoomsTarget:
    native_probabilities: torch.Tensor
    probabilities: torch.Tensor
    valid_target: torch.Tensor
    state: CTMState


class FourRoomsTeacher:
    def __init__(self, checkpoint, device="cpu"):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.source_snapshot_id = file_sha256(self.checkpoint)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if "args" not in payload or "model_state_dict" not in payload:
            raise ValueError("native checkpoint must contain args and model_state_dict")
        args = payload["args"]
        required = {"model_type": "ctm", "env_id": "MiniGrid-FourRooms-v0",
                    "iterations": 2, "memory_length": 40, "continuous_state_trace": True,
                    "neuron_select_type": "first-last", "max_environment_steps": 300}
        for name, value in required.items():
            if getattr(args, name, None) != value:
                raise ValueError(f"incompatible native teacher argument: {name}")
        self.args, self.device = args, torch.device(device)
        # Restoration must not consume the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            self.model = NativeAgent(args)
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(self.device).eval().requires_grad_(False)
        for parameter in self.model.parameters():
            parameter.grad = None
        self.metadata = {"checkpoint": str(self.checkpoint), "sha256": self.source_snapshot_id,
                         "original_args": vars(args), "key_mapping": "identity_strict",
                         "input_protocol": "native_uint8_B_7_7_3_no_normalization",
                         "output_protocol": "seven_softmax_then_wait_mass_split_to_five",
                         "state_protocol": "learned_initial_episode_reset_continuous_every_observation"}

    @torch.no_grad()
    def initial_state(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        model = self.model.recurrent_model
        return CTMState(model.start_trace.unsqueeze(0).expand(batch_size, -1, -1).clone(),
                        model.start_activated_trace.unsqueeze(0).expand(batch_size, -1, -1).clone())

    @torch.no_grad()
    def predict(self, observations, state, episode_start):
        if not isinstance(observations, np.ndarray) or observations.dtype != np.uint8 or observations.ndim != 4 or observations.shape[1:] != (7, 7, 3) or len(observations) == 0:
            raise ValueError("FourRooms teacher expects nonempty native uint8 [B,7,7,3]")
        batch = len(observations)
        reset = np.asarray(episode_start)
        if reset.dtype != np.bool_ or reset.shape != (batch,):
            raise ValueError("episode_start must be bool [B]")
        shape = (batch, self.args.d_model, self.args.memory_length)
        if any(x.shape != shape or x.device != self.device or x.dtype != torch.float32 or not torch.isfinite(x).all() for x in (state.pre, state.post)):
            raise ValueError("invalid teacher state shape/device/dtype/value")
        initial = self.initial_state(batch)
        mask = torch.as_tensor(reset, device=self.device).view(-1, 1, 1)
        # Reset only true episode boundaries. Window boundaries carry both learned traces.
        pre = torch.where(mask, initial.pre, state.pre)
        post = torch.where(mask, initial.post, state.post)
        obs = torch.as_tensor(observations, device=self.device)
        synch, new_state = self.model.recurrent_model(obs, (pre, post))
        logits = self.model.actor(synch)
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite teacher logits")
        native = torch.softmax(logits, dim=-1)
        return FourRoomsTarget(native, map_probabilities(native),
                               torch.ones(batch, dtype=torch.bool, device=self.device),
                               CTMState(new_state[0].detach(), new_state[1].detach()))

    def parameter_hash(self):
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()
