"""Online visual LSTM: hidden queries the current image at every internal tick."""
import math
from contextlib import nullcontext
import torch
from torch import nn
from ..common.policy import VisualPolicy
from ..common.state import LSTMState, reset
from ..common.vision import SpatialAttention, actor


class Controller(nn.Module):
    def __init__(self, hidden: int = 512, layers: int = 2):
        super().__init__()
        self.hidden,self.layers = hidden,layers
        self.attention = SpatialAttention(query_dim=hidden)
        self.lstm = nn.LSTM(128,hidden,layers,batch_first=True,dropout=0.)
        scale = math.sqrt(1/hidden)
        self.start_hidden = nn.Parameter(torch.empty(layers,hidden).uniform_(-scale,scale))
        self.start_cell = nn.Parameter(torch.empty(layers,hidden).uniform_(-scale,scale))

    def initial_state(self, batch: int) -> LSTMState:
        return LSTMState(self.start_hidden.unsqueeze(0).expand(batch,-1,-1).clone(),
                         self.start_cell.unsqueeze(0).expand(batch,-1,-1).clone())

    def tick(self, prepared, state: LSTMState) -> LSTMState:
        attended = self.attention.forward_prepared(prepared,state.h[:,-1])
        # Fisher scores require gradients in eval mode. cuDNN's inference RNN
        # does not retain its backward workspace; use the native differentiable
        # operator for this path without changing F's policy mode or dropout.
        context = (torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False)
                   if attended.is_cuda and torch.is_grad_enabled() and not self.lstm.training else nullcontext())
        with context:
            _,(h,c) = self.lstm(attended.unsqueeze(1),
                (state.h.transpose(0,1).contiguous(),state.c.transpose(0,1).contiguous()))
        return LSTMState(h.transpose(0,1).contiguous(),c.transpose(0,1).contiguous())


class Policy(VisualPolicy):
    def __init__(self, encoder_chunk: int = 32, checkpoint_encoder: bool = True,
                 hidden: int = 512, layers: int = 2):
        super().__init__(encoder_chunk,checkpoint_encoder)
        self.controller = Controller(hidden,layers)
        self.actor = actor(hidden)

    def initial_state(self, batch: int) -> LSTMState:
        return self.controller.initial_state(batch)

    def advance(self, features, state: LSTMState, starts, ticks: int, recurrence=None):
        state = reset(state,starts,self.initial_state(len(features)))
        prepared = self.controller.attention.prepare(features)
        for _ in range(ticks):
            state = self.controller.tick(prepared,state)
        return self.actor(state.h[:,-1]),state
