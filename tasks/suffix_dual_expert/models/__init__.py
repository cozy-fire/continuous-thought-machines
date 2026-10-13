"""Explicit two-model construction; shared training has no model registry."""
from .common.policy import VisualPolicy, frozen_copy
from .common.state import State, detached, select, pack_state, unpack_state
from .ctm.policy import Policy as CTMPolicy
from .ctm.teacher import DualTeacher, load_teacher
from .lstm.policy import Policy as LSTMPolicy


def build_policy(config) -> VisualPolicy:
    if config.model_type == 'ctm':
        return CTMPolicy(config.encoder_chunk,config.checkpoint_encoder)
    if config.model_type == 'lstm':
        return LSTMPolicy(config.encoder_chunk,config.checkpoint_encoder,config.lstm_hidden,config.lstm_layers)
    raise ValueError('model_type must be ctm or lstm')


def state_kind(policy) -> str:
    if isinstance(policy,DualTeacher):
        return 'ctm_dual'
    if isinstance(policy,CTMPolicy):
        return 'ctm'
    if isinstance(policy,LSTMPolicy):
        return 'lstm'
    raise TypeError('unknown policy architecture')
