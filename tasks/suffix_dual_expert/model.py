"""Compatibility imports for existing CTM weights, tooling and visualisations.

New shared training uses models.build_policy; model.Policy remains CTM-only.
"""
from .models.ctm.policy import Policy, Controller, WindowSynchrony
from .models.ctm.teacher import DualTeacher, LateralAdapter, load_teacher
from .models.common.state import (CTMState as State, DualState, state_map, detached,
                                  select, pack_state, unpack_state, reset)
from .models.common.vision import VisionEncoder, SpatialRoPE, SpatialAttention, actor, validate_rgb, rotate
from .models.common.policy import frozen_copy
