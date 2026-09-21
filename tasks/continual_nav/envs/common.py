"""Shared pixel conversion and action validation; no task-specific policy inputs."""
import numpy as np
from PIL import Image

from ..contracts import ImageArray


def pixels(rgb: ImageArray) -> ImageArray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("expected uint8 HWC RGB")
    # Nearest-neighbor resizing preserves categorical map colors and tile edges.
    image = np.asarray(Image.fromarray(rgb).resize((84, 84), Image.Resampling.NEAREST))
    return np.ascontiguousarray(image.transpose(2, 0, 1))


def validate_action(action: int) -> int:
    # bool and float must not silently become legitimate action indices.
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
        raise ValueError("action must be an integer in [0,5)")
    if not 0 <= action < 5:
        raise ValueError("action must be an integer in [0,5)")
    return int(action)
