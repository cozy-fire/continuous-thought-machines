"""Small synthetic maps; no downloads or access to the production dataset."""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image

from tasks.continual_nav.data.manifest import MazeEntry


def map_file(root: Path, index: int = 0, split: str = "train") -> MazeEntry:
    path = root / split / "0" / f"{index}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((19, 19, 3), dtype=np.uint8)
    row = 1 if split == "train" else 3
    rgb[row, 1:index+4] = 255
    rgb[row, 1] = (255, 0, 0)
    rgb[row, 2] = (0, 0, 255)
    rgb[row, index+3] = (0, 255, 0)
    Image.fromarray(rgb).save(path)
    return MazeEntry(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
