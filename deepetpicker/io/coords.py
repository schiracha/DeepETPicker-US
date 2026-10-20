"""Reading and writing DeepETPicker ``.coords`` files.

Native format: whitespace/tab separated ``class_id x y z`` in voxels of the
tomogram the picker ran on; a bare ``x y z`` file is class 1. Coordinates may
be fractional. Label generation needs integer voxel indices, so that step goes
through ``to_voxel_indices`` (rounding) rather than truncating.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_coords(path) -> np.ndarray:
    """Return an (N, 4) float array ``class_id, x, y, z``."""
    path = Path(path)
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) == 3:
            parts = ["1"] + parts
        if len(parts) != 4:
            raise ValueError(f"{path}:{lineno}: expected 3 or 4 columns, got {len(parts)}: {line!r}")
        rows.append([float(p) for p in parts])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def write_coords(path, coords: np.ndarray, integer: bool = False) -> None:
    """Write ``class_id x y z`` tab separated; ``integer`` rounds x y z."""
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c, x, y, z in coords:
            if integer:
                f.write(f"{int(c)}\t{int(np.rint(x))}\t{int(np.rint(y))}\t{int(np.rint(z))}\n")
            else:
                f.write(f"{int(c)}\t{x:.3f}\t{y:.3f}\t{z:.3f}\n")


def to_voxel_indices(xyz: np.ndarray) -> np.ndarray:
    """Round to the nearest voxel (the old import step truncated instead)."""
    return np.rint(np.asarray(xyz, dtype=np.float64)).astype(np.int64)
