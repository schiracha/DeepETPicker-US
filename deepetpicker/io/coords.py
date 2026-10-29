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


def read_coords_scored(path):
    """``(coords, scores)``: like ``read_coords`` but also accepts the five-column
    form predict writes to ``coords_with_area/``: ``class_id x y z score``.
    ``scores`` is an (N,) float array, or None when the file has no fifth column.
    (The column is DeepETPicker's confidence-like value for the pick: the peak of
    the smoothed 0/1 prediction mask, 1.0 for a solidly predicted particle; see
    ``deepetpicker.io.relion.SCORE_COLUMN``.)"""
    path = Path(path)
    rows, scores = [], []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) == 3:
            parts = ["1"] + parts
        if len(parts) not in (4, 5):
            raise ValueError(f"{path}:{lineno}: expected 3, 4 or 5 columns, got {len(parts)}: {line!r}")
        values = [float(p) for p in parts]
        rows.append(values[:4])
        if len(values) == 5:
            scores.append(values[4])
    coords = np.asarray(rows, dtype=np.float64).reshape(-1, 4)
    if not scores:
        return coords, None
    if len(scores) != len(rows):
        raise ValueError(f"{path}: some rows have a score column and some do not")
    return coords, np.asarray(scores, dtype=np.float64)


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
