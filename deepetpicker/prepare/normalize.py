"""Tomogram standardization / normalization without loading the volume twice.

Same arithmetic as ``utils/normalization.py`` (which holds a float64 copy of
the whole tomogram, 8 bytes per voxel, per worker), done slab by slab on a
memory-mapped input and written to a memory-mapped output, and the voxel size
is carried over to the output header (the old code dropped it).
"""
from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np

KINDS = ("standardization", "normalization")


def _slabs(n, step):
    for z0 in range(0, n, step):
        yield slice(z0, min(z0 + step, n))


SLAB_BYTES = 64 * 2**20


def _slab_step(shape):
    return max(1, SLAB_BYTES // (8 * shape[1] * shape[2]))


def volume_stats(data, kind):
    """(shift, scale) so that ``(data - shift) / scale`` is the result."""
    step = _slab_step(data.shape)
    if kind == "standardization":
        total = sum(float(data[s].sum(dtype=np.float64)) for s in _slabs(data.shape[0], step))
        mean = total / data.size
        ss = sum(float(((data[s].astype(np.float64) - mean) ** 2).sum()) for s in _slabs(data.shape[0], step))
        std = (ss / data.size) ** 0.5
        if std == 0:
            raise ValueError("constant volume: cannot standardize")
        return mean, std
    if kind == "normalization":
        lo = min(float(data[s].min()) for s in _slabs(data.shape[0], step))
        hi = max(float(data[s].max()) for s in _slabs(data.shape[0], step))
        if hi == lo:
            raise ValueError("constant volume: cannot normalize")
        return lo, hi - lo
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def normalize_file(src, dst, kind="standardization"):
    """Write the standardized copy of ``src`` to ``dst`` (float32 MRC)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.mmap(src, mode="r", permissive=True) as m:
        data = m.data
        if data.ndim != 3:
            raise ValueError(f"{src}: expected a 3-D volume, got shape {data.shape}")
        shift, scale = volume_stats(data, kind)
        vs = m.voxel_size.copy()
        with mrcfile.new_mmap(dst, shape=data.shape, mrc_mode=2, overwrite=True) as out:
            step = _slab_step(data.shape)
            for s in _slabs(data.shape[0], step):
                out.data[s] = ((data[s].astype(np.float64) - shift) / scale).astype(np.float32)
            out.voxel_size = vs
    return shift, scale
