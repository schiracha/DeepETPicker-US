"""Tiny synthetic cryo-ET-like data: Gaussian blobs in noise, with known coordinates."""
from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np


def make_raw_data(root, n_tomos=3, n_unannotated=1, shape=(40, 48, 48), n_particles=5, seed=0):
    """Write ``TS_00i.mrc`` (+ ``TS_00i.coords`` for the annotated ones) into
    ``root`` and return ``{name: coords (N, 4) of class,x,y,z}``.

    ``shape`` is (z, y, x), the numpy / MRC order. Coordinates are written
    ``class x y z`` as DeepETPicker expects.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    zz, yy, xx = np.indices(shape)
    truth = {}
    for i in range(n_tomos):
        name = f"TS_{i:03d}"
        vol = rng.normal(0, 1, shape).astype(np.float32)
        pts = np.column_stack([
            rng.integers(8, shape[2] - 8, n_particles),
            rng.integers(8, shape[1] - 8, n_particles),
            rng.integers(8, shape[0] - 8, n_particles),
        ])
        for x, y, z in pts:
            vol += 4 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
        with mrcfile.new(root / f"{name}.mrc", overwrite=True) as m:
            m.set_data(vol)
        if i < n_tomos - n_unannotated:
            arr = np.column_stack([np.ones(len(pts), int), pts])
            np.savetxt(root / f"{name}.coords", arr, fmt="%d", delimiter="\t")
            truth[name] = arr
    return truth
