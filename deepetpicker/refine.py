"""Optional sub-voxel refinement of picked peaks (experimental, off by default).

The inference path finds each particle as a *voxel*: the maximum of the
mean-pooled binary prediction (``_nms_v2`` in ``test.py``). The coordinates it
writes are therefore integers; they were never sub-voxel. This refines each
peak to the centre of mass of the binary prediction inside a cube around it.

It is a heuristic: it is exact for a symmetric blob wholly inside the cube and
biased towards the peak voxel if the cube cuts through the blob, so ``radius``
should be about the particle radius. Not validated against ground truth.
"""
from __future__ import annotations

import numpy as np


def refine_centroid(mask: np.ndarray, z: int, y: int, x: int, radius: int):
    """Centre of mass (dz, dy, dx offsets from the peak) of ``mask`` inside the
    cube of half-width ``radius`` around integer voxel ``(z, y, x)``. Returns
    zeros if the cube holds no mass or lies outside the array."""
    lo = [max(c - radius, 0) for c in (z, y, x)]
    hi = [min(c + radius + 1, n) for c, n in zip((z, y, x), mask.shape)]
    if any(h <= l for l, h in zip(lo, hi)):
        return 0.0, 0.0, 0.0
    w = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float64)
    total = w.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    zz, yy, xx = np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij")
    return (float((w * zz).sum() / total - z),
            float((w * yy).sum() / total - y),
            float((w * xx).sum() / total - x))


def refine_offsets(binmask, coords, radius):
    """``binmask``: torch tensor [B, C, Z, Y, X] (0/1); ``coords``: long tensor
    [N, 5] of ``b, c, z, y, x`` as returned by ``keep.nonzero()``. Returns a
    float tensor [N, 3] of ``dz, dy, dx`` on the same device."""
    import torch
    arr = binmask.detach().cpu().numpy()
    out = np.zeros((coords.shape[0], 3), dtype=np.float32)
    for i, (b, c, z, y, x) in enumerate(coords.detach().cpu().numpy()):
        out[i] = refine_centroid(arr[b, c], int(z), int(y), int(x), radius)
    return torch.from_numpy(out).to(coords.device)
