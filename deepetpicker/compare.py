"""Compare two sets of picks (old vs new code, or picks vs ground truth).

Prints a short text summary meant to be pasted into a message, so regression
checks on real (multi-GB) tomograms never require moving any volume.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .io.coords import read_coords


def match_points(a: np.ndarray, b: np.ndarray, tol: float):
    """Mutual nearest neighbours within ``tol`` voxels. Returns (ia, ib, dist)."""
    a, b = np.asarray(a, float).reshape(-1, 3), np.asarray(b, float).reshape(-1, 3)
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    nn_ab, nn_ba = d.argmin(1), d.argmin(0)
    ia = np.array([i for i in range(len(a)) if nn_ba[nn_ab[i]] == i and d[i, nn_ab[i]] <= tol], dtype=int)
    ib = nn_ab[ia]
    return ia, ib, d[ia, ib]


def compare_dirs(dir_a, dir_b, truth_dir=None, tol=5.0) -> str:
    """One line per tomogram plus totals. ``tol`` is in voxels of the grid the
    coordinates are on (use about half the particle radius)."""
    A = {p.stem: p for p in Path(dir_a).glob("*.coords")}
    B = {p.stem: p for p in Path(dir_b).glob("*.coords")}
    T = {p.stem: p for p in Path(truth_dir).glob("*.coords")} if truth_dir else {}
    names = sorted(set(A) & set(B))
    lines = [f"A={dir_a}", f"B={dir_b}", f"tolerance={tol:g} voxels",
             f"tomograms only in A: {sorted(set(A) - set(B))}  only in B: {sorted(set(B) - set(A))}",
             f"{'tomogram':<20}{'nA':>6}{'nB':>6}{'matched':>8}{'median_d':>9}{'max_d':>7}"
             + (f"{'P_A':>6}{'R_A':>6}{'P_B':>6}{'R_B':>6}" if T else "")]
    tot = dict(nA=0, nB=0, m=0)
    dists = []
    for n in names:
        a = read_coords(A[n])[:, 1:]
        b = read_coords(B[n])[:, 1:]
        _, _, d = match_points(a, b, tol)
        tot["nA"] += len(a); tot["nB"] += len(b); tot["m"] += len(d); dists.extend(d.tolist())
        row = f"{n:<20}{len(a):>6}{len(b):>6}{len(d):>8}{np.median(d) if len(d) else float('nan'):>9.2f}{d.max() if len(d) else float('nan'):>7.2f}"
        if n in T:
            t = read_coords(T[n])[:, 1:]
            for pts in (a, b):
                tp = len(match_points(pts, t, tol)[0])
                row += f"{tp / max(len(pts), 1):>6.2f}{tp / max(len(t), 1):>6.2f}"
        lines.append(row)
    med = f"{np.median(dists):.2f}" if dists else "n/a"
    lines.append(f"TOTAL nA={tot['nA']} nB={tot['nB']} matched={tot['m']} "
                 f"({tot['m'] / max(tot['nA'], 1):.1%} of A, {tot['m'] / max(tot['nB'], 1):.1%} of B) median_d={med}")
    return "\n".join(lines)
