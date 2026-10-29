"""Coordinate export formats (replaces the GUI's "coord conversion" panel).

``star`` / ``box`` / ``coords`` reproduce ``utils/coordFormatConvert.py``
byte for byte (tested), so existing downstream scripts keep working.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .coords import read_coords_scored
from .relion import write_particles_star


def _star(xyz, path):
    xyz = np.round(xyz, 1).astype(str)
    with open(path, "w") as f:
        f.write("\ndata_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnCoordinateZ #3\n")
        for item in xyz:
            f.write(" ".join(item) + "\n")


def _box(xyz, path):
    """EMAN2-style: x.1f, y.1f, z.0f (as before; no box size column)."""
    with open(path, "w") as f:
        for x, y, z in xyz:
            f.write(f"{x:.1f}\t{y:.1f}\t{z:.0f}\n")


def _coords(xyz, path):
    with open(path, "w") as f:
        for x, y, z in xyz:
            f.write(f"{x:.0f}\t{y:.0f}\t{z:.0f}\n")


WRITERS = {"star": (_star, ".star"), "box": (_box, ".box"), "coords": (_coords, ".coords")}
FORMATS = tuple(WRITERS) + ("relion5",)


def export(coords_dir, out_dir, fmt, cls=None, scale=1.0, relion_name="particles.star"):
    """Convert every ``*.coords`` in ``coords_dir``. ``cls``: keep one class
    (and drop the class column, as the GUI did); ``scale`` multiplies x y z.
    ``relion5`` writes one coordinate-only ``particles.star`` with all classes
    (or just ``cls``). Returns the list of files written."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    files = sorted(Path(coords_dir).glob("*.coords"))
    if not files:
        raise FileNotFoundError(f"no *.coords in {coords_dir}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per, scored = {}, {}
    for f in files:
        c, s = read_coords_scored(f)           # a five-column file (coords_with_area/) carries scores
        if cls is not None:
            keep = c[:, 0].astype(int) == cls
            c, s = c[keep], (s[keep] if s is not None else None)
        per[f.stem], scored[f.stem] = c, s
    if cls is not None and all(len(c) == 0 for c in per.values()):
        raise ValueError(f"class {cls} not found in any file")
    if fmt == "relion5":
        has_scores = all(s is not None for s in scored.values())
        write_particles_star(out_dir / relion_name, per, scale=scale, scores_by_name=scored if has_scores else None)
        return [out_dir / relion_name]
    writer, ext = WRITERS[fmt]
    written = []
    for name, c in per.items():
        path = out_dir / f"{name}{ext}"
        writer(c[:, 1:] * scale, path)
        written.append(path)
    return written
