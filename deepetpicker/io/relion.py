"""RELION STAR inputs and outputs.

Column names used here are the documented RELION-4/5 tomography ones for
``rlnTomoName`` / ``rlnCoordinateX/Y/Z`` / ``rlnClassNumber`` (the same ones
RELION-US's DeepETPicker bridge writes). The tomogram-file column of a
``tomograms.star`` is *not* assumed: it is a parameter, and a wrong name fails
with the list of columns that are present.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import starfile

DEFAULT_TOMO_COLUMN = "rlnTomoReconstructedTomogram"   # unverified default; override with --tomo_column
COORD_COLUMNS = ("rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ")


def _table_with(path, column):
    blocks = starfile.read(path, always_dict=True)
    for name, df in blocks.items():
        if isinstance(df, pd.DataFrame) and column in df.columns:
            return df
    present = {n: list(d.columns) for n, d in blocks.items() if isinstance(d, pd.DataFrame)}
    raise KeyError(f"{path}: no table has column {column!r}; tables and columns present: {present}")


def read_tomograms_star(path, column=DEFAULT_TOMO_COLUMN, project_dir=None) -> dict[str, Path]:
    """``rlnTomoName -> tomogram file``. Relative file names are resolved
    against ``project_dir`` (RELION paths are relative to the project root;
    default: current directory), then against the STAR file's directory."""
    df = _table_with(path, column)
    if "rlnTomoName" not in df.columns:
        raise KeyError(f"{path}: table with {column!r} has no rlnTomoName column: {list(df.columns)}")
    root = Path(project_dir) if project_dir else Path.cwd()
    out = {}
    for name, f in zip(df["rlnTomoName"].astype(str), df[column].astype(str)):
        p = Path(f)
        for cand in ([p] if p.is_absolute() else [root / p, Path(path).parent / p]):
            if cand.exists():
                out[name] = cand.resolve()
                break
        else:
            raise FileNotFoundError(f"{path}: tomogram file for {name!r} not found: {f}")
    return out


def read_particles_star(path, scale=1.0) -> dict[str, np.ndarray]:
    """``rlnTomoName -> (N, 4) float array [class, x, y, z]``. Coordinates are
    multiplied by ``scale`` (RELION pixels -> picking-grid voxels). Class is
    ``rlnClassNumber`` when present, else 1."""
    df = _table_with(path, COORD_COLUMNS[0])
    for c in ("rlnTomoName",) + COORD_COLUMNS:
        if c not in df.columns:
            raise KeyError(f"{path}: missing column {c}; present: {list(df.columns)}. "
                           "Only pixel coordinates (rlnCoordinateX/Y/Z) are supported, not centred Angstrom ones.")
    cls = df["rlnClassNumber"].to_numpy() if "rlnClassNumber" in df.columns else np.ones(len(df))
    xyz = df[list(COORD_COLUMNS)].to_numpy(dtype=np.float64) * scale
    arr = np.column_stack([cls, xyz]).astype(np.float64)
    names = df["rlnTomoName"].astype(str).to_numpy()
    return {n: arr[names == n] for n in dict.fromkeys(names)}


def write_particles_star(path, coords_by_name: dict, scale=1.0) -> None:
    """Coordinate-only RELION particles table (block ``particles``): the same
    columns as RELION-US's DeepETPicker bridge."""
    frames = []
    for name, c in coords_by_name.items():
        c = np.asarray(c, dtype=np.float64).reshape(-1, 4)
        frames.append(pd.DataFrame({
            "rlnTomoName": name,
            "rlnCoordinateX": c[:, 1] * scale,
            "rlnCoordinateY": c[:, 2] * scale,
            "rlnCoordinateZ": c[:, 3] * scale,
            "rlnClassNumber": c[:, 0].astype(int),
        }))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["rlnTomoName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnClassNumber"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    starfile.write({"particles": df}, path, overwrite=True)
