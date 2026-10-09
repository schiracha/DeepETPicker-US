"""``deepetpicker prepare``: standardized tomograms, weak labels, occupancy
maps and ``dataset.star`` for one job directory. Torch-free."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import mrcfile
import numpy as np

from ..io.coords import read_coords, to_voxel_indices, write_coords
from ..io.relion import DEFAULT_TOMO_COLUMN, read_particles_star, read_tomograms_star
from ..prepare import labels, normalize
from ..project.dataset import Dataset, TomoEntry, _match
from ..project.layout import JobDir
from ..project.params import write_params
from ..units import suggest_diameters


@dataclass
class PrepareParams:
    # inputs: a folder or a RELION STAR file, for tomograms and for coordinates
    tomo_dir: Optional[str] = None
    tomo_star: Optional[str] = None
    tomo_column: str = DEFAULT_TOMO_COLUMN
    tomo_ext: str = ".mrc"
    coords_dir: Optional[str] = None
    coords_star: Optional[str] = None
    coords_ext: str = ".coords"
    coords_scale: float = 1.0          # coordinate grid -> tomogram grid
    project_dir: Optional[str] = None  # base for relative paths inside STAR files
    # labels
    num_cls: int = 1
    label_type: str = "gaussian"
    label_diameter: Optional[int] = None       # voxels, odd
    ocp_diameter: Optional[str] = None         # voxels, odd, comma separated per class
    pixel_size: Optional[float] = None         # A/voxel of the tomograms; with particle_diameter_A
    particle_diameter_A: Optional[float] = None  # fills in the two diameters above when they are not given
    norm_type: str = "standardization"

    def resolve(self) -> "PrepareParams":
        p = PrepareParams(**asdict(self))
        if (p.tomo_dir is None) == (p.tomo_star is None):
            raise ValueError("give exactly one of tomo_dir / tomo_star")
        if (p.coords_dir is None) == (p.coords_star is None):
            raise ValueError("give exactly one of coords_dir / coords_star")
        if p.label_type not in labels.LABEL_TYPES:
            raise ValueError(f"label_type must be one of {labels.LABEL_TYPES}")
        if p.norm_type not in normalize.KINDS:
            raise ValueError(f"norm_type must be one of {normalize.KINDS}")
        if p.label_diameter is None or p.ocp_diameter is None:
            if not (p.pixel_size and p.particle_diameter_A):
                raise ValueError("give label_diameter and ocp_diameter, or pixel_size and particle_diameter_A")
            s = suggest_diameters(p.pixel_size, p.particle_diameter_A)
            if p.label_diameter is None:
                p.label_diameter = s.label_voxels
            if p.ocp_diameter is None:
                p.ocp_diameter = ",".join([str(s.particle_voxels)] * p.num_cls)
            if not s.in_recommended_range:
                print(f"warning: a {p.particle_diameter_A:g} A particle is {s.particle_voxels} voxels at "
                      f"{p.pixel_size:g} A/voxel; DeepETPicker's README recommends 20-30. Consider binning less.",
                      file=sys.stderr)
        ocp = [int(v) for v in str(p.ocp_diameter).split(",")]
        if len(ocp) != p.num_cls:
            raise ValueError(f"ocp_diameter needs {p.num_cls} comma-separated value(s), got {len(ocp)}")
        labels._check_diameters(ocp + [int(p.label_diameter)])
        return p


def _tomograms(p):
    if p.tomo_dir:
        return {n: f for n, f in _match(sorted(Path(p.tomo_dir).iterdir()), p.tomo_ext).items()}
    return read_tomograms_star(p.tomo_star, p.tomo_column, p.project_dir)


def _coords(p):
    if p.coords_star:
        return read_particles_star(p.coords_star, p.coords_scale)
    files = _match(sorted(Path(p.coords_dir).iterdir()), p.coords_ext)
    return {n: read_coords(f) * np.array([1, p.coords_scale, p.coords_scale, p.coords_scale]) for n, f in files.items()}


def run_prepare(job: JobDir, params: PrepareParams) -> Dataset:
    p = params.resolve()
    tomos, coords = _tomograms(p), _coords(p)
    orphans = sorted(set(coords) - set(tomos))
    if orphans:
        raise ValueError(f"coordinates for tomograms that were not found: {orphans}")
    if not coords:
        raise ValueError("no coordinates found: at least one annotated tomogram is needed to prepare training data")
    ocp = [int(v) for v in str(p.ocp_diameter).split(",")]
    job.mkdirs("tomograms_std", "coords", "labels", "occupancy")

    entries = []
    for i, name in enumerate(sorted(tomos)):
        print(f"[{i + 1}/{len(tomos)}] {name}", flush=True)
        std = job.tomograms_std / f"{name}.mrc"
        normalize.normalize_file(tomos[name], std, p.norm_type)
        coords_file = ""
        n = 0
        if name in coords:
            with mrcfile.mmap(std, mode="r", permissive=True) as m:
                shape = m.data.shape
            c = np.asarray(coords[name], dtype=np.float64)
            if not set(np.unique(c[:, 0]).astype(int)) <= set(range(1, p.num_cls + 1)):
                raise ValueError(f"{name}: class ids must be in 1..{p.num_cls}, found {sorted(set(c[:, 0].astype(int)))}")
            idx = to_voxel_indices(c[:, 1:])
            hi = np.array([shape[2], shape[1], shape[0]]) - 1       # x, y, z
            clipped = ((idx < 0) | (idx > hi)).any(axis=1).sum()
            if clipped:
                print(f"warning: {name}: {clipped} coordinate(s) outside the volume, moved to the border", file=sys.stderr)
            idx = np.clip(idx, 0, hi)
            ints = np.column_stack([c[:, 0].astype(np.int64), idx])
            # always 4 columns (class first): the dataloader oversamples rows whose *first* column is 15
            write_coords(job.coords / f"{name}.coords", ints, integer=True)
            labels.write_label_file(job.labels / f"{name}.mrc", std, ints, p.label_type, [p.label_diameter], p.num_cls)
            labels.write_label_file(job.occupancy / f"{name}.mrc", std, ints, labels.OCP, ocp, p.num_cls)
            coords_file, n = str(job.coords / f"{name}.coords"), len(ints)
        entries.append(TomoEntry(name, str(std), coords_file, n))
    ds = Dataset(entries)
    ds.write_star(job.dataset_star)
    write_params(job.params_json, "prepare", asdict(p))
    return ds
