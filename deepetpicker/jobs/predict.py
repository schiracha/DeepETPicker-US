"""``deepetpicker predict``: pick particles in tomograms with a trained model."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import compat
from ..io.coords import read_coords, read_coords_scored
from ..io.relion import DEFAULT_TOMO_COLUMN, read_tomograms_star, write_particles_star
from ..prepare import normalize
from ..project.dataset import _match
from ..project.layout import JobDir
from ..project.params import read_params, write_params


@dataclass
class PredictParams:
    train_job: str = ""
    model: Optional[str] = None            # default: <train_job>/checkpoints/best.ckpt
    tomo_dir: Optional[str] = None
    tomo_star: Optional[str] = None
    tomo_column: str = DEFAULT_TOMO_COLUMN
    tomo_ext: str = ".mrc"
    project_dir: Optional[str] = None
    already_standardized: bool = False     # inputs came from a Prepare job's tomograms_std
    batch_size: int = 8
    patch_size: int = 72
    pad_size: int = 12
    seg_thresh: float = 0.5
    gpu_ids: list = field(default_factory=lambda: [0])
    refine_centroids: bool = False         # experimental sub-voxel refinement, see deepetpicker/refine.py
    relion_scale: float = 1.0              # picking-grid voxels -> RELION tomogram pixels (particles.star only)
    min_score: float = 0.0                 # particles.star keeps picks scoring at least this (coords/ keeps all)

    def validate(self):
        if not self.train_job:
            raise ValueError("train_job is required")
        if (self.tomo_dir is None) == (self.tomo_star is None):
            raise ValueError("give exactly one of tomo_dir / tomo_star")
        if self.patch_size % 8 or 2 * self.pad_size >= self.patch_size or self.pad_size < 1:
            raise ValueError("patch_size must be a multiple of 8 and pad_size < patch_size / 2")
        if not 0 < self.seg_thresh < 1:
            raise ValueError("seg_thresh must be in (0, 1)")
        if self.relion_scale <= 0:
            raise ValueError("relion_scale must be positive")
        if not 0 <= self.min_score <= 1:
            raise ValueError("min_score must be in [0, 1]")


def prepare_inference_inputs(job: JobDir, p: PredictParams):
    """Standardize (or link) the input tomograms into this job and build the
    legacy config. Returns ``(cfg, cfg_path, names, checkpoint)``."""
    tj = JobDir(p.train_job)
    train_doc = read_params(tj.params_json)
    if train_doc["job_type"] != "train":
        raise ValueError(f"{p.train_job} is a {train_doc['job_type']!r} job, not a train job")
    ckpt = Path(p.model) if p.model else tj.checkpoints / "best.ckpt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    cfg = compat.read_legacy_config(tj.root / "legacy_config.py")
    prep = read_params(JobDir(train_doc["inputs"]["prepare_job"] if Path(train_doc["inputs"]["prepare_job"]).is_absolute()
                              else tj.root / train_doc["inputs"]["prepare_job"]).params_json)["params"]

    tomos = (_match(sorted(Path(p.tomo_dir).iterdir()), p.tomo_ext) if p.tomo_dir
             else read_tomograms_star(p.tomo_star, p.tomo_column, p.project_dir))
    if not tomos:
        raise FileNotFoundError("no input tomograms found")
    job.mkdirs("tomograms_std")
    names = sorted(tomos)
    for i, n in enumerate(names):
        dst = job.tomograms_std / f"{n}.mrc"
        print(f"[{i + 1}/{len(names)}] {n}", flush=True)
        if p.already_standardized:
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(Path(tomos[n]).resolve())
        else:
            normalize.normalize_file(tomos[n], dst, prep["norm_type"])
    compat.write_tomo_index(job.tomograms_std, names)

    cfg.update(tomo_path=str(job.tomograms_std), label_path=None, ocp_path=None,
               dset_name=cfg["dset_name"])
    cfg_path = compat.write_legacy_config(job.root / "legacy_config.py", cfg)
    return cfg, cfg_path, names, ckpt


def _scores_for(job: JobDir, name: str, n_picks: int):
    """The per-pick scores predict wrote to ``coords_with_area/<name>.coords``
    (fifth column), or None when they are missing or do not line up one to one
    with ``coords/<name>.coords`` (never attach scores to the wrong picks)."""
    f = job.root / "coords_with_area" / f"{name}.coords"
    if not f.exists() or f.stat().st_size == 0:
        return None
    try:
        _, scores = read_coords_scored(f)
    except ValueError:
        return None
    return scores if scores is not None and len(scores) == n_picks else None


def write_outputs(job: JobDir, p: PredictParams, names):
    """particles.star from the native coords the legacy code wrote, with each
    pick's score as ``rlnAutopickFigureOfMerit`` so it can be filtered on
    afterwards (RELION's Select on value, or RELION-US's Filter Picks). With
    ``p.min_score`` the star keeps only picks scoring at least that; the native
    ``coords/`` and ``coords_with_area/`` files always keep every pick."""
    coords, scores = {}, {}
    for n in names:
        f = job.predictions / f"{n}.coords"
        coords[n] = read_coords(f) if f.exists() and f.stat().st_size else np.zeros((0, 4))
        scores[n] = _scores_for(job, n, len(coords[n])) if len(coords[n]) else np.zeros(0)
    have_scores = all(s is not None for s in scores.values())
    if not have_scores:
        print("warning: no usable per-pick scores for some tomograms; particles.star is written without "
              "rlnAutopickFigureOfMerit", flush=True)
        scores = None
    if p.min_score > 0:
        if scores is None:
            raise ValueError("min_score needs the per-pick scores, which are missing (see the warning above)")
        for n in names:
            keep = scores[n] >= p.min_score
            coords[n], scores[n] = coords[n][keep], scores[n][keep]
    write_particles_star(job.particles_star, coords, scale=p.relion_scale, scores_by_name=scores)
    return {n: len(c) for n, c in coords.items()}


def run_predict(job: JobDir, p: PredictParams):
    p.validate()
    cfg, cfg_path, names, ckpt = prepare_inference_inputs(job, p)
    write_params(job.params_json, "predict", asdict(p), {"train_job": job.relative(p.train_job)})
    args = compat.build_predict_args(p, cfg, cfg_path, ckpt, len(names), job.root)
    compat.ensure_repo_on_path()
    import importlib.util
    spec = importlib.util.spec_from_file_location("detpicker_legacy_test", compat.REPO / "test.py")  # not stdlib `test`
    legacy_test = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_test)
    legacy_test.test_func(args, stdout=None)
    counts = write_outputs(job, p, names)
    print("particles per tomogram:", counts, flush=True)
    return counts
