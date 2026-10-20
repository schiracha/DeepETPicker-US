"""Where things live inside one DeepETPicker job directory.

RELION-US creates ``<Category>/jobNNN/`` and owns numbering, ``run.out`` /
``run.err`` and the ``RELION_JOB_EXIT_*`` markers; DeepETPicker only fills the
directory it is given and never writes outside it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_inside(root, path) -> Path:
    """Resolve ``path`` (relative paths are taken relative to ``root``) and
    refuse anything that escapes ``root``."""
    root = Path(root).resolve()
    p = Path(path)
    p = (root / p) if not p.is_absolute() else p
    p = p.resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"{p} is outside {root}")
    return p


@dataclass(frozen=True)
class JobDir:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    # prepare
    @property
    def tomograms_std(self) -> Path: return self.root / "tomograms_std"
    @property
    def coords(self) -> Path: return self.root / "coords"
    @property
    def labels(self) -> Path: return self.root / "labels"
    @property
    def occupancy(self) -> Path: return self.root / "occupancy"
    @property
    def dataset_star(self) -> Path: return self.root / "dataset.star"
    # train
    @property
    def checkpoints(self) -> Path: return self.root / "checkpoints"
    @property
    def metrics_csv(self) -> Path: return self.root / "metrics.csv"
    # predict
    @property
    def predictions(self) -> Path: return self.root / "coords"
    @property
    def particles_star(self) -> Path: return self.root / "particles.star"
    # all
    @property
    def params_json(self) -> Path: return self.root / "job_params.json"

    def mkdirs(self, *names: str) -> None:
        for n in names:
            getattr(self, n).mkdir(parents=True, exist_ok=True)

    def relative(self, path) -> str:
        """Path as stored in ``dataset.star`` / ``job_params.json``: relative
        to the job directory when possible, else absolute."""
        import os
        p = Path(path).resolve()
        try:
            return os.path.relpath(p, self.root.resolve())
        except ValueError:
            return str(p)
