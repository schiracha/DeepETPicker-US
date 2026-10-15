"""``deepetpicker train``: fit the network on a Prepare job's data."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .. import compat
from ..project.dataset import Dataset
from ..project.layout import JobDir
from ..project.params import read_params, write_params


@dataclass
class TrainParams:
    prepare_job: str = ""
    val_tomo: str = ""
    train_tomos: list = field(default_factory=list)   # empty: every annotated tomogram except val_tomo
    batch_size: int = 8
    patch_size: int = 72
    pad_size: int = 12
    lr: float = 0.001
    max_epochs: int = 60
    seg_thresh: float = 0.5
    gpu_ids: list = field(default_factory=lambda: [0])
    model_name: str = "ResUNet"
    resume: Optional[str] = None
    num_cls: int = 1                                  # filled from the Prepare job

    def validate(self):
        if not self.prepare_job:
            raise ValueError("prepare_job is required")
        if not self.val_tomo:
            raise ValueError("val_tomo is required (the tomogram used for validation)")
        if self.patch_size % 8 or self.patch_size < 32:
            raise ValueError("patch_size must be a multiple of 8 and at least 32")
        if self.pad_size < 1 or 2 * self.pad_size >= self.patch_size:
            raise ValueError("pad_size must satisfy 1 <= pad_size < patch_size / 2")
        if self.batch_size < 1 or self.max_epochs < 1 or not 0 < self.seg_thresh < 1 or self.lr <= 0:
            raise ValueError("batch_size, max_epochs, lr must be positive and seg_thresh in (0, 1)")
        if not self.gpu_ids:
            raise ValueError("at least one GPU id is required")


def prepare_training_inputs(job: JobDir, p: TrainParams):
    """Everything short of importing torch: roles, data_split, legacy config,
    coords index. Returns ``(cfg_path, data_split, order)``."""
    pj = JobDir(p.prepare_job)
    prep = read_params(pj.params_json)
    if prep["job_type"] != "prepare":
        raise ValueError(f"{p.prepare_job} is a {prep['job_type']!r} job, not a prepare job")
    ds = Dataset.read_star(pj.dataset_star)
    annotated = {e.name for e in ds.annotated()}
    train = list(p.train_tomos) or sorted(annotated - {p.val_tomo})
    if not train:
        train = [p.val_tomo]          # a single annotated tomogram trains and validates on itself
    ds = ds.with_roles(train, [p.val_tomo])
    order, split = ds.split()
    p.num_cls = int(prep["params"]["num_cls"])
    cfg = compat.legacy_cfg(pj, prep["params"], job.root.name, job.root, job.root / "coords_index",
                            pj.tomograms_std, p.model_name)
    ids = {n: i for i, n in enumerate(ds.names())}
    cfg.update(batch_size=p.batch_size, patch_size=p.patch_size, padding_size=p.pad_size, lr=p.lr,
               max_epochs=p.max_epochs, seg_thresh=p.seg_thresh, gpu_ids=",".join(map(str, p.gpu_ids)),
               train_set_ids=",".join(str(ids[n]) for n in train), val_set_ids=str(ids[p.val_tomo]))
    compat.write_coords_index(job.root / "coords_index", pj, ds, order)
    cfg_path = compat.write_legacy_config(job.root / "legacy_config.py", cfg)
    return cfg_path, split, order


def resolved_record(order, split) -> dict:
    """What the run actually used, by name. ``train_tomos`` in the parameters is
    empty when it defaulted to "every annotated tomogram except the validation
    one", so the effective sets are recorded here."""
    return {"train_tomos": list(order[:split[1]]), "val_tomo": order[split[2]],
            "order": list(order), "data_split": list(split)}


def run_train(job: JobDir, p: TrainParams):
    p.validate()
    job.mkdirs("checkpoints")
    cfg_path, split, order = prepare_training_inputs(job, p)
    write_params(job.params_json, "train", {**asdict(p), "resolved": resolved_record(order, split)},
                 {"prepare_job": job.relative(p.prepare_job)})
    print(f"training order: {order}; data_split={split}", flush=True)
    args = compat.build_train_args(p, cfg_path, split, job.root)
    compat.ensure_repo_on_path()
    import train as legacy_train                       # needs torch + pytorch-lightning
    for k, v in sorted(vars(args).items()):
        print(k, "=", v)
    legacy_train.train_func(args, stdout=None)
