"""Adapters between job directories and the legacy training/inference code.

``train.py``, ``test.py`` and the dataloader read one config dict (written as
``train_configs={json}``) with absolute ``*_path`` entries, and a
``num_name.csv`` index that they address *by position*. This module builds
those inputs from a Prepare job, inside the *new* job's own directory, so the
upstream job is never modified. Everything generated here is derived and can
be regenerated; ``job_params.json`` stays the record of truth.
"""
from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# fixed by the published architecture / the old GUI + bash scripts
F_MAPS = [24, 48, 72, 108]


def ensure_repo_on_path():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def write_legacy_config(path, cfg: dict, prefix="train_configs=") -> Path:
    path = Path(path)
    path.write_text(prefix + json.dumps(cfg, indent=1))
    return path


def read_legacy_config(path) -> dict:
    return json.loads(Path(path).read_text().partition("=")[2])


def legacy_cfg(prepare_job, prepare_params: dict, dset_name: str, base_path, coord_path, tomo_path,
               model_name="ResUNet") -> dict:
    """The config dict the old code expects. ``prepare_job`` is a JobDir."""
    p = prepare_params
    return {
        "dset_name": dset_name,
        "base_path": str(base_path),
        "coord_path": str(coord_path),
        "coord_format": ".coords",
        "tomo_path": str(tomo_path),
        "tomo_format": ".mrc",
        "num_cls": p["num_cls"],
        "label_name": f"{p['label_type']}{p['label_diameter']}",
        "label_path": str(prepare_job.labels),
        "label_type": p["label_type"],
        "label_diameter": int(p["label_diameter"]),
        "ocp_type": "sphere",
        "ocp_diameter": str(p["ocp_diameter"]),
        "ocp_name": "data_ocp",
        "ocp_path": str(prepare_job.occupancy),
        "norm_type": p["norm_type"],
        "model_name": model_name,
    }


def write_coords_index(index_dir, prepare_job, dataset, order) -> Path:
    """``index_dir`` gets symlinks to the Prepare job's ``.coords`` files and a
    ``num_name.csv`` listing ``order`` (the dataloader indexes it by position).
    Symlinks, so nothing in the upstream job is written to or duplicated."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    by_name = {e.name: e for e in dataset.entries}
    rows = []
    for i, name in enumerate(order):
        e = by_name[name]
        if e.coords_file:
            link = index_dir / f"{name}.coords"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(e.coords_file).resolve())
        rows.append(f"{float(e.n_particles)}\t{name}\t{i}")
    (index_dir / "num_name.csv").write_text("\n".join(rows) + "\n")
    return index_dir


def write_tomo_index(tomo_dir, names) -> None:
    """``num_name.csv`` of the inference input folder (``idx<TAB>name``)."""
    Path(tomo_dir, "num_name.csv").write_text("".join(f"{i}\t{n}\n" for i, n in enumerate(names)))


def _base_args(extra: dict) -> Namespace:
    ensure_repo_on_path()
    from options.option import BaseOptions
    args = BaseOptions().parser.parse_args([])     # defaults only; never read sys.argv
    for k, v in extra.items():
        setattr(args, k, v)
    return args


def class_setup(num_cls: int):
    """(num_classes seen by the network, use_sigmoid, use_softmax): one class
    is a sigmoid map; several classes get an extra background channel."""
    if num_cls == 1:
        return 1, True, False
    return num_cls + 1, False, True


def build_train_args(p, cfg_path, data_split, out_dir) -> Namespace:
    n, sig, soft = class_setup(p.num_cls)
    return _base_args(dict(
        use_bg=True, use_IP=True, use_coord=True, test_use_pad=True, meanPool_NMS=True,
        f_maps=list(F_MAPS), network=p.model_name,
        num_classes=n, use_sigmoid=sig, use_softmax=soft,
        batch_size=p.batch_size, val_batch_size=p.batch_size,
        block_size=p.patch_size, val_block_size=p.patch_size,
        pad_size=[p.pad_size], learning_rate=p.lr, max_epoch=p.max_epochs, threshold=p.seg_thresh,
        gpu_id=list(p.gpu_ids), configs=str(cfg_path), data_split=list(data_split),
        test_mode="val", random_num=0, loader_type="dataloader_DynamicLoad",
        checkpoints=p.resume, out_dir=str(out_dir),
    ))


def build_predict_args(p, cfg: dict, cfg_path, checkpoint, n_tomograms, out_dir) -> Namespace:
    n, sig, soft = class_setup(cfg["num_cls"])
    ocp = [int(i) for i in str(cfg["ocp_diameter"]).split(",")]
    return _base_args(dict(
        use_bg=True, use_IP=True, use_coord=True, test_use_pad=True, use_seg=True, meanPool_NMS=True,
        f_maps=list(F_MAPS), network=cfg["model_name"],
        num_classes=n, use_sigmoid=sig, use_softmax=soft,
        batch_size=p.batch_size, val_batch_size=p.batch_size,
        block_size=p.patch_size, val_block_size=p.patch_size,
        pad_size=[p.pad_size], threshold=p.seg_thresh, gpu_id=list(p.gpu_ids),
        configs=str(cfg_path), train_configs="", checkpoints=str(checkpoint),
        test_mode="test_only", out_name="PredictedLabels", de_duplication=True, de_dup_fmt="fmt4",
        mini_dist=sorted(d // 2 + 1 for d in ocp)[0],
        data_split=[0, 1, 0, 1, 0, 1], test_idxs=list(range(n_tomograms)),
        out_dir=str(out_dir), refine_centroids=bool(p.refine_centroids),
    ))
