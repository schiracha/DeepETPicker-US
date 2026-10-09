"""``python -m deepetpicker <prepare|train|predict|export|compare> --output_dir JOB ...``

Each command fills the job directory it is given (RELION-US creates
``<Category>/jobNNN/`` and handles numbering, logs and exit markers) and never
writes outside it. Errors exit non-zero with a one-line message for bad
input, or a traceback for anything unexpected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .project.layout import JobDir


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _gpus(s):
    return [int(x) for x in _csv(s)]


def _inputs(p, tomos=True, coords=False):
    if tomos:
        p.add_argument("--tomo_dir"); p.add_argument("--tomo_star")
        p.add_argument("--tomo_column", default=None, help="column of the tomograms STAR holding file names")
        p.add_argument("--tomo_ext", default=".mrc")
        p.add_argument("--project_dir", help="base for relative paths inside STAR files (default: cwd)")
    if coords:
        p.add_argument("--coords_dir"); p.add_argument("--coords_star")
        p.add_argument("--coords_ext", default=".coords")
        p.add_argument("--coords_scale", type=float, default=1.0,
                       help="multiply coordinates by this to reach the tomograms' voxel grid")


def build_parser():
    ap = argparse.ArgumentParser(prog="deepetpicker", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="standardize tomograms, make labels and occupancy maps")
    p.add_argument("--output_dir", required=True)
    _inputs(p, coords=True)
    p.add_argument("--num_cls", type=int, default=1)
    p.add_argument("--label_type", default="gaussian", choices=["gaussian", "sphere", "cubic"])
    p.add_argument("--label_diameter", type=int, help="voxels, odd")
    p.add_argument("--ocp_diameter", help="voxels, odd, one per class, comma separated")
    p.add_argument("--pixel_size", type=float, help="A/voxel of the tomograms")
    p.add_argument("--particle_diameter_A", type=float, help="fills in the two diameters when not given")
    p.add_argument("--norm_type", default="standardization", choices=["standardization", "normalization"])

    p = sub.add_parser("train", help="train on a prepare job")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prepare_job", required=True)
    p.add_argument("--val_tomo", required=True, help="tomogram name used for validation")
    p.add_argument("--train_tomos", type=_csv, default=[], help="names; default: all annotated except --val_tomo")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=72)
    p.add_argument("--pad_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--seg_thresh", type=float, default=0.5)
    p.add_argument("--gpu", type=_gpus, default=[0], dest="gpu_ids", help="e.g. 0 or 0,1")
    p.add_argument("--resume", help="checkpoint to continue from")

    p = sub.add_parser("predict", help="pick particles with a trained model")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_job", required=True)
    p.add_argument("--model", help="checkpoint (default: <train_job>/checkpoints/best.ckpt)")
    _inputs(p)
    p.add_argument("--already_standardized", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=72)
    p.add_argument("--pad_size", type=int, default=12)
    p.add_argument("--seg_thresh", type=float, default=0.5)
    p.add_argument("--gpu", type=_gpus, default=[0], dest="gpu_ids")
    p.add_argument("--refine_centroids", action="store_true", help="experimental sub-voxel refinement")
    p.add_argument("--relion_scale", type=float, default=1.0,
                   help="picking-grid voxels -> RELION tomogram pixels, applied to particles.star only")

    p = sub.add_parser("export", help="convert .coords to star / box / coords / RELION-5 particles.star")
    p.add_argument("--coords_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--fmt", required=True, choices=["star", "box", "coords", "relion5"])
    p.add_argument("--cls", type=int, help="keep only this class id")
    p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser("compare", help="compare two folders of .coords (old vs new, or picks vs truth)")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--truth"); p.add_argument("--tol", type=float, default=5.0, help="voxels")
    return ap


def _drop_none(ns, keys):
    return {k: v for k, v in vars(ns).items() if k in keys and v is not None}


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            from .jobs.prepare import PrepareParams, run_prepare
            keys = PrepareParams.__dataclass_fields__
            job = JobDir(args.output_dir)
            run_prepare(job, PrepareParams(**_drop_none(args, keys)))
        elif args.command == "train":
            from .jobs.train import TrainParams, run_train
            job = JobDir(args.output_dir)
            run_train(job, TrainParams(**_drop_none(args, TrainParams.__dataclass_fields__)))
        elif args.command == "predict":
            from .jobs.predict import PredictParams, run_predict
            job = JobDir(args.output_dir)
            run_predict(job, PredictParams(**_drop_none(args, PredictParams.__dataclass_fields__)))
        elif args.command == "compare":
            from .compare import compare_dirs
            print(compare_dirs(args.a, args.b, args.truth, args.tol))
        elif args.command == "export":
            from .io.export import export
            for f in export(args.coords_dir, args.output_dir, args.fmt, args.cls, args.scale):
                print(f)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
