"""The tomogram table that replaces the two ``num_name.csv`` files.

Tomograms are always identified by *name*. Integer ids exist only to read the
old ``--train_set_ids`` / ``--val_set_ids`` syntax and are positions in the
name-sorted table.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import starfile

BLOCK = "tomograms"
ROLES = ("train", "val", "train+val", "")


@dataclass(frozen=True)
class TomoEntry:
    name: str                 # file stem, e.g. "TS_001"
    tomo_file: str            # path as stored (relative to the star file's directory when possible)
    coords_file: str = ""     # "" when the tomogram has no annotation
    n_particles: int = 0
    role: str = ""            # train | val | train+val | ""


def parse_ids(text: str) -> list[int]:
    """Old id syntax: ``"0,2,5"`` and ranges ``"0-3"``. Ranges are
    *inclusive* (``0-3`` is 0,1,2,3), matching what the old GUI and bash
    scripts did; the old README said otherwise."""
    out: list[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            a, b = (int(i) for i in item.split("-"))
            if b < a:
                raise ValueError(f"descending range {item!r}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(item))
    return out


def _match(files: Iterable[Path], ext: str) -> dict[str, Path]:
    """stem -> path for files ending exactly in ``ext`` (not a substring test)."""
    ext = ext if ext.startswith(".") else "." + ext
    return {p.name[: -len(ext)]: p for p in files if p.is_file() and p.name.endswith(ext)}


class Dataset:
    def __init__(self, entries: list[TomoEntry]):
        names = [e.name for e in entries]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tomogram names")
        self.entries = sorted(entries, key=lambda e: e.name)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_folders(cls, tomo_dir, tomo_ext=".mrc", coord_dir=None, coord_ext=".coords") -> "Dataset":
        tomos = _match(sorted(Path(tomo_dir).iterdir()), tomo_ext)
        coords = _match(sorted(Path(coord_dir).iterdir()), coord_ext) if coord_dir else {}
        entries = []
        for name, tf in tomos.items():
            cf = coords.get(name)
            n = _count_lines(cf) if cf else 0
            entries.append(TomoEntry(name, str(tf), str(cf) if cf else "", n))
        orphans = set(coords) - set(tomos)
        if orphans:
            raise ValueError(f"coordinate files without a tomogram: {sorted(orphans)}")
        return cls(entries)

    # ---- access -------------------------------------------------------
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def annotated(self) -> list[TomoEntry]:
        return [e for e in self.entries if e.coords_file]

    def ids_to_names(self, ids: Iterable[int]) -> list[str]:
        names = self.names()
        bad = [i for i in ids if not 0 <= i < len(names)]
        if bad:
            raise ValueError(f"ids out of range 0..{len(names) - 1}: {bad}")
        return [names[i] for i in ids]

    def with_roles(self, train: Iterable[str], val: Iterable[str]) -> "Dataset":
        """Return a copy with roles set. A validation tomogram that is also in
        the training list gets role ``train+val`` (the train range then
        includes it, as the old GUI did)."""
        train, val = list(train), list(val)
        if len(val) != 1:
            raise ValueError("exactly one validation tomogram is supported")
        if len(set(train)) != len(train):
            raise ValueError("duplicate names in train set")
        known = {e.name: e for e in self.entries}
        for n in train + val:
            if n not in known:
                raise ValueError(f"unknown tomogram {n!r}")
            if not known[n].coords_file:
                raise ValueError(f"tomogram {n!r} has no coordinates; it cannot be used for training/validation")
        def role(n):
            t, v = n in train, n == val[0]
            return "train+val" if t and v else "val" if v else "train" if t else ""
        return Dataset([replace(e, role=role(e.name)) for e in self.entries])

    # ---- split --------------------------------------------------------
    def split(self) -> tuple[list[str], list[int]]:
        """(order, data_split) for the dataloader.

        ``order``: training tomograms (excluding the validation one), then the
        validation tomogram, then the rest, each group in name order.
        ``data_split`` is the six-number list the dataloader indexes ``order``
        with: ``[train_start, train_end, val_start, val_end, test_start,
        test_end]``.
        """
        val = [e.name for e in self.entries if e.role in ("val", "train+val")]
        if len(val) != 1:
            raise ValueError("dataset has no (or several) validation tomograms; call with_roles() first")
        train_only = [e.name for e in self.entries if e.role == "train"]
        rest = [e.name for e in self.entries if e.role == ""]
        order = train_only + val + rest
        val_first = len(train_only)
        in_train = any(e.role == "train+val" for e in self.entries)
        return order, [0, val_first + int(in_train), val_first, val_first + 1, val_first, val_first + 1]

    # ---- persistence --------------------------------------------------
    def write_star(self, path, base_dir=None) -> None:
        """Paths are stored relative to ``base_dir`` (default: the star file's
        directory) so a project can be moved."""
        import os
        path = Path(path)
        base = Path(base_dir) if base_dir else path.parent
        def rel(p):
            return os.path.relpath(Path(p).resolve(), base.resolve()) if p else "NA"
        df = pd.DataFrame({
            "rlnTomoName": [e.name for e in self.entries],
            "deepetTomoFile": [rel(e.tomo_file) for e in self.entries],
            "deepetCoordsFile": [rel(e.coords_file) for e in self.entries],
            "deepetNumParticles": [e.n_particles for e in self.entries],
            "deepetRole": [e.role or "NA" for e in self.entries],
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        starfile.write({BLOCK: df}, path, overwrite=True)

    @classmethod
    def read_star(cls, path, base_dir=None) -> "Dataset":
        path = Path(path)
        base = Path(base_dir) if base_dir else path.parent
        df = starfile.read(path, always_dict=True)[BLOCK]
        def absolutize(v):
            return "" if v == "NA" else str((base / v).resolve())
        return cls([
            TomoEntry(str(r.rlnTomoName), absolutize(r.deepetTomoFile), absolutize(r.deepetCoordsFile),
                      int(r.deepetNumParticles), "" if r.deepetRole == "NA" else str(r.deepetRole))
            for r in df.itertuples()
        ])


def _count_lines(path) -> int:
    with open(path) as f:
        return sum(1 for ln in f if ln.strip() and not ln.startswith("#"))
