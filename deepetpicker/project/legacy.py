"""Keeps the old ``coords/num_name.csv`` pipeline honest until it is replaced.

The old GUI rewrote ``coords/num_name.csv`` before training so that the
training tomograms came first, then the validation tomogram, then the rest;
the dataloader then indexes that file *by position* with ``data_split``.
``bin/train_bash.py`` computed ``data_split`` but never reordered the file,
so ``--train_set_ids``/``--val_set_ids`` did not select what they named.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import parse_ids


def reorder_num_name(rows: list[list[str]], train_ids: list[int], val_ids: list[int]):
    """``rows``: ``[count, name, id]`` as in ``coords/num_name.csv``. Returns
    ``(new_rows, data_split)``. Row *ids* (third column) never change; only
    their order does."""
    if len(val_ids) != 1:
        raise ValueError("exactly one validation tomogram is supported")
    ids = [int(r[2]) for r in rows]
    for i in list(train_ids) + list(val_ids):
        if i not in ids:
            raise ValueError(f"id {i} not in dataset (ids: {sorted(ids)})")
    val = val_ids[0]
    train_only = [r for r in rows if int(r[2]) in set(train_ids) and int(r[2]) != val]
    rest = [r for r in rows if int(r[2]) not in set(train_ids) and int(r[2]) != val]
    val_rows = [r for r in rows if int(r[2]) == val]
    val_first = len(train_only)
    train_end = val_first + (1 if val in train_ids else 0)
    return train_only + val_rows + rest, [0, train_end, val_first, val_first + 1, val_first, val_first + 1]


def reorder_num_name_csv(path, train_set_ids: str, val_set_ids: str) -> list[int]:
    """In-place version for ``coords/num_name.csv``; returns ``data_split``."""
    path = Path(path)
    rows = np.loadtxt(path, delimiter="\t", dtype=str).reshape(-1, 3).tolist()
    new_rows, split = reorder_num_name(rows, parse_ids(train_set_ids), parse_ids(val_set_ids))
    np.savetxt(path, np.array(new_rows), delimiter="\t", newline="\n", fmt="%s")
    return split
