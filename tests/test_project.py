import json
import random
import shutil

import numpy as np
import pytest

from deepetpicker.project.dataset import Dataset, parse_ids
from deepetpicker.project.layout import JobDir, resolve_inside
from deepetpicker.project.legacy import reorder_num_name, reorder_num_name_csv
from deepetpicker.project.params import load_legacy_config, read_params, write_params


# ---- ids -------------------------------------------------------------
def test_parse_ids_ranges_are_inclusive():
    assert parse_ids("0-3") == [0, 1, 2, 3]
    assert parse_ids("0, 2,5-6") == [0, 2, 5, 6]
    with pytest.raises(ValueError):
        parse_ids("3-1")


# ---- dataset ---------------------------------------------------------
def test_from_folders_exact_suffix_sorted(raw_data):
    d, truth = raw_data
    (d / "TS_000.mrc.bak").write_bytes(b"x")          # substring match would pick this up
    ds = Dataset.from_folders(d, ".mrc", d, ".coords")
    assert ds.names() == ["TS_000", "TS_001", "TS_002"]
    assert [e.n_particles for e in ds.entries] == [5, 5, 0]
    assert [bool(e.coords_file) for e in ds.entries] == [True, True, False]


def test_orphan_coords_rejected(raw_data):
    d, _ = raw_data
    (d / "GHOST.coords").write_text("1 2 3\n")
    with pytest.raises(ValueError, match="GHOST"):
        Dataset.from_folders(d, ".mrc", d, ".coords")


def test_star_roundtrip_and_relocatable(raw_data, tmp_path):
    d, _ = raw_data
    proj = tmp_path / "proj"
    (proj / "job001").mkdir(parents=True)
    ds = Dataset.from_folders(d, ".mrc", d, ".coords").with_roles(["TS_000"], ["TS_001"])
    star = proj / "job001" / "dataset.star"
    ds.write_star(star)
    text = star.read_text()
    assert str(tmp_path) not in text                    # paths stored relative
    moved = tmp_path / "moved"
    shutil.copytree(tmp_path / "raw_data", moved / "raw_data")
    shutil.copytree(proj, moved / "proj")
    back = Dataset.read_star(moved / "proj" / "job001" / "dataset.star")
    assert back.names() == ds.names()
    assert [e.role for e in back.entries] == ["train", "val", ""]
    assert back.entries[2].coords_file == ""
    assert back.entries[0].tomo_file == str((moved / "raw_data" / "TS_000.mrc").resolve())


def test_with_roles_validation(raw_data):
    d, _ = raw_data
    ds = Dataset.from_folders(d, ".mrc", d, ".coords")
    with pytest.raises(ValueError, match="no coordinates"):
        ds.with_roles(["TS_002"], ["TS_000"])
    with pytest.raises(ValueError, match="unknown"):
        ds.with_roles(["nope"], ["TS_000"])
    with pytest.raises(ValueError, match="exactly one"):
        ds.with_roles(["TS_000"], ["TS_000", "TS_001"])


def test_split_val_not_in_train(raw_data):
    d, _ = raw_data
    ds = Dataset.from_folders(d, ".mrc", d, ".coords").with_roles(["TS_001"], ["TS_000"])
    order, split = ds.split()
    assert order == ["TS_001", "TS_000", "TS_002"]
    assert split == [0, 1, 1, 2, 1, 2]


def test_split_val_inside_train(raw_data):
    d, _ = raw_data
    ds = Dataset.from_folders(d, ".mrc", d, ".coords").with_roles(["TS_000", "TS_001"], ["TS_001"])
    order, split = ds.split()
    assert order == ["TS_000", "TS_001", "TS_002"]
    assert split == [0, 2, 1, 2, 1, 2]                  # train range includes the val tomogram


def test_ids_to_names(raw_data):
    d, _ = raw_data
    ds = Dataset.from_folders(d, ".mrc", d, ".coords")
    assert ds.ids_to_names([2, 0]) == ["TS_002", "TS_000"]
    with pytest.raises(ValueError):
        ds.ids_to_names([3])


# ---- legacy num_name.csv reorder ------------------------------------
def _gui_reorder(csv_data, train_list, val_list):
    """Verbatim port of the old GUI's train_ok() reordering (main.py)."""
    csv_data_new, others = [], []
    for item in csv_data:
        if (int(item[2]) in train_list) and int(item[2]) not in val_list:
            csv_data_new.insert(0, item)
        elif int(item[2]) not in val_list:
            others.append(item)
    for item in csv_data:
        if int(item[2]) in val_list:
            csv_data_new.append(item)
    csv_data_new.extend(others)
    val_first = len(train_list) if val_list[0] not in train_list else len(train_list) - 1
    return csv_data_new, [0, len(train_list), val_first, val_first + 1, val_first, val_first + 1]


def test_reorder_matches_old_gui_up_to_train_order():
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randint(2, 8)
        rows = [[f"{rng.randint(1, 99)}.0", f"T{i}", str(i)] for i in range(n)]
        train = sorted(rng.sample(range(n), rng.randint(1, n)))
        val = [rng.randrange(n)]
        new, split = reorder_num_name(rows, train, val)
        old, old_split = _gui_reorder(rows, train, val)
        assert split == old_split
        # the GUI's insert(0, ...) reversed the training rows; everything else identical
        k = split[2]
        assert sorted(map(tuple, new[:k])) == sorted(map(tuple, old[:k]))
        assert new[k:] == old[k:]
        assert [r[2] for r in new[:k]] == sorted(r[2] for r in new[:k])


def test_reorder_makes_ids_select_what_they_name(tmp_path):
    """The CLI bug: ids 2 (train) / 1 (val) must end up first / second."""
    p = tmp_path / "num_name.csv"
    p.write_text("5.0\tA\t0\n6.0\tB\t1\n7.0\tC\t2\n")
    split = reorder_num_name_csv(p, "2", "1")
    names = [ln.split("\t")[1] for ln in p.read_text().splitlines()]
    assert names == ["C", "B", "A"] and split == [0, 1, 1, 2, 1, 2]


def test_reorder_rejects_bad_ids():
    rows = [["1.0", "A", "0"], ["1.0", "B", "1"]]
    with pytest.raises(ValueError):
        reorder_num_name(rows, [0], [5])
    with pytest.raises(ValueError):
        reorder_num_name(rows, [0], [0, 1])


# ---- params / layout -------------------------------------------------
def test_load_legacy_config_prefixes_and_tuples(tmp_path):
    cfg = {"dset_name": ["x"], "base_path": "/b", "label_diameter": 11, "train_set_ids": "0-2"}
    for prefix in ("pre_config=", "train_configs="):
        p = tmp_path / "c.py"
        p.write_text(prefix + json.dumps(cfg))
        out = load_legacy_config(p)
        assert out["dset_name"] == "x" and out["label_diameter"] == 11


def test_load_legacy_config_rejects_garbage(tmp_path):
    p = tmp_path / "c.py"
    p.write_text('{"a": 1}')
    with pytest.raises(ValueError):
        load_legacy_config(p)


def test_params_roundtrip(tmp_path):
    write_params(tmp_path / "job_params.json", "prepare", {"label_diameter": 9}, {"tomograms": "../x"})
    doc = read_params(tmp_path / "job_params.json")
    assert doc["job_type"] == "prepare" and doc["params"] == {"label_diameter": 9}
    (tmp_path / "bad.json").write_text('{"schema_version": 99}')
    with pytest.raises(ValueError):
        read_params(tmp_path / "bad.json")


def test_resolve_inside(tmp_path):
    assert resolve_inside(tmp_path, "a/b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "../x")
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "/etc/passwd")


def test_jobdir_relative(tmp_path):
    j = JobDir(tmp_path / "job001")
    j.mkdirs("labels", "coords")
    assert j.relative(j.labels / "a.mrc") == "labels/a.mrc"
