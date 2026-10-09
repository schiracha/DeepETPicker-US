"""Pins what the existing preprocessing does, on synthetic data, so the
Phase 2 refactor can be checked against it. Torch-free on purpose."""
import mrcfile
import numpy as np
import pandas as pd

from tests.conftest import load_module

coord_gen = load_module("utils/coord_gen.py")
normalization = load_module("utils/normalization.py")
coords2labels = load_module("utils/coords2labels.py")


def _prepare(raw, base):
    coord_gen.coords_gen(str(raw), ".coords", str(base))
    return base


def test_coord_gen_layout_and_num_name(raw_data, tmp_path):
    raw, truth = raw_data
    base = _prepare(raw, tmp_path / "base")
    assert sorted(p.name for p in (base / "coords").iterdir()) == ["TS_000.coords", "TS_001.coords", "num_name.csv"]
    rows = (base / "coords" / "num_name.csv").read_text().splitlines()
    assert rows == ["5.0\tTS_000\t0", "5.0\tTS_001\t1"]          # count is written as a float; idx is stable
    got = np.loadtxt(base / "coords" / "TS_000.coords", delimiter="\t", dtype=int)
    np.testing.assert_array_equal(got, truth["TS_000"])


def test_coord_gen_rounds_instead_of_truncating(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "A.coords").write_text("1\t10.9\t20.6\t30.4\n")
    coord_gen.coords_gen(str(raw), ".coords", str(tmp_path / "base"))
    assert (tmp_path / "base" / "coords" / "A.coords").read_text().strip() == "1\t11\t21\t30"


def test_normalization_standardizes(raw_data, tmp_path):
    raw, _ = raw_data
    base = tmp_path / "base"
    n = normalization.InputNorm(str(raw), ".mrc", str(base), "standardization")
    assert n.dir_list == ["TS_000.mrc", "TS_001.mrc", "TS_002.mrc"]   # sorted, exact suffix
    n.single_handle(0)
    with mrcfile.open(base / "data_std" / "TS_000.mrc") as m:
        assert m.data.dtype == np.float32
        assert abs(float(m.data.mean())) < 1e-4 and abs(float(m.data.std()) - 1) < 1e-3


def test_labels_cover_the_particles(raw_data, tmp_path):
    raw, truth = raw_data
    base = _prepare(raw, tmp_path / "base")
    c2l = coords2labels.Coord_to_Label(str(base), str(raw), ".coords", str(raw), ".mrc", 1, "gaussian", 9)
    assert c2l.dir_list == ["TS_000", "TS_001"]
    c2l.single_handle(0)
    with mrcfile.open(base / "gaussian9" / "TS_000.mrc") as m:
        lab = m.data
        assert lab.dtype == np.float32
        for _, x, y, z in truth["TS_000"]:
            assert lab[z, y, x] == 1                            # particle centre is labelled class 1
        assert 0 < (lab > 0).mean() < 0.2


def test_ocp_labels_use_per_particle_ids(raw_data, tmp_path):
    raw, truth = raw_data
    base = _prepare(raw, tmp_path / "base")
    c2l = coords2labels.Coord_to_Label(str(base), str(raw), ".coords", str(raw), ".mrc", 1, "data_ocp", "11")
    c2l.single_handle(0)
    with mrcfile.open(base / "data_ocp" / "TS_000.mrc") as m:
        ids = {int(m.data[z, y, x]) for _, x, y, z in truth["TS_000"]}
        assert ids <= set(range(1, 6)) and len(ids) >= 1
