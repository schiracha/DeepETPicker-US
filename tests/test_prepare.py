"""The new torch-free prepare code must reproduce the old preprocessing."""
import mrcfile
import numpy as np
import pytest

from deepetpicker.prepare import labels, normalize
from tests.conftest import load_module

coord_gen = load_module("utils/coord_gen.py")
old_norm = load_module("utils/normalization.py")
old_labels = load_module("utils/coords2labels.py")


@pytest.mark.parametrize("kind,dirname", [("standardization", "data_std"), ("normalization", "data_norm")])
@pytest.mark.parametrize("slab_bytes", [64 * 2**20, 8 * 48 * 48 * 3])   # one slab / several slabs
def test_normalize_matches_old(raw_data, tmp_path, monkeypatch, kind, dirname, slab_bytes):
    raw, _ = raw_data
    monkeypatch.setattr(normalize, "SLAB_BYTES", slab_bytes)
    old = old_norm.InputNorm(str(raw), ".mrc", str(tmp_path / "old"), kind)
    old.single_handle(0)
    normalize.normalize_file(raw / "TS_000.mrc", tmp_path / "new.mrc", kind)
    with mrcfile.open(tmp_path / "old" / dirname / "TS_000.mrc") as a, mrcfile.open(tmp_path / "new.mrc") as b:
        assert b.data.dtype == np.float32
        np.testing.assert_allclose(b.data, a.data, atol=1e-5)


def test_normalize_keeps_voxel_size(raw_data, tmp_path):
    raw, _ = raw_data
    with mrcfile.open(raw / "TS_000.mrc", mode="r+") as m:
        m.voxel_size = 9.36
    normalize.normalize_file(raw / "TS_000.mrc", tmp_path / "o.mrc")
    with mrcfile.open(tmp_path / "o.mrc") as m:
        assert float(m.voxel_size.x) == pytest.approx(9.36, rel=1e-4)


def test_normalize_rejects_constant_volume(tmp_path):
    with mrcfile.new(tmp_path / "c.mrc") as m:
        m.set_data(np.ones((4, 4, 4), np.float32))
    with pytest.raises(ValueError, match="constant"):
        normalize.normalize_file(tmp_path / "c.mrc", tmp_path / "o.mrc")


def _old_label(tmp_path, raw, label_type, diam, num_cls, name="TS_000"):
    base = tmp_path / "base"
    coord_gen.coords_gen(str(raw), ".coords", str(base))
    c = old_labels.Coord_to_Label(str(base), str(raw), ".coords", str(raw), ".mrc", num_cls, label_type, diam)
    c.single_handle(0)
    d = label_type if "ocp" in label_type else label_type + str(diam).split(",")[0]
    with mrcfile.open(base / d / f"{name}.mrc") as m:
        return m.data.copy()


@pytest.mark.parametrize("label_type,diam", [("gaussian", 9), ("sphere", 9), ("cubic", 7)])
def test_labels_match_old_single_class(raw_data, tmp_path, label_type, diam):
    raw, truth = raw_data
    ref = _old_label(tmp_path, raw, label_type, diam, 1)
    new = np.zeros(ref.shape, np.float32)
    labels.fill_labels(new, truth["TS_000"], label_type, [diam], 1)
    np.testing.assert_array_equal(new, ref)


def test_labels_and_ocp_match_old_multiclass_overlapping(tmp_path):
    """Overlapping particles of different classes: the later one overwrites."""
    raw = tmp_path / "raw"
    raw.mkdir()
    with mrcfile.new(raw / "A.mrc") as m:
        m.set_data(np.zeros((30, 30, 30), np.float32))
    coords = np.array([[1, 10, 10, 10], [2, 14, 12, 11], [3, 22, 20, 5], [2, 1, 1, 1], [1, 28, 28, 28]])
    np.savetxt(raw / "A.coords", coords, fmt="%d", delimiter="\t")
    ref_g = _old_label(tmp_path, raw, "gaussian", 9, 3, "A")
    ref_o = _old_label(tmp_path, raw, "data_ocp", "5,7,9", 3, "A")
    g = np.zeros(ref_g.shape, np.float32)
    labels.fill_labels(g, coords, "gaussian", [9], 3)
    o = np.zeros(ref_o.shape, np.float32)
    labels.fill_labels(o, coords, "data_ocp", [5, 7, 9], 3)
    np.testing.assert_array_equal(g, ref_g)
    np.testing.assert_array_equal(o, ref_o)
    assert set(np.unique(g)) <= {0, 1, 2, 3}


def test_even_diameter_rejected():
    with pytest.raises(ValueError, match="odd"):
        labels.fill_labels(np.zeros((9, 9, 9), np.float32), [[1, 4, 4, 4]], "gaussian", [10], 1)


def test_write_label_file_streams_to_disk(raw_data, tmp_path):
    raw, truth = raw_data
    labels.write_label_file(tmp_path / "l" / "TS_000.mrc", raw / "TS_000.mrc", truth["TS_000"], "gaussian", [9], 1)
    with mrcfile.open(tmp_path / "l" / "TS_000.mrc") as m:
        assert m.data.dtype == np.float32
        for _, x, y, z in truth["TS_000"]:
            assert m.data[z, y, x] == 1
