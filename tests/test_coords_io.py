import numpy as np
import pytest

from deepetpicker.io.coords import read_coords, to_voxel_indices, write_coords


def test_roundtrip_float(tmp_path):
    c = np.array([[1, 10.25, 20.5, 30.125], [2, 1.0, 2.0, 3.0]])
    write_coords(tmp_path / "a.coords", c)
    np.testing.assert_allclose(read_coords(tmp_path / "a.coords"), c, atol=1e-3)


def test_three_column_is_class_one(tmp_path):
    (tmp_path / "a.coords").write_text("1 2 3\n4 5 6\n")
    np.testing.assert_array_equal(read_coords(tmp_path / "a.coords"), [[1, 1, 2, 3], [1, 4, 5, 6]])


def test_bad_columns(tmp_path):
    (tmp_path / "a.coords").write_text("1 2 3 4 5\n")
    with pytest.raises(ValueError, match="expected 3 or 4"):
        read_coords(tmp_path / "a.coords")


def test_integer_write_rounds_not_truncates(tmp_path):
    write_coords(tmp_path / "a.coords", np.array([[1, 10.9, 20.6, 30.4]]), integer=True)
    assert (tmp_path / "a.coords").read_text().strip() == "1\t11\t21\t30"


def test_to_voxel_indices():
    np.testing.assert_array_equal(to_voxel_indices([[0.4, 1.6, 2.5]]), [[0, 2, 2]])
