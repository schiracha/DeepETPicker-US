import json

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from deepetpicker.io.relion import read_particles_star, read_tomograms_star, write_particles_star
from deepetpicker.jobs.prepare import PrepareParams, run_prepare
from deepetpicker.project.dataset import Dataset
from deepetpicker.project.layout import JobDir
from tests.conftest import load_module

old_pre = {m: load_module(f"utils/{m}.py") for m in ("coord_gen", "normalization", "coords2labels")}


def _params(raw, **kw):
    base = dict(tomo_dir=str(raw), coords_dir=str(raw), label_diameter=9, ocp_diameter="11")
    base.update(kw)
    return PrepareParams(**base)


def test_prepare_reproduces_old_pipeline(raw_data, tmp_path):
    raw, truth = raw_data
    job = JobDir(tmp_path / "DeepETPickerPrepare" / "job001")
    ds = run_prepare(job, _params(raw))

    # old pipeline on the same inputs
    base = tmp_path / "old"
    old_pre["coord_gen"].coords_gen(str(raw), ".coords", str(base))
    old_pre["normalization"].InputNorm(str(raw), ".mrc", str(base), "standardization").handle_parallel()
    for lt, d in (("gaussian", 9), ("data_ocp", "11")):
        old_pre["coords2labels"].Coord_to_Label(str(base), str(raw), ".coords", str(raw), ".mrc", 1, lt, d).gen_labels()

    for name in ("TS_000", "TS_001"):
        for new_dir, old_dir in (("tomograms_std", "data_std"), ("labels", "gaussian9"), ("occupancy", "data_ocp")):
            with mrcfile.open(job.root / new_dir / f"{name}.mrc") as a, mrcfile.open(base / old_dir / f"{name}.mrc") as b:
                np.testing.assert_allclose(a.data, b.data, atol=1e-5, err_msg=f"{name} {new_dir}")
        np.testing.assert_array_equal(
            np.loadtxt(job.coords / f"{name}.coords", dtype=int), truth[name])

    assert ds.names() == ["TS_000", "TS_001", "TS_002"]
    assert not (job.labels / "TS_002.mrc").exists()                 # unannotated: standardized only
    assert (job.tomograms_std / "TS_002.mrc").exists()
    back = Dataset.read_star(job.dataset_star)
    assert [e.n_particles for e in back.entries] == [5, 5, 0]
    doc = json.loads(job.params_json.read_text())
    assert doc["job_type"] == "prepare" and doc["params"]["label_diameter"] == 9


def test_prepare_from_relion_star_inputs(raw_data, tmp_path):
    raw, truth = raw_data
    tomo_star = tmp_path / "tomograms.star"
    starfile.write({"global": pd.DataFrame({
        "rlnTomoName": ["TS_000", "TS_001", "TS_002"],
        "rlnTomoReconstructedTomogram": [str(raw / f"TS_00{i}.mrc") for i in range(3)]})}, tomo_star)
    # particles on a grid 2x coarser than the tomograms: scale 2 brings them back
    coords_star = tmp_path / "particles.star"
    rows = [(n, *(c[1:] / 2.0)) for n, arr in truth.items() for c in arr]
    starfile.write({"particles": pd.DataFrame(rows, columns=["rlnTomoName", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ"])}, coords_star)
    job = JobDir(tmp_path / "job")
    p = PrepareParams(tomo_star=str(tomo_star), coords_star=str(coords_star), coords_scale=2.0,
                      label_diameter=9, ocp_diameter="11")
    run_prepare(job, p)
    np.testing.assert_array_equal(np.loadtxt(job.coords / "TS_000.coords", dtype=int), truth["TS_000"])


def test_diameters_from_angstrom(raw_data, tmp_path, capsys):
    raw, _ = raw_data
    job = JobDir(tmp_path / "job")
    p = PrepareParams(tomo_dir=str(raw), coords_dir=str(raw), pixel_size=9.36, particle_diameter_A=150)
    run_prepare(job, p)
    doc = json.loads(job.params_json.read_text())["params"]
    assert (doc["label_diameter"], doc["ocp_diameter"]) == (9, "17")
    assert "recommends 20-30" in capsys.readouterr().err


def test_out_of_volume_coordinates_are_clipped(raw_data, tmp_path, capsys):
    raw, _ = raw_data
    (raw / "TS_001.coords").write_text("1\t47.6\t3\t-2\n")
    job = JobDir(tmp_path / "job")
    run_prepare(job, _params(raw))
    assert (job.coords / "TS_001.coords").read_text().strip() == "1\t47\t3\t0"
    assert "outside the volume" in capsys.readouterr().err


@pytest.mark.parametrize("kw,msg", [
    (dict(label_diameter=10), "odd"),
    (dict(ocp_diameter="11,13"), "1 comma"),
    (dict(label_type="blob"), "label_type"),
    (dict(label_diameter=None, ocp_diameter=None), "pixel_size"),
    (dict(coords_dir=None), "exactly one"),
])
def test_bad_params(raw_data, tmp_path, kw, msg):
    raw, _ = raw_data
    with pytest.raises(ValueError, match=msg):
        run_prepare(JobDir(tmp_path / "job"), _params(raw, **kw))


def test_class_id_out_of_range(raw_data, tmp_path):
    raw, _ = raw_data
    (raw / "TS_001.coords").write_text("3\t5\t5\t5\n")
    with pytest.raises(ValueError, match="class ids"):
        run_prepare(JobDir(tmp_path / "job"), _params(raw))


def test_wrong_tomo_column_lists_columns(tmp_path):
    star = tmp_path / "t.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["a"], "rlnSomething": ["x"]})}, star)
    with pytest.raises(KeyError, match="rlnSomething"):
        read_tomograms_star(star)


def test_particles_star_roundtrip(tmp_path):
    c = {"A": np.array([[1, 1.5, 2.25, 3.0], [2, 4, 5, 6]]), "B": np.array([[1, 7, 8, 9.0]])}
    write_particles_star(tmp_path / "p.star", c, scale=2.0)
    back = read_particles_star(tmp_path / "p.star", scale=0.5)
    for k in c:
        np.testing.assert_allclose(back[k], c[k])
