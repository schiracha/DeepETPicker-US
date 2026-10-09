import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from deepetpicker import compat
from deepetpicker.cli import main
from deepetpicker.io.export import export
from deepetpicker.jobs.predict import PredictParams, prepare_inference_inputs, write_outputs
from deepetpicker.jobs.prepare import PrepareParams, run_prepare
from deepetpicker.jobs.train import TrainParams, prepare_training_inputs
from deepetpicker.project.layout import JobDir
from deepetpicker.project.params import read_params, write_params
from deepetpicker.refine import refine_centroid
from tests.conftest import load_module


def _tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def prepared(raw_data, tmp_path):
    raw, truth = raw_data
    job = JobDir(tmp_path / "DeepETPickerPrepare" / "job001")
    run_prepare(job, PrepareParams(tomo_dir=str(raw), coords_dir=str(raw), label_diameter=9, ocp_diameter="11"))
    return job, truth, tmp_path


# ---- CLI ---------------------------------------------------------------
def test_cli_prepare_and_errors(raw_data, tmp_path, capsys):
    raw, _ = raw_data
    out = tmp_path / "job"
    rc = main(["prepare", "--output_dir", str(out), "--tomo_dir", str(raw), "--coords_dir", str(raw),
               "--label_diameter", "9", "--ocp_diameter", "11"])
    assert rc == 0 and (out / "dataset.star").exists() and (out / "job_params.json").exists()
    rc = main(["prepare", "--output_dir", str(tmp_path / "j2"), "--tomo_dir", str(raw), "--coords_dir", str(raw),
               "--label_diameter", "10", "--ocp_diameter", "11"])
    assert rc == 1 and "error:" in capsys.readouterr().err


def test_cli_does_not_import_torch(prepared):
    job, _, tmp = prepared
    prepare_training_inputs(JobDir(tmp / "t"), TrainParams(prepare_job=str(job.root), val_tomo="TS_001"))
    assert "torch" not in sys.modules and "pytorch_lightning" not in sys.modules


# ---- train inputs ------------------------------------------------------
def test_train_inputs_by_name_and_upstream_untouched(prepared):
    job, _, tmp = prepared
    before = _tree_hash(job.root)
    tj = JobDir(tmp / "DeepETPickerTrain" / "job002")
    tj.root.mkdir(parents=True)
    p = TrainParams(prepare_job=str(job.root), val_tomo="TS_000", train_tomos=["TS_001"])
    cfg_path, split, order = prepare_training_inputs(tj, p)
    assert order == ["TS_001", "TS_000", "TS_002"] and split == [0, 1, 1, 2, 1, 2]
    rows = (tj.root / "coords_index" / "num_name.csv").read_text().splitlines()
    assert [r.split("\t")[1] for r in rows] == order
    assert (tj.root / "coords_index" / "TS_001.coords").is_symlink()
    assert not (tj.root / "coords_index" / "TS_002.coords").exists()          # unannotated
    cfg = compat.read_legacy_config(cfg_path)
    assert cfg["coord_path"] == str(tj.root / "coords_index")
    assert cfg["tomo_path"] == str(job.tomograms_std) and cfg["label_path"] == str(job.labels)
    assert cfg["label_name"] == "gaussian9" and cfg["ocp_diameter"] == "11" and cfg["norm_type"] == "standardization"
    assert cfg["train_set_ids"] == "1" and cfg["val_set_ids"] == "0"
    assert _tree_hash(job.root) == before                                      # the Prepare job was not written to


def test_train_defaults_and_validation(prepared):
    job, _, tmp = prepared
    tj = JobDir(tmp / "t")
    tj.root.mkdir()
    _, split, order = prepare_training_inputs(tj, TrainParams(prepare_job=str(job.root), val_tomo="TS_001"))
    assert order == ["TS_000", "TS_001", "TS_002"] and split == [0, 1, 1, 2, 1, 2]
    with pytest.raises(ValueError, match="no coordinates"):
        prepare_training_inputs(tj, TrainParams(prepare_job=str(job.root), val_tomo="TS_002"))
    for kw in (dict(patch_size=70), dict(pad_size=40), dict(seg_thresh=1.5), dict(val_tomo="")):
        with pytest.raises(ValueError):
            TrainParams(**{**dict(prepare_job="x", val_tomo="a"), **kw}).validate()


def test_train_args_match_old_scripts():
    p = TrainParams(prepare_job="x", val_tomo="a", num_cls=1, gpu_ids=[0, 1], batch_size=4, patch_size=64,
                    pad_size=10, lr=0.002, max_epochs=5, seg_thresh=0.4)
    a = compat.build_train_args(p, "cfg.py", [0, 2, 1, 2, 1, 2], "/out")
    assert (a.num_classes, a.use_sigmoid, a.use_softmax) == (1, True, False)
    assert a.pad_size == [10] and a.block_size == a.val_block_size == 64 and a.val_batch_size == 4
    assert a.gpu_id == [0, 1] and a.learning_rate == 0.002 and a.max_epoch == 5 and a.threshold == 0.4
    assert a.f_maps == [24, 48, 72, 108] and a.test_mode == "val" and a.data_split == [0, 2, 1, 2, 1, 2]
    assert a.use_bg and a.use_IP and a.use_coord and a.test_use_pad and a.meanPool_NMS
    assert (a.optim, a.scheduler, a.loss_func_seg, a.weight_decay) == ("AdamW", "OneCycleLR", "Dice", 0.01)
    assert a.configs == "cfg.py" and a.out_dir == "/out" and a.checkpoints is None
    p.num_cls = 3
    a = compat.build_train_args(p, "cfg.py", [0, 1, 1, 2, 1, 2], "/out")
    assert (a.num_classes, a.use_sigmoid, a.use_softmax) == (4, False, True)        # + background channel


# ---- predict inputs ----------------------------------------------------
def _fake_train_job(prepared):
    job, _, tmp = prepared
    tj = JobDir(tmp / "DeepETPickerTrain" / "job002")
    tj.root.mkdir(parents=True)
    prepare_training_inputs(tj, p := TrainParams(prepare_job=str(job.root), val_tomo="TS_001"))
    write_params(tj.params_json, "train", {}, {"prepare_job": tj.relative(job.root)})
    tj.checkpoints.mkdir()
    (tj.checkpoints / "best.ckpt").write_bytes(b"fake")
    return tj


def test_predict_inputs_normalize_new_tomograms(prepared, raw_data, tmp_path):
    raw, _ = raw_data
    tj = _fake_train_job(prepared)
    pj = JobDir(tmp_path / "DeepETPickerPredict" / "job003")
    p = PredictParams(train_job=str(tj.root), tomo_dir=str(raw), relion_scale=0.5)
    cfg, cfg_path, names, ckpt = prepare_inference_inputs(pj, p)
    assert names == ["TS_000", "TS_001", "TS_002"] and ckpt == tj.checkpoints / "best.ckpt"
    assert (pj.tomograms_std / "TS_002.mrc").exists()
    assert (pj.tomograms_std / "num_name.csv").read_text().splitlines() == ["0\tTS_000", "1\tTS_001", "2\tTS_002"]
    assert cfg["tomo_path"] == str(pj.tomograms_std) and cfg["label_path"] is None
    a = compat.build_predict_args(p, cfg, cfg_path, ckpt, len(names), pj.root)
    assert a.test_mode == "test_only" and a.mini_dist == 11 // 2 + 1 and a.test_idxs == [0, 1, 2]
    assert a.out_dir == str(pj.root) and a.refine_centroids is False and a.num_classes == 1


def test_predict_outputs_particles_star(prepared, tmp_path):
    tj = _fake_train_job(prepared)
    pj = JobDir(tmp_path / "pj")
    (pj.root / "coords").mkdir(parents=True)
    (pj.root / "coords" / "A.coords").write_text("1\t10.000\t20.500\t30.250\n")
    (pj.root / "coords" / "B.coords").write_text("")
    counts = write_outputs(pj, PredictParams(train_job=str(tj.root), relion_scale=2.0), ["A", "B"])
    assert counts == {"A": 1, "B": 0}
    from deepetpicker.io.relion import read_particles_star
    np.testing.assert_allclose(read_particles_star(pj.particles_star, scale=0.5)["A"], [[1, 10, 20.5, 30.25]])


# ---- export ------------------------------------------------------------
def test_export_formats_match_legacy_converter(tmp_path):
    legacy = load_module("utils/coordFormatConvert.py")
    src = tmp_path / "c"
    src.mkdir()
    (src / "A.coords").write_text("1\t10.26\t20.5\t30.75\n2\t1\t2\t3\n1\t4.04\t5.96\t6.5\n")
    data = np.loadtxt(src / "A.coords")[:, 1:]
    for fmt, fn in (("star", legacy.coords2star), ("box", legacy.coords2box), ("coords", legacy.coords2coords)):
        export(src, tmp_path / fmt, fmt)
        fn(data, tmp_path / f"legacy.{fmt}")
        assert (tmp_path / fmt / f"A.{fmt}").read_text() == (tmp_path / f"legacy.{fmt}").read_text(), fmt
    export(src, tmp_path / "c1", "coords", cls=1)
    assert (tmp_path / "c1" / "A.coords").read_text().splitlines() == ["10\t20\t31", "4\t6\t6"]
    with pytest.raises(ValueError, match="class 9"):
        export(src, tmp_path / "c9", "coords", cls=9)


def test_export_relion5(tmp_path):
    src = tmp_path / "c"
    src.mkdir()
    (src / "A.coords").write_text("1\t10\t20\t30\n")
    (out,) = export(src, tmp_path / "o", "relion5", scale=0.6)
    assert out.name == "particles.star" and "rlnCoordinateX" in out.read_text()


# ---- sub-voxel refinement ---------------------------------------------
def test_refine_recovers_float_centre_of_a_sphere():
    zz, yy, xx = np.indices((41, 41, 41))
    for c in [(20.0, 20.0, 20.0), (20.4, 19.7, 20.25), (19.6, 20.3, 20.1)]:
        mask = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= 6 ** 2)
        peak = [int(round(v)) for v in c]
        dz, dy, dx = refine_centroid(mask, *peak, radius=8)
        assert abs(peak[0] + dz - c[0]) < 0.2 and abs(peak[1] + dy - c[1]) < 0.2 and abs(peak[2] + dx - c[2]) < 0.2


def test_refine_edge_cases():
    assert refine_centroid(np.zeros((9, 9, 9)), 4, 4, 4, 2) == (0.0, 0.0, 0.0)
    assert refine_centroid(np.ones((5, 5, 5)), 99, 99, 99, 2) == (0.0, 0.0, 0.0)


# ---- static checks on the code we cannot run without torch -------------
def _args_attributes(path):
    import ast
    tree = ast.parse((compat.REPO / path).read_text())
    loads, stores = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args":
            (stores if isinstance(n.ctx, ast.Store) else loads).add(n.attr)
    return loads - stores


@pytest.mark.parametrize("path", ["train.py", "test.py", "dataset/dataloader_DynamicLoad.py",
                                  "model/model_loader.py", "model/residual_unet_att.py", "utils/loss.py"])
def test_every_args_attribute_the_legacy_code_reads_exists(path):
    """train/predict hand the legacy modules a Namespace; an attribute they read
    but we never set (and the parser has no default for) would only fail on a GPU run."""
    p = TrainParams(prepare_job="x", val_tomo="a")
    ns_train = compat.build_train_args(p, "c", [0, 1, 0, 1, 0, 1], "/o")
    cfg = {"num_cls": 1, "ocp_diameter": "11", "model_name": "ResUNet"}
    ns_pred = compat.build_predict_args(PredictParams(train_job="x"), cfg, "c", "k", 1, "/o")
    for ns, label in ((ns_train, "train"), (ns_pred, "predict")):
        missing = sorted(a for a in _args_attributes(path) if not hasattr(ns, a))
        if path == "train.py" and label == "predict":
            continue
        if path == "test.py" and label == "train":
            continue
        assert not missing, f"{path} reads args.{missing} which the {label} Namespace lacks"


def test_legacy_entry_points_compile():
    import py_compile
    for f in ("train.py", "test.py", "dataset/dataloader_DynamicLoad.py"):
        py_compile.compile(str(compat.REPO / f), doraise=True)


def test_legacy_config_has_every_key_the_legacy_code_reads(prepared):
    import ast
    job, _, tmp = prepared
    tj = JobDir(tmp / "t")
    tj.root.mkdir()
    cfg_path, _, _ = prepare_training_inputs(tj, TrainParams(prepare_job=str(job.root), val_tomo="TS_001"))
    cfg = compat.read_legacy_config(cfg_path)
    optional = {"classes", "classes_sort", "paf_path"}           # only read behind `if 'x' in cfg`
    wanted = set()
    for f in ("train.py", "test.py", "dataset/dataloader_DynamicLoad.py", "utils/misc.py"):
        for n in ast.walk(ast.parse((compat.REPO / f).read_text())):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str):
                v = n.value
                name = v.attr if isinstance(v, ast.Attribute) else getattr(v, "id", "")
                if name in ("cfg", "train_cfg", "val_cfg"):
                    wanted.add(n.slice.value)
    assert not (wanted - optional - set(cfg)), sorted(wanted - optional - set(cfg))


def test_compare(tmp_path, capsys):
    a, b, t = (tmp_path / d for d in "abt")
    for d in (a, b, t):
        d.mkdir()
    (t / "X.coords").write_text("1\t10\t10\t10\n1\t50\t50\t50\n1\t90\t90\t90\n")
    (a / "X.coords").write_text("1\t10\t10\t10\n1\t50\t50\t50\n1\t200\t200\t200\n")
    (b / "X.coords").write_text("1\t11\t10\t10\n1\t50\t51\t50\n")
    assert main(["compare", "--a", str(a), "--b", str(b), "--truth", str(t), "--tol", "3"]) == 0
    out = capsys.readouterr().out
    assert "TOTAL nA=3 nB=2 matched=2" in out and "X" in out
