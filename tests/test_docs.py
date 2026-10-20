"""The reference must not drift from the command line, and the README's example must run."""
import re
import shlex
from pathlib import Path

import pytest

from deepetpicker.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
DOC = (REPO / "docs" / "job_directories.md").read_text()
README = (REPO / "README.md").read_text()


def _subparsers():
    ap = build_parser()
    return [a for a in ap._actions if a.dest == "command"][0].choices


@pytest.mark.parametrize("command", ["prepare", "train", "predict", "export", "compare"])
def test_every_flag_is_documented(command):
    section = DOC.split(f"### {command}")[1].split("\n### ")[0].split("\n## ")[0]
    flags = {o for a in _subparsers()[command]._actions for o in a.option_strings if o.startswith("--") and o != "--help"}
    missing = sorted(f for f in flags if f"`{f}`" not in section)
    assert not missing, f"{command}: undocumented flags {missing}"


def test_documented_defaults_match_the_parser():
    sp = _subparsers()
    assert sp["train"].get_default("batch_size") == 8 and sp["train"].get_default("patch_size") == 72
    assert sp["train"].get_default("pad_size") == 12 and sp["train"].get_default("max_epochs") == 60
    assert sp["train"].get_default("lr") == 0.001 and sp["train"].get_default("seg_thresh") == 0.5
    assert sp["predict"].get_default("relion_scale") == 1.0 and sp["compare"].get_default("tol") == 5.0


def _commands(text):
    """`deepetpicker …` commands in fenced blocks, with backslash continuations joined."""
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.split(" #")[0].strip()
            if line.startswith("deepetpicker "):
                out.append(shlex.split(line)[1:])
    return out


def test_readme_examples_parse():
    cmds = _commands(README)
    assert [c[0] for c in cmds] == ["prepare", "train", "predict", "export"]
    ap = build_parser()
    for c in cmds:
        ap.parse_args(c)


def test_readme_example_numbers_are_what_the_code_computes():
    from deepetpicker.units import suggest_diameters
    s = suggest_diameters(6.3, 150)
    assert (s.particle_voxels, s.label_voxels) == (25, 13)       # "24 voxels ... made odd (25, label 13)"
    assert round(150 / 6.3) == 24 and round(6.3 / 10.5, 3) == 0.6
    # the unit table
    assert suggest_diameters(9.36, 150).particle_voxels == 17 and not suggest_diameters(9.36, 150).in_recommended_range
    assert suggest_diameters(6.24, 150).particle_voxels == 25 and suggest_diameters(6.24, 150).in_recommended_range


def test_documented_files_match_what_the_jobs_write(tmp_path):
    """Run prepare for real and compare with the tree in the reference."""
    from deepetpicker.jobs.prepare import PrepareParams, run_prepare
    from deepetpicker.project.layout import JobDir
    from tests.synth import make_raw_data
    make_raw_data(tmp_path / "raw")
    job = JobDir(tmp_path / "job")
    run_prepare(job, PrepareParams(tomo_dir=str(tmp_path / "raw"), coords_dir=str(tmp_path / "raw"),
                                   pixel_size=6.3, particle_diameter_A=150))
    written = {p.name if p.is_file() else p.name + "/" for p in job.root.iterdir()}
    for name in ("tomograms_std/", "coords/", "labels/", "occupancy/", "dataset.star", "job_params.json"):
        assert name in written and name.rstrip("/") in DOC
    import json
    params = json.loads(job.params_json.read_text())["params"]
    assert (params["label_diameter"], params["ocp_diameter"]) == (13, "25")
