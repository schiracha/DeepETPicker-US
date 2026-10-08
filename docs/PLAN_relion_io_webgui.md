# Plan (rev 2): DeepETPicker as RELION-US jobs, RELION-style I/O, retire the PyQt GUI

Rev 2 replaces the earlier "build our own FastAPI GUI" idea. After reading
[schiracha/relion-us](https://github.com/schiracha/relion-us) the web GUI,
job runner, SLURM path, tomogram viewer and manual picker already exist there,
so building a second one here would be duplicated work. Decisions from the
owner: target **RELION 5**; run on a **local GPU workstation now, SLURM
interactive node later**; **remove the old PyQt GUI only after the new path
reproduces everything it does, including the non-obvious parts** (section 4).

All claims below come from reading code; nothing was executed (no GPU/torch in
the planning container).

## 1. What exists

### DeepETPicker (this repo)
- Layout is implicit: one `base_path`, with `coords/`, `data_std/`, `gaussian<d>/`, `data_ocp/`, `runs/...` assembled by string concatenation in about six files. Predictions land inside the TensorBoard version folder (`'/'.join(checkpoint.split('/')[:-2]) + '/PredictedLabels'`, `test.py:90`).
- Config is a `.py` file holding JSON behind a `train_configs=` / `pre_config=` prefix, parsed with `.lstrip(...)` (strips a character set, not a prefix).
- `bin/*.py` CLI exists but is a second, diverging implementation of what `main.py` does (section 4, item 1).
- Core (model, dataloader, train/test loops) takes plain `args`/`cfg` and is reusable.

### RELION-US
- FastAPI backend (`backend/main.py`), vanilla-JS frontend, ~100 files, pytest + Playwright suites (`run_tests.sh`).
- Already has a **DeepETPickerImport** job (`custom_jobs.py`, `converters/deepetpicker_bridge.py`): one `.coords` file -> RELION-5 `particles.star`, with binning factor, Y/Z swap and per-axis mirror. It deliberately rejects the 5-column `Coords_withArea` files.
- Already has the pattern I need: **IsoNet2 jobs** (`isonet_jobs.py`) run an external deep-learning CLI as a real subprocess in a conda env, with SLURM submission, editable command box, per-job `<JobDir>/jobNNN/` output, and a live loss curve on the Progress tab. They go through `job_runner.start_subprocess_job`, not the in-process `custom_jobs` path.
- A tomogram viewer with pick overlay (laid out "the way DeepETPicker's picker is") and an in-browser manual picker (double-click add, right-click delete; saves into the job dir).
- Invariants worth inheriting (ARCHITECTURE.md "Invariants"): paths must stay inside the project dir; runners write only into their own job dir; overwrite means backup first; boundary-aware tomogram name matching (`TS_1` vs `TS_10`); subprocess group kill.

## 2. Target architecture

**Split by responsibility:**
- **DeepETPicker repo** = headless, importable package and a CLI that is a good citizen of a job directory. No GUI, no threads, no `sys.stdout` swapping.
- **RELION-US** = the GUI. DeepETPicker appears there as job types, exactly as IsoNet2 does.

### 2.1 DeepETPicker CLI contract (modelled on `isonet.py`)
Every subcommand takes `--output_dir <JobDir>/jobNNN/` (RELION-US creates it) and never writes outside it. Inputs are RELION STAR files or plain folders. Results are never written in place into an input.

| Command | Inputs | Writes into `--output_dir` |
|---|---|---|
| `deepetpicker prepare` | tomograms (`tomograms.star` or folder), training coords (`particles.star` or folder), label/occupancy diameters, normalization type, n classes | `tomograms_std/`, `coords/`, `labels/`, `occupancy/`, `dataset.star` (replaces both `num_name.csv` files; one row per tomogram: name, id, files, n particles, role train/val/test), `job_params.json` |
| `deepetpicker train` | `dataset.star` from prepare, train/val tomogram **names** (not positional ids), patch/pad/lr/epochs/GPUs | `checkpoints/`, `metrics.csv` (epoch, train/val loss), `job_params.json` |
| `deepetpicker predict` | checkpoint, tomograms (STAR or folder), patch/pad/threshold, GPUs | `coords/<tomo>.coords` (native 4-col `class_id x y z`), `coords_with_area/`, `particles.star`, `job_params.json` |
| `deepetpicker export` | `.coords` folder | `.star` (RELION 3 style), `.box` (EMAN2), per-class split (replaces GUI "coord conversion") |

Exit contract: nonzero exit on any failure, traceback to stderr (no bare `except:`), one progress line per epoch in a documented format so RELION-US can parse it.

### 2.2 RELION-US additions (separate repo, separate PRs)
New job types under a "DeepETPicker" category, same registration route as IsoNet2: `DeepETPickerPrepare`, `DeepETPickerTrain`, `DeepETPickerPredict` (field definitions + command builder + progress source for the Train loss curve). Keep the existing `DeepETPickerImport` for third-party `.coords` files. Predict's STAR conversion must reuse `deepetpicker_bridge` (single source of truth for coordinate conventions) rather than a second converter in this repo; the bridge needs a small batch mode (many `.coords`, one STAR), which its docstring already anticipates.

### 2.3 Layout
RELION-US owns `default_pipeline.star`, numbering and `jobNNN` directories; DeepETPicker only fills a job directory. Chain by file, like IsoNet2: Predict's `--model` default points at the Train job's `checkpoints/best.ckpt`, Train's `--dataset_star` at Prepare's `dataset.star`.

```
<project>/DeepETPicker/job001/   Prepare  (tomograms_std, labels, occupancy, coords, dataset.star)
<project>/DeepETPicker/job002/   Train    (checkpoints/, metrics.csv)
<project>/DeepETPicker/job003/   Predict  (coords/, particles.star)
```
Directory/category names must follow whatever RELION-US's `job_catalog.CUSTOM_JOBS` uses; check there before fixing them.

### 2.4 Local workstation now, SLURM later
Because jobs are plain subprocesses with all state in the job dir, the same command runs locally or via RELION-US's SLURM bridge. Things to keep true now so the SLURM move is free: no hard-coded GPU ids (take `--gpu` / respect `CUDA_VISIBLE_DEVICES`), no absolute paths inside `job_params.json`/`dataset.star` (store project-relative), no interactive prompts, stdout/stderr unbuffered, and `train` resumable from `last.ckpt`.

## 3. Phases

| # | Work | Repo | Done when |
|---|---|---|---|
| 0 | Tiny synthetic dataset (64³ tomogram + a few coords), characterization tests of current preprocess/labels/NMS output, working modern environment (`np.float` -> `float`, pin torch/lightning that run on the workstation's CUDA). Re-evaluate the paper's model on SHREC2021/EMPIAR-10045 before/after to quantify any drift. | DeepETPicker | Old pipeline reproduces on synthetic data; baseline numbers recorded |
| 1 | Package restructure + `ProjectLayout`/`dataset.star` + `job_params.json`; fix items 1-6 in section 4; legacy `pre_config=`/`train_configs=` importer | DeepETPicker | Unit tests; old config files still load |
| 2 | `prepare` / `train` / `predict` / `export` CLI to the contract above; `metrics.csv`; `particles.star` | DeepETPicker | CPU 1-epoch end-to-end on synthetic data; GPU run on workstation matches Phase-0 baseline |
| 3 | Prepare/Train/Predict job types, command builders, Progress source, tests in RELION-US's style; batch mode in `deepetpicker_bridge` | RELION-US | Backend pytest + one Playwright job-popup test |
| 4 | Viewer parity items from section 4 that RELION-US lacks | RELION-US | Parity checklist ticked |
| 5 | Docs: rewrite DeepETPicker README around job dirs and the RELION-US workflow; README change notes written in the owner's voice | both | Fresh-clone walk-through on the sample dataset |
| 6 | Remove PyQt GUI (`main.py`, `MainWindow_3D_PP.py`), PyQt5/pyqtgraph pins | DeepETPicker | **Only after every item in section 4 is "covered" or "dropped on purpose"**, signed off by owner |

One PR per phase; Phases 1-2 are the risky ones because they touch the training and inference paths.

## 4. Old-GUI behaviour audit (hidden functions to carry over, or drop on purpose)

Found by reading `main.py`. This list is the gate for removing the GUI; it should grow as I read the rest of the viewer code and as you use the old GUI.

| # | Hidden behaviour | Where | Why it matters | Disposition |
|---|---|---|---|---|
| 1 | **Training rewrites `coords/num_name.csv` on disk**, reordering tomograms (train ids first, then val, then the rest; train order is also reversed by `insert(0, ...)`), so the "ids" shown in *Dataset list* change after training starts. `data_split` indices are positions in this reordered file. | `main.py` `train_ok` ~L754-780 | `bin/train_bash.py` computes `data_split` the same way but **does not reorder the csv**, so with the CLI, `--train_set_ids`/`--val_set_ids` appear to select by file position, not by the ids given. Needs confirming by a run; if true, CLI users train on the wrong tomograms without any error. | Fix: select by tomogram name via `dataset.star`; no positional ids, no file rewriting |
| 2 | Inference **only runs on tomograms in `data_std/`** (normalized during preprocessing), and writes a fresh `num_name.csv` into `data_std/` from `os.listdir` (unsorted; `tomo_format in i` is a substring test, so `x.mrc.bak` matches). | `test_ok` ~L1020-1035, `bin/test_bash.py` | New tomograms must be standardized with the same method before predict, or results degrade silently. Order depends on filesystem. | Predict normalizes its own inputs per `job_params.json`; sorted, exact-suffix matching |
| 3 | `test.py` reads the **first** tomogram from `glob(tomo_path/*fmt)[0]` for the volume shape regardless of which tomogram is predicted. | `test.py:45` | Wrong shape for any tomogram of a different size. | Fix: per-tomogram shape |
| 4 | Coordinates are **truncated to integers** on import (`astype(float).astype(int)`), so sub-voxel RELION coordinates lose precision before label generation. | `utils/coord_gen.py` | Systematic up-to-1-voxel shift (toward zero). Matters for small particles at binned scale. | Round, not truncate; document. Decide with owner whether to preserve floats |
| 5 | Multi-class models use `num_classes + 1` outputs (background channel) with softmax; single class uses sigmoid. Predict derives `mini_dist` from the smallest `ocp_diameter // 2 + 1` and mean-pool NMS size from the largest. | `train_ok`, `test.py` | Easy to break when refactoring params; affects duplicate removal. | Keep behaviour; derive in one function, test it |
| 6 | Normalization launches one process per tomogram (`Pool(len(dir_list))`) and loads each tomogram fully. | `utils/normalization.py` | Memory blow-up and fork storm on large datasets. | Bounded worker count |
| 7 | Coordinate-format conversion: picks one class, drops the class column, writes `.star` (1 decimal), `.box` (x, y, z rounded), `.coords` (rounded ints) into `<path>_cls<N>_<fmt>/`. | `convert_ok`, `utils/coordFormatConvert.py` | Users have downstream scripts that expect these folders. `.box` has no box size/centre convention encoded. | `deepetpicker export`; RELION-5 particles.star comes via the bridge |
| 8 | Dataset/config round trip: *Save/Load configs* repopulate every tab (preprocess params feed the training and inference tabs, including `ocp_diameter` and `gpu_ids`). | `train_saveConfigs`, `test_load_trainConfigs` | Inference silently depends on `ocp_diameter` from preprocessing. | `job_params.json` chains inputs from upstream job |
| 9 | Stop = injecting an exception into the thread (`stop_thread`) inside the GUI process. | `train_stop`, `test_stop` | Unreliable, can leave GPU memory held. | Process-group kill via RELION-US |
| 10 | Viewer: Gaussian filter, histogram equalization, label overlay with per-class colours, circle diameter/width, saving a **video (mp4/avi)** of a z-range with ROI crop, XY/ZY/XZ linked views, Up/Down arrow for z. | `main.py` viewer, `utils/colors.py` | RELION-US has linked views and overlay; filters and video export need checking. | Audit RELION-US viewer; add missing or drop on purpose |
| 11 | Manual picking: double-click **toggles**: within `circle_diameter // 2` of an existing pick it deletes it, otherwise adds. Per-class id and colour. Saves tab-separated `class x y z` to a user-chosen name. | `mouseClicked`, `mpick_save` | RELION-US uses add / right-click delete and writes STAR, with no class id. | Decide behaviour; keep `.coords` export |

Anything else the old GUI does implicitly (e.g. defaults baked into `train_ok`: AdamW, weight decay 0.01, OneCycleLR, Dice loss, `f_maps=[24,48,72,108]`, `use_bg/use_IP/use_coord=True`) must be moved into explicit, recorded parameters in `job_params.json` rather than hidden in the GUI.

## 5. Sub-agents

- **Phases 0-2: no.** One person (or agent) should own the path/`dataset.star` contract; it is small enough (~8k lines) that splitting it creates conflicting layouts.
- **Phase 3-4: yes, two.** The RELION-US job types and the viewer-parity work live in a different repo and only depend on the frozen CLI contract (section 2.1), so a worktree-isolated agent can do them against a stub CLI while Phase 2 finishes.
- **Review: yes, one read-only agent** after Phase 2 for the audit in section 4 and for coordinate-order bugs (x,y,z vs z,y,x; 0- vs 1-based), which corrupt results without raising errors.
- Docs stay with me/you so the README notes are in your voice.

## 6. Risks and things I could not check
- No GPU or torch in the planning container, so training/inference parity must be verified on your workstation.
- RELION-5 particle coordinate units: RELION-US's bridge treats `rlnCoordinateX/Y/Z` as pixels of the tomogram used, with a binning factor. I have not verified this against the RELION-5 tomography documentation; do that before Phase 2 finalizes `particles.star`.
- Upgrading torch/lightning (from torch 1.7.1 / lightning 1.1.0) may change results. Hence the Phase-0 baseline.
- Push access: GitHub write access for this session to `schiracha/DeepETPicker_Sc` currently fails (403), so nothing here is on the remote yet. Read access to `schiracha/relion-us` works; I cannot push there either.

## 7. Open questions
1. Directory/category names: follow RELION-US's `CUSTOM_JOBS` convention (I will read it in Phase 3), or do you want `DeepETPicker/` fixed now?
2. Should `predict` also be allowed to write float (sub-voxel) coordinates? Currently integers (centroids cast with `.astype(int)`).
3. Item 1 in section 4: do you remember whether results you got from the CLI path looked off? A one-tomogram test on the workstation would settle it before I refactor.
4. Can you share one small tomogram + coords and the published checkpoint for the Phase-0 baseline (or point me at the Zenodo sample set to use)?
5. Okay to put the RELION-US job types in a branch of `schiracha/relion-us` (needs push access to that repo as well)?
