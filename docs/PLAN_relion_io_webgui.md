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
| 1 | **Training rewrites `coords/num_name.csv` on disk**, moving training rows first, then the validation row, then the rest. The id column does not change, only row order, but `data_split` indexes the file **by position**. | `main.py` `train_ok` ~L754-780 | `bin/train_bash.py` computed `data_split` but **never reordered the file**, so with the CLI `--train_set_ids`/`--val_set_ids` did not select what they named (e.g. ids `2`/`1` trained on the first row). From reading the code, not from a run. | **Fixed in Phase 1** for the legacy CLI (`deepetpicker/project/legacy.py`, tested against a port of the GUI algorithm); name-based selection via `dataset.star` replaces it in Phase 2 |
| 2 | Inference **only runs on tomograms in `data_std/`** (normalized during preprocessing), and writes a fresh `num_name.csv` into `data_std/` from `os.listdir` (unsorted; `tomo_format in i` is a substring test, so `x.mrc.bak` matches). | `test_ok` ~L1020-1035, `bin/test_bash.py` | New tomograms must be standardized with the same method before predict, or results degrade silently. Order depends on filesystem. | Predict normalizes its own inputs per `job_params.json`; sorted, exact-suffix matching |
| 3 | `test.py:45` opened the **first** tomogram in the folder (`glob(...)[0]`) with `mrcfile.open` (reads the whole volume, never closed) on every test index, only to print its shape. | `test.py:45` | Not a correctness bug (the value is unused), but loads a multi-GB volume per tomogram for nothing. | **Fixed in Phase 1**: header-only read of the right file |
| 4 | Coordinates are **truncated to integers** on import (`astype(float).astype(int)`), so sub-voxel RELION coordinates lose precision before label generation. | `utils/coord_gen.py` | Systematic up-to-1-voxel shift (toward zero). Matters for small particles at binned scale. | **Changed in Phase 1:** `coords_gen` now rounds. Label generation needs integer voxels, so full-precision floats live in the native `.coords` / `particles.star` and are rounded only for labels (`deepetpicker/io/coords.py`). Labels move by up to 1 voxel vs. models trained before this change; re-check on the paper checkpoint |
| 5 | Multi-class models use `num_classes + 1` outputs (background channel) with softmax; single class uses sigmoid. Predict derives `mini_dist` from the smallest `ocp_diameter // 2 + 1` and mean-pool NMS size from the largest. | `train_ok`, `test.py` | Easy to break when refactoring params; affects duplicate removal. | Keep behaviour; derive in one function, test it |
| 6 | Normalization launches one process per tomogram (`Pool(len(dir_list))`) and loads each tomogram fully. | `utils/normalization.py` | Memory blow-up and fork storm on large datasets. | Bounded worker count |
| 7 | Coordinate-format conversion: picks one class, drops the class column, writes `.star` (1 decimal), `.box` (x, y, z rounded), `.coords` (rounded ints) into `<path>_cls<N>_<fmt>/`. | `convert_ok`, `utils/coordFormatConvert.py` | Users have downstream scripts that expect these folders. `.box` has no box size/centre convention encoded. | `deepetpicker export`; RELION-5 particles.star comes via the bridge |
| 8 | Dataset/config round trip: *Save/Load configs* repopulate every tab (preprocess params feed the training and inference tabs, including `ocp_diameter` and `gpu_ids`). | `train_saveConfigs`, `test_load_trainConfigs` | Inference silently depends on `ocp_diameter` from preprocessing. | `job_params.json` chains inputs from upstream job |
| 9 | Stop = injecting an exception into the thread (`stop_thread`) inside the GUI process. | `train_stop`, `test_stop` | Unreliable, can leave GPU memory held. | Process-group kill via RELION-US |
| 10 | Viewer: Gaussian filter, histogram equalization, label overlay with per-class colours, circle diameter/width, saving a **video (mp4/avi)** of a z-range with ROI crop, XY/ZY/XZ linked views, Up/Down arrow for z. | `main.py` viewer, `utils/colors.py` | RELION-US has linked views and overlay; filters and video export need checking. | **Decided (owner):** video export dropped; histogram equalization dropped; Gaussian filter kept; class colours kept (8, not 14). Done in relion-us PR #83: Gaussian σ slider as a true 3-D blur (the old viewer filtered the whole volume with skimage; its "kernel size" box did nothing and is not offered), eight class colours with a legend. Contrast sliders, linked views and the class overlay already existed. |
| 11 | Manual picking: double-click **toggles**: within `circle_diameter // 2` of an existing pick it deletes it, otherwise adds. Per-class id and colour. Saves tab-separated `class x y z` to a user-chosen name. | `mouseClicked`, `mpick_save` | RELION-US uses add / right-click delete and writes STAR, with no class id. | Decide behaviour; keep `.coords` export |
| 12 | The dataloader repeats every training row whose **first column equals 15** 13 times (`dataloader_DynamicLoad.py` ~L283). That is a SHREC2021 hack for class 15. | `dataset/dataloader_DynamicLoad.py` | For a single-class dataset with 3-column `x y z` files, any particle with x = 15 is silently oversampled 13x. | `prepare` always writes 4 columns (class first), so it cannot trigger on x. The class-15 rule itself is untouched and should become an explicit option |
| 13 | `train_func` swallowed every training exception (bare `except:`) and returned normally. | `train.py` | `bin/train_bash.py` exited 0 after a crash. | Re-raised when run from the command line (no GUI stdout) |
| 14 | Even label/occupancy diameters crash label generation with a broadcasting error (template has `dim` voxels, window `2*int(dim/2)+1`). | `utils/coords2labels.py` | All sample configs use odd diameters, so it never showed up. | `prepare` rejects even diameters up front; `units.suggest_diameters` forces odd |
| 15 | Overlapping particles: a later particle's window overwrites every non-zero voxel it covers, including a neighbour's label. | `utils/coords2labels.py` | Changes multi-class training labels in crowded regions. Models depend on it. | Kept bit-for-bit (tested against the old class); worth a deliberate decision later |

| 16 | Inference with a model that flags (almost) every voxel: `test.py`'s `_nms_v2` has no cap on candidate peaks (the validation `_nms_v2` in `train.py` has: 2000), it loops over them in Python, and `de_dup` is quadratic. | `test.py`, `utils/misc.py:de_dup` | Found on a real run: a 3-epoch model (val_precision 0.012) turned a 344x1376x1376 tomogram into an ~80 minute prediction that would then have stalled in `de_dup`. | Batches with more than 2000 candidates are skipped, counted, and reported in a warning (also in the legacy path) |

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

## 6b. Status (Phases 0-1, this PR)
Done, with tests (`python -m pytest tests`, 31 passing, torch-free, also run by `.github/workflows/tests.yml`):
- `deepetpicker/` package: `units.py`, `io/coords.py`, `project/{layout,params,dataset,legacy}.py`.
- Synthetic tomogram generator (`tests/synth.py`) and characterization tests of `coord_gen`, `normalization`, `coords2labels` (including that the label centre sits at the coordinate, i.e. the x,y,z vs z,y,x order is right).
- Legacy-pipeline fixes: `np.float` removed (NumPy >= 1.24), `lstrip`-based config parsing replaced, `train_bash.py` reorder bug (audit 1), `test.py` whole-volume read (audit 3), unsorted/substring tomogram listing (audit 2, in `bin/test_bash.py` and `coords2labels`), `Pool(len(tomograms))` capped at 4 workers, label volumes `float32` instead of `float64`, rounding instead of truncation (audit 4), a pandas >= 3 crash in `coord_gen` found by the tests.
- `main.py` / `MainWindow_3D_PP.py` were deliberately left alone (to be removed); they still contain `np.float` and the `lstrip` parse.

Not done / not verifiable here: anything importing torch (`train.py`, `test.py`, `dataloader_DynamicLoad.py`, `utils/misc.py`: edits there are syntax-checked only), and `tools/regression.py`, which is only useful once there is a new code path to compare against (Phase 2).

### Numbers for your data
| raw pixel | bin | picking pixel | 150 A particle | DeepETPicker README range 20-30 voxels |
|---|---|---|---|---|
| 3.12 A | 3 | 9.36 A | 16 voxels (label 9, particle 17 forced odd) | below |
| 3.12 A | 2 | 6.24 A | 24 voxels | in range |
| 2.1 A | 3 | 6.3 A | 24 voxels (particle 25, label 13) | in range |
So 2.1 A data at bin 3 suits DeepETPicker directly, while RELION's ~10 A/px initial processing would be bin 5 (10.5 A). Picking and RELION need not use the same binning: picks are rescaled by `coordinate_scale(picking_px, relion_px)` (6.3 -> 10.5 is x0.6), the same factor as RELION-US's `binning_factor`. For the 3.12 A set at bin 3 the particle is below the README's recommended size; bin 2 for picking is worth testing. These ranges come from the README and sample configs, not from a benchmark.

## 6c. Status (Phase 2, this PR)
Commands (run from the repository checkout, after `pip install -e .` or with `python -m deepetpicker`):

```bash
deepetpicker prepare --output_dir P/DeepETPickerPrepare/job001 --tomo_dir RAW --coords_dir RAW \
    --pixel_size 6.3 --particle_diameter_A 150          # or --label_diameter 13 --ocp_diameter 25
deepetpicker train   --output_dir P/DeepETPickerTrain/job002 --prepare_job P/DeepETPickerPrepare/job001 \
    --val_tomo TS_002 --gpu 0
deepetpicker predict --output_dir P/DeepETPickerPredict/job003 --train_job P/DeepETPickerTrain/job002 \
    --tomo_dir NEW_TOMOS --relion_scale 0.6 [--refine_centroids]
deepetpicker export  --coords_dir P/DeepETPickerPredict/job003/coords --fmt star --output_dir OUT --cls 1
deepetpicker compare --a OLD_Coords_All --b P/DeepETPickerPredict/job003/coords --truth RAW --tol 5
```
Tested here (torch-free, 75 tests): `prepare` reproduces the old standardized tomograms, labels and occupancy maps exactly (including the multi-class overlap behaviour); the train and predict *inputs* (legacy config, `coords_index/` symlinks, `num_name.csv`, `data_split`, normalized inference inputs, `particles.star`) and that the Prepare job is never written to; argument Namespaces for `train.py`/`test.py` (an AST check that every `args.*` and `cfg[...]` the legacy code reads is provided); export formats byte-identical to the old converter; the refinement function; `compare`.

**Not run here (needs torch, pytorch-lightning 1.1.0 and a GPU):** `train.py` (job-directory checkpoints, `metrics.csv`, `best.ckpt` copy, error propagation), `test.py` (job-directory outputs, refinement hook), and the real `train`/`predict` commands. Both files are only byte-compiled. In particular the `ModelCheckpoint(dirpath=…, filename=…)` arguments are written for lightning 1.1 from memory of its API.

### Workstation check (you run, then paste the output)
1. `python -m pytest tests -q` in the checkout (should say 75 passed; it needs no GPU).
2. Old vs new on one annotated tomogram you already have a trained model for:
   - old: your usual GUI/bash `Coords_All` folder;
   - new: `prepare` -> `train` (a few epochs is enough for a smoke test) -> `predict` on the same tomograms;
   - `deepetpicker compare --a <old Coords_All> --b <new coords> --truth <your .coords> --tol 5`. Expect near-total agreement if you reuse the old *checkpoint* with `--model`; a freshly trained model will differ.
3. A `--refine_centroids` run next to a plain one, and `compare` the two (median shift should be well under a voxel).
4. Check `train` prints one `epoch=… train_loss=… val_loss=…` line per epoch, and that `checkpoints/best.ckpt`, `last.ckpt`, `metrics.csv` exist.

## 6d. Status (Phase 3, schiracha/relion-us PR #82)
`DeepETPickerPrepare`, `DeepETPickerTrain` and `DeepETPickerPredict` job types exist on `claude/deepetpicker-jobs` of relion-us, built like the IsoNet2 jobs (see that PR and its `docs/ARCHITECTURE.md` section). This repo's side of the contract: `train` keeps `metrics.csv` and `loss_full.png` current (`deepetpicker/plots.py`, written atomically, never allowed to stop a run), and every flag the relion-us command builder can emit parses with `deepetpicker.cli.build_parser()` (checked once by hand; relion-us keeps the flag list in its own test).

Differences from the plan above worth knowing: validation of the fields happens only while the command box still holds a drafted command; the Predict job writes both native `coords/` and `particles.star` itself (so no separate post-step in relion-us); the existing `DeepETPickerImport` converter was left as is.

## 6e. Status (Phase 5, docs)
README: a new "About this fork" (why the layout changed, what differs from the original, that RELION-US needs this fork), a quick start with a worked example for 2.1 Å data binned 3x and a 150 Å particle, and a pointer to RELION-US; the original text follows unchanged under "Original documentation". `docs/job_directories.md` is the reference: every flag, each job directory's contents, `job_params.json` / `dataset.star` / `particles.star` / `metrics.csv` fields, the pixel size and binning table, old-to-new mapping, and what is not verified. `tests/test_docs.py` fails if a flag is missing from the reference, if a README example stops parsing with the real parser, or if the example's numbers differ from what the code computes. Also: a train job's `job_params.json` now records the tomograms it actually used (`resolved`), because `train_tomos` is empty when left to default. Verified while writing: the original label code raises on even diameters (broadcast error), as the audit said.

## 7. Decisions so far
- **RELION-US work goes on a branch, never main:** `claude/deepetpicker-jobs` on `schiracha/relion-us` (created from main at 3904d1e; nothing committed yet). Its own PR, separate from this one.
- **Sub-voxel coordinates: wanted, but they did not exist, and I got this wrong earlier.** I said `get_centroids` already returned float centroids that `Coords_All` then truncated. That function belongs to the evaluation path. The inference path (`test.py`) takes the maximum of the mean-pooled binary prediction (`_nms_v2`), which is an integer voxel, and `de_dup` only chooses among such voxels, so the predictions were always integers. Phase 2 therefore adds an opt-in `--refine_centroids` (`deepetpicker/refine.py`): the centre of mass of the binary prediction in a cube of about one particle radius around each peak. It is off by default, unit-tested on synthetic spheres (recovers a float centre to within 0.2 voxel), but not yet validated on real picks. Float coordinates are carried end to end (native `.coords`, `particles.star`, `export`), and RELION-US's bridge already multiplies by a binning factor, so floats are normal there. I have not confirmed from RELION-5 source that every downstream program reads those columns as doubles; the first real test is one Extract job.
- **No data upload.** Tomograms of 2-5 GB stay on the workstation (and the container's proxy blocks Zenodo, so I could not fetch the sample sets either). Validation split:
  1. In the container: synthetic 64^3 tomograms I generate, CPU only, for unit and end-to-end tests.
  2. On the workstation: I write a `tools/regression.py` that runs the old and the new path on your real data and prints a small text summary (per-tomogram particle counts, mean nearest-neighbour distance between old and new picks, precision/recall against `.coords` if given). You paste the summary back; no volumes leave the machine.

## 8. Open questions
1. Directory/category names: follow RELION-US's `CUSTOM_JOBS` convention (I will read it in Phase 3), or fix `DeepETPicker/` now?
2. Audit item 1 (`num_name.csv` reorder): worth one 2-tomogram CLI run on the workstation before the refactor. `tools/regression.py` will include this check.
3. Which tomogram set and binning do you normally train on (pixel size, patch/label diameter)? Only needed to choose defaults and the synthetic data scale.
