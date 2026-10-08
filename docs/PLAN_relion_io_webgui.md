# Plan: RELION-style project I/O + web GUI for DeepETPicker

## 1. What the code does today (findings)

**Directory layout is implicit and scattered across modules.** One `base_path` is chosen by the user and everything else is derived by string concatenation in several places:

| Artifact | Where it ends up | Written by | Read by |
|---|---|---|---|
| raw tomograms + `.coords` | `raw_data/` (user-made, any name) | user | `coord_gen.py`, `normalization.py` |
| copied coords + `num_name.csv` | `<base>/coords/` | `utils/coord_gen.py` | `dataloader_DynamicLoad.py:84-102` |
| standardized tomograms | `<base>/data_std/` (or `data_norm/`) | `utils/normalization.py` | dataloader, `test_bash.py` |
| weak labels | `<base>/gaussian<diam>/` | `utils/coords2labels.py:36-40` | dataloader |
| occupancy masks | `<base>/data_ocp/` | `utils/coords2labels.py` | dataloader |
| checkpoints / TB logs | `<base>/runs/<dset>/<long_hyperparam_name>/version_N/checkpoints/` | `train.py:347` | `test.py` |
| predictions | `<checkpoint_dir>/../../PredictedLabels/Coords_All` and `Coords_withArea` | `test.py:90-101` | user, `coordFormatConvert.py`, `coords_to_relion4.py` |

Consequences for the "outputs are not well organized" complaint:
- Predictions live *inside* a TensorBoard version folder, derived by `'/'.join(args.checkpoints.split('/')[:-2])`. Re-running inference overwrites silently; there is no record of which weights/threshold produced which coords.
- `model_name = args.checkpoints.split('/')[-4] ...` (`test.py:29`) breaks if the checkpoint path has a different depth.
- Config is a Python file holding `json` after a `train_configs=` / `pre_config=` prefix, parsed with `.lstrip('train_configs=')`. `lstrip` strips a *character set*, not a prefix, so it only works by luck.
- `num_name.csv` is the only index linking tomogram names to integer ids, in two places (`coords/` and `data_std/`), and `test.py:45` takes `glob(...)[0]` for the data shape, i.e. the *first* tomogram regardless of which one is being tested.
- RELION output is a bolt-on (`coords_to_relion4.py`, `__main__` with a hard-coded `coord_path/Coords_All`), writes `relion4` as the first line, and zeroes angles; it is not wired to the pipeline.

**GUI fragility (PyQt5 `main.py`, 1.9k lines + generated `MainWindow_3D_PP.py`, 1.8k lines).** Logic, path handling, config parsing, and thread control are in the same class. Training/inference run via `threading.Thread` with stdout swapped via a `stdout` shim; failures are caught by bare `except:` that write "… Exception!" and drop the traceback (`coord_gen.py`, `train.py`). `coord_gen_show`'s error branch calls `stdout.flush()` when `stdout` is `None` in CLI mode, which itself raises. Stop is implemented by pid handling. Old pins (`PyQt5==5.15.4`, `pytorch_lightning==1.1.0`, `torch 1.7.1`, `numpy==1.24.0` while code still uses `np.float`, 18 hits, removed in NumPy 1.24) make installs fragile.

**Good news:** the core (dataloader, model, train/test loops, coordinate NMS) is already CLI-callable via `bin/*.py` and takes plain `args`/`cfg` dicts. The I/O contract is concentrated in about six files, so a new layout can be introduced without touching model code.

## 2. Target design

### 2.1 RELION-like project directory
A project folder with numbered job directories, one per run, mirroring RELION's `JobType/jobNNN/` convention. Exact names to be checked against the RELION 5 tomography docs and the Relion-Us spec before freezing (see open questions).

```
<project>/
  default_pipeline.star            # job list + input/output edges (RELION convention)
  DeepETPicker/
    job001/                        # Preprocess
      job.star                     # all parameters for this job (replaces pre_config .py)
      note.txt  run.out  run.err
      RELION_JOB_EXIT_SUCCESS      # or _FAILURE / _ABORTED
      tomograms_std/  coords/  labels/  occupancy/
      manifest.star                # tomo name <-> id <-> files (replaces num_name.csv)
    job002/                        # Train
      job.star  run.out  run.err
      checkpoints/best.ckpt  logs/ (TensorBoard)  metrics.csv
      train_manifest.star          # which tomos were train/val, input = job001
    job003/                        # Predict
      job.star  run.out  run.err
      coords/<tomo>.coords         # DeepETPicker-native
      particles.star               # RELION-style (rlnTomoName, rlnCoordinateX/Y/Z, ...)
      candidate_coords.star        # input to a later manual-review job
  Tomograms/ ...                   # read from upstream RELION jobs, never modified
```

Design rules:
- Inputs are referenced by path from an upstream RELION job (e.g. `Tomograms/jobNNN/tomograms.star`) as well as by plain folder, so no more mandatory `raw_data/` copy convention.
- Every job is self-describing: `job.star` has the full parameter set, plus input job IDs, so a result can always be traced to its weights and threshold.
- Re-running creates a new `jobNNN`; nothing is overwritten.
- One internal `ProjectLayout` class owns every path. No other module concatenates paths.
- Legacy `pre_config=`/`train_configs=` files remain loadable through a one-way importer; old `base_path` datasets can be opened read-only via `--legacy-layout`.

### 2.2 Code structure
```
deepetpicker/
  core/            # unchanged model/dataset/train/test, only path args refactored
  project/         # ProjectLayout, job.star read/write, pipeline.star updater, status files
  io/              # coords readers/writers (.coords/.csv/.txt/.star/.box), RELION STAR
  jobs/            # preprocess.py train.py predict.py: pure functions (params -> files)
  cli/             # `deepetpicker preprocess|train|predict|export|serve`
  web/             # FastAPI backend + frontend (section 2.3)
```
Jobs are pure functions taking a typed params object (pydantic) and a `ProjectLayout`, writing logs to files rather than swapping `sys.stdout`. This is the main fix for GUI fragility: the GUI never shares a process with training.

### 2.3 Web GUI
- **Backend:** FastAPI. Each job is launched as a **subprocess** (`python -m deepetpicker.jobs.train --job-dir ...`), so a CUDA OOM or crash cannot take down the server, stop is `SIGTERM` to the process group, and status comes from the RELION exit-marker files plus `run.out`. Live logs via Server-Sent Events tailing `run.out`.
- **Endpoints (REST + OpenAPI, which is what makes later RELION-Us integration practical):** `POST /jobs/{type}` (validate params, create `jobNNN`, start), `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/log`, `POST /jobs/{id}/abort`, `GET /tomograms`, `GET /tomograms/{name}/slice?axis=z&i=…` (PNG slice for viewer), `GET/PUT /jobs/{id}/coords/{tomo}` (manual edit).
- **Frontend:** a small, dependency-light SPA (plain TypeScript + Vite, or htmx). Pages: Project/job list, per-job parameter form generated from the pydantic JSON schema (so CLI, GUI and docs cannot drift), log/progress view, slice viewer with coordinate overlay and manual add/delete (parity with current PyQt viewer).
- **Integration shape:** run as a stateless service pointed at a project directory (`deepetpicker serve --project DIR --port N`). Auth is left to the host (RELION-Us) at this stage; bind to localhost by default.
- PyQt GUI is kept in `legacy/` until the web GUI reaches viewer parity, then removed.

## 3. Phases

| Phase | Deliverable | Verifiable how |
|---|---|---|
| 0 | Characterization tests + tiny synthetic dataset (one 64³ tomogram, a few particles). Pin a working modern env (`np.float` fix, current torch / lightning) in `pyproject.toml`. | Old pipeline reproduces labels/coords on synthetic data; CPU-only smoke tests in CI |
| 1 | `ProjectLayout`, `job.star` I/O, legacy-config importer, replace `lstrip` parsing, replace `num_name.csv` with manifest, fix `glob(...)[0]` shape bug | Unit tests on layout and round-trip of config |
| 2 | Refactor preprocess/train/predict into `jobs/*` + `cli`; predictions written to `jobNNN/` with `particles.star` (RELION-correct columns, real exit markers, `default_pipeline.star` updates); remove bare `except:`; log to files | End-to-end CLI run on synthetic data (CPU, 1 epoch); STAR output parsed back by a reader |
| 3 | FastAPI backend + job runner + SSE logs + OpenAPI schema | API tests with a fake job; abort test |
| 4 | Frontend: job forms, logs, slice viewer, manual picking | Playwright smoke test (Chromium is available in this environment) |
| 5 | Docs: rewrite README around the project layout, one worked example, and a layout diagram; document `job.star` fields and RELION-Us integration notes | Fresh-clone walkthrough on the sample dataset |
| 6 | Integration with RELION-Us after your testing | Depends on its spec |

Each phase is a separate PR to keep review manageable; Phase 1–2 are the risky ones because they touch training/inference paths.

## 4. Sub-agents: recommendation

**Do not parallelize Phases 0–2.** The path contract (`ProjectLayout` + `job.star` schema) is the thing every other piece depends on, the codebase is small (~8k lines), and splitting that work would produce conflicting layouts and re-derivation of the same context in each agent.

**Use a few after the contract and API schema are frozen (start of Phase 3):**
1. *Frontend agent* (worktree-isolated): builds the SPA against the OpenAPI schema and a mock server; independent of core code.
2. *Test/QA agent*: writes API and Playwright tests and a synthetic dataset generator while backend is implemented.
3. *Review agent* at the end of Phase 2 (read-only, `code-review`) for adversarial checks of the refactor, especially label/coordinate conventions (x,y,z vs z,y,x ordering, 0 vs 1 origin), which silently corrupt results rather than failing.

Docs (Phase 5) I would write myself, because the README notes should be in your voice.

## 5. Risks / things I cannot verify from this container
- No GPU here and torch is not installed; training/inference on real data will need to be validated on your side. I can only run CPU smoke tests on synthetic data.
- Coordinate convention: RELION 5 tomography STAR coordinates (`rlnCoordinateX/Y/Z`, pixel units, tomogram binning, origin offsets in Å) need to match the specific tomogram job you feed in; the current `coords_to_relion4.py` writes RELION-4 style unbinned voxel coordinates with zeroed origins. Needs checking against the RELION docs, not assumed.
- Changing torch / lightning versions may shift numerical results from the published model; the checkpoint from the paper should be re-evaluated on the sample datasets (SHREC2021, EMPIAR-10045) before and after.

## 6. Open questions (needed before Phase 1)
1. **Relion-Us:** I do not know this project and will not guess its API. Is there a repo/spec or doc for how external tools plug in (job launch, status, file conventions)?
2. Target RELION version for the STAR files (RELION 4 vs 5 tomography)?
3. Where will the server run (cluster login node behind SSH tunnel vs. a node with its own port)? This decides auth and whether subprocess launch should go through SLURM.
4. Keep the PyQt GUI in the repo, or drop it once the web GUI works?
5. Can a trained checkpoint from the original paper and one small test tomogram be shared for regression testing?
