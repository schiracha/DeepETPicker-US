# Job directories: reference

Every `deepetpicker` command fills one directory that you give it with `--output_dir`, and writes nothing outside it. A later command names an earlier job's directory to read from; an earlier job is only ever read. Re-running a command into a new directory never overwrites a previous result.

[RELION-US](https://github.com/schiracha/RELION-US) creates those directories (`<Category>/jobNNN/`), numbers them, and keeps `run.out` / `run.err` and the exit markers itself. Run by hand, you pick the directory names; the examples use the same ones RELION-US does.

For a short walk-through see the *Quick start* in the [README](../README.md). This page lists the details.

## Commands and flags

`python -m deepetpicker <command> --help` prints the same information. Every command exits non-zero on failure: a one-line `error: …` on stderr for bad input (missing file, wrong number of class values, unknown tomogram name), a traceback for anything unexpected.

### prepare

Standardizes the tomograms and paints weak labels and occupancy maps from known particle coordinates. Needs no GPU and no PyTorch.

Give the tomograms as `--tomo_dir` **or** `--tomo_star`, and the known particles as `--coords_dir` **or** `--coords_star`. Give both diameters in voxels, **or** `--pixel_size` and `--particle_diameter_A` and let the job work them out.

| Flag | Default | Meaning |
|---|---|---|
| `--output_dir` | **required** | the job directory to fill, e.g. `DeepETPickerPrepare/job001` |
| `--tomo_dir` | — | folder of `<name>.mrc` tomograms (this OR `--tomo_star`) |
| `--tomo_star` | — | RELION tomograms STAR file (this OR `--tomo_dir`) |
| `--tomo_column` | `rlnTomoReconstructedTomogram` | column of the tomograms STAR holding file names. This default has not been checked against every RELION-5 release; a wrong name stops the job and lists the columns the file has |
| `--tomo_ext` | `.mrc` | tomogram file extension for `--tomo_dir` |
| `--project_dir` | current directory | base for relative paths inside STAR files |
| `--coords_dir` | — | folder of `<name>.coords` files, columns `class x y z` (this OR `--coords_star`) |
| `--coords_star` | — | RELION particles STAR file with `rlnTomoName` and `rlnCoordinateX/Y/Z` (this OR `--coords_dir`) |
| `--coords_ext` | `.coords` | coordinate file extension for `--coords_dir` |
| `--coords_scale` | `1.0` | multiply coordinates by this to reach the tomograms' voxel grid; a number or a ratio of pixel sizes such as `10.5/6.3` |
| `--num_cls` | `1` | number of particle classes; class ids must be 1..N |
| `--label_type` | `gaussian` | shape painted at each particle: `gaussian`, `sphere` or `cubic` |
| `--label_diameter` | — | weak-label diameter in voxels, odd |
| `--ocp_diameter` | — | real particle diameter in voxels, odd, one value per class, comma separated |
| `--pixel_size` | — | Å per voxel of the tomograms |
| `--particle_diameter_A` | — | particle diameter in Å; fills in whichever diameter above was not given |
| `--norm_type` | `standardization` | `standardization` (zero mean, unit standard deviation) or `normalization` (scale to [0, 1]); `predict` applies the same one |

Notes:

- A tomogram with no coordinates is still standardized (and listed in `dataset.star`), but gets no labels. It can be used for prediction but not for training or validation.
- Coordinates are rounded to the nearest voxel (the old code truncated them) and, if they fall outside the volume, moved to its border with a warning on stderr.
- Even diameters are rejected. The label template is centred on a voxel, and the old code crashed on even sizes.
- `--coords_star` accepts pixel coordinates (`rlnCoordinateX/Y/Z`) only, not the centred Ångström columns. `rlnClassNumber` is the class (1 if absent).

### train

Trains the network on a prepare job. Needs PyTorch, pytorch-lightning 1.1.0 and a GPU.

| Flag | Default | Meaning |
|---|---|---|
| `--output_dir` | **required** | the job directory to fill, e.g. `DeepETPickerTrain/job002` |
| `--prepare_job` | **required** | a prepare job directory (read only) |
| `--val_tomo` | **required** | name of the one annotated tomogram used for validation |
| `--train_tomos` | all annotated except `--val_tomo` | comma-separated names. With a single annotated tomogram it trains and validates on itself |
| `--batch_size` | `8` | reduce if the GPU runs out of memory |
| `--patch_size` | `72` | sub-volume size in voxels; a multiple of 8, at least 32 |
| `--pad_size` | `12` | overlap-tile padding in voxels; must be below `patch_size / 2` |
| `--lr` | `0.001` | peak learning rate (one-cycle schedule) |
| `--max_epochs` | `60` | |
| `--seg_thresh` | `0.5` | threshold for the validation metrics, in (0, 1) |
| `--gpu` | `0` | comma-separated GPU ids, e.g. `0` or `0,1` |
| `--resume` | — | checkpoint to continue from |

Tomograms are chosen **by name**, never by position. The optimiser (AdamW, weight decay 0.01, one-cycle schedule), Dice loss and the network widths are the settings the original interface used and are not options here.

### predict

Picks particles with a trained model. Needs the same environment as `train`.

| Flag | Default | Meaning |
|---|---|---|
| `--output_dir` | **required** | the job directory to fill, e.g. `DeepETPickerPredict/job003` |
| `--train_job` | **required** | a train job directory (read only) |
| `--model` | `<train_job>/checkpoints/best.ckpt` | checkpoint to use instead |
| `--tomo_dir`, `--tomo_star`, `--tomo_column`, `--tomo_ext`, `--project_dir` | as for `prepare` | the tomograms to pick in. They do not need coordinates |
| `--already_standardized` | off | the inputs are a prepare job's `tomograms_std/`; otherwise they are standardized here with the method the model was trained with |
| `--batch_size` | `8` | reduce if the GPU runs out of memory |
| `--patch_size` | `72` | normally the value used for training |
| `--pad_size` | `12` | normally the value used for training |
| `--seg_thresh` | `0.5` | probability above which a voxel counts as particle |
| `--gpu` | `0` | comma-separated GPU ids |
| `--refine_centroids` | off | experimental sub-voxel refinement, see below |
| `--relion_scale` | `1.0` | picking-grid voxels → RELION tomogram pixels, applied to `particles.star` only: picking pixel size / RELION tomogram pixel size, as a number or a ratio such as `10/3.12` |
| `--min_score` | `0` | `particles.star` keeps only picks whose score (`rlnAutopickFigureOfMerit`, 0 to 1) is at least this; `coords/` and `coords_with_area/` keep every pick. Leave at 0 to filter afterwards without re-running |

Every pick has a score: the peak of the smoothed binary prediction mask around it, close to 1 for a pick inside a solidly predicted particle and low for a marginal blob (it is not the raw network probability; the prediction is binarized first). It is written to `particles.star` as `rlnAutopickFigureOfMerit`, RELION's own pick-score column, and as the fifth column of `coords_with_area/<tomogram>.coords` (`class_id x y z score`), row for row with `coords/<tomogram>.coords`. If the two files ever disagree in length, no scores are attached rather than risk attaching them to the wrong picks. `export --fmt relion5` on `coords_with_area/` writes a `particles.star` with the scores, for a job run before they were added.

By default a pick is a whole voxel: the maximum of the smoothed binary prediction. With `--refine_centroids` each pick moves to the centre of mass of the binary prediction in a cube of about one particle radius around it, which gives fractional coordinates. It is a heuristic, exact for a symmetric blob inside the cube and biased towards the peak voxel if the cube cuts through the blob, and has not been validated against ground truth.

### export

Converts a folder of native `.coords` into other formats.

| Flag | Default | Meaning |
|---|---|---|
| `--coords_dir` | **required** | folder of `<name>.coords` files (e.g. a predict job's `coords/`) |
| `--output_dir` | **required** | where to write the converted files |
| `--fmt` | **required** | `star` (RELION-3 style, one file per tomogram), `box` (EMAN2), `coords` (rounded integers), or `relion5` (one coordinate-only `particles.star`) |
| `--cls` | all | keep only this class id (the class column is then dropped, as the old interface did) |
| `--scale` | `1.0` | multiply x y z by this; a number or a ratio such as `10/3.12` |

`star`, `box` and `coords` reproduce the old converter byte for byte (tested), so scripts that read its output keep working. `relion5` is not the 13-column RELION-4 table that `utils/coords_to_relion4.py` wrote: it has only the columns listed under `particles.star` below.

### compare

Compares two folders of `.coords`, for example old and new picks, or picks and ground truth, and prints a short text table: counts, how many picks match (mutual nearest neighbours within `--tol` voxels), the median and maximum distance of the matches, and precision and recall against `--truth` if given. It exists so that a check on large tomograms never needs a volume to leave the machine.

| Flag | Default | Meaning |
|---|---|---|
| `--a`, `--b` | **required** | the two folders |
| `--truth` | — | folder of ground-truth `.coords` |
| `--tol` | `5.0` | match distance in voxels (about half a particle radius) |

## What each job directory contains

```
DeepETPickerPrepare/job001/
    tomograms_std/<name>.mrc     standardized copy of every tomogram (float32, voxel size kept)
    coords/<name>.coords         the known particles: class x y z, integer voxels, always 4 columns
    labels/<name>.mrc            weak labels: the class number inside each label, 0 elsewhere
    occupancy/<name>.mrc         particle extents: a running particle number inside each particle
    dataset.star                 one row per tomogram
    job_params.json

DeepETPickerTrain/job002/
    checkpoints/                 epochNNN*.ckpt, last.ckpt, best.ckpt   (written by training)
    logs/tb/                     TensorBoard logs                       (written by training)
    metrics.csv, loss_full.png   one row / one point per validation pass
    coords_index/                derived: links to the prepare job's coords + num_name.csv
    legacy_config.py             derived: the config dict the original training code reads
    job_params.json

DeepETPickerPredict/job003/
    coords/<name>.coords         the picks: class x y z, voxels of the tomogram that was picked in
    coords_with_area/<name>.coords   the same plus a fifth column, see below
    particles.star               the picks as a RELION table
    tomograms_std/               derived: standardized inputs, plus num_name.csv
    legacy_config.py             derived
    job_params.json
```

"Derived" files exist because the original training and inference code reads its inputs that way. They are regenerated on every run and are not the record; `job_params.json` is. The checkpoints, logs, `metrics.csv`, `loss_full.png` and a predict job's `coords/` are produced by the original GPU code, which this repository's tests do not run.

### `job_params.json`

```json
{
  "schema_version": 1,
  "job_type": "train",
  "created": "2026-10-15T17:42:31+00:00",
  "inputs": { "prepare_job": "../../DeepETPickerPrepare/job001" },
  "params": { "val_tomo": "TS_002", "batch_size": 8, "patch_size": 72, "...": "...",
              "resolved": { "train_tomos": ["TS_001"], "val_tomo": "TS_002",
                            "order": ["TS_001", "TS_002", "TS_003"], "data_split": [0, 1, 1, 2, 1, 2] } }
}
```

- `job_type`: `prepare`, `train` or `predict`.
- `inputs`: the earlier jobs this one read, as paths **relative to this job directory**, so the project can be moved.
- `params`: every parameter, exactly as given (so relative paths stay relative if you gave relative paths), including the ones left at their defaults. A train job also has `resolved`: the tomograms it actually used by name, the order the data loader saw them in, and its `data_split`; `train_tomos` above is empty when it was left to default.
- A prepare job records the derived `label_diameter` and `ocp_diameter` when you gave pixel size and particle size instead.

### `dataset.star`

One block, `data_tomograms`, one row per tomogram in name order:

| Column | Meaning |
|---|---|
| `rlnTomoName` | the file name without extension |
| `deepetTomoFile` | the standardized tomogram, relative to the STAR file's directory |
| `deepetCoordsFile` | its `coords/…` file, or `NA` if it has no known particles |
| `deepetNumParticles` | number of known particles (0 if none) |
| `deepetRole` | `train`, `val`, `train+val` or `NA`. Prepare writes `NA`; which tomograms a training run used is recorded in that train job's `job_params.json` |

### Coordinate files

`<name>.coords` is tab-separated, no header: `class x y z`, voxels of the tomogram it belongs to, x fastest (the same axes as the volume's columns, rows and sections). A 3-column `x y z` file is read as class 1. Training copies in a prepare job are rounded to integers; the picks of a predict job are integers unless `--refine_centroids` was used (three decimals).

`coords_with_area/<name>.coords` has a fifth column: the value of the pooled prediction at the pick, between 0 and 1. Higher means more of the surrounding voxels were above threshold. The original code and folder name call it "area".

### `particles.star`

A coordinate-only table, block `particles`:

| Column | Meaning |
|---|---|
| `rlnTomoName` | tomogram name |
| `rlnCoordinateX/Y/Z` | the pick multiplied by `--relion_scale` |
| `rlnClassNumber` | class id |

It has no optics block, angles or origins; RELION-US's **Import from DeepETPicker** bridge and its viewer read this form.

### `metrics.csv` and the progress line

`metrics.csv` has an `epoch` column and every scalar metric the model has logged (`train_loss`, `val_loss`, `val_precision`, `val_recall`, `val_f1`, `val_iou`; `val_f1` is the Dice score and `val_iou` the intersection over union), one row after each training epoch (not for the validation sanity check that lightning runs before epoch 0). The same row is printed on its own line, `epoch=3 train_loss=0.123 val_loss=0.234 …`, and `loss_full.png` (train and validation loss) is redrawn at the same time, atomically, which is what RELION-US shows on the Progress tab.

## Pixel size, binning and diameters

DeepETPicker works in voxels of whatever tomogram it is given. Particle sizes are known in Å, so with `--pixel_size` and `--particle_diameter_A` the job works out the voxel diameters: the particle diameter is the nearest odd whole number of voxels (at least 3), and the label diameter is the nearest odd number to half of it. The sample configs use about that ratio; it is a starting point, not a rule.

The original README recommends binning until the particle is 20–30 voxels across, and the job warns on stderr when it is not. For a 150 Å particle:

| Raw pixel | Bin | Pixel for picking | Particle | In 20–30? |
|---|---|---|---|---|
| 3.12 Å | 3 | 9.36 Å | 16 voxels (17 as odd) | no |
| 3.12 Å | 2 | 6.24 Å | 24 voxels (25) | yes |
| 2.1 Å | 3 | 6.3 Å | 24 voxels (25) | yes |

You do not have to pick on the grid RELION uses. Picks made on a 6.3 Å/px tomogram are brought onto a 10.5 Å/px RELION tomogram by `--relion_scale 0.6` (picking pixel size divided by RELION pixel size); `export --scale` does the same for the other formats. RELION-US's own import bridge calls this factor `binning_factor`.

## What replaced what

| Original | Now |
|---|---|
| everything under one `base_path` | one directory per job |
| `coords/num_name.csv` and `data_std/num_name.csv`, tomograms addressed by position | `dataset.star`, tomograms addressed by name (the index file the original code needs is derived inside the job) |
| `data_std/`, `gaussian<d>/`, `data_ocp/` | `tomograms_std/`, `labels/`, `occupancy/` |
| `configs/*.py` with `pre_config=` / `train_configs=` | `job_params.json` (the old files are still read by `bin/`) |
| `runs/<dataset>/<long name>/version_N/checkpoints/…` | `checkpoints/` with `best.ckpt` and `last.ckpt` |
| `…/PredictedLabels/Coords_All` and `Coords_withArea` | `coords/` and `coords_with_area/` |
| the interface's "coordinate format conversion" | `deepetpicker export` |
| `utils/coords_to_relion4.py` | `deepetpicker export --fmt relion5` (coordinate-only, not the 13-column table) |
| the interface's visualisation and manual picking | RELION-US's viewer and picker |

## Troubleshooting

- **`no tomogram for the coordinates of: 'X' (nearest tomogram: 'Y')`.** A coordinate file must have the same name as its tomogram: `<name>.coords` for `<name>.mrc`. Rename one of them (or copy to a working folder with matching names). The message lists the tomograms it found.
- **`predict` is very slow, or finds nothing, after a short training run.** After only a few epochs the network marks almost every voxel as particle (look at `val_precision` in `metrics.csv`: a value near 0.01 with `val_recall` near 0.8 is exactly that). Every voxel then becomes a candidate peak. Batches with more than 2000 candidates are now skipped and counted in a warning, instead of being worked through one by one, and the warning says so. Train longer (60 epochs is the default for a reason) before judging the picks.
- **What the `val_*` numbers measure.** They are scored on 72-voxel blocks (the default block size) cut around the annotated particles, with the particle placed at a random offset of up to block size / 5 (14 voxels) from the block centre, not on whole tomograms, and the network is also given the position inside the block as an input channel. A model that ignored the image and only learned "a particle is somewhere near the middle" would already score a Dice of roughly 0.14 for a 17-voxel particle (simulated, binary label). So read a validation Dice near that value as "no evidence of detection yet", and judge a model with `predict` followed by `compare --truth` on an annotated tomogram.
- **Training and validation on the same tomogram.** With a single annotated tomogram `train` trains and validates on it, so `val_*` numbers say nothing about how well the model generalises.
- **`--relion_scale 10/3.12`** is accepted: write the two pixel sizes instead of doing the division.

## Not done, not verified

- `train` and `predict` run the original GPU code, edited to write into the job directory. Everything around them (inputs, generated config, outputs afterwards) is tested without a GPU; the GPU runs themselves have not been.
- The class-15 oversampling rule in the data loader (`dataset/dataloader_DynamicLoad.py`: rows whose first column is 15 are repeated 13 times) is left as it was; it was written for the SHREC 2021 dataset.
- When two particles of different classes overlap, the later one overwrites the earlier label inside its window. That is how the original worked, and models were trained with it, so it is reproduced exactly (tested against the original code).
- Only one validation tomogram is supported.
