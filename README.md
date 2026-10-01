# Parkinson-Data-Challenge

Solution to the [SFMN DaTscan Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/) on DrivenData: predict, from a 3D DaTscan SPECT volume (NIfTI), the probability that the exam is pathologic (`is_pathologic`).

## Leaderboard result

| | |
|---|---|
| **Private leaderboard rank** | **#25** |
| **Team / user** | Dufouranto |
| **Submissions** | 14 |
| **Scores shown on the leaderboard** | **0.2944** (log loss) and **0.9428** (AUROC) |

Leaderboard: <https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/leaderboard/>

The submission is a **5-fold ensemble of small 3D ResNet-18 + CBAM classifiers**, each calibrated with temperature scaling, trained on a striatum-centered crop of every scan and on a training set enriched with synthetically down-sampled scans (see [Pipeline](#pipeline)). Everything needed to reproduce it is in this repo; the exact commands are in [Reproducing the #25 run](#reproducing-the-25-run).

---

## Pipeline

```
 raw NIfTI (train_set/niftis, train_labels.csv)
        │
        │  00_split_by_site.py        cluster voxel spacings -> 9 pseudo-"sites"
        ▼
 site_assignments.csv  (uid, is_pathologic, sx, sy, sz, site)
        │
        │  04_resample_low_res_sites.py   synthetic low-resolution copies of high-res scans
        ▼
 resample_train_set/niftis + site_assignments.csv (real + synthetic, with source_uid)
        │
        │  module/preprocess_dataset.py   striatum-centered crop -> 90^3 @ 1.5 mm, normalised .npy
        ▼
 cache_dir/*.npy
        │
        │  train.py  (+ module/dataloader.py, module/model.py)
        │  5-fold StratifiedGroupKFold, augmentation, calibration
        ▼
 results/fold_{1..5}/{model.pt, config.json}  +  results/fallback_probability.json
        │
        │  main.py   (runs inside the DrivenData container)
        ▼
 submission.csv   (uid, is_pathologic)
```

### Why "sites" and why synthetic resampling?

The training labels do not say which centre/scanner produced each exam, but acquisition resolution is a strong proxy for it. Two things follow:

1. **Stratification.** Folds must contain every kind of scanner, so `00_split_by_site.py` clusters the voxel spacings (k-means, 9 clusters) into pseudo-sites, and the CV stratifies on `site + label`.
2. **Shortcut learning.** Every scan is finally resampled to the same 1.5 mm grid, but a 1.2 mm scan is slightly *down*-sampled while a 3.3 mm or 4.4 mm scan is strongly *up*-sampled. The interpolation blur is a signature of the site, not of the disease, and a network can learn it. Sites 1, 2, 3, 6, 7 and 9 are rare (9 to 52 subjects each, versus 679 for site 5), so `04_resample_low_res_sites.py` fabricates extra copies of high-resolution scans degraded to those rare sites' median resolution. The model then sees every pathology label at every resolution.

In the reference run this turned 1,362 real exams into **4,174 rows** (1,362 real + 2,812 synthetic).

---

## Repository layout

```
.
├── 00_split_by_site.py            # step 0 – pseudo-site assignment from voxel spacing
├── 04_resample_low_res_sites.py   # step 1 – synthetic low-resolution NIfTIs
├── train.py                       # step 3 – cross-validated training
├── main.py                        # inference entry point for the submission container
├── project.toml                   # runtime dependencies (uv / Python 3.12, torch 2.12 + cu129)
├── module/
│   ├── preprocess_dataset.py      # step 2 – NIfTI -> cropped, normalised .npy
│   ├── dataloader.py              # Dataset, augmentations, group-aware CV splits
│   └── model.py                   # 3D ResNet + attention, losses, calibration, metrics
└── results/                       # written by train.py, shipped with the submission
    ├── fallback_probability.json
    └── fold_1 … fold_5/{model.pt, config.json}
```

`train.py` and `main.py` import from the `module` package (`from module.model import ...`), so run them from the repository root. Scripts `00_` and `04_` are standalone.

### File reference

#### `00_split_by_site.py`
Reads the voxel spacing (`sx, sy, sz`) from each NIfTI header, clusters them with k-means into `N_SITES = 9` groups, and numbers the sites by increasing spacing so that numbering is reproducible. It writes `site_assignments.csv`, `site_summary.csv` and one `site_XX/` directory per site (symlinks by default). It is configured by constants at the top of the file (`DATA`, `OUTPUT`, `N_SITES`, `USE_SYMLINKS`, `RANDOM_STATE`), not by CLI arguments.

```bash
python 00_split_by_site.py
```

#### `04_resample_low_res_sites.py`
Builds synthetic scans:

1. Scans all native NIfTI headers.
2. Picks the "under-represented" sites (fewer than `--small-site-threshold` subjects) and uses the median spacing of each as a target resolution.
3. For every subject whose native resolution is better than `--better-than` on all three axes, writes a down-sampled copy for each target that is strictly coarser on all three axes (by at least `--min-downsample-ratio`). It never up-samples, and applies a Gaussian anti-aliasing filter before down-sampling.
4. Writes `output-dir/niftis/*.nii.gz` and a combined `site_assignments.csv` containing real and synthetic rows, with `source_uid` (the real subject each row derives from) and `is_synthetic`.

| Option | Default | Meaning |
|---|---|---|
| `--input-dir` | required | native NIfTI folder |
| `--site-assignments` | required | CSV from step 0 |
| `--output-dir` | `resample_train_set` | output folder |
| `--better-than` | `2.5 2.5 2.5` | only scans finer than this (mm) are degraded |
| `--small-site-threshold` | `100` | sites with fewer subjects are targets |
| `--min-target-count` | `4` | min real subjects for a site to define a target |
| `--min-downsample-ratio` | `1.05` | min target/native ratio per axis |
| `--jobs` | `4` | parallel workers |
| `--dry-run` | off | print targets and job counts only |

#### `module/preprocess_dataset.py`
Turns a NIfTI into a fixed-size `.npy` volume centred on the striatum. Defaults: `DEFAULT_CROP_SIZE = (90, 90, 90)` voxels at `DEFAULT_TARGET_SPACING = (1.5, 1.5, 1.5)` mm, i.e. a 135 mm cube. Steps:

1. Load, reorient to LPS, fix inverted or negative intensities.
2. Rough head mask from the top-K brightest voxels (K derived from a ~1,650 cm³ maximum brain volume and the native voxel size, so it is resolution-independent), and its centre of mass.
3. Three-pass "funnel": a Gaussian distance weighting around the current centre (σ = 40, 30, 20 mm) followed by a 2.5 mm smoothing and a top-K threshold (K from 30,000 / 12,000 / 5,000 mm³ target volumes). Each pass refines the centre of mass onto the striatal uptake.
4. The left-right (X) coordinate of the final centre is **re-anchored to the rough head centre**, because the refined centre is pulled towards the more avid hemisphere, which would correlate the framing with the very asymmetry that signals disease.
5. Crop in native space (with padding if needed), resample to the target grid with a B-spline, normalise between the 1st and 99.95th percentile of the crop, clip to [0, 1], save as `<uid>.npy`.

`preprocess_files()` is shared with `main.py`, so inference uses exactly the same code as training. `quiet=True` suppresses per-file logs, since logging test-set identities is forbidden by the competition rules.

```bash
python module/preprocess_dataset.py \
    --input_dir  /path/to/niftis \
    --cache_dir  /path/to/cache_dir \
    --jobs 24
# optional: --crop-size 90 90 90 --target-spacing 1.5 1.5 1.5
```

#### `module/dataloader.py`
* `DaTscanDataset`: loads `<uid>.npy` from the cache and, in training mode, applies `random_augment`. Returns `(volume, label, uid)`.
* `AugmentConfig` / `random_augment`: augmentation stack, in order: synthetic hot-spot "blobs" (mimics reconstruction artefacts, added in numpy before the geometric transforms so they get deformed too), Gaussian noise, random affine, 3D elastic deformation, intensity shift, Gaussian blur, coarse dropout, left-right flip. Every probability and magnitude is a `train.py` CLI flag.
* `worker_init_fn`: re-seeds each DataLoader worker every epoch so augmentations are not repeated from one epoch to the next.
* `get_stratified_group_splits`: `StratifiedGroupKFold` on `site + label`, **grouped by `source_uid`**, so a subject and all its synthetic variants always land in the same fold (no leakage). It raises if any group is found on both sides of a fold.
* `get_stratified_splits`: the ungrouped version, kept for datasets without synthetic copies.

#### `module/model.py`
* `ResNet3D` / `build_model`: a light 3D ResNet (`resnet10` = 1 block per stage, `resnet18` = 2). The stem is a single stride-1 3×3×3 conv (the input is already small, so no early down-sampling). Four stages with `base_channels × {1, 2, 4, 8}` channels, strides `{1, 2, 2, 2}`. Optional attention on each residual branch: `none`, `se` (channel), or `cbam` (channel + spatial, useful for "look at the left vs right striatum"). Head: global `avg`/`max`/`avg_max` pooling, then a 1- or 2-layer MLP with dropout, giving one logit.
* `FocalLoss`: optional alternative to BCE for training.
* `fit_temperature_robust`: temperature scaling on validation logits, averaged in log-space over 50 bootstrap resamples.
* `collect_logits`, `evaluate_metrics` (log loss and AUROC), `probs_from_logits`, `EarlyStopping`, `mixup_batch`, and other helpers.

#### `train.py`
Runs the full k-fold cross-validation (default 5 folds). For each fold: build the datasets and loaders, train with Adam, `ReduceLROnPlateau` on validation log loss, and early stopping. **Checkpoint selection always uses validation log loss**, whatever the training loss is. Then it restores the best weights, fits the calibration temperature on the fold's validation logits and writes:

* `results/fold_k/model.pt`: best weights.
* `results/fold_k/config.json`: architecture hyper-parameters, `temperature`, validation metrics, `crop_size` / `target_spacing`, and the full CLI arguments (`train_args`).
* `results/fallback_probability.json`: global training prevalence, used at inference when a scan cannot be processed.

Training is resumable: a fold that already has a `model.pt` and a `config.json` with identical hyper-parameters is skipped on restart.

#### `main.py`
The inference entry point for the DrivenData runtime container (`data/submission_format.csv` and `data/niftis/*.nii.gz` in, `submission.csv` out):

1. Read `crop_size` / `target_spacing` from the folds' `config.json` (and fail if folds disagree), so training and inference preprocessing cannot drift apart.
2. Preprocess the test NIfTIs into a temporary cache (no per-exam logging).
3. Rebuild every fold's model from its `config.json`, load `model.pt`, and predict.
4. Convert each fold's logits to probabilities with that fold's own temperature, then **average the fold probabilities**.
5. Any exam that failed to preprocess or load gets `fallback_probability.json` (or 0.5 if that file is missing).

#### `project.toml`
Runtime dependencies for the container: Python 3.12, `torch==2.12.1+cu129`, `monai`, `SimpleITK`, `nibabel`, `scikit-learn`, `pandas`, `loguru`, `tqdm`, and others.

---

## How to run

All commands are run from the repository root.

```bash
# 0. Pseudo-site assignment (edit the DATA / OUTPUT constants first)
python 00_split_by_site.py

# 1. Synthetic low-resolution scans
python 04_resample_low_res_sites.py \
    --input-dir        /data/train_set/niftis \
    --site-assignments /data/train_set/site_assignments.csv \
    --output-dir       /data/resample_train_set \
    --better-than 2.5 2.5 2.5 \
    --jobs 8

# 2. Preprocess BOTH the synthetic and the native scans into the same cache
python module/preprocess_dataset.py --input_dir /data/resample_train_set/niftis \
    --cache_dir /data/resample_train_set/iso_cache --jobs 24
python module/preprocess_dataset.py --input_dir /data/train_set/niftis \
    --cache_dir /data/resample_train_set/iso_cache --jobs 24

# 3. Train (5-fold CV)
python train.py \
    --train_labels /data/resample_train_set/site_assignments.csv \
    --cache_dir    /data/resample_train_set/iso_cache \
    --results_dir  results

# 4. Inference (inside the submission container: data/ + results/ present)
python main.py
```

Sanity check after step 2: the cache should contain one `.npy` per row of the combined CSV (4,174 in the reference run).

### Reproducing the #25 run

The command below is the reference training run (ResNet-18, base 16 channels, CBAM, max pooling, single linear head, BCE loss, seed 3):

```bash
python train.py \
  --train_labels /data/resample_train_set/site_assignments.csv \
  --cache_dir    /data/resample_train_set/iso_cache \
  --results_dir  results/ResNet18_res22 \
  --model ResNet --depth resnet18 --base_channels 16 \
  --attention cbam --mlp_layers 1 --pool_mode max --dropout 0.25 \
  --loss bce --seed 3 \
  --batch_size 32 --epochs 300 --patience 30 \
  --lr 0.001 --lr_patience 15 --lr_factor 0.3 \
  --max_rotate_deg 40 --max_translate_vox 20 --max_scale_delta 0.15 \
  --p_noise 0.95 --noise_std 0.12 \
  --p_elastic 0.5 --elastic_alpha 5 \
  --p_shift_intensity 0.2 --intensity_shift_delta 0.2 \
  --p_blur 0.55 --blur_sigma_max 0.8 \
  --dropout_holes 5 --dropout_frac 0.09 \
  --num_workers 19
```

Its five folds (`results/ResNet18_res22/fold_1` … `fold_5`, plus `fallback_probability.json`) are what `main.py` ensembles into the leaderboard submission (log loss **0.2944**, AUROC **0.9428**, rank **#25**).

## Design notes

* **Log loss is the metric that matters**, so it drives checkpoint selection, LR scheduling and early stopping, and each fold is calibrated with a temperature fitted on its held-out data.
* **No leakage by construction.** Synthetic scans share a `source_uid` with their parent and the CV is grouped on it.
* **Train/inference consistency.** Preprocessing parameters travel with the weights in `config.json`; the same `preprocess_files` function is used in both places.
* **Robust inference.** A failing exam never crashes the run: it receives the training prevalence, and no test identifiers are ever logged.
