# DAGFS — Directed Additive Graph Feature Selection (Multi-label FS)

This repository contains the reference Python implementation of **DAGFS**, an embedded method for **multi-label feature selection (MLFS)**.
At a high level, DAGFS produces a **single feature ranking per training fold** by combining:

- a **nonnegative reconstruction** template with group sparsity (stable multiplicative updates),
- a **signed-deviation feature lift** to represent *two-sided evidence* under nonnegativity,
- **rarity-aware instance reweighting** (training-only) to stabilise learning under heterogeneous supervision,
- a **directed label-transfer regulariser** to control correlation transfer across labels.

The code is intended to be usable independently of the paper experiments: you can run DAGFS on your own datasets and evaluate any downstream classifier you prefer.

## What is (and is not) included

- ✅ **Included (ours):** DAGFS implementation (`src/`) and scripts to run DAGFS and evaluate rankings (`scripts/`).
- ❌ **Not included:** datasets, third‑party baselines, and all generated result folders. 

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[experiments]"
```

Core DAGFS only requires NumPy. The `experiments` extra installs the scientific stack used by the provided scripts.

## Quickstart (API)

```python
import numpy as np
from dagfs import DAGFSParams, dagfs

# X: (n_samples, n_features) float, preferably scaled to [0,1]
# Y: (n_samples, n_labels) in {0,1}

params = DAGFSParams(
    alpha=0.10,
    beta=0.10,
    max_iter=40,
    feature_lift="center_split",
    paired_penalty=True,
)

ranking_1based, W, info = dagfs(X, Y, params)
print(ranking_1based[:10], info)
```

`ranking_1based` is a 1‑based permutation of feature indices (MATLAB‑friendly). Use `ranking_1based - 1` for 0‑based Python indexing.

## Running experiments (scripts)

The scripts assume a **folded dataset layout** (one folder per dataset, one file per fold):

```
<DATA_DIR>/<DatasetName>/
  fold0.mat
  fold1.mat
  ...
```

Each `fold*.mat` must contain:

- `X_train`: shape `(n_train, d)`
- `Y_train`: shape `(n_train, L)` in `{0,1}`
- `X_test`:  shape `(n_test,  d)`
- `Y_test`:  shape `(n_test,  L)` in `{0,1}`

### 1) Produce DAGFS rankings

```bash
python3 scripts/run_dagfs_custom.py \
  --data-dir <DATA_DIR> \
  --output-dir <RESULTS_DIR> \
  --method-name DAGFS \
  --folds 5
```

This writes:

- `<RESULTS_DIR>/DAGFS/<dataset>_fold<k>_ranking.csv`

### 2) Evaluate rankings on a p-grid with ML-kNN (Python)

```bash
python3 scripts/eval_rankings_py_pgrid_fast.py \
  --results-dir <RESULTS_DIR> \
  --data-dir <DATA_DIR> \
  --methods DAGFS \
  --folds 5 \
  --p-min 0.05 --p-max 0.50 --p-step 0.05 --p-target 0.20
```

This produces per fold:

- `<RESULTS_DIR>/DAGFS/<dataset>_fold<k>_pgrid_metrics.json`

### 3) Aggregate results + statistical tests + plots

```bash
python3 scripts/aggregate_kgrid_and_make_tables.py \
  --results-dir <RESULTS_DIR> \
  --grid-mode pgrid --p-target 0.2 \
  --methods DAGFS \
  --out-dir outputs/tables

python3 scripts/make_pgrid_curves.py \
  --results-dir <RESULTS_DIR> \
  --methods DAGFS \
  --out-dir outputs/figures
```

## Reproducibility notes

- **Training-only preprocessing:** any statistic used by DAGFS (lift mean, label frequencies, label similarity) is computed on the training fold only.
- **Scaling:** DAGFS is designed for nonnegative data. For most benchmarks we use **min–max scaling to `[0,1]` per fold** (fit on train, applied to test).
- **Folds:** for multi-label data, use **iterative stratification** to preserve label proportions across folds; see `scripts/export_cv_splits_to_mat.py`.

## Repository structure

- `src/`: DAGFS implementation (public API)
- `scripts/`: runnable pipelines (ranking, evaluation, aggregation, plotting)

## Citation

If you use this code, please cite the accompanying paper.

## Smoke test

After installation (ideally with the `experiments` extra), you can run a quick end-to-end sanity check:

```bash
python3 scripts/smoke_test.py
```

## License

MIT License. See `LICENSE`.
