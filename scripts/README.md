# Scripts (public)

This folder contains the minimal runnable pipeline to:

1. prepare multi-label folds (optional),
2. run DAGFS on training folds to obtain feature rankings,
3. evaluate rankings over a feature-ratio grid with ML-kNN (Python),
4. aggregate results, statistical tests, and plots.

## Main entrypoints

- `run_dagfs_custom.py`: compute DAGFS rankings (`*_ranking.csv`) for all datasets×folds.
- `eval_rankings_py_pgrid_fast.py`: evaluate rankings with ML-kNN on a p-grid and write `*_pgrid_metrics.json`.
- `aggregate_kgrid_and_make_tables.py`: aggregate metrics across datasets, compute ranks/tests, and generate LaTeX tables.
- `make_pgrid_curves.py`: plot mean p-grid curves (and optionally per-dataset grids) from the JSON outputs.
- `smoke_test.py`: a lightweight end-to-end check (API + minimal ranking→eval pipeline).

## Dataset preparation utilities

- `export_cv_splits_to_mat.py`: create stratified folds (iterative stratification) and export `fold*.mat`.
- `prepare_new_datasets.py`: helper for converting specific Mulan ARFF datasets to the local benchmark format.

## Internal scripts

For ongoing research and paper-specific debugging we keep additional scripts under `scripts/_internal/`.
That folder is intentionally ignored by `.gitignore` (kept locally, not committed).
