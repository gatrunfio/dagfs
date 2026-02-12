#!/usr/bin/env python3
"""
Export multi-label train/test splits to MATLAB .mat files.

This repo contains datasets in NPZ (sparse CSR components) under:
  - data/paper_protocol/<Dataset>/{train.npz,test.npz}
  - data/asc2025/raw/<Dataset>/{train.npz,test.npz}  (ASC suite)

The MATLAB baselines expect, for each dataset and fold:
  X_train, Y_train, X_test, Y_test

We generate K repeated stratified holdout splits ("folds") by applying
iterative stratification on the full dataset (train+test concatenated),
with approximately the same train/test proportion as the provided split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy import sparse
from scipy.io import savemat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skmultilearn.model_selection import IterativeStratification, iterative_train_test_split


def load_xy_npz(npz_path: Path) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    npz = np.load(npz_path, allow_pickle=True)
    x = sparse.csr_matrix(
        (npz["X_data"], npz["X_indices"], npz["X_indptr"]), shape=tuple(npz["X_shape"])
    )
    y = sparse.csr_matrix(
        (npz["Y_data"], npz["Y_indices"], npz["Y_indptr"]), shape=tuple(npz["Y_shape"])
    )
    return x, y


def find_datasets(root: Path, datasets: Iterable[str] | None) -> list[str]:
    if datasets is not None and len(list(datasets)) > 0:
        return list(datasets)
    # autodiscovery
    ds = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if (p / "train.npz").exists() and (p / "test.npz").exists():
            ds.append(p.name)
    if not ds:
        raise SystemExit(f"No datasets found under {root}")
    return ds


def scale_train_test(
    x_train: np.ndarray, x_test: np.ndarray, scaler_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    if scaler_name == "none":
        return x_train, x_test
    if scaler_name == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True)
        return scaler.fit_transform(x_train), scaler.transform(x_test)
    if scaler_name == "minmax":
        scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        return scaler.fit_transform(x_train), scaler.transform(x_test)
    raise ValueError(f"Unknown scaler: {scaler_name}")


def export_dataset(
    dataset_dir: Path,
    out_dir: Path,
    dataset_name: str,
    n_folds: int,
    seed: int,
    scaler: str,
    split_mode: str,
) -> None:
    x_tr, y_tr = load_xy_npz(dataset_dir / "train.npz")
    x_te, y_te = load_xy_npz(dataset_dir / "test.npz")

    x_full = sparse.vstack([x_tr, x_te]).toarray().astype(np.float64)
    y_full = sparse.vstack([y_tr, y_te]).toarray().astype(np.int8)

    n_total = x_full.shape[0]
    n_test_ref = x_te.shape[0]
    test_size = float(n_test_ref) / float(n_total)

    ds_out = out_dir / dataset_name
    ds_out.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": dataset_name,
        "n_total": int(n_total),
        "n_features": int(x_full.shape[1]),
        "n_labels": int(y_full.shape[1]),
        "reference_split": {"train": int(x_tr.shape[0]), "test": int(x_te.shape[0])},
        "n_folds": int(n_folds),
        "seed": int(seed),
        "test_size": test_size,
        "scaler": scaler,
        "split_mode": split_mode,
    }
    (ds_out / "meta.json").write_text(json.dumps(meta, indent=2))

    if split_mode == "repeated_holdout":
        for fold in range(n_folds):
            # Shuffle rows before splitting: iterative_train_test_split is
            # deterministic w.r.t. input order and ignores np.random.seed,
            # so we must permute the data ourselves to obtain distinct folds.
            rng = np.random.RandomState(seed + fold)
            perm = rng.permutation(n_total)
            x_shuf = x_full[perm]
            y_shuf = y_full[perm]

            x_train, y_train, x_test, y_test = iterative_train_test_split(
                x_shuf, y_shuf, test_size=test_size
            )

            x_train, x_test = scale_train_test(x_train, x_test, scaler)

            out_path = ds_out / f"fold{fold}.mat"
            savemat(
                out_path,
                {
                    "X_train": x_train,
                    "Y_train": y_train,
                    "X_test": x_test,
                    "Y_test": y_test,
                },
                do_compression=True,
            )
        return

    if split_mode == "kfold":
        # Build a single shuffle permutation (for determinism), then run an
        # iterative-stratified K-fold splitter on the shuffled data.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_total)
        x_shuf = x_full[perm]
        y_shuf = y_full[perm]

        splitter = IterativeStratification(n_splits=n_folds, order=1)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x_shuf, y_shuf)):
            x_train = x_shuf[train_idx]
            y_train = y_shuf[train_idx]
            x_test = x_shuf[test_idx]
            y_test = y_shuf[test_idx]

            x_train, x_test = scale_train_test(x_train, x_test, scaler)

            out_path = ds_out / f"fold{fold}.mat"
            savemat(
                out_path,
                {
                    "X_train": x_train,
                    "Y_train": y_train,
                    "X_test": x_test,
                    "Y_test": y_test,
                },
                do_compression=True,
            )
        return

    raise ValueError(f"Unknown split_mode: {split_mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/paper_protocol"),
        help="Dataset root containing <Dataset>/{train.npz,test.npz}",
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names (default: autodiscover all under data-root)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/asc2025_matlab"),
        help="Output directory for MATLAB fold .mat files",
    )
    ap.add_argument("--folds", type=int, default=5, help="Number of folds (repeats)")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument(
        "--scaler",
        choices=["none", "zscore", "minmax"],
        default="minmax",
        help="Feature scaling applied per fold using train statistics (default: minmax)",
    )
    ap.add_argument(
        "--split-mode",
        choices=["repeated_holdout", "kfold"],
        default="repeated_holdout",
        help="How to create folds: repeated holdout (SRFS-style repeats) or iterative-stratified k-fold CV.",
    )
    args = ap.parse_args()

    data_root: Path = args.data_root
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_list = find_datasets(data_root, args.datasets)
    scaler = args.scaler

    for ds in ds_list:
        export_dataset(
            dataset_dir=data_root / ds,
            out_dir=out_dir,
            dataset_name=ds,
            n_folds=args.folds,
            seed=args.seed,
            scaler=scaler,
            split_mode=args.split_mode,
        )
        print(f"✓ Exported {ds} -> {out_dir / ds}")


if __name__ == "__main__":
    main()
