#!/usr/bin/env python3
"""
Run a single DAGFS configuration on all datasets × folds and save rankings.

This is a lightweight runner useful for quick ablations without editing code.

Outputs:
  <output_dir>/<method_name>/<dataset>_fold<fold>_ranking.csv
  <output_dir>/<method_name>/<dataset>_fold<fold>_time.txt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.io as sio

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mlfs.dagfs_v2 import dagfs_v2


def infer_datasets(data_dir: Path) -> List[str]:
    """Infer dataset names as subdirectories containing a `fold0.mat`."""
    out: List[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "fold0.mat").exists():
            out.append(p.name)
    if not out:
        raise SystemExit(f"No datasets found under {data_dir} (expected <Dataset>/fold0.mat).")
    return out


def load_fold(data_dir: Path, dataset: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(data_dir / dataset / f"fold{fold}.mat"))
    return (
        np.asarray(mat["X_train"], dtype=np.float64),
        np.asarray(mat["Y_train"], dtype=np.float64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/paper_matlab_minmax"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/bench_paper_kgrid"))
    ap.add_argument("--method-name", type=str, required=True)
    ap.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help="Optional JSON file with DAGFS parameters (merged over defaults).",
    )
    ap.add_argument("--params", type=str, default="{}", help="Inline JSON dict with DAGFS parameters.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Dataset names (default: infer all subdirectories under --data-dir with fold0.mat).",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base: Dict[str, Any] = {
        "alpha": 0.10,
        "beta": 0.10,
        "max_iter": 40,
        "rarity_gamma": 2.0,
        "tau_dir": 0.50,
        "topK": 10,
        "kappa": 1.50,
        "s_max": 3.0,
        "seed": 0,
        "feature_lift": "center_split",
        "transfer_graph": "directed",
    }

    overrides = json.loads(str(args.params))
    if args.params_json is not None:
        overrides_json = json.loads(args.params_json.read_text(encoding="utf-8"))
        overrides.update(overrides_json)
    base.update(overrides)

    out_dir = Path(args.output_dir) / str(args.method_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(args.datasets) if args.datasets else infer_datasets(Path(args.data_dir))
    total = len(datasets) * int(args.folds)
    done = 0
    for dataset in datasets:
        for fold in range(int(args.folds)):
            done += 1
            base_name = f"{dataset}_fold{fold}"
            rank_file = out_dir / f"{base_name}_ranking.csv"
            time_file = out_dir / f"{base_name}_time.txt"
            if rank_file.exists() and time_file.exists() and not args.overwrite:
                print(f"[{done}/{total}] SKIP {dataset} fold{fold}")
                continue

            Xtr, Ytr = load_fold(Path(args.data_dir), dataset, fold)
            t0 = time.time()
            ranking, _w, _info = dagfs_v2(Xtr, Ytr, base)
            elapsed = time.time() - t0
            np.savetxt(str(rank_file), ranking, fmt="%d", delimiter=",")
            time_file.write_text(f"{elapsed:.6f}\n", encoding="utf-8")
            print(f"[{done}/{total}] {dataset} fold{fold} | {elapsed:.2f}s")


if __name__ == "__main__":
    main()
