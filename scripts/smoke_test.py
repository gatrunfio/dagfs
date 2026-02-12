#!/usr/bin/env python3
"""
Smoke test for the public DAGFS repo.

This is intentionally lightweight and fast. It checks that:
  1) the `dagfs` public API imports and runs on a small synthetic dataset;
  2) the main scripts can run end-to-end on a tiny generated fold:
     run_dagfs_custom.py -> eval_rankings_py_pgrid_fast.py

Run:
  python3 scripts/smoke_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}\n"
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    # --- 1) API smoke test ---
    sys.path.insert(0, str(repo_root / "src"))
    import dagfs as dagfs_pkg  # noqa: E402

    rng = np.random.default_rng(0)
    X = rng.random((60, 30), dtype=np.float64)
    Y = (rng.random((60, 7)) > 0.85).astype(np.int8)

    ranking, W, info = dagfs_pkg.dagfs(X, Y, {"max_iter": 2})
    assert ranking.shape == (30,)
    assert W.shape[1] == 7
    assert info["feature_lift"] in ("none", "center_split")

    # --- 2) Script-level smoke test on a temporary fold ---
    try:
        import scipy.io as sio  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing optional dependency for script smoke test: scipy.\n"
            "Install with: python3 -m pip install -e \".[experiments]\"\n"
            f"Original import error: {e}"
        )

    from scipy.io import savemat  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data_dir = td_path / "data"
        results_dir = td_path / "results"
        ds_dir = data_dir / "Toy"
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Create a tiny fold0.mat
        Xtr = rng.random((40, 30), dtype=np.float64)
        Ytr = (rng.random((40, 7)) > 0.85).astype(np.int8)
        Xte = rng.random((20, 30), dtype=np.float64)
        Yte = (rng.random((20, 7)) > 0.85).astype(np.int8)
        savemat(
            ds_dir / "fold0.mat",
            {"X_train": Xtr, "Y_train": Ytr, "X_test": Xte, "Y_test": Yte},
            do_compression=True,
        )

        _run(
            [
                sys.executable,
                "scripts/run_dagfs_custom.py",
                "--data-dir",
                str(data_dir),
                "--output-dir",
                str(results_dir),
                "--method-name",
                "DAGFS",
                "--datasets",
                "Toy",
                "--folds",
                "1",
                "--overwrite",
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                "scripts/eval_rankings_py_pgrid_fast.py",
                "--results-dir",
                str(results_dir),
                "--data-dir",
                str(data_dir),
                "--methods",
                "DAGFS",
                "--datasets",
                "Toy",
                "--folds",
                "1",
                "--p-min",
                "0.2",
                "--p-max",
                "0.4",
                "--p-step",
                "0.2",
                "--p-target",
                "0.2",
            ],
            cwd=repo_root,
        )

        out_json = results_dir / "DAGFS" / "Toy_fold0_pgrid_metrics.json"
        obj = json.loads(out_json.read_text(encoding="utf-8"))
        for key in (
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "micro_pr_auc",
            "macro_pr_auc",
            "avg_precision",
            "one_error",
        ):
            assert isinstance(obj.get(key), list) and len(obj[key]) > 0, key

    print("✓ DAGFS smoke test passed.")


if __name__ == "__main__":
    main()

