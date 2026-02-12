#!/usr/bin/env python3
"""
Figure 7 remake — lift split artefact under an unpaired penalty (all datasets, all folds).

We quantify a diagnostic split ratio for the lifted-but-unpaired variant:
  - run DAGFS with center-split lifting (2d) and unpaired ℓ2,1 penalty over lifted rows
  - rank lifted components by row norms ||W_{u:}||_2
  - select top-(2k) lifted components where k = round(p * d)
  - a touched original feature j is "split" if exactly one of (j+, j-) is selected
  - SplitRatio = (#split) / (#touched)

Paired DAGFS has SplitRatio = 0 by design (selection is done on original-feature groups).

Outputs:
  - results/lift_split_ratio/lift_split_ratio.csv
  - paper/figures/paired_invariance_splitrate.pdf
  - paper/tables/table_paired_invariance.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mlfs.dagfs_v2 import dagfs_v2


def load_fold_train(data_dir: Path, dataset: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(data_dir / dataset / f"fold{fold}.mat"))
    return (
        np.asarray(mat["X_train"], dtype=np.float64),
        np.asarray(mat["Y_train"], dtype=np.float64),
    )


def load_paired_ranking(results_dir: Path, method: str, dataset: str, fold: int) -> np.ndarray:
    path = results_dir / method / f"{dataset}_fold{fold}_ranking.csv"
    r = np.loadtxt(str(path), delimiter=",").astype(np.int64).reshape(-1)
    return r  # 1-based indices (original features)


def jaccard_set(a: np.ndarray, b: np.ndarray) -> float:
    sa = set(np.asarray(a, dtype=np.int64).reshape(-1).tolist())
    sb = set(np.asarray(b, dtype=np.int64).reshape(-1).tolist())
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(len(sa | sb))


def _ds_title(ds: str) -> str:
    ds = str(ds)
    return (ds[:1].upper() + ds[1:].lower()) if ds else ds


def mean_std(x: List[float]) -> Tuple[float, float]:
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return m, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/paper_matlab_minmax"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/bench_paper_kgrid"))
    ap.add_argument("--paired-method", type=str, default="DAGFS", help="Method name holding paired rankings in results-dir.")
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "Arts",
            "Business",
            "Education",
            "Entertain",
            "Health",
            "Recreation",
            "Reference",
            "Science",
            "Social",
            "bibtex",
            "corel5k",
            "emotions",
            "genbase",
            "medical",
            "yeast",
        ],
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--p", type=float, default=0.20)
    ap.add_argument("--out-dir", type=Path, default=Path("results/lift_split_ratio"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/lift_split_ratio/lift_split_ratio.csv"))
    ap.add_argument("--out-fig", type=Path, default=Path("paper/figures/paired_invariance_splitrate.pdf"))
    ap.add_argument("--out-tex", type=Path, default=Path("paper/tables/table_paired_invariance.tex"))
    args = ap.parse_args()

    p = float(args.p)
    folds = int(args.folds)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)

    # Match paper defaults (unpaired only changes how ℓ2,1 surrogate couples (+/-) rows).
    base_params: Dict[str, float | int | str | bool] = {
        "alpha": 0.10,
        "beta": 0.10,
        "max_iter": 40,
        "rarity_gamma": 2.0,
        "tau_dir": 0.50,
        "topK": 10,
        "kappa": 1.50,
        "s_max": 3.0,
        "seed": 0,
        "transfer_graph": "directed",
        "feature_lift": "center_split",
        "paired_penalty": False,
    }

    rows: List[Dict[str, float | int | str]] = []

    for ds in args.datasets:
        for fold in range(folds):
            Xtr, Ytr = load_fold_train(args.data_dir, ds, int(fold))
            n, d = Xtr.shape
            L = int(Ytr.shape[1])
            k = int(max(1, round(p * d)))
            m = int(min(2 * k, 2 * d))

            # Unpaired lifted run (training-only).
            rank_lift, W_u, _info = dagfs_v2(Xtr, Ytr, base_params)
            rank_lift0 = (np.asarray(rank_lift, dtype=np.int64).reshape(-1) - 1)  # 0..2d-1
            top = rank_lift0[:m]

            sel_plus = {int(u) for u in top if int(u) < d}
            sel_minus = {int(u - d) for u in top if int(u) >= d}
            touched = sel_plus | sel_minus
            split = sel_plus ^ sel_minus

            split_ratio = float(len(split)) / float(len(touched)) if len(touched) > 0 else float("nan")

            # Jaccard between paired (from stored paper runs) and unpaired-derived original ranking.
            paired_rank = load_paired_ranking(args.results_dir, str(args.paired_method), ds, int(fold))
            paired_topk = paired_rank[:k]

            Wu = np.asarray(W_u, dtype=np.float64)
            Wplus = Wu[:d]
            Wminus = Wu[d:]
            splus = np.sqrt((Wplus * Wplus).sum(axis=1))
            sminus = np.sqrt((Wminus * Wminus).sum(axis=1))
            s_orig = np.maximum(splus, sminus)
            unpaired_orig_rank = np.argsort(-s_orig, kind="mergesort") + 1  # 1..d
            unpaired_topk = unpaired_orig_rank[:k]

            j_paired_unpaired = jaccard_set(paired_topk, unpaired_topk)

            rows.append(
                {
                    "dataset": str(ds),
                    "fold": int(fold),
                    "p": float(p),
                    "split_ratio_unpaired": float(split_ratio),
                    "n_touched": int(len(touched)),
                    "n_split": int(len(split)),
                    "d": int(d),
                    "L": int(L),
                    "n": int(n),
                    "j_paired_vs_unpaired": float(j_paired_unpaired),
                }
            )

    # Write CSV
    import csv

    fieldnames = [
        "dataset",
        "fold",
        "p",
        "split_ratio_unpaired",
        "n_touched",
        "n_split",
        "d",
        "L",
        "n",
        "j_paired_vs_unpaired",
    ]
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate by dataset over folds
    by_ds: Dict[str, List[Dict[str, float | int | str]]] = {}
    for r in rows:
        by_ds.setdefault(str(r["dataset"]), []).append(r)

    # LaTeX table: mean±std over folds, per dataset + overall mean±std over datasets.
    table_rows: List[Tuple[str, float, float, float, float]] = []
    for ds in args.datasets:
        items = by_ds.get(str(ds), [])
        split_vals = [float(it["split_ratio_unpaired"]) for it in items]
        jac_vals = [float(it["j_paired_vs_unpaired"]) for it in items]
        m_split, s_split = mean_std(split_vals)
        m_j, s_j = mean_std(jac_vals)
        table_rows.append((str(ds), m_split, s_split, m_j, s_j))

    overall_split_mean, overall_split_std = mean_std([tr[1] for tr in table_rows])
    overall_j_mean, overall_j_std = mean_std([tr[3] for tr in table_rows])

    lines: List[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{5pt}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Dataset & SplitRatio (unpaired) & $J$(paired, unpaired) \\\\")
    lines.append("\\midrule")
    for ds, ms, ss, mj, sj in table_rows:
        name = _ds_title(ds)
        lines.append(f"{name} & {ms:.3f} $\\pm$ {ss:.3f} & {mj:.3f} $\\pm$ {sj:.3f} \\\\")
    lines.append("\\midrule")
    lines.append(
        f"Mean over datasets & {overall_split_mean:.3f} $\\pm$ {overall_split_std:.3f} & {overall_j_mean:.3f} $\\pm$ {overall_j_std:.3f} \\\\"
    )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Lift split artefact diagnostic at $p=20\\%$ selected features, aggregated over all datasets and folds. "
        "SplitRatio is computed on the unpaired lifted variant by selecting the top-$(2k)$ lifted components (with $k=\\mathrm{round}(p\\,d)$) "
        "and measuring the fraction of \\emph{touched} original features for which exactly one of $(j^+)$ or $(j^-)$ is selected. "
        "$J$(paired, unpaired) is the Jaccard overlap between the top-$k$ original feature sets induced by the paired model (paper ranking) and by the unpaired model (derived by max pooling the $(+/-)$ row norms). "
        "For paired DAGFS, SplitRatio is $0$ by design.}"
    )
    lines.append("\\label{tab:paired_invariance}")
    lines.append("\\end{table}")
    args.out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: per-dataset distribution across folds (boxplots).
    import matplotlib.pyplot as plt

    data_by_ds: List[List[float]] = []
    labels: List[str] = []
    for ds in args.datasets:
        items = by_ds.get(str(ds), [])
        vals = [float(it["split_ratio_unpaired"]) for it in items]
        data_by_ds.append(vals)
        labels.append(_ds_title(ds))

    fig, ax = plt.subplots(1, 1, figsize=(12.2, 7.2))
    ax_title_fs = 14
    ax_label_fs = 13
    tick_fs = 11
    legend_fs = 11
    bp = ax.boxplot(
        data_by_ds,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "C2", "alpha": 0.55},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
    )
    for i, vals in enumerate(data_by_ds, start=1):
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, num=len(vals))
        ax.scatter(i + jitter, vals, s=14, color="C2", alpha=0.95, edgecolors="none")

    ax.set_ylabel("SplitRatio (unpaired lift)", fontsize=ax_label_fs)
    ax.set_title(
        "Lift-induced split artefact without paired penalty (all datasets, all folds)",
        fontsize=ax_title_fs,
    )
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(
        0.99,
        0.97,
        "Paired variant: SplitRatio = 0 by design (off-scale)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=legend_fs,
        style="italic",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=tick_fs)
    plt.setp(ax.get_yticklabels(), fontsize=tick_fs)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.28)
    fig.savefig(args.out_fig, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    print("✓ Wrote:", args.out_csv)
    print("✓ Wrote:", args.out_tex)
    print("✓ Wrote:", args.out_fig)


if __name__ == "__main__":
    main()
