#!/usr/bin/env python3
"""
Attempt A (Monotonicity barrier) — synthetic 1D/1-label toy.

We show that a nonnegative reconstruction model without lifting can only
express monotone non-decreasing evidence in x, and therefore struggles when
the positive label is associated with *low* x.

We compare:
  - Baseline nonnegative reconstruction (no lift)
  - DAGFS with signed-deviation center-split lift only (no graph, no inst)

Outputs:
  paper/figures/toy_monotonicity_scatter.pdf
  paper/figures/toy_monotonicity_model.pdf
  paper/figures/toy_monotonicity_panels.pdf
  paper/tables/toy_monotonicity_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mlfs.dagfs_v2 import dagfs_v2


def f1_from_preds(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / float(denom))


def best_threshold_f1(scores: np.ndarray, y_true01: np.ndarray) -> Tuple[float, float]:
    """
    Pick the threshold that maximizes F1 on the provided scores.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(y_true01, dtype=np.int8).reshape(-1)
    uniq = np.unique(scores)
    if uniq.size == 0:
        return 0.5, 0.0
    thr_candidates = np.concatenate([uniq, uniq + 1e-12])
    best_f1 = -1.0
    best_thr = float(thr_candidates[0])
    for thr in thr_candidates:
        pred = (scores >= thr).astype(np.int8)
        f1 = f1_from_preds(y, pred)
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flip", type=float, default=0.05, help="Optional label flip probability.")
    ap.add_argument("--x_thr", type=float, default=0.30, help="y=1 iff x < x_thr (before noise).")
    ap.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    ap.add_argument("--out-json", type=Path, default=Path("paper/tables/toy_monotonicity_summary.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    n = int(args.n)
    x = rng.random((n, 1), dtype=np.float64)  # Uniform(0,1)
    y = (x[:, 0] < float(args.x_thr)).astype(np.int8)
    if float(args.flip) > 0:
        flip = rng.random((n,), dtype=np.float64) < float(args.flip)
        y = np.where(flip, 1 - y, y)

    X = x
    Y = y.reshape(-1, 1).astype(np.float64)

    # Baseline: nonnegative reconstruction without lift, no graph, no inst.
    base_params = {
        "alpha": 0.0,
        "beta": 1e-3,
        "max_iter": 200,
        "rarity_gamma": 0.0,
        "kappa": 0.0,
        "s_max": 1.0,
        "feature_lift": "none",
        "transfer_graph": "directed",
        "tau_dir": 0.0,
        "topK": 0,
        "seed": 0,
    }
    _rank0, W0, _info0 = dagfs_v2(X, Y, base_params)
    w0 = float(W0.reshape(-1)[0])
    score0 = (X[:, 0] * w0)
    thr0, f10 = best_threshold_f1(score0, y)

    # Lift only: center-split, paired penalty (to map back), no graph, no inst.
    lift_params = dict(base_params)
    lift_params.update({"feature_lift": "center_split", "paired_penalty": True})
    _rank1, W1, _info1 = dagfs_v2(X, Y, lift_params)
    mu = float(X.mean(axis=0)[0])
    Xpos = np.maximum(X[:, 0] - mu, 0.0)
    Xneg = np.maximum(mu - X[:, 0], 0.0)
    wpos = float(W1[0, 0])
    wneg = float(W1[1, 0])
    score1 = Xpos * wpos + Xneg * wneg
    thr1, f11 = best_threshold_f1(score1, y)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    # Figure 1: scatter x vs y
    fig1, ax1 = plt.subplots(1, 1, figsize=(5.2, 2.9))
    ax1.axvspan(0.0, float(args.x_thr), color="C1", alpha=0.10, label=f"positive region ($x<{args.x_thr:.2f}$)")
    ax1.scatter(X[:, 0], y + 0.02 * rng.standard_normal((n,)), s=10, alpha=0.25, edgecolors="none")
    ax1.axvline(float(args.x_thr), color="black", linewidth=1.0, linestyle="--", label=f"true threshold ({args.x_thr:.2f})")
    ax1.set_xlabel("Feature value $x$")
    ax1.set_ylabel("Label $y$ (jittered)")
    ax1.set_title("Synthetic dataset: positives are low-$x$")
    ax1.grid(True, alpha=0.25)
    # Place the legend slightly lower to avoid occluding the upper-right points.
    ax1.legend(
        frameon=False,
        fontsize=8,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.82),
        borderaxespad=0.0,
    )
    fig1.tight_layout()
    fig1.savefig(out_dir / "toy_monotonicity_scatter.pdf")
    plt.close(fig1)

    # Figure 2: model scores vs x
    xs = np.linspace(0.0, 1.0, 400, dtype=np.float64)
    score0_line = xs * w0
    Xpos_line = np.maximum(xs - mu, 0.0)
    Xneg_line = np.maximum(mu - xs, 0.0)
    score1_line = Xpos_line * wpos + Xneg_line * wneg

    fig2, ax2 = plt.subplots(1, 1, figsize=(5.2, 3.0))
    ax2.plot(xs, score0_line, label="No lift (monotone non-decreasing)", linewidth=2.0)
    ax2.plot(xs, score1_line, label="Signed-deviation lift (can be decreasing)", linewidth=2.0)
    ax2.axhline(thr0, color="C0", linestyle=":", linewidth=1.2, alpha=0.9, label=f"best thr (no lift): {thr0:.3f}")
    ax2.axhline(thr1, color="C1", linestyle=":", linewidth=1.2, alpha=0.9, label=f"best thr (lift): {thr1:.3f}")
    ax2.set_xlabel("Feature value $x$")
    ax2.set_ylabel("Reconstruction score $g(x)$")
    ax2.set_title("Synthetic model expressivity under nonnegativity")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "toy_monotonicity_model.pdf")
    plt.close(fig2)

    # Figure panels: (a) data distribution (b) score profiles
    pos_rate = float(np.mean(y))
    figP, (axL, axR) = plt.subplots(1, 2, figsize=(10.8, 3.0))

    axL.axvspan(0.0, float(args.x_thr), color="C1", alpha=0.10)
    axL.scatter(X[:, 0], y + 0.02 * rng.standard_normal((n,)), s=10, alpha=0.22, edgecolors="none")
    axL.axvline(float(args.x_thr), color="black", linewidth=1.0, linestyle="--")
    axL.set_xlabel("Feature value $x$")
    axL.set_ylabel("Label $y$ (jittered)")
    axL.set_title(f"(a) Synthetic data (pos rate={pos_rate:.2f}, flip={float(args.flip):.2f})")
    axL.grid(True, alpha=0.25)

    axR.plot(xs, score0_line, label="No lift ($g(x)=wx$)", linewidth=2.0)
    axR.plot(xs, score1_line, label="Lift ($\\Phi(x)=[(x-\\mu)_+, (\\mu-x)_+]$)", linewidth=2.0)
    axR.axhline(thr0, color="C0", linestyle=":", linewidth=1.2, alpha=0.9)
    axR.axhline(thr1, color="C1", linestyle=":", linewidth=1.2, alpha=0.9)
    axR.set_xlabel("Feature value $x$")
    axR.set_ylabel("Score $g(x)$")
    axR.set_title("(b) Nonnegative score profiles")
    axR.grid(True, alpha=0.25)
    axR.legend(frameon=False, fontsize=8, loc="upper right")

    figP.tight_layout()
    figP.savefig(out_dir / "toy_monotonicity_panels.pdf")
    plt.close(figP)

    summary: Dict[str, float] = {
        "n": float(n),
        "x_thr": float(args.x_thr),
        "flip": float(args.flip),
        "pos_rate": float(pos_rate),
        "mu": float(mu),
        "no_lift_w": float(w0),
        "no_lift_best_thr": float(thr0),
        "no_lift_train_f1": float(f10),
        "lift_w_pos": float(wpos),
        "lift_w_neg": float(wneg),
        "lift_best_thr": float(thr1),
        "lift_train_f1": float(f11),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("✓ Wrote toy figures:")
    print("  -", out_dir / "toy_monotonicity_scatter.pdf")
    print("  -", out_dir / "toy_monotonicity_model.pdf")
    print("✓ Wrote summary:", args.out_json)


if __name__ == "__main__":
    main()
