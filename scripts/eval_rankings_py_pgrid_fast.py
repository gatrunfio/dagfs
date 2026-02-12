#!/usr/bin/env python3
"""
Fast p-grid evaluation of feature rankings with ML-kNN (NumPy incremental distances).

This is a Python alternative to the MATLAB/Octave evaluator, producing JSON files
compatible with scripts/aggregate_kgrid_and_make_tables.py and scripts/make_pgrid_curves.py.

Inputs:
  results_dir/<METHOD>/<DATASET>_fold<fold>_ranking.csv   (1-based indices)
  data_dir/<DATASET>/fold<fold>.mat                      (X_train/Y_train/X_test/Y_test)

Outputs:
  results_dir/<METHOD>/<DATASET>_fold<fold><out_suffix>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio


def micro_f1(pre: np.ndarray, y: np.ndarray) -> float:
    tp = int(np.sum((pre == 1) & (y == 1)))
    fp = int(np.sum((pre == 1) & (y == -1)))
    fn = int(np.sum((pre == -1) & (y == 1)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else float((2 * tp) / float(denom))


def macro_f1(pre: np.ndarray, y: np.ndarray) -> float:
    L = pre.shape[0]
    out = 0.0
    for j in range(L):
        tp = int(np.sum((pre[j] == 1) & (y[j] == 1)))
        fp = int(np.sum((pre[j] == 1) & (y[j] == -1)))
        fn = int(np.sum((pre[j] == -1) & (y[j] == 1)))
        denom = 2 * tp + fp + fn
        out += 0.0 if denom == 0 else float((2 * tp) / float(denom))
    return float(out / float(L))


def hamming_loss(pre: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(pre != y))

def _average_precision_binary(scores: np.ndarray, y01: np.ndarray) -> float:
    """
    Average precision (AP) for a binary task.

    Matches the AP definition used in `baselines/_eval_mlknn/PR_AUC_micro_macro.m`:
      AP = mean_k precision@rank(pos_k),
    where pos_k are the ranks of positive examples after sorting by decreasing score.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    y01 = (np.asarray(y01).reshape(-1) != 0)
    n_pos = int(np.sum(y01))
    if n_pos == 0:
        return float("nan")
    idx = np.argsort(-scores, kind="mergesort")
    y_sorted = y01[idx]
    pos_idx = np.flatnonzero(y_sorted) + 1  # ranks are 1-based
    if pos_idx.size == 0:
        return float("nan")
    k = np.arange(1, pos_idx.size + 1, dtype=np.float64)
    return float(np.mean(k / pos_idx.astype(np.float64)))


def pr_auc_micro_macro(outputs: np.ndarray, y_pm: np.ndarray) -> tuple[float, float, int, int]:
    """
    Compute micro/macro PR-AUC (via AP) from ML-kNN posteriors.

    outputs: (L, n) in [0,1]
    y_pm   : (L, n) in {-1,+1}
    """
    L = int(y_pm.shape[0])
    labels_total = L

    y01_micro = (y_pm.reshape(-1) == 1).astype(np.int8)
    s_micro = outputs.reshape(-1)
    micro_ap = _average_precision_binary(s_micro, y01_micro)

    aps: list[float] = []
    for ell in range(L):
        y01 = (y_pm[ell] == 1)
        if np.all(~y01) or np.all(y01):
            continue
        aps.append(_average_precision_binary(outputs[ell], y01))
    labels_used = len(aps)
    macro_ap = float(np.mean(aps)) if aps else float("nan")
    return float(micro_ap), float(macro_ap), int(labels_used), int(labels_total)


def avg_precision_example_based(outputs: np.ndarray, y_pm: np.ndarray) -> float:
    """
    Example-based average precision (label-ranking AP), matching `Average_precision.m`.
    """
    outputs = np.asarray(outputs, dtype=np.float64)
    y_pm = np.asarray(y_pm, dtype=np.int8)
    L, n = y_pm.shape
    if n == 0 or L == 0:
        return float("nan")

    ap_sum = 0.0
    for i in range(n):
        pos = np.flatnonzero(y_pm[:, i] == 1)
        if pos.size == 0:
            continue
        pos_set = set(pos.tolist())
        idx = np.argsort(-outputs[:, i], kind="mergesort")
        hit = 0
        prec_sum = 0.0
        # iterate all ranks; accumulate precision at each relevant label
        for r, lab in enumerate(idx, start=1):
            if int(lab) in pos_set:
                hit += 1
                prec_sum += hit / float(r)
        ap_sum += prec_sum / float(pos.size)
    return float(ap_sum / float(n))


def one_error(outputs: np.ndarray, y_pm: np.ndarray) -> float:
    """One-error, matching `One_error.m` (lower is better)."""
    outputs = np.asarray(outputs, dtype=np.float64)
    y_pm = np.asarray(y_pm, dtype=np.int8)
    L, n = y_pm.shape
    if n == 0 or L == 0:
        return float("nan")
    top = np.argmax(outputs, axis=0)  # (n,)
    err = np.sum(y_pm[top, np.arange(n)] != 1)
    return float(err / float(n))


def load_fold(data_dir: Path, dataset: str, fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat = sio.loadmat(str(data_dir / dataset / f"fold{fold}.mat"))
    Xtr = np.asarray(mat["X_train"], dtype=np.float64)
    Ytr = np.asarray(mat["Y_train"], dtype=np.float64)
    Xte = np.asarray(mat["X_test"], dtype=np.float64)
    Yte = np.asarray(mat["Y_test"], dtype=np.float64)
    return Xtr, Ytr, Xte, Yte


def load_ranking(results_dir: Path, method: str, dataset: str, fold: int) -> np.ndarray:
    path = results_dir / method / f"{dataset}_fold{fold}_ranking.csv"
    r = np.loadtxt(str(path), delimiter=",").astype(np.int64).reshape(-1)
    # 1-based -> 0-based
    r0 = r - 1
    return r0


def mlknn_eval_kgrid_incremental(
    Xtr_ranked: np.ndarray,
    Ytr_pm: np.ndarray,  # (L, n_train) in {-1,+1}
    Xte_ranked: np.ndarray,
    Yte_pm: np.ndarray,  # (L, n_test) in {-1,+1}
    k_values: List[int],
    *,
    Num: int,
    Smooth: float,
) -> Dict[str, List[float]]:
    """
    Replicates baselines/_eval_mlknn/MLKNN_eval_kgrid.m logic.

    Note: MATLAB's `mink()` has deterministic tie-breaking. For exact
    equivalence on these benchmark folds (which can have many distance ties),
    we use a stable full sort (`argsort(kind='mergesort')`) rather than
    `argpartition`.
    """
    epsv = np.finfo(np.float64).eps
    L, n_train = Ytr_pm.shape
    n_test = Yte_pm.shape[1]

    k_values = [int(k) for k in k_values]
    k_values = sorted(list(dict.fromkeys([k for k in k_values if k >= 1])))
    k_max = int(Xtr_ranked.shape[1])
    k_values = [k for k in k_values if k <= k_max]
    if not k_values:
        raise ValueError("k_values is empty after clipping.")

    # Prior/PriorN (independent of selected features)
    Prior = np.zeros((L,), dtype=np.float64)
    PriorN = np.zeros((L,), dtype=np.float64)
    for j in range(L):
        temp_Ci = int(np.sum(Ytr_pm[j] == 1))
        Prior[j] = (Smooth + temp_Ci) / (Smooth * 2.0 + n_train)
        PriorN[j] = 1.0 - Prior[j]

    Cond = np.zeros((L, Num + 1), dtype=np.float64)
    CondN = np.zeros((L, Num + 1), dtype=np.float64)

    train_norms = np.zeros((n_train,), dtype=np.float64)
    test_norms = np.zeros((n_test,), dtype=np.float64)
    dot_tt = np.zeros((n_train, n_train), dtype=np.float64)
    dot_te_tr = np.zeros((n_test, n_train), dtype=np.float64)

    micro: List[float] = []
    macro: List[float] = []
    hamming: List[float] = []
    micro_pr_auc: List[float] = []
    macro_pr_auc: List[float] = []
    avgprec: List[float] = []
    oneerr: List[float] = []
    macro_labels_used: int | None = None
    macro_labels_total: int | None = None

    k_prev = 0
    for k in k_values:
        if k > k_prev:
            Xtr_add = Xtr_ranked[:, k_prev:k]
            Xte_add = Xte_ranked[:, k_prev:k]
            train_norms += np.sum(Xtr_add * Xtr_add, axis=1)
            test_norms += np.sum(Xte_add * Xte_add, axis=1)
            dot_tt += Xtr_add @ Xtr_add.T
            dot_te_tr += Xte_add @ Xtr_add.T
            k_prev = k

        dist2_tt = train_norms[:, None] + train_norms[None, :] - 2.0 * dot_tt
        dist2_tt = np.maximum(dist2_tt, 0.0)
        np.fill_diagonal(dist2_tt, np.inf)

        # Training neighbors (exclude self); indices shape (n_train, Num)
        NeighTr = np.argsort(dist2_tt, axis=1, kind="mergesort")[:, :Num]

        # Counts: (L, n_train) in 0..Num
        neigh_labels = Ytr_pm[:, NeighTr.reshape(-1)].reshape(L, n_train, Num)
        cnt = np.sum(neigh_labels == 1, axis=2).astype(np.int64)

        # Conditional probabilities per label
        for j in range(L):
            cvec = cnt[j]
            pos = (Ytr_pm[j] == 1)
            temp_Ci = np.bincount(cvec[pos], minlength=Num + 1).astype(np.float64)
            temp_NCi = np.bincount(cvec[~pos], minlength=Num + 1).astype(np.float64)
            sum_Ci = float(np.sum(temp_Ci))
            sum_NCi = float(np.sum(temp_NCi))
            Cond[j, :] = (Smooth + temp_Ci) / (Smooth * (Num + 1) + sum_Ci + epsv)
            CondN[j, :] = (Smooth + temp_NCi) / (Smooth * (Num + 1) + sum_NCi + epsv)

        # Test neighbors
        dist2_te = test_norms[:, None] + train_norms[None, :] - 2.0 * dot_te_tr
        dist2_te = np.maximum(dist2_te, 0.0)
        NeighTe = np.argsort(dist2_te, axis=1, kind="mergesort")[:, :Num]  # (n_test, Num)

        neigh_labels_te = Ytr_pm[:, NeighTe.reshape(-1)].reshape(L, n_test, Num)
        cnt_te = np.sum(neigh_labels_te == 1, axis=2).astype(np.int64)  # (L, n_test)

        idx = cnt_te  # 0..Num
        cond_vals = Cond[np.arange(L)[:, None], idx]
        condn_vals = CondN[np.arange(L)[:, None], idx]

        Prob_in = Prior[:, None] * cond_vals
        Prob_out = PriorN[:, None] * condn_vals
        denom = Prob_in + Prob_out
        Outputs = np.where(denom == 0.0, Prior[:, None], Prob_in / denom)
        Pre = np.where(Outputs < 0.5, -1, 1).astype(np.int8)

        micro.append(micro_f1(Pre, Yte_pm))
        macro.append(macro_f1(Pre, Yte_pm))
        hamming.append(hamming_loss(Pre, Yte_pm))
        mi_ap, ma_ap, labels_used, labels_total = pr_auc_micro_macro(Outputs, Yte_pm)
        micro_pr_auc.append(mi_ap)
        macro_pr_auc.append(ma_ap)
        macro_labels_used = int(labels_used)
        macro_labels_total = int(labels_total)
        avgprec.append(avg_precision_example_based(Outputs, Yte_pm))
        oneerr.append(one_error(Outputs, Yte_pm))

    return {
        "k_values": k_values,
        "micro_f1": micro,
        "macro_f1": macro,
        "hamming_loss": hamming,
        "micro_pr_auc": micro_pr_auc,
        "macro_pr_auc": macro_pr_auc,
        "avg_precision": avgprec,
        "one_error": oneerr,
        "macro_pr_auc_labels_used": int(macro_labels_used) if macro_labels_used is not None else 0,
        "macro_pr_auc_labels_total": int(macro_labels_total) if macro_labels_total is not None else int(L),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=10, help="Number of folds (default: 10).")
    ap.add_argument("--mlknn-k", type=int, default=10)
    ap.add_argument("--mlknn-smooth", type=float, default=1.0)
    ap.add_argument("--p-min", type=float, default=0.05)
    ap.add_argument("--p-max", type=float, default=0.50)
    ap.add_argument("--p-step", type=float, default=0.05)
    ap.add_argument("--p-target", type=float, default=0.20)
    ap.add_argument("--out-suffix", type=str, default="_pgrid_metrics.json")
    ap.add_argument("--skip-existing", action="store_true", default=False)
    args = ap.parse_args()

    results_dir: Path = args.results_dir
    data_dir: Path = args.data_dir
    folds = int(args.folds)

    if args.datasets is None:
        datasets = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and (p / "fold0.mat").exists()])
    else:
        datasets = list(args.datasets)

    p_values = np.arange(float(args.p_min), float(args.p_max) + 1e-12, float(args.p_step), dtype=np.float64)
    p_values = p_values[(p_values > 0) & (p_values <= 1)]
    if p_values.size == 0:
        raise SystemExit("Empty p grid.")
    p_target = float(args.p_target)
    p_pct = int(round(100 * p_target))

    total = len(args.methods) * len(datasets) * folds
    done = 0

    for method in args.methods:
        method_dir = results_dir / method
        if not method_dir.exists():
            print(f"WARNING: method dir missing: {method_dir}")
            continue
        for ds in datasets:
            for fold in range(folds):
                done += 1
                base = f"{ds}_fold{fold}"
                ranking_path = method_dir / f"{base}_ranking.csv"
                out_path = method_dir / f"{base}{args.out_suffix}"
                if not ranking_path.exists():
                    continue
                if args.skip_existing and out_path.exists():
                    try:
                        obj = json.loads(out_path.read_text(encoding="utf-8"))
                        if "micro_f1" in obj and "macro_f1" in obj and "p_values" in obj:
                            continue
                    except Exception:
                        pass

                Xtr, Ytr, Xte, Yte = load_fold(data_dir, ds, fold)
                n_features = int(Xtr.shape[1])
                n_labels = int(Ytr.shape[1])

                ranking = load_ranking(results_dir, method, ds, fold)
                k_for_p = np.maximum(1, np.round(p_values * n_features).astype(int))
                k_for_p = np.minimum(k_for_p, n_features)
                k_unique = list(dict.fromkeys(k_for_p.tolist()))
                kmax = int(max(k_unique))

                ranked_prefix = ranking[:kmax]
                Xtr_ranked = Xtr[:, ranked_prefix]
                Xte_ranked = Xte[:, ranked_prefix]

                # ML-kNN expects {-1,+1} with shape (L, n)
                Ytr_pm = (Ytr > 0).astype(np.int8).T
                Ytr_pm[Ytr_pm == 0] = -1
                Yte_pm = (Yte > 0).astype(np.int8).T
                Yte_pm[Yte_pm == 0] = -1

                t0 = time.time()
                curve = mlknn_eval_kgrid_incremental(
                    Xtr_ranked,
                    Ytr_pm,
                    Xte_ranked,
                    Yte_pm,
                    k_unique,
                    Num=int(args.mlknn_k),
                    Smooth=float(args.mlknn_smooth),
                )
                eval_time = time.time() - t0

                # Map evaluated curve back to the user-defined p grid.
                micro = np.zeros((p_values.size,), dtype=np.float64)
                macro = np.zeros((p_values.size,), dtype=np.float64)
                hl = np.zeros((p_values.size,), dtype=np.float64)
                mi_pr = np.zeros((p_values.size,), dtype=np.float64)
                ma_pr = np.zeros((p_values.size,), dtype=np.float64)
                ap = np.zeros((p_values.size,), dtype=np.float64)
                oe = np.zeros((p_values.size,), dtype=np.float64)
                k_arr = np.asarray(curve["k_values"], dtype=int)
                for i, kk in enumerate(k_for_p):
                    idx = int(np.where(k_arr == int(kk))[0][0])
                    micro[i] = float(curve["micro_f1"][idx])
                    macro[i] = float(curve["macro_f1"][idx])
                    hl[i] = float(curve["hamming_loss"][idx])
                    mi_pr[i] = float(curve["micro_pr_auc"][idx])
                    ma_pr[i] = float(curve["macro_pr_auc"][idx])
                    ap[i] = float(curve["avg_precision"][idx])
                    oe[i] = float(curve["one_error"][idx])

                # Convenience: pick metric at p_target (closest).
                idx_t = int(np.argmin(np.abs(p_values - p_target)))
                p_used = float(p_values[idx_t])

                obj = {
                    "grid_mode": "pgrid",
                    "p_values": [float(x) for x in p_values.tolist()],
                    "k_values": [int(x) for x in k_for_p.tolist()],
                    "micro_f1": [float(x) for x in micro.tolist()],
                    "macro_f1": [float(x) for x in macro.tolist()],
                    "hamming_loss": [float(x) for x in hl.tolist()],
                    "micro_pr_auc": [float(x) for x in mi_pr.tolist()],
                    "macro_pr_auc": [float(x) for x in ma_pr.tolist()],
                    "avg_precision": [float(x) for x in ap.tolist()],
                    "one_error": [float(x) for x in oe.tolist()],
                    "p_target": float(p_target),
                    "p_target_used": p_used,
                    "micro_f1_at_p_target": float(micro[idx_t]),
                    "macro_f1_at_p_target": float(macro[idx_t]),
                    "hamming_loss_at_p_target": float(hl[idx_t]),
                    "micro_pr_auc_at_p_target": float(mi_pr[idx_t]),
                    "macro_pr_auc_at_p_target": float(ma_pr[idx_t]),
                    "avg_precision_at_p_target": float(ap[idx_t]),
                    "one_error_at_p_target": float(oe[idx_t]),
                    "micro_f1_mean": float(np.mean(micro)),
                    "macro_f1_mean": float(np.mean(macro)),
                    "hamming_loss_mean": float(np.mean(hl)),
                    "micro_pr_auc_mean": float(np.mean(mi_pr)),
                    "macro_pr_auc_mean": float(np.mean(ma_pr)),
                    "avg_precision_mean": float(np.mean(ap)),
                    "one_error_mean": float(np.mean(oe)),
                    "macro_pr_auc_labels_used": int(curve.get("macro_pr_auc_labels_used", 0)),
                    "macro_pr_auc_labels_total": int(curve.get("macro_pr_auc_labels_total", n_labels)),
                    "n_features": int(n_features),
                    "n_labels": int(n_labels),
                    "time_eval_seconds": float(eval_time),
                }

                out_path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
                if done % 10 == 0:
                    print(f"[{done}/{total}] {method} | {ds} | fold{fold} ✓ (p@{p_pct}% MiF1={obj['micro_f1_at_p_target']:.4f})")

    print("✓ Done.")


if __name__ == "__main__":
    main()
