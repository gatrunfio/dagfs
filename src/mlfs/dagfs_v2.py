"""
DAGFS v2 (Python) — training-fold feature ranking for multi-label feature selection.

This is the **paper-aligned** implementation used throughout the experiments.
Given training features `X` and labels `Y`, it learns a nonnegative weight
matrix `W` and returns a **single feature ranking** (descending importance).

Key ingredients (all *training-only*, no test leakage):

1) Signed-deviation lift (center-split)
   - Expands the nonnegative feature space to represent **two-sided evidence**
     while preserving `W >= 0` and multiplicative updates.
   - `X -> [ (X - μ)_+ , (μ - X)_+ ]` where `μ` is the per-feature mean on the
     training fold.

2) Rarity-aware instance reweighting
   - Builds an instance weight from a rarity prior over labels (labels with
     fewer positives receive higher weight).
   - Stabilizes learning when supervision density is heterogeneous.

3) Directed label-transfer regularization
   - Learns features for rare labels while allowing controlled transfer from
     higher-support labels via a row-stochastic directed graph `C`.

4) Embedded nonnegative reconstruction + group sparsity
   - Optimizes a convex-in-`W` objective with multiplicative updates.
   - Ranks features by row/group norms; with lifting, ranking is performed in
     the *original* feature space via paired group norms (by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DAGFSv2Params:
    # ── core DAGFS parameters ───────────────────────────────────────
    alpha: float = 0.10
    beta: float = 0.10
    max_iter: int = 40
    rarity_gamma: float = 2.0
    tau_dir: float = 0.50
    topK: int = 10

    # Instance reweighting (rarity-aware)
    kappa: float = 1.50
    s_max: float = 3.0

    # Reproducibility / init
    seed: int = 0
    W0: Optional[np.ndarray] = None

    # Label-transfer graph
    transfer_graph: str = "directed"  # only 'directed' is supported
    feature_lift: str = "center_split"  # 'none' or 'center_split'
    paired_penalty: bool = True  # if lifted: use paired group sparsity and paired ranking


# ════════════════════════════════════════════════════════════════════
# Helpers: label transfer graph
# ════════════════════════════════════════════════════════════════════


def _build_C_directed(S_base: np.ndarray, freq: np.ndarray, tau: float, topK: int, epsv: float) -> np.ndarray:
    L = int(S_base.shape[0])
    # Reliability bias: prefer transfer from higher-support labels (teachers) to
    # lower-support labels (students). With row-stochastic normalization, this
    # corresponds to scaling outgoing weights by (f_row / f_col)^tau.
    ratio = np.outer(freq, np.ones(L)) / np.outer(np.ones(L), freq)
    ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)
    B = ratio ** float(tau)
    C = S_base * B

    if 0 < int(topK) < L:
        Csp = np.zeros_like(C)
        for i in range(L):
            idx = np.argsort(-C[i], kind="mergesort")[: int(topK)]
            Csp[i, idx] = C[i, idx]
        C = Csp

    C = C / (C.sum(axis=1)[:, None] + epsv)
    return C


def _split_psd(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = 0.5 * (M + M.T)
    Mabs = np.abs(M)
    Mplus = 0.5 * (Mabs + M)
    Mminus = 0.5 * (Mabs - M)
    return Mplus, Mminus


# ════════════════════════════════════════════════════════════════════
# Helpers: feature lift
# ════════════════════════════════════════════════════════════════════


def _lift_center_split(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    Xpos = np.maximum(X - mu, 0.0)
    Xneg = np.maximum(mu - X, 0.0)
    return np.concatenate([Xpos, Xneg], axis=1), mu


# ════════════════════════════════════════════════════════════════════
# Main algorithm
# ════════════════════════════════════════════════════════════════════


def dagfs_v2(
    X: np.ndarray,
    Y: np.ndarray,
    params: DAGFSv2Params | Dict[str, Any] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Compute a training-fold feature ranking with DAGFS.

    Parameters
    ----------
    X:
        Training features, shape `(n_samples, d)`. DAGFS is designed for
        nonnegative data; in the paper we use min–max scaling to `[0,1]`
        computed on the training fold.
    Y:
        Binary multi-label matrix, shape `(n_samples, L)` with entries in
        `{0,1}`.
    params:
        Either a `DAGFSv2Params` instance or a dict overriding defaults.

    Returns
    -------
    ranking:
        1-based feature indices, shape `(d,)`, sorted by decreasing importance.
        (Use `ranking - 1` for 0-based Python indexing.)
    W:
        Learned nonnegative weight matrix, shape `(d_lift, L)` where
        `d_lift = d` (no lift) or `2d` (center-split lift).
    info:
        Small dict with the effective configuration (useful for logging).
    """
    if params is None:
        p = DAGFSv2Params()
    elif isinstance(params, DAGFSv2Params):
        p = params
    else:
        p = DAGFSv2Params(**params)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    epsv = np.finfo(np.float64).eps

    n, d_orig = X.shape
    L = int(Y.shape[1])

    # ── Rarity prior (for instance weights and directed bias) ──────
    freq = Y.sum(axis=0) + 1.0
    prior = freq ** (-float(p.rarity_gamma))
    prior = prior / (prior.sum() + epsv)

    # ── Instance reweighting ───────────────────────────────────────
    # s_i = 1 + κ <y_i, prior>, clipped and re-centered to mean 1.
    s = 1.0 + float(p.kappa) * (Y @ prior)
    s = np.minimum(s, float(p.s_max))
    s = s / (s.mean() + epsv)
    sw = np.sqrt(s)
    Yw = Y * sw[:, None]

    # ── Label similarity and label-transfer graph ──────────────────
    # Cosine similarity in the (reweighted) label space.
    yn = np.sqrt((Yw * Yw).sum(axis=0))
    denom = np.outer(yn, yn) + epsv
    S_label = (Yw.T @ Yw) / denom
    S_label = np.nan_to_num(S_label, nan=0.0, posinf=0.0, neginf=0.0)
    S_label = np.maximum(S_label, 0.0)
    np.fill_diagonal(S_label, 0.0)

    graph_mode = str(p.transfer_graph).lower().strip()
    if graph_mode != "directed":
        raise ValueError("Only transfer_graph='directed' is supported in the cleaned implementation.")
    C = _build_C_directed(S_label, freq, float(p.tau_dir), int(p.topK), epsv)

    I = np.eye(L, dtype=np.float64)
    M = (I - C) @ (I - C).T
    Mplus, Mminus = _split_psd(M)

    # ── Feature lift ───────────────────────────────────────────────
    lift_mode = str(p.feature_lift).lower().strip()
    if lift_mode == "none":
        X_lift = X
        d = d_orig
    elif lift_mode == "center_split":
        X_lift, _mu = _lift_center_split(X)
        d = 2 * d_orig
    else:
        raise ValueError(f"Unknown feature_lift: {p.feature_lift!r}")

    Xw = X_lift * sw[:, None]

    # ── Initialise W ───────────────────────────────────────────────
    if p.W0 is not None:
        W = np.asarray(p.W0, dtype=np.float64).copy()
        if W.shape != (d, L):
            raise ValueError(f"W0 has shape {W.shape}, expected {(d, L)}")
    else:
        rng = np.random.default_rng(int(p.seed))
        W = rng.random((d, L), dtype=np.float64)

    XtX = Xw.T @ Xw
    XtY = Xw.T @ Yw

    paired = bool(p.paired_penalty)

    for _it in range(int(p.max_iter)):
        # ℓ2,1 subgradient weights (paired for lifted features when requested)
        if d == d_orig:
            row_norms = np.sqrt((W * W).sum(axis=1)) + epsv
            inv2norm = 1.0 / (2.0 * row_norms)
        elif d == 2 * d_orig:
            if paired:
                Wp = W[:d_orig]
                Wn = W[d_orig:]
                pair_norms = np.sqrt((Wp * Wp).sum(axis=1) + (Wn * Wn).sum(axis=1)) + epsv
                inv2 = 1.0 / (2.0 * pair_norms)
                inv2norm = np.concatenate([inv2, inv2], axis=0)
            else:
                row_norms = np.sqrt((W * W).sum(axis=1)) + epsv
                inv2norm = 1.0 / (2.0 * row_norms)
        else:
            raise ValueError("Lifted dimension incompatible.")

        Up = XtY + float(p.alpha) * (W @ Mminus)
        Dw = (XtX @ W) + float(p.alpha) * (W @ Mplus) + float(p.beta) * (inv2norm[:, None] * W)
        W = W * (Up / np.maximum(Dw, epsv))

    # ── Scores & ranking ───────────────────────────────────────────
    # Default (paper): if lifted, rank original features by paired group norms.
    # For analysis, we optionally allow an unpaired penalty/ranking over 2d rows.
    if d == d_orig:
        scores = np.sqrt((W * W).sum(axis=1))
        ranking0 = np.argsort(-scores, kind="mergesort")
        ranking = (ranking0 + 1).astype(np.int64)
    elif d == 2 * d_orig:
        if paired:
            Wp = W[:d_orig]
            Wn = W[d_orig:]
            scores = np.sqrt((Wp * Wp).sum(axis=1) + (Wn * Wn).sum(axis=1))
            ranking0 = np.argsort(-scores, kind="mergesort")
            ranking = (ranking0 + 1).astype(np.int64)
        else:
            scores = np.sqrt((W * W).sum(axis=1))
            ranking0 = np.argsort(-scores, kind="mergesort")
            ranking = (ranking0 + 1).astype(np.int64)
    else:
        raise ValueError("Lifted dimension incompatible.")

    info: Dict[str, Any] = {
        "feature_lift": lift_mode,
        "transfer_graph": graph_mode,
        "paired_penalty": paired,
    }
    return ranking, W, info
