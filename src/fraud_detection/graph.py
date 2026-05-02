"""Card-merchant bipartite graph features with leakage-safe fraud_density."""

from __future__ import annotations

import gc
import json
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

GRAPH_FEATURES = [
    "g_card_degree",
    "g_card_weighted_degree",
    "g_clustering",
    "g_pagerank",
    "g_avg_amount",
    "g_amount_var",
    "g_merchant_diversity",
]


def _lookup(d: dict, keys: np.ndarray, default: float = 0.0) -> np.ndarray:
    vals = np.array([d.get(k, default) for k in keys], dtype=np.float64)
    return np.clip(vals, np.finfo(np.float32).min, np.finfo(np.float32).max).astype(np.float32)


def build_graph_features(
    df: pd.DataFrame,
    train_txids: set[int],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Compute all graph features; return DataFrame with TransactionID + Class."""
    max_merch = cfg["graph"]["max_merchant_size"]
    pr_alpha = cfg["graph"]["pagerank_alpha"]
    pr_iters = cfg["graph"]["pagerank_iters"]
    pr_tol = cfg["graph"]["pagerank_tol"]

    df = df.copy()
    df["card_id"] = df["card1"].fillna(0).astype(int)
    df["merchant_id"] = (
        df["ProductCD"].fillna("X").astype(str)
        + "_"
        + df["card4"].fillna("unk").astype(str)
        + "_M"
    )
    df["log_amount"] = np.log1p(df["Amount"].fillna(0))
    df = df.sort_values("Time").reset_index(drop=True)

    cards = df["card_id"].unique()
    card_to_idx = {c: i for i, c in enumerate(cards)}
    n_cards = len(cards)

    # ── Step 1: pandas metrics ───────────────────────────────────────────
    logger.info("Step 1/4: pandas metrics ...")
    t0 = time.time()

    df_train = df[df["TransactionID"].isin(train_txids)]
    wdeg_train = df_train.groupby("card_id").size()
    fraud_per_card = df_train.groupby("card_id")["Class"].sum()
    fraud_density = (fraud_per_card / wdeg_train.clip(lower=1)).to_dict()
    del df_train, wdeg_train, fraud_per_card

    degree_s = df.groupby("card_id")["merchant_id"].nunique()
    wdeg_s = df.groupby("card_id").size()
    amt_stats = df.groupby("card_id")["log_amount"].agg(["mean", "var"])
    degree_dict = degree_s.to_dict()
    weighted_degree = wdeg_s.to_dict()
    avg_amt = amt_stats["mean"].fillna(0).to_dict()
    amt_var = amt_stats["var"].fillna(0).to_dict()
    merch_div = {c: degree_dict.get(c, 0) / max(weighted_degree.get(c, 1), 1) for c in cards}
    del degree_s, wdeg_s, amt_stats
    logger.info("  %.1fs", time.time() - t0)

    # ── Step 2: sparse adjacency (shared-merchant card pairs) ────────────
    logger.info("Step 2/4: sparse adjacency ...")
    t0 = time.time()
    merchant_groups = df.groupby("merchant_id")["card_id"].apply(list)
    rows_list, cols_list = [], []
    for mc in merchant_groups:
        idxs = [card_to_idx[c] for c in mc if c in card_to_idx]
        if len(idxs) < 2 or len(idxs) > max_merch:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                rows_list.append(idxs[a])
                cols_list.append(idxs[b])
                rows_list.append(idxs[b])
                cols_list.append(idxs[a])
    del merchant_groups
    rows = np.array(rows_list, dtype=np.int32)
    cols = np.array(cols_list, dtype=np.int32)
    del rows_list, cols_list
    A = sp.csr_matrix(
        sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(n_cards, n_cards),
        )
    )
    del rows, cols
    A.data[:] = 1.0
    A.eliminate_zeros()
    logger.info("  adj shape=%s  nnz=%s  %.1fs", A.shape, f"{A.nnz:,}", time.time() - t0)

    # ── Step 3: local clustering coefficient ────────────────────────────
    logger.info("Step 3/4: clustering coefficients ...")
    t0 = time.time()
    A2 = A.dot(A)
    tri_arr = np.array(A.multiply(A2.T).sum(axis=1)).ravel() / 2
    del A2
    deg_arr = np.array(A.sum(axis=1)).ravel()
    denom = deg_arr * (deg_arr - 1)
    with np.errstate(invalid="ignore"):
        cc_arr = np.where(denom > 0, 2.0 * tri_arr / denom, 0.0).astype(np.float32)
    del tri_arr, denom
    logger.info("  mean_cc=%.4f  %.1fs", cc_arr.mean(), time.time() - t0)

    # ── Step 4: PageRank ─────────────────────────────────────────────────
    logger.info("Step 4/4: PageRank ...")
    t0 = time.time()
    with np.errstate(divide="ignore"):
        deg_inv = np.where(deg_arr > 0, 1.0 / deg_arr, 0.0)
    del deg_arr

    merch_groups2 = df.groupby("merchant_id")["card_id"].apply(list)
    t_rows, t_cols, t_vals = [], [], []
    for mc in merch_groups2:
        idxs = [card_to_idx[c] for c in mc if c in card_to_idx]
        if len(idxs) < 2 or len(idxs) > max_merch:
            continue
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                t_rows.append(idxs[b])
                t_cols.append(idxs[a])
                t_vals.append(deg_inv[idxs[a]])
                t_rows.append(idxs[a])
                t_cols.append(idxs[b])
                t_vals.append(deg_inv[idxs[b]])
    del merch_groups2, deg_inv

    T = sp.csr_matrix(
        (
            np.array(t_vals, dtype=np.float64),
            (np.array(t_rows, dtype=np.int32), np.array(t_cols, dtype=np.int32)),
        ),
        shape=(n_cards, n_cards),
    )
    del t_rows, t_cols, t_vals

    r = np.full(n_cards, 1.0 / n_cards, dtype=np.float64)
    for _ in range(pr_iters):
        r_new = pr_alpha * T.dot(r) + (1 - pr_alpha) / n_cards
        if np.linalg.norm(r_new - r, 1) < pr_tol:
            break
        r = r_new
    del T, r
    pagerank = {cards[i]: float(r_new[i]) for i in range(n_cards)}
    del r_new
    logger.info("  %.1fs", time.time() - t0)

    # ── Assemble feature matrix ──────────────────────────────────────────
    idx_arr = np.array([card_to_idx.get(c, 0) for c in df["card_id"].values], dtype=np.int32)
    del card_to_idx, A

    X_graph = np.column_stack(
        [
            np.take(_lookup(degree_dict, cards), idx_arr),
            np.take(_lookup(weighted_degree, cards), idx_arr),
            np.take(cc_arr, idx_arr),
            np.take(_lookup(pagerank, cards), idx_arr),
            np.take(_lookup(avg_amt, cards), idx_arr),
            np.take(_lookup(amt_var, cards), idx_arr),
            np.take(_lookup(merch_div, cards), idx_arr),
        ]
    )
    del degree_dict, weighted_degree, cc_arr, pagerank, avg_amt, amt_var, merch_div, idx_arr, cards

    gdf = pd.DataFrame(X_graph, columns=GRAPH_FEATURES)
    del X_graph
    gdf["TransactionID"] = df["TransactionID"].values
    gdf["Class"] = df["Class"].values
    gc.collect()

    logger.info("Graph features: %s | fraud=%s", gdf.shape, gdf["Class"].sum())
    return gdf


def run_graph(
    df: pd.DataFrame,
    train_txids: set[int],
    test_txids: set[int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build graph features, scale, and save to processed/."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)

    gdf = build_graph_features(df, train_txids, cfg)

    y_g = gdf["Class"].values.astype(np.int8)
    X_g = gdf[GRAPH_FEATURES].values.astype(np.float32)
    tx_g = gdf["TransactionID"].values
    del gdf
    gc.collect()

    X_g = np.nan_to_num(X_g, nan=0.0, posinf=0.0, neginf=0.0)

    tr_mask = np.array([tid in train_txids for tid in tx_g])
    te_mask = np.array([tid in test_txids for tid in tx_g])
    tr_i = np.where(tr_mask)[0]
    te_i = np.where(te_mask)[0]

    assert len(set(tx_g[tr_i].tolist()) & set(tx_g[te_i].tolist())) == 0

    gs = RobustScaler()
    X_g[tr_i] = gs.fit_transform(X_g[tr_i])
    X_g[te_i] = gs.transform(X_g[te_i])
    X_g = np.clip(X_g, -10.0, 10.0)

    np.save(proc_dir / "graph_features_train.npy", X_g[tr_i])
    np.save(proc_dir / "graph_features_test.npy", X_g[te_i])
    np.save(proc_dir / "graph_tx_ids_train.npy", tx_g[tr_i])
    np.save(proc_dir / "graph_tx_ids_test.npy", tx_g[te_i])
    np.save(proc_dir / "y_graph_train.npy", y_g[tr_i])
    np.save(proc_dir / "y_graph_test.npy", y_g[te_i])

    with open(proc_dir / "graph_scaler.pkl", "wb") as fh:
        pickle.dump(gs, fh)

    meta = {"graph_features": GRAPH_FEATURES, "n_features": len(GRAPH_FEATURES)}
    with open(proc_dir / "graph_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "Saved graph arrays — train: %s (fraud=%s %.2f%%) | test: %s (fraud=%s %.2f%%)",
        X_g[tr_i].shape,
        y_g[tr_i].sum(),
        y_g[tr_i].mean() * 100,
        X_g[te_i].shape,
        y_g[te_i].sum(),
        y_g[te_i].mean() * 100,
    )
    return meta