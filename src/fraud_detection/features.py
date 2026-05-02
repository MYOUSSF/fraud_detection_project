"""Feature engineering: tabular features and train/test saving.

No sequences — all 590k transactions are preserved as flat feature vectors.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

ENGINEERED = ["log_amount", "time_delta", "rolling_amount_mean_3", "amount_vs_mean"]


def get_v_features(df: pd.DataFrame, n: int = 28) -> list[str]:
    """Return the first n V-column names, zero-padding if the dataset has fewer."""
    for i in range(len([c for c in df.columns if c.startswith("V")]) + 1, n + 1):
        df[f"V{i}"] = 0.0
    return [f"V{i}" for i in range(1, n + 1)]


def engineer_features(df: pd.DataFrame, n_v: int = 28) -> tuple[pd.DataFrame, list[str]]:
    """Add engineered columns in-place; return (df, feature_names)."""
    v_feats = get_v_features(df, n_v)

    df["log_amount"] = np.log1p(df["Amount"].fillna(0))
    df["time_delta"] = df.groupby("user_id")["Time"].diff().fillna(0)
    df["rolling_amount_mean_3"] = (
        df.groupby("user_id")["log_amount"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["amount_vs_mean"] = df["log_amount"] - df["rolling_amount_mean_3"]

    feature_names = v_feats + ENGINEERED
    df[feature_names] = df[feature_names].fillna(0)
    logger.info(
        "Feature columns: %s  (%s V + %s engineered)",
        len(feature_names), len(v_feats), len(ENGINEERED),
    )
    return df, feature_names


def fit_scaler(
    df: pd.DataFrame,
    feature_names: list[str],
    train_txids: set[int],
    proc_dir: Path,
) -> RobustScaler:
    """Fit RobustScaler on training-set normal transactions only; transform df in-place."""
    mask = df["TransactionID"].isin(train_txids) & (df["Class"] == 0)
    scaler = RobustScaler()
    scaler.fit(df.loc[mask, feature_names])
    df[feature_names] = scaler.transform(df[feature_names])
    with open(proc_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info(
        "Scaler fitted on %s training-normal rows → saved scaler.pkl",
        f"{mask.sum():,}",
    )
    return scaler


def run_features(
    df: pd.DataFrame,
    train_txids: set[int],
    test_txids: set[int],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Engineer features, scale, split, and save flat arrays.

    Saves
    -----
    X_train_tab_all.npy   (n_train, n_features)  float32
    X_test_tab.npy        (n_test,  n_features)  float32
    y_train_all.npy       (n_train,)  int8
    y_test.npy            (n_test,)   int8
    tx_ids_train_all.npy  (n_train,)  int64
    tx_ids_test_tab.npy   (n_test,)   int64
    scaler.pkl
    feature_meta.json
    """
    proc_dir = Path(cfg["paths"]["processed_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    n_v = cfg["features"]["n_v_features"]

    df, feature_names = engineer_features(df, n_v)
    fit_scaler(df, feature_names, train_txids, proc_dir)

    # Flat feature matrix — all rows, clipped to +-5
    X = np.clip(df[feature_names].values, -5, 5).astype(np.float32)
    y = df["Class"].values.astype(np.int8)
    tx_ids = df["TransactionID"].values.astype(np.int64)

    tr_mask = np.isin(tx_ids, np.fromiter(train_txids, dtype=np.int64))
    te_mask = np.isin(tx_ids, np.fromiter(test_txids,  dtype=np.int64))

    assert not np.any(tr_mask & te_mask), "Train/test overlap in feature arrays!"

    np.save(proc_dir / "X_train_tab_all.npy",  X[tr_mask])
    np.save(proc_dir / "X_test_tab.npy",        X[te_mask])
    np.save(proc_dir / "y_train_all.npy",       y[tr_mask])
    np.save(proc_dir / "y_test.npy",            y[te_mask])
    np.save(proc_dir / "tx_ids_train_all.npy",  tx_ids[tr_mask])
    np.save(proc_dir / "tx_ids_test_tab.npy",   tx_ids[te_mask])

    # Save raw TransactionDT for time-period evaluation (seconds offset)
    tx_dt = df["Time"].values.astype(np.float64)
    np.save(proc_dir / "tx_dt_train.npy", tx_dt[tr_mask])
    np.save(proc_dir / "tx_dt_test.npy",  tx_dt[te_mask])

    meta = {
        "n_features": len(feature_names),
        "features": feature_names,
        "train_size": int(tr_mask.sum()),
        "test_size":  int(te_mask.sum()),
        "train_fraud": int(y[tr_mask].sum()),
        "test_fraud":  int(y[te_mask].sum()),
    }
    with open(proc_dir / "feature_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    logger.info(
        "Saved: train=%s (fraud=%s, %.2f%%) | test=%s (fraud=%s, %.2f%%)",
        f"{tr_mask.sum():,}", f"{y[tr_mask].sum():,}", y[tr_mask].mean() * 100,
        f"{te_mask.sum():,}", f"{y[te_mask].sum():,}", y[te_mask].mean() * 100,
    )
    return meta