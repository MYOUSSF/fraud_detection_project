"""Data loading and initial preparation for IEEE-CIS Fraud Detection."""

# IMPORTANT NOTE ON EVALUATION
# For a production system you'd want walk-forward CV for
# model selection and hyperparameter tuning, but the per-fold graph
# rebuild is usually skipped in practice — the approximation error
# from static graph features is small relative to the gains from
# correct temporal evaluation.
# The current pipeline's stratified split is a reasonable starting
# point but the AUROC of 0.8065 is likely 0.03–0.05 points
# optimistic compared to what you'd see in deployment.

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Merge train_transaction + train_identity, sort by time."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    tx_path = raw_dir / cfg["data"]["train_transaction"]
    id_path = raw_dir / cfg["data"]["train_identity"]

    logger.info("Loading %s ...", tx_path.name)
    tx = pd.read_csv(tx_path)
    logger.info("  transactions: %s rows x %s cols", *tx.shape)

    logger.info("Loading %s ...", id_path.name)
    id_ = pd.read_csv(id_path)
    logger.info("  identity    : %s rows x %s cols", *id_.shape)

    df = tx.merge(id_, on="TransactionID", how="left")
    del tx, id_
    gc.collect()

    df = df.rename(
        columns={
            "isFraud": "Class",
            "TransactionAmt": "Amount",
            "TransactionDT": "Time",
        }
    )
    df["user_id"] = df["card1"].fillna(0).astype(int)
    df = df.sort_values(["user_id", "Time"]).reset_index(drop=True)

    fraud_count = int(df["Class"].sum())
    pct = fraud_count / len(df) * 100
    logger.info(
        "Merged: %s rows | fraud=%s (%.3f%%) | cardholders=%s",
        f"{len(df):,}",
        f"{fraud_count:,}",
        pct,
        f"{df['user_id'].nunique():,}",
    )
    return df


def make_split(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[set[int], set[int]]:
    """Stratified train/test split on TransactionID.

    Returns
    -------
    train_txids, test_txids : sets of integers
    """
    all_txids = df["TransactionID"].values
    labels = df["Class"].values.astype(np.int8)

    tr_idx, te_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=labels,
    )
    train_txids = set(all_txids[tr_idx].tolist())
    test_txids = set(all_txids[te_idx].tolist())

    assert len(train_txids & test_txids) == 0, "Train/test overlap!"
    logger.info(
        "Split: train=%s  test=%s | train_fraud=%.2f%%  test_fraud=%.2f%%",
        f"{len(train_txids):,}",
        f"{len(test_txids):,}",
        labels[tr_idx].mean() * 100,
        labels[te_idx].mean() * 100,
    )
    return train_txids, test_txids
