"""
graph_export.py
---------------
Exports a per-card graph feature lookup table that the API uses at inference
time to retrieve graph features for known cards in < 1 ms.

Run once after the training pipeline completes:

    python scripts/export_graph_lookup.py

Output
------
data/processed/card_graph_lookup.json
    { "<card_id>": { "g_card_degree": …, "g_clustering": …, … }, … }

This file is loaded by api/app.py at startup into an in-memory dict.
For unseen cards (not in training data) the API defaults all graph features to 0.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_detection.config import load_config
from fraud_detection.graph import GRAPH_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("export_graph")


def main() -> None:
    cfg = load_config()
    proc_dir = Path(cfg["paths"]["processed_dir"])
    out_path = proc_dir / "card_graph_lookup.json"

    # Load raw transaction data to get card_id <-> TransactionID mapping
    import gc
    import pandas as pd

    raw_dir = Path(cfg["paths"]["raw_dir"])
    tx_path = raw_dir / cfg["data"]["train_transaction"]

    logger.info("Loading transaction CSV for card_id mapping ...")
    tx = pd.read_csv(tx_path, usecols=["TransactionID", "card1"])
    tx["card_id"] = tx["card1"].fillna(0).astype(int)
    txid_to_card = tx.set_index("TransactionID")["card_id"].to_dict()
    del tx
    gc.collect()

    # Load graph features (train set)
    g_feats = np.load(proc_dir / "graph_features_train.npy")
    g_txids = np.load(proc_dir / "graph_tx_ids_train.npy")

    logger.info("Building card lookup from %s training transactions ...", f"{len(g_txids):,}")

    # Aggregate per card: take the mean of all feature vectors for that card
    from collections import defaultdict

    card_feat_sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(len(GRAPH_FEATURES)))
    card_counts: dict[int, int] = defaultdict(int)

    for i, tid in enumerate(g_txids):
        card_id = txid_to_card.get(int(tid))
        if card_id is None:
            continue
        card_feat_sums[card_id] += g_feats[i]
        card_counts[card_id] += 1

    lookup: dict[str, dict[str, float]] = {}
    for card_id, feat_sum in card_feat_sums.items():
        n = card_counts[card_id]
        feat_vec = (feat_sum / n).tolist()
        lookup[str(card_id)] = {f: round(v, 6) for f, v in zip(GRAPH_FEATURES, feat_vec)}

    with open(out_path, "w") as fh:
        json.dump(lookup, fh)

    logger.info("Saved card_graph_lookup.json  (%s unique cards)  →  %s", f"{len(lookup):,}", out_path)


if __name__ == "__main__":
    main()
