#!/usr/bin/env python
"""End-to-end fraud detection pipeline with MLflow experiment tracking.

Usage
-----
# Full pipeline
python scripts/run_pipeline.py

# Specific stages
python scripts/run_pipeline.py --stages eda features graph train

# Custom config
python scripts/run_pipeline.py --config configs/config.yaml

# Named MLflow run
python scripts/run_pipeline.py --run-name experiment_v2
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow

from fraud_detection.config import flat_params, get_path, load_config
from fraud_detection.data import load_raw, make_split
from fraud_detection.eda import run_eda
from fraud_detection.lgbm import run_lgb
from fraud_detection.evaluate import run_evaluation
from fraud_detection.features import run_features
from fraud_detection.graph import run_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

ALL_STAGES = ["eda", "features", "graph", "train"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IEEE-CIS Fraud Detection Pipeline")
    p.add_argument("--config",    default=None, help="Path to YAML config")
    p.add_argument("--stages",    nargs="+", default=ALL_STAGES, choices=ALL_STAGES,
                   help="Pipeline stages to run (default: all)")
    p.add_argument("--run-name",  default=None, help="MLflow run name (default: timestamp)")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)
    stages = args.stages

    # ── MLflow ───────────────────────────────────────────────────────────
    mlruns_dir = get_path(cfg, "mlruns_dir")
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    db_path = mlruns_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    run_name = args.run_name or cfg["mlflow"].get("run_name") or datetime.now().strftime("%Y%m%d_%H%M%S")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flat_params(cfg))
        mlflow.log_param("stages", ",".join(stages))

        t_total = time.time()
        logger.info("=" * 60)
        logger.info("Run: %s  |  Stages: %s", run_name, stages)
        logger.info("=" * 60)

        # ── Load raw data (shared across eda / features / graph) ──────────
        df = None
        train_txids: set[int] = set()
        test_txids:  set[int] = set()

        if any(s in stages for s in ["eda", "features", "graph"]):
            logger.info("Loading raw data ...")
            df = load_raw(cfg)
            train_txids, test_txids = make_split(df, cfg)

        # ── EDA ──────────────────────────────────────────────────────────
        if "eda" in stages:
            logger.info("\n[EDA]")
            t0 = time.time()
            run_eda(df, cfg)
            mlflow.log_metric("eda_time_s", round(time.time() - t0, 1))

        # ── Features ─────────────────────────────────────────────────────
        if "features" in stages:
            logger.info("\n[FEATURES]")
            t0 = time.time()
            feat_meta = run_features(df, train_txids, test_txids, cfg)
            mlflow.log_params({
                "feat_n_features": feat_meta["n_features"],
                "feat_train_size": feat_meta["train_size"],
                "feat_test_size":  feat_meta["test_size"],
            })
            mlflow.log_metric("features_time_s", round(time.time() - t0, 1))

        # ── Graph ─────────────────────────────────────────────────────────
        if "graph" in stages:
            logger.info("\n[GRAPH]")
            t0 = time.time()
            run_graph(df, train_txids, test_txids, cfg)
            mlflow.log_metric("graph_time_s", round(time.time() - t0, 1))

        # Free raw data before training
        if df is not None:
            del df
            gc.collect()

        # ── Train LightGBM ────────────────────────────────────────────────
        if "train" in stages:
            logger.info("\n[TRAIN — LightGBM]")
            t0     = time.time()
            result = run_lgb(cfg)
            eval_metrics = run_evaluation(result, cfg)
            mlflow.log_metric("train_time_s", round(time.time() - t0, 1))

            logger.info("\n" + "=" * 60)
            logger.info("FINAL RESULTS")
            logger.info("  LightGBM AUROC : %.4f", result["auroc"])
            logger.info("  LightGBM AUPRC : %.4f", result["auprc"])
            logger.info("=" * 60)

        total_time = round(time.time() - t_total, 1)
        mlflow.log_metric("total_time_s", total_time)
        logger.info("Pipeline finished in %.1fs", total_time)
        logger.info("MLflow run: %s", mlflow.active_run().info.run_id)
        logger.info("View results: mlflow ui --backend-store-uri sqlite:///%s", db_path.resolve())


if __name__ == "__main__":
    main()