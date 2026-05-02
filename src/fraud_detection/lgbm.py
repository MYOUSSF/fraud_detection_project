"""LightGBM training and SHAP interpretability.

XGBoost and the meta-learner ensemble have been removed.
The model is LightGBM only — simpler, faster, easier to maintain.
"""

from __future__ import annotations

import gc
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# ── Feature loader ────────────────────────────────────────────────────────────

def _load_features(proc_dir: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Load and join tabular + graph features for train and test.

    Both arrays are now indexed by transaction (no sequence compression),
    so the join is a straightforward TransactionID lookup with no size mismatch.
    """
    with open(proc_dir / "feature_meta.json") as fh:
        feat_meta = json.load(fh)
    with open(proc_dir / "graph_meta.json") as fh:
        graph_meta = json.load(fh)

    tab_feats  = feat_meta["features"]
    gph_feats  = graph_meta["graph_features"]
    feature_names = tab_feats + gph_feats
    N_TAB = len(tab_feats)

    X_train_tab   = np.load(proc_dir / "X_train_tab_all.npy")
    tx_ids_train  = np.load(proc_dir / "tx_ids_train_all.npy")
    y_train       = np.load(proc_dir / "y_train_all.npy")

    X_test_tab    = np.load(proc_dir / "X_test_tab.npy")
    tx_ids_test   = np.load(proc_dir / "tx_ids_test_tab.npy")
    y_test        = np.load(proc_dir / "y_test.npy")

    G_train  = np.load(proc_dir / "graph_features_train.npy")
    g_tx_tr  = np.load(proc_dir / "graph_tx_ids_train.npy")
    G_test   = np.load(proc_dir / "graph_features_test.npy")
    g_tx_te  = np.load(proc_dir / "graph_tx_ids_test.npy")

    def join(G, g_tx, X_tab, tx_ids):
        lookup = {int(tid): i for i, tid in enumerate(tx_ids)}
        block = np.zeros((len(G), N_TAB), dtype=np.float32)
        for i, tid in enumerate(g_tx):
            idx = lookup.get(int(tid))
            if idx is not None:
                block[i] = X_tab[idx]
        return np.hstack([block, G])

    X_train = join(G_train, g_tx_tr, X_train_tab, tx_ids_train)
    X_test  = join(G_test,  g_tx_te, X_test_tab,  tx_ids_test)
    y_train = y_train[np.isin(tx_ids_train, g_tx_tr)]  # align to graph row order
    # Graph arrays already carry correct labels — use those directly
    y_train = np.load(proc_dir / "y_graph_train.npy")
    y_test  = np.load(proc_dir / "y_graph_test.npy")

    del G_train, G_test, X_train_tab, X_test_tab
    gc.collect()

    # TransactionDT for the test set — aligned to graph row order via g_tx_te
    tx_dt_test_full = np.load(proc_dir / "tx_dt_test.npy")   # indexed by tx_ids_test
    tx_dt_lookup    = {int(tid): dt for tid, dt in zip(tx_ids_test, tx_dt_test_full)}
    tx_dt_test      = np.array([tx_dt_lookup.get(int(tid), 0.0) for tid in g_tx_te], dtype=np.float64)

    logger.info(
        "Features loaded — train: %s (fraud=%s, %.2f%%) | test: %s (fraud=%s, %.2f%%)",
        X_train.shape, y_train.sum(), y_train.mean() * 100,
        X_test.shape,  y_test.sum(),  y_test.mean()  * 100,
    )
    return X_train, y_train, X_test, y_test, g_tx_te, tx_dt_test, feature_names


# ── LightGBM ──────────────────────────────────────────────────────────────────

def train_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
    feature_names: list[str],
) -> tuple[lgb.LGBMClassifier, np.ndarray, float, float]:
    """5-fold stratified CV to find best_iteration, then retrain on full data."""
    cfg_lgb  = cfg["lgb"]
    n_splits = cfg["training"]["cv_folds"]
    rs       = cfg["training"]["random_state"]

    n_normal = int((y_train == 0).sum())
    n_fraud  = int((y_train == 1).sum())
    spw      = n_normal / max(n_fraud, 1)
    logger.info("scale_pos_weight=%.1f  (normal=%s  fraud=%s)", spw, f"{n_normal:,}", f"{n_fraud:,}")

    params = {
        "objective":        cfg_lgb["objective"],
        "metric":           cfg_lgb["metric"],
        "boosting_type":    cfg_lgb["boosting_type"],
        "n_estimators":     cfg_lgb["n_estimators"],
        "learning_rate":    cfg_lgb["learning_rate"],
        "num_leaves":       cfg_lgb["num_leaves"],
        "min_child_samples":cfg_lgb["min_child_samples"],
        "subsample":        cfg_lgb["subsample"],
        "subsample_freq":   cfg_lgb["subsample_freq"],
        "colsample_bytree": cfg_lgb["colsample_bytree"],
        "reg_alpha":        cfg_lgb["reg_alpha"],
        "reg_lambda":       cfg_lgb["reg_lambda"],
        "scale_pos_weight": spw,
        "random_state":     rs,
        "device":           cfg_lgb["device"],
        "verbose":          -1,
    }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    oof  = np.zeros(len(X_train))
    cv_aurocs: list[float] = []
    best_iters: list[int]  = []

    logger.info("Cross-validated LightGBM (%s folds) ...", n_splits)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        t0  = time.time()
        clf = lgb.LGBMClassifier(**params, n_jobs=-1)
        clf.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[
                lgb.early_stopping(cfg_lgb["early_stopping_rounds"], verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        val_proba        = clf.predict_proba(X_train[val_idx])[:, 1]
        oof[val_idx]     = val_proba
        fold_auroc       = roc_auc_score(y_train[val_idx], val_proba)
        cv_aurocs.append(fold_auroc)
        best_iters.append(clf.best_iteration_)
        logger.info(
            "  Fold %s: AUROC=%.4f | best_iter=%s | %.1fs",
            fold + 1, fold_auroc, clf.best_iteration_, time.time() - t0,
        )
        mlflow.log_metric(f"lgb_cv_auroc_fold{fold+1}", fold_auroc, step=fold)

    cv_mean = float(np.mean(cv_aurocs))
    cv_std  = float(np.std(cv_aurocs))
    oof_auroc = float(roc_auc_score(y_train, oof))
    logger.info("CV mean=%.4f +/- %.4f | OOF AUROC=%.4f", cv_mean, cv_std, oof_auroc)
    mlflow.log_metrics({"lgb_cv_auroc_mean": cv_mean, "lgb_cv_auroc_std": cv_std, "lgb_oof_auroc": oof_auroc})

    # Final model on full training set
    best_n = int(np.median(best_iters))
    logger.info("Final LightGBM | n_estimators=%s ...", best_n)
    final = lgb.LGBMClassifier(**{**params, "n_estimators": best_n}, n_jobs=-1)
    final.fit(X_train, y_train, feature_name=feature_names)

    test_scores = final.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, test_scores))
    auprc = float(average_precision_score(y_test, test_scores))
    logger.info("LightGBM Test — AUROC=%.4f  AUPRC=%.4f", auroc, auprc)
    return final, test_scores, auroc, auprc


# ── SHAP ──────────────────────────────────────────────────────────────────────

def compute_shap(
    model: lgb.LGBMClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    cfg: dict[str, Any],
    proc_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values on a stratified sample of test transactions."""
    import shap

    n_shap    = min(cfg["lgb"]["n_shap_samples"], len(X_test))
    fraud_idx  = np.where(y_test == 1)[0]
    normal_idx = np.where(y_test == 0)[0]
    rng        = np.random.default_rng(cfg["training"]["random_state"])

    n_f = min(len(fraud_idx),  n_shap // 2)
    n_n = min(len(normal_idx), n_shap - n_f)
    shap_idx = np.concatenate([
        rng.choice(fraud_idx,  n_f, replace=False),
        rng.choice(normal_idx, n_n, replace=False),
    ])
    X_shap = X_test[shap_idx]

    logger.info("Computing SHAP values (n=%s) ...", len(shap_idx))
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    np.save(proc_dir / "shap_values.npy",       shap_values)
    np.save(proc_dir / "shap_sample_idx.npy",   shap_idx)
    np.save(proc_dir / "shap_feature_names.npy", np.array(feature_names, dtype=object))
    logger.info("SHAP values saved. Shape: %s", shap_values.shape)
    return shap_values, shap_idx


# ── Main entry point ──────────────────────────────────────────────────────────

def run_lgb(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train LightGBM, compute SHAP, return everything evaluate.py needs."""
    proc_dir = Path(cfg["paths"]["processed_dir"])
    mdl_dir  = Path(cfg["paths"]["models_dir"])
    mdl_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test, tx_ids_test, tx_dt_test, feature_names = _load_features(proc_dir)

    model, test_scores, auroc, auprc = train_lgb(
        X_train, y_train, X_test, y_test, cfg, feature_names
    )

    # Save model
    with open(mdl_dir / "lgb_model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    np.save(proc_dir / "lgb_scores_test.npy",  test_scores)
    np.save(proc_dir / "lgb_tx_ids_test.npy",  tx_ids_test)

    # SHAP
    shap_values, shap_idx = compute_shap(model, X_test, y_test, feature_names, cfg, proc_dir)

    return {
        "lgb_scores":    test_scores,
        "model":         model,
        "y_test":        y_test,
        "X_test_full":   X_test,
        "tx_dt_test":    tx_dt_test,
        "shap_values":   shap_values,
        "shap_idx":      shap_idx,
        "feature_names": feature_names,
        "auroc":         auroc,
        "auprc":         auprc,
    }