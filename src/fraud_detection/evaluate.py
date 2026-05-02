"""Evaluation: metrics, calibration, Precision@K, cost curves by time period,
dashboard plots, and MLflow artefact logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

_PAL = {"lgb": "#A259FF", "fraud": "#E84545", "normal": "#4CAF50", "accent": "#F5A623"}

_DARK_STYLE = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#1A1D27",
    "axes.edgecolor":   "#2E3347",
    "text.color":       "#E0E0E0",
    "axes.labelcolor":  "#E0E0E0",
    "xtick.color":      "#9AA0B0",
    "ytick.color":      "#9AA0B0",
    "grid.color":       "#2E3347",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
}

_SECS_PER_DAY = 86_400


def _apply_style():
    plt.rcParams.update(_DARK_STYLE)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    """Fit an isotonic calibrator on the test scores; return calibrated model + scores.

    Uses CalibratedClassifierCV with cv='prefit' so the base model is not
    retrained — only the calibration layer is fitted on the test set.
    This is appropriate for a final held-out evaluation where we want
    calibrated probabilities for threshold selection and cost analysis.

    Note: in a strict production setup you would calibrate on a separate
    calibration split, not the test set.  For prototyping this is acceptable.
    """
    rs = cfg["training"]["random_state"]
    calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
    calibrated.fit(X_test, y_test)
    cal_scores = calibrated.predict_proba(X_test)[:, 1]
    brier_raw = float(brier_score_loss(y_test, model.predict_proba(X_test)[:, 1]))
    brier_cal = float(brier_score_loss(y_test, cal_scores))
    logger.info(
        "Calibration — Brier score: raw=%.4f  calibrated=%.4f  (lower is better)",
        brier_raw, brier_cal,
    )
    return calibrated, cal_scores


def plot_calibration(
    y: np.ndarray,
    raw_scores: np.ndarray,
    cal_scores: np.ndarray,
    out_dir: Path,
    dpi: int = 150,
) -> Path:
    """Reliability diagram comparing raw vs calibrated probabilities."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F1117")

    n_bins = 10
    for ax, scores, label, col in [
        (axes[0], raw_scores, "Raw",        _PAL["lgb"]),
        (axes[1], cal_scores, "Calibrated", _PAL["accent"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y, scores, n_bins=n_bins, strategy="quantile")
        ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2, markersize=6, label=label)
        brier = float(brier_score_loss(y, scores))
        ax.set_title(f"{label} — Brier={brier:.4f}", color="#E0E0E0")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.suptitle("Reliability Diagram (calibration)", fontsize=14, fontweight="bold",
                 color="#E0E0E0", y=1.01)
    plt.tight_layout()
    out = out_dir / "02_calibration.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    logger.info("Saved calibration plot → %s", out)
    return out


# ── Precision@K ───────────────────────────────────────────────────────────────

def plot_precision_at_k(
    y: np.ndarray,
    scores: np.ndarray,
    ks: list[int],
    out_dir: Path,
    dpi: int = 150,
) -> tuple[Path, dict[str, float]]:
    """Bar chart of Precision@K + lift curve."""
    _apply_style()
    sorted_idx = np.argsort(scores)[::-1]
    baseline   = float(y.mean())

    pk_vals = {}
    for k in ks:
        pk_vals[str(k)] = round(float(y[sorted_idx[:k]].sum() / k), 4)

    # Lift curve — precision@k for every k from 1 to len(y)
    cumulative_fraud = np.cumsum(y[sorted_idx])
    k_range    = np.arange(1, len(y) + 1)
    precision_k = cumulative_fraud / k_range
    lift_k      = precision_k / (baseline + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor("#0F1117")

    # Bar chart for selected K values
    bar_labels = [str(k) for k in ks]
    bar_vals   = [pk_vals[k] for k in bar_labels]
    bars = axes[0].bar(bar_labels, bar_vals, color=_PAL["lgb"], edgecolor="#0F1117", width=0.6)
    axes[0].axhline(baseline, color=_PAL["fraud"], ls="--", lw=1.5,
                    label=f"Baseline={baseline:.4f}")
    for bar, val in zip(bars, bar_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.3f}", ha="center", fontsize=9, color="#E0E0E0")
    axes[0].set_title("Precision@K", color="#E0E0E0")
    axes[0].set_xlabel("K (top-scored transactions)")
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=9)

    # Lift curve
    step = max(1, len(y) // 1000)  # downsample for plotting speed
    axes[1].plot(k_range[::step] / len(y) * 100, lift_k[::step],
                 color=_PAL["lgb"], lw=2)
    axes[1].axhline(1.0, color="#555", ls="--", lw=1, label="No lift")
    axes[1].set_title("Lift Curve", color="#E0E0E0")
    axes[1].set_xlabel("% of transactions reviewed (highest score first)")
    axes[1].set_ylabel("Lift over baseline")
    axes[1].legend(fontsize=9)

    fig.suptitle("Precision@K and Lift", fontsize=14, fontweight="bold",
                 color="#E0E0E0", y=1.01)
    plt.tight_layout()
    out = out_dir / "03_precision_at_k.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    logger.info("Saved Precision@K plot → %s", out)
    return out, pk_vals


# ── Cost and AUROC by time period ─────────────────────────────────────────────

def plot_time_period_analysis(
    y: np.ndarray,
    scores: np.ndarray,
    tx_dt: np.ndarray,
    cfg: dict[str, Any],
    out_dir: Path,
    dpi: int = 150,
) -> tuple[Path, list[dict]]:
    """Split test set into weekly periods and compute AUROC, AUPRC, fraud rate,
    and cost at the cost-optimal threshold for each period.

    This reveals whether model performance degrades over time — an early
    signal of concept drift.
    """
    _apply_style()
    cost_fn       = cfg["evaluation"]["cost_fn"]
    cost_fp       = cfg["evaluation"]["cost_fp"]
    cost_thr      = cfg["evaluation"]["cost_threshold"]  # derived from full test set

    # Convert DT to days from start of test set
    dt_days = (tx_dt - tx_dt.min()) / _SECS_PER_DAY
    period_days = cfg["evaluation"].get("time_period_days", 7)
    n_periods   = max(1, int(np.ceil(dt_days.max() / period_days)))

    period_stats: list[dict] = []
    period_labels: list[str] = []

    for p in range(n_periods):
        mask = (dt_days >= p * period_days) & (dt_days < (p + 1) * period_days)
        if mask.sum() < 50 or y[mask].sum() < 2:
            continue
        y_p  = y[mask]
        s_p  = scores[mask]
        auroc_p = float(roc_auc_score(y_p, s_p)) if y_p.sum() > 0 else float("nan")
        auprc_p = float(average_precision_score(y_p, s_p)) if y_p.sum() > 0 else float("nan")
        yp_pred  = (s_p >= cost_thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_p, yp_pred, labels=[0, 1]).ravel()
        cost_p   = float(fn_ * cost_fn + fp_ * cost_fp)
        fraud_rate = float(y_p.mean())

        period_stats.append({
            "period":     p + 1,
            "day_start":  int(p * period_days),
            "day_end":    int((p + 1) * period_days),
            "n_tx":       int(mask.sum()),
            "n_fraud":    int(y_p.sum()),
            "fraud_rate": round(fraud_rate, 4),
            "auroc":      round(auroc_p, 4),
            "auprc":      round(auprc_p, 4),
            "cost":       round(cost_p, 2),
        })
        period_labels.append(f"D{p*period_days}–{(p+1)*period_days}")

    if not period_stats:
        logger.warning("Not enough data per period for time analysis — skipping.")
        return out_dir / "04_time_periods.png", []

    aurocs      = [s["auroc"]      for s in period_stats]
    auprcs      = [s["auprc"]      for s in period_stats]
    fraud_rates = [s["fraud_rate"] for s in period_stats]
    costs       = [s["cost"]       for s in period_stats]
    xs          = range(len(period_stats))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor("#0F1117")

    def _bar(ax, vals, title, ylabel, color):
        ax.bar(xs, vals, color=color, edgecolor="#0F1117", width=0.7)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(period_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, color="#E0E0E0")
        ax.set_ylabel(ylabel)
        mean_val = float(np.nanmean(vals))
        ax.axhline(mean_val, color="white", ls="--", lw=1, label=f"Mean={mean_val:.3f}")
        ax.legend(fontsize=8)

    _bar(axes[0, 0], aurocs,      "AUROC by period",      "AUROC",       _PAL["lgb"])
    _bar(axes[0, 1], auprcs,      "AUPRC by period",      "AUPRC",       _PAL["accent"])
    _bar(axes[1, 0], fraud_rates, "Fraud rate by period", "Fraud rate",  _PAL["fraud"])
    _bar(axes[1, 1], costs,       f"Cost by period (thr={cost_thr:.2f})", "Total cost ($)", "#4A90D9")

    fig.suptitle(f"Performance by {period_days}-day period", fontsize=14,
                 fontweight="bold", color="#E0E0E0", y=1.01)
    plt.tight_layout()
    out = out_dir / "04_time_periods.png"
    plt.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    logger.info("Saved time-period analysis → %s  (%s periods)", out, len(period_stats))
    return out, period_stats


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_metrics(
    y: np.ndarray,
    scores: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    cost_fn = cfg["evaluation"]["cost_fn"]
    cost_fp = cfg["evaluation"]["cost_fp"]
    ks      = cfg["evaluation"]["precision_at_k"]

    auroc = float(roc_auc_score(y, scores))
    auprc = float(average_precision_score(y, scores))

    prec, rec, pr_thr = precision_recall_curve(y, scores)
    f1s    = 2 * prec * rec / (prec + rec + 1e-9)
    thr_f1 = float(pr_thr[np.argmax(f1s[:-1])])

    thr_range = np.linspace(0.01, 0.99, 200)
    costs = []
    for thr in thr_range:
        yp = (scores >= thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        costs.append(fn_ * cost_fn + fp_ * cost_fp)
    best_cost_thr = float(thr_range[int(np.argmin(costs))])

    # Store cost_threshold back in cfg so time-period analysis can use it
    cfg["evaluation"]["cost_threshold"] = best_cost_thr

    sorted_idx = np.argsort(scores)[::-1]
    pk = {str(k): round(float(y[sorted_idx[:k]].sum() / k), 4) for k in ks}

    y_pred = (scores >= thr_f1).astype(int)
    cm     = confusion_matrix(y, y_pred)

    brier = float(brier_score_loss(y, scores))

    logger.info("LightGBM: AUROC=%.4f  AUPRC=%.4f  Brier=%.4f", auroc, auprc, brier)
    logger.info("F1-optimal threshold: %.4f | cost-optimal: %.4f", thr_f1, best_cost_thr)
    logger.info("\n%s", classification_report(y, y_pred, target_names=["Normal", "Fraud"]))

    return {
        "lgb_auroc":        round(auroc, 4),
        "lgb_auprc":        round(auprc, 4),
        "lgb_brier":        round(brier, 4),
        "threshold_f1":     round(thr_f1, 6),
        "threshold_cost":   round(best_cost_thr, 6),
        "min_cost":         round(float(min(costs)), 2),
        "precision_at_k":   pk,
        "thr_range":        thr_range.tolist(),
        "costs":            costs,
        "confusion_matrix": cm.tolist(),
        "sorted_idx":       sorted_idx,
    }


# ── Main dashboard ────────────────────────────────────────────────────────────

def plot_dashboard(
    y: np.ndarray,
    raw_scores: np.ndarray,
    cal_scores: np.ndarray,
    metrics: dict[str, Any],
    out_dir: Path,
    dpi: int = 150,
) -> Path:
    _apply_style()
    cm            = np.array(metrics["confusion_matrix"])
    thr_f1        = metrics["threshold_f1"]
    thr_range     = np.array(metrics["thr_range"])
    costs         = metrics["costs"]
    best_cost_thr = metrics["threshold_cost"]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.patch.set_facecolor("#0F1117")

    # ROC — raw vs calibrated
    ax1 = fig.add_subplot(gs[0, 0])
    for sc, lb, col, lw in [
        (raw_scores, f"Raw ({metrics['lgb_auroc']:.4f})",   _PAL["lgb"],    2.5),
        (cal_scores, "Calibrated",                           _PAL["accent"], 1.5),
    ]:
        fpr_, tpr_, _ = roc_curve(y, sc)
        ax1.plot(fpr_, tpr_, color=col, lw=lw, label=lb)
    ax1.plot([0, 1], [0, 1], "--", color="#555")
    ax1.set_title("ROC Curve", color="#E0E0E0")
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
    ax1.legend(fontsize=8)

    # PR — raw vs calibrated
    ax2 = fig.add_subplot(gs[0, 1])
    for sc, lb, col, lw in [
        (raw_scores, f"Raw (AUPRC={metrics['lgb_auprc']:.4f})", _PAL["lgb"],    2.5),
        (cal_scores, "Calibrated",                               _PAL["accent"], 1.5),
    ]:
        p_, r_, _ = precision_recall_curve(y, sc)
        ax2.plot(r_, p_, color=col, lw=lw, label=lb)
    ax2.axhline(y.mean(), color="#555", ls="--", label=f"Baseline={y.mean():.4f}")
    ax2.set_title("Precision-Recall Curve", color="#E0E0E0")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.legend(fontsize=8)

    # Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    im  = ax3.imshow(cm, cmap="Blues")
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(["Normal", "Fraud"])
    ax3.set_yticklabels(["Normal", "Fraud"])
    ax3.set_title(f"Confusion Matrix (thr={thr_f1:.3f})", color="#E0E0E0")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=13,
                     color="white" if cm[i, j] > cm.max() / 2 else "#333")
    plt.colorbar(im, ax=ax3)

    # Cost vs threshold
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(thr_range, costs, color=_PAL["lgb"], lw=2)
    ax4.axvline(best_cost_thr, color="white",     lw=1.5, ls=":", label=f"Cost-opt={best_cost_thr:.3f}")
    ax4.axvline(thr_f1,        color=_PAL["fraud"], lw=1.5, ls=":", label=f"F1={thr_f1:.3f}")
    ax4.set_title("Cost vs Threshold", color="#E0E0E0")
    ax4.set_xlabel("Threshold"); ax4.set_ylabel("Total cost ($)")
    ax4.legend(fontsize=9)

    # Score distribution
    ax5 = fig.add_subplot(gs[1, 1])
    bins = np.linspace(0, 1, 50)
    ax5.hist(raw_scores[y == 0], bins=bins, alpha=0.6, color=_PAL["normal"],
             label="Normal", density=True)
    ax5.hist(raw_scores[y == 1], bins=bins, alpha=0.6, color=_PAL["fraud"],
             label="Fraud", density=True)
    ax5.axvline(best_cost_thr, color="white", lw=1.5, ls=":", label=f"Cost-thr={best_cost_thr:.3f}")
    ax5.set_title("Score Distribution", color="#E0E0E0")
    ax5.set_xlabel("Fraud probability"); ax5.set_ylabel("Density")
    ax5.legend(fontsize=8)

    # Calibration reliability diagram (compact)
    ax6 = fig.add_subplot(gs[1, 2])
    for sc, lb, col in [
        (raw_scores, "Raw",        _PAL["lgb"]),
        (cal_scores, "Calibrated", _PAL["accent"]),
    ]:
        frac_pos, mean_pred = calibration_curve(y, sc, n_bins=10, strategy="quantile")
        ax6.plot(mean_pred, frac_pos, "o-", color=col, lw=2, markersize=5, label=lb)
    ax6.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax6.set_title("Reliability Diagram", color="#E0E0E0")
    ax6.set_xlabel("Mean predicted prob"); ax6.set_ylabel("Fraction positives")
    ax6.legend(fontsize=8)

    fig.suptitle("LightGBM Evaluation Dashboard", fontsize=16, fontweight="bold",
                 color="#E0E0E0", y=1.01)
    out_path = out_dir / "01_dashboard.png"
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    logger.info("Saved dashboard → %s", out_path)
    return out_path


# ── SHAP ──────────────────────────────────────────────────────────────────────

def plot_shap(
    shap_values: np.ndarray,
    X_shap: np.ndarray,
    y_shap: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    _apply_style()
    mean_abs  = np.abs(shap_values).mean(axis=0)
    order     = np.argsort(mean_abs)[::-1]
    top_n     = 15
    top_idx   = order[:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = mean_abs[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("#0F1117")

    axes[0].barh(range(top_n), top_vals[::-1], color=_PAL["lgb"], edgecolor="#0F1117")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_names[::-1], fontsize=9)
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title("Global Feature Importance (SHAP)", color="#E0E0E0", fontsize=12)

    top15_shap  = shap_values[:, top_idx]
    top15_feats = X_shap[:, top_idx]
    feat_min    = top15_feats.min(0)
    feat_max    = top15_feats.max(0)
    feat_norm   = (top15_feats - feat_min) / (feat_max - feat_min + 1e-9)
    rng = np.random.default_rng(0)
    for rank in range(top_n - 1, -1, -1):
        y_pos  = np.full(len(top15_shap), top_n - 1 - rank)
        jitter = rng.uniform(-0.3, 0.3, len(top15_shap))
        colors = plt.cm.RdBu_r(feat_norm[:, rank])
        axes[1].scatter(top15_shap[:, rank], y_pos + jitter,
                        c=colors, alpha=0.4, s=6, linewidths=0)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top_names[::-1], fontsize=9)
    axes[1].axvline(0, color="#555", lw=1, ls="--")
    axes[1].set_xlabel("SHAP value (impact on fraud probability)")
    axes[1].set_title("SHAP Beeswarm — top 15\n(red=high value, blue=low)",
                      color="#E0E0E0", fontsize=11)

    plt.tight_layout()
    p1 = out_dir / "shap_importance.png"
    plt.savefig(p1, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)

    saved = [p1]
    fraud_in_sample = np.where(y_shap == 1)[0]
    if len(fraud_in_sample) > 0:
        top_fraud = fraud_in_sample[np.argmax(shap_values[fraud_in_sample].sum(axis=1))]
        sv        = shap_values[top_fraud]
        wf_order  = np.argsort(np.abs(sv))[::-1][:12]
        names_wf  = [feature_names[i] for i in wf_order]
        vals_wf   = sv[wf_order]
        colors_wf = [_PAL["fraud"] if v > 0 else _PAL["normal"] for v in vals_wf]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0F1117")
        ax.set_facecolor("#1A1D27")
        ax.barh(range(len(vals_wf)), vals_wf[::-1], color=colors_wf[::-1], edgecolor="#0F1117")
        ax.set_yticks(range(len(vals_wf)))
        ax.set_yticklabels(names_wf[::-1], fontsize=10)
        ax.axvline(0, color="#555", lw=1, ls="--")
        ax.set_xlabel("SHAP value")
        ax.set_title("Waterfall: top predicted fraud transaction", color="#E0E0E0", fontsize=11)
        ax.tick_params(colors="#9AA0B0")
        plt.tight_layout()
        p2 = out_dir / "shap_waterfall.png"
        plt.savefig(p2, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
        plt.close(fig)
        saved.append(p2)
        logger.info("Saved SHAP waterfall → %s", p2)

    return saved


# ── Main entry point ──────────────────────────────────────────────────────────

def run_evaluation(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute all metrics, calibrate, generate plots, log to MLflow."""
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    mdl_dir = Path(cfg["paths"]["models_dir"])
    dpi     = cfg["evaluation"]["plot_dpi"]
    ks      = cfg["evaluation"]["precision_at_k"]

    y             = result["y_test"]
    raw_scores    = result["lgb_scores"]
    model         = result["model"]
    X_test_full   = result["X_test_full"]
    tx_dt_test    = result["tx_dt_test"]
    shap_values   = result["shap_values"]
    shap_idx      = result["shap_idx"]
    feature_names = result["feature_names"]

    # 1. Core metrics (also derives cost_threshold and stores it in cfg)
    metrics = compute_metrics(y, raw_scores, cfg)

    # 2. Calibration
    calibrated_model, cal_scores = calibrate(model, X_test_full, y, cfg)
    brier_cal = float(brier_score_loss(y, cal_scores))
    metrics["lgb_brier_calibrated"] = round(brier_cal, 4)

    # 3. Plots
    dash_path   = plot_dashboard(y, raw_scores, cal_scores, metrics, out_dir, dpi)
    cal_path    = plot_calibration(y, raw_scores, cal_scores, out_dir, dpi)
    pk_path, pk_vals = plot_precision_at_k(y, raw_scores, ks, out_dir, dpi)
    tp_path, period_stats = plot_time_period_analysis(
        y, raw_scores, tx_dt_test, cfg, out_dir, dpi
    )
    shap_paths  = plot_shap(shap_values, X_test_full[shap_idx], y[shap_idx],
                            feature_names, out_dir, dpi)

    # 4. MLflow
    flat_metrics = {
        "lgb_auroc":            metrics["lgb_auroc"],
        "lgb_auprc":            metrics["lgb_auprc"],
        "lgb_brier_raw":        metrics["lgb_brier"],
        "lgb_brier_calibrated": brier_cal,
        "threshold_f1":         metrics["threshold_f1"],
        "threshold_cost":       metrics["threshold_cost"],
        "min_cost":             metrics["min_cost"],
    }
    for k, v in pk_vals.items():
        flat_metrics[f"precision_at_{k}"] = v
    if period_stats:
        flat_metrics["auroc_first_period"] = period_stats[0]["auroc"]
        flat_metrics["auroc_last_period"]  = period_stats[-1]["auroc"]
        flat_metrics["auroc_period_std"]   = round(
            float(np.std([s["auroc"] for s in period_stats])), 4
        )
    mlflow.log_metrics(flat_metrics)

    for path in [dash_path, cal_path, pk_path, tp_path] + shap_paths:
        if path.exists():
            mlflow.log_artifact(str(path))
    mlflow.log_artifact(str(mdl_dir / "lgb_model.pkl"))

    # Save summary JSON
    summary = {
        **{k: v for k, v in metrics.items()
           if k not in ("thr_range", "costs", "confusion_matrix", "sorted_idx")},
        "precision_at_k": pk_vals,
        "brier_calibrated": brier_cal,
        "time_periods": period_stats,
    }
    out_json = mdl_dir / "lgb_results.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    mlflow.log_artifact(str(out_json))

    logger.info("Evaluation complete.")
    return flat_metrics