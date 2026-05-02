"""Exploratory data analysis plots for the IEEE-CIS fraud dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_PAL = {
    "normal": "#4A90D9",
    "fraud": "#E84545",
    "accent": "#F5A623",
    "green": "#4CAF50",
    "purple": "#A259FF",
}

_DARK = {
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#2E3347",
    "text.color": "#E0E0E0",
    "axes.labelcolor": "#E0E0E0",
    "xtick.color": "#9AA0B0",
    "ytick.color": "#9AA0B0",
    "grid.color": "#2E3347",
    "grid.linestyle": "--",
    "font.family": "monospace",
    "legend.facecolor": "#1A1D27",
    "legend.edgecolor": "#2E3347",
}


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    plt.rcParams.update(_DARK)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="#0F1117")
    plt.close(fig)
    return path


def plot_class_distribution(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    plt.rcParams.update(_DARK)
    normal_count = int((df["Class"] == 0).sum())
    fraud_count = int(df["Class"].sum())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Class Distribution", fontsize=15, fontweight="bold", color="#E0E0E0")
    bars = axes[0].bar(["Normal", "Fraud"], [normal_count, fraud_count],
                       color=[_PAL["normal"], _PAL["fraud"]], edgecolor="#0F1117", width=0.5)
    for bar, val in zip(bars, [normal_count, fraud_count]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                     f"{val:,}", ha="center", fontsize=11, color="#E0E0E0")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("Transaction counts", color="#E0E0E0")
    axes[1].pie([normal_count, fraud_count], labels=["Normal", "Fraud"],
                colors=[_PAL["normal"], _PAL["fraud"]], autopct="%1.3f%%",
                textprops={"color": "#E0E0E0"},
                wedgeprops={"edgecolor": "#0F1117", "linewidth": 2}, startangle=90)
    axes[1].set_title("Class proportion", color="#E0E0E0")
    plt.tight_layout()
    return _savefig(fig, out_dir / "01_class_distribution.png", dpi)


def plot_amount_analysis(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    plt.rcParams.update(_DARK)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Transaction Amount Analysis", fontsize=14, fontweight="bold", color="#E0E0E0")

    for cls, lbl, col in [(0, "Normal", _PAL["normal"]), (1, "Fraud", _PAL["fraud"])]:
        vals = np.log1p(df[df["Class"] == cls]["Amount"].dropna())
        axes[0].hist(vals, bins=80, alpha=0.6, color=col, label=lbl, density=True)
    axes[0].set_xlabel("log(1 + Amount)")
    axes[0].set_title("Log-amount distribution", color="#E0E0E0")
    axes[0].legend()

    norm_amt = np.log1p(df[df["Class"] == 0]["Amount"].dropna().values)
    fraud_amt = np.log1p(df[df["Class"] == 1]["Amount"].dropna().values)
    bp = axes[1].boxplot([norm_amt, fraud_amt], labels=["Normal", "Fraud"],
                         patch_artist=True, notch=True, medianprops={"color": "white", "lw": 2})
    for patch, col in zip(bp["boxes"], [_PAL["normal"], _PAL["fraud"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps", "fliers"]:
        for line in bp[element]:
            line.set_color("#9AA0B0")
    axes[1].set_ylabel("log(1 + Amount)")
    axes[1].set_title("Amount distribution by class", color="#E0E0E0")

    df2 = df.copy()
    df2["amount_bracket"] = pd.cut(df2["Amount"],
        bins=[0, 10, 50, 100, 500, 1000, df2["Amount"].max() + 1],
        labels=["$0-10", "$10-50", "$50-100", "$100-500", "$500-1K", "$1K+"])
    bracket_fraud = df2.groupby("amount_bracket", observed=True)["Class"].agg(["sum", "count"])
    bracket_fraud["rate"] = bracket_fraud["sum"] / bracket_fraud["count"] * 100
    bracket_fraud["rate"].plot(kind="bar", ax=axes[2], color=_PAL["fraud"], edgecolor="#0F1117", rot=30)
    axes[2].set_ylabel("Fraud rate (%)")
    axes[2].set_title("Fraud rate by amount bracket", color="#E0E0E0")

    plt.tight_layout()
    return _savefig(fig, out_dir / "02_amount_analysis.png", dpi)


def plot_ks_features(df: pd.DataFrame, out_dir: Path, dpi: int) -> tuple[Path, pd.DataFrame]:
    plt.rcParams.update(_DARK)
    v_features = [c for c in df.columns if c.startswith("V")][:28]
    ks_results = []
    for feat in v_features:
        a = df.loc[df["Class"] == 0, feat].dropna()
        b = df.loc[df["Class"] == 1, feat].dropna()
        if len(a) > 10 and len(b) > 10:
            stat, pval = stats.ks_2samp(a, b)
            ks_results.append({"feature": feat, "ks_stat": stat, "p_value": pval,
                                "normal_mean": a.mean(), "fraud_mean": b.mean()})
    ks_df = pd.DataFrame(ks_results).sort_values("ks_stat", ascending=False)
    top6 = ks_df.head(6)["feature"].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Top 6 V-Features: Normal vs Fraud", fontsize=14, fontweight="bold", color="#E0E0E0")
    for ax, feat in zip(axes.flat, top6):
        ks_val = ks_df.loc[ks_df["feature"] == feat, "ks_stat"].values[0]
        for cls, label, color in [(0, "Normal", _PAL["normal"]), (1, "Fraud", _PAL["fraud"])]:
            vals = df[df["Class"] == cls][feat].dropna()
            vals = vals.clip(vals.quantile(0.01), vals.quantile(0.99))
            ax.hist(vals, bins=60, alpha=0.6, color=color, label=label, density=True)
        ax.set_title(f"{feat}  (KS={ks_val:.3f})", color="#E0E0E0", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return _savefig(fig, out_dir / "03_ks_features.png", dpi), ks_df


def run_eda(df: pd.DataFrame, cfg: dict[str, Any]) -> list[Path]:
    """Run all EDA plots and log to MLflow."""
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = cfg["evaluation"]["plot_dpi"]

    paths: list[Path] = []
    paths.append(plot_class_distribution(df, out_dir, dpi))
    paths.append(plot_amount_analysis(df, out_dir, dpi))
    p, ks_df = plot_ks_features(df, out_dir, dpi)
    paths.append(p)

    for path in paths:
        mlflow.log_artifact(str(path))

    logger.info("EDA complete — saved %s plots to %s", len(paths), out_dir)
    return paths
