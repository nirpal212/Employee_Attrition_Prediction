"""
Plot helpers for the Streamlit dashboard and the EDA notebook.

Every chart uses titles, axis labels, and a restrained navy/teal palette.
Figures are returned so Streamlit can display them with st.pyplot.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_loader import encode_target
import config

NAVY = "#1e3a5f"
TEAL = "#2a9d8f"
CORAL = "#c44536"
LIGHT = "#e8eef5"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_class_distribution(df: pd.DataFrame):
    _style()
    counts = df[config.TARGET_COLUMN].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    colors = [TEAL if str(idx).lower() in ("no", "0") else CORAL for idx in counts.index]
    counts.plot(kind="bar", ax=ax, color=colors, width=0.6)
    ax.set_title("Attrition class distribution")
    ax.set_xlabel("Attrition")
    ax.set_ylabel("Number of employees")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_attrition_rate_by(df: pd.DataFrame, column: str, title: str | None = None):
    _style()
    work = df.copy()
    work["_flag"] = encode_target(work[config.TARGET_COLUMN])
    rates = work.groupby(column)["_flag"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(rates))))
    rates.plot(kind="barh", ax=ax, color=NAVY)
    ax.set_xlabel("Attrition rate")
    ax.set_ylabel(column)
    ax.set_title(title or f"Attrition rate by {column}")
    ax.set_xlim(0, max(0.5, float(rates.max()) * 1.15))
    fig.tight_layout()
    return fig


def plot_years_at_company(df: pd.DataFrame):
    _style()
    if "YearsAtCompany" not in df.columns:
        return None
    work = df.copy()
    work["_flag"] = encode_target(work[config.TARGET_COLUMN])
    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.histplot(
        data=work,
        x="YearsAtCompany",
        hue=config.TARGET_COLUMN,
        bins=20,
        ax=ax,
        palette={config.POSITIVE_LABEL: CORAL, config.NEGATIVE_LABEL: TEAL, "Yes": CORAL, "No": TEAL},
        multiple="stack",
    )
    ax.set_title("Years at company by attrition status")
    ax.set_xlabel("Years at company")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame, columns: list[str]):
    _style()
    cols = [c for c in columns if c in df.columns][:6]
    if not cols:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6))
    axes = axes.ravel()
    for i, column in enumerate(cols):
        sns.histplot(df[column].dropna(), ax=axes[i], color=NAVY, bins=20)
        axes[i].set_title(column)
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Count")
    for j in range(len(cols), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Numerical feature distributions")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    _style()
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="Blues", center=0, linewidths=0.3)
    ax.set_title("Correlation heatmap (numerical features)")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int):
    _style()
    matrix = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred Stay (No)", "Pred Leave (Yes)"],
        yticklabels=["Actual Stay (No)", "Actual Leave (Yes)"],
        cbar=False,
    )
    ax.set_title("Confusion matrix (selected model, test set)")
    fig.tight_layout()
    return fig


def plot_feature_importance(table: pd.DataFrame, top_n: int = 15):
    _style()
    if table is None or table.empty:
        return None
    top = table.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top["feature"], top["importance"], color=TEAL)
    ax.set_title("Feature importance (selected model)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig
