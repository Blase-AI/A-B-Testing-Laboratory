from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from matplotlib.figure import Figure

from ...core.plotting import new_figure, new_figure_grid
from .stats import mean_difference_ci


def plot_distribution(data_a: np.ndarray, data_b: np.ndarray) -> Figure:
    fig, ax = new_figure(figsize=(12, 6))
    bins = max(10, int(np.sqrt(len(data_a) + len(data_b))))
    sns.histplot(data_a, color="tab:blue", label="Group A", kde=True, stat="density", bins=bins, ax=ax, alpha=0.45)
    sns.histplot(data_b, color="tab:orange", label="Group B", kde=True, stat="density", bins=bins, ax=ax, alpha=0.45)
    ax.set_title("Metric distribution (A vs B)")
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def plot_boxplot(data_a: np.ndarray, data_b: np.ndarray) -> Figure:
    df = pd.DataFrame({"Group": ["A"] * len(data_a) + ["B"] * len(data_b), "Value": np.concatenate([data_a, data_b])})
    fig, ax = new_figure(figsize=(8, 6))
    sns.boxplot(
        x="Group",
        y="Value",
        data=df,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    sns.stripplot(x="Group", y="Value", data=df, ax=ax, color="0.2", alpha=0.25, jitter=0.2, size=2)
    ax.set_title("Boxplot (with points)")
    ax.set_xlabel("Group")
    ax.set_ylabel("Metric Value")
    return fig


def plot_mean_difference_ci(data_a: np.ndarray, data_b: np.ndarray, *, confidence: float = 0.95) -> Figure:
    diff, (ci_low, ci_high) = mean_difference_ci(data_a, data_b, confidence=confidence)
    fig, ax = new_figure(figsize=(8, 6))
    ax.errorbar(0, diff, yerr=[[diff - ci_low], [ci_high - diff]], fmt="o", color="blue", capsize=10, markersize=8)
    ax.axhline(0, color="red", linestyle="--", label="There is no difference")
    ax.set_xticks([])
    ax.set_ylabel("Difference (B - A)")
    ax.set_title(f"Difference in Means with {int(confidence*100)}% Confidence Interval")
    ax.legend()
    return fig


def plot_qq(data_a: np.ndarray, data_b: np.ndarray) -> Figure:
    fig, ax = new_figure_grid(1, 2, figsize=(14, 6))

    (osm, _osr), (slope, intercept, r) = stats.probplot(data_a, dist="norm", plot=ax[0])
    ax[0].plot(osm, slope * osm + intercept, color="red", linestyle="--", label=f"R² = {r**2:.2f}\nSlope = {slope:.2f}")
    ax[0].set_title("Group A", fontsize=12, pad=10)
    ax[0].set_xlabel("Theoretical Quantiles", fontsize=10)
    ax[0].set_ylabel("Sample Quantiles", fontsize=10)
    ax[0].legend(loc="upper left")

    (osm, _osr), (slope, intercept, r) = stats.probplot(data_b, dist="norm", plot=ax[1])
    ax[1].plot(osm, slope * osm + intercept, color="red", linestyle="--", label=f"R² = {r**2:.2f}\nSlope = {slope:.2f}")
    ax[1].set_title("Group B", fontsize=12, pad=10)
    ax[1].set_xlabel("Theoretical Quantiles", fontsize=10)
    ax[1].set_ylabel("Sample Quantiles")
    ax[1].legend(loc="upper left")

    fig.suptitle("Q-Q Plots for Normality Check", y=1.02, fontsize=14, weight="bold")
    fig.tight_layout()
    return fig

