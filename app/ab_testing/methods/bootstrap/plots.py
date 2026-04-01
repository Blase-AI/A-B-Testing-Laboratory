from __future__ import annotations

from typing import Optional

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from ...core.plotting import new_figure
from .stats import compute_bootstrap_diff, compute_bootstrap_means


def plot_bootstrap_distributions(
    group_a: np.ndarray, group_b: np.ndarray, *, n_bootstrap: int = 10000, random_state: Optional[int] = 42
) -> Figure:
    means_a, means_b = compute_bootstrap_means(group_a, group_b, n_bootstrap=n_bootstrap, random_state=random_state)
    fig, ax = new_figure(figsize=(12, 6))
    sns.kdeplot(means_a, label="Bootstrapped means A", fill=True, color="tab:blue", ax=ax)
    sns.kdeplot(means_b, label="Bootstrapped means B", fill=True, color="tab:orange", ax=ax)
    ax.set_xlabel("Metric Mean")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap mean distributions")
    ax.legend()
    return fig


def plot_bootstrap_diff_hist(
    group_a: np.ndarray, group_b: np.ndarray, *, n_bootstrap: int = 10000, random_state: Optional[int] = 42
) -> Figure:
    boot_diff = compute_bootstrap_diff(group_a, group_b, n_bootstrap=n_bootstrap, random_state=random_state)
    fig, ax = new_figure(figsize=(12, 6))
    vmin = float(np.min(boot_diff))
    vmax = float(np.max(boot_diff))
    if np.isclose(vmin, vmax):
        ax.axvline(vmin, color="purple", linewidth=2, label="Bootstrap diff")
        ax.set_ylabel("Density")
        ax.set_title("Bootstrap difference distribution (B - A)")
        ax.text(0.01, 0.95, "Degenerate distribution", transform=ax.transAxes, va="top")
    else:
        uniq = int(np.unique(boot_diff).size)
        bins = max(10, min(60, uniq))
        sns.histplot(boot_diff, bins=bins, kde=True, color="purple", ax=ax, alpha=0.65)
    ax.axvline(0, color="red", linestyle="dashed", label="No difference")
    ax.set_xlabel("Difference in Means (B - A)")
    ax.legend()
    return fig


def plot_bootstrap_diff_ecdf(
    group_a: np.ndarray, group_b: np.ndarray, *, n_bootstrap: int = 10000, random_state: Optional[int] = 42
) -> Figure:
    boot_diff = compute_bootstrap_diff(group_a, group_b, n_bootstrap=n_bootstrap, random_state=random_state)
    fig, ax = new_figure(figsize=(12, 6))
    sns.ecdfplot(boot_diff, color="darkgreen", ax=ax)
    ax.axvline(0, color="red", linestyle="dashed")
    ax.set_xlabel("Difference in Means (B - A)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Bootstrap difference ECDF (B - A)")
    return fig

