from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from ...core.plotting import new_figure
from .stats import difference_stats, sample_posteriors


def plot_posteriors(posterior_a, posterior_b, *, num_samples: int = 10000) -> Figure:
    samples_a, samples_b = sample_posteriors(posterior_a, posterior_b, num_samples=num_samples)
    fig, ax = new_figure(figsize=(12, 6))
    bins = max(20, int(np.sqrt(num_samples)))
    sns.histplot(
        samples_a,
        color="tab:blue",
        label="Group A",
        kde=True,
        stat="density",
        bins=bins,
        ax=ax,
        alpha=0.45,
    )
    sns.histplot(
        samples_b,
        color="tab:orange",
        label="Group B",
        kde=True,
        stat="density",
        bins=bins,
        ax=ax,
        alpha=0.45,
    )
    ax.set_title("Posterior distributions (A vs B)")
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Density")
    ax.legend()
    return fig


def plot_boxplot(
    posterior_a, posterior_b, *, n_a: int, n_b: int, num_samples: int | None = None
) -> Figure:
    if num_samples is None:
        samples_a = posterior_a.rvs(n_a)
        samples_b = posterior_b.rvs(n_b)
    else:
        samples_a, samples_b = sample_posteriors(posterior_a, posterior_b, num_samples=num_samples)

    data = pd.DataFrame(
        {
            "Group": ["A"] * len(samples_a) + ["B"] * len(samples_b),
            "Value": np.concatenate([samples_a, samples_b]),
        }
    )
    fig, ax = new_figure(figsize=(8, 6))
    sns.boxplot(
        x="Group",
        y="Value",
        data=data,
        ax=ax,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
    )
    ax.set_title("Posterior boxplot")
    return fig


def plot_kde(posterior_a, posterior_b, *, num_samples: int = 10000) -> Figure:
    samples_a, samples_b = sample_posteriors(posterior_a, posterior_b, num_samples=num_samples)
    fig, ax = new_figure(figsize=(12, 6))
    sns.kdeplot(samples_a, label="Group A", fill=True, color="blue", ax=ax)
    sns.kdeplot(samples_b, label="Group B", fill=True, color="orange", ax=ax)
    ax.set_xlabel("Conversion Rate")
    ax.set_ylabel("Density")
    ax.set_title("Posterior KDE (A vs B)")
    ax.legend()
    return fig


def plot_difference_hist_kde(posterior_a, posterior_b, *, num_samples: int = 10000) -> Figure:
    diff = difference_stats(posterior_a, posterior_b, num_samples=num_samples)
    delta = diff["delta"]
    fig, ax = new_figure(figsize=(12, 6))
    vmin = float(np.min(delta))
    vmax = float(np.max(delta))
    if np.isclose(vmin, vmax):
        ax.axvline(vmin, color="purple", linewidth=2, label="Posterior diff")
        ax.text(0.01, 0.95, "Degenerate distribution", transform=ax.transAxes, va="top")
    else:
        uniq = int(np.unique(delta).size)
        bins = max(10, min(60, uniq))
        sns.histplot(delta, bins=bins, kde=True, color="purple", ax=ax, alpha=0.65)
    ax.axvline(0, color="red", linestyle="dashed")
    ax.set_xlabel("Difference in Conversion Rate (B - A)")
    ax.set_ylabel("Density")
    ax.set_title("Posterior difference (B - A)")
    return fig


def plot_difference_cdf(posterior_a, posterior_b, *, num_samples: int = 10000) -> Figure:
    diff = difference_stats(posterior_a, posterior_b, num_samples=num_samples)
    delta = diff["delta"]
    fig, ax = new_figure(figsize=(12, 6))
    sns.ecdfplot(delta, color="darkgreen", ax=ax)
    ax.axvline(0, color="red", linestyle="dashed")
    ax.set_xlabel("Difference in Conversion Rate (B - A)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Posterior difference CDF")
    return fig
