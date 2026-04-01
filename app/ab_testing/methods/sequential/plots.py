from __future__ import annotations

from matplotlib.figure import Figure

from ...core.plotting import new_figure
from .model import SPRTResult


def plot_sprt_history(result: SPRTResult) -> Figure:
    fig, ax = new_figure(figsize=(10, 6))
    ax.plot(range(1, len(result.history) + 1), result.history, marker="o", label="log(LR)")
    ax.axhline(result.upper_bound, color="green", linestyle="--", label="H1 boundary")
    ax.axhline(result.lower_bound, color="red", linestyle="--", label="H0 boundary")
    ax.set_xlabel("Observation number")
    ax.set_ylabel("Log-likelihood ratio")
    ax.set_title("SPRT Log-Likelihood Ratio Evolution")
    ax.legend()
    return fig

