from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure

from ...base import TwoGroupContinuousABTest
from ...core.power import perform_power_analysis
from . import plots, stats


class BootstrapABTest(TwoGroupContinuousABTest):
    @property
    def name(self) -> str:
        return "bootstrap"

    def summary(self) -> dict[str, Any]:
        return {
            "mean_a": float(np.mean(self.data_a)),
            "mean_b": float(np.mean(self.data_b)),
            "diff": float(np.mean(self.data_b) - np.mean(self.data_a)),
            "cohens_d": stats.cohens_d(self.data_a, self.data_b),
        }

    def compute_bootstrap_means(self, n_bootstrap: int = 10000, random_state: int | None = 42):
        return stats.compute_bootstrap_means(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

    def percentile_ci(self, boot_stats: np.ndarray, ci: float = 95):
        return stats.percentile_ci(boot_stats, ci=ci)

    def compute_bootstrap_diff(
        self, n_bootstrap: int = 10000, random_state: int | None = 42
    ) -> np.ndarray:
        return stats.compute_bootstrap_diff(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

    def compute_bootstrap_p_value(self, boot_diff: np.ndarray) -> float:
        return stats.compute_bootstrap_p_value(boot_diff)

    def get_bootstrap_statistics(
        self, n_bootstrap: int = 10000, random_state: int | None = 42, ci: float = 95
    ):
        return stats.get_bootstrap_statistics(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state, ci=ci
        )

    def plot_bootstrap_distributions(
        self, n_bootstrap: int = 10000, random_state: int | None = 42
    ) -> Figure:
        return plots.plot_bootstrap_distributions(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

    def plot_bootstrap_diff_hist(
        self, n_bootstrap: int = 10000, random_state: int | None = 42
    ) -> Figure:
        return plots.plot_bootstrap_diff_hist(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

    def plot_bootstrap_diff_ecdf(
        self, n_bootstrap: int = 10000, random_state: int | None = 42
    ) -> Figure:
        return plots.plot_bootstrap_diff_ecdf(
            self.data_a, self.data_b, n_bootstrap=n_bootstrap, random_state=random_state
        )

    @staticmethod
    def perform_power_analysis(
        effect_size: float,
        alpha: float = 0.05,
        nobs1: int = 100,
        desired_power: float = 0.8,
        sample_sizes: np.ndarray | None = None,
    ):
        return perform_power_analysis(
            effect_size,
            alpha=alpha,
            nobs1=nobs1,
            desired_power=desired_power,
            sample_sizes=sample_sizes,
        )
