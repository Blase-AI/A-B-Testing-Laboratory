from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from matplotlib.figure import Figure

from ...base import TwoGroupContinuousABTest
from ...core.power import perform_power_analysis
from . import plots, stats


class ClassicABTest(TwoGroupContinuousABTest):
    @property
    def name(self) -> str:
        return "classic"

    def summary(self) -> Dict[str, Any]:
        return {
            "mean_a": float(np.mean(self.data_a)),
            "mean_b": float(np.mean(self.data_b)),
            "diff": float(np.mean(self.data_b) - np.mean(self.data_a)),
            "cohens_d": stats.cohens_d(self.data_a, self.data_b),
        }

    def descriptive_stats(self):
        return stats.descriptive_stats(self.data_a, self.data_b)

    def test_normality(self) -> Dict[str, Any]:
        return stats.test_normality(self.data_a, self.data_b)

    def test_variance(self) -> Dict[str, Any]:
        return stats.test_variance(self.data_a, self.data_b)

    def perform_t_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        return stats.perform_t_or_u_test(self.data_a, self.data_b, alpha=alpha)

    def calculate_cohens_d(self) -> float:
        return stats.cohens_d(self.data_a, self.data_b)

    def plot_distribution(self) -> Figure:
        return plots.plot_distribution(self.data_a, self.data_b)

    def plot_boxplot(self) -> Figure:
        return plots.plot_boxplot(self.data_a, self.data_b)

    def plot_mean_difference_ci(self, confidence: float = 0.95) -> Figure:
        return plots.plot_mean_difference_ci(self.data_a, self.data_b, confidence=confidence)

    def plot_qq(self) -> Figure:
        return plots.plot_qq(self.data_a, self.data_b)

    @staticmethod
    def perform_power_analysis(
        effect_size: float,
        alpha: float = 0.05,
        nobs1: int = 100,
        desired_power: float = 0.8,
        sample_sizes: Optional[np.ndarray] = None,
    ):
        return perform_power_analysis(
            effect_size, alpha=alpha, nobs1=nobs1, desired_power=desired_power, sample_sizes=sample_sizes
        )

