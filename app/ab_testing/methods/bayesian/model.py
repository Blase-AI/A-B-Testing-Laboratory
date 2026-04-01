from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure

from ...base import BaseABTest
from ...core.power import perform_power_analysis
from . import plots, stats


class BayesianABTest(BaseABTest):
    def __init__(
        self,
        n_a: int,
        successes_a: int,
        n_b: int,
        successes_b: int,
        *,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
    ) -> None:
        if n_a <= 0 or n_b <= 0:
            raise ValueError("Number of observations must be positive.")
        if not (0 <= successes_a <= n_a):
            raise ValueError("Number of successes in group A must be between 0 and n_a.")
        if not (0 <= successes_b <= n_b):
            raise ValueError("Number of successes in group B must be between 0 and n_b.")

        self.n_a = int(n_a)
        self.successes_a = int(successes_a)
        self.n_b = int(n_b)
        self.successes_b = int(successes_b)
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)

        self.posterior_a, self.posterior_b = stats.make_posteriors(
            self.n_a,
            self.successes_a,
            self.n_b,
            self.successes_b,
            alpha_prior=self.alpha_prior,
            beta_prior=self.beta_prior,
        )

    @property
    def name(self) -> str:
        return "bayesian"

    def summary(self) -> dict[str, Any]:
        prob = self.compute_prob_B_better()
        return {"p_b_better": float(prob)}

    def sample_posteriors(self, num_samples: int = 10000):
        return stats.sample_posteriors(self.posterior_a, self.posterior_b, num_samples=num_samples)

    def compute_prob_B_better(self, num_samples: int = 10000) -> float:
        return stats.prob_b_better(self.posterior_a, self.posterior_b, num_samples=num_samples)

    def compute_difference_stats(self, num_samples: int = 10000, hdi_prob: float = 0.95):
        return stats.difference_stats(
            self.posterior_a, self.posterior_b, num_samples=num_samples, hdi_prob=hdi_prob
        )

    def plot_posteriors(self, num_samples: int = 10000) -> Figure:
        return plots.plot_posteriors(self.posterior_a, self.posterior_b, num_samples=num_samples)

    def plot_boxplot(self, num_samples: int | None = None) -> Figure:
        return plots.plot_boxplot(
            self.posterior_a, self.posterior_b, n_a=self.n_a, n_b=self.n_b, num_samples=num_samples
        )

    def plot_kde(self, num_samples: int = 10000) -> Figure:
        return plots.plot_kde(self.posterior_a, self.posterior_b, num_samples=num_samples)

    def plot_difference_hist_kde(self, num_samples: int = 10000) -> Figure:
        return plots.plot_difference_hist_kde(
            self.posterior_a, self.posterior_b, num_samples=num_samples
        )

    def plot_difference_cdf(self, num_samples: int = 10000) -> Figure:
        return plots.plot_difference_cdf(
            self.posterior_a, self.posterior_b, num_samples=num_samples
        )

    @staticmethod
    def perform_power_analysis(
        effect_size: float, alpha: float, nobs1: int, desired_power: float, sample_sizes=None
    ):
        return perform_power_analysis(
            effect_size,
            alpha=alpha,
            nobs1=nobs1,
            desired_power=desired_power,
            sample_sizes=sample_sizes,
        )
