from __future__ import annotations

from typing import Any, Dict, Tuple

import arviz as az
import numpy as np
import scipy.stats as stats


def make_posteriors(
    n_a: int,
    successes_a: int,
    n_b: int,
    successes_b: int,
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> Tuple[stats.rv_continuous, stats.rv_continuous]:
    posterior_a = stats.beta(alpha_prior + successes_a, beta_prior + (n_a - successes_a))
    posterior_b = stats.beta(alpha_prior + successes_b, beta_prior + (n_b - successes_b))
    return posterior_a, posterior_b


def sample_posteriors(posterior_a, posterior_b, *, num_samples: int = 10000):
    return posterior_a.rvs(num_samples), posterior_b.rvs(num_samples)


def prob_b_better(posterior_a, posterior_b, *, num_samples: int = 10000) -> float:
    a, b = sample_posteriors(posterior_a, posterior_b, num_samples=num_samples)
    return float(np.mean(b > a))


def difference_stats(posterior_a, posterior_b, *, num_samples: int = 10000, hdi_prob: float = 0.95) -> Dict[str, Any]:
    a, b = sample_posteriors(posterior_a, posterior_b, num_samples=num_samples)
    delta = b - a
    hdi_interval = az.hdi(delta, hdi_prob=hdi_prob)
    return {"delta": delta, "hdi_interval": hdi_interval}

