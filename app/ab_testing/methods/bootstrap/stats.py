from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_statistic(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float] = np.mean,
    *,
    n_bootstrap: int = 10000,
    random_state: int | None = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    n = len(data)
    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(sample)
    return boot_stats


def compute_bootstrap_means(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        bootstrap_statistic(group_a, np.mean, n_bootstrap=n_bootstrap, random_state=random_state),
        bootstrap_statistic(group_b, np.mean, n_bootstrap=n_bootstrap, random_state=random_state),
    )


def percentile_ci(boot_stats: np.ndarray, *, ci: float = 95) -> tuple[float, float]:
    lower = np.percentile(boot_stats, (100 - ci) / 2)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
    return float(lower), float(upper)


def compute_bootstrap_diff(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    random_state: int | None = 42,
) -> np.ndarray:
    means_a, means_b = compute_bootstrap_means(
        group_a, group_b, n_bootstrap=n_bootstrap, random_state=random_state
    )
    return means_b - means_a


def compute_bootstrap_p_value(boot_diff: np.ndarray) -> float:
    left = float(np.mean(boot_diff <= 0))
    right = float(np.mean(boot_diff >= 0))
    return float(2 * min(left, right))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2)
    )
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def get_bootstrap_statistics(
    group_a: np.ndarray,
    group_b: np.ndarray,
    *,
    n_bootstrap: int = 10000,
    random_state: int | None = 42,
    ci: float = 95,
) -> dict[str, object]:
    means_a, means_b = compute_bootstrap_means(
        group_a, group_b, n_bootstrap=n_bootstrap, random_state=random_state
    )
    ci_a = percentile_ci(means_a, ci=ci)
    ci_b = percentile_ci(means_b, ci=ci)
    boot_diff = means_b - means_a
    ci_diff = percentile_ci(boot_diff, ci=ci)
    p_value = compute_bootstrap_p_value(boot_diff)
    effect_size = cohens_d(group_a, group_b)
    return {
        "boot_means_A": means_a,
        "boot_means_B": means_b,
        "ci_A": ci_a,
        "ci_B": ci_b,
        "boot_diff": boot_diff,
        "ci_diff": ci_diff,
        "p_value": p_value,
        "cohen_d": effect_size,
    }
