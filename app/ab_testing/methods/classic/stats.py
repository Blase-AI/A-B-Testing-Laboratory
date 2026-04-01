from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats


def descriptive_stats(data_a: np.ndarray, data_b: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Group": ["A", "B"],
            "Mean": [float(np.mean(data_a)), float(np.mean(data_b))],
            "Std": [float(np.std(data_a, ddof=1)), float(np.std(data_b, ddof=1))],
        }
    )


def test_normality(data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, Dict[str, Any]]:
    shapiro_a = stats.shapiro(data_a)
    shapiro_b = stats.shapiro(data_b)
    return {
        "Group A": {"p_value": float(shapiro_a.pvalue), "normal": bool(shapiro_a.pvalue > 0.05)},
        "Group B": {"p_value": float(shapiro_b.pvalue), "normal": bool(shapiro_b.pvalue > 0.05)},
    }


def test_variance(data_a: np.ndarray, data_b: np.ndarray) -> Dict[str, Any]:
    levene_stat, levene_p = stats.levene(data_a, data_b)
    return {"levene_stat": float(levene_stat), "p_value": float(levene_p), "equal_variance": bool(levene_p > 0.05)}


def perform_t_or_u_test(data_a: np.ndarray, data_b: np.ndarray, *, alpha: float = 0.05) -> Dict[str, Any]:
    norm_results = test_normality(data_a, data_b)
    variance_results = test_variance(data_a, data_b)

    if norm_results["Group A"]["normal"] and norm_results["Group B"]["normal"]:
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=variance_results["equal_variance"])
        result = {"test": "t-test", "t_stat": float(t_stat), "p_value": float(p_value)}
    else:
        u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
        result = {"test": "Mann-Whitney U-test", "u_stat": float(u_stat), "p_value": float(p_value)}

    result["alpha"] = float(alpha)
    result["significant"] = bool(result["p_value"] < alpha)
    return result


def cohens_d(data_a: np.ndarray, data_b: np.ndarray) -> float:
    nx, ny = len(data_a), len(data_b)
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(data_a, ddof=1) ** 2 + (ny - 1) * np.std(data_b, ddof=1) ** 2) / (nx + ny - 2)
    )
    return float((np.mean(data_a) - np.mean(data_b)) / pooled_std)


def mean_difference_ci(data_a: np.ndarray, data_b: np.ndarray, *, confidence: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    diff = float(np.mean(data_b) - np.mean(data_a))
    n_a, n_b = len(data_a), len(data_b)
    var_a = np.var(data_a, ddof=1)
    var_b = np.var(data_b, ddof=1)

    se_diff = np.sqrt(var_a / n_a + var_b / n_b)
    dof = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a**2) / (n_a**2 * (n_a - 1)) + (var_b**2) / (n_b**2 * (n_b - 1))
    )
    ci = stats.t.interval(confidence, dof, loc=diff, scale=se_diff)
    return diff, (float(ci[0]), float(ci[1]))

