import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
import scipy.stats as stats
import arviz as az
from typing import Callable, Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class BootstrapABTest:
    """
    A class for conducting A/B testing using bootstrap analysis.
    
    The class methods allow:
      - Validating input data.
      - Performing bootstrap sampling to compute statistics (e.g., mean).
      - Calculating percentile confidence intervals.
      - Estimating the bootstrap distribution of the difference in means and calculating p-values.
      - Computing the effect size (Cohen's d).
      - Plotting distributions, differences, and ECDFs.
      - Exporting key bootstrap statistics for further analysis.
    """
    
    def __init__(self, group_A: np.ndarray, group_B: np.ndarray) -> None:
        """
        Initializes the A/B test object with two datasets.
        
        :param group_A: Data array for group A.
        :param group_B: Data array for group B.
        :raises ValueError: If data is empty or non-numeric.
        """
        self.group_A = self._validate_data(group_A, "Group A")
        self.group_B = self._validate_data(group_B, "Group B")
    
    @staticmethod
    def _validate_data(data: np.ndarray, group_name: str) -> np.ndarray:
        """
        Validates that the data is not empty and is numeric.
        
        :param data: Input data array.
        :param group_name: Group name for error messages.
        :return: Flattened numeric array.
        :raises ValueError: If data is invalid.
        """
        arr = np.array(data)
        if arr.size == 0:
            raise ValueError(f"Data for {group_name} is empty.")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Data for {group_name} must be numeric.")
        return arr.flatten()
    
    def bootstrap_statistic(self, data: np.ndarray, statistic_func: Callable[[np.ndarray], float] = np.mean,
                            n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> np.ndarray:
        """
        Performs bootstrap sampling to compute a specified statistic.
        
        :param data: Input data array.
        :param statistic_func: Function to compute the statistic (default is np.mean).
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Array of bootstrap statistic values.
        """
        rng = np.random.default_rng(random_state)
        n = len(data)
        boot_stats = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=n, replace=True)
            boot_stats[i] = statistic_func(sample)
        return boot_stats
    
    def compute_bootstrap_means(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes bootstrap distributions of means for groups A and B.
        
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Tuple of two arrays: (boot_means_A, boot_means_B).
        """
        boot_means_A = self.bootstrap_statistic(self.group_A, np.mean, n_bootstrap, random_state)
        boot_means_B = self.bootstrap_statistic(self.group_B, np.mean, n_bootstrap, random_state)
        return boot_means_A, boot_means_B
    
    @staticmethod
    def percentile_ci(boot_stats: np.ndarray, ci: float = 95) -> Tuple[float, float]:
        """
        Computes a percentile confidence interval for the bootstrap distribution.
        
        :param boot_stats: Array of bootstrap statistics.
        :param ci: Confidence interval level (in percent).
        :return: Tuple (lower bound, upper bound).
        """
        lower = np.percentile(boot_stats, (100 - ci) / 2)
        upper = np.percentile(boot_stats, 100 - (100 - ci) / 2)
        return lower, upper
    
    def compute_bootstrap_diff(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> np.ndarray:
        """
        Computes the bootstrap distribution of the difference in means (Group B - Group A).
        
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Array of differences between groups.
        """
        boot_means_A, boot_means_B = self.compute_bootstrap_means(n_bootstrap, random_state)
        return boot_means_B - boot_means_A
    
    def compute_bootstrap_p_value(self, boot_diff: np.ndarray) -> float:
        """
        Computes the p-value based on the bootstrap distribution of the difference.
        
        :param boot_diff: Array of differences (Group B - Group A).
        :return: Two-sided p-value.
        """
        p_value = 2 * min(np.mean(boot_diff <= 0), np.mean(boot_diff >= 0))
        return p_value
    
    @staticmethod
    def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the effect size Cohen's d between two groups.
        
        :param x: Data for group x.
        :param y: Data for group y.
        :return: Cohen's d value.
        """
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    def plot_bootstrap_distributions(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> plt.Figure:
        """
        Plots KDE densities of bootstrap distributions of means for groups A and B.
        
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Figure object with the plotted graph.
        """
        boot_means_A, boot_means_B = self.compute_bootstrap_means(n_bootstrap, random_state)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.kdeplot(boot_means_A, label='Bootstrapped Means A', fill=True, color='blue', ax=ax)
        sns.kdeplot(boot_means_B, label='Bootstrapped Means B', fill=True, color='orange', ax=ax)
        ax.set_xlabel('Metric Mean')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distributions of Means for Groups A and B')
        ax.legend()
        return fig
    
    def plot_bootstrap_diff_hist(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> plt.Figure:
        """
        Plots a histogram with KDE for the bootstrap distribution of the difference in means (Group B - Group A).
    
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Figure object with the plotted graph.
        """
        boot_diff = self.compute_bootstrap_diff(n_bootstrap, random_state)
    
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(boot_diff, bins=50, kde=True, color='purple', ax=ax)
        ax.axvline(0, color='red', linestyle='dashed', label='No Difference')
        ax.set_xlabel('Difference in Means (B - A)')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distribution of Difference in Means')
        ax.legend()
    
        return fig

    def plot_bootstrap_diff_ecdf(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42) -> plt.Figure:
        """
        Plots an ECDF graph for the bootstrap distribution of the difference in means (Group B - Group A).
    
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :return: Figure object with the plotted graph.
        """
        boot_diff = self.compute_bootstrap_diff(n_bootstrap, random_state)
    
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.ecdfplot(boot_diff, color='green', ax=ax)
        ax.axvline(0, color='red', linestyle='dashed')
        ax.set_xlabel('Difference in Means (B - A)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('ECDF: Bootstrap Distribution of Difference (B - A)')
    
        return fig

    def get_bootstrap_statistics(self, n_bootstrap: int = 10000, random_state: Optional[int] = 42,
                                 ci: float = 95) -> Dict[str, Any]:
        """
        Computes and returns key bootstrap statistics:
          - Distributions of means for groups A and B.
          - Percentile confidence intervals for means.
          - Distribution of the difference (B - A) and its confidence interval.
          - Two-sided bootstrap p-value.
          - Effect size Cohen's d for the original data.
        
        :param n_bootstrap: Number of bootstrap samples.
        :param random_state: Seed for random number generation.
        :param ci: Confidence interval level (in percent).
        :return: Dictionary with computed statistics.
        """
        boot_means_A, boot_means_B = self.compute_bootstrap_means(n_bootstrap, random_state)
        ci_A = self.percentile_ci(boot_means_A, ci)
        ci_B = self.percentile_ci(boot_means_B, ci)
        
        boot_diff = boot_means_B - boot_means_A
        ci_diff = self.percentile_ci(boot_diff, ci)
        p_value = self.compute_bootstrap_p_value(boot_diff)
        effect_size = self.cohen_d(self.group_A, self.group_B)
        
        stats_dict = {
            'boot_means_A': boot_means_A,
            'boot_means_B': boot_means_B,
            'ci_A': ci_A,
            'ci_B': ci_B,
            'boot_diff': boot_diff,
            'ci_diff': ci_diff,
            'p_value': p_value,
            'cohen_d': effect_size
        }
        return stats_dict

def perform_power_analysis(effect_size: float, alpha: float, nobs1: int, desired_power: float,
                           sample_sizes: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Performs power analysis and visualizes the relationship between power and sample size.
    
    :param effect_size: Effect size (Cohen's d).
    :param alpha: Significance level.
    :param nobs1: Sample size for group A (for equal groups).
    :param desired_power: Desired power of the test (e.g., 0.8 for 80%).
    :param sample_sizes: (Optional) Array of sample size values for plotting.
                         If not provided, a range from 100 to 2000 with a step of 50 is used.
    :return: Dictionary with current power, required sample size for desired power,
             and the Figure object with the plot.
    """
    analysis = TTestIndPower()
    
    current_power = analysis.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=1.0)
    logging.info(f"Test power at n = {nobs1}: {current_power:.4f}")
    
    required_n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0)
    logging.info(f"Required sample size per group for {desired_power*100:.0f}%power: {required_n:.0f}")
    
    if sample_sizes is None:
        sample_sizes = np.arange(100, 2000, 50)
    
    powers = analysis.power(effect_size=effect_size, nobs1=sample_sizes, alpha=alpha, ratio=1.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, powers, marker='o', linestyle='-', label='Test Power')
    ax.axhline(desired_power, color='red', linestyle='--', label=f'Desired Power: {desired_power}')
    ax.set_xlabel('Sample Size per Group')
    ax.set_ylabel('Test Power')
    ax.set_title("Power Analysis: Power vs. Sample Size")
    ax.legend()
    ax.grid(True)
    
    return {'current_power': current_power, 'required_n': required_n, 'figure': fig}
