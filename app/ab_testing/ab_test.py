import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from typing import Dict, Any, Optional, Union

class ABTest:
    """
    A class for conducting A/B testing with data analysis and visualization.

    This class encapsulates methods for:
      - Descriptive analysis
      - Normality testing
      - Variance equality testing
      - Performing t-tests or alternative non-parametric U-tests
      - Calculating effect size (Cohen's d)
      - Plotting distributions and boxplots
    """
    
    def __init__(self, data_A: np.ndarray, data_B: np.ndarray) -> None:
        """
        Initializes the A/B test with two datasets.

        Args:
            data_A: Array of data for group A.
            data_B: Array of data for group B.
        
        Raises:
            ValueError: If the input data is empty or contains non-numeric values.
        """
        self.data_A = self._validate_data(data_A, "A")
        self.data_B = self._validate_data(data_B, "B")
    
    @staticmethod
    def _validate_data(data: Union[np.ndarray, list], group_name: str) -> np.ndarray:
        """
        Validates that the data is a non-empty array of numbers.

        Args:
            data: Input data.
            group_name: Group name for error messages.
        
        Returns:
            A flattened numpy array of numbers.
        
        Raises:
            ValueError: If the data is invalid.
        """
        arr = np.array(data)
        if arr.size == 0:
            raise ValueError(f"Data for {group_name} is empty.")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"Data for {group_name} must be numeric.")
        return arr.flatten()
    
    def descriptive_stats(self) -> pd.DataFrame:
        """
        Computes descriptive statistics (mean, standard deviation) for both groups.

        Returns:
            DataFrame with descriptive statistics.
        """
        stats_dict = {
            'Group': ['A', 'B'],
            'Mean': [np.mean(self.data_A), np.mean(self.data_B)],
            'Std': [np.std(self.data_A, ddof=1), np.std(self.data_B, ddof=1)]
        }
        return pd.DataFrame(stats_dict)
    
    def test_normality(self) -> Dict[str, Any]:
        """
        Performs the Shapiro-Wilk test to check the normality of the distribution.

        Returns:
            Dictionary with p-values for groups A and B.
        """
        shapiro_A = stats.shapiro(self.data_A)
        shapiro_B = stats.shapiro(self.data_B)
        results = {
            'Group A': {'p_value': shapiro_A.pvalue, 'normal': shapiro_A.pvalue > 0.05},
            'Group B': {'p_value': shapiro_B.pvalue, 'normal': shapiro_B.pvalue > 0.05}
        }
        return results
    
    def test_variance(self) -> Dict[str, Any]:
        """
        Performs Levene's test to check the equality of variances between the two groups.

        Returns:
            Dictionary with p-value and a flag indicating equal variances.
        """
        levene_stat, levene_p = stats.levene(self.data_A, self.data_B)
        return {'levene_stat': levene_stat, 'p_value': levene_p, 'equal_variance': levene_p > 0.05}
    
    def perform_t_test(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Performs a classical t-test for independent samples. If normality is not met in at least one group,
        a non-parametric Mann-Whitney U-test is performed.

        Args:
            alpha: Significance level (default is 0.05).

        Returns:
            Dictionary with test results: statistic, p-value, and information about the test used.
        """
        norm_results = self.test_normality()
        variance_results = self.test_variance()
        results = {}
        
        if norm_results['Group A']['normal'] and norm_results['Group B']['normal']:
            t_stat, p_value = stats.ttest_ind(self.data_A, self.data_B, equal_var=variance_results['equal_variance'])
            results['test'] = 't-test'
            results['t_stat'] = t_stat
            results['p_value'] = p_value
        else:
            u_stat, p_value = stats.mannwhitneyu(self.data_A, self.data_B, alternative='two-sided')
            results['test'] = 'Mann-Whitney U-test'
            results['u_stat'] = u_stat
            results['p_value'] = p_value
        
        results['alpha'] = alpha
        results['significant'] = p_value < alpha  
        return results
    
    def plot_qq(self) -> plt.Figure:
        """
        Plots Q-Q plots to check the normality of the distribution for both groups.

        Returns:
            matplotlib.figure.Figure: Figure object with Q-Q plots for groups A and B.
        """
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    
        (osm, osr), (slope, intercept, r) = stats.probplot(self.data_A, dist="norm", plot=ax[0])
        ax[0].plot(osm, slope*osm + intercept, color='red', linestyle='--', label=f'R² = {r**2:.2f}\nSlope = {slope:.2f}')
        ax[0].set_title("Group A", fontsize=12, pad=10)
        ax[0].set_xlabel("Theoretical Quantiles", fontsize=10)
        ax[0].set_ylabel("Sample Quantiles", fontsize=10)
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(loc='upper left')

        (osm, osr), (slope, intercept, r) = stats.probplot(self.data_B, dist="norm", plot=ax[1])
        ax[1].plot(osm, slope*osm + intercept, color='red', linestyle='--',label=f'R² = {r**2:.2f}\nSlope = {slope:.2f}')
        ax[1].set_title("Group B", fontsize=12, pad=10)
        ax[1].set_xlabel("Theoretical Quantiles", fontsize=10)
        ax[1].set_ylabel("Sample Quantiles")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(loc='upper left')

        fig.suptitle("Q-Q Plots for Normality Check", y=1.02, fontsize=14, weight='bold')
        fig.tight_layout()
    
        return fig

    def plot_mean_difference_ci(self, confidence: float = 0.95) -> plt.Figure:
        """
         Plots the difference in means with a confidence interval.

        Args:
            confidence: Confidence level (default is 0.95).

        Returns:
            matplotlib.figure.Figure: Figure object with the plot.
        """
        diff = np.mean(self.data_B) - np.mean(self.data_A)
        n_A, n_B = len(self.data_A), len(self.data_B)
        var_A = np.var(self.data_A, ddof=1)
        var_B = np.var(self.data_B, ddof=1)
        
        se_diff = np.sqrt(var_A / n_A + var_B / n_B)
        
        dof = (var_A / n_A + var_B / n_B)**2 / (
            (var_A**2) / (n_A**2 * (n_A - 1)) + 
            (var_B**2) / (n_B**2 * (n_B - 1))
        )
        
        ci = stats.t.interval(confidence, dof, loc=diff, scale=se_diff)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.errorbar(0, diff, yerr=[[diff - ci[0]], [ci[1] - diff]], 
                    fmt='o', color='blue', capsize=10, markersize=8)
        ax.axhline(0, color='red', linestyle='--', label="There is no difference")
        ax.set_xticks([])
        ax.set_ylabel("Difference (B - A)")
        ax.set_title(f"Difference in Means with {int(confidence*100)}% Confidence Interval")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig


    def calculate_cohens_d(self) -> float:
        """
        Calculates the effect size (Cohen's d).

        Returns:
            Cohen's d value.
        """
        nx, ny = len(self.data_A), len(self.data_B)
        pooled_std = np.sqrt(((nx - 1) * np.std(self.data_A, ddof=1) ** 2 + (ny - 1) * np.std(self.data_B, ddof=1) ** 2) / (nx + ny - 2))
        return (np.mean(self.data_A) - np.mean(self.data_B)) / pooled_std
    
    def plot_distribution(self) -> plt.Figure:
        """
        Plots the distribution (histograms with KDE) for groups A and B.

        Returns:
            matplotlib.figure.Figure: Figure object with the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(self.data_A, color='blue', label='Group A', kde=True, stat="density", bins=15, ax=ax)
        sns.histplot(self.data_B, color='orange', label='Group B', kde=True, stat="density", bins=15, ax=ax)
        ax.set_title('Distribution of Metric for Groups A and B')
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Density')
        ax.legend()
        return fig
    
    def plot_boxplot(self) -> plt.Figure:
        """
        Plots a boxplot to compare the distributions of groups A and B.

        Returns:
            matplotlib.figure.Figure: Figure object with the boxplot.
        """
        df = pd.DataFrame({
            'Group': ['A'] * len(self.data_A) + ['B'] * len(self.data_B),
            'Value': np.concatenate([self.data_A, self.data_B])
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Group', y='Value', data=df, ax=ax)
        ax.set_title('Boxplot for Groups A and B')
        ax.set_xlabel('Group')
        ax.set_ylabel('Metric Value')
        return fig
    
    @staticmethod
    def perform_power_analysis(effect_size: float, alpha: float = 0.05, nobs1: int = 100, desired_power: float = 0.8,
                               sample_sizes: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Performs power analysis and visualizes the relationship between power and sample size.

        Args:
            effect_size: Effect size (Cohen's d).
            alpha: Significance level.
            nobs1: Sample size for group A (for equal groups).
            desired_power: Desired power of the test (e.g., 0.8 for 80%).
            sample_sizes: (Optional) Array of sample sizes for plotting.
                          If not provided, a range from 50 to 2000 with a step of 50 is used.

        Returns:
            Dictionary with current power, required sample size for desired power,
            and a matplotlib.figure.Figure object with the plot.
        """
        analysis = TTestIndPower()
        
        current_power = analysis.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=1.0)
        print(f"Power of the test at n  = {nobs1}: {current_power:.4f}")
        
        required_n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0)
        print(f"Required sample size per group for {desired_power*100:.0f}% power: {required_n:.0f}")
        
        if sample_sizes is None:
            sample_sizes = np.arange(50, 2000, 50)
        
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
