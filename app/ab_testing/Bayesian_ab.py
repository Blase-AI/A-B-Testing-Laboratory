import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import arviz as az
from statsmodels.stats.power import TTestIndPower
from typing import Tuple, Dict, Any, Optional, Union

class BayesianABTest:
    """
    A class for conducting Bayesian A/B testing using Beta distributions.
    
    Performs:
      - Calculation of posterior distributions for conversion rates in groups A and B.
      - Estimation of the probability that the conversion rate of group B is higher than that of group A.
      - Calculation of the difference between groups and the HDI confidence interval for this difference.
      - Visualization of posterior distributions and the difference between groups.
    """
    
    def __init__(self, n_A: int, successes_A: int, n_B: int, successes_B: int,
                 alpha_prior: float = 1.0, beta_prior: float = 1.0) -> None:
        """
        Initializes the Bayesian A/B test object.
        
        :param n_A: Total number of observations in group A.
        :param successes_A: Number of successful outcomes in group A.
        :param n_B: Total number of observations in group B.
        :param successes_B: Number of successful outcomes in group B.
        :param alpha_prior: Alpha parameter of the prior Beta distribution.
        :param beta_prior: Beta parameter of the prior Beta distribution.
        :raises ValueError: If input data is invalid.
        """
        if n_A <= 0 or n_B <= 0:
            raise ValueError("Number of observations must be positive.")
        if not (0 <= successes_A <= n_A):
            raise ValueError("Number of successes in group A must be between 0 and n_A.")
        if not (0 <= successes_B <= n_B):
            raise ValueError("Number of successes in group B must be between 0 and n_B.")
        
        self.n_A = n_A
        self.successes_A = successes_A
        self.n_B = n_B
        self.successes_B = successes_B
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        self.posterior_A = stats.beta(self.alpha_prior + self.successes_A,
                                      self.beta_prior + (self.n_A - self.successes_A))
        self.posterior_B = stats.beta(self.alpha_prior + self.successes_B,
                                      self.beta_prior + (self.n_B - self.successes_B))
    
    def sample_posteriors(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples from the posterior distributions for groups A and B.
        
        :param num_samples: Number of samples.
        :return: Tuple (samples_A, samples_B).
        """
        samples_A = self.posterior_A.rvs(num_samples)
        samples_B = self.posterior_B.rvs(num_samples)
        return samples_A, samples_B
    
    def compute_prob_B_better(self, num_samples: int = 10000) -> float:
        """
        Computes the probability that the conversion rate of group B is higher than that of group A.
        
        :param num_samples: Number of samples for estimation.
        :return: Probability (between 0 and 1).
        """
        samples_A, samples_B = self.sample_posteriors(num_samples)
        return np.mean(samples_B > samples_A)
    
    def compute_difference_stats(self, num_samples: int = 10000, hdi_prob: float = 0.95) -> Dict[str, Any]:
        """
        Computes the difference between samples of groups B and A and returns the HDI confidence interval for the difference.
        
        :param num_samples: Number of samples for estimating the difference.
        :param hdi_prob: Probability for computing HDI.
        :return: Dictionary with keys 'delta' (difference) and 'hdi_interval' (confidence interval).
        """
        samples_A, samples_B = self.sample_posteriors(num_samples)
        delta = samples_B - samples_A
        hdi_interval = az.hdi(delta, hdi_prob=hdi_prob)
        return {'delta': delta, 'hdi_interval': hdi_interval}
    
    def plot_posteriors(self, num_samples: int = 10000) -> plt.Figure:
        """
        Plots the posterior distributions for groups A and B and returns the Figure object.
        
        :param num_samples: Number of samples for estimation.
        :return: Figure object with the plotted graph.
        """
        samples_A, samples_B = self.sample_posteriors(num_samples)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(samples_A, color='blue', label='Group A', kde=True, stat="density", bins=15, ax=ax)
        sns.histplot(samples_B, color='orange', label='Group B', kde=True, stat="density", bins=15, ax=ax)
        ax.set_title('Posterior Distributions of Conversion Rates for Groups A and B')
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Density')
        ax.legend()
        return fig
    
    def plot_boxplot(self, num_samples: Optional[int] = None) -> plt.Figure:
        """
        Plots a boxplot for groups A and B and returns the Figure object.
        
        :param num_samples: If specified, used for sampling; otherwise, n_A and n_B samples are taken.
        :return: Figure object with the plotted graph.
        """
        if num_samples is None:
            samples_A = self.posterior_A.rvs(self.n_A)
            samples_B = self.posterior_B.rvs(self.n_B)
        else:
            samples_A, samples_B = self.sample_posteriors(num_samples)
        
        data = pd.DataFrame({
            'Group': ['A'] * len(samples_A) + ['B'] * len(samples_B),
            'Value': np.concatenate([samples_A, samples_B])
        })
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Group', y='Value', data=data, ax=ax)
        ax.set_title('Boxplot for Groups A and B')
        return fig
    
    def plot_kde(self, num_samples: int = 10000) -> plt.Figure:
        """
        Plots KDE densities for the posterior distributions of groups A and B and returns the Figure object.
        
        :param num_samples: Number of samples for estimation.
        :return: Figure object with the plotted graph.
        """
        samples_A, samples_B = self.sample_posteriors(num_samples)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.kdeplot(samples_A, label='Group A', fill=True, color='blue', ax=ax)
        sns.kdeplot(samples_B, label='Group B', fill=True, color='orange', ax=ax)
        ax.set_xlabel('Conversion Rate')
        ax.set_ylabel('Density')
        ax.set_title('Posterior Distributions of Conversion Rates for A/B Test')
        ax.legend()
        return fig
    
    def plot_difference_hist_kde(self, num_samples: int = 10000) -> plt.Figure:
        """
        Plots a histogram with KDE for the difference between groups.
    
        :param num_samples: Number of samples for estimating the difference.
        :return: Figure object with the plotted graph.
        """
        diff_stats = self.compute_difference_stats(num_samples=num_samples)
        delta = diff_stats['delta']
    
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(delta, bins=50, kde=True, color='purple', ax=ax)
        ax.axvline(0, color='red', linestyle='dashed')
        ax.set_xlabel('Difference in Conversion Rate (B - A)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Difference Between B and A')
    
        return fig
    
    def plot_difference_cdf(self, num_samples: int = 10000) -> plt.Figure:
        """
        Plots a CDF graph for the difference between groups.
    
        :param num_samples: Number of samples for estimating the difference.
        :return: Figure object with the plotted graph.
        """
        diff_stats = self.compute_difference_stats(num_samples=num_samples)
        delta = diff_stats['delta']
    
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.ecdfplot(delta, color='darkgreen', ax=ax)
        ax.axvline(0, color='red', linestyle='dashed')
        ax.set_xlabel('Difference in Conversion Rate (B - A)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('CDF: Probability that Difference ≤ X')
    
        return fig

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
    print(f"Test power at n  = {nobs1}: {current_power:.4f}")
    
    required_n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0)
    print(f"Required sample size per group for {desired_power*100:.0f}%power: {required_n:.0f}")
    
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
