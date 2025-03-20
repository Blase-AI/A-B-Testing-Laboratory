import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Callable, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class SequentialTest:
    """
    A class for conducting sequential testing using the Sequential Probability Ratio Test (SPRT).
    Parameters:
        f0: Probability density function for H0. Takes an observation and returns f0(x)
        f1: Probability density function for H1. Takes an observation and returns f1(x)
        alpha: Significance level (Type I error)
        beta: Type II error probability (power = 1 - beta)
        verbose: If True, outputs detailed test progress information
    """

    def __init__(self, 
                 f0: Callable[[float], float], 
                 f1: Callable[[float], float], 
                 alpha: float = 0.05, 
                 beta: float = 0.2, 
                 verbose: bool = True) -> None:
        self.f0 = f0
        self.f1 = f1
        self.alpha = alpha
        self.beta = beta
        self.verbose = verbose

        self.upper_bound: float = np.log((1 - beta) / alpha)
        self.lower_bound: float = np.log(beta / (1 - alpha))
        if self.verbose:
            logging.info(f"Bounds set: upper = {self.upper_bound:.4f}, lower = {self.lower_bound:.4f}")

    def run(self, data: np.ndarray) -> Tuple[str, int, float, List[float]]:
        """
        Executes sequential testing on the provided data sequence.
        
        Args:
            data: Array of observations
            
        Returns:
            Tuple containing:
                - decision: "H0", "H1" or "No decision"
                - n_used: Number of observations used
                - final_log_lr: Final log-likelihood ratio value
                - history: List of log(LR) values after each observation
        """
        log_lr: float = 0.0
        history: List[float] = []
        n_used: int = 0

        for i, x in enumerate(data, start=1):
            delta = np.log(self.f1(x)) - np.log(self.f0(x))
            log_lr += delta
            history.append(log_lr)
            if self.verbose:
                logging.info(f"Observation {i}: log(LR) = {log_lr:.4f}")

            if log_lr >= self.upper_bound:
                if self.verbose:
                    logging.info(f"H1 decision made at  {i} observations "
                                 f"(log(LR) >= {self.upper_bound:.4f}).")
                return "H1", i, log_lr, history
            elif log_lr <= self.lower_bound:
                if self.verbose:
                    logging.info(f"H0 decision made at {i} observations "
                                 f"(log(LR) <= {self.lower_bound:.4f}).")
                return "H0", i, log_lr, history

        if self.verbose:
            logging.info("No decision made. Reached end of sample without crossing boundaries.")
        return "No decision", len(data), log_lr, history

    def plot_history(self) -> plt.Figure:
        """
        Plots the log(LR) evolution with decision boundaries.
        
        Returns:
            matplotlib.figure.Figure: Figure object for integration with visualization tools
            
        Raises:
            ValueError: If test hasn't been executed yet
        """
        decision, n_used, final_log_lr, history = self.run(np.array([]))  
        if not history:
            raise ValueError("Test not executed. Call run() first.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(history) + 1), history, marker='o', label='log(LR)')
        ax.axhline(self.upper_bound, color='green', linestyle='--', label='H1 boundary')
        ax.axhline(self.lower_bound, color='red', linestyle='--', label='H0 boundary')
        ax.set_xlabel('Observation number')
        ax.set_ylabel('Log-likelihood ratio')
        ax.set_title('SPRT Log-Likelihood Ratio Evolution')
        ax.legend()
        ax.grid(True)
        return fig


class SequentialTestSimulator:
    """
    A class for simulating sequential testing with normally distributed data.
    
    Parameters:
        mu0: Mean under H0
        mu1: Mean under H1
        sigma: Common standard deviation
        n: Maximum number of observations per simulation
        alpha: Significance level (Type I error)
        beta: Type II error probability
        true_state: "H0" or "H1" - true state for data generation
        random_seed: Seed for reproducibility
        verbose: If True, outputs detailed simulation info
        stop_threshold: Maximum number of observations before stopping (optional)
    """

    def __init__(self,
                 mu0: float,
                 mu1: float,
                 sigma: float,
                 n: int,
                 alpha: float = 0.05,
                 beta: float = 0.2,
                 true_state: str = "H0",
                 random_seed: Optional[int] = None,
                 verbose: bool = True,
                 stop_threshold: Optional[int] = None) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)

        self.mu0 = mu0
        self.mu1 = mu1
        self.sigma = sigma
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.true_state = true_state
        self.verbose = verbose
        self.stop_threshold = stop_threshold

        if self.true_state == "H0":
            self.data = np.random.normal(mu0, sigma, n)
        elif self.true_state == "H1":
            self.data = np.random.normal(mu1, sigma, n)
        else:
            raise ValueError("true_state must be 'H0' or 'H1'.")

        self.f0: Callable[[float], float] = lambda x: norm.pdf(x, loc=mu0, scale=sigma)
        self.f1: Callable[[float], float] = lambda x: norm.pdf(x, loc=mu1, scale=sigma)

        self.sequential_test = SequentialTest(self.f0, self.f1, alpha, beta, verbose)
        self.decision: Optional[str] = None
        self.n_used: Optional[int] = None
        self.log_lr: Optional[float] = None
        self.history: Optional[List[float]] = None

    def run(self) -> Tuple[str, int, float, List[float]]:
        """
        Executes sequential test on generated data.

        :return: result (decision, n_used, final_log_lr, history).
        """
        if self.stop_threshold is not None:
            data = self.data[:self.stop_threshold]
        else:
            data = self.data
        
        self.decision, self.n_used, self.log_lr, self.history = self.sequential_test.run(data)
        return self.decision, self.n_used, self.log_lr, self.history

    def plot_history(self) -> plt.Figure:
        """
        Visualizes log(LR) evolution for the simulation.
        """
        if self.history is None:
            raise ValueError("Test not executed. Call run() first.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(self.history) + 1), self.history, marker='o', label='log(LR)')
        ax.axhline(self.sequential_test.upper_bound, color='green', linestyle='--', label='H1 boundary')
        ax.axhline(self.sequential_test.lower_bound, color='red', linestyle='--', label='H0 boundary')
        ax.set_xlabel('Observation number')
        ax.set_ylabel('Log-likelihood ratio')
        ax.set_title('SPRT Log-Likelihood Ratio Evolution')
        ax.legend()
        ax.grid(True)
        return fig

    def run_simulations(self, n_simulations: int = 100) -> dict:
        """
        Performs a series of sequential testing simulations with new data generated
        for each simulation. Returns aggregated results:
            - Frequencies of H0, H1, and "No decision" outcomes.
            - Average number of observations used.
            - Array with the number of observations for each simulation.
        
            :param n_simulations: Number of simulations.
            :return: Dictionary with aggregated results.
        """
        decisions = []
        n_used_list = []
        final_log_lr_list = []

        for _ in range(n_simulations):
            if self.true_state == "H0":
                data = np.random.normal(self.mu0, self.sigma, self.n)
            else:
                data = np.random.normal(self.mu1, self.sigma, self.n)
            decision, n_used, final_log_lr, _ = self.sequential_test.run(data)
            decisions.append(decision)
            n_used_list.append(n_used)
            final_log_lr_list.append(final_log_lr)

        freq_H0 = decisions.count("H0")
        freq_H1 = decisions.count("H1")
        freq_no = decisions.count("No decision")
        avg_n_used = np.mean(n_used_list)

        logging.info(f"Simulations: {n_simulations}. H0: {freq_H0}, H1: {freq_H1}, No decision: {freq_no}.")
        logging.info(f"Average observations used: {avg_n_used:.2f}")

        return {
            'n_simulations': n_simulations,
            'decision_counts': {'H0': freq_H0, 'H1': freq_H1, 'No decision': freq_no},
            'avg_n_used': avg_n_used,
            'n_used_list': n_used_list,
            'final_log_lr_list': final_log_lr_list
        }

    def plot_simulation_results(self, simulation_results: dict) -> plt.Figure:
        """
        Visualizes distribution of observations used across simulations.
        
            :param simulation_results: A dictionary derived from run_simulations.
            :return: A Figure object with a built graph.
        """
        n_used_list = simulation_results['n_used_list']
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(n_used_list, bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Observations used')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Observations Used in Simulations')
        ax.grid(True)
        return fig
