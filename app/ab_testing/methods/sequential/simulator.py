from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .model import SequentialSPRT, SPRTResult


@dataclass
class NormalSPRTSimulator:
    mu0: float
    mu1: float
    sigma: float
    n: int
    alpha: float = 0.05
    beta: float = 0.2
    true_state: str = "H0"
    random_seed: int | None = None
    stop_threshold: int | None = None

    def _generate(self) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        mu = self.mu0 if self.true_state == "H0" else self.mu1
        return rng.normal(mu, self.sigma, int(self.n))

    def run(self) -> SPRTResult:
        data = self._generate()
        if self.stop_threshold is not None:
            data = data[: int(self.stop_threshold)]

        def f0(x: float) -> float:
            return float(norm.pdf(x, loc=self.mu0, scale=self.sigma))

        def f1(x: float) -> float:
            return float(norm.pdf(x, loc=self.mu1, scale=self.sigma))

        sprt = SequentialSPRT(f0, f1, alpha=self.alpha, beta=self.beta, verbose=False)
        return sprt.run(data)

    def run_simulations(self, n_simulations: int = 100) -> dict[str, object]:
        decisions = []
        n_used_list = []
        final_log_lr_list = []

        for i in range(int(n_simulations)):
            self.random_seed = None if self.random_seed is None else (self.random_seed + i)
            res = self.run()
            decisions.append(res.decision)
            n_used_list.append(res.n_used)
            final_log_lr_list.append(res.final_log_lr)

        return {
            "n_simulations": int(n_simulations),
            "decision_counts": {
                "H0": int(decisions.count("H0")),
                "H1": int(decisions.count("H1")),
                "No decision": int(decisions.count("No decision")),
            },
            "avg_n_used": float(np.mean(n_used_list)) if n_used_list else 0.0,
            "n_used_list": n_used_list,
            "final_log_lr_list": final_log_lr_list,
        }
