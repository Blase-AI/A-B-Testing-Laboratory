from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

import numpy as np


Decision = Literal["H0", "H1", "No decision"]


@dataclass(frozen=True)
class SPRTResult:
    decision: Decision
    n_used: int
    final_log_lr: float
    history: List[float]
    upper_bound: float
    lower_bound: float


class SequentialSPRT:
    def __init__(
        self,
        f0: Callable[[float], float],
        f1: Callable[[float], float],
        *,
        alpha: float = 0.05,
        beta: float = 0.2,
        verbose: bool = False,
    ) -> None:
        self.f0 = f0
        self.f1 = f1
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.verbose = bool(verbose)

        self.upper_bound = float(np.log((1 - self.beta) / self.alpha))
        self.lower_bound = float(np.log(self.beta / (1 - self.alpha)))

    def run(self, data: np.ndarray) -> SPRTResult:
        log_lr = 0.0
        history: List[float] = []

        for i, x in enumerate(data, start=1):
            delta = float(np.log(self.f1(float(x))) - np.log(self.f0(float(x))))
            log_lr += delta
            history.append(log_lr)

            if log_lr >= self.upper_bound:
                return SPRTResult("H1", i, log_lr, history, self.upper_bound, self.lower_bound)
            if log_lr <= self.lower_bound:
                return SPRTResult("H0", i, log_lr, history, self.upper_bound, self.lower_bound)

        return SPRTResult("No decision", int(len(data)), log_lr, history, self.upper_bound, self.lower_bound)

