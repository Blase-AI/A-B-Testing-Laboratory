from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.power import TTestIndPower


def perform_power_analysis(
    effect_size: float,
    *,
    alpha: float = 0.05,
    nobs1: int = 100,
    desired_power: float = 0.8,
    sample_sizes: np.ndarray | None = None,
) -> dict[str, Any]:
    analysis = TTestIndPower()
    current_power = analysis.power(effect_size=effect_size, nobs1=nobs1, alpha=alpha, ratio=1.0)
    required_n = analysis.solve_power(
        effect_size=effect_size, power=desired_power, alpha=alpha, ratio=1.0
    )

    if sample_sizes is None:
        sample_sizes = np.arange(50, 2000, 50)

    powers = analysis.power(effect_size=effect_size, nobs1=sample_sizes, alpha=alpha, ratio=1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sample_sizes, powers, marker="o", linestyle="-", label="Test Power")
    ax.axhline(desired_power, color="red", linestyle="--", label=f"Desired Power: {desired_power}")
    ax.set_xlabel("Sample Size per Group")
    ax.set_ylabel("Test Power")
    ax.set_title("Power Analysis: Power vs. Sample Size")
    ax.legend()
    ax.grid(True)

    return {"current_power": current_power, "required_n": required_n, "figure": fig}
