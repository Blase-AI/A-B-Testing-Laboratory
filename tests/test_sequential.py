import numpy as np
from scipy.stats import norm


def test_sequential_sprt_returns_result_and_plots():
    from app.ab_testing import SequentialSPRT, plot_sprt_history

    def f0(x: float) -> float:
        return float(norm.pdf(x, loc=0.0, scale=1.0))

    def f1(x: float) -> float:
        return float(norm.pdf(x, loc=0.3, scale=1.0))

    sprt = SequentialSPRT(f0, f1, alpha=0.05, beta=0.2, verbose=False)
    res = sprt.run(np.random.normal(0.3, 1.0, 50))
    assert res.decision in ("H0", "H1", "No decision")
    assert res.n_used > 0
    assert len(res.history) == res.n_used
    assert plot_sprt_history(res) is not None


def test_normal_sprt_simulator_smoke():
    from app.ab_testing import NormalSPRTSimulator

    sim = NormalSPRTSimulator(
        mu0=0.0, mu1=0.3, sigma=1.0, n=50, alpha=0.05, true_state="H1", stop_threshold=30
    )
    res = sim.run()
    assert res.n_used <= 30
    agg = sim.run_simulations(n_simulations=10)
    assert agg["n_simulations"] == 10
