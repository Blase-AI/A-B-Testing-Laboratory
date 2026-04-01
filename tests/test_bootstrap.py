import numpy as np


def test_bootstrap_stats_and_plots_smoke():
    from app.ab_testing import BootstrapABTest

    a = np.random.normal(0, 1, 200)
    b = np.random.normal(0.2, 1, 200)

    test = BootstrapABTest(a, b)
    stats = test.get_bootstrap_statistics(n_bootstrap=2000)
    assert "p_value" in stats and "ci_diff" in stats and "cohen_d" in stats

    assert test.plot_bootstrap_distributions(n_bootstrap=2000) is not None
    assert test.plot_bootstrap_diff_hist(n_bootstrap=2000) is not None
    assert test.plot_bootstrap_diff_ecdf(n_bootstrap=2000) is not None


def test_bootstrap_diff_hist_degenerate_does_not_crash():
    from app.ab_testing import BootstrapABTest

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    test = BootstrapABTest(a, b)
    assert test.plot_bootstrap_diff_hist(n_bootstrap=2000) is not None


def test_bootstrap_p_value_in_range():
    from app.ab_testing.methods.bootstrap import stats

    diff = np.array([-1.0, 0.0, 1.0, 2.0])
    p = stats.compute_bootstrap_p_value(diff)
    assert 0.0 <= p <= 1.0
