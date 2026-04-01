def test_bayesian_stats_and_plots_smoke():
    from app.ab_testing import BayesianABTest

    test = BayesianABTest(100, 40, 120, 55)
    p = test.compute_prob_B_better(num_samples=2000)
    assert 0 <= p <= 1

    diff = test.compute_difference_stats(num_samples=2000)
    assert "delta" in diff and "hdi_interval" in diff

    assert test.plot_posteriors(num_samples=2000) is not None
    assert test.plot_boxplot(num_samples=2000) is not None
    assert test.plot_kde(num_samples=2000) is not None
    assert test.plot_difference_hist_kde(num_samples=2000) is not None
    assert test.plot_difference_cdf(num_samples=2000) is not None


def test_bayesian_difference_hist_degenerate_does_not_crash():
    from app.ab_testing import BayesianABTest

    test = BayesianABTest(10, 0, 10, 0)
    assert test.plot_difference_hist_kde(num_samples=2000) is not None

