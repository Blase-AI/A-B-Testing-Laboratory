import numpy as np


def test_classic_stats_and_plots_smoke():
    from app.ab_testing import ClassicABTest

    a = np.random.normal(0, 1, 200)
    b = np.random.normal(0.2, 1, 200)

    test = ClassicABTest(a, b)
    norm = test.test_normality()
    assert "Group A" in norm and "Group B" in norm

    res = test.perform_t_test(alpha=0.05)
    assert "p_value" in res and "test" in res

    d = test.calculate_cohens_d()
    assert isinstance(d, float)

    assert test.plot_distribution() is not None
    assert test.plot_boxplot() is not None
    assert test.plot_mean_difference_ci() is not None
    assert test.plot_qq() is not None

