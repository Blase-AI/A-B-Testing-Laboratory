import numpy as np


def test_public_api_exports():
    from app import ab_testing as ab

    assert ab.ABTest is ab.ClassicABTest

    for attr in [
        "ClassicABTest",
        "BootstrapABTest",
        "BayesianABTest",
        "SequentialSPRT",
        "SPRTResult",
        "NormalSPRTSimulator",
        "plot_sprt_history",
        "BaseABTest",
        "TwoGroupContinuousABTest",
        "classic_stats",
        "classic_plots",
        "bootstrap_stats",
        "bootstrap_plots",
        "bayesian_stats",
        "bayesian_plots",
        "validate_numeric_1d",
        "perform_power_analysis",
        "new_figure",
        "new_figure_grid",
    ]:
        assert hasattr(ab, attr)


def test_top_level_classes_work():
    from app import ab_testing as ab

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert ab.ABTest(a, b).perform_t_test()["p_value"] is not None
    assert ab.BootstrapABTest(a, b).get_bootstrap_statistics(n_bootstrap=2000)["p_value"] is not None
    assert 0 <= ab.BayesianABTest(10, 2, 10, 3).compute_prob_B_better() <= 1

