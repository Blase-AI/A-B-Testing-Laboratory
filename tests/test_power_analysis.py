import numpy as np


def test_perform_power_analysis_returns_valid_outputs():
    from app.ab_testing import perform_power_analysis

    res = perform_power_analysis(effect_size=0.5, alpha=0.05, nobs1=100, desired_power=0.8)
    assert 0.0 <= float(res["current_power"]) <= 1.0
    assert float(res["required_n"]) > 0
    assert res["figure"] is not None


def test_perform_power_analysis_respects_sample_sizes():
    from app.ab_testing import perform_power_analysis

    sample_sizes = np.array([50, 100, 150])
    res = perform_power_analysis(
        effect_size=0.2, alpha=0.05, nobs1=50, desired_power=0.8, sample_sizes=sample_sizes
    )
    assert res["figure"] is not None
