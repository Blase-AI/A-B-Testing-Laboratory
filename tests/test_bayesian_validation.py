import pytest


def test_bayesian_rejects_invalid_counts():
    from app.ab_testing import BayesianABTest

    with pytest.raises(ValueError):
        BayesianABTest(0, 0, 10, 1)

    with pytest.raises(ValueError):
        BayesianABTest(10, -1, 10, 1)

    with pytest.raises(ValueError):
        BayesianABTest(10, 11, 10, 1)

    with pytest.raises(ValueError):
        BayesianABTest(10, 1, 10, 11)
