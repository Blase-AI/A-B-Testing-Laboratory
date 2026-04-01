import pytest


def test_validate_numeric_1d_accepts_list_and_flattens():
    from app.ab_testing import validate_numeric_1d

    arr = validate_numeric_1d([[1, 2], [3, 4]], name="X")
    assert arr.shape == (4,)
    assert arr.dtype.kind == "f"


def test_validate_numeric_1d_rejects_empty():
    from app.ab_testing import validate_numeric_1d

    with pytest.raises(ValueError):
        validate_numeric_1d([], name="X")


def test_validate_numeric_1d_rejects_non_numeric():
    from app.ab_testing import validate_numeric_1d

    with pytest.raises(ValueError):
        validate_numeric_1d(["a", "b"], name="X")
