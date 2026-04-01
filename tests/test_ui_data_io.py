import logging

import numpy as np


def test_parse_manual_csv_floats_happy_path():
    from app.ui.data_io import parse_manual_csv_floats

    arr = parse_manual_csv_floats("1, 2, 3.5", "A", logging.getLogger("test"))
    assert arr is not None
    assert np.allclose(arr, np.array([1.0, 2.0, 3.5]))


def test_parse_manual_csv_floats_invalid_returns_none(monkeypatch):
    # Patch streamlit BEFORE importing the module (it imports `streamlit as st` at import time).
    import importlib
    import sys

    from tests.ui_stubs import FakeStreamlit

    monkeypatch.setitem(sys.modules, "streamlit", FakeStreamlit())

    import app.ui.data_io as data_io

    importlib.reload(data_io)
    arr = data_io.parse_manual_csv_floats("1, a, 2", "A", logging.getLogger("test"))
    assert arr is None


def test_demo_continuous_groups_shapes():
    from app.ui.data_io import demo_continuous_groups

    d = demo_continuous_groups(n=123)
    assert d.a.shape == (123,)
    assert d.b.shape == (123,)
