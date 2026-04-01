from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from tests.ui_stubs import FakeStreamlit


def _patch_streamlit(monkeypatch, *, radio_value: str | None = None) -> None:
    fake = FakeStreamlit(radio_value=radio_value)
    # Ensure `import streamlit as st` resolves to our stub for subsequent imports.
    monkeypatch.setitem(sys.modules, "streamlit", fake)


def test_render_classic_smoke(monkeypatch):
    _patch_streamlit(monkeypatch)
    from app.ui.sections.classic import render_classic_ab

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.2, 2.2, 3.2, 4.2])
    render_classic_ab(a, b, alpha=0.05, min_sample_size=10, logger=logging.getLogger("test"))
    plt.close("all")


def test_render_bootstrap_smoke(monkeypatch):
    _patch_streamlit(monkeypatch)
    from app.ui.sections.bootstrap import render_bootstrap

    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.1, 2.1, 3.1, 4.1])
    render_bootstrap(a, b, n_bootstrap=500, ci_level=0.95, logger=logging.getLogger("test"))
    plt.close("all")


def test_render_bayesian_smoke_binary_mode(monkeypatch):
    _patch_streamlit(monkeypatch, radio_value="Binary metric (0/1)")
    from app.ui.sections.bayesian import render_bayesian

    a = np.array([0, 1, 0, 1, 1])
    b = np.array([0, 1, 1, 1, 1])
    render_bayesian(a, b, alpha_prior=1.0, beta_prior=1.0, logger=logging.getLogger("test"))
    plt.close("all")
