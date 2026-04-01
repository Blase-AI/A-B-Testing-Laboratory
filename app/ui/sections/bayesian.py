from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from app.ab_testing import BayesianABTest


def render_bayesian(data_a: np.ndarray, data_b: np.ndarray, *, alpha_prior: float, beta_prior: float, logger: logging.Logger) -> None:
    st.header("Bayesian Analysis")
    with st.spinner("Performing Bayesian analysis..."):
        n_a, success_a = len(data_a), int(np.sum(data_a > np.mean(data_a)))
        n_b, success_b = len(data_b), int(np.sum(data_b > np.mean(data_b)))

        bab = BayesianABTest(
            n_a,
            success_a,
            n_b,
            success_b,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )
        prob = bab.compute_prob_B_better()
        diff_stats = bab.compute_difference_stats()

        cols = st.columns(3)
        cols[0].metric("Probability B > A", f"{prob*100:.1f}%")
        cols[1].metric("HDI Lower", f"{diff_stats['hdi_interval'][0]:.3f}")
        cols[2].metric("HDI Upper", f"{diff_stats['hdi_interval'][1]:.3f}")

        logger.info("Bayesian: P(B>A)=%.4f", prob)

        with st.expander("Posterior Distributions"):
            st.pyplot(bab.plot_posteriors())

        with st.expander("Boxplot"):
            st.pyplot(bab.plot_boxplot())

        with st.expander("KDE for Posterior Distributions"):
            st.pyplot(bab.plot_kde())

        with st.expander("KDE Plot for Difference Between Groups"):
            st.pyplot(bab.plot_difference_hist_kde())

        with st.expander("CDF Plot for Difference Between Groups"):
            st.pyplot(bab.plot_difference_cdf())

