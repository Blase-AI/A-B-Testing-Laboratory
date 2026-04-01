from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from app.ab_testing import BootstrapABTest


def render_bootstrap(
    data_a: np.ndarray,
    data_b: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
    logger: logging.Logger,
) -> None:
    st.header("Bootstrap Analysis")
    with st.spinner("Performing bootstrap analysis..."):
        boot = BootstrapABTest(data_a, data_b)

        with st.expander("Results"):
            means_a, means_b = boot.compute_bootstrap_means(n_bootstrap=n_bootstrap)
            ci_a = boot.percentile_ci(means_a, ci=ci_level * 100)
            ci_b = boot.percentile_ci(means_b, ci=ci_level * 100)
            diff = boot.compute_bootstrap_diff(n_bootstrap=n_bootstrap)
            ci_diff = boot.percentile_ci(diff, ci=ci_level * 100)

            cols = st.columns(3)
            cols[0].metric("CI Group A", f"{ci_a[0]:.2f} - {ci_a[1]:.2f}")
            cols[1].metric("CI Group B", f"{ci_b[0]:.2f} - {ci_b[1]:.2f}")
            cols[2].metric("Difference (B-A)", f"{ci_diff[0]:.2f} - {ci_diff[1]:.2f}")

            logger.info("Bootstrap: n=%s ci=%.2f", n_bootstrap, ci_level)

        with st.expander("Distribution of the difference in averages"):
            st.pyplot(boot.plot_bootstrap_distributions(n_bootstrap=n_bootstrap))

        with st.expander("KDE Plot"):
            st.pyplot(boot.plot_bootstrap_diff_hist(n_bootstrap=n_bootstrap))

        with st.expander("ECDF Plot"):
            st.pyplot(boot.plot_bootstrap_diff_ecdf(n_bootstrap=n_bootstrap))

