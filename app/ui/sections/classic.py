from __future__ import annotations

import logging

import numpy as np
import streamlit as st

from app.ab_testing import ABTest


def render_classic_ab(data_a: np.ndarray, data_b: np.ndarray, *, alpha: float, min_sample_size: int, logger: logging.Logger) -> None:
    st.header("Classic A/B Test")
    with st.spinner("Performing analysis..."):
        ab = ABTest(data_a, data_b)

        with st.expander("Descriptive Statistics", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Mean A", f"{np.mean(data_a):.2f}")
            cols[1].metric("Mean B", f"{np.mean(data_b):.2f}")
            cols[2].metric("Difference", f"{np.mean(data_b) - np.mean(data_a):.2f}")

        with st.expander("Statistical Analysis"):
            norm = ab.test_normality()
            test_res = ab.perform_t_test(alpha=alpha)
            col1, col2 = st.columns(2)
            col1.metric(
                "Normality A",
                "✅" if norm["Group A"]["normal"] else "❌",
                f"p={norm['Group A']['p_value']:.4f}",
            )
            col2.metric(
                "Normality B",
                "✅" if norm["Group B"]["normal"] else "❌",
                f"p={norm['Group B']['p_value']:.4f}",
            )
            st.subheader("Test Results")
            st.write(f"**Method:** {test_res['test']}")
            st.metric("p-value", f"{test_res['p_value']:.4f}", help="Significance of differences")

            d = ab.calculate_cohens_d()
            effect_size = "small" if abs(d) < 0.2 else "medium" if abs(d) < 0.5 else "large"
            st.metric("Cohen's d", f"{d:.3f} ({effect_size} effect)")

            st.write("Used parameters:")
            st.write(f"- Significance level: {alpha}")
            st.write(f"- Minimum sample size: {min_sample_size}")

            logger.info("Classic A/B: test=%s p=%s d=%s", test_res.get("test"), test_res.get("p_value"), d)

        with st.expander("Distribution Visualization"):
            st.pyplot(ab.plot_distribution())

        with st.expander("Boxplot"):
            st.pyplot(ab.plot_boxplot())

        with st.expander("Mean Difference Plot with Confidence Interval"):
            st.pyplot(ab.plot_mean_difference_ci(confidence=0.95))

        with st.expander("Q-Q Plot for Normality Check"):
            st.pyplot(ab.plot_qq())

