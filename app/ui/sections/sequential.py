from __future__ import annotations

import logging

import streamlit as st

from app.ab_testing import NormalSPRTSimulator, plot_sprt_history


def render_sequential(*, stop_threshold: int, logger: logging.Logger) -> None:
    st.header("Sequential Analysis")

    with st.expander("Simulation Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            mu0 = st.number_input("μ for H0", value=0.0)
            mu1 = st.number_input("μ for H1", value=1.0)
        with col2:
            sigma = st.number_input("σ", value=1.0, min_value=0.1)
            n = st.number_input("Max observations", value=100, min_value=10)
        with col3:
            true_state = st.radio("True hypothesis", ["H0", "H1"])
            alpha = st.slider("Significance level", 0.01, 0.2, 0.05)

    if st.button("Start Simulation"):
        with st.spinner("Running simulation..."):
            simulator = NormalSPRTSimulator(
                mu0=float(mu0),
                mu1=float(mu1),
                sigma=float(sigma),
                n=int(n),
                alpha=float(alpha),
                true_state=str(true_state),
                stop_threshold=int(stop_threshold),
            )

            result = simulator.run()
            st.subheader("Test Results")
            cols = st.columns(3)
            cols[0].metric("Decision", result.decision)
            cols[1].metric("Observations Used", result.n_used)
            cols[2].metric("Efficiency", f"{(result.n_used/int(n))*100:.1f}%")

            st.pyplot(plot_sprt_history(result))

            sim_results = simulator.run_simulations(n_simulations=50)
            st.subheader("Aggregated Simulation Results")
            st.bar_chart(sim_results["decision_counts"])

            logger.info("Sequential: decision=%s n_used=%s", result.decision, result.n_used)
