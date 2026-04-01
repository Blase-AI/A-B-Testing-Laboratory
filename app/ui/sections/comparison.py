from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from app.ab_testing import ABTest
from app.ui.data_io import load_csv_first_column, parse_manual_csv_floats
from app.ui.results import display_results_table


def render_experiment_comparison(*, alpha: float, logger: logging.Logger) -> None:
    st.header("📊 Multiple Experiment Comparison")
    num_experiments = st.number_input(
        "Number of experiments to compare", min_value=2, max_value=10, value=2, step=1
    )
    experiment_results = []

    for i in range(int(num_experiments)):
        st.subheader(f"Experiment {i+1}")
        exp_data_input_method = st.radio(
            f"Data input method for experiment {i+1}:",
            ["Demo Data", "Upload CSV", "Manual Input"],
            key=f"input_method_{i}",
        )

        data_a, data_b = None, None

        if exp_data_input_method == "Demo Data":
            mean_shift = i * 2
            data_a = np.random.normal(50, 10, 1000)
            data_b = np.random.normal(50 + mean_shift + 5, 10, 1000)
            st.info(f"Experiment {i+1}: Group A ~ N(50,10), Group B ~ N({50+mean_shift+5},10)")
            logger.info("Experiment %s: demo data created mean_shift=%s", i + 1, mean_shift)

        elif exp_data_input_method == "Upload CSV":
            col1, col2 = st.columns(2)
            with col1:
                a_file = st.file_uploader(
                    f"CSV for group A of experiment {i+1}",
                    type=["csv"],
                    key=f"file_a_{i}",
                )
                data_a = load_csv_first_column(a_file, f"A (experiment {i+1})", logger)
            with col2:
                b_file = st.file_uploader(
                    f"CSV for group B of experiment {i+1}",
                    type=["csv"],
                    key=f"file_b_{i}",
                )
                data_b = load_csv_first_column(b_file, f"B (experiment {i+1})", logger)

        elif exp_data_input_method == "Manual Input":
            col1, col2 = st.columns(2)
            with col1:
                raw_a = st.text_area(
                    f"Data for group A (comma-separated) for experiment {i+1}",
                    "50, 55, 60",
                    key=f"raw_a_{i}",
                )
                data_a = parse_manual_csv_floats(raw_a, f"A (experiment {i+1})", logger)
            with col2:
                raw_b = st.text_area(
                    f"Data for group B (comma-separated) for experiment {i+1}",
                    "55, 60, 65",
                    key=f"raw_b_{i}",
                )
                data_b = parse_manual_csv_floats(raw_b, f"B (experiment {i+1})", logger)

        if data_a is None or data_b is None:
            continue

        if len(data_a) < 10 or len(data_b) < 10:
            st.warning("⚠️ Recommended to use samples with ≥10 observations")
            logger.warning(
                "Experiment %s: insufficient observations A=%s B=%s",
                i + 1,
                len(data_a),
                len(data_b),
            )
            continue

        ab = ABTest(data_a, data_b)
        test_res = ab.perform_t_test(alpha=alpha)
        cohens_d = ab.calculate_cohens_d()
        result_summary = {
            "Experiment": f"Experiment {i+1}",
            "Mean A": float(np.mean(data_a)),
            "Mean B": float(np.mean(data_b)),
            "Difference": float(np.mean(data_b) - np.mean(data_a)),
            "p-value": float(test_res["p_value"]),
            "Cohen's d": float(cohens_d),
        }
        experiment_results.append(result_summary)
        st.success(f"Experiment {i+1} completed successfully")
        logger.info("Experiment %s: p=%.4f d=%.3f", i + 1, test_res["p_value"], cohens_d)

    if not experiment_results:
        return

    display_results_table(experiment_results)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=pd.DataFrame(experiment_results), x="Experiment", y="p-value", marker="o", ax=ax
    )
    ax.set_title("p-value Dynamics Across Experiments")
    st.pyplot(fig)
