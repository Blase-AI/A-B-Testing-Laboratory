from __future__ import annotations

import numpy as np
import streamlit as st

from app.ui.config import setup_app
from app.ui.data_io import demo_continuous_groups, load_csv_first_column, parse_manual_csv_floats
from app.ui.guidelines import show_guidelines
from app.ui.sections.bayesian import render_bayesian
from app.ui.sections.bootstrap import render_bootstrap
from app.ui.sections.classic import render_classic_ab
from app.ui.sections.comparison import render_experiment_comparison
from app.ui.sections.sequential import render_sequential


def _warn_small_samples(data_a: np.ndarray, data_b: np.ndarray, *, threshold: int = 10) -> None:
    if len(data_a) < threshold or len(data_b) < threshold:
        st.warning("Recommended to use samples with ≥10 observations")


def _two_group_input_block(
    *, data_input_method: str, logger
) -> tuple[np.ndarray | None, np.ndarray | None]:
    data_a, data_b = None, None

    with st.expander("⚙️ Data Settings", expanded=True):
        if data_input_method == "Demo Data":
            demo = demo_continuous_groups()
            data_a, data_b = demo.a, demo.b
            st.info("Using demo data: Group A ~ N(50,10), Group B ~ N(55,10)")
            logger.info("Demo data used for groups A and B")
        elif data_input_method == "Upload CSV":
            col1, col2 = st.columns(2)
            with col1:
                a_file = st.file_uploader("CSV for group A", type=["csv"])
                data_a = load_csv_first_column(a_file, "A", logger)
            with col2:
                b_file = st.file_uploader("CSV for group B", type=["csv"])
                data_b = load_csv_first_column(b_file, "B", logger)
        elif data_input_method == "Manual Input":
            col1, col2 = st.columns(2)
            with col1:
                raw_a = st.text_area("Data for group A (comma-separated)", "50, 55, 60")
                data_a = parse_manual_csv_floats(raw_a, "A", logger)
            with col2:
                raw_b = st.text_area("Data for group B (comma-separated)", "55, 60, 65")
                data_b = parse_manual_csv_floats(raw_b, "B", logger)

    if data_a is not None and data_b is not None:
        _warn_small_samples(data_a, data_b)

    return data_a, data_b


def main() -> None:
    logger = setup_app()

    st.title("A/B Testing Laboratory")
    st.sidebar.header("Experiment Settings")

    test_type = st.sidebar.selectbox(
        "Test Type",
        ["Classic A/B", "Bayesian", "Bootstrap", "Sequential", "Experiment Comparison"],
    )

    st.sidebar.header("Test Parameters")

    # Parameters per test type (avoid undefined variables bugs).
    alpha = None
    min_sample_size = None
    alpha_prior = None
    beta_prior = None
    n_bootstrap = None
    ci_level = None
    stop_threshold = None

    if test_type == "Classic A/B":
        alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.2, 0.05, key="alpha_classic")
        min_sample_size = st.sidebar.number_input(
            "Minimum sample size", 10, 1000, 30, key="min_sample"
        )
    elif test_type == "Bayesian":
        col1, col2 = st.sidebar.columns(2)
        alpha_prior = col1.number_input("Alpha prior", 0.1, 10.0, 1.0, key="alpha_prior")
        beta_prior = col2.number_input("Beta prior", 0.1, 10.0, 1.0, key="beta_prior")
    elif test_type == "Bootstrap":
        n_bootstrap = st.sidebar.number_input(
            "Number of iterations", 100, 100000, 10000, key="n_bootstrap"
        )
        ci_level = st.sidebar.slider("Confidence level", 0.8, 0.99, 0.95, key="ci_level")
    elif test_type == "Sequential":
        stop_threshold = st.sidebar.number_input("Stop threshold", 1, 100, 10, key="stop_threshold")
    elif test_type == "Experiment Comparison":
        alpha = st.sidebar.slider("Significance level (α)", 0.01, 0.2, 0.05, key="alpha_compare")

    if test_type != "Sequential":
        data_input_method = st.sidebar.radio(
            "Data input method:", ["Demo Data", "Upload CSV", "Manual Input"]
        )
        data_a, data_b = _two_group_input_block(data_input_method=data_input_method, logger=logger)
    else:
        data_a = data_b = None

    try:
        if test_type == "Classic A/B" and data_a is not None and data_b is not None:
            assert alpha is not None
            assert min_sample_size is not None
            render_classic_ab(
                data_a,
                data_b,
                alpha=float(alpha),
                min_sample_size=int(min_sample_size),
                logger=logger,
            )
        elif test_type == "Bayesian" and data_a is not None and data_b is not None:
            assert alpha_prior is not None
            assert beta_prior is not None
            render_bayesian(
                data_a,
                data_b,
                alpha_prior=float(alpha_prior),
                beta_prior=float(beta_prior),
                logger=logger,
            )
        elif test_type == "Bootstrap" and data_a is not None and data_b is not None:
            assert n_bootstrap is not None
            assert ci_level is not None
            render_bootstrap(
                data_a,
                data_b,
                n_bootstrap=int(n_bootstrap),
                ci_level=float(ci_level),
                logger=logger,
            )
        elif test_type == "Sequential":
            assert stop_threshold is not None
            render_sequential(stop_threshold=int(stop_threshold), logger=logger)
        elif test_type == "Experiment Comparison":
            assert alpha is not None
            render_experiment_comparison(alpha=float(alpha), logger=logger)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.exception("Unhandled error in Streamlit UI")
        st.error("Check the correctness of input data and parameters")

    st.markdown("---")
    col_help = st.columns([1, 2, 1])
    with col_help[1]:
        if st.button(
            "How to interpret results?",
            help="Open guidelines for testing methods and interpretation",
            use_container_width=True,
        ):
            try:
                st.switch_page("pages/Glossary.py")
            except Exception:
                show_guidelines()


if __name__ == "__main__":
    main()
