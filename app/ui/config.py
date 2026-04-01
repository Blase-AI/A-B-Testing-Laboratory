import logging

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def setup_app() -> logging.Logger:
    """
    One place for Streamlit + plotting configuration.
    Returns module-level logger to reuse across UI modules.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    st.set_page_config(page_title="A/B Testing Lab", layout="wide")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    return logging.getLogger("ab_testing_lab")

