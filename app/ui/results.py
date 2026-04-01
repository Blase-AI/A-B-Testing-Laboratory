from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd
import streamlit as st


def display_results_table(results: Iterable[Mapping[str, Any]]) -> None:
    """Display a summary table of experiment results and allow downloading as CSV."""
    df_results = pd.DataFrame(results)
    st.subheader("Summary Table of Experiment Results")
    st.dataframe(df_results)
    csv = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results", data=csv, file_name="experiment_results.csv", mime="text/csv"
    )
