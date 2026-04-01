from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class TwoGroupData:
    a: np.ndarray
    b: np.ndarray


@st.cache_data
def load_csv_first_column(uploader, group_name: str, logger: logging.Logger) -> Optional[np.ndarray]:
    """Load numeric data from the first CSV column."""
    try:
        if uploader is None:
            return None
        df = pd.read_csv(uploader)
        data = [float(x) for x in df.iloc[:, 0].values]
        st.success(f"Data for group {group_name} successfully loaded!")
        logger.info("Loaded CSV data for group %s, count=%s", group_name, len(data))
        return np.array(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        logger.exception("Error loading data for group %s", group_name)
        return None


def parse_manual_csv_floats(raw_input: str, group_name: str, logger: logging.Logger) -> Optional[np.ndarray]:
    """Parse comma-separated floats from textarea input."""
    try:
        values = [float(x.strip()) for x in raw_input.split(",") if x.strip()]
        logger.info("Parsed manual input for group %s, count=%s", group_name, len(values))
        return np.array(values)
    except ValueError as e:
        st.error(f"Invalid input data for group {group_name}: {str(e)}")
        logger.exception("Validation error for manual input for group %s", group_name)
        return None


def demo_continuous_groups(mean_a: float = 50, mean_b: float = 55, sigma: float = 10, n: int = 1000) -> TwoGroupData:
    a = np.random.normal(mean_a, sigma, n)
    b = np.random.normal(mean_b, sigma, n)
    return TwoGroupData(a=a, b=b)

