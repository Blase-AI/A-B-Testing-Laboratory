from __future__ import annotations

import numpy as np


def validate_numeric_1d(data: np.ndarray | list, *, name: str) -> np.ndarray:
    arr = np.array(data)
    if arr.size == 0:
        raise ValueError(f"Data for {name} is empty.")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"Data for {name} must be numeric.")
    return arr.astype(float, copy=False).flatten()
