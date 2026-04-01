from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.validators import validate_numeric_1d


class BaseABTest(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def summary(self) -> dict[str, Any]: ...


@dataclass
class TwoGroupContinuousABTest(BaseABTest):
    data_a: np.ndarray
    data_b: np.ndarray

    def __post_init__(self) -> None:
        self.data_a = validate_numeric_1d(self.data_a, name="A")
        self.data_b = validate_numeric_1d(self.data_b, name="B")
