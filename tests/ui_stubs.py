from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any


class _FakeColumn:
    def metric(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None


class FakeStreamlit:
    """
    Minimal Streamlit stub for unit-testing render_* functions.

    It intentionally ignores most arguments and only provides the attributes used by our UI modules.
    """

    def __init__(self, *, radio_value: str | None = None):
        self._radio_value = radio_value

    def header(self, *args: Any, **kwargs: Any) -> None:
        return None

    def subheader(self, *args: Any, **kwargs: Any) -> None:
        return None

    def write(self, *args: Any, **kwargs: Any) -> None:
        return None

    def metric(self, *args: Any, **kwargs: Any) -> None:
        return None

    def caption(self, *args: Any, **kwargs: Any) -> None:
        return None

    def warning(self, *args: Any, **kwargs: Any) -> None:
        return None

    def success(self, *args: Any, **kwargs: Any) -> None:
        return None

    def error(self, *args: Any, **kwargs: Any) -> None:
        return None

    def pyplot(self, *args: Any, **kwargs: Any) -> None:
        return None

    def columns(
        self, n: int | list[int] | tuple[int, ...], *args: Any, **kwargs: Any
    ) -> list[_FakeColumn]:
        count = n if isinstance(n, int) else len(n)
        return [_FakeColumn() for _ in range(count)]

    @contextmanager
    def spinner(self, *args: Any, **kwargs: Any):
        yield None

    @contextmanager
    def expander(self, *args: Any, **kwargs: Any):
        yield None

    def radio(self, _label: str, options: list[str], *args: Any, **kwargs: Any) -> str:
        if self._radio_value is not None:
            return self._radio_value
        return options[0]

    def cache_data(self, func: Callable[..., Any] | None = None, *args: Any, **kwargs: Any):
        # Decorator passthrough.
        if func is None:
            return lambda f: f
        return func
