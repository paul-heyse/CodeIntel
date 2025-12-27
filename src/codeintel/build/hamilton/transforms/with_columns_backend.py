"""Backend selector for Hamilton with_columns decorators."""

from __future__ import annotations

from collections.abc import Callable

from hamilton.function_modifiers.base import NodeTransformLifecycle
from hamilton.plugins.h_pandas import with_columns as with_columns_pd
from hamilton.plugins.h_polars import with_columns as with_columns_pl
from hamilton.plugins.h_polars_lazyframe import with_columns as with_columns_pl_lazy

WithColumnsFactory = Callable[..., NodeTransformLifecycle]


def select_with_columns(df_backend: str) -> WithColumnsFactory:
    """Select the correct with_columns decorator for the configured backend.

    Returns
    -------
    WithColumnsFactory
        Decorator factory for the configured backend.

    Raises
    ------
    ValueError
        If the backend name is not supported.
    """
    if df_backend == "pandas":
        return with_columns_pd
    if df_backend == "polars":
        return with_columns_pl
    if df_backend == "polars_lazy":
        return with_columns_pl_lazy
    msg = f"Unsupported df_backend={df_backend!r}"
    raise ValueError(msg)


__all__ = ["WithColumnsFactory", "select_with_columns"]
