"""Backend selector for Hamilton with_columns decorators."""

from __future__ import annotations

from typing import Callable

from hamilton.plugins.h_pandas import with_columns as with_columns_pd
from hamilton.plugins.h_polars import with_columns as with_columns_pl
from hamilton.plugins.h_polars_lazyframe import with_columns as with_columns_pl_lazy


WithColumnsFactory = Callable[..., Callable[..., object]]


def select_with_columns(df_backend: str) -> WithColumnsFactory:
    """Select the correct with_columns decorator for the configured backend."""
    if df_backend == "pandas":
        return with_columns_pd
    if df_backend == "polars":
        return with_columns_pl
    if df_backend == "polars_lazy":
        return with_columns_pl_lazy
    msg = f"Unsupported df_backend={df_backend!r}"
    raise ValueError(msg)


__all__ = ["WithColumnsFactory", "select_with_columns"]
