"""Stable wrappers for columnar transform modifiers."""

from __future__ import annotations

from hamilton.function_modifiers import pipe_input, pipe_output
from hamilton.plugins.h_polars_lazyframe import with_columns as with_columns_lazy

__all__ = [
    "pipe_input",
    "pipe_output",
    "with_columns_lazy",
]
