"""Ingestion preprocessing pipelines for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl
from hamilton.function_modifiers import pipe_input, resolve_from_config, step, value
from hamilton.function_modifiers.base import NodeTransformLifecycle


def _drop_null_rows(
    frame: pl.LazyFrame,
    *,
    required_cols: tuple[str, ...],
) -> pl.LazyFrame:
    if not required_cols:
        return frame
    return frame.drop_nulls(list(required_cols))


def _pipe_ingest_rows(
    _clean_mode: str,
    *,
    required_cols: tuple[str, ...],
    input_name: str,
) -> NodeTransformLifecycle:
    return pipe_input(
        step(_drop_null_rows, required_cols=value(required_cols)).when(clean_mode="strict"),
        on_input=input_name,
        namespace="prep",
    )


def pipe_ingest_rows(
    *,
    required_cols: Sequence[str] = (),
    input_name: str = "rows",
) -> NodeTransformLifecycle:
    """Return a resolve_from_config pipe_input decorator for ingestion frames.

    Parameters
    ----------
    required_cols
        Column names required for strict-mode filtering.
    input_name
        Input parameter name to target for ingestion row cleanup.

    Returns
    -------
    DecoratorFactory
        Decorator applying ingestion row cleanup steps.
    """
    required_tuple = tuple(str(name) for name in required_cols)

    def _factory(clean_mode: str) -> NodeTransformLifecycle:
        return _pipe_ingest_rows(
            clean_mode,
            required_cols=required_tuple,
            input_name=input_name,
        )

    return resolve_from_config(decorate_with=_factory)


__all__ = ["pipe_ingest_rows"]
