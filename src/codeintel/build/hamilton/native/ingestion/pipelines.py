"""Ingestion preprocessing pipelines for Hamilton DAGs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from hamilton.function_modifiers import pipe_input, resolve_from_config, step, value

DecoratorFactory = Callable[[Callable[..., object]], Callable[..., object]]


def _drop_null_rows(
    rows: tuple[tuple[object, ...], ...],
    *,
    required_indices: tuple[int, ...],
) -> tuple[tuple[object, ...], ...]:
    if not required_indices:
        return rows
    return tuple(
        row
        for row in rows
        if all(row[idx] is not None for idx in required_indices)
    )


def _pipe_ingest_rows(
    clean_mode: str,
    *,
    required_indices: tuple[int, ...],
) -> DecoratorFactory:
    if clean_mode == "off":
        return lambda fn: fn
    return pipe_input(
        step(_drop_null_rows, required_indices=value(required_indices)).when(clean_mode="strict"),
        on_input="rows",
        namespace="prep",
    )


def pipe_ingest_rows(
    *,
    required_indices: Sequence[int] = (),
) -> DecoratorFactory:
    """Return a resolve_from_config pipe_input decorator for ingestion rows.

    Parameters
    ----------
    required_indices
        Tuple of column indices required for strict-mode filtering.

    Returns
    -------
    DecoratorFactory
        Decorator applying ingestion row cleanup steps.
    """
    required_tuple = tuple(int(idx) for idx in required_indices)

    def _factory(clean_mode: str) -> DecoratorFactory:
        return _pipe_ingest_rows(clean_mode, required_indices=required_tuple)

    return resolve_from_config(decorate_with=_factory)


__all__ = ["pipe_ingest_rows"]
