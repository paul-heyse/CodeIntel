"""Typed Polars collection helpers for streaming execution."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars import DataFrame, LazyFrame

    type PolarsLazyFrame = LazyFrame
    type PolarsDataFrame = DataFrame
else:
    type PolarsLazyFrame = object
    type PolarsDataFrame = object


class CollectKwargs(TypedDict, total=False):
    engine: str
    streaming: bool
    optimization_flags: object
    query_opt_flags: object
    optimizations: object


class CollectBatchesKwargs(CollectKwargs, total=False):
    batch_size: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class PolarsExecutionOptions:
    """Execution options for Polars LazyFrame collection."""

    streaming: bool = True
    query_opt_flags: object | None = None
    inspect: bool = False
    streaming_fallback: bool = True


def collect_lazyframe(
    lazyframe: PolarsLazyFrame,
    *,
    options: PolarsExecutionOptions,
) -> PolarsDataFrame:
    """Collect a LazyFrame with typed option handling.

    Returns
    -------
    polars.DataFrame
        Collected DataFrame.
    """
    collect_target = cast("Callable[..., object]", lazyframe.collect)
    kwargs = _collect_kwargs(collect_target, options=options)
    collect_fn = _collect_callable(lazyframe.collect)
    return collect_fn(**kwargs)


def collect_batches(
    lazyframe: PolarsLazyFrame,
    *,
    batch_size: int,
    options: PolarsExecutionOptions,
) -> Sequence[PolarsDataFrame]:
    """Collect a LazyFrame into batches with typed option handling.

    Returns
    -------
    Sequence[polars.DataFrame]
        Collected DataFrame batches.
    """
    collect_target = cast("Callable[..., object]", lazyframe.collect_batches)
    kwargs = _collect_batches_kwargs(
        collect_target,
        batch_size=batch_size,
        options=options,
    )
    collect_fn = _collect_batches_callable(lazyframe.collect_batches)
    return collect_fn(**kwargs)


def _collect_kwargs(
    func: Callable[..., object],
    *,
    options: PolarsExecutionOptions,
) -> CollectKwargs:
    params = _param_names(func)
    kwargs: CollectKwargs = {}
    _populate_streaming_kwargs(kwargs, params, streaming=options.streaming)
    _populate_opt_flags(kwargs, params, options.query_opt_flags)
    return kwargs


def _collect_batches_kwargs(
    func: Callable[..., object],
    *,
    batch_size: int,
    options: PolarsExecutionOptions,
) -> CollectBatchesKwargs:
    params = _param_names(func)
    kwargs: CollectBatchesKwargs = {}
    if "chunk_size" in params:
        kwargs["chunk_size"] = batch_size
    elif "batch_size" in params:
        kwargs["batch_size"] = batch_size
    _populate_streaming_kwargs(kwargs, params, streaming=options.streaming)
    _populate_opt_flags(kwargs, params, options.query_opt_flags)
    return kwargs


def _populate_streaming_kwargs(
    kwargs: CollectKwargs,
    params: frozenset[str],
    *,
    streaming: bool,
) -> None:
    if "engine" in params:
        if streaming:
            kwargs["engine"] = "streaming"
        return
    if "streaming" in params:
        kwargs["streaming"] = streaming


def _populate_opt_flags(
    kwargs: CollectKwargs,
    params: frozenset[str],
    query_opt_flags: object | None,
) -> None:
    if query_opt_flags is None:
        return
    if "optimization_flags" in params:
        kwargs["optimization_flags"] = query_opt_flags
    elif "query_opt_flags" in params:
        kwargs["query_opt_flags"] = query_opt_flags
    elif "optimizations" in params:
        kwargs["optimizations"] = query_opt_flags


def _param_names(func: Callable[..., object]) -> frozenset[str]:
    try:
        params = signature(func).parameters
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(params)


def _collect_callable(
    func: object,
) -> Callable[..., PolarsDataFrame]:
    return cast("Callable[..., PolarsDataFrame]", func)


def _collect_batches_callable(
    func: object,
) -> Callable[..., Sequence[PolarsDataFrame]]:
    return cast("Callable[..., Sequence[PolarsDataFrame]]", func)


__all__ = ["PolarsExecutionOptions", "collect_batches", "collect_lazyframe"]
