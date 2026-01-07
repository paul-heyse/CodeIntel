"""Error adapters for rustworkx execution."""

from __future__ import annotations

from collections.abc import Callable

import rustworkx as rx
from rustworkx.visit import PruneSearch, StopSearch

RX_EXCEPTIONS = (
    rx.DAGHasCycle,
    rx.DAGWouldCycle,
    rx.FailedToConverge,
    rx.GraphNotBipartite,
    rx.InvalidMapping,
    rx.InvalidNode,
    rx.JSONDeserializationError,
    rx.JSONSerializationError,
    rx.NegativeCycle,
    rx.NoEdgeBetweenNodes,
    rx.NoPathFound,
    rx.NoSuitableNeighbors,
    rx.NullGraph,
    PruneSearch,
    StopSearch,
)


class RxGraphError(RuntimeError):
    """Exception wrapper that preserves rustworkx error context."""

    kind: str
    operation: str
    cause: BaseException | None

    def __init__(
        self,
        *,
        kind: str,
        operation: str,
        message: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.operation = operation
        self.cause = cause


def run_rx[T](operation: str, func: Callable[[], T]) -> T:
    """Execute a rustworkx callable and normalize expected errors.

    Returns
    -------
    T
        Result of the callable when it succeeds.

    Raises
    ------
    RxGraphError
        If rustworkx raises an expected error type.
    """
    try:
        return func()
    except RX_EXCEPTIONS as exc:
        message = f"rustworkx error during {operation}: {type(exc).__name__}"
        raise RxGraphError(
            kind=type(exc).__name__,
            operation=operation,
            message=message,
            cause=exc,
        ) from exc


__all__ = ["RxGraphError", "run_rx"]
