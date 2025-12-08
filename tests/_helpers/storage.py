"""Storage-focused test helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from codeintel.storage.gateway.protocol import DuckDBConnection

__all__ = ["CapturedInsert", "capture_executor"]


@dataclass(frozen=True)
class CapturedInsert:
    """Captured insert invocation."""

    table: str
    rows: list[tuple[object, ...]]


def capture_executor() -> tuple[
    Callable[[DuckDBConnection, str, Iterable[tuple[object, ...]]], None],
    list[CapturedInsert],
]:
    """Create an executor that records table keys and rows.

    Returns
    -------
    tuple[Callable[..., None], list[CapturedInsert]]
        Executor callable and the list that will collect captured invocations.
    """
    calls: list[CapturedInsert] = []

    def _executor(
        _con: DuckDBConnection,
        table_key: str,
        rows: Iterable[tuple[object, ...]],
    ) -> None:
        calls.append(CapturedInsert(table=table_key, rows=list(rows)))

    return _executor, calls
