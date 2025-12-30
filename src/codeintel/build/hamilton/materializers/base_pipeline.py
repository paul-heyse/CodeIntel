"""Shared materialization pipeline helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers.base import (
    MaterializationContext,
    MaterializationContextError,
    duration_ms,
    resolve_materialization_context,
)
from codeintel.core.execution.materialization import failed_table_result

type CleanupHandler = Callable[[], None]


type MaterializeHandler = Callable[
    [MaterializationContext, object, str | None, float],
    MaterializationResult,
]


@dataclass(frozen=True, slots=True)
class MaterializationPipelineInput:
    """Input payload for running a materialization pipeline."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    table_key: str
    data: object
    recoverable_exceptions: tuple[type[Exception], ...]
    none_error: str
    unknown_error: str


def run_materialization_pipeline(
    *,
    payload: MaterializationPipelineInput,
    materialize: MaterializeHandler,
    cleanup: CleanupHandler | None = None,
) -> MaterializationResult:
    """Execute a standard materialization pipeline with shared error handling.

    Parameters
    ----------
    payload
        Input payload describing materialization context and errors.
    materialize
        Callback that performs the actual write and returns a result.
    cleanup
        Optional cleanup callback executed regardless of success or failure.

    Returns
    -------
    MaterializationResult
        Materialization result describing success or failure.
    """
    start = perf_counter()
    input_hash: str | None = None
    try:
        prepared = resolve_materialization_context(
            env=payload.env,
            catalog=payload.catalog,
            target_name=payload.target_name,
        )
        if isinstance(prepared, MaterializationContextError):
            return failed_table_result(
                table_key=payload.table_key,
                duration_ms=duration_ms(start),
                input_hash=prepared.input_hash or "",
                error=prepared.message,
            )
        input_hash = prepared.input_hash
        if payload.data is None:
            return failed_table_result(
                table_key=payload.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=payload.none_error,
            )
        result = materialize(prepared, payload.data, input_hash, start)
    except payload.recoverable_exceptions as exc:
        return failed_table_result(
            table_key=payload.table_key,
            duration_ms=duration_ms(start),
            input_hash=input_hash or "",
            error=str(exc),
        )
    else:
        if result is None:
            return failed_table_result(
                table_key=payload.table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=payload.unknown_error,
            )
        return result
    finally:
        if cleanup is not None:
            cleanup()
