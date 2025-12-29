"""Shared materialization pipeline helpers."""

from __future__ import annotations

from collections.abc import Callable
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


def run_materialization_pipeline(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
    table_key: str,
    data: object,
    recoverable_exceptions: tuple[type[Exception], ...],
    none_error: str,
    unknown_error: str,
    materialize: MaterializeHandler,
    cleanup: CleanupHandler | None = None,
) -> MaterializationResult:
    """Execute a standard materialization pipeline with shared error handling.

    Parameters
    ----------
    env
        Build environment containing snapshot, gateway, and configuration.
    catalog
        DAG catalog describing build dependencies.
    target_name
        Name of the target being materialized.
    table_key
        Fully qualified table key for the output.
    data
        Data value produced by the upstream compute node.
    recoverable_exceptions
        Exception types that should map to a failed materialization result.
    none_error
        Error message when data is None.
    unknown_error
        Error message when materialization returns no result.
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
            env=env,
            catalog=catalog,
            target_name=target_name,
        )
        if isinstance(prepared, MaterializationContextError):
            return failed_table_result(
                table_key=table_key,
                duration_ms=duration_ms(start),
                input_hash=prepared.input_hash or "",
                error=prepared.message,
            )
        input_hash = prepared.input_hash
        if data is None:
            return failed_table_result(
                table_key=table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=none_error,
            )
        result = materialize(prepared, data, input_hash, start)
        if result is None:
            return failed_table_result(
                table_key=table_key,
                duration_ms=duration_ms(start),
                input_hash=input_hash or "",
                error=unknown_error,
            )
        return result
    except recoverable_exceptions as exc:
        return failed_table_result(
            table_key=table_key,
            duration_ms=duration_ms(start),
            input_hash=input_hash or "",
            error=str(exc),
        )
    finally:
        if cleanup is not None:
            cleanup()
