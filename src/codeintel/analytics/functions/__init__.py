"""Function-level analytics public API.

This module centralizes the main entrypoints for per-function analytics so
callers do not need to import individual implementation modules.

For typedness utilities (ParamStats, TypednessFlags, compute_param_stats,
compute_typedness_flags), import directly from:
    codeintel.analytics.compute.functions.typedness

For Hamilton native execution, use ``build_function_history_rows`` to get row
tuples, then materialize with ``materialize_rows``.

The Hamilton native module for function history is at:
``codeintel.build.hamilton.native.analytics.function_history``
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, cast

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.function_history import (
    FUNCTION_HISTORY_COLS,
    build_function_history_rows,
)
from codeintel.analytics.utilities.lazy_module import make_lazy_getattr

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.analytics.functions.function_effects import (
        FunctionEffectsInputs,
        FunctionEffectsOptions,
    )
    from codeintel.analytics.functions.metrics import (
        FunctionAnalyticsResult as _FunctionAnalyticsResult,
    )
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "FUNCTION_HISTORY_COLS",
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsResult",
    "build_function_history_rows",
    "compute_function_analytics_result",
    "compute_function_contracts",
    "compute_function_effects",
    "compute_function_metrics_and_types",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "compute_function_analytics_result": (
        "codeintel.analytics.functions.metrics",
        "compute_function_analytics_result",
    ),
    "compute_function_contracts": (
        "codeintel.analytics.functions.function_contracts",
        "compute_function_contracts",
    ),
    "compute_function_effects": (
        "codeintel.analytics.functions.function_effects",
        "compute_function_effects",
    ),
    "compute_function_metrics_and_types": (
        "codeintel.analytics.functions.metrics",
        "compute_function_metrics_and_types",
    ),
    "FunctionAnalyticsResult": (
        "codeintel.analytics.functions.metrics",
        "FunctionAnalyticsResult",
    ),
}


def _load(name: str) -> Callable[..., object]:
    module_path, attr_name = _LAZY_ATTRS[name]
    return getattr(import_module(module_path), attr_name)


def compute_function_contracts(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    function_ast_map: dict[int, FunctionAst] | None = None,
    catalog: FunctionCatalogProvider | None = None,
    max_conditions_per_func: int = 64,
) -> None:
    """Compute contract coverage for functions using the configured backend.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    function_ast_map
        Mapping of GOID to parsed function AST (from AstProvider).
    catalog
        Function catalog provider (from CatalogProvider).
    max_conditions_per_func
        Maximum number of preconditions/postconditions/raises per function.
    """
    func = cast(
        "Callable[..., None]",
        _load("compute_function_contracts"),
    )
    return func(
        gateway,
        snapshot,
        function_ast_map=function_ast_map,
        catalog=catalog,
        max_conditions_per_func=max_conditions_per_func,
    )


def compute_function_effects(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: FunctionEffectsOptions | None = None,
    inputs: FunctionEffectsInputs | None = None,
) -> None:
    """Compute function side effects and control-flow metadata.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    options
        Configuration options for effects detection.
    inputs
        Optional inputs containing catalog, runtime, AST map, and missing GOIDs.
    """
    func = cast(
        "Callable[..., None]",
        _load("compute_function_effects"),
    )
    return func(gateway, snapshot, options=options, inputs=inputs)


def compute_function_analytics_result(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> _FunctionAnalyticsResult:
    """Compute pure function analytics result without persisting.

    This is the pure compute path for Hamilton DAG-visible I/O. It returns
    rows ready for materialization via SaveToDecorator/DuckDBRowsSaver.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    options
        Optional configuration for metrics computation.

    Returns
    -------
    FunctionAnalyticsResult
        Container with metrics_rows, types_rows, and validation reporter.
    """
    func = cast(
        "Callable[..., _FunctionAnalyticsResult]",
        _load("compute_function_analytics_result"),
    )
    return func(gateway, snapshot, options=options)


def compute_function_metrics_and_types(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    options: FunctionAnalyticsOptions | None = None,
    fail_on_missing_spans: bool = False,
) -> dict[str, int]:
    """Compute combined metrics and typedness for functions.

    Parameters
    ----------
    gateway
        Storage gateway providing DuckDB access.
    snapshot
        Repository and commit identifiers.
    options
        Optional configuration for metrics computation.
    fail_on_missing_spans
        Whether to raise an error if any spans are missing.

    Returns
    -------
    dict[str, int]
        Mapping of output table names to row counts.
    """
    func = cast(
        "Callable[..., dict[str, int]]",
        _load("compute_function_metrics_and_types"),
    )
    return func(gateway, snapshot, options=options, fail_on_missing_spans=fail_on_missing_spans)


# Fallback for any attribute access
__getattr__ = make_lazy_getattr(_LAZY_ATTRS, __name__)


def __dir__() -> list[str]:
    """List public attributes for IDE support.

    Returns
    -------
    list[str]
        Sorted list of public attribute names.
    """
    return sorted(__all__)
