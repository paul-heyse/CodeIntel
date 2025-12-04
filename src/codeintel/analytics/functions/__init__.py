"""Function-level analytics public API.

This module centralizes the main entrypoints for per-function analytics so
callers do not need to import individual implementation modules.

For typedness utilities (ParamStats, TypednessFlags, compute_param_stats,
compute_typedness_flags), import directly from:
    codeintel.analytics.compute.functions.typedness
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.config import FunctionAnalyticsStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.function_ast_cache import FunctionAst
    from codeintel.analytics.functions.function_effects import FunctionEffectsInputs
    from codeintel.config import (
        FunctionContractsStepConfig,
        FunctionEffectsStepConfig,
        FunctionHistoryStepConfig,
    )
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.ingestion.tools.infrastructure import ToolRunner
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "FunctionAnalyticsOptions",
    "FunctionAnalyticsStepConfig",
    "compute_function_contracts",
    "compute_function_effects",
    "compute_function_history",
    "compute_function_metrics_and_types",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "compute_function_contracts": (
        "codeintel.analytics.functions.function_contracts",
        "compute_function_contracts",
    ),
    "compute_function_effects": (
        "codeintel.analytics.functions.function_effects",
        "compute_function_effects",
    ),
    "compute_function_history": (
        "codeintel.analytics.functions.function_history",
        "compute_function_history",
    ),
    "compute_function_metrics_and_types": (
        "codeintel.analytics.functions.metrics",
        "compute_function_metrics_and_types",
    ),
}


def _load(name: str) -> Callable[..., Any]:
    try:
        module_path, attr_name = _LAZY_ATTRS[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg) from exc
    return getattr(import_module(module_path), attr_name)


def compute_function_contracts(
    gateway: StorageGateway,
    cfg: FunctionContractsStepConfig,
    *,
    function_ast_map: dict[int, FunctionAst] | None = None,
    catalog: FunctionCatalogProvider | None = None,
) -> None:
    """Compute contract coverage for functions using the configured backend.

    Returns
    -------
    None
        Results are written to storage; return value is unused.
    """
    func = cast(
        "Callable[..., None]",
        _load("compute_function_contracts"),
    )
    return func(
        gateway,
        cfg,
        function_ast_map=function_ast_map,
        catalog=catalog,
    )


def compute_function_effects(
    gateway: StorageGateway,
    cfg: FunctionEffectsStepConfig,
    *,
    inputs: FunctionEffectsInputs | None = None,
) -> None:
    """Compute function side effects and control-flow metadata.

    Returns
    -------
    None
        Results are persisted; return value is not used.
    """
    func = cast(
        "Callable[..., None]",
        _load("compute_function_effects"),
    )
    return func(gateway, cfg, inputs=inputs)


def compute_function_history(
    gateway: StorageGateway,
    cfg: FunctionHistoryStepConfig,
    *,
    runner: ToolRunner | None = None,
) -> None:
    """Compute historical metrics for functions from SCM or tool outputs.

    Returns
    -------
    None
        This function persists results via storage; return is for parity.
    """
    func = cast(
        "Callable[..., None]",
        _load("compute_function_history"),
    )
    return func(gateway, cfg, runner=runner)


def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsStepConfig,
    *,
    options: FunctionAnalyticsOptions | None = None,
) -> dict[str, int]:
    """Compute combined metrics and typedness for functions.

    Returns
    -------
    dict[str, int]
        Mapping of output table names to row counts.
    """
    func = cast(
        "Callable[..., dict[str, int]]",
        _load("compute_function_metrics_and_types"),
    )
    return func(gateway, cfg, options=options)


def __getattr__(name: str) -> object:
    """Lazily import heavy function analytics entrypoints to avoid cycles.

    Returns
    -------
    object
        Imported attribute from the deferred module.
    """
    return _load(name)


def __dir__() -> list[str]:
    return sorted(__all__)
