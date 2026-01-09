"""Orchestration helpers for external dependency analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.dependencies.compute import (
    EXTERNAL_DEPENDENCIES_COLS,
    EXTERNAL_DEPENDENCY_CALLS_COLS,
    DependencyCallsResult,
    ExternalDependenciesInputs,
    ExternalDependenciesResult,
)
from codeintel.build.analytics.compute.dependencies.compute import (
    compute_dependency_calls_pure as _compute_dependency_calls_pure,
)
from codeintel.build.analytics.compute.dependencies.compute import (
    compute_external_dependencies_pure as _compute_external_dependencies_pure,
)
from codeintel.build.analytics.dependencies.core import (
    build_alias_maps,
    load_dependency_patterns,
)
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext

if TYPE_CHECKING:
    from pathlib import Path

    import pyarrow as pa

    from codeintel.build.analytics.compute.dependencies.compute import ExternalDependencyInputs
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.columnar.execution_context import ExecutionContext

log = logging.getLogger(__name__)


def compute_dependency_calls_pure(
    snapshot: SnapshotRef,
    inputs: ExternalDependencyInputs,
    dependency_patterns_path: Path | None = None,
) -> DependencyCallsResult:
    """Compute external dependency calls without writing to database.

    Returns
    -------
    DependencyCallsResult
        Dependency call rows ready for materialization.
    """
    patterns = load_dependency_patterns(snapshot.repo_root, dependency_patterns_path)
    if not patterns:
        log.warning("No dependency patterns loaded; returning empty result")
        return DependencyCallsResult(rows=())
    alias_maps = build_alias_maps(snapshot.repo_root, inputs.module_map)
    return _compute_dependency_calls_pure(
        snapshot,
        inputs,
        patterns=patterns,
        alias_maps=alias_maps,
    )


@dataclass(frozen=True, slots=True)
class ExternalDependenciesRequest:
    """Inputs required to aggregate external dependencies."""

    dependency_calls_frame: pa.Table | None = None
    config_values_frame: pa.Table | None = None
    dependency_patterns_path: Path | None = None
    language: str = "python"
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


def compute_external_dependencies_pure(
    snapshot: SnapshotRef,
    request: ExternalDependenciesRequest,
) -> ExternalDependenciesResult:
    """Compute aggregated external dependencies without writing to database.

    Returns
    -------
    ExternalDependenciesResult
        Aggregated dependency rows ready for materialization.
    """
    patterns = load_dependency_patterns(snapshot.repo_root, request.dependency_patterns_path)
    if not patterns:
        log.warning("No dependency patterns loaded; returning empty result")
        return ExternalDependenciesResult(rows=())
    return _compute_external_dependencies_pure(
        snapshot,
        ExternalDependenciesInputs(
            dependency_calls_frame=request.dependency_calls_frame,
            config_values_frame=request.config_values_frame,
            patterns=patterns,
            language=request.language,
            ctx=request.ctx,
        ),
    )


__all__ = [
    "EXTERNAL_DEPENDENCIES_COLS",
    "EXTERNAL_DEPENDENCY_CALLS_COLS",
    "DependencyCallsResult",
    "ExternalDependenciesRequest",
    "ExternalDependenciesResult",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
]
