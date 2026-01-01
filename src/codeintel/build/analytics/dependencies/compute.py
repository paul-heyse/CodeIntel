"""Pure compute functions for external dependency analysis.

This module provides pure compute functions that return row data without
performing any database writes. The materialization is handled by the
Hamilton native module in `build/hamilton/native/analytics/dependencies.py`.

The functions analyze external dependency usage across functions and
return structured result containers that can be materialized to DuckDB
tables by the build system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.dependencies.core import (
    EXTERNAL_DEPENDENCIES_COLS,
    EXTERNAL_DEPENDENCY_CALLS_COLS,
    DependencyCallVisitor,
    DependencyContext,
    _aggregate_dependency_calls,
    _build_alias_maps,
    _config_keys_from_frame,
    _dependency_call_rows_from_frame,
    _group_calls,
    _load_dependency_patterns,
    _serialize_dependency_rows,
)
from codeintel.core.hashing import sha1_short
from codeintel.core.paths import normalize_path

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from codeintel.build.analytics.dependencies.core import ExternalDependencyInputs
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DependencyCallsResult:
    """Result container for external dependency calls computation.

    Contains row data for the external_dependency_calls table without
    performing writes. The rows are tuples matching the column
    specifications in the schema.

    Attributes
    ----------
    rows
        Rows for analytics.external_dependency_calls table.
    """

    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ExternalDependenciesResult:
    """Result container for aggregated external dependencies.

    Contains row data for the external_dependencies table without
    performing writes. The rows are tuples matching the column
    specifications in the schema.

    Attributes
    ----------
    rows
        Rows for analytics.external_dependencies table.
    """

    rows: tuple[tuple[object, ...], ...]


def _decimal(value: int) -> Decimal:
    """Convert an integer to Decimal for GOID storage.

    Parameters
    ----------
    value
        Integer value to convert.

    Returns
    -------
    Decimal
        The integer as a Decimal.
    """
    return Decimal(value)


def _function_call_rows_pure(
    *,
    goid: int,
    func_ast: FunctionAst,
    context: DependencyContext,
) -> list[tuple[object, ...]]:
    """Build row tuples for a single function's dependency calls.

    Parameters
    ----------
    goid
        The function's global object ID.
    func_ast
        The function's AST data.
    context
        Shared dependency context with patterns and catalog.

    Returns
    -------
    list[tuple[object, ...]]
        Row tuples for external_dependency_calls table.
    """
    feature_vector = context.features.get(goid)
    if feature_vector is not None and not (
        feature_vector.io_flags.uses_network
        or feature_vector.db_libs
        or feature_vector.http_client_libs
        or feature_vector.message_libs
    ):
        return []

    alias_map = context.alias_maps.get(normalize_path(func_ast.rel_path), {})
    visitor = DependencyCallVisitor(
        alias_map,
        context.patterns,
        func_ast.rel_path,
        func_ast.lines,
    )
    visitor.visit(func_ast.node)
    grouped = _group_calls(visitor.calls)
    if not grouped:
        return []

    module = context.module_map.get(func_ast.rel_path)
    if module is None:
        return []

    urn = context.catalog.urn_for_goid(goid) or ""
    rows: list[tuple[object, ...]] = []

    for library, calls in grouped.items():
        pattern = context.patterns[library]
        dep_id = _dep_id(context.repo, context.commit, library)
        modes = sorted({mode for call in calls for mode in call.modes})
        collector = EvidenceCollector()

        for call in calls:
            collector.add_sample(
                path=func_ast.rel_path,
                line_span=(call.lineno, call.end_lineno),
                snippet=call.snippet,
                details={
                    "target": call.target,
                    "modes": call.modes,
                    "matched_pattern": call.matched_pattern,
                    "severity": call.severity,
                    "criticality": call.criticality,
                },
                tags=(library,),
            )

        evidence = collector.to_dicts()
        rows.append(
            (
                context.repo,
                context.commit,
                dep_id,
                library,
                pattern.service_name or library,
                _decimal(goid),
                urn,
                func_ast.rel_path,
                module,
                func_ast.qualname,
                len(calls),
                modes,
                evidence,
                context.now,
            )
        )

    return rows


def _dep_id(repo: str, commit: str, library: str) -> str:
    """Generate a stable dependency ID from repo, commit, and library.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    library
        Library name.

    Returns
    -------
    str
        A 16-character hex hash ID.
    """
    raw = f"{repo}:{commit}:{library}"
    return sha1_short(raw, length=16, used_for_security=False)


def compute_dependency_calls_pure(
    snapshot: SnapshotRef,
    inputs: ExternalDependencyInputs,
    dependency_patterns_path: Path | None = None,
) -> DependencyCallsResult:
    """Compute external dependency calls without writing to database.

    Analyze AST data to detect external dependency usage across functions,
    returning row data that can be materialized separately.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    inputs
        Grouped inputs containing catalog, module map, AST data, and features.
    dependency_patterns_path
        Optional path to dependency patterns YAML file.

    Returns
    -------
    DependencyCallsResult
        Container with rows for external_dependency_calls table.

    Notes
    -----
    This function is a pure transformation that reads configuration but
    does not write. The materialization is handled by the Hamilton native
    module to ensure proper asset catalog tracking.
    """
    patterns = _load_dependency_patterns(snapshot.repo_root, dependency_patterns_path)
    if not patterns:
        log.warning("No dependency patterns loaded; returning empty result")
        return DependencyCallsResult(rows=())

    missing = inputs.missing_goids or set()
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during dependency analysis",
            len(missing),
        )

    alias_maps = _build_alias_maps(snapshot.repo_root, inputs.module_map)
    now = datetime.now(tz=UTC)

    dep_context = DependencyContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        alias_maps=alias_maps,
        patterns=patterns,
        module_map=inputs.module_map,
        catalog=inputs.catalog_provider,
        now=now,
        features=inputs.features_map,
    )

    rows: list[tuple[object, ...]] = []
    for goid, func_ast in inputs.ast_by_goid.items():
        rows.extend(
            _function_call_rows_pure(
                goid=goid,
                func_ast=func_ast,
                context=dep_context,
            )
        )

    log.info(
        "dependency_calls computed: %d rows for %s@%s",
        len(rows),
        snapshot.repo,
        snapshot.commit,
    )

    return DependencyCallsResult(rows=tuple(rows))


def compute_external_dependencies_pure(
    snapshot: SnapshotRef,
    *,
    dependency_calls_frame: pl.DataFrame | None = None,
    config_values_frame: pl.DataFrame | None = None,
    dependency_patterns_path: Path | None = None,
    language: str = "python",
) -> ExternalDependenciesResult:
    """Compute aggregated external dependencies without writing to database.

    Aggregate dependency usage from the external_dependency_calls table
    into summary rows, returning row data that can be materialized separately.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    dependency_calls_frame
        External dependency calls table snapshot.
    config_values_frame
        Config values table snapshot.
    dependency_patterns_path
        Optional path to dependency patterns YAML file.
    language
        Programming language for the dependencies.

    Returns
    -------
    ExternalDependenciesResult
        Container with rows for external_dependencies table.

    Notes
    -----
    This function expects dependency_calls_frame to come from a materialized
    external_dependency_calls table for the target snapshot.
    """
    patterns = _load_dependency_patterns(snapshot.repo_root, dependency_patterns_path)
    if not patterns:
        log.warning("No dependency patterns loaded; returning empty result")
        return ExternalDependenciesResult(rows=())

    config_keys_by_module = _config_keys_from_frame(
        config_values_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    call_rows = _dependency_call_rows_from_frame(
        dependency_calls_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    aggregates = _aggregate_dependency_calls(call_rows, patterns)
    dep_rows = _serialize_dependency_rows(
        aggregates,
        config_keys_by_module,
        snapshot,
        language=language,
    )

    log.info(
        "external_dependencies computed: %d rows for %s@%s",
        len(dep_rows),
        snapshot.repo,
        snapshot.commit,
    )

    return ExternalDependenciesResult(rows=tuple(dep_rows))


__all__ = [
    "EXTERNAL_DEPENDENCIES_COLS",
    "EXTERNAL_DEPENDENCY_CALLS_COLS",
    "DependencyCallsResult",
    "ExternalDependenciesResult",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
]
