"""Pure compute functions for external dependency analysis."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.dependencies.classification import (
    LibraryPattern,
    risk_level,
    risk_score,
)
from codeintel.build.analytics.compute.dependencies.detection import (
    DependencyCallVisitor,
    group_calls_by_library,
)
from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.compute.row_builders import rows_to_tuples_for_table
from codeintel.build.analytics.utilities.snapshot import (
    SnapshotContext,
    require_columns,
    snapshot_table,
)
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
from codeintel.core.hashing import sha1_short
from codeintel.core.paths import normalize_path
from codeintel.core.schemas.row_models import columns_for_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.columnar.execution_context import ExecutionContext
    from codeintel.core.execution.context import ExecutionContext as RuntimeExecutionContext
    from codeintel.storage.catalog import FunctionCatalogProvider

log = logging.getLogger(__name__)

EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY = "analytics.external_dependency_calls"
EXTERNAL_DEPENDENCIES_TABLE_KEY = "analytics.external_dependencies"


def _columns_for_table(table_key: str) -> list[str]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return list(columns)


EXTERNAL_DEPENDENCY_CALLS_COLS = _columns_for_table(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY)
EXTERNAL_DEPENDENCIES_COLS = _columns_for_table(EXTERNAL_DEPENDENCIES_TABLE_KEY)


@dataclass
class DependencyAggregate:
    """Aggregated usage for a dependency."""

    library: str
    service_name: str | None
    category: str | None
    severity: str | None = None
    criticality: float | None = None
    risk_score: float | None = None
    modules: set[str] = field(default_factory=set)
    functions: set[int] = field(default_factory=set)
    callsite_count: int = 0
    modes: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DependencyContext:
    """Shared context for dependency call classification."""

    repo: str
    commit: str
    alias_maps: dict[str, dict[str, str]]
    patterns: dict[str, LibraryPattern]
    module_map: dict[str, str]
    catalog: FunctionCatalogProvider
    now: datetime
    features: dict[int, FunctionAstFeatures]


@dataclass(frozen=True)
class ExternalDependencyInputs:
    """Inputs for external dependency call analysis."""

    catalog_provider: FunctionCatalogProvider
    module_map: dict[str, str]
    ast_by_goid: dict[int, FunctionAst]
    features_map: dict[int, FunctionAstFeatures]
    missing_goids: set[int] | None = None


@dataclass(frozen=True)
class ExternalDependenciesInputs:
    """Inputs for aggregated external dependencies."""

    dependency_calls_frame: pa.Table | None = None
    config_values_frame: pa.Table | None = None
    patterns: Mapping[str, LibraryPattern] | None = None
    language: str = "python"
    ctx: ExecutionContext | RuntimeExecutionContext | None = None


@dataclass(frozen=True)
class DependencyCallsResult:
    """Result container for external dependency calls computation."""

    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class ExternalDependenciesResult:
    """Result container for aggregated external dependencies."""

    rows: tuple[tuple[object, ...], ...]


def _decimal(value: int) -> Decimal:
    return Decimal(value)


def _dep_id(repo: str, commit: str, library: str) -> str:
    raw = f"{repo}:{commit}:{library}"
    return sha1_short(raw, length=16, used_for_security=False)


def _normalize_module_map(module_map: Mapping[str, str]) -> dict[str, str]:
    return {normalize_path(path): module for path, module in module_map.items()}


def _normalize_alias_maps(
    alias_maps: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    return {normalize_path(path): dict(alias_map) for path, alias_map in alias_maps.items()}


def _function_call_rows_pure(
    *,
    goid: int,
    func_ast: FunctionAst,
    context: DependencyContext,
) -> list[tuple[object, ...]]:
    feature_vector = context.features.get(goid)
    if feature_vector is not None and not (
        feature_vector.io_flags.uses_network
        or feature_vector.db_libs
        or feature_vector.http_client_libs
        or feature_vector.message_libs
    ):
        return []

    rel_path = normalize_path(func_ast.rel_path)
    alias_map = context.alias_maps.get(rel_path, {})
    visitor = DependencyCallVisitor(
        alias_map,
        context.patterns,
        func_ast.rel_path,
        func_ast.lines,
    )
    visitor.visit(func_ast.node)
    grouped = group_calls_by_library(visitor.calls)
    if not grouped:
        return []

    module = context.module_map.get(rel_path)
    if module is None:
        return []

    urn = context.catalog.urn_for_goid(goid) or ""
    row_dicts: list[dict[str, object]] = []

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
        row_dicts.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "dep_id": dep_id,
                "library": library,
                "service_name": pattern.service_name or library,
                "function_goid_h128": _decimal(goid),
                "function_urn": urn,
                "rel_path": func_ast.rel_path,
                "module": module,
                "qualname": func_ast.qualname,
                "callsite_count": len(calls),
                "extras": {
                    "modes": modes,
                    "evidence": evidence,
                },
                "created_at": context.now,
            }
        )

    return rows_to_tuples_for_table(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY, row_dicts)


def _dependency_call_rows_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> Iterable[tuple[object, ...]]:
    if frame is None or frame.num_rows == 0:
        return ()
    filtered = _rows_for_snapshot(frame, repo=repo, commit=commit, ctx=ctx)
    rows: list[tuple[object, ...]] = []
    for row in filtered:
        extras = row.get("extras")
        modes = extras.get("modes") if isinstance(extras, Mapping) else row.get("modes")
        rows.append(
            (
                row.get("dep_id"),
                row.get("library"),
                row.get("function_goid_h128"),
                row.get("module"),
                row.get("callsite_count"),
                modes,
                row.get("severity"),
                row.get("criticality"),
                row.get("risk_score"),
            )
        )
    return rows


def _aggregate_dependency_calls(
    rows: Iterable[tuple[object, ...]],
    patterns: Mapping[str, LibraryPattern],
) -> dict[str, DependencyAggregate]:
    aggregates: dict[str, DependencyAggregate] = {}
    for (
        dep_id,
        library,
        function_goid,
        module,
        callsite_count,
        modes_obj,
        severity,
        criticality,
        risk_score_value,
    ) in rows:
        if library is None or dep_id is None:
            continue
        lib_key = str(library)
        pattern = patterns.get(lib_key)
        severity_value = _as_str(severity) or (pattern.severity if pattern else None)
        criticality_value = _as_float(criticality) if criticality is not None else None
        aggregate = aggregates.setdefault(
            str(dep_id),
            DependencyAggregate(
                library=lib_key,
                service_name=(pattern.service_name if pattern else None) or lib_key,
                category=pattern.category if pattern else None,
                severity=severity_value,
                criticality=(
                    criticality_value
                    if criticality_value is not None
                    else (pattern.criticality if pattern else None)
                ),
                risk_score=None,
            ),
        )
        if module:
            aggregate.modules.add(str(module))
        function_goid_value = _as_int(function_goid)
        if function_goid_value is not None:
            aggregate.functions.add(function_goid_value)
        aggregate.callsite_count += _as_int(callsite_count) or 0
        aggregate.modes.update(_ensure_str_list(modes_obj))
        agg_score = (
            _as_float(risk_score_value)
            if risk_score_value is not None
            else risk_score(severity_value, criticality_value)
        )
        if agg_score is not None:
            prev_score = aggregate.risk_score or 0.0
            if agg_score > prev_score:
                aggregate.risk_score = agg_score
        if severity_value and aggregate.severity is None:
            aggregate.severity = severity_value
        if criticality_value is not None and aggregate.criticality is None:
            aggregate.criticality = criticality_value
    return aggregates


def _serialize_dependency_rows(
    aggregates: dict[str, DependencyAggregate],
    config_keys_by_module: dict[str, set[str]],
    snapshot: SnapshotRef,
    *,
    language: str,
    now: datetime,
) -> list[tuple[object, ...]]:
    row_dicts: list[dict[str, object]] = []
    for dep_id, aggregate in aggregates.items():
        config_keys: set[str] = set()
        for module in aggregate.modules:
            config_keys.update(config_keys_by_module.get(module, set()))
        resolved_risk_level = aggregate.severity or risk_level(
            aggregate.modes,
            aggregate.callsite_count,
        )
        row_dicts.append(
            {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "dep_id": dep_id,
                "library": aggregate.library,
                "service_name": aggregate.service_name,
                "category": aggregate.category,
                "language": language,
                "severity": aggregate.severity,
                "criticality": aggregate.criticality,
                "risk_score": aggregate.risk_score,
                "function_count": len(aggregate.functions),
                "callsite_count": aggregate.callsite_count,
                "extras": {
                    "modules": sorted(aggregate.modules),
                    "usage_modes": sorted(aggregate.modes),
                    "config_keys": sorted(config_keys) if config_keys else None,
                },
                "risk_level": resolved_risk_level,
                "created_at": now,
            }
        )
    return rows_to_tuples_for_table(EXTERNAL_DEPENDENCIES_TABLE_KEY, row_dicts)


def _as_int(value: object | None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: object | None) -> float | None:
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_str(value: object | None) -> str | None:
    return str(value) if isinstance(value, str) else None


def _ensure_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _config_keys_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    if frame is None or frame.num_rows == 0:
        return mapping
    filtered = _rows_for_snapshot(frame, repo=repo, commit=commit, ctx=ctx)
    for row in filtered:
        extras = row.get("extras")
        ref_modules = extras.get("reference_modules") if isinstance(extras, Mapping) else None
        key = row.get("key")
        if key is None or ref_modules is None:
            continue
        modules = _ensure_str_list(ref_modules)
        for module in modules:
            mapping.setdefault(module, set()).add(str(key))
    return mapping


def _rows_for_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None,
) -> list[dict[str, object]]:
    require_columns(frame, ("repo", "commit"))
    filtered = snapshot_table(
        frame,
        context=SnapshotContext(repo=repo, commit=commit, ctx=ctx),
    )
    return list(iter_rows(filtered))


def load_config_key_map(
    config_values_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
    ctx: ExecutionContext | RuntimeExecutionContext | None = None,
) -> dict[str, set[str]]:
    """Load config keys keyed by module for a repo snapshot.

    Returns
    -------
    dict[str, set[str]]
        Mapping of module path to referenced config keys.
    """
    return _config_keys_from_frame(config_values_frame, repo=repo, commit=commit, ctx=ctx)


def compute_dependency_calls_pure(
    snapshot: SnapshotRef,
    inputs: ExternalDependencyInputs,
    *,
    patterns: Mapping[str, LibraryPattern] | None,
    alias_maps: Mapping[str, Mapping[str, str]] | None,
    now: datetime | None = None,
) -> DependencyCallsResult:
    """Compute external dependency calls without writing to database.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    inputs
        Grouped inputs containing catalog, module map, AST data, and features.
    patterns
        Dependency patterns to classify external calls.
    alias_maps
        Import alias maps keyed by relative path.
    now
        Timestamp override for deterministic tests.

    Returns
    -------
    DependencyCallsResult
        Result container with ordered dependency call rows.
    """
    patterns_map = dict(patterns or {})
    if not patterns_map:
        log.warning("No dependency patterns loaded; returning empty result")
        return DependencyCallsResult(rows=())

    missing = inputs.missing_goids or set()
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during dependency analysis",
            len(missing),
        )

    normalized_alias_maps = _normalize_alias_maps(alias_maps or {})
    normalized_module_map = _normalize_module_map(inputs.module_map)
    resolved_now = now or datetime.now(tz=UTC)

    dep_context = DependencyContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        alias_maps=normalized_alias_maps,
        patterns=patterns_map,
        module_map=normalized_module_map,
        catalog=inputs.catalog_provider,
        now=resolved_now,
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
    inputs: ExternalDependenciesInputs,
    now: datetime | None = None,
) -> ExternalDependenciesResult:
    """Compute aggregated external dependencies without writing to database.

    Parameters
    ----------
    snapshot
        Repository and commit snapshot reference.
    inputs
        Bundled inputs for dependency aggregation.
    now
        Timestamp override for deterministic tests.

    Returns
    -------
    ExternalDependenciesResult
        Result container with ordered dependency rows.
    """
    patterns_map = dict(inputs.patterns or {})
    if not patterns_map:
        log.warning("No dependency patterns loaded; returning empty result")
        return ExternalDependenciesResult(rows=())

    config_keys_by_module = _config_keys_from_frame(
        inputs.config_values_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    call_rows = _dependency_call_rows_from_frame(
        inputs.dependency_calls_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
        ctx=inputs.ctx,
    )
    aggregates = _aggregate_dependency_calls(call_rows, patterns_map)
    resolved_now = now or datetime.now(tz=UTC)
    dep_rows = _serialize_dependency_rows(
        aggregates,
        config_keys_by_module,
        snapshot,
        language=inputs.language,
        now=resolved_now,
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
    "DependencyAggregate",
    "DependencyCallsResult",
    "DependencyContext",
    "ExternalDependenciesInputs",
    "ExternalDependenciesResult",
    "ExternalDependencyInputs",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
    "load_config_key_map",
]
