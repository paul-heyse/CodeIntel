"""Function profile recipe helpers."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from codeintel.build.analytics.profiles.types import (
    CoverageSummary,
    FunctionBaseInfo,
    FunctionContractView,
    FunctionDocView,
    FunctionEffectsView,
    FunctionProfileFrames,
    FunctionProfileInputs,
    FunctionRiskView,
    FunctionRoleView,
)
from codeintel.build.analytics.profiles.utils import (
    CATALOG_MODULE_TABLE,
    DEFAULT_MODULE_TABLE,
)
from codeintel.build.analytics.utilities.type_coercion import (
    int_or_default,
    optional_float,
    optional_int,
    optional_str,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.analytics.profiles.types import FunctionGraphFeatures
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsFunctionProfileRow as FunctionProfileRowModel,
    )

log = logging.getLogger(__name__)

SLOW_TEST_THRESHOLD_MS = 1000.0


def _scope_frame(frame: pl.DataFrame, repo: str, commit: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "repo" in frame.columns and "commit" in frame.columns:
        return frame.filter((pl.col("repo") == repo) & (pl.col("commit") == commit))
    return frame


def _types_by_goid(types: pl.DataFrame) -> dict[int, dict[str, object]]:
    types_by_goid: dict[int, dict[str, object]] = {}
    for row in types.iter_rows(named=True):
        goid = optional_int(row.get("function_goid_h128"))
        if goid is None:
            continue
        types_by_goid[goid] = row
    return types_by_goid


def _module_by_path(modules: pl.DataFrame) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    for row in modules.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_by_path[path] = module
    return module_by_path


def _typedness_by_path(typedness: pl.DataFrame) -> dict[str, float | None]:
    typedness_by_path: dict[str, float | None] = {}
    for row in typedness.iter_rows(named=True):
        path = row.get("path")
        if isinstance(path, str):
            typedness_by_path[path] = _extract_annotation_ratio(row.get("annotation_ratio"))
    return typedness_by_path


def _diagnostics_by_path(diagnostics: pl.DataFrame) -> dict[str, tuple[int, bool]]:
    diagnostics_by_path: dict[str, tuple[int, bool]] = {}
    for row in diagnostics.iter_rows(named=True):
        path = row.get("rel_path")
        if isinstance(path, str):
            diagnostics_by_path[path] = (
                int_or_default(row.get("total_errors")),
                bool(row.get("has_errors")),
            )
    return diagnostics_by_path


def _extract_annotation_ratio(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        ratio = value.get("params")
        return float(ratio) if isinstance(ratio, (int, float)) else None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(decoded, dict):
            ratio = decoded.get("params")
            return float(ratio) if isinstance(ratio, (int, float)) else None
    return None


def _coerce_goid(record: Mapping[str, object]) -> int | None:
    return optional_int(record.get("function_goid_h128"))


@dataclass(frozen=True)
class FunctionProfileViews:
    """Container for per-function views used to assemble profiles."""

    base_by_func: Mapping[int, FunctionBaseInfo]
    risk_by_func: Mapping[int, FunctionRiskView]
    coverage_by_func: Mapping[int, CoverageSummary]
    graph_by_func: Mapping[int, FunctionGraphFeatures]
    effects_by_func: Mapping[int, FunctionEffectsView]
    contracts_by_func: Mapping[int, FunctionContractView]
    roles_by_func: Mapping[int, FunctionRoleView]
    docs_by_func: Mapping[int, FunctionDocView]


def compute_function_profile_inputs(
    snapshot: SnapshotRef,
    frames: FunctionProfileFrames,
    *,
    slow_test_threshold_ms: float = SLOW_TEST_THRESHOLD_MS,
) -> FunctionProfileInputs:
    """
    Normalize snapshot inputs for function profile computations.

    The returned object is intentionally lightweight; heavy lifting happens in
    downstream helpers.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    frames
        Frame bundle for function profile inputs.
    slow_test_threshold_ms
        Threshold for slow tests in milliseconds.

    Returns
    -------
    FunctionProfileInputs
        Snapshot handle used by downstream helpers.
    """
    return FunctionProfileInputs(
        repo=snapshot.repo,
        commit=snapshot.commit,
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=slow_test_threshold_ms,
        function_metrics=frames.function_metrics,
        function_types=frames.function_types,
        modules=frames.modules,
        typedness=frames.typedness,
        diagnostics=frames.diagnostics,
        goid_risk_factors=frames.goid_risk_factors,
        coverage_functions=frames.coverage_functions,
        graph_metrics_functions=frames.graph_metrics_functions,
        function_effects=frames.function_effects,
        function_contracts=frames.function_contracts,
        semantic_roles_functions=frames.semantic_roles_functions,
        docstrings=frames.docstrings,
        hotspots=frames.hotspots,
        call_graph_edges=frames.call_graph_edges,
        call_graph_nodes=frames.call_graph_nodes,
        test_coverage_edges=frames.test_coverage_edges,
        test_catalog=frames.test_catalog,
    )


def load_function_base_info(
    inputs: FunctionProfileInputs,
    *,
    module_table: str = DEFAULT_MODULE_TABLE,
) -> Mapping[int, FunctionBaseInfo]:
    """Load static per-function metadata from metrics and typedness tables.

    Returns
    -------
    Mapping[int, FunctionBaseInfo]
        Mapping keyed by function GOID with static attributes.

    Raises
    ------
    ValueError
        If an unexpected module table name is provided.
    """
    if module_table not in {DEFAULT_MODULE_TABLE, CATALOG_MODULE_TABLE}:
        msg = f"Unexpected module table: {module_table}"
        raise ValueError(msg)

    metrics = _scope_frame(inputs.function_metrics, inputs.repo, inputs.commit)
    if metrics.is_empty():
        return {}

    types_by_goid = _types_by_goid(_scope_frame(inputs.function_types, inputs.repo, inputs.commit))
    module_by_path = _module_by_path(_scope_frame(inputs.modules, inputs.repo, inputs.commit))
    typedness_by_path = _typedness_by_path(
        _scope_frame(inputs.typedness, inputs.repo, inputs.commit)
    )
    diagnostics_by_path = _diagnostics_by_path(
        _scope_frame(inputs.diagnostics, inputs.repo, inputs.commit)
    )

    result: dict[int, FunctionBaseInfo] = {}
    for record in metrics.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid_int = goid_value
        type_row = types_by_goid.get(goid_int, {})
        rel_path = optional_str(record.get("rel_path")) or ""
        module = module_by_path.get(rel_path)
        typed_ratio = typedness_by_path.get(rel_path)
        static_errors = diagnostics_by_path.get(rel_path, (0, False))
        result[goid_int] = FunctionBaseInfo(
            function_goid_h128=goid_int,
            urn=optional_str(record.get("urn")),
            repo=str(record.get("repo")),
            commit=str(record.get("commit")),
            rel_path=rel_path,
            module=module,
            language=optional_str(record.get("language")),
            kind=optional_str(record.get("kind")),
            qualname=optional_str(record.get("qualname")),
            start_line=optional_int(record.get("start_line")),
            end_line=optional_int(record.get("end_line")),
            loc=int_or_default(record.get("loc")),
            logical_loc=int_or_default(record.get("logical_loc")),
            cyclomatic_complexity=int_or_default(record.get("cyclomatic_complexity")),
            complexity_bucket=optional_str(record.get("complexity_bucket")),
            param_count=int_or_default(record.get("param_count")),
            positional_params=int_or_default(record.get("positional_params")),
            keyword_params=int_or_default(record.get("keyword_only_params")),
            vararg=bool(record.get("has_varargs")),
            kwarg=bool(record.get("has_varkw")),
            max_nesting_depth=optional_int(record.get("max_nesting_depth")),
            stmt_count=optional_int(record.get("stmt_count")),
            decorator_count=optional_int(record.get("decorator_count")),
            has_docstring=bool(record.get("has_docstring")),
            total_params=int_or_default(type_row.get("total_params")),
            annotated_params=int_or_default(type_row.get("annotated_params")),
            return_type=optional_str(type_row.get("return_type")),
            param_types=type_row.get("param_types"),
            fully_typed=bool(type_row.get("fully_typed")),
            partial_typed=bool(type_row.get("partial_typed")),
            untyped=bool(type_row.get("untyped")),
            typedness_bucket=optional_str(type_row.get("typedness_bucket")),
            typedness_source=optional_str(type_row.get("typedness_source")),
            file_typed_ratio=typed_ratio,
            static_error_count=static_errors[0],
            has_static_errors=static_errors[1],
        )
    return result


def join_function_risk(inputs: FunctionProfileInputs) -> Mapping[int, FunctionRiskView]:
    """
    Collect risk scores, levels, and ownership metadata.

    Returns
    -------
    Mapping[int, FunctionRiskView]
        Mapping keyed by function GOID.
    """
    metrics = _scope_frame(inputs.function_metrics, inputs.repo, inputs.commit)
    if metrics.is_empty():
        return {}

    risk_factors = _scope_frame(inputs.goid_risk_factors, inputs.repo, inputs.commit)
    modules = _scope_frame(inputs.modules, inputs.repo, inputs.commit)
    hotspots = _scope_frame(inputs.hotspots, inputs.repo, inputs.commit)

    risk_by_goid: dict[int, dict[str, object]] = {}
    for row in risk_factors.iter_rows(named=True):
        goid = _coerce_goid(row)
        if goid is None:
            continue
        risk_by_goid[goid] = row

    tags_by_path: dict[str, tuple[object, object]] = {}
    for row in modules.iter_rows(named=True):
        path = row.get("path")
        if isinstance(path, str):
            tags_by_path[path] = (row.get("tags"), row.get("owners"))

    hotspots_by_path: dict[str, float | None] = {}
    for row in hotspots.iter_rows(named=True):
        rel_path = row.get("rel_path")
        if isinstance(rel_path, str):
            hotspots_by_path[rel_path] = optional_float(row.get("score"))

    result: dict[int, FunctionRiskView] = {}
    for record in metrics.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        risk_row = risk_by_goid.get(goid, {})
        rel_path = optional_str(record.get("rel_path")) or ""
        tags, owners = tags_by_path.get(rel_path, ("[]", "[]"))
        result[goid] = FunctionRiskView(
            function_goid_h128=goid,
            risk_score=optional_float(risk_row.get("risk_score")) or 0.0,
            risk_level=optional_str(risk_row.get("risk_level")),
            hotspot_score=hotspots_by_path.get(rel_path),
            tags=tags if tags is not None else "[]",
            owners=owners if owners is not None else "[]",
        )
    return result


@dataclass(frozen=True)
class _CoverageTestSets:
    tests_touching: dict[int, set[str]]
    failing_tests: dict[int, set[str]]
    slow_tests: dict[int, set[str]]
    flaky_tests: dict[int, set[str]]
    status_sets: dict[int, dict[str, set[str]]]


def _catalog_by_test(catalog: pl.DataFrame) -> dict[str, dict[str, object]]:
    catalog_by_test: dict[str, dict[str, object]] = {}
    for row in catalog.iter_rows(named=True):
        test_id = row.get("test_id")
        if isinstance(test_id, str):
            catalog_by_test[test_id] = row
    return catalog_by_test


def _coverage_sets_from_edges(
    edges: pl.DataFrame,
    catalog_by_test: dict[str, dict[str, object]],
    *,
    slow_threshold: float,
) -> _CoverageTestSets:
    tests_touching: dict[int, set[str]] = {}
    failing_tests: dict[int, set[str]] = {}
    slow_tests: dict[int, set[str]] = {}
    flaky_tests: dict[int, set[str]] = {}
    status_sets: dict[int, dict[str, set[str]]] = {}

    for row in edges.iter_rows(named=True):
        goid = optional_int(row.get("function_goid_h128"))
        test_id = row.get("test_id")
        if goid is None or not isinstance(test_id, str):
            continue
        tests_touching.setdefault(goid, set()).add(test_id)
        test_meta = catalog_by_test.get(test_id)
        if test_meta is None:
            continue
        status = optional_str(test_meta.get("status")) or "unknown"
        status_sets.setdefault(goid, {}).setdefault(status, set()).add(test_id)
        if status in {"failed", "error"}:
            failing_tests.setdefault(goid, set()).add(test_id)
        duration = optional_float(test_meta.get("duration_ms")) or 0.0
        if duration > slow_threshold:
            slow_tests.setdefault(goid, set()).add(test_id)
        if bool(test_meta.get("flaky")):
            flaky_tests.setdefault(goid, set()).add(test_id)

    return _CoverageTestSets(
        tests_touching=tests_touching,
        failing_tests=failing_tests,
        slow_tests=slow_tests,
        flaky_tests=flaky_tests,
        status_sets=status_sets,
    )


def _dominant_status_by_goid(
    status_sets: dict[int, dict[str, set[str]]],
) -> dict[int, str | None]:
    dominant_status: dict[int, str | None] = {}
    for goid, status_map in status_sets.items():
        if not status_map:
            dominant_status[goid] = None
            continue
        max_count = max(len(vals) for vals in status_map.values())
        candidates = sorted(
            [status for status, vals in status_map.items() if len(vals) == max_count]
        )
        dominant_status[goid] = candidates[0] if candidates else None
    return dominant_status


def join_function_coverage(inputs: FunctionProfileInputs) -> Mapping[int, CoverageSummary]:
    """
    Aggregate coverage and test metrics per function.

    Returns
    -------
    Mapping[int, CoverageSummary]
        Mapping keyed by function GOID.
    """
    coverage = _scope_frame(inputs.coverage_functions, inputs.repo, inputs.commit)
    if coverage.is_empty():
        return {}

    edges = _scope_frame(inputs.test_coverage_edges, inputs.repo, inputs.commit)
    catalog = _scope_frame(inputs.test_catalog, inputs.repo, inputs.commit)
    slow_threshold = float(inputs.slow_test_threshold_ms)
    catalog_by_test = _catalog_by_test(catalog)
    coverage_sets = _coverage_sets_from_edges(
        edges,
        catalog_by_test,
        slow_threshold=slow_threshold,
    )
    dominant_status = _dominant_status_by_goid(coverage_sets.status_sets)

    result: dict[int, CoverageSummary] = {}
    for record in coverage.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        tests = coverage_sets.tests_touching.get(goid, set())
        failing = coverage_sets.failing_tests.get(goid, set())
        slow = coverage_sets.slow_tests.get(goid, set())
        flaky = coverage_sets.flaky_tests.get(goid, set())
        dominant = dominant_status.get(goid)
        result[goid] = CoverageSummary(
            function_goid_h128=goid,
            executable_lines=int_or_default(record.get("executable_lines")),
            covered_lines=int_or_default(record.get("covered_lines")),
            coverage_ratio=optional_float(record.get("coverage_ratio")),
            tested=bool(record.get("tested")),
            untested_reason=optional_str(record.get("untested_reason")),
            tests_touching=len(tests),
            failing_tests=len(failing),
            slow_tests=len(slow),
            flaky_tests=len(flaky),
            last_test_status=dominant or "untested",
            dominant_test_status=dominant,
        )
    return result


def join_function_effects(inputs: FunctionProfileInputs) -> Mapping[int, FunctionEffectsView]:
    """
    Collect effect summaries from analytics.function_effects.

    Returns
    -------
    Mapping[int, FunctionEffectsView]
        Mapping keyed by function GOID.
    """
    effects = _scope_frame(inputs.function_effects, inputs.repo, inputs.commit)
    if effects.is_empty():
        return {}
    result: dict[int, FunctionEffectsView] = {}
    for record in effects.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        result[goid] = FunctionEffectsView(
            function_goid_h128=goid,
            is_pure=bool(record.get("is_pure")),
            uses_io=bool(record.get("uses_io")),
            touches_db=bool(record.get("touches_db")),
            uses_time=bool(record.get("uses_time")),
            uses_randomness=bool(record.get("uses_randomness")),
            modifies_globals=bool(record.get("modifies_globals")),
            modifies_closure=bool(record.get("modifies_closure")),
            spawns_threads_or_tasks=bool(record.get("spawns_threads_or_tasks")),
            has_transitive_effects=bool(record.get("has_transitive_effects")),
            purity_confidence=optional_float(record.get("purity_confidence")),
        )
    return result


def join_function_contracts(inputs: FunctionProfileInputs) -> Mapping[int, FunctionContractView]:
    """
    Collect contract metadata from analytics.function_contracts.

    Returns
    -------
    Mapping[int, FunctionContractView]
        Mapping keyed by function GOID.
    """
    contracts = _scope_frame(inputs.function_contracts, inputs.repo, inputs.commit)
    if contracts.is_empty():
        return {}
    result: dict[int, FunctionContractView] = {}
    for record in contracts.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        preconditions = record.get("preconditions_json")
        postconditions = record.get("postconditions_json")
        raises_json = record.get("raises_json")
        result[goid] = FunctionContractView(
            function_goid_h128=goid,
            param_nullability_json=record.get("param_nullability_json"),
            return_nullability=optional_str(record.get("return_nullability")),
            has_preconditions=bool(preconditions) if preconditions not in {"", None} else False,
            has_postconditions=bool(postconditions) if postconditions not in {"", None} else False,
            has_raises=bool(raises_json) if raises_json not in {"", None} else False,
            contract_confidence=optional_float(record.get("contract_confidence")),
        )
    return result


def join_function_roles(inputs: FunctionProfileInputs) -> Mapping[int, FunctionRoleView]:
    """
    Collect semantic roles per function.

    Returns
    -------
    Mapping[int, FunctionRoleView]
        Mapping keyed by function GOID.
    """
    roles = _scope_frame(inputs.semantic_roles_functions, inputs.repo, inputs.commit)
    if roles.is_empty():
        return {}
    result: dict[int, FunctionRoleView] = {}
    for record in roles.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        result[goid] = FunctionRoleView(
            function_goid_h128=goid,
            role=optional_str(record.get("role")),
            framework=optional_str(record.get("framework")),
            role_confidence=optional_float(record.get("role_confidence")),
            role_sources_json=(
                record.get("role_sources_json")
                if record.get("role_sources_json") is not None
                else "[]"
            ),
        )
    return result


def join_function_docs(inputs: FunctionProfileInputs) -> Mapping[int, FunctionDocView]:
    """
    Collect docstring surfaces per function.

    Returns
    -------
    Mapping[int, FunctionDocView]
        Mapping keyed by function GOID.
    """
    metrics = _scope_frame(inputs.function_metrics, inputs.repo, inputs.commit)
    if metrics.is_empty():
        return {}

    docs = _scope_frame(inputs.docstrings, inputs.repo, inputs.commit)
    doc_by_key: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in docs.iter_rows(named=True):
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        kind = row.get("kind")
        if isinstance(rel_path, str) and isinstance(qualname, str) and isinstance(kind, str):
            doc_by_key[rel_path, qualname, kind] = row

    result: dict[int, FunctionDocView] = {}
    for record in metrics.iter_rows(named=True):
        goid_value = _coerce_goid(record)
        if goid_value is None:
            continue
        goid = goid_value
        rel_path = optional_str(record.get("rel_path")) or ""
        qualname = optional_str(record.get("qualname")) or ""
        kind = optional_str(record.get("kind")) or ""
        doc_row = doc_by_key.get((rel_path, qualname, kind), {})
        result[goid] = FunctionDocView(
            function_goid_h128=goid,
            doc_short=optional_str(doc_row.get("short_desc")),
            doc_long=optional_str(doc_row.get("long_desc")),
            doc_params=doc_row.get("params"),
            doc_returns=doc_row.get("returns"),
        )
    return result


def build_function_profile_rows(
    inputs: FunctionProfileInputs,
    views: FunctionProfileViews,
) -> Iterable[FunctionProfileRowModel]:
    """
    Assemble FunctionProfileRowModel values from per-concern mappings.

    Yields
    ------
    FunctionProfileRowModel
        Row models ready for insertion into ``analytics.function_profile``.
    """
    for goid, base in views.base_by_func.items():
        risk = views.risk_by_func.get(goid)
        coverage = views.coverage_by_func.get(goid)
        graph = views.graph_by_func.get(goid)
        effects = views.effects_by_func.get(goid)
        contract = views.contracts_by_func.get(goid)
        role = views.roles_by_func.get(goid)
        doc = views.docs_by_func.get(goid)

        coverage_ratio = coverage.coverage_ratio if coverage is not None else None
        risk_component_coverage = (
            (1.0 - coverage_ratio) * 0.4 if coverage_ratio is not None else 0.4
        )
        risk_component_complexity = 0.0
        if base.complexity_bucket == "high":
            risk_component_complexity = 0.4
        elif base.complexity_bucket == "medium":
            risk_component_complexity = 0.2
        risk_component_static = 0.2 if base.has_static_errors else 0.0
        risk_component_hotspot = (
            0.1
            if risk is not None
            and risk.hotspot_score is not None
            and float(risk.hotspot_score) > 0.0
            else 0.0
        )

        row: FunctionProfileRowModel = {
            "function_goid_h128": goid,
            "urn": base.urn,
            "repo": base.repo,
            "commit": base.commit,
            "rel_path": base.rel_path,
            "module": base.module,
            "language": base.language,
            "kind": base.kind,
            "qualname": base.qualname,
            "start_line": base.start_line,
            "end_line": base.end_line,
            "loc": base.loc,
            "logical_loc": base.logical_loc,
            "cyclomatic_complexity": base.cyclomatic_complexity,
            "complexity_bucket": base.complexity_bucket,
            "param_count": base.param_count,
            "positional_params": base.positional_params,
            "keyword_params": base.keyword_params,
            "vararg": base.vararg,
            "kwarg": base.kwarg,
            "max_nesting_depth": base.max_nesting_depth,
            "stmt_count": base.stmt_count,
            "decorator_count": base.decorator_count,
            "has_docstring": base.has_docstring,
            "total_params": base.total_params,
            "annotated_params": base.annotated_params,
            "return_type": base.return_type,
            "param_types": base.param_types,
            "fully_typed": base.fully_typed,
            "partial_typed": base.partial_typed,
            "untyped": base.untyped,
            "typedness_bucket": base.typedness_bucket,
            "typedness_source": base.typedness_source,
            "file_typed_ratio": base.file_typed_ratio,
            "static_error_count": base.static_error_count,
            "has_static_errors": base.has_static_errors,
            "executable_lines": coverage.executable_lines if coverage is not None else 0,
            "covered_lines": coverage.covered_lines if coverage is not None else 0,
            "coverage_ratio": coverage_ratio,
            "tested": coverage.tested if coverage is not None else False,
            "untested_reason": coverage.untested_reason if coverage is not None else None,
            "tests_touching": coverage.tests_touching if coverage is not None else 0,
            "failing_tests": coverage.failing_tests if coverage is not None else 0,
            "slow_tests": coverage.slow_tests if coverage is not None else 0,
            "flaky_tests": coverage.flaky_tests if coverage is not None else 0,
            "last_test_status": coverage.last_test_status if coverage is not None else None,
            "dominant_test_status": (
                coverage.dominant_test_status if coverage is not None else None
            ),
            "slow_test_threshold_ms": inputs.slow_test_threshold_ms,
            "created_in_commit": None,
            "created_at_history": None,
            "last_modified_commit": None,
            "last_modified_at": None,
            "age_days": None,
            "commit_count": 0,
            "author_count": 0,
            "lines_added": 0,
            "lines_deleted": 0,
            "churn_score": 0.0,
            "stability_bucket": "unknown",
            "call_fan_in": graph.call_fan_in if graph is not None else 0,
            "call_fan_out": graph.call_fan_out if graph is not None else 0,
            "call_edge_in_count": graph.call_edge_in_count if graph is not None else 0,
            "call_edge_out_count": graph.call_edge_out_count if graph is not None else 0,
            "call_is_leaf": graph.call_is_leaf if graph is not None else False,
            "call_is_entrypoint": graph.call_is_entrypoint if graph is not None else False,
            "call_is_public": graph.call_is_public if graph is not None else False,
            "risk_score": risk.risk_score if risk is not None else 0.0,
            "risk_level": risk.risk_level if risk is not None else None,
            "risk_component_coverage": risk_component_coverage,
            "risk_component_complexity": risk_component_complexity,
            "risk_component_static": risk_component_static,
            "risk_component_hotspot": risk_component_hotspot,
            "is_pure": effects.is_pure if effects is not None else False,
            "uses_io": effects.uses_io if effects is not None else False,
            "touches_db": effects.touches_db if effects is not None else False,
            "uses_time": effects.uses_time if effects is not None else False,
            "uses_randomness": effects.uses_randomness if effects is not None else False,
            "modifies_globals": effects.modifies_globals if effects is not None else False,
            "modifies_closure": effects.modifies_closure if effects is not None else False,
            "spawns_threads_or_tasks": (
                effects.spawns_threads_or_tasks if effects is not None else False
            ),
            "has_transitive_effects": (
                effects.has_transitive_effects if effects is not None else False
            ),
            "purity_confidence": effects.purity_confidence if effects is not None else None,
            "param_nullability_json": (
                contract.param_nullability_json if contract is not None else None
            ),
            "return_nullability": contract.return_nullability if contract is not None else None,
            "has_preconditions": contract.has_preconditions if contract is not None else False,
            "has_postconditions": contract.has_postconditions if contract is not None else False,
            "has_raises": contract.has_raises if contract is not None else False,
            "contract_confidence": (contract.contract_confidence if contract is not None else None),
            "role": role.role if role is not None else None,
            "framework": role.framework if role is not None else None,
            "role_confidence": role.role_confidence if role is not None else None,
            "role_sources_json": role.role_sources_json if role is not None else None,
            "tags": risk.tags if risk is not None else "[]",
            "owners": risk.owners if risk is not None else "[]",
            "doc_short": doc.doc_short if doc is not None else None,
            "doc_long": doc.doc_long if doc is not None else None,
            "doc_params": doc.doc_params if doc is not None else None,
            "doc_returns": doc.doc_returns if doc is not None else None,
            "created_at": inputs.created_at,
        }

        yield row
